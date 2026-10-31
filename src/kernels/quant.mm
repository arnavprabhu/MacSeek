// Host encode functions of the quant module.
#include "kernels/quant.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "core/env.hpp"

namespace ms::q {

namespace {

struct QMvArgs {  // must match quant_common.h
  uint32_t rows, cols, row_bytes, rows_per_group, x_group_stride, accumulate, ksplit, n, x_stride, y_stride;
};

const char* kname(QType t) {
  switch (t) {
    case QType::F32: return "f32"; case QType::F16: return "f16"; case QType::BF16: return "bf16";
    case QType::Q8_0: return "q8_0"; case QType::Q5_K: return "q5_k"; case QType::Q6_K: return "q6_k";
    case QType::IQ1_S: return "iq1_s"; case QType::IQ2_XXS: return "iq2_xxs";
    default: return nullptr;
  }
}

// GPU block geometry (dense types use a virtual 256-element block)
int gpu_qk(QType t) { return block_elems(t) == 1 ? 256 : block_elems(t); }

[[noreturn]] void die(const char* what, QType t, int rows, int cols) {
  fprintf(stderr, "macseek quant: %s unsupported (type %s rows %d cols %d)\n", what, type_name(t), rows, cols);
  abort();
}

// lanes cooperating on one block in the matvec kernels (must match quant_blocks.h)
int gpu_lpb(QType t) {
  switch (t) {
    case QType::F32: case QType::F16: case QType::BF16: return 32;
    case QType::Q8_0: return 4;
    default: return 16;
  }
}

// Launch heuristic (from bench_qmatvec --sweep on M5 Pro): split K until each simdgroup runs ~1-2 block iterations
// (memory-level parallelism matters more than x reuse), 256-thread threadgroups, more rows per simdgroup only for
// tall matrices.
MvCfg auto_cfg(QType t, int rows, int cols) {
  MvCfg c;
  const int bpi = 32 / gpu_lpb(t);
  const int iters = (cols / gpu_qk(t) + bpi - 1) / bpi;
  if (t == QType::IQ1_S || t == QType::IQ2_XXS) {  // ALU-bound decode: maximize x reuse instead of K split
    c.nr = rows >= 1024 ? 4 : 1; c.ks = iters >= 8 ? 2 : 1; c.rsg = 2;
    return c;
  }
  c.nr = rows >= 8192 ? 4 : rows >= 2048 ? 2 : 1;
  c.ks = 1;
  while (c.ks < 32 && c.ks * 2 <= iters && (c.ks < 8 || (long)rows / c.nr * c.ks < 4096)) c.ks *= 2;
  c.rsg = std::max(1, 8 / c.ks);
  return c;
}

}  // namespace

bool matvec_supported(QType t) { return kname(t) != nullptr; }

void encode_matvec(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& x, const Buffer& y,
                   bool accumulate, int groups, MvCfg cfg) {
  const char* kn = kname(t);
  if (!kn || cols % gpu_qk(t) != 0 || groups < 1 || rows % groups != 0) die("matvec", t, rows, cols);
  MvCfg a = auto_cfg(t, rows, cols);
  if (cfg.nr) a.nr = cfg.nr;
  if (cfg.rsg) a.rsg = cfg.rsg;
  if (cfg.ks) a.ks = cfg.ks;
  static const long env_nr = env_int("MACSEEK_MV_NR", 0), env_rsg = env_int("MACSEEK_MV_RSG", 0),
                    env_ks = env_int("MACSEEK_MV_KS", 0);
  if (env_nr) a.nr = (int)env_nr;
  if (env_rsg) a.rsg = (int)env_rsg;
  if (env_ks) a.ks = (int)env_ks;
  if (groups > 1) {  // every simdgroup's NR rows must stay inside one group
    while ((rows / groups) % a.nr) a.nr /= 2;
  }
  QMvArgs args{(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes(t, cols), (uint32_t)(rows / groups),
               (uint32_t)cols, accumulate ? 1u : 0u, (uint32_t)a.ks, 1, (uint32_t)cols, (uint32_t)rows};
  std::string fn = std::string("q_mv_") + kn + "_r" + std::to_string(a.nr);
  e.set_pipeline(fn.c_str());
  e.set_buffer(0, W); e.set_buffer(1, x); e.set_buffer(2, y);
  e.set_value(3, args);
  const int rows_per_tg = a.nr * a.rsg;
  e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * a.ks);
  e.dispatch_groups((rows + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * a.rsg * a.ks);
}

void encode_matvec_n(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                     bool accumulate, int groups) {
  if (n == 1) { encode_matvec(e, t, W, rows, cols, X, Y, accumulate, groups); return; }
  const char* kn = kname(t);
  if (!kn || n < 1 || n > 16 || cols % gpu_qk(t) != 0 || groups < 1 || rows % groups != 0) die("matvec_n", t, rows, cols);
  const int nt = n <= 2 ? 2 : n <= 4 ? 4 : n <= 8 ? 8 : 16;
  MvCfg a = auto_cfg(t, rows, cols);
  a.nr = nt >= 8 ? 1 : 2;
  static const long env_nr = env_int("MACSEEK_MVN_NR", 0);
  if (env_nr) a.nr = (int)env_nr;
  if (groups > 1) while ((rows / groups) % a.nr) a.nr /= 2;
  QMvArgs args{(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes(t, cols), (uint32_t)(rows / groups), (uint32_t)cols,
               accumulate ? 1u : 0u, (uint32_t)a.ks, (uint32_t)n, (uint32_t)(cols * groups), (uint32_t)rows};
  std::string fn = std::string("q_mvn_") + kn + "_r" + std::to_string(a.nr) + "_n" + std::to_string(nt);
  e.set_pipeline(fn.c_str());
  e.set_buffer(0, W); e.set_buffer(1, X); e.set_buffer(2, Y);
  e.set_value(3, args);
  const int rows_per_tg = a.nr * a.rsg;
  e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * a.ks * nt);
  e.dispatch_groups((rows + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * a.rsg * a.ks);
}

// ---------------------------------------------------------------- MoE decode
namespace {
struct QMoeArgs {  // must match quant_moe.metal
  uint32_t n_routed, has_shared, dim, inter, rb_gu_r, rb_d_r, rb_gu_s, rb_d_s, ksplit, tiles;
};
const char* moe_pair(QType r, QType s) {
  if (r == QType::IQ1_S && s == QType::Q6_K) return "iq1_s_q6_k";
  if (r == QType::IQ1_S && s == QType::Q8_0) return "iq1_s_q8_0";
  if (r == QType::IQ2_XXS && s == QType::Q6_K) return "iq2_xxs_q6_k";
  if (r == QType::IQ2_XXS && s == QType::Q8_0) return "iq2_xxs_q8_0";
  return nullptr;
}
}  // namespace

size_t moe_decode_scratch_bytes() { return sizeof(float) * (kMaxRouted + 1) * kExpertInter; }

void encode_moe_decode(Enc& e, const MoeDecodeArgs& a) {
  if (a.n_routed < 0 || a.n_routed > kMaxRouted) die("moe_decode n_routed", a.t_gate_up, a.n_routed, 0);
  const int n_exp = a.n_routed + (a.has_shared ? 1 : 0);
  if (n_exp == 0) return;
  const char* p_gu = moe_pair(a.t_gate_up, a.t_shared);
  const char* p_d = moe_pair(a.t_down, a.t_shared);
  if (!p_gu || !p_d) die("moe_decode type combo", a.t_gate_up, 0, 0);
  if (a.scratch.size < moe_decode_scratch_bytes()) die("moe_decode scratch too small", a.t_gate_up, 0, 0);
  static const long env_gu_nr = env_int("MACSEEK_MOE_GU_NR", 2), env_gu_ks = env_int("MACSEEK_MOE_GU_KS", 2),
                    env_gu_rsg = env_int("MACSEEK_MOE_GU_RSG", 2), env_d_nr = env_int("MACSEEK_MOE_D_NR", 2),
                    env_d_ks = env_int("MACSEEK_MOE_D_KS", 2), env_d_rsg = env_int("MACSEEK_MOE_D_RSG", 2);
  QMoeArgs args{(uint32_t)a.n_routed, a.has_shared ? 1u : 0u, (uint32_t)kExpertDim, (uint32_t)kExpertInter,
                (uint32_t)row_bytes(a.t_gate_up, kExpertDim), (uint32_t)row_bytes(a.t_down, kExpertInter),
                (uint32_t)row_bytes(a.t_shared, kExpertDim), (uint32_t)row_bytes(a.t_shared, kExpertInter), 1, 0};
  // any valid buffer for unused bindings (never dereferenced)
  const ExpertSlot& any = a.n_routed > 0 ? a.slot[0] : a.shared;
  auto slot_or = [&](int i, const Buffer ExpertSlot::*m) -> const Buffer& {
    return i < a.n_routed ? a.slot[i].*m : any.*m;
  };
  // 1) gate + up + swiglu -> scratch h[n_exp][2304]
  {
    const int nr = (int)env_gu_nr, ks = (int)env_gu_ks, rsg = (int)env_gu_rsg;
    const int rows_per_tg = nr * rsg;
    args.ksplit = ks;
    args.tiles = (kExpertInter + rows_per_tg - 1) / rows_per_tg;
    std::string fn = std::string("q_moe_gu_") + p_gu + "_r" + std::to_string(nr);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, a.x); e.set_buffer(1, a.scratch); e.set_value(2, args);
    e.set_buffer(3, a.weights.buf ? a.weights : a.scratch);
    for (int i = 0; i < kMaxRouted; i++) { e.set_buffer(4 + i, slot_or(i, &ExpertSlot::gate)); e.set_buffer(10 + i, slot_or(i, &ExpertSlot::up)); }
    const ExpertSlot& sh = a.has_shared ? a.shared : any;
    e.set_buffer(16, sh.gate); e.set_buffer(17, sh.up);
    e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * 2 * ks);
    e.dispatch_groups(args.tiles * n_exp, 1, 1, 32 * rsg * ks);
  }
  e.barrier();
  // 2) down for all experts, accumulated into y
  {
    const int nr = (int)env_d_nr, ks = (int)env_d_ks, rsg = (int)env_d_rsg;
    const int rows_per_tg = nr * rsg;
    args.ksplit = ks;
    std::string fn = std::string("q_moe_down_") + p_d + "_r" + std::to_string(nr);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, a.scratch); e.set_buffer(1, a.y); e.set_value(2, args);
    for (int i = 0; i < kMaxRouted; i++) e.set_buffer(4 + i, slot_or(i, &ExpertSlot::down));
    e.set_buffer(10, a.has_shared ? a.shared.down : any.down);
    e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * ks);
    e.dispatch_groups((kExpertDim + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * rsg * ks);
  }
}

}  // namespace ms::q
