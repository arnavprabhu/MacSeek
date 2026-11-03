// Host encode functions of the quant module.
#import <Metal/Metal.h>
#include "kernels/quant.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>
#include "core/env.hpp"
#include "kernels/quant_grids.h"

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
    case QType::IQ1_M: return "iq1_m"; case QType::IQ2_XS: return "iq2_xs";
    default: return nullptr;
  }
}
// grid-table (IQ) types: ALU-bound decode, routed-expert types of the MoE kernels
bool is_iq(QType t) { return t == QType::IQ1_S || t == QType::IQ2_XXS || t == QType::IQ1_M || t == QType::IQ2_XS; }
// shared-expert types of the MoE kernels
bool is_moe_shared(QType t) { return t == QType::Q6_K || t == QType::Q8_0 || t == QType::Q5_K; }

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
// elements per lane slice (q_deq_f16 grid width = cols / epl)
int gpu_epl(QType t) { return gpu_qk(t) / gpu_lpb(t); }

// Launch heuristic (from bench_qmatvec --sweep on M5 Pro): split K until each simdgroup runs ~1-2 block iterations
// (memory-level parallelism matters more than x reuse), 256-thread threadgroups, more rows per simdgroup only for
// tall matrices.
MvCfg auto_cfg(QType t, int rows, int cols, int version) {
  MvCfg c;
  const int bpi = 32 / gpu_lpb(t);
  const int iters = (cols / gpu_qk(t) + bpi - 1) / bpi;
  if (t == QType::Q5_K && version >= 1) {
    // Q5_K (GSQ-RCO Coder attention / head / engram / shared expert), bench_qmatvec --sweep: one row per simdgroup and
    // a K split leaving each simdgroup ~2-3 block iterations (2 blocks each), 1-simdgroup row tiles; short rows
    // (attn_q_b, 5 blocks): 4 rows per simdgroup, no K split. 95-99% of BW vs 85-95% with the generic rule.
    const int nb = cols / gpu_qk(t);
    c.nr = nb <= 8 ? 4 : 1;
    c.ks = 1;
    while (c.ks < 16 && 8 * c.ks <= nb) c.ks *= 2;
    c.rsg = c.ks == 1 ? 2 : 1;
    return c;
  }
  if (t == QType::IQ2_XS) {  // sweep: K split and 4 rows/simdgroup on short rows lose 10-20% (register pressure)
    c.nr = rows >= 1024 ? (cols >= 4096 ? 4 : 2) : 1; c.ks = 1; c.rsg = 2;
    return c;
  }
  if (is_iq(t)) {  // ALU-bound decode: maximize x reuse instead of K split
    c.nr = rows >= 1024 ? 4 : 1; c.ks = iters >= 8 ? 2 : 1; c.rsg = 2;
    return c;
  }
  c.nr = rows >= 8192 ? 4 : rows >= 2048 ? 2 : 1;
  c.ks = 1;
  while (c.ks < 32 && c.ks * 2 <= iters && (c.ks < 8 || (long)rows / c.nr * c.ks < 4096)) c.ks *= 2;
  c.rsg = std::max(1, 8 / c.ks);
  return c;
}

// launch-rule version: MACSEEK_MV_CFG (1 = current: Q5_K rule above; 0 = the generic rule for every type)
int cfg_version() { static const int v = (int)env_int("MACSEEK_MV_CFG", 1); return v; }

}  // namespace

MvCfg matvec_cfg_rule(QType t, int rows, int cols, int version) { return auto_cfg(t, rows, cols, version); }

bool matvec_supported(QType t) { return kname(t) != nullptr; }

void encode_matvec(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& x, const Buffer& y,
                   bool accumulate, int groups, MvCfg cfg) {
  const char* kn = kname(t);
  if (!kn || cols % gpu_qk(t) != 0 || groups < 1 || rows % groups != 0) die("matvec", t, rows, cols);
  MvCfg a = auto_cfg(t, rows, cols, cfg_version());
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

MvCfg matvec_cfg(QType t, int rows, int cols) {
  MvCfg a = auto_cfg(t, rows, cols, cfg_version());
  static const long env_nr = env_int("MACSEEK_MV_NR", 0), env_rsg = env_int("MACSEEK_MV_RSG", 0),
                    env_ks = env_int("MACSEEK_MV_KS", 0);
  if (env_nr) a.nr = (int)env_nr;
  if (env_rsg) a.rsg = (int)env_rsg;
  if (env_ks) a.ks = (int)env_ks;
  return a;
}

void encode_matvec_rmsnorm(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& x, const Buffer& norm_w,
                           float eps, const Buffer& y, int tiles_per_tg) {
  const char* kn = kname(t);
  if (!kn || is_iq(t) || cols % gpu_qk(t) != 0 || cols <= 1024)
    die("matvec_rmsnorm", t, rows, cols);
  const MvCfg a = matvec_cfg(t, rows, cols);
  QMvArgs args{(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes(t, cols), (uint32_t)rows, (uint32_t)cols, 0u,
               (uint32_t)a.ks, 1, (uint32_t)cols, (uint32_t)rows};
  // arch_rmsnorm launch for one row of `cols`: T = min(1024, max(32, ceil(cols/4 / 32) * 32)) threads
  static const int env_tpt = (int)env_int("MACSEEK_MV_RN_TPT", 8);
  const int tpt = std::max(1, tiles_per_tg > 0 ? tiles_per_tg : env_tpt);
  struct { uint32_t dim; float eps; uint32_t nt, tpt; } na{(uint32_t)cols, eps,
      (uint32_t)std::min(1024, std::max(32, (cols / 4 + 31) / 32 * 32)), (uint32_t)tpt};
  std::string fn = std::string("q_mv_rn_") + kn + "_r" + std::to_string(a.nr);
  e.set_pipeline(fn.c_str());
  e.set_buffer(0, W); e.set_buffer(1, x); e.set_buffer(2, y);
  e.set_value(3, args); e.set_buffer(4, norm_w); e.set_bytes(5, &na, sizeof na);
  const int rows_per_tile = a.nr * a.rsg, ntiles = (rows + rows_per_tile - 1) / rows_per_tile;
  e.set_threadgroup_memory(0, sizeof(float) * rows_per_tile * a.ks);
  e.dispatch_groups((ntiles + tpt - 1) / tpt, 1, 1, 32 * a.rsg * a.ks);
}

void encode_matvec_n(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                     bool accumulate, int groups) {
  if (n == 1) { encode_matvec(e, t, W, rows, cols, X, Y, accumulate, groups); return; }
  // n >= 8: FP32 ALU work (n FMAs per weight) exceeds the weight-streaming time; the tensor-API tile path wins
  static const long mm_min = env_int("MACSEEK_MVN_MM_MIN", 8);
  if (mm_min > 0 && n >= mm_min && groups == 1 && cols % 128 == 0 && kname(t)) {
    encode_matmul(e, t, W, rows, cols, X, Y, n, accumulate);
    return;
  }
  const char* kn = kname(t);
  if (!kn || n < 1 || n > 16 || cols % gpu_qk(t) != 0 || groups < 1 || rows % groups != 0) die("matvec_n", t, rows, cols);
  const int nt = n <= 2 ? 2 : n <= 4 ? 4 : n <= 8 ? 8 : 16;
  static const long env_nr = env_int("MACSEEK_MVN_NR", 0), env_sg = env_int("MACSEEK_MVN_SG", 0);
  int nr = env_nr ? (int)env_nr : (nt >= 8 ? 1 : 2);
  int nsg = env_sg ? (int)env_sg : 8;
  while (nsg > 1 && (long)rows / (nr * nsg) < 160) nsg /= 2;  // keep enough threadgroups for small matrices
  if (groups > 1) while ((rows / groups) % (nr * nsg)) { if (nsg > 1) nsg /= 2; else nr /= 2; }
  QMvArgs args{(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes(t, cols), (uint32_t)(rows / groups), (uint32_t)cols,
               accumulate ? 1u : 0u, 1u, (uint32_t)n, (uint32_t)(cols * groups), (uint32_t)rows};
  std::string fn = std::string("q_mvn_") + kn + "_r" + std::to_string(nr) + "_n" + std::to_string(nt);
  e.set_pipeline(fn.c_str());
  e.set_buffer(0, W); e.set_buffer(1, X); e.set_buffer(2, Y);
  e.set_value(3, args);
  const int ck = (32 / gpu_lpb(t)) * gpu_qk(t);
  e.set_threadgroup_memory(0, sizeof(uint16_t) * n * ck);
  const int rows_per_tg = nr * nsg;
  e.dispatch_groups((rows + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * nsg);
}

// ---------------------------------------------------------------- prefill matmul
void encode_matmul(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                   bool accumulate);
namespace {
struct QMmArgs { uint32_t n, rows, cols, row_bytes, x_stride, y_stride, accumulate; };
struct MmTile { int tm, tn, nsg, nbuf; };
// fused dequant-tile kernel shape: 64-token tiles for small n, 128-token tiles otherwise (bench_qmatmul)
MmTile mm_tile(int n) {
  static const std::string env = env_str("MACSEEK_MM_TILE", "");
  MmTile r = n <= 32 ? MmTile{32, 64, 4, 2} : n <= 96 ? MmTile{64, 64, 4, 2} : MmTile{128, 64, 4, 2};
  if (!env.empty() && sscanf(env.c_str(), "%dx%d_b%d", &r.tm, &r.tn, &r.nbuf) == 3) {
    r.nsg = (r.tm == 128 && r.tn == 128) ? 8 : 4;
    int sg = 0;
    if (sscanf(env.c_str(), "%*dx%*d_b%*d_s%d", &sg) == 1) r.nsg = sg;
  }
  return r;
}
Buffer& mm_scratch(size_t bytes) {
  static Buffer b;
  if (b.size < bytes) b = Device::get().alloc(std::max(bytes, (size_t)64 << 20), "quant.mm.x16");
  return b;
}
Buffer& deq_scratch(size_t bytes) {
  static Buffer b;
  if (b.size < bytes) b = Device::get().alloc(std::max(bytes, (size_t)64 << 20), "quant.mm.w16");
  return b;
}
// n at/above which the weight is dequantized to half once and multiplied by the dense GEMM (0 = never)
int predeq_min_n() { static const int v = (int)env_int("MACSEEK_MM_PREDEQ_N", 512); return v; }
}  // namespace

void encode_f32_to_f16(Enc& e, const Buffer& X, const Buffer& X16, size_t count) {
  uint32_t n4 = (uint32_t)(count / 4);
  e.set_pipeline("q_f32_to_f16");
  e.set_buffer(0, X); e.set_buffer(1, X16); e.set_value(2, n4);
  e.dispatch_threads(n4, 1, 1, 256);
}

void encode_matmul_h(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X16, const Buffer& Y, int n,
                     bool accumulate, int x_stride) {
  const char* kn = kname(t);
  if (!kn || cols % 128 != 0 || n < 1) die("matmul", t, rows, cols);
  QMmArgs args{(uint32_t)n, (uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes(t, cols),
               (uint32_t)(x_stride ? x_stride : cols), (uint32_t)rows, accumulate ? 1u : 0u};
  std::string fn;
  int tm, tn, nsg;
  const bool predeq = block_elems(t) > 1 && predeq_min_n() > 0 && n >= predeq_min_n() &&
                      (size_t)rows * cols * 2 <= ((size_t)1 << 30);
  if (predeq) {  // dequantize W once to half (shared module scratch; barriers around it), then the dense GEMM
    Buffer& w16 = deq_scratch((size_t)rows * cols * 2);
    e.barrier();
    uint32_t d[4] = {(uint32_t)rows, (uint32_t)cols, args.row_bytes, 0};
    std::string dn = std::string("q_deq_f16_") + kn;
    e.set_pipeline(dn.c_str());
    e.set_buffer(0, W); e.set_buffer(1, w16); e.set_bytes(2, d, 16);
    e.dispatch_threads(cols / gpu_epl(t), rows, 1, 64, 1);
    e.barrier();
    e.set_pipeline("q_mm_f16_64x128");
    args.row_bytes = (uint32_t)(cols * 2);
    e.set_buffer(0, w16); e.set_buffer(1, X16); e.set_buffer(2, Y);
    e.set_value(3, args);
    e.dispatch_groups((rows + 127) / 128, (n + 63) / 64, 1, 128);
    return;
  }
  if (block_elems(t) == 1) {  // dense: right operand straight from device memory
    tm = 64; tn = 128; nsg = 4;
    fn = std::string("q_mm_") + kn + "_64x128";
    e.set_pipeline(fn.c_str());
  } else {
    MmTile mt = mm_tile(n);
    tm = mt.tm; tn = mt.tn; nsg = mt.nsg;
    fn = std::string("q_mm_") + kn + "_" + std::to_string(tm) + "x" + std::to_string(tn) + "_b" + std::to_string(mt.nbuf) +
         (nsg == 4 || (tm == 128 && tn == 128) ? "" : "_s" + std::to_string(nsg));
    e.set_pipeline(fn.c_str());
    e.set_threadgroup_memory(0, sizeof(uint16_t) * tn * 128 * mt.nbuf);
  }
  e.set_buffer(0, W); e.set_buffer(1, X16); e.set_buffer(2, Y);
  e.set_value(3, args);
  e.dispatch_groups((rows + tn - 1) / tn, (n + tm - 1) / tm, 1, 32 * nsg);
}

void encode_matmul(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                   bool accumulate) {
  if (cols % 4) die("matmul (cols % 4)", t, rows, cols);
  Buffer& x16 = mm_scratch((size_t)n * cols * 2);
  e.barrier();  // earlier users of the shared scratch must finish
  encode_f32_to_f16(e, X, x16, (size_t)n * cols);
  e.barrier();
  encode_matmul_h(e, t, W, rows, cols, x16, Y, n, accumulate);
}

// ---------------------------------------------------------------- MoE decode
namespace {
struct QMoeArgs {  // must match quant_moe.metal
  uint32_t n_routed, has_shared, dim, inter, rb_gu_r, rb_d_r, rb_gu_s, rb_d_s, ksplit, tiles;
};
// kernel-name suffix "<routed>_<shared>" (e.g. "iq1_m_q5_k"); empty if the pair is not instantiated
// (quant_moe.metal Q_MOE_PAIRS: routed IQ1_S / IQ2_XXS / IQ1_M / IQ2_XS x shared Q6_K / Q8_0 / Q5_K)
std::string moe_pair(QType r, QType s) {
  if (!is_iq(r) || !is_moe_shared(s)) return {};
  return std::string(kname(r)) + "_" + kname(s);
}
// table (decode) variants: MACSEEK_MOE_DEC_V=0 selects the previous IQ1_M decoder and 16-lane IQ2_XXS gate/up
// (instantiated with the Coder's Q5_K shared expert)
std::string moe_pair_t(QType r, QType s) {
  static const bool v0 = env_int("MACSEEK_MOE_DEC_V", 1) == 0;
  if (v0 && r == QType::IQ1_M && s == QType::Q5_K) return "iq1_m0_q5_k";
  if (v0 && r == QType::IQ2_XXS && s == QType::Q5_K) return "iq2_xxs0_q5_k";
  return moe_pair(r, s);
}
}  // namespace

size_t moe_decode_scratch_bytes() { return sizeof(float) * (kMaxRouted + 1) * kExpertInter; }

void encode_moe_decode(Enc& e, const MoeDecodeArgs& a) {
  if (a.n_routed < 0 || a.n_routed > kMaxRouted) die("moe_decode n_routed", a.t_gate_up, a.n_routed, 0);
  const int n_exp = a.n_routed + (a.has_shared ? 1 : 0);
  if (n_exp == 0) return;
  const std::string p_gu = moe_pair(a.t_gate_up, a.t_shared), p_d = moe_pair(a.t_down, a.t_shared);
  if (p_gu.empty() || p_d.empty()) die("moe_decode type combo", a.t_gate_up, 0, 0);
  if (a.scratch.size < moe_decode_scratch_bytes()) die("moe_decode scratch too small", a.t_gate_up, 0, 0);
  static const long env_gu_nr = env_int("MACSEEK_MOE_GU_NR", 2), env_gu_ks = env_int("MACSEEK_MOE_GU_KS", 1),
                    env_gu_rsg = env_int("MACSEEK_MOE_GU_RSG", 4), env_d_nr = env_int("MACSEEK_MOE_D_NR", 4),
                    env_d_ks = env_int("MACSEEK_MOE_D_KS", 1), env_d_rsg = env_int("MACSEEK_MOE_D_RSG", 2);
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

void encode_moe_decode_table(Enc& e, const MoeDecodeArgs& a, const Buffer& table) {
  encode_moe_decode_table_part(e, a, table, MoePart::All);
}

void encode_moe_decode_table_part(Enc& e, const MoeDecodeArgs& a, const Buffer& table, MoePart part) {
  const std::string p_gu = moe_pair_t(a.t_gate_up, a.t_shared), p_d = moe_pair_t(a.t_down, a.t_shared);
  if (p_gu.empty() || p_d.empty()) die("moe_decode_table type combo", a.t_gate_up, 0, 0);
  if ((part == MoePart::SharedGu || part == MoePart::Shared || part == MoePart::SharedDown) && !a.has_shared) return;
  static const long env_gu_nr = env_int("MACSEEK_MOE_GU_NR", 2), env_gu_ks = env_int("MACSEEK_MOE_GU_KS", 1),
                    env_gu_rsg = env_int("MACSEEK_MOE_GU_RSG", 4), env_d_nr = env_int("MACSEEK_MOE_D_NR", 4),
                    env_d_ks = env_int("MACSEEK_MOE_D_KS", 1), env_d_rsg = env_int("MACSEEK_MOE_D_RSG", 2);
  // which pieces this call encodes
  const bool gu_r = part == MoePart::All || part == MoePart::AfterSharedGu || part == MoePart::Routed;
  const bool gu_s = a.has_shared && (part == MoePart::All || part == MoePart::SharedGu || part == MoePart::Shared);
  const bool d_r = gu_r;
  const bool d_s = a.has_shared && (part == MoePart::All || part == MoePart::AfterSharedGu || part == MoePart::Shared ||
                                    part == MoePart::SharedDown);
  const bool shared_only = !gu_r;   // n_routed = 0 views: scratch rebased to the shared expert's h row
  const Buffer h = shared_only ? a.scratch.sub((size_t)a.n_routed * kExpertInter * 4, (size_t)kExpertInter * 4) : a.scratch;
  QMoeArgs args{shared_only ? 0u : (uint32_t)a.n_routed, d_s ? 1u : 0u, (uint32_t)kExpertDim, (uint32_t)kExpertInter,
                (uint32_t)row_bytes(a.t_gate_up, kExpertDim), (uint32_t)row_bytes(a.t_down, kExpertInter),
                (uint32_t)row_bytes(a.t_shared, kExpertDim), (uint32_t)row_bytes(a.t_shared, kExpertInter), 1, 0};
  const int n_gu = (gu_r ? a.n_routed : 0) + (gu_s ? 1 : 0);
  if (n_gu > 0) {
    // IQ2_XXS routed gate/up (8-lane decoder): 4 rows per simdgroup x 4 (bench --moe-sweep: -2%; nr/rsg are bit-identical)
    static const bool env_set = getenv("MACSEEK_MOE_GU_NR") || getenv("MACSEEK_MOE_GU_RSG");
    const bool x4 = !env_set && gu_r && a.t_gate_up == QType::IQ2_XXS;
    const int nr = a.gu_nr ? a.gu_nr : x4 ? 4 : (int)env_gu_nr, ks = a.gu_ks ? a.gu_ks : (int)env_gu_ks,
              rsg = a.gu_rsg ? a.gu_rsg : x4 ? 4 : (int)env_gu_rsg;
    const int rows_per_tg = nr * rsg;
    args.ksplit = ks;
    args.tiles = (kExpertInter + rows_per_tg - 1) / rows_per_tg;
    std::string fn = std::string("q_moe_gu_t_") + p_gu + "_r" + std::to_string(nr);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, a.x); e.set_buffer(1, h); e.set_value(2, args);
    e.set_buffer(3, a.weights); e.set_buffer(4, table);
    e.set_buffer(16, a.shared.gate); e.set_buffer(17, a.shared.up);
    e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * 2 * ks);
    // threadgroup e*tiles.. computes expert e; e >= args.n_routed is the shared expert, so the routed-only variant
    // just dispatches n_routed experts' tiles (same per-row code => bit-identical to the fused dispatch)
    e.dispatch_groups(args.tiles * n_gu, 1, 1, 32 * rsg * ks);
  }
  if (!d_r && !d_s) return;
  if (n_gu > 0) e.barrier();
  {
    const int nr = a.d_nr ? a.d_nr : (int)env_d_nr, ks = a.d_ks ? a.d_ks : (int)env_d_ks, rsg = a.d_rsg ? a.d_rsg : (int)env_d_rsg;
    const int rows_per_tg = nr * rsg;
    args.ksplit = ks;
    std::string fn = std::string("q_moe_down_t_") + p_d + "_r" + std::to_string(nr);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, h); e.set_buffer(1, a.y); e.set_value(2, args);
    e.set_buffer(4, table); e.set_buffer(10, a.shared.down);
    e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * ks);
    e.dispatch_groups((kExpertDim + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * rsg * ks);
  }
}

// ---------------------------------------------------------------- MoE prefill
namespace {
constexpr int kMoeTN = 64;
int moe_tm() { static const int v = (int)env_int("MACSEEK_MOEP_TM", 64); return v; }
// MoE prefill kernel family: 0 = threadgroup dequant tiles (q_moe_mm_*), 1 = per-simdgroup register dequant (q_moe_r*)
int moe_path() { static const int v = (int)env_int("MACSEEK_MOEP_PATH", 1); return v; }
struct MoeRKern { int tm, tn, tk, nsg; };
// shared expert through pre-dequantized half weights + dense GEMMs (1, default) or the grouped kernels (0)
bool moe_shared_dense() { static const bool v = env_int("MACSEEK_MOEP_SHDENSE", 1) != 0; return v; }
// register-path launch shapes (bench_qmatmul --real), "TMxTNkTKsNSG" ("0" = class off). Token-tile classes of 64, 48,
// 32 and 16 tokens: an expert with m tokens gets ceil(m/64) tiles (one weight dequantization per tile), full 64-token
// tiles plus the smallest class that holds the remainder (tensor work padded to 16 rows). Defaults per routed type
// (gate/up K step 64 for IQ1_S / IQ2_XS, 32 for IQ2_XXS / IQ1_M: their dequant needs more registers).
constexpr int kMoeTileCls = 4;
MoeRKern moe_rk(QType t, bool down, int cls) {
  static const char* envs[2][kMoeTileCls] = {
      {"MACSEEK_MOEP_RGU64", "MACSEEK_MOEP_RGU48", "MACSEEK_MOEP_RGU", "MACSEEK_MOEP_RGU16"},
      {"MACSEEK_MOEP_RD64", "MACSEEK_MOEP_RD48", "MACSEEK_MOEP_RD", "MACSEEK_MOEP_RD16"}};
  const int tk_gu = (t == QType::IQ1_S || t == QType::IQ2_XS) ? 64 : 32;  // IQ1_M: K 64 spills (-12%)
  static const MoeRKern dg[kMoeTileCls] = {{64, 16, 0, 4}, {48, 16, 0, 4}, {32, 32, 0, 4}, {16, 32, 0, 4}};
  static const MoeRKern dd[kMoeTileCls] = {{64, 32, 32, 4}, {48, 32, 32, 4}, {32, 64, 32, 4}, {16, 64, 32, 4}};
  MoeRKern d = down ? dd[cls] : dg[cls];
  if (!down && d.tm) d.tk = tk_gu;
  std::string v = env_str(envs[down][cls], "");
  if (v == "0") return MoeRKern{0, 0, 0, 0};
  if (!v.empty()) sscanf(v.c_str(), "%dx%dk%ds%d", &d.tm, &d.tn, &d.tk, &d.nsg);
  return d;
}
struct QMoePArgs { uint32_t n, total, dim, inter, rb_gu, rb_d, k_per_token; };
// device copy of the IQ grid tables for the register-path kernels (QMoeTbl in quant_moe_prefill.metal): IQ1_S half
// grid (8192 uint) | IQ2_XXS half grid (1024) | IQ2 sign masks (512) | IQ2_XS half grid (2048). Filled on the CPU,
// read-only afterwards. Offsets must match kQTbl* in quant_moe_prefill.metal.
const Buffer& moe_tables() {
  static const Buffer b = [] {
    static const uint32_t g1[] = {Q_IQ1S_GRID_HALF_VALUES}, g2[] = {Q_IQ2XXS_GRID_HALF_VALUES}, s2[] = {Q_IQ2_SIGNH_VALUES},
                          g3[] = {Q_IQ2XS_GRID_HALF_VALUES};
    static_assert(sizeof(g1) == 8192 * 4 && sizeof(g2) == 1024 * 4 && sizeof(s2) == 512 * 4 && sizeof(g3) == 2048 * 4,
                  "grid table sizes");
    Buffer t = Device::get().alloc(sizeof(g1) + sizeof(g2) + sizeof(s2) + sizeof(g3), "quant.moe.tbl");
    uint8_t* p = (uint8_t*)t.cpu();
    memcpy(p, g1, sizeof(g1)); p += sizeof(g1);
    memcpy(p, g2, sizeof(g2)); p += sizeof(g2);
    memcpy(p, s2, sizeof(s2)); p += sizeof(s2);
    memcpy(p, g3, sizeof(g3));
    return t;
  }();
  return b;
}
// threadgroup memory of the register-path kernels: the IQ2_XXS grid is copied there
size_t moe_tg_bytes(QType t) { return t == QType::IQ2_XXS ? 4096 : 16; }
const char* mm_moe_name(QType t) { return is_iq(t) || is_moe_shared(t) ? kname(t) : nullptr; }
size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }
size_t moe_header_bytes(int total, int n, int n_exp) {
  const size_t tiles = (size_t)n_exp + total / 16 + n / 16 + 4;  // token tiles of >= 16
  return align256(24 * (size_t)(n_exp + 1) + 4 * (size_t)(n_exp + 2) + 8 * tiles);
}
}  // namespace

size_t moe_prefill_scratch_bytes(int total, int n, int n_exp) {
  const size_t rows = (size_t)total + n;
  return moe_header_bytes(total, n, n_exp) + align256(rows * kExpertDim * 2) + align256(rows * kExpertInter * 2) +
         align256(rows * kExpertDim * 4) + (size_t)2 * kExpertInter * kExpertDim * 2;  // + shared expert gate|up half
}

void encode_moe_prefill(Enc& e, const MoePrefillArgs& a) {
  const int n = a.n_tokens, ne = a.n_exp;
  if (n <= 0) return;
  const int total = ne > 0 ? a.offs[ne] : 0;
  const char *k_gu = mm_moe_name(a.t_gate_up), *k_d = mm_moe_name(a.t_down), *k_s = mm_moe_name(a.t_shared);
  if (!k_gu || !k_d || !k_s) die("moe_prefill types", a.t_gate_up, 0, 0);
  if (a.scratch.size < moe_prefill_scratch_bytes(total, n, ne)) die("moe_prefill scratch too small", a.t_gate_up, total, n);
  // ---- host-written header: expert pointer table, offsets, tile tables
  uint8_t* hdr = (uint8_t*)a.scratch.cpu();
  const size_t hbytes = moe_header_bytes(total, n, ne);
  uint64_t* ptrs = (uint64_t*)hdr;
  uint32_t* offs = (uint32_t*)(hdr + 24 * (ne + 1));
  uint32_t* tiles = (uint32_t*)(hdr + 24 * (ne + 1) + 4 * (ne + 2));  // uint2 {expert, first token}
  std::unordered_set<void*> bufs;
  auto addr = [&](const Buffer& b) -> uint64_t {
    bufs.insert(b.buf);
    return [(__bridge id<MTLBuffer>)b.buf gpuAddress] + b.off;
  };
  for (int j = 0; j < ne; j++) {
    ptrs[3 * j] = addr(a.slots[j].gate); ptrs[3 * j + 1] = addr(a.slots[j].up); ptrs[3 * j + 2] = addr(a.slots[j].down);
    offs[j] = (uint32_t)a.offs[j];
  }
  offs[ne] = (uint32_t)total;
  offs[ne + 1] = (uint32_t)(total + n);
  if (a.has_shared) { ptrs[3 * ne] = addr(a.shared.gate); ptrs[3 * ne + 1] = addr(a.shared.up); ptrs[3 * ne + 2] = addr(a.shared.down); }
  const int path = a.path >= 0 ? a.path : moe_path();
  // routed tile classes; kernels for gate/up use a.t_gate_up's shapes, down a.t_down's
  MoeRKern kg[kMoeTileCls], kd[kMoeTileCls];
  int cls_tm[kMoeTileCls];  // token-tile size per class (0 = off)
  for (int c = 0; c < kMoeTileCls; c++) {
    kg[c] = moe_rk(a.t_gate_up, false, c); kd[c] = moe_rk(a.t_down, true, c);
    if (path == 1 && ((kg[c].tm == 0) != (kd[c].tm == 0) || kg[c].tm != kd[c].tm))
      die("moe_prefill: gate/up and down token tiles differ", a.t_gate_up, kg[c].tm, kd[c].tm);
    cls_tm[c] = kg[c].tm;
  }
  constexpr int kMid = 2;  // the 32-token class (always on; also the shared expert's tiles on the grouped path)
  if (path == 1 && kg[kMid].tm == 0) die("moe_prefill: MACSEEK_MOEP_RGU/_RD cannot be 0", a.t_gate_up, 0, 0);
  const int tm = path == 1 ? kg[kMid].tm : moe_tm();
  const bool sh_dense = a.has_shared && (a.shared_dense >= 0 ? a.shared_dense != 0 : moe_shared_dense());
  // tile tables: [class 0..3 routed | shared (grouped-kernel shared expert only)], uint2 {expert, first token}
  std::vector<uint32_t> tl[kMoeTileCls + 1];
  int tmax = tm, cmax = kMid;
  if (path == 1) for (int c = 0; c < kMoeTileCls; c++) if (cls_tm[c] > tmax) { tmax = cls_tm[c]; cmax = c; }
  for (int j = 0; j < ne; j++) {
    const int cnt = a.offs[j + 1] - a.offs[j];
    int t0 = 0;
    if (path != 1) {
      for (; t0 < cnt; t0 += tm) { tl[kMid].push_back(j); tl[kMid].push_back(t0); }
      continue;
    }
    for (; cnt - t0 > tmax; t0 += tmax) { tl[cmax].push_back(j); tl[cmax].push_back(t0); }
    int best = cmax;  // smallest enabled class holding the remainder
    for (int c = 0; c < kMoeTileCls; c++) if (cls_tm[c] > 0 && cls_tm[c] >= cnt - t0 && cls_tm[c] < cls_tm[best]) best = c;
    tl[best].push_back(j); tl[best].push_back(t0);
  }
  if (a.has_shared && !sh_dense)
    for (int t0 = 0; t0 < n; t0 += tm) { tl[kMoeTileCls].push_back(ne); tl[kMoeTileCls].push_back(t0); }
  int ntl[kMoeTileCls + 1];
  size_t toff[kMoeTileCls + 1], tw = 0;
  for (int c = 0; c <= kMoeTileCls; c++) {
    ntl[c] = (int)tl[c].size() / 2;
    memcpy(tiles + tw, tl[c].data(), tl[c].size() * 4);
    toff[c] = tw * 4; tw += tl[c].size();
  }
  const size_t rows = (size_t)total + n;
  const Buffer s_hdr = a.scratch.sub(0, hbytes);
  const Buffer s_xg = a.scratch.sub(hbytes, rows * kExpertDim * 2);
  const size_t off_h = hbytes + align256(rows * kExpertDim * 2);
  const Buffer s_h = a.scratch.sub(off_h, rows * kExpertInter * 2);
  const size_t off_z = off_h + align256(rows * kExpertInter * 2);
  const Buffer s_z = a.scratch.sub(off_z, rows * kExpertDim * 4);
  const size_t off_w16 = off_z + align256(rows * kExpertDim * 4);
  const Buffer s_w16 = a.scratch.sub(off_w16, (size_t)2 * kExpertInter * kExpertDim * 2);
  auto deq16 = [&](QType t, const Buffer& W, int r, int c, size_t dst_off) {  // whole matrix -> half in s_w16
    uint32_t d[4] = {(uint32_t)r, (uint32_t)c, (uint32_t)row_bytes(t, c), 0};
    e.set_pipeline((std::string("q_deq_f16_") + kname(t)).c_str());
    e.set_buffer(0, W); e.set_buffer(1, s_w16.sub(dst_off, (size_t)r * c * 2)); e.set_bytes(2, d, 16);
    e.dispatch_threads(c / gpu_epl(t), r, 1, 64, 1);
  };
  const Buffer b_ptrs = s_hdr;
  const Buffer b_offs = a.scratch.sub(24 * (ne + 1), 4 * (ne + 2));
  const size_t off_t = 24 * (ne + 1) + 4 * (ne + 2);
  Buffer b_tl[kMoeTileCls + 1];
  for (int c = 0; c <= kMoeTileCls; c++) b_tl[c] = a.scratch.sub(off_t + toff[c], 8 * (size_t)std::max(ntl[c], 1));
  id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)e.encoder();
  for (void* b : bufs) [enc useResource:(__bridge id<MTLBuffer>)b usage:MTLResourceUsageRead];

  // PROFILING ONLY (wrong output): MACSEEK_MOEP_SKIP bit mask 1 gather, 2 routed gu, 4 shared gu, 8 routed down,
  // 16 shared down, 32 reduce, 64 << c: routed tile class c
  static const int skip = (int)env_int("MACSEEK_MOEP_SKIP", 0);
  QMoePArgs args{(uint32_t)n, (uint32_t)total, (uint32_t)kExpertDim, (uint32_t)kExpertInter, 0, 0, (uint32_t)a.k_per_token};
  // 1) gather + convert (+ shared gate/up -> half)
  if (!(skip & 1)) {
    e.set_pipeline("q_moe_gather");
    e.set_buffer(0, a.X); e.set_buffer(1, s_xg); e.set_buffer(2, a.tok.buf ? a.tok : a.scratch); e.set_value(3, args);
    e.dispatch_threads(kExpertDim / 4, (uint32_t)(a.has_shared ? rows : total), 1, 64, 1);
  }
  if (sh_dense && !(skip & 4)) {
    deq16(a.t_shared, a.shared.gate, kExpertInter, kExpertDim, 0);
    deq16(a.t_shared, a.shared.up, kExpertInter, kExpertDim, (size_t)kExpertInter * kExpertDim * 2);
  }
  e.barrier();
  auto rname = [&](const char* kind, const char* kn, const MoeRKern& k) {
    return std::string(kind) + kn + "_" + std::to_string(k.tm) + "x" + std::to_string(k.tn) + "k" + std::to_string(k.tk) +
           "s" + std::to_string(k.nsg);
  };
  // 2) gate + up + swiglu -> H (all dispatches of a phase run concurrently)
  auto gu = [&](const char* kn, QType t, const Buffer& tl, int nt, const MoeRKern& k) {
    if (nt == 0) return;
    QMoePArgs ar = args; ar.rb_gu = (uint32_t)row_bytes(t, kExpertDim);
    e.set_buffer(0, s_xg); e.set_buffer(1, s_h); e.set_value(2, ar); e.set_buffer(3, a.wt.buf ? a.wt : a.scratch);
    e.set_buffer(4, b_ptrs); e.set_buffer(5, b_offs); e.set_buffer(6, tl);
    if (path == 1) {
      e.set_buffer(7, moe_tables());
      e.set_threadgroup_memory(0, moe_tg_bytes(t));  // IQ2 grid copy
      e.set_pipeline(rname("q_moe_rgu_", kn, k).c_str());
      const int per_tg = k.tn * k.nsg;
      e.dispatch_groups((kExpertInter + per_tg - 1) / per_tg, nt, 1, 32 * k.nsg);
    } else {
      e.set_pipeline((std::string("q_moe_mm_gu_") + kn + "_m" + std::to_string(tm)).c_str());
      e.set_threadgroup_memory(0, 2 * 2 * kMoeTN * 128);
      e.dispatch_groups(kExpertInter / kMoeTN, nt, 1, 128);
    }
  };
  if (!(skip & 2))
    for (int c = 0; c < kMoeTileCls; c++) if (!(skip & (64 << c))) gu(k_gu, a.t_gate_up, b_tl[c], ntl[c], kg[c]);
  if (!(skip & 4)) {
    if (sh_dense) {
      e.set_pipeline("q_moe_dgu");
      e.set_buffer(0, s_xg); e.set_buffer(1, s_h); e.set_value(2, args); e.set_buffer(3, s_w16);
      e.dispatch_groups(kExpertInter / 64, (n + 63) / 64, 1, 128);
    } else {
      gu(k_s, a.t_shared, b_tl[kMoeTileCls], ntl[kMoeTileCls], moe_rk(a.t_shared, false, kMid));
    }
  }
  e.barrier();
  // 3) down -> Z
  auto dn = [&](const char* kn, QType t, const Buffer& tl, int nt, const MoeRKern& k) {
    if (nt == 0) return;
    QMoePArgs ar = args; ar.rb_d = (uint32_t)row_bytes(t, kExpertInter);
    e.set_buffer(0, s_h); e.set_buffer(1, s_z); e.set_value(2, ar);
    e.set_buffer(4, b_ptrs); e.set_buffer(5, b_offs); e.set_buffer(6, tl);
    if (path == 1) {
      e.set_buffer(7, moe_tables());
      e.set_threadgroup_memory(0, moe_tg_bytes(t));
      e.set_pipeline(rname("q_moe_rdown_", kn, k).c_str());
      const int per_tg = k.tn * k.nsg;
      e.dispatch_groups((kExpertDim + per_tg - 1) / per_tg, nt, 1, 32 * k.nsg);
    } else {
      e.set_pipeline((std::string("q_moe_mm_down_") + kn + "_m" + std::to_string(tm)).c_str());
      e.set_threadgroup_memory(0, 2 * 2 * kMoeTN * 128);
      e.dispatch_groups(kExpertDim / kMoeTN, nt, 1, 128);
    }
  };
  if (!(skip & 8))
    for (int c = 0; c < kMoeTileCls; c++) if (!(skip & (64 << c))) dn(k_d, a.t_down, b_tl[c], ntl[c], kd[c]);
  if (sh_dense) {  // W16 is free once the gate/up phase is done: dequantize down, then the dense GEMM into Z rows
    if (!(skip & 16)) deq16(a.t_shared, a.shared.down, kExpertDim, kExpertInter, 0);
    e.barrier();
    if (!(skip & 16)) {
      QMmArgs mm{(uint32_t)n, (uint32_t)kExpertDim, (uint32_t)kExpertInter, (uint32_t)(kExpertInter * 2),
                 (uint32_t)kExpertInter, (uint32_t)kExpertDim, 0u};
      e.set_pipeline("q_mm_f16_64x128");
      e.set_buffer(0, s_w16); e.set_buffer(1, s_h.sub((size_t)total * kExpertInter * 2, (size_t)n * kExpertInter * 2));
      e.set_buffer(2, s_z.sub((size_t)total * kExpertDim * 4, (size_t)n * kExpertDim * 4));
      e.set_value(3, mm);
      e.dispatch_groups(kExpertDim / 128, (n + 63) / 64, 1, 128);
    }
  } else if (!(skip & 16)) {
    dn(k_s, a.t_shared, b_tl[kMoeTileCls], ntl[kMoeTileCls], moe_rk(a.t_shared, true, kMid));
  }
  e.barrier();
  // 4) deterministic reduction into Y
  if (skip & 32) return;
  uint32_t hs = a.has_shared ? 1u : 0u;
  e.set_pipeline("q_moe_reduce");
  e.set_buffer(0, s_z); e.set_buffer(1, a.Y); e.set_value(2, args);
  e.set_buffer(3, a.slot_of.buf ? a.slot_of : a.scratch); e.set_value(4, hs);
  e.dispatch_threads(kExpertDim / 4, n, 1, 64, 1);
}

}  // namespace ms::q
