// Host encode functions of the quant module.
#import <Metal/Metal.h>
#include "kernels/quant.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>
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
    const int epl = (t == QType::Q8_0) ? 8 : 16;
    e.dispatch_threads(cols / epl, rows, 1, 64, 1);
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
  const char* p_gu = moe_pair(a.t_gate_up, a.t_shared);
  const char* p_d = moe_pair(a.t_down, a.t_shared);
  if (!p_gu || !p_d) die("moe_decode_table type combo", a.t_gate_up, 0, 0);
  static const long env_gu_nr = env_int("MACSEEK_MOE_GU_NR", 2), env_gu_ks = env_int("MACSEEK_MOE_GU_KS", 1),
                    env_gu_rsg = env_int("MACSEEK_MOE_GU_RSG", 4), env_d_nr = env_int("MACSEEK_MOE_D_NR", 4),
                    env_d_ks = env_int("MACSEEK_MOE_D_KS", 1), env_d_rsg = env_int("MACSEEK_MOE_D_RSG", 2);
  const int n_exp = a.n_routed + (a.has_shared ? 1 : 0);
  QMoeArgs args{(uint32_t)a.n_routed, a.has_shared ? 1u : 0u, (uint32_t)kExpertDim, (uint32_t)kExpertInter,
                (uint32_t)row_bytes(a.t_gate_up, kExpertDim), (uint32_t)row_bytes(a.t_down, kExpertInter),
                (uint32_t)row_bytes(a.t_shared, kExpertDim), (uint32_t)row_bytes(a.t_shared, kExpertInter), 1, 0};
  {
    const int nr = (int)env_gu_nr, ks = (int)env_gu_ks, rsg = (int)env_gu_rsg;
    const int rows_per_tg = nr * rsg;
    args.ksplit = ks;
    args.tiles = (kExpertInter + rows_per_tg - 1) / rows_per_tg;
    std::string fn = std::string("q_moe_gu_t_") + p_gu + "_r" + std::to_string(nr);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, a.x); e.set_buffer(1, a.scratch); e.set_value(2, args);
    e.set_buffer(3, a.weights); e.set_buffer(4, table);
    e.set_buffer(16, a.shared.gate); e.set_buffer(17, a.shared.up);
    e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * 2 * ks);
    e.dispatch_groups(args.tiles * n_exp, 1, 1, 32 * rsg * ks);
  }
  e.barrier();
  {
    const int nr = (int)env_d_nr, ks = (int)env_d_ks, rsg = (int)env_d_rsg;
    const int rows_per_tg = nr * rsg;
    args.ksplit = ks;
    std::string fn = std::string("q_moe_down_t_") + p_d + "_r" + std::to_string(nr);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, a.scratch); e.set_buffer(1, a.y); e.set_value(2, args);
    e.set_buffer(4, table); e.set_buffer(10, a.shared.down);
    e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * ks);
    e.dispatch_groups((kExpertDim + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * rsg * ks);
  }
}

// ---------------------------------------------------------------- MoE prefill
namespace {
constexpr int kMoeTN = 64;
int moe_tm() { static const int v = (int)env_int("MACSEEK_MOEP_TM", 64); return v; }
struct QMoePArgs { uint32_t n, total, dim, inter, rb_gu, rb_d, k_per_token; };
const char* mm_moe_name(QType t) {
  switch (t) {
    case QType::IQ1_S: return "iq1_s"; case QType::IQ2_XXS: return "iq2_xxs";
    case QType::Q6_K: return "q6_k"; case QType::Q8_0: return "q8_0";
    default: return nullptr;
  }
}
size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }
size_t moe_header_bytes(int total, int n, int n_exp) {
  const size_t tiles = (size_t)n_exp + total / 64 + n / 64 + 4;
  return align256(24 * (size_t)(n_exp + 1) + 4 * (size_t)(n_exp + 2) + 8 * tiles);
}
}  // namespace

size_t moe_prefill_scratch_bytes(int total, int n, int n_exp) {
  const size_t rows = (size_t)total + n;
  return moe_header_bytes(total, n, n_exp) + align256(rows * kExpertDim * 2) + align256(rows * kExpertInter * 2) +
         align256(rows * kExpertDim * 4);
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
  uint32_t* tiles = (uint32_t*)(hdr + 24 * (ne + 1) + 4 * (ne + 2));
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
  const int tm = moe_tm();
  int nt_r = 0;
  for (int j = 0; j < ne; j++)
    for (int t0 = 0; t0 < a.offs[j + 1] - a.offs[j]; t0 += tm) { tiles[2 * nt_r] = j; tiles[2 * nt_r + 1] = t0; nt_r++; }
  int nt_s = 0;
  if (a.has_shared)
    for (int t0 = 0; t0 < n; t0 += tm) { tiles[2 * (nt_r + nt_s)] = ne; tiles[2 * (nt_r + nt_s) + 1] = t0; nt_s++; }
  const size_t rows = (size_t)total + n;
  const Buffer s_hdr = a.scratch.sub(0, hbytes);
  const Buffer s_xg = a.scratch.sub(hbytes, rows * kExpertDim * 2);
  const size_t off_h = hbytes + align256(rows * kExpertDim * 2);
  const Buffer s_h = a.scratch.sub(off_h, rows * kExpertInter * 2);
  const size_t off_z = off_h + align256(rows * kExpertInter * 2);
  const Buffer s_z = a.scratch.sub(off_z, rows * kExpertDim * 4);
  const Buffer b_ptrs = s_hdr;
  const Buffer b_offs = a.scratch.sub(24 * (ne + 1), 4 * (ne + 2));
  const Buffer b_tiles_r = a.scratch.sub(24 * (ne + 1) + 4 * (ne + 2), 8 * (size_t)std::max(nt_r, 1));
  const Buffer b_tiles_s = a.scratch.sub(24 * (ne + 1) + 4 * (ne + 2) + 8 * (size_t)nt_r, 8 * (size_t)std::max(nt_s, 1));
  id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)e.encoder();
  for (void* b : bufs) [enc useResource:(__bridge id<MTLBuffer>)b usage:MTLResourceUsageRead];

  QMoePArgs args{(uint32_t)n, (uint32_t)total, (uint32_t)kExpertDim, (uint32_t)kExpertInter, 0, 0, (uint32_t)a.k_per_token};
  // 1) gather + convert
  e.set_pipeline("q_moe_gather");
  e.set_buffer(0, a.X); e.set_buffer(1, s_xg); e.set_buffer(2, a.tok.buf ? a.tok : a.scratch); e.set_value(3, args);
  e.dispatch_threads(kExpertDim / 4, (uint32_t)(a.has_shared ? rows : total), 1, 64, 1);
  e.barrier();
  // 2) gate + up + swiglu -> H (routed and shared dispatches run concurrently)
  auto gu = [&](const char* kn, QType t, const Buffer& tl, int nt) {
    if (nt == 0) return;
    std::string fn = std::string("q_moe_mm_gu_") + kn + "_m" + std::to_string(tm);
    QMoePArgs ar = args; ar.rb_gu = (uint32_t)row_bytes(t, kExpertDim);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, s_xg); e.set_buffer(1, s_h); e.set_value(2, ar); e.set_buffer(3, a.wt.buf ? a.wt : a.scratch);
    e.set_buffer(4, b_ptrs); e.set_buffer(5, b_offs); e.set_buffer(6, tl);
    e.set_threadgroup_memory(0, 2 * 2 * kMoeTN * 128);
    e.dispatch_groups(kExpertInter / kMoeTN, nt, 1, 128);
  };
  gu(k_gu, a.t_gate_up, b_tiles_r, nt_r);
  gu(k_s, a.t_shared, b_tiles_s, nt_s);
  e.barrier();
  // 3) down -> Z
  auto dn = [&](const char* kn, QType t, const Buffer& tl, int nt) {
    if (nt == 0) return;
    std::string fn = std::string("q_moe_mm_down_") + kn + "_m" + std::to_string(tm);
    QMoePArgs ar = args; ar.rb_d = (uint32_t)row_bytes(t, kExpertInter);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, s_h); e.set_buffer(1, s_z); e.set_value(2, ar);
    e.set_buffer(4, b_ptrs); e.set_buffer(5, b_offs); e.set_buffer(6, tl);
    e.set_threadgroup_memory(0, 2 * 2 * kMoeTN * 128);
    e.dispatch_groups(kExpertDim / kMoeTN, nt, 1, 128);
  };
  dn(k_d, a.t_down, b_tiles_r, nt_r);
  dn(k_s, a.t_shared, b_tiles_s, nt_s);
  e.barrier();
  // 4) deterministic reduction into Y
  uint32_t hs = a.has_shared ? 1u : 0u;
  e.set_pipeline("q_moe_reduce");
  e.set_buffer(0, s_z); e.set_buffer(1, a.Y); e.set_value(2, args);
  e.set_buffer(3, a.slot_of.buf ? a.slot_of : a.scratch); e.set_value(4, hs);
  e.dispatch_threads(kExpertDim / 4, n, 1, 64, 1);
}

}  // namespace ms::q
