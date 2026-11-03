#pragma once
// Quantized matrix kernels (quant module): host-side encode functions. See docs/modules/quant.md.
// Weight buffer offsets must be >= 4-byte aligned (IQ1_M / Q5_K decoders use 32-bit loads).
// All functions only encode into the given Enc (never commit/wait). Weights are raw GGUF block rows: a [in -> out]
// matrix is `rows = out` rows of `cols = in` elements. Activations/outputs are F32 unless documented otherwise.
#include "core/metal.hpp"
#include "kernels/quant_types.hpp"

namespace ms::q {

// Launch shape for the matvec family: nr rows per simdgroup, rsg row-simdgroups per threadgroup, ks simdgroups
// splitting K inside a threadgroup (threadgroup = rsg*ks simdgroups). 0 = auto (tuned table, see quant.mm).
struct MvCfg { int nr = 0, rsg = 0, ks = 0; };

bool matvec_supported(QType t);

// Launch shape encode_matvec would use for this matrix (auto table + MACSEEK_MV_* overrides): splitting a matrix into
// row chunks that all use this shape (chunk rows a multiple of nr*rsg) gives bit-identical rows.
MvCfg matvec_cfg(QType t, int rows, int cols);
// The auto launch rule without env overrides; version 0 = the generic rule, 1 = with the Q5_K rule (default,
// MACSEEK_MV_CFG). Benchmarks compare versions in one process.
MvCfg matvec_cfg_rule(QType t, int rows, int cols, int version);

// Decode matvec with the input RMSNorm fused into the x loads: y = W . (x * rsqrt(mean(x^2) + eps) * norm_w), x has
// `cols` elements. Bit-identical to arch::encode_rmsnorm(x, norm_w, tmp, 1, cols, eps) -> barrier -> encode_matvec(tmp)
// for cols > 1024 (the RMSNorm reduction is replayed exactly); types F32 F16 BF16 Q8_0 Q5_K Q6_K.
// tiles_per_tg: row tiles (nr*rsg rows) per threadgroup, amortizing the norm prologue (0 = MACSEEK_MV_RN_TPT, default 8).
void encode_matvec_rmsnorm(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& x, const Buffer& norm_w,
                           float eps, const Buffer& y, int tiles_per_tg = 0);

// y[rows] (+)= W[rows x cols] . x[cols]. groups > 1: block-diagonal, rows split into `groups` equal row groups and
// group g reads x[g*cols .. g*cols+cols) (attn_output_a: groups=8, cols=4096, rows=8192, x has 32768 elements).
void encode_matvec(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& x, const Buffer& y,
                   bool accumulate = false, int groups = 1, MvCfg cfg = {});

// Y[n x rows] (+)= X[n x cols] . W^T for n in 1..16 (one weight read serves all n columns). X row stride = cols
// (or groups*cols for block-diagonal), Y row stride = rows.
void encode_matvec_n(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                     bool accumulate = false, int groups = 1);

// Prefill: Y[n x rows] (+)= X[n x cols] . W^T for n >= 32 (any n works). Weight tiles are dequantized to half in
// threadgroup memory and multiplied with the Metal 4 tensor API (MPP matmul2d, F32 accumulate). The tensor units need
// half activations: encode_matmul converts X into an internal module-owned scratch (barrier before + after);
// when one X feeds several matmuls, convert once with encode_f32_to_f16 and call encode_matmul_h.
void encode_matmul(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                   bool accumulate = false);
void encode_f32_to_f16(Enc& e, const Buffer& X, const Buffer& X16, size_t count);  // count % 4 == 0
// X16: half[n x x_stride] (x_stride = cols if 0), Y: F32 [n x rows]
void encode_matmul_h(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X16, const Buffer& Y, int n,
                     bool accumulate = false, int x_stride = 0);

// ---- MoE ----
constexpr int kExpertDim = 5120, kExpertInter = 2304, kMaxRouted = 6;

// One routed expert slot: three contiguous raw-block byte ranges (gate [5120->2304], up [5120->2304],
// down [2304->5120]), each an ms::Buffer (buf + byte offset) -- may live in the same arena buffer.
struct ExpertSlot { Buffer gate, up, down; };

struct MoeDecodeArgs {
  QType t_gate_up = QType::IQ1_S;   // routed gate/up type: IQ1_S, IQ2_XXS, IQ1_M or IQ2_XS
  QType t_down = QType::IQ1_S;      // routed down type (same set; independent of t_gate_up)
  int n_routed = 6;                 // 0..6
  ExpertSlot slot[kMaxRouted];
  Buffer weights;                   // F32[n_routed] routing weights (device buffer, may be GPU-written)
  bool has_shared = true;
  QType t_shared = QType::Q6_K;     // shared expert type (Q6_K, Q8_0 or Q5_K), weight 1
  ExpertSlot shared;
  Buffer x;                         // F32[5120] (normed FFN input)
  Buffer y;                         // F32[5120], result is ACCUMULATED into y
  Buffer scratch;                   // >= moe_decode_scratch_bytes()
  // launch shape overrides of the table variants (0 = MACSEEK_MOE_* env / default). Rows per simdgroup (nr) and
  // row-simdgroups per threadgroup (rsg) do not change any row's summation order: results are bit-identical.
  int gu_nr = 0, gu_rsg = 0, d_nr = 0, d_rsg = 0;
  // K-split simdgroups per row (0 = MACSEEK_MOE_{GU,D}_KS / default): changes the summation grouping (not bit-identical)
  int gu_ks = 0, d_ks = 0;
};
size_t moe_decode_scratch_bytes();
// Two dispatches (gate+up+swiglu for all experts incl. shared, then down for all experts accumulated into y) with a
// barrier between them. swiglu: u = clamp(up, -10, 10), g = min(gate, 10), h = silu(g) * u * weight.
void encode_moe_decode(Enc& e, const MoeDecodeArgs& a);
// Same, but routed expert matrices come from `table`: kMaxRouted entries of 3 device addresses {gate, up, down}
// (uint64, filled by the CPU before the GPU runs this). a.slot[] is ignored. The caller makes the arena resident.
void encode_moe_decode_table(Enc& e, const MoeDecodeArgs& a, const Buffer& table);
// Pieces of encode_moe_decode_table, so independent work can run while the CPU resolves the routed experts
// (decode handshake overlap, MACSEEK_DEC_SYNC):
//   SharedGu then AfterSharedGu  == All, bit-identical (each h element is computed by the same thread code)
//   Shared then Routed           == All up to float summation order (y += shared; y += routed instead of one sum)
//   SharedGu then SharedDown     == Shared, bit-identical (the shared expert's gate/up and its down projection alone)
// SharedGu / Shared / SharedDown never read `table`.
enum class MoePart { All, SharedGu, AfterSharedGu, Shared, Routed, SharedDown };
void encode_moe_decode_table_part(Enc& e, const MoeDecodeArgs& a, const Buffer& table, MoePart part);

// Prefill MoE (grouped GEMM over experts, tensor API). Routing is given in expert-sorted order: routed expert j
// (0..n_exp-1, any subset of the layer's experts) owns entries [offs[j], offs[j+1]) of tok[] (token row in X) and
// wt[] (routing weight). slot_of[t*k_per_token + k] = entry index of token t's k-th routed expert (-1 = none); the
// final reduction sums Y[t] += shared(t) + sum_k Z[slot_of[t][k]] in that fixed order (deterministic).
// Scratch layout (caller-owned, shared storage; must not be in use by an in-flight command buffer because the host
// writes the expert pointer / tile tables into its header at encode time):
//   header | Xg half[(total+n) x 5120] | H half[(total+n) x 2304] | Z f32[(total+n) x 5120] | W16 half[2 x 2304 x 5120]
// (W16: the shared expert's gate|up, later down, dequantized to half for the dense GEMMs)
struct MoePrefillArgs {
  QType t_gate_up = QType::IQ1_S, t_down = QType::IQ1_S;  // routed: IQ1_S / IQ2_XXS / IQ1_M / IQ2_XS, any combination
  int n_tokens = 0;                   // rows of X / Y
  int n_exp = 0;                      // routed experts with >= 1 token
  const ExpertSlot* slots = nullptr;  // host [n_exp]
  const int* offs = nullptr;          // host [n_exp + 1], offs[0] = 0, offs[n_exp] = total
  Buffer tok;                         // device int32[total]
  Buffer wt;                          // device F32[total]
  Buffer slot_of;                     // device int32[n_tokens * k_per_token]
  int k_per_token = 6;
  bool has_shared = true;
  QType t_shared = QType::Q6_K;       // Q6_K / Q8_0 / Q5_K
  ExpertSlot shared;
  Buffer X;                           // F32[n_tokens x 5120]
  Buffer Y;                           // F32[n_tokens x 5120], ACCUMULATED
  Buffer scratch;                     // >= moe_prefill_scratch_bytes(total, n_tokens, n_exp)
  // kernel selection (-1 = env / default): path 1 = per-simdgroup register-dequant kernels (MACSEEK_MOEP_PATH, default),
  // 0 = threadgroup dequant-tile kernels (previous implementation); shared_dense 1 = shared expert through half
  // pre-dequantization + dense GEMMs (MACSEEK_MOEP_SHDENSE, default), 0 = through the grouped kernels
  int path = -1;
  int shared_dense = -1;
};
size_t moe_prefill_scratch_bytes(int total_entries, int n_tokens, int n_exp = 384);
void encode_moe_prefill(Enc& e, const MoePrefillArgs& a);

}  // namespace ms::q
