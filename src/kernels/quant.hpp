#pragma once
// Quantized matrix kernels (quant module): host-side encode functions. See docs/modules/quant.md.
// All functions only encode into the given Enc (never commit/wait). Weights are raw GGUF block rows: a [in -> out]
// matrix is `rows = out` rows of `cols = in` elements. Activations/outputs are F32 unless documented otherwise.
#include "core/metal.hpp"
#include "kernels/quant_types.hpp"

namespace ms::q {

// Launch shape for the matvec family: nr rows per simdgroup, rsg row-simdgroups per threadgroup, ks simdgroups
// splitting K inside a threadgroup (threadgroup = rsg*ks simdgroups). 0 = auto (tuned table, see quant.mm).
struct MvCfg { int nr = 0, rsg = 0, ks = 0; };

bool matvec_supported(QType t);

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
  QType t_gate_up = QType::IQ1_S;   // routed gate/up type (IQ1_S or IQ2_XXS)
  QType t_down = QType::IQ1_S;      // routed down type (IQ1_S or IQ2_XXS)
  int n_routed = 6;                 // 0..6
  ExpertSlot slot[kMaxRouted];
  Buffer weights;                   // F32[n_routed] routing weights (device buffer, may be GPU-written)
  bool has_shared = true;
  QType t_shared = QType::Q6_K;     // shared expert type (Q6_K or Q8_0), weight 1
  ExpertSlot shared;
  Buffer x;                         // F32[5120] (normed FFN input)
  Buffer y;                         // F32[5120], result is ACCUMULATED into y
  Buffer scratch;                   // >= moe_decode_scratch_bytes()
};
size_t moe_decode_scratch_bytes();
// Two dispatches (gate+up+swiglu for all experts incl. shared, then down for all experts accumulated into y) with a
// barrier between them. swiglu: u = clamp(up, -10, 10), g = min(gate, 10), h = silu(g) * u * weight.
void encode_moe_decode(Enc& e, const MoeDecodeArgs& a);

// Prefill MoE (grouped GEMM over experts, tensor API). Routing is given in expert-sorted order: routed expert j
// (0..n_exp-1, any subset of the layer's experts) owns entries [offs[j], offs[j+1]) of tok[] (token row in X) and
// wt[] (routing weight). slot_of[t*k_per_token + k] = entry index of token t's k-th routed expert (-1 = none); the
// final reduction sums Y[t] += shared(t) + sum_k Z[slot_of[t][k]] in that fixed order (deterministic).
// Scratch layout (caller-owned, shared storage; must not be in use by an in-flight command buffer because the host
// writes the expert pointer / tile tables into its header at encode time):
//   header | Xg half[(total+n) x 5120] | H half[(total+n) x 2304] | Z f32[(total+n) x 5120]
struct MoePrefillArgs {
  QType t_gate_up = QType::IQ1_S, t_down = QType::IQ1_S;
  int n_tokens = 0;                   // rows of X / Y
  int n_exp = 0;                      // routed experts with >= 1 token
  const ExpertSlot* slots = nullptr;  // host [n_exp]
  const int* offs = nullptr;          // host [n_exp + 1], offs[0] = 0, offs[n_exp] = total
  Buffer tok;                         // device int32[total]
  Buffer wt;                          // device F32[total]
  Buffer slot_of;                     // device int32[n_tokens * k_per_token]
  int k_per_token = 6;
  bool has_shared = true;
  QType t_shared = QType::Q6_K;
  ExpertSlot shared;
  Buffer X;                           // F32[n_tokens x 5120]
  Buffer Y;                           // F32[n_tokens x 5120], ACCUMULATED
  Buffer scratch;                     // >= moe_prefill_scratch_bytes(total, n_tokens, n_exp)
};
size_t moe_prefill_scratch_bytes(int total_entries, int n_tokens, int n_exp = 384);
void encode_moe_prefill(Enc& e, const MoePrefillArgs& a);

}  // namespace ms::q
