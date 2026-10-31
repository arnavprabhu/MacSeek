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

// Prefill: Y[n x rows] (+)= X[n x cols] . W^T for n >= 32 (any n works). Dequantized tiles + tensor-API / simdgroup
// matrix multiply, F32 accumulate. X/Y F32 row-major.
void encode_matmul(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int n,
                   bool accumulate = false);

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

// Prefill MoE (grouped GEMM over experts). Routing is given in expert-sorted order: for routed expert j (0..n_exp-1)
// its tokens are entries [offs[j], offs[j+1]) of tok[] (token row in X) and wt[] (routing weight).
struct MoePrefillArgs {
  QType t_gate_up = QType::IQ1_S, t_down = QType::IQ1_S;
  int n_tokens = 0;                 // rows of X / Y
  int n_exp = 0;                    // number of experts with >= 1 token
  const ExpertSlot* slots = nullptr;  // [n_exp]
  const int* offs = nullptr;        // host [n_exp + 1] (host needs counts to size the grids)
  Buffer tok;                       // device int32[total] token ids, expert-sorted (total = offs[n_exp])
  Buffer wt;                        // device F32[total] routing weights, same order
  Buffer slot_of;                   // device int32[n_tokens * k_per_token]: for token t, its entries' positions in
  int k_per_token = 6;              //   tok[] (-1 = unused); used by the deterministic final reduction
  bool has_shared = true;
  QType t_shared = QType::Q6_K;
  ExpertSlot shared;
  Buffer X;                         // F32[n_tokens x 5120]
  Buffer Y;                         // F32[n_tokens x 5120], ACCUMULATED
  Buffer scratch;                   // >= moe_prefill_scratch_bytes(total, n_tokens)
};
size_t moe_prefill_scratch_bytes(int total_entries, int n_tokens);
void encode_moe_prefill(Enc& e, const MoePrefillArgs& a);

}  // namespace ms::q
