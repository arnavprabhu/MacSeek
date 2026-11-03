#pragma once
// arch: model-specific small kernels for DeepSeek-V4.1-Flash (hyper-connections + Sinkhorn, hc_pre/hc_post + RMSNorm,
// MoE router, engram gate, engram/embedding row dequant, argmax/top-k sampling, log_softmax, rmsnorm).
// Host encode functions never commit or wait. Barrier convention: every encode_* inserts barriers between its OWN
// dependent dispatches; the caller inserts e.barrier() between dependent encode_* calls. See docs/modules/arch.md.
#include <cstdint>
#include "core/metal.hpp"

namespace ms::arch {

constexpr int kDim = 5120, kHc = 4, kHcDim = 20480, kMix = 24;
constexpr int kExperts = 384, kTopK = 6;   // kExperts = max routed experts (buffers); Options::n_experts = actual
constexpr int kVocab = 129280;
constexpr int kEngramRows = 24, kEngramRowVals = 256, kEngramRowBytes = 144, kEngramKv = 25600;
constexpr int kEmbedRowBytes = 4200;   // Q6_K row of 5120
constexpr float kNormEps = 1e-20f, kHcEps = 1e-6f, kRouteScale = 1.5f, kEngramClamp = 1e-6f;
constexpr int kSinkhornIters = 20;
constexpr int kSampleK = 64;           // top-k returned by encode_topk (max)

// Small device scratch shared by the single-dispatch ("last threadgroup finishes") kernels: partial records and
// zero-initialized atomic counters (each kernel resets its counter). One Scratch per stream of dependent work; kernel
// families use separate counter slots, but two concurrent (non-barriered) dispatches of the same family need two
// Scratch objects.
struct Scratch {
  Buffer hc_part;        // 64 groups x 32 floats
  Buffer counters;       // uint32 [64 + 2*max_rows]
  Buffer router_logits;  // [384] logits written by encode_router_fused (diagnostic)
  Buffer rstd;           // [max_tokens] prefill rms scales
  Buffer gate;           // [max_tokens][4] engram gates (diagnostic)
  Buffer sample_part;    // argmax / top-k partials
  Buffer dummy;          // bound for unused optional inputs
  int max_tokens = 0, max_rows = 0;
  static Scratch create(int max_tokens = 4096, int max_sample_rows = 16);
};

// Runtime switches (initialized once from env; tests/bench may modify them).
struct Options {
  int hc_mode = 0;             // MACSEEK_ARCH_HC: 0 fused (1 dispatch), 1 split (step + finalize), 2 rows (post_pre + rows)
  int hc_threads = 512;        // MACSEEK_ARCH_HC_T: decode step threadgroup size (128/256/512/1024; 5120/T groups)
  bool hc_prefill_rows = false;  // MACSEEK_ARCH_HC_PREFILL=rows: per-token mixes kernel instead of the MMA GEMM
  bool hc_prefill_mma8 = false;  // MACSEEK_ARCH_HC_PREFILL=mma8: 8-token MMA tiles instead of 32-token tiles
  int router_groups = 96;      // MACSEEK_ARCH_ROUTER_G: fused router threadgroups (divides n_experts)
  int n_experts = 384;         // routed experts per layer (384 or 192); the model sets it from the GGUF
  int hc_row_splits = 1;       // MACSEEK_ARCH_HC_SPLITS: decode step threadgroups per dim chunk (mix rows split; only
                               // when h_out != h_in; bit-identical). HcStep::row_splits overrides.
};
Options& options();

// ---------------- hyper-connections ----------------
// One sublayer boundary (SPEC 3), n tokens:
//   h_new  = x_sub ? post[i]*x_sub + sum_j comb[j][i]*h_in[j] : h_in            (written to h_out if x_sub)
//   mix_out = fn ? hc_split_sinkhorn(fn . h_new * rsqrt(mean(h_new^2)+1e-20), scale, base) : -
//   x_out  = rmsnorm(sum_j mix_prev.pre[j] * h_new[j]) * norm_w
// mix layout [n][24] = pre[4], post[4], comb[4][4] (comb[j][k] at 8+4j+k). mix_prev supplies post/comb of the
// previous hc_mixes AND the lagged pre (attention uses ffn_pre of the previous layer, FFN uses attn_pre).
// Layer 0: mix_prev = identity mix (fill_identity_mix), no x_sub. Final head: no fn, norm_w = output_norm.
// Engram layers (1, 14), between layer L-1's FFN and layer L's attention:
//   A) {x_sub = ffn out, mix_prev = ffn mixes of L-1, no fn, no x_out}     post only
//   B) encode_engram_gate(h)
//   C) {no x_sub, mix_prev = same ffn mixes of L-1 (lagged pre), fn = hc_attn of L, x_out = attention input}
struct HcStep {
  Buffer h_in, h_out;          // [n][4][5120]; h_out may equal h_in
  Buffer x_sub;                // [n][5120] sublayer output (empty: no hc_post)
  Buffer mix_prev;             // [n][24]
  Buffer fn, scale, base;      // hc_*_fn [24][20480], hc_*_scale [3], hc_*_base [24] (fn empty: no mixes)
  Buffer mix_out;              // [n][24], must not alias mix_prev
  Buffer norm_w;               // [5120]
  Buffer x_out;                // [n][5120] (empty: no hc_pre / RMSNorm)
  int n = 1;
  int row_splits = 0;          // decode fused step: threadgroups per dim chunk (0 = Options::hc_row_splits). Needs
                               // h_out != h_in when there is an x_sub (falls back to 1 otherwise). Bit-identical.
};
// Decode (n == 1) mode: MACSEEK_ARCH_HC = fused (default, 1 dispatch) | split (2) | rows (2).
// Prefill (n > 1): post/pre kernel + mixes GEMM (simdgroup MMA for n >= 8, MACSEEK_ARCH_HC_PREFILL=rows to force
// the per-token kernel).
void encode_hc_step(Enc& e, const HcStep& a, const Scratch& s);
void fill_identity_mix(float* mix24);                      // pre = [1,0,0,0], post = 0, comb = 0
void encode_hc_mean(Enc& e, Buffer h, Buffer out, int n);  // out[t][d] = mean_j h[t][j][d] (dspark main_hidden)

// ---------------- router ----------------
// Decode, fused F32 matvec (ffn_gate_inp [384][5120]) + sqrt(softplus) + bias top-6 + normalize*1.5, 1 dispatch.
// ids int32 [6], w f32 [6] (descending by score+bias). Shared storage: CPU reads ids via Buffer::cpu() once the GPU
// work is complete (use a split command buffer / MTLSharedEvent for mid-token expert fetch).
void encode_router_fused(Enc& e, Buffer x, Buffer W, Buffer bias, Buffer ids, Buffer w, const Scratch& s);
// From precomputed logits [n][384] (any n).
void encode_router_topk(Enc& e, Buffer logits, Buffer bias, Buffer ids, Buffer w, int n);
// Prefill grouped-GEMM routing tables, deterministic: entries (t,k) = t*6+k sorted by expert, then entry index.
struct RouteLists {
  Buffer counts;      // u32 [384]
  Buffer offsets;     // u32 [385] exclusive scan, offsets[384] = 6n
  Buffer entry;       // u32 [6n]  entry index t*6+k
  Buffer tok;         // u32 [6n]  token t
  Buffer w;           // f32 [6n]  routing weight
  Buffer slot_of;     // u32 [6n]  position of entry t*6+k in the sorted lists
  int max_tokens = 0;
  static RouteLists create(int max_tokens);
};
void encode_route_lists(Enc& e, Buffer ids, Buffer w, int n, const RouteLists& L);

// ---------------- engram ----------------
// h [n][4][5120] += gate_j * value, kv [n][25600] = engram_kv output (key [4][5120] then value [5120]),
// w = engram_gate [4][5120] (the q_weight*k_weight product). gate_out [n][4] optional (else scratch).
void encode_engram_gate(Enc& e, Buffer h, Buffer kv, Buffer w, int n, const Scratch& s, Buffer gate_out = {});
// IQ4_NL rows (144 B, 256 values) -> F32 [n_rows][256]; src rows row_stride bytes apart (even).
void encode_iq4nl_rows(Enc& e, Buffer src, uint32_t row_stride, Buffer dst, int n_rows);

// ---------------- embedding ----------------
// Q6_K rows of token_embd (4200 B each, row_stride bytes apart) -> h [n][copies][5120] (copies = 4: hc expand).
void encode_embed_q6k(Enc& e, Buffer rows, uint32_t row_stride, Buffer h, int n, int copies = 4);
// Q5_K rows (3520 B each) -> same layout.
void encode_embed_q5k(Enc& e, Buffer rows, uint32_t row_stride, Buffer h, int n, int copies = 4);
// dispatch by ggml type id (14 = Q6_K, 13 = Q5_K)
void encode_embed(Enc& e, int ggml_type, Buffer rows, uint32_t row_stride, Buffer h, int n, int copies = 4);

// ---------------- sampling / logits ----------------
// argmax per row (ties -> lowest index), single dispatch. temperature > 0: Gumbel-max sampling of
// softmax(logits/temperature) (exact multinomial over the full vocab, counter-based RNG keyed by seed).
// out_idx u32 [rows], out_val f32 [rows] (winning (noisy) value).
void encode_argmax(Enc& e, Buffer logits, int rows, int V, Buffer out_idx, Buffer out_val, const Scratch& s,
                   float temperature = 0.0f, uint64_t seed = 0);
// top-k (k <= 64) per row, descending raw logits + ids, and lse[row] = logsumexp(logits/temperature) over the full row.
// out_v f32 [rows][64], out_i u32 [rows][64], out_lse f32 [rows]. Single dispatch.
void encode_topk(Enc& e, Buffer logits, int rows, int V, int k, float temperature, Buffer out_v, Buffer out_i,
                 Buffer out_lse, const Scratch& s);
// out = in - logsumexp(in) per row (out may alias in).
void encode_log_softmax(Enc& e, Buffer in, Buffer out, int rows, int V);
// CPU: temperature + top-p sampling from encode_topk output. p_i = exp(v_i/T - lse); keeps the smallest prefix with
// cumulative mass >= top_p (renormalized); u in [0,1). Returns the token id.
uint32_t sample_top_p(const float* vals, const uint32_t* ids, int k, float lse, float temperature, float top_p,
                      double u);

// ---------------- rmsnorm ----------------
// y = x * rsqrt(mean(x^2) + eps) * w (w empty: no weight). Strides in floats (0 = dim). y may alias x.
void encode_rmsnorm(Enc& e, Buffer x, Buffer w, Buffer y, int rows, int dim, float eps, int in_stride = 0,
                    int out_stride = 0);

}  // namespace ms::arch
