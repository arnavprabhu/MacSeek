#pragma once
// Decode (n = 1) per-layer encoding: the attention side (hc boundary, projections, attention, o-proj, hc boundary,
// router) and the table-variant MoE with the router handshake (docs/modules/dec_gpu.md). Used by Model::forward's
// table decode path and by bench/decode_layer.mm (synthetic weights) so both run exactly the same sequence.
//
// DecOptions::fuse = 0 reproduces the previous decode sequence dispatch for dispatch (the old lambda in ds41.mm is kept
// behind MACSEEK_DEC_LAYER=0). The fused variants are documented per switch in dec_gpu.md.
#include <cstdint>
#include <vector>
#include "core/metal.hpp"
#include "kernels/arch.hpp"
#include "kernels/attn.hpp"
#include "kernels/quant_types.hpp"
#include "model/config.hpp"

namespace ms::ds41 {

struct W {  // one weight: buffer + kernel type + shape
  Buffer b; q::QType t = q::QType::F32; int rows = 0, cols = 0;
  explicit operator bool() const { return (bool)b; }
};
struct LayerW {
  W attn_norm, ffn_norm, q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b, sinks;
  W c_kv, c_gate, c_norm, idx_qb, idx_proj, idx_k, idx_knorm;
  W gate_inp, bias;
  W sh_gate, sh_up, sh_down;
  W hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
  W engram_kv, engram_gate;
};

// Activation buffers of one decode token (all F32, shared storage).
struct DecBufs {
  Buffer h;            // [4][5120]
  Buffer h2;           // [4][5120] second hyper-connection state for ping-pong hc steps (hc_splits > 1); may be empty
  Buffer x, sub;       // [5120]
  Buffer mixA, mixF;   // [24]
  Buffer mix_ident;    // optional [24] identity mix read by layer 0 instead of mixF (several tokens per command buffer)
  Buffer qa, qrn, q, kv, ckv, cgate, iq, iw, aout, oa;
  Buffer rids, rw;     // router ids int32[6], weights [6]
  Buffer plog;         // [pred_depth][384] predicted router logits of the next layers
  Buffer ekv[2];       // engram kv [25600] of engram layers 0/1 (cfg.engram_layers)
  Buffer moe_scratch;  // >= q::moe_decode_scratch_bytes()
  Buffer edeq;         // engram rows dequantized, [24][256] per engram layer (prologue)
};

// Switches (env, read once by from_env). fuse = MACSEEK_DEC_FUSE master switch: reorder + combine + sgu_router 1 +
// engram_bg 2 + hc_splits 2 (every one bit-identical); each can be overridden (MACSEEK_DEC_<NAME>=value).
struct DecOptions {
  bool fuse = false;
  bool reorder = false;    // MACSEEK_DEC_REORDER: kv window/compressor in the q_a-norm stage, fill(sub) in the router stage
  bool norm_qb = false;    // MACSEEK_DEC_NORMQB: q_a_norm RMSNorm fused into the q_b / indexer q_b x loads (not in
                           // fuse: measured slower, every q_b threadgroup replays the 1280-element norm reduction)
  int norm_qb_tpt = 0;     // row tiles per threadgroup of the norm-fused matvecs (0 = MACSEEK_MV_RN_TPT, default 8)
  bool attn_comb = false;  // MACSEEK_DEC_COMBINE: sparse attention split-K combine done by the last threadgroup
  int sgu_router = 0;      // MACSEEK_DEC_SGU_ROUTER: 1 = shared expert gate/up in the router stage (the handshake gap
                           // keeps only the shared down projection); 2 = also the shared down after the go wait, next
                           // to the routed gate/up (no gap work: fewest stages, exposes the handshake). Bit-identical.
  int hc_splits = 1;       // MACSEEK_DEC_HC_SPLIT: threadgroups per dim chunk of the fused hc step (mix rows split, 2x
                           // the parallelism of the 10-threadgroup step; h ping-pongs between B.h and B.h2)
  int engram_bg = 0;       // MACSEEK_DEC_ENGRAM_BG: engram_kv matvecs run as background chunks inside latency-bound
                           // stages (n chunks per engram layer; 0 = off: both up front as before)
  int moe_gu_nr = 0, moe_gu_rsg = 0, moe_d_nr = 0, moe_d_rsg = 0;   // MACSEEK_DEC_MOE_{GU,D}_{NR,RSG}: table-MoE launch
                           // shapes (0 = quant module default; bit-identical for any value)
  int probe_dispatches = 0;  // PROFILING (bench): extra tiny dispatches per layer in the router stage (dispatch cost)
  bool actq_out = false;   // ModelOptions::actq_out
  bool pred_on = true;     // MACSEEK_PRED
  int pred_depth = 1;      // MACSEEK_PRED_DEPTH
  bool pred_q8 = true;     // MACSEEK_PRED_Q8
  uint32_t skip = 0;       // PROFILING ONLY (bench): DecSkip bits not encoded (wrong outputs)
  static DecOptions from_env();
};

enum DecSkip : uint32_t {
  kSkipHcA = 1u << 0, kSkipProj = 1u << 1, kSkipNorm = 1u << 2, kSkipQb = 1u << 3, kSkipKvWin = 1u << 4,
  kSkipIndex = 1u << 5, kSkipSparse = 1u << 6, kSkipOa = 1u << 7, kSkipOb = 1u << 8, kSkipHcF = 1u << 9,
  kSkipRouter = 1u << 10, kSkipPred = 1u << 11, kSkipFill = 1u << 12, kSkipMoeShared = 1u << 13,
  kSkipMoeRouted = 1u << 14, kSkipBarriers = 1u << 15, kSkipEvents = 1u << 16, kSkipEngram = 1u << 17,
};

// Everything the per-layer encoders need (non-owning).
struct DecCtx {
  const Config* cfg = nullptr;
  const std::vector<LayerW>* L = nullptr;
  const std::vector<W>* pred_w = nullptr;   // Q8_0 router copies (prediction), may be empty
  DecBufs B;
  attn::State* ast = nullptr;
  const arch::Scratch* asc = nullptr;
  DecOptions o;
  struct Bg { int chunks = 0, next = 0; } bg[2];   // engram_bg: chunk plan per engram layer (dec_bg_begin)
  int hcur = 0;            // which of B.h / B.h2 holds the current hyper-connection state (reset by the prologue)
};
// the buffer holding the current hyper-connection state h (after the last encoded hc step)
inline Buffer dec_cur_h(const DecCtx& c) { return c.hcur ? c.B.h2 : c.B.h; }

// Table-MoE handshake: per layer the GPU signals rdy = value_base + 1 + l after the router and waits go (same value)
// before the routed experts. mode 0 event | 1 sgu | 2 shared (docs/modules/dec_sync.md). rdy/go = id<MTLSharedEvent>;
// both null = no handshake (the whole MoE right after the router; profiling / bench).
struct DecSync {
  void* rdy = nullptr;
  void* go = nullptr;
  uint64_t value_base = 0;
  int mode = 2;
};

// Background work interleaved into the layer stages (engram_bg): call dec_bg_begin once per token after the
// engram rows are dequantized; encode_dec_* then emit chunks of it into latency-bound stages.
void dec_bg_begin(DecCtx& c);
void dec_bg_flush(Enc& e, DecCtx& c, int upto_engram);  // encode all remaining chunks of engram layers < upto

// Token prologue after the embedding: engram rows (IQ4_NL, already in `erow`) -> edeq -> engram_kv for both engram
// layers (or only the dequant when engram_bg is on: the matvecs are deferred into the layer stages).
void encode_dec_prologue(Enc& e, DecCtx& c, const Buffer& erow);
// Attention side of layer l for the token at position pos (ends after the router; sub zeroed for the MoE).
void encode_dec_attn_side(Enc& e, DecCtx& c, int l, int pos);
// MoE of layer l, table variant: routed experts read from `table` ([6][3] device addresses, CPU-filled after rdy).
// `arena` = buffers the table may point into (made resident). Ends with a barrier.
void encode_dec_moe_table(Enc& e, DecCtx& c, int l, q::QType t_gate_up, q::QType t_down, const Buffer& table,
                          const std::vector<Buffer>& arena, const DecSync& s);

}  // namespace ms::ds41
