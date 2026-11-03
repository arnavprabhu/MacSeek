// Decode per-layer encoding (see dec_layer.hpp, docs/modules/dec_gpu.md).
#import <Metal/Metal.h>
#include "model/dec_layer.hpp"

#include <algorithm>
#include <cstdio>
#include "core/env.hpp"
#include "kernels/quant.hpp"

namespace ms::ds41 {

using q::QType;

DecOptions DecOptions::from_env() {
  DecOptions o;
  o.fuse = env_int("MACSEEK_DEC_FUSE", 0) != 0;
  const long f = o.fuse ? 1 : 0;
  o.reorder = env_int("MACSEEK_DEC_REORDER", f) != 0;
  o.norm_qb = env_int("MACSEEK_DEC_NORMQB", 0) != 0;
  o.attn_comb = env_int("MACSEEK_DEC_COMBINE", f) != 0;
  o.sgu_router = (int)env_int("MACSEEK_DEC_SGU_ROUTER", f);
  o.engram_bg = (int)env_int("MACSEEK_DEC_ENGRAM_BG", o.fuse ? 2 : 0);
  o.hc_splits = (int)env_int("MACSEEK_DEC_HC_SPLIT", o.fuse ? 2 : 1);
  o.moe_gu_nr = (int)env_int("MACSEEK_DEC_MOE_GU_NR", 0); o.moe_gu_rsg = (int)env_int("MACSEEK_DEC_MOE_GU_RSG", 0);
  o.moe_d_nr = (int)env_int("MACSEEK_DEC_MOE_D_NR", 0); o.moe_d_rsg = (int)env_int("MACSEEK_DEC_MOE_D_RSG", 0);
  o.actq_out = env_int("MACSEEK_ACTQ_OUT", env_int("MACSEEK_ACTQ", 0)) != 0;
  o.pred_on = env_int("MACSEEK_PRED", 1) != 0;
  o.pred_depth = std::min(4, (int)env_int("MACSEEK_PRED_DEPTH", 1));
  o.pred_q8 = env_int("MACSEEK_PRED_Q8", 1) != 0;
  o.skip = (uint32_t)env_int("MACSEEK_DEC_SKIP", 0);
  return o;
}

namespace {

inline bool sk(const DecCtx& c, uint32_t bit) { return (c.o.skip & bit) != 0; }
inline void bar(Enc& e, const DecCtx& c) { if (!sk(c, kSkipBarriers)) e.barrier(); }

void matvec(Enc& e, const W& w, const Buffer& x, const Buffer& y) {
  q::encode_matvec(e, w.t, w.b, w.rows, w.cols, x, y, false);
}

void fill0(Enc& e, const Buffer& y, size_t nfloats) {
  uint32_t n4 = (uint32_t)(nfloats / 4);
  float v = 0.f;
  e.set_pipeline("m_fill"); e.set_buffer(0, y); e.set_value(1, v); e.set_value(2, n4);
  e.dispatch_threads(n4, 1, 1, 256);
}

}  // namespace

// ---- engram_kv as background chunks (engram_bg) ----
// engram_kv (129 MB Q6_K per engram layer, 25600 x 6144) depends only on the token ids, so the second engram layer's
// (layer 14) does not have to run before layer 0: its rows are split into chunks encoded into latency-bound stages of
// layers 0..13, each right AFTER the stage's latency-critical dispatch (the concurrent encoder starts that one first; the
// chunk's threadgroups fill the idle cores). Every chunk uses the full matrix's launch shape, so each row is computed
// exactly as by the single matvec (bit-identical). The first engram layer's kv (needed by layer 1) stays in the prologue.
// Slots: indexer compaction / final select and sparse attention (/ combine) stages (engram_bg >= 1), hc steps (>= 2).
namespace {
enum { kSlotHcA = 0, kSlotIdx0, kSlotIdx1, kSlotSparse, kSlotComb, kSlotHcF, kSlots };
bool slot_on(const DecCtx& c, int l, int slot) {
  const int lv = c.o.engram_bg;
  switch (slot) {
    case kSlotHcA: case kSlotHcF: return lv >= 2;
    case kSlotIdx0: case kSlotIdx1: return lv >= 1 && c.cfg->is_index_source(l) && c.ast->opt.dec_fast_index;
    case kSlotSparse: return lv >= 1;
    case kSlotComb: return lv >= 1 && !c.o.attn_comb;
  }
  return false;
}
int bg_slots_before(const DecCtx& c, int layer) {
  int n = 0;
  for (int l = 0; l < layer; l++) for (int k = 0; k < kSlots; k++) n += slot_on(c, l, k);
  return n;
}
void bg_chunk(Enc& e, DecCtx& c, int s, int k) {
  const W& w = (*c.L)[c.cfg->engram_layers[s]].engram_kv;
  const q::MvCfg mc = q::matvec_cfg(w.t, w.rows, w.cols);
  const int rpt = mc.nr * mc.rsg;
  const int units = (w.rows + rpt - 1) / rpt, n = c.bg[s].chunks;
  const int r0 = std::min(w.rows, (int)((long)units * k / n) * rpt), r1 = std::min(w.rows, (int)((long)units * (k + 1) / n) * rpt);
  if (r1 <= r0) return;
  const size_t rb = q::row_bytes(w.t, w.cols);
  q::encode_matvec(e, w.t, w.b.sub((size_t)r0 * rb, (size_t)(r1 - r0) * rb), r1 - r0, w.cols,
                   c.B.edeq.sub((size_t)s * 6144 * 4, 6144 * 4), c.B.ekv[s].sub((size_t)r0 * 4, (size_t)(r1 - r0) * 4),
                   false, 1, mc);
}
// encode the chunk planned for (layer l, slot): call inside the slot's stage, after its latency-critical dispatch
void bg_slot(Enc& e, DecCtx& c, int l, int slot) {
  if (c.o.engram_bg <= 0 || !slot_on(c, l, slot) || sk(c, kSkipEngram)) return;
  for (int s = 0; s < 2; s++)
    if (l < c.cfg->engram_layers[s] && c.bg[s].next < c.bg[s].chunks) bg_chunk(e, c, s, c.bg[s].next++);
}
}  // namespace

void dec_bg_begin(DecCtx& c) {
  for (int s = 0; s < 2; s++) {
    const int le = c.cfg->engram_layers[s];
    c.bg[s].next = 0;
    c.bg[s].chunks = (c.o.engram_bg > 0 && le >= 2) ? std::max(1, bg_slots_before(c, le)) : 0;
  }
}
void dec_bg_flush(Enc& e, DecCtx& c, int upto) {
  bool any = false;
  for (int s = 0; s < 2; s++)
    if (c.cfg->engram_layers[s] < upto || upto < 0)
      while (c.bg[s].next < c.bg[s].chunks) { bg_chunk(e, c, s, c.bg[s].next++); any = true; }
  if (any) bar(e, c);
}

void encode_dec_prologue(Enc& e, DecCtx& c, const Buffer& erow) {
  const Config& cfg = *c.cfg;
  c.hcur = 0;   // the embedding wrote B.h
  if (sk(c, kSkipEngram)) return;
  if (c.o.engram_bg > 0) {   // dequantize both engram layers' rows; the later engram layers' matvecs become chunks
    dec_bg_begin(c);
    for (int s = 0; s < 2; s++)
      arch::encode_iq4nl_rows(e, erow.sub((size_t)s * 24 * 144, 24 * 144), 144, c.B.edeq.sub((size_t)s * 6144 * 4, 6144 * 4), 24);
    bar(e, c);
    bool any = false;
    for (int s = 0; s < 2; s++)
      if (c.bg[s].chunks == 0) {
        matvec(e, (*c.L)[cfg.engram_layers[s]].engram_kv, c.B.edeq.sub((size_t)s * 6144 * 4, 6144 * 4), c.B.ekv[s]);
        any = true;
      }
    if (any) bar(e, c);
    return;
  }
  for (int s = 0; s < 2; s++) {
    const int l = cfg.engram_layers[s];
    arch::encode_iq4nl_rows(e, erow.sub((size_t)s * 24 * 144, 24 * 144), 144, c.B.edeq, 24);
    bar(e, c);
    matvec(e, (*c.L)[l].engram_kv, c.B.edeq, c.B.ekv[s]);
    bar(e, c);
  }
}

void encode_dec_attn_side(Enc& e, DecCtx& c, int l, int pos) {
  const Config& cfg = *c.cfg;
  const LayerW& L = (*c.L)[l];
  const DecBufs& B = c.B;
  const arch::Scratch& asc = *c.asc;
  const DecOptions& o = c.o;
  const bool kvsrc = cfg.is_kv_source(l), idxsrc = cfg.is_index_source(l);
  // ---- hyper-connection boundary into attention (+ engram) ----
  if (cfg.is_engram(l) && o.engram_bg > 0) dec_bg_flush(e, c, l + 1);   // safety: nothing left with the default plan
  // hc steps with an x_sub update h: in place (hc_splits 1) or ping-pong B.h <-> B.h2 (hc_splits > 1, needs h2)
  const bool pp = o.hc_splits > 1 && (bool)B.h2;
  auto post_io = [&](arch::HcStep& a) {
    a.h_in = dec_cur_h(c);
    if (pp) { c.hcur ^= 1; a.h_out = dec_cur_h(c); a.row_splits = o.hc_splits; }
    else a.h_out = a.h_in;
  };
  if (!sk(c, kSkipHcA)) {
    if (cfg.is_engram(l)) {
      arch::HcStep a; post_io(a); a.x_sub = B.sub; a.mix_prev = B.mixF; a.n = 1;
      arch::encode_hc_step(e, a, asc); bar(e, c);
      const Buffer kvb = l == cfg.engram_layers[0] ? B.ekv[0] : B.ekv[1];
      arch::encode_engram_gate(e, dec_cur_h(c), kvb, L.engram_gate.b, 1, asc); bar(e, c);
      arch::HcStep h2; h2.h_in = dec_cur_h(c); h2.h_out = h2.h_in; h2.mix_prev = B.mixF; h2.fn = L.hc_attn_fn.b; h2.scale = L.hc_attn_scale.b;
      h2.base = L.hc_attn_base.b; h2.mix_out = B.mixA; h2.norm_w = L.attn_norm.b; h2.x_out = B.x; h2.n = 1;
      if (pp) h2.row_splits = o.hc_splits;
      arch::encode_hc_step(e, h2, asc);
      bg_slot(e, c, l, kSlotHcA);
      bar(e, c);
    } else {
      arch::HcStep a;
      if (l > 0) { post_io(a); a.x_sub = B.sub; }
      else { a.h_in = dec_cur_h(c); a.h_out = a.h_in; if (pp) a.row_splits = o.hc_splits; }
      a.mix_prev = (l == 0 && B.mix_ident) ? B.mix_ident : B.mixF;
      a.fn = L.hc_attn_fn.b; a.scale = L.hc_attn_scale.b; a.base = L.hc_attn_base.b; a.mix_out = B.mixA;
      a.norm_w = L.attn_norm.b; a.x_out = B.x; a.n = 1;
      arch::encode_hc_step(e, a, asc);
      bg_slot(e, c, l, kSlotHcA);
      bar(e, c);
    }
  }
  // ---- attention projections ----
  if (!sk(c, kSkipProj)) {
    matvec(e, L.q_a, B.x, B.qa);
    matvec(e, L.kv, B.x, B.kv);
    if (kvsrc) { matvec(e, L.c_kv, B.x, B.ckv); if (L.c_gate) matvec(e, L.c_gate, B.x, B.cgate); }
    if (idxsrc) matvec(e, L.idx_proj, B.x, B.iw);
  }
  bar(e, c);
  attn::State& S = *c.ast;
  attn::LayerIO io; io.layer = l; io.q = B.q; io.kv = B.kv; io.kv_norm = L.kv_norm.b; io.sinks = L.sinks.b; io.out = B.aout;
  if (kvsrc) { io.c_kv = B.ckv; io.c_gate = L.c_gate ? B.cgate : Buffer{}; io.c_norm = L.c_norm.b; io.idx_wk = L.idx_k.b; io.idx_knorm = L.idx_knorm.b; }
  if (idxsrc) { io.idx_q = B.iq; io.idx_w = B.iw; }
  // kv window write + compressor only need the projections: with `reorder` they run in the next stage (q_a norm or
  // the norm-fused q_b) instead of a stage of their own after q_b
  auto kv_stage = [&]() {
    if (sk(c, kSkipKvWin)) return;
    attn::encode_kv_window(e, S, l, io.kv, io.kv_norm, pos, 1);
    if (kvsrc) attn::encode_compress(e, S, l, io, pos, 1, false);
  };
  if (o.norm_qb) {
    if (!sk(c, kSkipQb)) {
      q::encode_matvec_rmsnorm(e, L.q_b.t, L.q_b.b, L.q_b.rows, L.q_b.cols, B.qa, L.q_a_norm.b, cfg.norm_eps, B.q, o.norm_qb_tpt);
      if (idxsrc) q::encode_matvec_rmsnorm(e, L.idx_qb.t, L.idx_qb.b, L.idx_qb.rows, L.idx_qb.cols, B.qa, L.q_a_norm.b, cfg.norm_eps, B.iq, o.norm_qb_tpt);
    }
    if (o.reorder) kv_stage();
    bar(e, c);
  } else {
    if (!sk(c, kSkipNorm)) arch::encode_rmsnorm(e, B.qa, L.q_a_norm.b, B.qrn, 1, 1280, cfg.norm_eps);
    if (o.reorder) kv_stage();
    if (!sk(c, kSkipNorm) || o.reorder) bar(e, c);
    if (!sk(c, kSkipQb)) {
      matvec(e, L.q_b, B.qrn, B.q);
      if (idxsrc) matvec(e, L.idx_qb, B.qrn, B.iq);
    }
    bar(e, c);
  }
  if (!o.reorder && !sk(c, kSkipKvWin)) { kv_stage(); bar(e, c); }
  // ---- attention: indexer (index source layers) + sparse attention ----
  attn::StageHook hook;
  if (o.engram_bg > 0)
    hook = [&](int st) { bg_slot(e, c, l, st == 0 ? kSlotIdx0 : st == 1 ? kSlotIdx1 : st == 2 ? kSlotSparse : kSlotComb); };
  if (idxsrc && !sk(c, kSkipIndex)) attn::encode_index_decode(e, S, l, io.idx_q, io.idx_w, pos, hook);
  if (!sk(c, kSkipSparse)) attn::encode_sparse(e, S, l, io.q, io.sinks, io.out, pos, 1, o.attn_comb, hook);
  if (o.actq_out) { attn::encode_quant_sim(e, B.aout, 32768u, 0); bar(e, c); }
  // ---- wo_a (block-diagonal, 8 groups) then wo_b ----
  if (!sk(c, kSkipOa)) { q::encode_matvec(e, L.o_a.t, L.o_a.b, 8192, 4096, B.aout, B.oa, false, 8); bar(e, c); }
  if (!sk(c, kSkipOb)) { matvec(e, L.o_b, B.oa, B.sub); bar(e, c); }
  // ---- hyper-connection boundary into the FFN ----
  if (!sk(c, kSkipHcF)) {
    arch::HcStep a; post_io(a); a.x_sub = B.sub; a.mix_prev = B.mixA;
    a.fn = L.hc_ffn_fn.b; a.scale = L.hc_ffn_scale.b; a.base = L.hc_ffn_base.b; a.mix_out = B.mixF;
    a.norm_w = L.ffn_norm.b; a.x_out = B.x; a.n = 1;
    arch::encode_hc_step(e, a, asc);
    bg_slot(e, c, l, kSlotHcF);
    bar(e, c);
  }
  // ---- router (+ next-layer expert prediction: logits only) ----
  if (!sk(c, kSkipRouter)) arch::encode_router_fused(e, B.x, L.gate_inp.b, L.bias.b, B.rids, B.rw, asc);
  if (o.pred_on && !sk(c, kSkipPred))
    for (int d = 1; d <= o.pred_depth && l + d < cfg.n_layers; d++) {
      const W& g = (o.pred_q8 && c.pred_w && !c.pred_w->empty()) ? (*c.pred_w)[l + d] : (*c.L)[l + d].gate_inp;
      q::encode_matvec(e, g.t, g.b, g.rows, g.cols, B.x, B.plog.sub((size_t)(d - 1) * g.rows * 4, (size_t)g.rows * 4));
    }
  for (int k = 0; k < o.probe_dispatches; k++) fill0(e, asc.dummy.sub(16 * (k % 8), 16), 4);   // PROFILING probe
  // the MoE accumulates into sub (consumed by the hc step above): zero it in the router stage with `reorder`
  if (o.reorder && !sk(c, kSkipFill)) fill0(e, B.sub, 5120);
  if (o.sgu_router && !sk(c, kSkipMoeShared)) {   // shared expert gate/up next to the router (needs only x)
    q::MoeDecodeArgs m;
    m.n_routed = 6; m.weights = B.rw; m.has_shared = true; m.t_shared = L.sh_gate.t;
    m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
    m.x = B.x; m.y = B.sub; m.scratch = B.moe_scratch;
    m.t_gate_up = QType::IQ1_S; m.t_down = QType::IQ1_S;   // routed types are not used by the SharedGu part
    m.gu_nr = o.moe_gu_nr; m.gu_rsg = o.moe_gu_rsg;
    q::encode_moe_decode_table_part(e, m, B.moe_scratch, q::MoePart::SharedGu);
  }
  bar(e, c);
  if (!o.reorder && !sk(c, kSkipFill)) fill0(e, B.sub, 5120);
}

void encode_dec_moe_table(Enc& e, DecCtx& c, int l, QType t_gate_up, QType t_down, const Buffer& table,
                          const std::vector<Buffer>& arena, const DecSync& s) {
  const LayerW& L = (*c.L)[l];
  q::MoeDecodeArgs m;
  m.t_gate_up = t_gate_up; m.t_down = t_down;
  m.n_routed = 6; m.weights = c.B.rw; m.has_shared = true; m.t_shared = L.sh_gate.t;
  m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
  m.x = c.B.x; m.y = c.B.sub; m.scratch = c.B.moe_scratch;
  m.gu_nr = c.o.moe_gu_nr; m.gu_rsg = c.o.moe_gu_rsg; m.d_nr = c.o.moe_d_nr; m.d_rsg = c.o.moe_d_rsg;
  id<MTLDevice> dev = (__bridge id<MTLDevice>)Device::get().device();
  static id<MTLFence> f_att = [dev newFence];
  static id<MTLFence> f_gap = [dev newFence];
  auto ce = [&]() { return (__bridge id<MTLComputeCommandEncoder>)e.encoder(); };
  id<MTLSharedEvent> rdy = (__bridge id<MTLSharedEvent>)s.rdy, go = (__bridge id<MTLSharedEvent>)s.go;
  const uint64_t v = s.value_base + 1 + (uint64_t)l;
  const bool hs = rdy && go && !sk(c, kSkipEvents);
  const bool skip_s = sk(c, kSkipMoeShared), skip_r = sk(c, kSkipMoeRouted);
  // pieces: `gap` = shared-expert work that does not need the routing (runs between the rdy signal and the go wait),
  // `after` = what needs the routed experts' table.
  //   mode 0 event : gap -           after All              (sgu_router: after AfterSharedGu)
  //   mode 1 sgu   : gap SharedGu    after AfterSharedGu    (sgu_router: gap -, after AfterSharedGu)
  //   mode 2 shared: gap Shared      after Routed           (sgu_router: gap SharedDown, after Routed)
  // (sgu_router: the shared gate/up already ran in the router stage; every combination is bit-identical to its mode.)
  bool has_gap = false;
  q::MoePart gap = q::MoePart::Shared, after = q::MoePart::All;
  if (s.mode == 1) { has_gap = !c.o.sgu_router; gap = q::MoePart::SharedGu; after = q::MoePart::AfterSharedGu; }
  else if (s.mode == 2) { has_gap = true; gap = c.o.sgu_router ? q::MoePart::SharedDown : q::MoePart::Shared; after = q::MoePart::Routed; }
  else if (c.o.sgu_router) after = q::MoePart::AfterSharedGu;
  // sgu_router 2: the shared down projection runs after the go wait, concurrently with the routed gate/up (same
  // accumulation order into y: shared down completes before the routed down's stage)
  const bool late_sd = s.mode == 2 && c.o.sgu_router >= 2 && !skip_s;
  if (late_sd) has_gap = false;
  if (skip_s) has_gap = false;
  const bool do_after = !skip_r && !(skip_s && after == q::MoePart::All);
  if (!hs) {
    // no handshake: everything in the current encoder (the table must already be valid)
    if (has_gap) { q::encode_moe_decode_table_part(e, m, table, gap); bar(e, c); }
    e.use_resources(arena);
    if (late_sd) q::encode_moe_decode_table_part(e, m, table, q::MoePart::SharedDown);
    if (do_after) q::encode_moe_decode_table_part(e, m, table, after);
  } else {
    // Consecutive compute encoders over untracked buffers are not ordered by themselves: fence A orders the gap
    // encoder after the attention side, fence B the routed MoE after the gap. (The go wait alone orders the MoE only
    // through the CPU, which sets go after it saw rdy; with a pre-signaled go, e.g. bench/test, that ordering is gone.)
    [ce() updateFence:f_att];
    e.between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:rdy value:v]; });
    [ce() waitForFence:f_att];
    if (has_gap) q::encode_moe_decode_table_part(e, m, table, gap);
    [ce() updateFence:f_gap];
    e.between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:go value:v]; });
    [ce() waitForFence:f_gap];
    e.use_resources(arena);
    if (late_sd) q::encode_moe_decode_table_part(e, m, table, q::MoePart::SharedDown);
    if (do_after) q::encode_moe_decode_table_part(e, m, table, after);
  }
  bar(e, c);
}

}  // namespace ms::ds41
