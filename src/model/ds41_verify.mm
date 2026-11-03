// Speculative-decoding verify: k tokens through the decode (table) path at once, bit-identical per token to decode().
// See ds41.hpp (Model::verify) and docs/speculative.md.
//
// Per token everything is computed by the decode kernels themselves (hc steps, router, engram gate, attention kernels,
// routed MoE table kernels) or by k-token kernels that reproduce the decode kernels' per-lane arithmetic exactly
// (kernels/spec.hpp: dense matvecs, shared expert). Attention state: the k new window rows go to a per-layer staging
// chunk (the ring is not touched; queries read positions >= start from the chunk), the compressor writes its rows /
// partial state as a k-token call (bit-identical group pooling), and verify_keep(m) commits m chunk rows into the
// rings and repairs the compressor's partial-group state; compressed rows of rejected tokens are invisible until the
// token that completes their group rewrites them.
#import <Metal/Metal.h>
#include "model/ds41.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "core/env.hpp"
#include "core/timer.hpp"
#include "gguf/gguf.hpp"
#include "kernels/arch.hpp"
#include "kernels/attn.hpp"
#include "kernels/quant.hpp"
#include "kernels/spec.hpp"
#include "model/dec_layer.hpp"
#include "model/ds41_impl.hpp"
#include "model/dspark.hpp"
#include "model/engram_hash.hpp"
#include "store/engram.hpp"
#include "store/experts.hpp"
#include "store/weights.hpp"

namespace ms::ds41 {

using q::QType;

struct VerifyState {
  int kmax = 0;
  Buffer embrows, erow, edeq, ekv, h, x, sub, mixA, mixF, qa, qrn, q, kvb, iq, iw, aout, oa, rids, rw, plog, logits,
      amax_i, amax_v, hS, moe_scr, tbl, feat;
  Buffer ckv[4], cgate[4];          // c_kv / c_gate rows of the k tokens per kv source (compressor state repair)
  Buffer chunk[40];                 // new window rows (half [k][512]) per layer
  std::vector<arch::Scratch> sc;    // per token: the decode single-dispatch kernels' counters / partials
  std::vector<attn::DecQScratch> qs;   // per token: indexer / top-k / sparse partials
  // the last verify (for verify_keep)
  int start = -1, k = 0;
  std::vector<int32_t> toks;
  EngramHasher::State hs;
};

Model::Impl::Impl() = default;
Model::Impl::~Impl() = default;

bool Model::verify_supported(std::string* why) {
  auto fail = [&](const char* w) { if (why) *why = w; return false; };
  if (env_int("MACSEEK_DEC_TABLE", 1) == 0 || env_int("MACSEEK_DEC_LAYER", 1) == 0 || env_int("MACSEEK_PROF_FORCE", 0) != 0)
    return fail("needs the table decode path (MACSEEK_DEC_TABLE=1, MACSEEK_DEC_LAYER=1)");
  const std::string sync = env_str("MACSEEK_DEC_SYNC", "shared");
  if (!(sync == "2" || sync == "shared")) return fail("needs MACSEEK_DEC_SYNC=shared (the default)");
  const DecOptions o = DecOptions::from_env();
  if (o.fuse || o.reorder || o.norm_qb || o.attn_comb || o.sgu_router || o.engram_bg || o.hc_splits != 1 || o.skip ||
      o.probe_dispatches)
    return fail("needs the default decode layer options (MACSEEK_DEC_FUSE etc. off)");
  if (arch::options().hc_mode != 0) return fail("needs MACSEEK_ARCH_HC=fused (the default)");
  return true;
}

const float* Model::verify_features() const { return p_->vs ? (const float*)p_->vs->feat.cpu() : nullptr; }

void Model::verify(const int32_t* toks, int k, int32_t* next, float* logits_out) {
  auto& P = *p_;
  if (k < 1 || k > sv::kMaxK) { fprintf(stderr, "macseek: verify k=%d out of range\n", k); abort(); }
  if (pos_ + k > max_pos()) { fprintf(stderr, "macseek: verify past max_pos\n"); abort(); }
  {
    static const bool ok = [] { std::string w; bool r = verify_supported(&w); if (!r) fprintf(stderr, "macseek: verify: %s\n", w.c_str()); return r; }();
    if (!ok) abort();
  }
  if (pinned_) { cache_->unpin_resident(); pinned_ = 0; }
  auto& d = Device::get();
  const int NE = cfg_.n_experts, V = cfg_.vocab, D = 5120;
  if (!P.vs) {
    auto vs = std::make_unique<VerifyState>();
    const int K = sv::kMaxK;
    vs->kmax = K;
    auto A = [&](size_t b, const char* l) { return d.alloc(b, l); };
    vs->embrows = A((size_t)K * te_->row_bytes() + 16384, "v.embrows"); vs->erow = A((size_t)2 * K * 24 * 144 + 16384, "v.erow");
    vs->edeq = A((size_t)2 * K * 6144 * 4, "v.edeq"); vs->ekv = A((size_t)2 * K * 25600 * 4, "v.ekv");
    vs->h = A((size_t)K * 20480 * 4, "v.h"); vs->x = A((size_t)K * D * 4, "v.x"); vs->sub = A((size_t)K * D * 4, "v.sub");
    vs->mixA = A((size_t)K * 24 * 4, "v.mixA"); vs->mixF = A((size_t)K * 24 * 4, "v.mixF");
    vs->qa = A((size_t)K * 1280 * 4, "v.qa"); vs->qrn = A((size_t)K * 1280 * 4, "v.qrn"); vs->q = A((size_t)K * 32768 * 4, "v.q");
    vs->kvb = A((size_t)K * 512 * 4, "v.kv"); vs->iq = A((size_t)K * 4096 * 4, "v.iq"); vs->iw = A((size_t)K * 32 * 4, "v.iw");
    vs->aout = A((size_t)K * 32768 * 4, "v.aout"); vs->oa = A((size_t)K * 8192 * 4, "v.oa");
    vs->rids = A((size_t)K * 64, "v.rids"); vs->rw = A((size_t)K * 64, "v.rw"); vs->plog = A((size_t)K * NE * 4, "v.plog");
    vs->logits = A((size_t)K * V * 4, "v.logits"); vs->amax_i = A(256, "v.amax_i"); vs->amax_v = A(256, "v.amax_v");
    vs->hS = A((size_t)K * 2304 * 4, "v.hS"); vs->moe_scr = A(q::moe_decode_scratch_bytes() * K, "v.moe");
    vs->tbl = A((size_t)cfg_.n_layers * 1024, "v.tbl"); vs->feat = A((size_t)3 * K * D * 4, "v.feat");
    for (int s = 0; s < 4; s++) { vs->ckv[s] = A((size_t)K * 512 * 4, "v.ckv"); vs->cgate[s] = A((size_t)K * 512 * 4, "v.cgate"); }
    for (int l = 0; l < cfg_.n_layers; l++) vs->chunk[l] = A((size_t)K * 512 * 2, "v.chunk");
    for (int t = 0; t < K; t++) { vs->sc.push_back(arch::Scratch::create(64, 16)); vs->qs.push_back(attn::make_dec_q_scratch(P.ast)); }
    P.vs = std::move(vs);
  }
  VerifyState& S = *P.vs;
  Timer tt;
  const int start = pos_;
  S.start = start; S.k = k; S.toks.assign(toks, toks + k); S.hs = hasher_->state();

  // ---- CPU: engram rows + embedding rows (as forward(): layer-0 rows of all tokens, then layer-1 rows) ----
  Timer te;
  {
    std::vector<uint32_t> hs((size_t)k * 48);
    hasher_->hash(toks, k, hs.data());
    std::vector<uint64_t> r0((size_t)k * 24), r1((size_t)k * 24);
    for (int t = 0; t < k; t++)
      for (int c = 0; c < 24; c++) { r0[t * 24 + c] = hs[t * 48 + c]; r1[t * 24 + c] = hs[t * 48 + 24 + c]; }
    char* dst = (char*)S.erow.cpu();
    EngramFetcher::Ticket t0 = ef_->fetch(r0.data(), r0.size(), dst);
    EngramFetcher::Ticket t1 = ef_->fetch(r1.data(), r1.size(), dst + (size_t)k * 24 * 144);
    te_->fetch(toks, k, S.embrows.cpu());
    t0.wait(); t1.wait();
  }
  const double engram_ms = te.ms();
  const LayerStats st0 = cache_->total_stats();

  auto rowb = [](const Buffer& b, int t, size_t bytes) { return b.sub((size_t)t * bytes, bytes); };
  const size_t HB = 20480 * 4, XB = (size_t)D * 4;
  auto e = std::make_unique<Enc>();
  arch::encode_embed(*e, (int)te_->type(), S.embrows, (uint32_t)te_->row_bytes(), S.h, k, 4);
  e->barrier();
  { float* mf = (float*)S.mixF.cpu(); for (int t = 0; t < k; t++) arch::fill_identity_mix(mf + t * 24); }
  // prologue: engram rows -> edeq -> engram_kv for both engram layers (decode: encode_dec_prologue, engram_bg off)
  Buffer ekv_s[2];
  for (int s = 0; s < 2; s++) {
    const int l = cfg_.engram_layers[s];
    const Buffer edq = S.edeq.sub((size_t)s * k * 6144 * 4, (size_t)k * 6144 * 4);
    ekv_s[s] = S.ekv.sub((size_t)s * k * 25600 * 4, (size_t)k * 25600 * 4);
    arch::encode_iq4nl_rows(*e, S.erow.sub((size_t)s * k * 24 * 144, (size_t)k * 24 * 144), 144, edq, k * 24);
    e->barrier();
    const W& w = P.L[l].engram_kv;
    sv::encode_matvec_k(*e, w.t, w.b, w.rows, w.cols, edq, ekv_s[s], k);
    e->barrier();
  }
  // per-token MoE table pieces (decode encode_dec_moe_table, mode 2 "shared": shared part in the handshake gap,
  // routed part after the go wait)
  auto moe_args = [&](int l, int t, QType tgu, QType td) {
    const LayerW& L = P.L[l];
    q::MoeDecodeArgs m;
    m.t_gate_up = tgu; m.t_down = td; m.n_routed = 6; m.weights = rowb(S.rw, t, 64); m.has_shared = true;
    m.t_shared = L.sh_gate.t; m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
    m.x = rowb(S.x, t, XB); m.y = rowb(S.sub, t, XB);
    m.scratch = rowb(S.moe_scr, t, q::moe_decode_scratch_bytes());
    return m;
  };
  id<MTLDevice> dev = (__bridge id<MTLDevice>)d.device();
  static id<MTLSharedEvent> rdy = [dev newSharedEvent];
  static id<MTLSharedEvent> go = [dev newSharedEvent];
  static id<MTLFence> f_att = [dev newFence];
  static id<MTLFence> f_gap = [dev newFence];
  static uint64_t vb = 0;
  const uint64_t b0 = vb;
  vb += cfg_.n_layers;
  auto ce = [&]() { return (__bridge id<MTLComputeCommandEncoder>)e->encoder(); };
  const std::vector<Buffer> arena = cache_->buffers();
  static const bool pred_on = env_int("MACSEEK_PRED", 1) != 0;
  static const int pred_k = (int)env_int("MACSEEK_PRED_K", 6);
  static const bool actq_out = o_.actq_out;

  Timer tenc;
  for (int l = 0; l < cfg_.n_layers; l++) {
    const LayerW& L = P.L[l];
    const bool kvsrc = cfg_.is_kv_source(l), idxsrc = cfg_.is_index_source(l);
    // ---- hc boundary into attention (+ engram) ----
    if (cfg_.is_engram(l)) {
      const int se = l == cfg_.engram_layers[0] ? 0 : 1;
      for (int t = 0; t < k; t++) {
        arch::HcStep a; a.h_in = rowb(S.h, t, HB); a.h_out = a.h_in; a.x_sub = rowb(S.sub, t, XB);
        a.mix_prev = rowb(S.mixF, t, 96); a.n = 1;
        arch::encode_hc_step(*e, a, S.sc[t]);
      }
      e->barrier();
      for (int t = 0; t < k; t++)
        arch::encode_engram_gate(*e, rowb(S.h, t, HB), rowb(ekv_s[se], t, 25600 * 4), L.engram_gate.b, 1, S.sc[t]);
      e->barrier();
      for (int t = 0; t < k; t++) {
        arch::HcStep a; a.h_in = rowb(S.h, t, HB); a.h_out = a.h_in; a.mix_prev = rowb(S.mixF, t, 96);
        a.fn = L.hc_attn_fn.b; a.scale = L.hc_attn_scale.b; a.base = L.hc_attn_base.b; a.mix_out = rowb(S.mixA, t, 96);
        a.norm_w = L.attn_norm.b; a.x_out = rowb(S.x, t, XB); a.n = 1;
        arch::encode_hc_step(*e, a, S.sc[t]);
      }
      e->barrier();
    } else {
      for (int t = 0; t < k; t++) {
        arch::HcStep a; a.h_in = rowb(S.h, t, HB); a.h_out = a.h_in;
        if (l > 0) a.x_sub = rowb(S.sub, t, XB);
        a.mix_prev = rowb(S.mixF, t, 96);
        a.fn = L.hc_attn_fn.b; a.scale = L.hc_attn_scale.b; a.base = L.hc_attn_base.b; a.mix_out = rowb(S.mixA, t, 96);
        a.norm_w = L.attn_norm.b; a.x_out = rowb(S.x, t, XB); a.n = 1;
        arch::encode_hc_step(*e, a, S.sc[t]);
      }
      e->barrier();
    }
    // dspark target features: mean over the hc copies of this layer's attention input (h after the hc step)
    for (int ti = 0; ti < 3; ti++)
      if (cfg_.dspark_targets[ti] == l)
        arch::encode_hc_mean(*e, S.h.sub(0, (size_t)k * HB), S.feat.sub((size_t)ti * k * XB, (size_t)k * XB), k);
    // ---- projections ----
    const int si = kvsrc ? attn::src_slot(l) : 0;
    sv::encode_matvec_k(*e, L.q_a.t, L.q_a.b, L.q_a.rows, L.q_a.cols, S.x, S.qa, k);
    sv::encode_matvec_k(*e, L.kv.t, L.kv.b, L.kv.rows, L.kv.cols, S.x, S.kvb, k);
    if (kvsrc) {
      sv::encode_matvec_k(*e, L.c_kv.t, L.c_kv.b, L.c_kv.rows, L.c_kv.cols, S.x, S.ckv[si], k);
      if (L.c_gate) sv::encode_matvec_k(*e, L.c_gate.t, L.c_gate.b, L.c_gate.rows, L.c_gate.cols, S.x, S.cgate[si], k);
    }
    if (idxsrc) sv::encode_matvec_k(*e, L.idx_proj.t, L.idx_proj.b, L.idx_proj.rows, L.idx_proj.cols, S.x, S.iw, k);
    e->barrier();
    arch::encode_rmsnorm(*e, S.qa, L.q_a_norm.b, S.qrn, k, 1280, cfg_.norm_eps);
    e->barrier();
    sv::encode_matvec_k(*e, L.q_b.t, L.q_b.b, L.q_b.rows, L.q_b.cols, S.qrn, S.q, k);
    if (idxsrc) sv::encode_matvec_k(*e, L.idx_qb.t, L.idx_qb.b, L.idx_qb.rows, L.idx_qb.cols, S.qrn, S.iq, k);
    e->barrier();
    // ---- kv window rows -> staging chunk; compressor (k tokens at once: same group pooling per row) ----
    attn::encode_kv_window_to(*e, P.ast, l, S.kvb, L.kv_norm.b, start, k, S.chunk[l]);
    if (kvsrc) {
      attn::LayerIO io; io.layer = l; io.c_kv = S.ckv[si]; io.c_gate = L.c_gate ? S.cgate[si] : Buffer{};
      io.c_norm = L.c_norm.b; io.idx_wk = L.idx_k.b; io.idx_knorm = L.idx_knorm.b;
      attn::encode_compress(*e, P.ast, l, io, start, k, false);
    }
    e->barrier();
    if (idxsrc) attn::encode_index_decode_k(*e, P.ast, l, S.iq, S.iw, start, k, S.qs);
    attn::encode_sparse_k(*e, P.ast, l, S.q, L.sinks.b, S.aout, start, k, S.chunk[l], S.qs);
    if (actq_out) {
      for (int t = 0; t < k; t++) attn::encode_quant_sim(*e, rowb(S.aout, t, 32768 * 4), 32768u, 0);
      e->barrier();
    }
    sv::encode_matvec_k(*e, L.o_a.t, L.o_a.b, 8192, 4096, S.aout, S.oa, k, false, 8);
    e->barrier();
    sv::encode_matvec_k(*e, L.o_b.t, L.o_b.b, L.o_b.rows, L.o_b.cols, S.oa, S.sub, k);
    e->barrier();
    // ---- hc boundary into the FFN, router, prediction ----
    for (int t = 0; t < k; t++) {
      arch::HcStep a; a.h_in = rowb(S.h, t, HB); a.h_out = a.h_in; a.x_sub = rowb(S.sub, t, XB); a.mix_prev = rowb(S.mixA, t, 96);
      a.fn = L.hc_ffn_fn.b; a.scale = L.hc_ffn_scale.b; a.base = L.hc_ffn_base.b; a.mix_out = rowb(S.mixF, t, 96);
      a.norm_w = L.ffn_norm.b; a.x_out = rowb(S.x, t, XB); a.n = 1;
      arch::encode_hc_step(*e, a, S.sc[t]);
    }
    e->barrier();
    for (int t = 0; t < k; t++)
      arch::encode_router_fused(*e, rowb(S.x, t, XB), L.gate_inp.b, L.bias.b, rowb(S.rids, t, 64), rowb(S.rw, t, 64), S.sc[t]);
    if (pred_on && l + 1 < cfg_.n_layers) {
      const W& g = P.pred_w[l + 1];
      sv::encode_matvec_k(*e, g.t, g.b, g.rows, g.cols, S.x, S.plog, k);
    }
    e->barrier();
    {   // fill(sub) = 0: the MoE accumulates into it
      uint32_t n4 = (uint32_t)(k * D / 4); float zero = 0.f;
      e->set_pipeline("m_fill"); e->set_buffer(0, S.sub); e->set_value(1, zero); e->set_value(2, n4);
      e->dispatch_threads(n4, 1, 1, 256);
    }
    // ---- MoE: rdy -> gap (shared expert, k tokens) -> go -> routed experts per token ----
    const auto& lay = cache_->layout().layers[l];
    const QType tgu = (QType)(int)lay.mat[0].type, tdn = (QType)(int)lay.mat[2].type;
    const uint64_t v = b0 + 1 + (uint64_t)l;
    [ce() updateFence:f_att];
    e->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:rdy value:v]; });
    [ce() waitForFence:f_att];
    sv::encode_moe_shared_k(*e, L.sh_gate.t, q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b}, S.x, S.hS, S.sub, k);
    [ce() updateFence:f_gap];
    e->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:go value:v]; });
    [ce() waitForFence:f_gap];
    e->use_resources(arena);
    const Buffer tl = S.tbl.sub((size_t)l * 1024, 1024);
    for (int t = 0; t < k; t++)
      q::encode_moe_decode_table_part(*e, moe_args(l, t, tgu, tdn), tl.sub((size_t)t * 144, 144), q::MoePart::RoutedGu);
    e->barrier();
    for (int t = 0; t < k; t++)
      q::encode_moe_decode_table_part(*e, moe_args(l, t, tgu, tdn), tl.sub((size_t)t * 144, 144), q::MoePart::RoutedDown);
    e->barrier();
  }
  // ---- head ----
  for (int t = 0; t < k; t++) {
    arch::HcStep a; a.h_in = rowb(S.h, t, HB); a.h_out = a.h_in; a.x_sub = rowb(S.sub, t, XB); a.mix_prev = rowb(S.mixF, t, 96);
    a.norm_w = P.out_norm.b; a.x_out = rowb(S.x, t, XB); a.n = 1;
    arch::encode_hc_step(*e, a, S.sc[t]);
  }
  e->barrier();
  sv::encode_matvec_k(*e, P.head.t, P.head.b, P.head.rows, P.head.cols, S.x, S.logits, k);
  e->barrier();
  arch::encode_argmax(*e, S.logits, k, V, S.amax_i, S.amax_v, S.sc[0], 0.f, 0);
  const double enc_ms = tenc.ms();
  e->commit();

  // ---- CPU: per layer, the union of the k tokens' experts ----
  double wait_ms = 0, exp_ms = 0;
  std::vector<ExpertLease> leases;
  leases.reserve(cfg_.n_layers);
  for (int l = 0; l < cfg_.n_layers; l++) {
    Timer tw;
    while (rdy.signaledValue < b0 + 1 + (uint64_t)l) { }
    wait_ms += tw.ms();
    Timer tx;
    const int* ids = (const int*)S.rids.cpu();
    int uni[sv::kMaxK * 6], nu = 0, slot_of[sv::kMaxK * 6];
    for (int t = 0; t < k; t++)
      for (int j = 0; j < 6; j++) {
        const int x = ids[t * 16 + j];
        int u = 0;
        while (u < nu && uni[u] != x) u++;
        if (u == nu) uni[nu++] = x;
        slot_of[t * 6 + j] = u;
      }
    ExpertLease lease = cache_->ensure(l, uni, nu);
    uint64_t* tb = (uint64_t*)((char*)S.tbl.cpu() + (size_t)l * 1024);
    for (int t = 0; t < k; t++)
      for (int j = 0; j < 6; j++)
        for (int m = 0; m < 3; m++) tb[t * 18 + j * 3 + m] = lease.views[slot_of[t * 6 + j]].gpu[m];
    auto prefetch = [&]() {
      if (!pred_on || l + 1 >= cfg_.n_layers) return;
      const float* bias = (const float*)P.L[l + 1].bias.b.cpu();
      std::vector<int> want;
      for (int t = 0; t < k; t++) {
        const float* lg = (const float*)S.plog.cpu() + (size_t)t * NE;
        std::pair<float, int> sc[384];
        for (int x = 0; x < NE; x++) { float vv = lg[x]; float sp = vv > 20.f ? vv : std::log1p(std::exp(vv)); sc[x] = {std::sqrt(sp) + bias[x], x}; }
        const int kk = std::min(pred_k, NE);
        std::partial_sort(sc, sc + kk, sc + NE, [](auto& a, auto& b) { return a.first > b.first; });
        for (int i = 0; i < kk; i++) if (std::find(want.begin(), want.end(), sc[i].second) == want.end()) want.push_back(sc[i].second);
      }
      cache_->prefetch(l + 1, want.data(), (int)want.size());
    };
    if (!lease.ticket) { go.signaledValue = b0 + 1 + (uint64_t)l; prefetch(); }
    else { prefetch(); lease.wait(); go.signaledValue = b0 + 1 + (uint64_t)l; }
    leases.push_back(std::move(lease));
    exp_ms += tx.ms();
  }
  Timer tw;
  e->wait();
  wait_ms += tw.ms();
  for (int t = 0; t < k; t++) next[t] = (int32_t)((const uint32_t*)S.amax_i.cpu())[t];
  if (logits_out) memcpy(logits_out, S.logits.cpu(), (size_t)k * V * 4);
  pos_ += k;
  const LayerStats st1 = cache_->total_stats();
  sprof_.passes++; sprof_.tokens += k; sprof_.total_ms += tt.ms(); sprof_.gpu_ms += e->gpu_ms(); sprof_.wait_ms += wait_ms;
  sprof_.expert_ms += exp_ms; sprof_.engram_ms += engram_ms; sprof_.io_stall_ms += st1.stall_ms - st0.stall_ms;
  sprof_.io_bytes += st1.bytes_read - st0.bytes_read;
  (void)enc_ms;
}

void Model::verify_keep(int m) {
  auto& P = *p_;
  if (!P.vs || P.vs->start < 0) { fprintf(stderr, "macseek: verify_keep without verify\n"); abort(); }
  VerifyState& S = *P.vs;
  if (m < 1 || m > S.k || pos_ != S.start + S.k) { fprintf(stderr, "macseek: verify_keep(%d) bad state\n", m); abort(); }
  Timer t;
  {
    Enc e;
    for (int l = 0; l < cfg_.n_layers; l++) attn::encode_window_commit_from(e, P.ast, l, S.start, m, S.chunk[l]);
    // compressor partial group: if the last kept token opens a group (ratio 2), its values must be the state
    const int last = S.start + m - 1;
    for (int s : cfg_.kv_sources)
      if (cfg_.compress_ratio[s] > 1 && (last + 1) % cfg_.compress_ratio[s] != 0) {
        const int si = attn::src_slot(s);
        attn::encode_compress_state_row(e, P.ast, s, S.ckv[si], S.cgate[si], S.start, last);
      }
    e.wait();
  }
  if (m < S.k) {
    hasher_->set_state(S.hs);
    std::vector<uint32_t> tmp((size_t)m * 48);
    hasher_->hash(S.toks.data(), m, tmp.data());
  }
  pos_ = S.start + m;
  S.start = -1;
  sprof_.keep_ms += t.ms();
}

int Model::spec_step(int32_t tok, std::vector<int32_t>& out, int k) {
  if (!dsp_) { out.push_back(decode(tok)); return 0; }
  Timer ts;
  if (k <= 0) k = o_.spec;
  k = std::min({k, dsp_->block_size(), sv::kMaxK - 1, max_pos() - pos_ - 1});
  if (k <= 0) { out.push_back(decode(tok)); return 0; }
  const int P = pos_;
  int32_t dr[sv::kMaxK], toks[sv::kMaxK], nx[sv::kMaxK];
  sstats_.draft_ms += dsp_->draft(tok, P, k, dr);
  toks[0] = tok;
  for (int i = 0; i < k; i++) toks[i + 1] = dr[i];
  verify(toks, k + 1, nx);
  int m = 0;
  while (m < k && dr[m] == nx[m]) m++;
  verify_keep(m + 1);
  // features of the kept tokens (positions P..P+m) for the next draft
  const float* F = verify_features();
  std::vector<float> f(3 * 5120);
  for (int j = 0; j <= m; j++) {
    for (int ti = 0; ti < 3; ti++) memcpy(f.data() + ti * 5120, F + ((size_t)ti * (k + 1) + j) * 5120, 5120 * 4);
    dsp_->add_feature(P + j, f.data());
  }
  for (int j = 0; j < m; j++) out.push_back(dr[j]);
  out.push_back(nx[m]);
  sstats_.steps++; sstats_.drafted += k; sstats_.accepted += m; sstats_.produced += m + 1; sstats_.step_ms += ts.ms();
  return m;
}

}  // namespace ms::ds41
