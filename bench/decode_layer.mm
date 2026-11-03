// Decode token GPU-time benchmark on synthetic weights with the real per-layer shapes/types (tests/dec_synth.hpp):
// embed + engram prologue + 40 x (encode_dec_attn_side + encode_dec_moe_table) + head, exactly the sequence of
// Model::forward's table decode path (src/model/dec_layer.*), T tokens per command buffer. Layout: --layout coder
// (default: GSQ-RCO Coder, 192 experts, attention state at position 260000 of 262144) or --layout ud (UD-IQ1_S, 384
// experts, position 200000).
// The router handshake events are encoded as in the model, with `go` pre-signaled (GPU never waits on the CPU), so the
// numbers are the pure GPU time of a token incl. the signal/encoder-split costs. docs/modules/dec_gpu.md has results.
//
//   bench_decode_layer [--layout coder|ud] [--tokens 20] [--reps 7] [--copies 2] [--pos P] [--nohead] [--check]
//                      [--breakdown] [--bw GBS] [--sync shared|sgu|event|none] [--cpu US] [--layers N] cfg...
// --cpu US: real handshake: a CPU thread spins on rdy and sets go US microseconds later (emulates ensure() + table
//           writes on all-hit layers), instead of a pre-signaled go. Then the GPU time includes the exposed handshake.
// cfg: "base" (all switches off), "fuse" (MACSEEK_DEC_FUSE=1 defaults), or comma-separated key=value over the base:
//   fuse reorder normqb(+tpt) combine sgur(0..2) engbg(0..2) hcs(hc row splits) gunr gursg dnr drsg (MoE launch)
//   probe(extra tiny dispatches) skip(hex/dec DecSkip bits, PROFILING) sync(0..2|-1 none)
// Configs are run round-robin per rep (interleaved; a model job may share the GPU); best and median per config.
// --breakdown: base config with each component removed (PROFILING: wrong outputs), delta = component's cost; the
//              table lists each component's bytes read per token and its floor at --bw GB/s (default 285).
// --check: restore the attention state, run one command buffer per config and compare h / logits / state with cfg 0.
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include "../tests/dec_synth.hpp"
#include "core/env.hpp"
#include "kernels/arch.hpp"
#include "kernels/attn.hpp"
#include "kernels/quant.hpp"
#include "model/config.hpp"
#include "model/dec_layer.hpp"

using namespace ms;
using namespace ms::ds41;
using q::QType;

namespace {

struct Cfg {
  std::string name;
  DecOptions o;
  int sync = 2;   // -1 none, 0 event, 1 sgu, 2 shared
  bool nohead = false;   // breakdown: output head (+ argmax) not encoded
};

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

Cfg parse_cfg(const std::string& s, int dsync) {
  Cfg c; c.name = s; c.sync = dsync;
  c.o = DecOptions{};    // all off
  c.o.pred_on = true; c.o.pred_depth = 1; c.o.pred_q8 = true;
  // "fuse" = DecOptions::from_env with MACSEEK_DEC_FUSE=1 and nothing else set
  auto set_fuse = [&](bool f) {
    c.o.fuse = f; c.o.reorder = f; c.o.attn_comb = f; c.o.sgu_router = f ? 1 : 0; c.o.engram_bg = f ? 2 : 0;
    c.o.hc_splits = f ? 2 : 1;
  };
  if (s == "base") return c;
  if (s == "fuse") { set_fuse(true); return c; }
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
    std::string kv = s.substr(p, q - p);
    p = q + 1;
    size_t eq = kv.find('=');
    std::string k = kv.substr(0, eq);
    long v = eq == std::string::npos ? 1 : strtol(kv.c_str() + eq + 1, nullptr, 0);
    if (k == "fuse") set_fuse(v != 0);
    else if (k == "reorder") c.o.reorder = v;
    else if (k == "normqb") c.o.norm_qb = v;
    else if (k == "combine") c.o.attn_comb = v;
    else if (k == "sgur") c.o.sgu_router = v;
    else if (k == "engbg") c.o.engram_bg = (int)v;
    else if (k == "skip") c.o.skip |= (uint32_t)v;
    else if (k == "sync") c.sync = (int)v;
    else if (k == "tpt") c.o.norm_qb_tpt = (int)v;
    else if (k == "hcs") c.o.hc_splits = (int)v;
    else if (k == "probe") c.o.probe_dispatches = (int)v;
    else if (k == "gunr") c.o.moe_gu_nr = (int)v;
    else if (k == "gursg") c.o.moe_gu_rsg = (int)v;
    else if (k == "dnr") c.o.moe_d_nr = (int)v;
    else if (k == "drsg") c.o.moe_d_rsg = (int)v;
    else { fprintf(stderr, "unknown cfg key %s\n", k.c_str()); exit(1); }
  }
  return c;
}

struct Bench {
  Config cfg;
  dsyn::Pool P;
  std::vector<LayerW> L;
  std::vector<W> pred;
  attn::State S;
  arch::Scratch asc;
  DecBufs B;
  Buffer embrows, erow, logits, amax_i, amax_v, tbl;
  size_t emb_rb = 0;
  W out_norm, head;
  std::vector<Buffer> arena;
  id<MTLSharedEvent> rdy, go;
  uint64_t evbase = 0;
  int pos0 = -1, T = 20, nlayers = 40;
  bool nohead = false;
  Buffer last_h;        // current hyper-connection state buffer after the last encoded token (ping-pong)
  double cpu_us = -1;   // < 0: go pre-signaled
  // saved state (check mode)
  std::vector<std::vector<uint8_t>> saved;

  void setup(int copies) {
    auto& d = Device::get();
    double t0 = now_ms();
    L.resize(cfg.n_layers); pred.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) { L[l] = dsyn::make_layer(P, cfg, l, copies); pred[l] = dsyn::make_pred(P, l, copies); }
    out_norm = P.vec("out_norm", 5120, 0, 1.f, 0.1f);
    if (!nohead) head = P.get("head", dsyn::head_type(), 129280, 5120, 0, 0.014f);
    attn::Options ao = attn::Options::from_env();
    ao.max_chunk = 1;
    ao.max_pos = std::max(262144, pos0 + 1024);
    attn::init(S, ao);
    attn::reset(S);
    dsyn::fill_attn_state(S, pos0, 77);
    asc = arch::Scratch::create(64, 16);
    auto A = [&](size_t n, const char* lb) { return d.alloc(n, lb); };
    B.h = A(20480 * 4, "h"); B.h2 = A(20480 * 4, "h2"); B.x = A(5120 * 4, "x"); B.sub = A(5120 * 4, "sub");
    B.mixA = A(24 * 4, "mixA"); B.mixF = A(24 * 4, "mixF"); B.mix_ident = A(24 * 4, "mix_ident");
    arch::fill_identity_mix((float*)B.mix_ident.cpu());
    B.qa = A(1280 * 4, "qa"); B.qrn = A(1280 * 4, "qrn"); B.q = A(32768 * 4, "q"); B.kv = A(512 * 4, "kv");
    B.ckv = A(512 * 4, "ckv"); B.cgate = A(512 * 4, "cgate"); B.iq = A(4096 * 4, "iq"); B.iw = A(32 * 4, "iw");
    B.aout = A(32768 * 4, "aout"); B.oa = A(8192 * 4, "oa"); B.rids = A(64, "rids"); B.rw = A(64, "rw");
    B.plog = A(4 * 384 * 4, "plog");   // >= pred_depth * n_experts
    B.ekv[0] = A(25600 * 4, "ekv0"); B.ekv[1] = A(25600 * 4, "ekv1");
    B.moe_scratch = A(q::moe_decode_scratch_bytes(), "moe_dec");
    B.edeq = A(2 * 6144 * 4, "edeq");
    emb_rb = q::row_bytes(dsyn::head_type(), 5120);
    embrows = A(emb_rb + 64, "embrows");
    dsyn::fill_rows(dsyn::head_type(), (uint8_t*)embrows.cpu(), 1, 5120, 999, 1.0f);
    erow = A(2 * 24 * 144, "erow");
    dsyn::fill_rows(QType::IQ4_NL, (uint8_t*)erow.cpu(), 48, 256, 998, 1.0f);
    logits = A(129280 * 4, "logits"); amax_i = A(64, "amax_i"); amax_v = A(64, "amax_v");
    // routed experts: per (gate/up, down) type combo a pool of NE experts; layer l uses experts (6l + k) % NE
    const int NE = 12;
    tbl = A(40 * 6 * 3 * 8, "tbl");
    struct Combo { QType gu, dn; Buffer b; size_t gb, db; };
    std::vector<Combo> combos;
    for (int l = 0; l < cfg.n_layers; l++) {
      QType gu = dsyn::exp_gu_type(l), dn = dsyn::exp_down_type(l);
      int ci = -1;
      for (size_t i = 0; i < combos.size(); i++) if (combos[i].gu == gu && combos[i].dn == dn) ci = (int)i;
      if (ci < 0) {
        Combo c{gu, dn, {}, q::row_bytes(gu, 5120) * 2304, q::row_bytes(dn, 2304) * 5120};
        c.b = d.alloc((2 * c.gb + c.db) * NE, "experts");
        for (int e = 0; e < NE; e++) {
          uint8_t* p = (uint8_t*)c.b.cpu() + (2 * c.gb + c.db) * e;
          dsyn::fill_rows(gu, p, 2304, 5120, 5000 + e * 3, 0.014f);
          dsyn::fill_rows(gu, p + c.gb, 2304, 5120, 5001 + e * 3, 0.014f);
          dsyn::fill_rows(dn, p + 2 * c.gb, 5120, 2304, 5002 + e * 3, 0.021f);
        }
        P.bytes += (2 * c.gb + c.db) * NE;
        combos.push_back(c);
        arena.push_back(c.b);
        ci = (int)combos.size() - 1;
      }
      const Combo& c = combos[ci];
      const uint64_t base = [(__bridge id<MTLBuffer>)c.b.buf gpuAddress] + c.b.off;
      uint64_t* t = (uint64_t*)tbl.cpu() + (size_t)l * 18;
      for (int k = 0; k < 6; k++) {
        const int e = (6 * l + k) % NE;
        const uint64_t a = base + (2 * c.gb + c.db) * e;
        t[k * 3 + 0] = a; t[k * 3 + 1] = a + c.gb; t[k * 3 + 2] = a + 2 * c.gb;
      }
    }
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d.device();
    rdy = [dev newSharedEvent];
    go = [dev newSharedEvent];
    if (cpu_us < 0) go.signaledValue = 1ull << 60;   // every go wait is already satisfied
    fprintf(stderr, "setup: %.2f GB synthetic weights, %.1f s\n", P.bytes / 1e9, (now_ms() - t0) / 1e3);
  }

  // encode one command buffer of T tokens
  void encode(Enc& e, const Cfg& c) {
    DecCtx ctx; ctx.cfg = &cfg; ctx.L = &L; ctx.pred_w = &pred; ctx.B = B; ctx.ast = &S; ctx.asc = &asc; ctx.o = c.o;
    for (int t = 0; t < T; t++) {
      const int pos = pos0 + t;
      arch::encode_embed(e, (int)dsyn::head_type(), embrows, (uint32_t)emb_rb, B.h, 1, 4);
      encode_dec_prologue(e, ctx, erow);
      DecSync s;
      if (c.sync >= 0) { s.rdy = (__bridge void*)rdy; s.go = (__bridge void*)go; s.mode = c.sync; }
      else s.mode = 0;
      s.value_base = evbase;
      evbase += cfg.n_layers;
      for (int l = 0; l < nlayers; l++) {
        encode_dec_attn_side(e, ctx, l, pos);
        encode_dec_moe_table(e, ctx, l, dsyn::exp_gu_type(l), dsyn::exp_down_type(l), tbl.sub((size_t)l * 144, 144), arena, s);
      }
      dec_bg_flush(e, ctx, -1);   // engram_bg: chunks not placed yet (none with the full 40 layers)
      last_h = dec_cur_h(ctx);
      arch::HcStep a; a.h_in = last_h; a.h_out = last_h; a.x_sub = B.sub; a.mix_prev = B.mixF; a.norm_w = out_norm.b; a.x_out = B.x; a.n = 1;
      arch::encode_hc_step(e, a, asc); e.barrier();
      if (!nohead && !c.nohead) {
        q::encode_matvec(e, head.t, head.b, head.rows, head.cols, B.x, logits, false);
        e.barrier();
        arch::encode_argmax(e, logits, 1, 129280, amax_i, amax_v, asc, 0.f, 0);
        e.barrier();
      }
    }
  }

  // bytes read per token by breakdown component k (weights + attention state), for the bandwidth floor
  double comp_bytes(int k) const {
    double b = 0;
    auto wb = [](const W& w) { return w ? (double)q::row_bytes(w.t, w.cols) * w.rows : 0.0; };
    const int pos = pos0;
    for (int l = 0; l < nlayers; l++) {
      const LayerW& w = L[l];
      switch (k) {
        case 0: b += wb(w.hc_attn_fn) + 2 * 20480 * 4 + (cfg.is_engram(l) ? wb(w.engram_gate) : 0); break;
        case 1: b += wb(w.q_a) + wb(w.kv) + wb(w.c_kv) + wb(w.c_gate) + wb(w.idx_proj); break;
        case 3: b += wb(w.q_b) + wb(w.idx_qb); break;
        case 4: b += wb(w.idx_k); break;
        case 5:
          if (cfg.is_index_source(l)) {
            const int r = attn::ratio_of(l);
            b += attn::uses_cand(l) ? 2048.0 * 8 * attn::kID * 2 : (double)(pos / r) * attn::kID * 2;
          }
          break;
        case 6: b += (attn::kWin + (attn::ratio_of(l) ? attn::kTopk : 0)) * (double)attn::kD * 2; break;
        case 7: b += wb(w.o_a); break;
        case 8: b += wb(w.o_b); break;
        case 9: b += wb(w.hc_ffn_fn) + 2 * 20480 * 4; break;
        case 10: b += wb(w.gate_inp); break;
        case 11: if (l + 1 < cfg.n_layers) b += wb(pred[l + 1]); break;
        case 13: b += wb(w.sh_gate) + wb(w.sh_up) + wb(w.sh_down); break;
        case 14: b += 6.0 * (2.0 * q::row_bytes(dsyn::exp_gu_type(l), 5120) * 2304 + (double)q::row_bytes(dsyn::exp_down_type(l), 2304) * 5120); break;
        case 17: if (cfg.is_engram(l)) b += wb(w.engram_kv); break;
        default: break;
      }
    }
    if (k == 18 && !nohead) b += wb(head) + 129280 * 4.0 * 2;
    return b;
  }

  struct Res { double gpu_ms, wall_ms; EncStats st; };
  Res run(const Cfg& c) {
    EncStats s0 = enc_stats();
    Enc e;
    encode(e, c);
    EncStats s1 = enc_stats();
    double t0 = now_ms();
    const uint64_t v0 = evbase - (uint64_t)T * cfg.n_layers;
    std::thread resp;
    if (cpu_us >= 0 && c.sync >= 0 && !(c.o.skip & kSkipEvents))
      resp = std::thread([&, v0] {   // answer every layer's rdy like Model::forward does on all-hit layers
        for (int t = 0; t < T; t++)
          for (int l = 0; l < nlayers; l++) {
            const uint64_t v = v0 + (uint64_t)t * cfg.n_layers + 1 + l;
            while (rdy.signaledValue < v) {}
            if (cpu_us > 0) { const double t0 = now_ms(); while (now_ms() - t0 < cpu_us * 1e-3) {} }
            go.signaledValue = v;
          }
      });
    e.commit();
    e.wait();
    if (resp.joinable()) resp.join();
    Res r{e.gpu_ms(), now_ms() - t0, {}};
    r.st.dispatches = s1.dispatches - s0.dispatches; r.st.barriers = s1.barriers - s0.barriers;
    r.st.stages = s1.stages - s0.stages; r.st.encoders = s1.encoders - s0.encoders;
    return r;
  }

  // attention state written by T decode tokens at pos0.. (check mode restores it before every run)
  std::vector<Buffer> state_bufs(std::vector<std::pair<size_t, size_t>>& rng) {
    std::vector<Buffer> v;
    rng.clear();
    for (int l = 0; l < attn::kLayers; l++) { v.push_back(S.win[l]); rng.push_back({0, S.win[l].size}); }
    const int srcs[4] = {2, 8, 14, 20};
    for (int i = 0; i < 4; i++) {
      const int r = attn::ratio_of(srcs[i]);
      const size_t j0 = pos0 / r, j1 = (pos0 + T) / r + 1;
      v.push_back(S.ckv[i]); rng.push_back({j0 * attn::kD * 2, (j1 - j0) * attn::kD * 2});
      v.push_back(S.ikey[i]); rng.push_back({j0 * attn::kID * 2, (j1 - j0) * attn::kID * 2});
      v.push_back(S.cst[i]); rng.push_back({0, S.cst[i].size});
    }
    return v;
  }
  void save_state() {
    std::vector<std::pair<size_t, size_t>> rng;
    auto v = state_bufs(rng);
    saved.clear();
    for (size_t i = 0; i < v.size(); i++) {
      const uint8_t* p = (const uint8_t*)v[i].cpu() + rng[i].first;
      saved.emplace_back(p, p + rng[i].second);
    }
  }
  void restore_state() {
    std::vector<std::pair<size_t, size_t>> rng;
    auto v = state_bufs(rng);
    for (size_t i = 0; i < v.size(); i++) memcpy((uint8_t*)v[i].cpu() + rng[i].first, saved[i].data(), rng[i].second);
    memset(B.h.cpu(), 0, B.h.size); memset(B.h2.cpu(), 0, B.h2.size); memset(B.sub.cpu(), 0, B.sub.size);
  }
  std::vector<uint8_t> snapshot() {
    std::vector<uint8_t> out;
    auto add = [&](const void* p, size_t n) { const uint8_t* b = (const uint8_t*)p; out.insert(out.end(), b, b + n); };
    add(last_h.cpu(), B.h.size); add(B.x.cpu(), B.x.size); add(B.sub.cpu(), B.sub.size);
    if (!nohead) add(logits.cpu(), logits.size);
    std::vector<std::pair<size_t, size_t>> rng;
    auto v = state_bufs(rng);
    for (size_t i = 0; i < v.size(); i++) add((const uint8_t*)v[i].cpu() + rng[i].first, rng[i].second);
    return out;
  }
};

constexpr int kNComp = 19;
const char* kComp[kNComp] = {"hcA", "proj", "norm", "qb", "kvwin", "index", "sparse", "oa", "ob", "hcF", "router", "pred",
                             "fill", "moe_shared", "moe_routed", "barriers", "events", "engram", "head"};

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

}  // namespace

int main(int argc, char** argv) {
  int reps = 7, copies = 2, dsync = 2;
  double bw = 285.0;
  std::string lay = "coder";
  bool check = false, breakdown = false;
  Bench b;
  std::vector<std::string> names;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (a == "--tokens") b.T = std::stoi(nx());
    else if (a == "--reps") reps = std::stoi(nx());
    else if (a == "--copies") copies = std::stoi(nx());
    else if (a == "--pos") b.pos0 = std::stoi(nx());
    else if (a == "--nohead") b.nohead = true;
    else if (a == "--layers") b.nlayers = std::stoi(nx());
    else if (a == "--cpu") b.cpu_us = std::stod(nx());
    else if (a == "--check") check = true;
    else if (a == "--breakdown") breakdown = true;
    else if (a == "--layout") lay = nx();
    else if (a == "--bw") bw = std::stod(nx());
    else if (a == "--sync") { std::string v = nx(); dsync = v == "none" ? -1 : v == "event" ? 0 : v == "sgu" ? 1 : 2; }
    else names.push_back(a);
  }
  if (names.empty()) names.push_back("base");
  if (lay != "coder" && lay != "ud") { fprintf(stderr, "--layout coder|ud\n"); return 1; }
  dsyn::layout() = lay == "coder" ? dsyn::Layout::Coder : dsyn::Layout::UD;
  arch::options().n_experts = dsyn::n_experts();
  if (b.pos0 < 0) b.pos0 = dsyn::coder() ? 260000 : 200000;
  std::vector<Cfg> cfgs;
  for (auto& n : names) cfgs.push_back(parse_cfg(n, dsync));
  if (breakdown) {
    const Cfg base = cfgs[0];
    for (int k = 0; k < kNComp; k++) {
      Cfg c = base; c.name = base.name + "-" + kComp[k];
      if (k == 18) c.nohead = true; else c.o.skip |= 1u << k;
      cfgs.push_back(c);
    }
  }
  b.setup(copies);
  if (check) {
    b.save_state();
    std::vector<uint8_t> ref, prev;
    for (size_t i = 0; i < cfgs.size(); i++) {
      b.restore_state();
      b.run(cfgs[i]);
      std::vector<uint8_t> s = b.snapshot();
      const float* hf = (const float*)s.data();
      double hr = 0; bool fin = true;
      for (int k = 0; k < 20480; k++) { hr += (double)hf[k] * hf[k]; fin = fin && std::isfinite(hf[k]); }
      if (i == 0) {
        ref = s; prev = s;
        printf("check %-28s h rms %.4g finite %d argmax %u\n", cfgs[i].name.c_str(), std::sqrt(hr / 20480), (int)fin,
               *(uint32_t*)b.amax_i.cpu());
        continue;
      }
      size_t diff = 0; double maxrel = 0;
      // float compare over the first part (h, x, sub, logits); bytes for the state
      const size_t nf = (20480 + 5120 + 5120 + (b.nohead ? 0 : 129280));
      const float* a = (const float*)ref.data(); const float* c = (const float*)s.data();
      for (size_t k = 0; k < nf; k++) {
        if (memcmp(a + k, c + k, 4)) { diff++; double den = std::max(1e-30, (double)std::fabs(a[k])); maxrel = std::max(maxrel, std::fabs((double)a[k] - c[k]) / den); }
      }
      size_t sdiff = 0;
      for (size_t k = nf * 4; k < s.size(); k++) sdiff += ref[k] != s[k];
      const bool same_prev = prev == s;
      printf("check %-28s vs %s: float diffs %zu (max rel %.3g), state byte diffs %zu, argmax %u, same as previous run %d -> %s\n",
             cfgs[i].name.c_str(), cfgs[0].name.c_str(), diff, maxrel, sdiff, *(uint32_t*)b.amax_i.cpu(), (int)same_prev,
             (diff == 0 && sdiff == 0) ? "BIT-IDENTICAL" : "DIFFERENT");
      prev = s;
    }
    return 0;
  }
  // warmup (pipelines + clocks)
  for (auto& c : cfgs) b.run(c);
  std::vector<std::vector<double>> g(cfgs.size()), w(cfgs.size());
  std::vector<EncStats> st(cfgs.size());
  for (int r = 0; r < reps; r++)
    for (size_t i = 0; i < cfgs.size(); i++) {
      auto res = b.run(cfgs[i]);
      g[i].push_back(res.gpu_ms / b.T); w[i].push_back(res.wall_ms / b.T); st[i] = res.st;
    }
  printf("decode token, layout %s, pos %d, %d tokens/CB, %d reps, sync %d%s, go %s\n", lay.c_str(), b.pos0, b.T, reps, dsync, b.nohead ? ", no head" : "",
         b.cpu_us < 0 ? "pre-signaled" : ("CPU responder +" + std::to_string((int)b.cpu_us) + " us").c_str());
  printf("%-30s %9s %9s %9s %8s | %6s %6s %6s %4s  per token\n", "config", "best ms", "med ms", "wall med", "d best", "disp", "barr", "stages", "enc");
  const double b0 = *std::min_element(g[0].begin(), g[0].end());
  for (size_t i = 0; i < cfgs.size(); i++) {
    const double best = *std::min_element(g[i].begin(), g[i].end());
    printf("%-30s %9.3f %9.3f %9.3f %+8.3f | %6.1f %6.1f %6.1f %4.1f\n", cfgs[i].name.c_str(), best, median(g[i]), median(w[i]),
           best - b0, st[i].dispatches / (double)b.T, st[i].barriers / (double)b.T, st[i].stages / (double)b.T,
           st[i].encoders / (double)b.T);
  }
  if (breakdown) {
    printf("\nbreakdown (delta = base best - best without the component; floor = bytes / %.0f GB/s)\n", bw);
    printf("%-12s %9s %9s %9s %9s %7s\n", "component", "delta ms", "d med ms", "MB/token", "floor ms", "%floor");
    double tb = 0, tf = 0;
    const double m0 = median(g[0]);
    for (int k = 0; k < kNComp; k++) {
      const size_t i = names.size() + k;
      const double best = *std::min_element(g[i].begin(), g[i].end());
      const double by = b.comp_bytes(k), fl = by / bw / 1e6;
      tb += by; tf += fl;
      printf("%-12s %9.3f %9.3f %9.1f %9.3f %6.0f%%\n", kComp[k], b0 - best, m0 - median(g[i]), by / 1e6, fl,
             fl > 0 && b0 - best > 0 ? 100.0 * fl / (b0 - best) : 0.0);
    }
    printf("%-12s %9s %9s %9.1f %9.3f   (base best %.3f ms, median %.3f)\n", "total", "", "", tb / 1e6, tf, b0, m0);
  }
  return 0;
}
