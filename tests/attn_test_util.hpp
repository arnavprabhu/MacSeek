// Shared helpers for the attention tests (header-only; not a test itself).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>
#include "core/metal.hpp"
#include "kernels/attn.hpp"
#include "ref/attn_ref.hpp"

namespace at {

using namespace ms;
namespace A = ms::attn;
namespace R = attnref;

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t seed) : g(seed) {}
  float normal(float sd = 1) { return std::normal_distribution<float>(0, sd)(g); }
  float uni(float a, float b) { return std::uniform_real_distribution<float>(a, b)(g); }
  void fill(float* p, size_t n, float sd) { for (size_t i = 0; i < n; i++) p[i] = normal(sd); }
};
inline uint64_t seed_of(int L, int pos, int what) { return (uint64_t)L * 1000003ull + (uint64_t)pos * 7919ull + what * 104729ull + 12345; }

inline uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
inline float h2f(uint16_t h) { return R::f16_to_f32(h); }

// ---- per-layer synthetic weights and per-token inputs ----
struct LayerW {
  std::vector<float> kv_norm, sinks, c_norm, knorm;
  std::vector<uint16_t> wk;
};
inline LayerW make_weights(int L) {
  LayerW w;
  Rng r(seed_of(L, 0, 99));
  w.kv_norm.resize(512); for (auto& x : w.kv_norm) x = r.uni(0.5f, 1.5f);
  w.sinks.resize(64); for (auto& x : w.sinks) x = r.normal(1.0f);
  w.c_norm.resize(512); for (auto& x : w.c_norm) x = r.uni(0.5f, 1.5f);
  w.knorm.resize(128); for (auto& x : w.knorm) x = r.uni(0.5f, 1.5f);
  w.wk.resize(128 * 512); for (auto& x : w.wk) x = f2bf(r.normal(1.0f / std::sqrt(512.0f)));
  return w;
}
struct TokIn { float q[64 * 512], kv[512], ckv[512], cgate[512], iq[32 * 128], iw[32]; };
inline void make_tok(int L, int p, TokIn& t) {
  Rng r(seed_of(L, p, 1));
  r.fill(t.q, 64 * 512, 0.1f);
  r.fill(t.kv, 512, 1.0f);
  r.fill(t.ckv, 512, 1.0f);
  r.fill(t.cgate, 512, 2.0f);
  r.fill(t.iq, 32 * 128, 1.0f);
  r.fill(t.iw, 32, 1.0f);
}

// ---- GPU layer harness: buffers for LayerIO of up to nmax tokens ----
struct GpuLayer {
  int L;
  LayerW w;
  Buffer kv_norm, sinks, c_norm, knorm, wk;
  Buffer q, kv, ckv, cgate, iq, iw, out;
  GpuLayer(int L_, int nmax) : L(L_), w(make_weights(L_)) {
    auto& d = Device::get();
    kv_norm = d.alloc(512 * 4); memcpy(kv_norm.cpu(), w.kv_norm.data(), 512 * 4);
    sinks = d.alloc(64 * 4); memcpy(sinks.cpu(), w.sinks.data(), 64 * 4);
    c_norm = d.alloc(512 * 4); memcpy(c_norm.cpu(), w.c_norm.data(), 512 * 4);
    knorm = d.alloc(128 * 4); memcpy(knorm.cpu(), w.knorm.data(), 128 * 4);
    wk = d.alloc(128 * 512 * 2); memcpy(wk.cpu(), w.wk.data(), 128 * 512 * 2);
    q = d.alloc((size_t)nmax * 64 * 512 * 4); kv = d.alloc((size_t)nmax * 512 * 4);
    ckv = d.alloc((size_t)nmax * 512 * 4); cgate = d.alloc((size_t)nmax * 512 * 4);
    iq = d.alloc((size_t)nmax * 32 * 128 * 4); iw = d.alloc((size_t)nmax * 32 * 4);
    out = d.alloc((size_t)nmax * 64 * 512 * 4);
  }
  void fill(int start, int n) {
    TokIn t;
    for (int i = 0; i < n; i++) {
      make_tok(L, start + i, t);
      memcpy((float*)q.cpu() + (size_t)i * 64 * 512, t.q, sizeof t.q);
      memcpy((float*)kv.cpu() + (size_t)i * 512, t.kv, sizeof t.kv);
      memcpy((float*)ckv.cpu() + (size_t)i * 512, t.ckv, sizeof t.ckv);
      memcpy((float*)cgate.cpu() + (size_t)i * 512, t.cgate, sizeof t.cgate);
      memcpy((float*)iq.cpu() + (size_t)i * 32 * 128, t.iq, sizeof t.iq);
      memcpy((float*)iw.cpu() + (size_t)i * 32, t.iw, sizeof t.iw);
    }
  }
  A::LayerIO io() const {
    A::LayerIO x;
    x.layer = L; x.q = q; x.kv = kv; x.kv_norm = kv_norm; x.sinks = sinks; x.out = out;
    x.c_kv = ckv; x.c_gate = cgate; x.c_norm = c_norm; x.idx_wk = wk; x.idx_knorm = knorm; x.idx_q = iq; x.idx_w = iw;
    return x;
  }
};

// ---- CPU stack reference (token by token) ----
struct RefStack {
  bool actq = true;
  std::vector<float> tp, ty;
  std::map<int, LayerW> w;
  std::map<int, std::map<int, std::vector<float>>> win;  // layer -> pos -> row
  R::Source src[4];
  std::vector<int> cand;  // layer-20 candidate blocks of the current token
  RefStack(int max_pos, bool actq_) : actq(actq_) {
    tp = R::rope_table(max_pos, false); ty = R::rope_table(max_pos, true);
    const int s[4] = {2, 8, 14, 20};
    for (int i = 0; i < 4; i++) R::source_init(src[i], A::ratio_of(s[i]));
  }
  // initialize from a GPU state at position S0 (ring rows, caches, compressor state)
  void load_gpu(const A::State& st, const std::vector<int>& layers, int S0) {
    for (int L : layers) {
      const uint16_t* ring = (const uint16_t*)st.win[L].cpu();
      for (int p = std::max(0, S0 - 128); p < S0; p++) {
        std::vector<float> row(512);
        for (int i = 0; i < 512; i++) row[i] = h2f(ring[(size_t)(p % 128) * 512 + i]);
        win[L][p] = row;
      }
    }
    for (int i = 0; i < 4; i++) {
      int r = src[i].ratio, rows = S0 / r;
      const uint16_t* c = (const uint16_t*)st.ckv[i].cpu();
      const uint16_t* k = (const uint16_t*)st.ikey[i].cpu();
      src[i].ckv.resize((size_t)rows * 512); src[i].ikey.resize((size_t)rows * 128);
      for (size_t j = 0; j < (size_t)rows * 512; j++) src[i].ckv[j] = h2f(c[j]);
      for (size_t j = 0; j < (size_t)rows * 128; j++) src[i].ikey[j] = h2f(k[j]);
      const float* cs = (const float*)st.cst[i].cpu();
      memcpy(src[i].st_kv.data(), cs, (size_t)r * 512 * 4);
      memcpy(src[i].st_sc.data(), cs + (size_t)r * 512, (size_t)r * 512 * 4);
    }
  }
  const LayerW& weights(int L) { if (!w.count(L)) w[L] = make_weights(L); return w[L]; }
  // one token of one layer; topk_out (index sources) gets the selected compressed rows
  void token(int L, int p, float* out, std::vector<int>* topk_out, std::vector<int>& shared_topk) {
    const LayerW& lw = weights(L);
    TokIn t; make_tok(L, p, t);
    const int r = A::ratio_of(L);
    const float* cs = &(r ? ty : tp)[(size_t)p * 64];
    std::vector<float> row(512);
    R::window_row(t.kv, lw.kv_norm.data(), cs, actq, row.data());
    win[L][p] = row;
    if (A::is_kv_source(L))
      R::compress_token(src[A::src_slot(L)], p, t.ckv, t.cgate, lw.c_norm.data(), lw.wk.data(), lw.knorm.data(), ty, actq);
    if (A::is_index_source(L)) {
      const R::Source& S = src[A::src_slot(A::kv_source_of(L))];
      int len = (p + 1) / r;
      float qq[32 * 128], ww[32];
      R::index_q(t.iq, t.iw, cs, actq, qq, ww);
      std::vector<float> sc;
      R::index_scores(qq, ww, S.ikey.data(), len, sc);
      if (A::uses_cand(L)) {
        std::vector<char> keep(len, 0);
        for (int b : cand) for (int i = b * 8; i < std::min(len, b * 8 + 8); i++) keep[i] = 1;
        for (int i = 0; i < len; i++) if (!keep[i]) sc[i] = -INFINITY;
      }
      shared_topk = R::topk_sorted(sc, R::TOPK);
      if (A::is_cand_source(L)) cand = R::candidate_blocks(sc, len);
      if (topk_out) *topk_out = shared_topk;
    }
    std::vector<const float*> rows;
    for (int wp = std::max(0, p - 127); wp <= p; wp++) rows.push_back(win[L][wp].data());
    if (r) {
      const R::Source& S = src[A::src_slot(A::kv_source_of(L))];
      for (int id : shared_topk) rows.push_back(&S.ckv[(size_t)id * 512]);
    }
    R::sparse_attn(t.q, cs, rows, lw.sinks.data(), out);
    // keep memory bounded
    if (win[L].size() > 300) win[L].erase(win[L].begin());
  }
};

struct ErrStat {
  double max_abs = 0, max_ref = 0, sum_sq = 0, sum_ref_sq = 0; size_t n = 0;
  void add(const float* g, const float* r, size_t cnt) {
    for (size_t i = 0; i < cnt; i++) {
      double d = std::fabs((double)g[i] - r[i]);
      if (!(d <= max_abs)) max_abs = d;  // NaN propagates
      max_ref = std::max(max_ref, (double)std::fabs(r[i]));
      sum_sq += d * d; sum_ref_sq += (double)r[i] * r[i]; n++;
    }
  }
  double rel_rms() const { return std::sqrt(sum_sq / std::max(sum_ref_sq, 1e-30)); }
  double rel_max() const { return max_abs / std::max(max_ref, 1e-30); }
};

inline int g_fail = 0;
inline void check(bool ok, const char* what) {
  printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) g_fail++;
}

}  // namespace at
