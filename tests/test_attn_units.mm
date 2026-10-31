// Unit tests: quant-dequant sims (bit-exact), rope tables, top-k selection, indexer decode scoring, sparse attention.
#include "attn_test_util.hpp"
#include <algorithm>
#include <set>

using namespace at;

static void test_quant() {
  printf("quant sims (bit-exact vs CPU reference)\n");
  auto& d = Device::get();
  const uint32_t n = 1 << 20;
  std::vector<float> x(n);
  Rng r(7);
  for (uint32_t b = 0; b < n; b += 32) {
    float sc = std::ldexp(1.0f, (int)r.uni(-30, 12));
    int kind = (int)r.uni(0, 10);
    for (int i = 0; i < 32; i++) {
      float v = r.normal(sc);
      if (kind == 0) v = 0;                                              // all-zero block
      if (kind == 1) v = std::ldexp((float)(int)r.uni(-64, 64) * 0.5f, (int)r.uni(-10, 4));  // many exact ties
      x[b + i] = v;
    }
  }
  Buffer g = d.alloc(n * 4);
  for (int mode = 0; mode < 3; mode++) {
    memcpy(g.cpu(), x.data(), n * 4);
    { Enc e; A::encode_quant_sim(e, g, n, mode); e.wait(); }
    std::vector<float> ref = x;
    if (mode == 0) R::fp8_sim(ref.data(), n);
    if (mode == 1) R::fp4_sim_e8m0(ref.data(), n);
    if (mode == 2) R::fp4_sim_e4m3(ref.data(), n);
    size_t bad = 0;
    const float* gp = (const float*)g.cpu();
    for (uint32_t i = 0; i < n; i++) if (memcmp(&gp[i], &ref[i], 4) != 0) { if (bad < 3) printf("    i=%u x=%a gpu=%a ref=%a\n", i, x[i], gp[i], ref[i]); bad++; }
    char msg[128]; snprintf(msg, sizeof msg, "mode %d: %zu / %u mismatches", mode, bad, n);
    check(bad == 0, msg);
  }
  // spot values of the formats
  check(R::round_e4m3(448) == 448 && R::round_e4m3(0x1p-10f) == 0 && R::round_e4m3(0x1.8p-10f) == 0x1p-9f &&
        R::round_e4m3(17) == 16 && R::round_e4m3(19) == 20, "e4m3 rounding spot checks");
  check(R::round_e2m1(0.25f) == 0 && R::round_e2m1(0.75f) == 1 && R::round_e2m1(2.5f) == 2 && R::round_e2m1(5) == 4 &&
        R::round_e2m1(5.5f) == 6, "e2m1 rounding spot checks");
}

static void test_topk() {
  printf("top-k selection (attn_topk_dec)\n");
  auto& d = Device::get();
  const int NMAX = 262144;
  Buffer sc = d.alloc(NMAX * 4), bl = d.alloc(2048 * 4), out = d.alloc(2048 * 4);
  auto run = [&](const std::vector<float>& v, int k) {
    memcpy(sc.cpu(), v.data(), v.size() * 4);
    Enc e;
    struct { int n, k, lm; } a{(int)v.size(), k, 0};
    e.set_pipeline("attn_topk_dec"); e.set_buffer(0, sc); e.set_buffer(1, bl); e.set_buffer(2, out); e.set_value(3, a);
    e.dispatch_groups(1, 1, 1, 1024); e.wait();
    std::vector<int> o((int*)out.cpu(), (int*)out.cpu() + k);
    return o;
  };
  auto expect = [&](const std::vector<float>& v, int k) {
    auto ref = R::topk_sorted(v, k);
    ref.resize(k, -1);
    return ref;
  };
  Rng r(3);
  struct Case { const char* name; int n, k, kind; } cases[] = {
      {"n=0", 0, 512, 0}, {"n=5", 5, 512, 0}, {"n=512", 512, 512, 0}, {"n=513", 513, 512, 0},
      {"n=100000 random", 100000, 512, 0}, {"n=200000 random", 200000, 512, 0}, {"n=262144 random", 262144, 512, 0},
      {"ties: all equal n=5000", 5000, 512, 1}, {"ties: few levels n=30000", 30000, 512, 2},
      {"with -inf holes n=50000", 50000, 512, 3}, {"blocks k=2048 n=25000", 25000, 2048, 0},
      {"negative scores n=10000", 10000, 512, 4}};
  for (auto& c : cases) {
    std::vector<float> v(c.n);
    for (int i = 0; i < c.n; i++) {
      v[i] = r.normal(1);
      if (c.kind == 1) v[i] = 1.5f;
      if (c.kind == 2) v[i] = (float)(int)r.uni(0, 8);
      if (c.kind == 3 && (i % 3)) v[i] = -INFINITY;
      if (c.kind == 4) v[i] = -std::fabs(v[i]) - 1;
    }
    if (c.n > 10) v[c.n / 2] = INFINITY;  // pinned element must always be selected
    auto g = run(v, c.k), e = expect(v, c.k);
    check(g == e, c.name);
  }
}

static void test_sparse() {
  printf("sparse attention (decode split-K and prefill-shaped) vs CPU reference\n");
  A::Options o; o.max_pos = 8192; o.max_chunk = 64;
  A::State st; A::init(st, o);
  auto& d = Device::get();
  const int L = 3;  // ratio 2 layer (source 2)
  Rng r(11);
  // random window ring and compressed rows (any half values)
  uint16_t* ring = (uint16_t*)st.win[L].cpu();
  for (int i = 0; i < 128 * 512; i++) ring[i] = R::f32_to_f16(r.normal(1));
  uint16_t* ckv = (uint16_t*)st.ckv[0].cpu();
  const int crow = 3000;
  for (int i = 0; i < crow * 512; i++) ckv[i] = R::f32_to_f16(r.normal(1));
  Buffer q = d.alloc(64 * 512 * 4), sinks = d.alloc(64 * 4), out = d.alloc(64 * 512 * 4);
  float* qp = (float*)q.cpu();
  for (int i = 0; i < 64 * 512; i++) qp[i] = r.normal(0.1f);
  for (int i = 0; i < 64; i++) ((float*)sinks.cpu())[i] = r.normal(1);
  auto tab = R::rope_table(o.max_pos, true);
  for (int pos : {5, 127, 1000, 6000}) {
    for (int nvalid : {0, 7, 512}) {
      int* tk = (int*)st.topk.cpu();
      std::vector<int> ids;
      for (int i = 0; i < 512; i++) tk[i] = -1;
      for (int i = 0; i < nvalid; i++) { tk[i] = (int)r.uni(0, crow - 1); ids.push_back(tk[i]); }
      { Enc e; A::encode_sparse(e, st, L, q, sinks, out, pos, 1); e.wait(); }
      std::vector<const float*> rows;
      std::vector<std::vector<float>> store;
      store.reserve(700);
      for (int wp = std::max(0, pos - 127); wp <= pos; wp++) {
        store.emplace_back(512);
        for (int i = 0; i < 512; i++) store.back()[i] = h2f(ring[(size_t)(wp % 128) * 512 + i]);
        rows.push_back(store.back().data());
      }
      for (int id : ids) {
        store.emplace_back(512);
        for (int i = 0; i < 512; i++) store.back()[i] = h2f(ckv[(size_t)id * 512 + i]);
        rows.push_back(store.back().data());
      }
      std::vector<float> ref(64 * 512);
      R::sparse_attn(qp, &tab[(size_t)pos * 64], rows, (float*)sinks.cpu(), ref.data());
      ErrStat es; es.add((float*)out.cpu(), ref.data(), 64 * 512);
      char msg[160];
      snprintf(msg, sizeof msg, "decode pos=%d topk=%d: rel_rms %.2e rel_max %.2e", pos, nvalid, es.rel_rms(), es.rel_max());
      check(es.rel_rms() < 3e-3 && es.rel_max() < 1e-2, msg);
    }
  }
}

static void test_index_decode() {
  printf("indexer decode scoring + top-k (full scan, candidate blocks, list mode) vs CPU reference\n");
  A::Options o; o.max_pos = 65536; o.max_chunk = 16;
  A::State st; A::init(st, o);
  auto& d = Device::get();
  Rng r(5);
  // source 20 (ratio 1) keys: fp4-simulated random vectors
  const int pos = 40000, len = pos + 1;
  uint16_t* ik = (uint16_t*)st.ikey[3].cpu();
  std::vector<float> keys((size_t)len * 128);
  for (int t = 0; t < len; t++) {
    float k[128];
    for (int i = 0; i < 128; i++) k[i] = r.normal(1);
    R::fp4_sim_e8m0(k, 128);
    for (int i = 0; i < 128; i++) { ik[(size_t)t * 128 + i] = R::f32_to_f16(k[i]); keys[(size_t)t * 128 + i] = k[i]; }
  }
  Buffer iq = d.alloc(32 * 128 * 4), iw = d.alloc(32 * 4);
  float* qp = (float*)iq.cpu();
  for (int i = 0; i < 4096; i++) qp[i] = r.normal(1);
  for (int i = 0; i < 32; i++) ((float*)iw.cpu())[i] = r.normal(1);
  auto tab = R::rope_table(o.max_pos, true);
  float qq[4096], ww[32];
  R::index_q(qp, (float*)iw.cpu(), &tab[(size_t)pos * 64], true, qq, ww);
  std::vector<float> sc;
  R::index_scores(qq, ww, keys.data(), len, sc);
  // layer 20: candidate source
  { Enc e; A::encode_index_decode(e, st, 20, iq, iw, pos); e.wait(); }
  ErrStat es; es.add((float*)st.dec_scores.cpu(), sc.data(), len);
  char msg[160];
  snprintf(msg, sizeof msg, "layer 20 scores: rel_max %.2e", es.rel_max());
  check(es.rel_max() < 1e-5, msg);
  auto ref_top = R::topk_sorted(sc, 512);
  std::vector<int> g((int*)st.topk.cpu(), (int*)st.topk.cpu() + 512);
  check(g == ref_top, "layer 20 top-512 == reference");
  auto ref_cand = R::candidate_blocks(sc, len);
  std::vector<int> gc((int*)st.cand_list.cpu(), (int*)st.cand_list.cpu() + 2048);
  check(gc == ref_cand, "layer 20 candidate blocks == reference (2048, newest pinned)");
  // layer 24: candidate user, list mode
  for (int i = 0; i < 4096; i++) qp[i] = r.normal(1);
  R::index_q(qp, (float*)iw.cpu(), &tab[(size_t)pos * 64], true, qq, ww);
  R::index_scores(qq, ww, keys.data(), len, sc);
  std::vector<char> keep(len, 0);
  for (int b : ref_cand) for (int i = b * 8; i < std::min(len, b * 8 + 8); i++) keep[i] = 1;
  for (int i = 0; i < len; i++) if (!keep[i]) sc[i] = -INFINITY;
  { Enc e; A::encode_index_decode(e, st, 24, iq, iw, pos); e.wait(); }
  ref_top = R::topk_sorted(sc, 512);
  g.assign((int*)st.topk.cpu(), (int*)st.topk.cpu() + 512);
  check(g == ref_top, "layer 24 (candidate-restricted) top-512 == reference");
  // small len (< 512 visible)
  { Enc e; A::encode_index_decode(e, st, 20, iq, iw, 100); e.wait(); }
  g.assign((int*)st.topk.cpu(), (int*)st.topk.cpu() + 512);
  bool ok = true;
  for (int i = 0; i < 512; i++) ok &= g[i] == (i <= 100 ? i : -1);
  check(ok, "pos=100: all 101 visible rows selected, rest -1");
}

int main() {
  test_quant();
  test_topk();
  test_index_decode();
  test_sparse();
  printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
