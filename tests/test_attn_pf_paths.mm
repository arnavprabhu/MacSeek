// Prefill fast paths vs their fallbacks (GPU vs GPU) and the threshold-filtered top-k merge vs the CPU reference:
//   1. merge: attn_topk_merge_f == attn_topk_merge == CPU top-k over several key chunks (random, -inf holes, ties)
//   2. a stack of layers {0, 2, 20, 24} at depth (random 40K-position caches, candidate masking active), one prefill
//      chunk: every fast-path switch on (defaults) vs all off: identical top-k lists, attention outputs within 5e-4
//      rel rms; plus each switch off on its own
//   3. shallow (from position 0, many invalid rows / -1 top-k padding -> skipped tiles): defaults vs all off
#include "attn_test_util.hpp"
#include <algorithm>
#include <set>

using namespace at;

struct MergeArgs { int n, k, cnt, ld_s, base, first; };

static void test_merge_filter() {
  printf("threshold-filtered running top-k merge vs full merge vs CPU reference\n");
  auto& d = Device::get();
  const int nq = 8, KC = 4096, k = 512;
  Rng r(21);
  struct Case { const char* name; int total, kind; } cases[] = {
      {"random, 5 chunks", 5 * KC, 0}, {"random, 3.5 chunks", 3 * KC + 2000, 0}, {"-inf holes", 4 * KC, 1},
      {"increasing scores (late chunks win: survivor overflow -> full path)", 4 * KC, 2},
      {"ties at few levels (threshold ties inside chunk 0 only)", 4 * KC, 3}};
  Buffer S = d.alloc((size_t)nq * KC * 4), Rk[2], Ri[2], Rc[2];
  for (int v = 0; v < 2; v++) { Rk[v] = d.alloc(nq * k * 4); Ri[v] = d.alloc(nq * k * 4); Rc[v] = d.alloc(nq * 4); }
  for (auto& c : cases) {
    std::vector<std::vector<float>> all(nq, std::vector<float>(c.total));
    for (int q = 0; q < nq; q++)
      for (int i = 0; i < c.total; i++) {
        float x = r.normal(1);
        if (c.kind == 1 && (i % 3)) x = -INFINITY;
        if (c.kind == 2) x = (float)i * 1e-3f + 0.01f * x;
        if (c.kind == 3) x = i < KC ? (float)(int)r.uni(0, 4) + 100 : (float)(int)r.uni(0, 104);  // ties; later <= 103
        all[q][i] = x;
      }
    for (int v = 0; v < 2; v++) {
      for (int kc0 = 0; kc0 < c.total; kc0 += KC) {
        const int cnt = std::min(KC, c.total - kc0);
        for (int q = 0; q < nq; q++) memcpy((float*)S.cpu() + (size_t)q * KC, all[q].data() + kc0, cnt * 4);
        Enc e;
        MergeArgs a{nq, k, cnt, KC, kc0, kc0 == 0};
        e.set_pipeline(v ? "attn_topk_merge_f" : "attn_topk_merge");
        e.set_buffer(0, S); e.set_buffer(1, Rk[v]); e.set_buffer(2, Ri[v]); e.set_buffer(3, Rc[v]); e.set_value(4, a);
        e.dispatch_groups(nq, 1, 1, 1024);
        e.wait();
      }
    }
    bool ok = true;
    for (int q = 0; q < nq; q++) {
      auto ref = R::topk_sorted(all[q], k);
      for (int v = 0; v < 2; v++) {
        int m = ((int*)Rc[v].cpu())[q];
        std::vector<int> g((int*)Ri[v].cpu() + q * k, (int*)Ri[v].cpu() + q * k + m);
        std::sort(g.begin(), g.end());
        if (g != ref) { ok = false; if (q == 0) printf("    q0 variant %d: %zu vs ref %zu\n", v, g.size(), ref.size()); }
      }
    }
    check(ok, c.name);
  }
}

static void fill_state(A::State& st, const std::vector<int>& layers, int S0, uint64_t seed) {
  Rng r(seed);
  for (int L : layers) {
    uint16_t* ring = (uint16_t*)st.win[L].cpu();
    for (int i = 0; i < 128 * 512; i++) ring[i] = R::f32_to_f16(r.normal(1));
  }
  const int srcs[4] = {2, 8, 14, 20};
  for (int i = 0; i < 4; i++) {
    int rr = A::ratio_of(srcs[i]), rows = S0 / rr;
    uint16_t* c = (uint16_t*)st.ckv[i].cpu();
    uint16_t* k = (uint16_t*)st.ikey[i].cpu();
    for (size_t j = 0; j < (size_t)rows * 512; j++) c[j] = R::f32_to_f16(r.normal(1));
    for (size_t j = 0; j < (size_t)rows * 128; j++) k[j] = R::f32_to_f16(r.normal(1));
    float* cs = (float*)st.cst[i].cpu();
    for (int x = 0; x < 2 * rr * 512; x++) cs[x] = r.normal(1);
  }
}

struct Out { std::map<int, std::vector<float>> out; std::map<int, std::vector<int>> topk; };

static Out run(const A::Options& o, const std::vector<int>& layers, int S0, int n, uint64_t seed) {
  A::State st; A::init(st, o);
  if (S0 > 0) fill_state(st, layers, S0, seed);
  Out r;
  for (int L : layers) {
    GpuLayer g(L, n);
    g.fill(S0, n);
    { Enc e; A::encode_layer(e, st, g.io(), S0, n); e.wait(); }
    r.out[L].assign((float*)g.out.cpu(), (float*)g.out.cpu() + (size_t)n * 64 * 512);
    if (A::is_index_source(L)) r.topk[L].assign((int*)st.topk.cpu(), (int*)st.topk.cpu() + (size_t)n * 512);
  }
  return r;
}

static void compare(const Out& a, const Out& b, const char* what, double tol) {
  double worst = 0;
  bool lists = true;
  size_t diff_lists = 0;
  for (auto& [L, v] : a.out) {
    ErrStat es; es.add(v.data(), b.out.at(L).data(), v.size());
    worst = std::max(worst, es.rel_rms());
  }
  for (auto& [L, v] : a.topk) {
    const auto& w = b.topk.at(L);
    for (size_t q = 0; q < v.size() / 512; q++)
      if (!std::equal(v.begin() + q * 512, v.begin() + q * 512 + 512, w.begin() + q * 512)) { lists = false; diff_lists++; }
  }
  char msg[200];
  snprintf(msg, sizeof msg, "%s: worst out rel_rms %.2e, top-k lists %s (%zu differ)", what, worst, lists ? "identical" : "DIFFER", diff_lists);
  check(lists && worst <= tol, msg);
}

static A::Options all_off(A::Options o) {
  o.pf_score_reg = false; o.pf_fold_softmax = false; o.pf_merge_filter = false;
  return o;
}

int main() {
  test_merge_filter();
  const std::vector<int> layers = {0, 2, 20, 24};
  {
    printf("at depth: S0=40001, chunk of 160 (3 key chunks, candidate masking + union active)\n");
    A::Options o; o.max_pos = 65536; o.max_chunk = 160; o.pf_qblock = 64;  // 3 query blocks
    Out on = run(o, layers, 40001, 160, 9);
    Out off = run(all_off(o), layers, 40001, 160, 9);
    compare(on, off, "all fast paths vs all off", 5e-4);
    A::Options v = o; v.pf_score_reg = false; compare(on, run(v, layers, 40001, 160, 9), "pf_score_reg off", 5e-4);
    v = o; v.pf_fold_softmax = false; compare(on, run(v, layers, 40001, 160, 9), "pf_fold_softmax off", 5e-4);
    v = o; v.pf_merge_filter = false; compare(on, run(v, layers, 40001, 160, 9), "pf_merge_filter off", 0.0);
  }
  {
    printf("shallow: from position 0, chunk of 300 (invalid rows, -1 padding, skipped tiles)\n");
    A::Options o; o.max_pos = 4096; o.max_chunk = 300; o.key_chunk = 256;
    compare(run(o, layers, 0, 300, 0), run(all_off(o), layers, 0, 300, 0), "all fast paths vs all off", 5e-4);
  }
  printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
