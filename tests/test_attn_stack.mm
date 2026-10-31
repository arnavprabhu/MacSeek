// Algorithm tests over a stack of layers (0 ratio-0, 2 ratio-2 kv+index source, 3 reuse, 20 ratio-1 kv+index+candidate
// source, 21 reuse, 24 candidate-restricted index source):
//   1. from position 0: GPU (prefill chunks + decode steps) vs CPU token-by-token reference
//   2. chunked prefill + decode == one-shot prefill (GPU vs GPU)
//   3. at depth (random caches, 40K positions, multi key-chunk indexer, candidate masking active): prefill chunk vs
//      reference, and chunked == one-shot
#include "attn_test_util.hpp"
#include <set>

using namespace at;

struct Run {
  std::map<int, std::map<int, std::vector<float>>> out;
  std::map<int, std::map<int, std::vector<int>>> topk;
};

static std::vector<std::vector<uint8_t>> snapshot(const A::State& st) {
  std::vector<std::vector<uint8_t>> s;
  auto add = [&](const Buffer& b) { s.emplace_back((uint8_t*)b.cpu(), (uint8_t*)b.cpu() + b.size); };
  for (int L = 0; L < A::kLayers; L++) add(st.win[L]);
  for (int i = 0; i < 4; i++) { add(st.ckv[i]); add(st.ikey[i]); add(st.cst[i]); }
  return s;
}
static void restore(A::State& st, const std::vector<std::vector<uint8_t>>& s) {
  size_t k = 0;
  auto put = [&](Buffer& b) { memcpy(b.cpu(), s[k++].data(), b.size); };
  for (int L = 0; L < A::kLayers; L++) put(st.win[L]);
  for (int i = 0; i < 4; i++) { put(st.ckv[i]); put(st.ikey[i]); put(st.cst[i]); }
}

static Run run_gpu(A::State& st, std::vector<GpuLayer>& gl, int S0, const std::vector<int>& plan, const std::set<int>& keep) {
  Run r;
  int pos = S0;
  for (int n : plan) {
    for (auto& g : gl) {
      g.fill(pos, n);
      { Enc e; A::encode_layer(e, st, g.io(), pos, n); e.wait(); }
      for (int i = 0; i < n; i++) {
        if (!keep.count(pos + i)) continue;
        const float* o = (const float*)g.out.cpu() + (size_t)i * 64 * 512;
        r.out[g.L][pos + i].assign(o, o + 64 * 512);
        if (A::is_index_source(g.L)) {
          const int* t = (const int*)st.topk.cpu() + (size_t)i * 512;
          std::vector<int> v;
          for (int j = 0; j < 512; j++) if (t[j] >= 0) v.push_back(t[j]);
          r.topk[g.L][pos + i] = v;
        }
      }
    }
    pos += n;
  }
  return r;
}

static Run run_ref(RefStack& rs, const std::vector<int>& layers, int S0, int N, const std::set<int>& keep) {
  Run r;
  std::vector<float> o(64 * 512);
  for (int p = S0; p < S0 + N; p++) {
    std::vector<int> shared;
    for (int L : layers) {
      bool full = keep.count(p) > 0;
      std::vector<int> tk;
      if (full) {
        rs.token(L, p, o.data(), A::is_index_source(L) ? &tk : nullptr, shared);
        r.out[L][p] = o;
        if (A::is_index_source(L)) r.topk[L][p] = tk;
      } else {
        rs.token_state_only(L, p);
      }
    }
  }
  return r;
}

static void compare(const Run& a, const Run& b, const char* what, double tol_rms, double tol_max, double min_topk_match) {
  ErrStat es;
  double worst_rms = 0;
  size_t tk_tot = 0, tk_same = 0, tk_lists = 0, tk_exact = 0;
  for (auto& [L, m] : a.out)
    for (auto& [p, v] : m) {
      auto itL = b.out.find(L);
      if (itL == b.out.end() || !itL->second.count(p)) continue;
      ErrStat e1; e1.add(v.data(), itL->second.at(p).data(), v.size());
      es.add(v.data(), itL->second.at(p).data(), v.size());
      worst_rms = std::max(worst_rms, e1.rel_rms());
    }
  for (auto& [L, m] : a.topk)
    for (auto& [p, v] : m) {
      auto& w = b.topk.at(L).at(p);
      std::set<int> sa(v.begin(), v.end());
      size_t same = 0;
      for (int x : w) same += sa.count(x);
      tk_tot += std::max(v.size(), w.size()); tk_same += same; tk_lists++; tk_exact += (v == w);
    }
  double tk_frac = tk_tot ? (double)tk_same / tk_tot : 1.0;
  char msg[256];
  snprintf(msg, sizeof msg, "%s: out rel_rms %.2e (worst %.2e) rel_max %.2e | top-k overlap %.5f, identical lists %zu/%zu",
           what, es.rel_rms(), worst_rms, es.rel_max(), tk_frac, tk_exact, tk_lists);
  check(worst_rms <= tol_rms && es.rel_max() <= tol_max && tk_frac >= min_topk_match && es.n > 0, msg);
}

static void fill_random_state(A::State& st, const std::vector<int>& layers, int S0, uint64_t seed) {
  Rng r(seed);
  for (int L : layers) {
    uint16_t* ring = (uint16_t*)st.win[L].cpu();
    for (int s = 0; s < 128; s++) {
      float row[512];
      for (int i = 0; i < 512; i++) row[i] = r.normal(1);
      R::fp8_sim(row, 512);
      for (int i = 0; i < 512; i++) ring[s * 512 + i] = R::f32_to_f16(row[i]);
    }
  }
  const int srcs[4] = {2, 8, 14, 20};
  for (int i = 0; i < 4; i++) {
    int rr = A::ratio_of(srcs[i]), rows = S0 / rr;
    uint16_t* c = (uint16_t*)st.ckv[i].cpu();
    uint16_t* k = (uint16_t*)st.ikey[i].cpu();
    for (int j = 0; j < rows; j++) {
      float row[512], key[128];
      for (int x = 0; x < 512; x++) row[x] = r.normal(1);
      for (int x = 0; x < 128; x++) key[x] = r.normal(1);
      R::fp4_sim_e4m3(row, 512); R::fp4_sim_e8m0(key, 128);
      for (int x = 0; x < 512; x++) c[(size_t)j * 512 + x] = R::f32_to_f16(row[x]);
      for (int x = 0; x < 128; x++) k[(size_t)j * 128 + x] = R::f32_to_f16(key[x]);
    }
    float* cs = (float*)st.cst[i].cpu();
    for (int x = 0; x < rr * 512; x++) { cs[x] = r.normal(1); cs[rr * 512 + x] = r.normal(2); }
  }
}

int main() {
  const std::vector<int> layers = {0, 2, 3, 20, 21, 24};
  {
    printf("from position 0: N=600 tokens\n");
    A::Options o; o.max_pos = 4096; o.max_chunk = 600; o.key_chunk = 256;  // small key chunk: exercises merging
    A::State st; A::init(st, o);
    std::vector<GpuLayer> gl;
    for (int L : layers) gl.emplace_back(L, 600);
    const int N = 600;
    std::set<int> keep;
    for (int p = 0; p < N; p += 9) keep.insert(p);
    for (int p : {0, 1, 2, 3, 126, 127, 128, 129, 255, 256, 257, 300, 301, 302, 303, 304, 597, 598, 599}) keep.insert(p);
    auto snap = snapshot(st);
    Run one = run_gpu(st, gl, 0, {N}, keep);
    restore(st, snap);
    Run chunked = run_gpu(st, gl, 0, {1, 1, 131, 1, 1, 1, 122, 1, 33, 7, 1, 1, 300}, keep);
    restore(st, snap);
    Run dec = run_gpu(st, gl, 0, std::vector<int>(N, 1), keep);
    restore(st, snap);
    Run pchunks = run_gpu(st, gl, 0, {2, 131, 122, 33, 7, 305}, keep);  // prefill chunks only (no decode steps)
    RefStack rs(o.max_pos, true);
    Run ref = run_ref(rs, layers, 0, N, keep);
    compare(one, ref, "one-shot prefill vs CPU ref", 2e-3, 2e-2, 0.999);
    compare(chunked, ref, "chunked prefill+decode vs CPU ref", 2e-3, 2e-2, 0.999);
    compare(dec, ref, "pure decode vs CPU ref", 2e-3, 2e-2, 0.999);
    compare(pchunks, one, "prefill chunks vs one-shot (GPU, must be bit-exact)", 0.0, 0.0, 1.0);
    compare(chunked, one, "chunked+decode vs one-shot (GPU; decode split-K rounding)", 5e-4, 2e-3, 1.0);
    compare(dec, one, "decode vs one-shot (GPU; decode split-K rounding)", 5e-4, 2e-3, 1.0);
  }
  {
    const int S0 = 40001, N = 96;
    printf("at depth: S0=%d, chunk of %d (key_chunk 16384 -> 3 key chunks; candidate masking active)\n", S0, N);
    A::Options o; o.max_pos = 65536; o.max_chunk = 128;
    A::State st; A::init(st, o);
    fill_random_state(st, layers, S0, 77);
    std::vector<GpuLayer> gl;
    for (int L : layers) gl.emplace_back(L, 128);
    std::set<int> keep;
    for (int p = S0; p < S0 + N; p += 5) keep.insert(p);
    keep.insert(S0 + N - 1);
    auto snap = snapshot(st);
    RefStack rs(o.max_pos, true);
    rs.load_gpu(st, layers, S0);
    Run one = run_gpu(st, gl, S0, {N}, keep);
    restore(st, snap);
    Run chunked = run_gpu(st, gl, S0, {33, 1, 62}, keep);
    Run ref = run_ref(rs, layers, S0, N, keep);
    compare(one, ref, "depth one-shot vs CPU ref", 2e-3, 2e-2, 0.999);
    compare(chunked, ref, "depth chunked vs CPU ref", 2e-3, 2e-2, 0.999);
    compare(chunked, one, "depth chunked vs one-shot (GPU)", 5e-4, 2e-3, 1.0);
  }
  printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
