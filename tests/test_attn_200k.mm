// 200K-context decode test: random caches filled to depth, one decode token per layer of the stack
// {0, 2, 3, 20, 21, 24} at positions 200000 and 262143 (max), GPU vs CPU reference. Also ACTQ=0 path at shallow depth.
#include "attn_test_util.hpp"
#include <set>

using namespace at;

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
  A::Options o; o.max_pos = 262144; o.max_chunk = 4;
  A::State st; A::init(st, o);
  std::vector<GpuLayer> gl;
  for (int L : layers) gl.emplace_back(L, 4);
  for (int pos : {200000, 262143}) {
    printf("decode at pos %d (layer 2/3: %d compressed rows, layer 20/21/24: %d)\n", pos, (pos + 1) / 2, pos + 1);
    fill_random_state(st, layers, pos, 1234 + pos);
    RefStack rs(o.max_pos, true);
    rs.load_gpu(st, layers, pos);
    std::vector<int> shared;
    std::vector<float> ref(64 * 512);
    for (auto& g : gl) {
      g.fill(pos, 1);
      Enc e; A::encode_layer(e, st, g.io(), pos, 1); e.wait();
      std::vector<int> tk;
      rs.token(g.L, pos, ref.data(), A::is_index_source(g.L) ? &tk : nullptr, shared);
      ErrStat es; es.add((const float*)g.out.cpu(), ref.data(), 64 * 512);
      char msg[200];
      if (A::is_index_source(g.L)) {
        std::vector<int> gt((const int*)st.topk.cpu(), (const int*)st.topk.cpu() + 512);
        snprintf(msg, sizeof msg, "layer %d: top-512 identical to reference", g.L);
        check(gt == tk, msg);
      }
      snprintf(msg, sizeof msg, "layer %d: out rel_rms %.2e rel_max %.2e", g.L, es.rel_rms(), es.rel_max());
      check(es.rel_rms() < 2e-3 && es.rel_max() < 2e-2, msg);
    }
  }
  {
    printf("ACTQ=0 (no quant simulation): 300 tokens from 0, prefill 250 + 50 decode vs reference\n");
    A::Options o2; o2.max_pos = 4096; o2.max_chunk = 256; o2.actq = false;
    A::State s2; A::init(s2, o2);
    std::vector<GpuLayer> g2;
    for (int L : layers) g2.emplace_back(L, 256);
    RefStack rs(o2.max_pos, false);
    ErrStat es;
    int pos = 0;
    std::vector<int> plan = {250};
    for (int i = 0; i < 50; i++) plan.push_back(1);
    std::vector<float> ref(64 * 512);
    bool topk_ok = true;
    for (int n : plan) {
      // reference tokens
      std::map<int, std::vector<float>> want;  // (layer*1000 + i) for last token of chunk only
      for (int p = pos; p < pos + n; p++) {
        std::vector<int> shared;
        for (int L : layers) {
          if (p == pos + n - 1) { rs.token(L, p, ref.data(), nullptr, shared); want[L] = ref; }
          else rs.token_state_only(L, p);
        }
      }
      for (auto& g : g2) {
        g.fill(pos, n);
        Enc e; A::encode_layer(e, s2, g.io(), pos, n); e.wait();
        es.add((const float*)g.out.cpu() + (size_t)(n - 1) * 64 * 512, want[g.L].data(), 64 * 512);
      }
      pos += n;
    }
    char msg[160];
    snprintf(msg, sizeof msg, "ACTQ=0 last-token outputs: rel_rms %.2e rel_max %.2e", es.rel_rms(), es.rel_max());
    check(es.rel_rms() < 2e-3 && es.rel_max() < 2e-2 && topk_ok, msg);
  }
  printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
