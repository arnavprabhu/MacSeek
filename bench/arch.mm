// Microbenchmarks for the arch kernels: per-dispatch GPU latency in a dependent chain (N dispatches separated by
// barriers in one command buffer, as in a decode token), median of trials. Usage: bench_arch [N=200] [trials=7]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "core/metal.hpp"
#include "kernels/arch.hpp"

using namespace ms;

static int N = 200, TRIALS = 7;

static Buffer rnd(size_t nfloat, float sd, uint64_t seed) {
  Buffer b = Device::get().alloc(nfloat * 4);
  std::mt19937_64 g(seed);
  std::normal_distribution<float> nd(0.f, sd);
  float* p = (float*)b.cpu();
  for (size_t i = 0; i < nfloat; i++) p[i] = nd(g);
  return b;
}

// returns median us per dispatch-chain element
static double bench(const char* name, const std::function<void(Enc&)>& enc, double bytes = 0, int reps = 0) {
  if (!reps) reps = N;
  for (int w = 0; w < 2; w++) { Enc e; for (int i = 0; i < reps; i++) { enc(e); e.barrier(); } e.wait(); }
  std::vector<double> t;
  for (int k = 0; k < TRIALS; k++) {
    Enc e;
    for (int i = 0; i < reps; i++) { enc(e); e.barrier(); }
    e.wait();
    t.push_back(e.gpu_ms() * 1e3 / reps);
  }
  std::sort(t.begin(), t.end());
  double med = t[t.size() / 2];
  if (bytes > 0) printf("  %-52s %8.2f us  (min %.2f)  %6.1f GB/s\n", name, med, t[0], bytes / (med * 1e3));
  else printf("  %-52s %8.2f us  (min %.2f)\n", name, med, t[0]);
  return med;
}

int main(int argc, char** argv) {
  if (argc > 1) N = atoi(argv[1]);
  if (argc > 2) TRIALS = atoi(argv[2]);
  auto& d = Device::get();
  auto s = arch::Scratch::create(2048, 16);
  printf("arch microbench: %d dependent dispatches per command buffer, median of %d trials (GPU time / dispatch)\n", N, TRIALS);

  // baseline: trivial kernel (dispatch + barrier floor)
  Buffer a = rnd(32, 1, 1), b = rnd(32, 1, 2), c = d.alloc(128);
  bench("baseline: trivial 32-thread kernel", [&](Enc& e) {
    uint32_t n = 32;
    e.set_pipeline("smoke_add"); e.set_buffer(0, a); e.set_buffer(1, b); e.set_buffer(2, c); e.set_value(3, n);
    e.dispatch_threads(32, 1, 1, 32);
  });

  // bandwidth reference: streaming read of distinct buffers (DRAM) of the sizes these kernels touch
  {
    std::vector<Buffer> big;
    for (int i = 0; i < 40; i++) big.push_back(rnd(384 * 5120, 1, 300 + i));
    Buffer o = d.alloc(1 << 20);
    for (uint32_t mb : {2u, 8u}) {
      uint32_t n4 = mb == 2 ? 24 * 20480 / 4 : 384 * 5120 / 4;
      int it = 0;
      char nm[96];
      snprintf(nm, sizeof nm, "reference: stream read %.1f MB (distinct buffers)", n4 * 16 / 1e6);
      bench(nm, [&](Enc& e) {
        e.set_pipeline("arch_bench_read"); e.set_buffer(0, big[it++ % 40]); e.set_buffer(1, o); e.set_value(2, n4);
        e.dispatch_groups((n4 + 2047) / 2048, 1, 1, 256);
      }, n4 * 16.0);
    }
  }

  // ---- hc step (decode) ----
  Buffer h = rnd(20480 * 2048, 0.7f, 3), xs = rnd(5120 * 2048, 1, 4), fn = rnd(24 * 20480, 0.01f, 5);
  std::vector<Buffer> fns;  // 80 distinct copies (one per sublayer): weights stream from DRAM as in a real token
  for (int i = 0; i < 80; i++) fns.push_back(rnd(24 * 20480, 0.01f, 100 + i));
  Buffer sc = rnd(3, 1, 6), ba = rnd(24, 1, 7), nw = rnd(5120, 1, 8), xo = d.alloc(5120 * 2048 * 4);
  Buffer mp = d.alloc(24 * 2048 * 4), mo = d.alloc(24 * 2048 * 4);
  { float* m = (float*)mp.cpu(); for (int i = 0; i < 24 * 2048; i++) m[i] = 0.25f; }
  auto step = [&](int n, bool post, bool mixes) {
    arch::HcStep st;
    st.h_in = h; st.h_out = h; st.x_sub = post ? xs : Buffer{}; st.mix_prev = mp; st.fn = mixes ? fn : Buffer{};
    st.scale = sc; st.base = ba; st.mix_out = mo; st.norm_w = nw; st.x_out = xo; st.n = n;
    return st;
  };
  const double hc_bytes = 24.0 * 20480 * 4 + 20480 * 4 * 2 + 5120 * 4 * 3;
  printf("hc_step decode (hc_post + hc_mixes/Sinkhorn + hc_pre + RMSNorm), 80 per token:\n");
  const char* mn[] = {"fused", "split", "rows"};
  for (int m = 0; m < 3; m++) {
    arch::options().hc_mode = m;
    for (int T : {128, 256, 512, 1024}) {
      if (m == 2 && T != 256) continue;
      arch::options().hc_threads = T;
      char nm[96];
      snprintf(nm, sizeof nm, "%s T=%d (%d groups)%s", mn[m], T, 5120 / T, m == 2 ? " [post_pre + rows]" : "");
      auto st = step(1, true, true);
      int it = 0;
      bench(nm, [&](Enc& e) { st.fn = fns[it++ % 80]; arch::encode_hc_step(e, st, s); }, hc_bytes);
      if (m == 0 && T == 256) {
        snprintf(nm, sizeof nm, "%s T=%d, same fn every dispatch (cache-hot)", mn[m], T);
        bench(nm, [&](Enc& e) { arch::encode_hc_step(e, st, s); }, hc_bytes);
      }
    }
  }
  arch::options().hc_mode = 0;
  arch::options().hc_threads = 512;
  { auto st = step(1, true, false); bench("fused, no mixes (final head)", [&](Enc& e) { arch::encode_hc_step(e, st, s); }); }

  printf("hc_step prefill, per chunk (reported per token):\n");
  for (int n : {256, 2048}) {
    for (int rows = 0; rows < 3; rows++) {
      arch::options().hc_prefill_rows = rows == 1;
      arch::options().hc_prefill_mma8 = rows == 2;
      auto st = step(n, true, true);
      char nm[96];
      snprintf(nm, sizeof nm, "n=%d %s", n, rows == 1 ? "post_pre + rows" : rows == 2 ? "post_pre + mma8" : "post_pre + mma32");
      double us = bench(nm, [&](Enc& e) { arch::encode_hc_step(e, st, s); }, 0, 10);
      printf("  %-52s %8.3f us/token  (%.1f GFLOP/s mixes)\n", "", us / n, 2.0 * 24 * 20480 * n / (us * 1e3));
    }
  }
  arch::options().hc_prefill_rows = false;
  arch::options().hc_prefill_mma8 = false;

  // ---- router ----
  printf("router (40 per token):\n");
  std::vector<Buffer> Ws;  // 40 distinct router matrices (one per layer)
  for (int i = 0; i < 40; i++) Ws.push_back(rnd(384 * 5120, 0.02f, 200 + i));
  Buffer W = Ws[0], bias = rnd(384, 0.1f, 10), x = rnd(5120, 1, 11);
  Buffer ids = d.alloc(2048 * 24), wt = d.alloc(2048 * 24), lg = rnd(384 * 2048, 3, 12);
  for (int G : {24, 48, 96, 192}) {
    arch::options().router_groups = G;
    char nm[96];
    snprintf(nm, sizeof nm, "router_fused F32 matvec 5120->384 + top6, G=%d", G);
    int it = 0;
    bench(nm, [&](Enc& e) { arch::encode_router_fused(e, x, Ws[it++ % 40], bias, ids, wt, s); }, 384.0 * 5120 * 4);
  }
  arch::options().router_groups = 96;
  bench("router_topk n=1 (from logits)", [&](Enc& e) { arch::encode_router_topk(e, lg, bias, ids, wt, 1); });
  {
    double us = bench("router_topk n=2048", [&](Enc& e) { arch::encode_router_topk(e, lg, bias, ids, wt, 2048); }, 0, 20);
    (void)us;
    auto L = arch::RouteLists::create(2048);
    bench("route_lists n=2048 (count + scatter)", [&](Enc& e) { arch::encode_route_lists(e, ids, wt, 2048, L); }, 0, 20);
  }

  // ---- engram / embedding ----
  printf("engram / embedding:\n");
  Buffer kv = rnd(25600 * 64, 0.5f, 13), wg = rnd(20480, 1, 14), hh = rnd(20480 * 64, 1, 15);
  bench("engram_gate n=1", [&](Enc& e) { arch::encode_engram_gate(e, hh, kv, wg, 1, s); });
  bench("engram_gate n=64", [&](Enc& e) { arch::encode_engram_gate(e, hh, kv, wg, 64, s); });
  Buffer raw = d.alloc(4200 * 64);
  { uint8_t* p = (uint8_t*)raw.cpu(); for (int i = 0; i < 4200 * 64; i++) p[i] = (uint8_t)(i * 37); }
  Buffer rows_out = d.alloc(24 * 256 * 4 * 64);
  bench("iq4nl_rows 24 rows (one token)", [&](Enc& e) { arch::encode_iq4nl_rows(e, raw, 144, rows_out, 24); });
  bench("embed_q6k n=1 -> 4 copies", [&](Enc& e) { arch::encode_embed_q6k(e, raw, 4200, hh, 1, 4); });

  // ---- sampling ----
  printf("sampling / logits (V=129280):\n");
  Buffer logits = rnd((size_t)arch::kVocab * 16, 3, 16), oi = d.alloc(16 * 64 * 4), ov = d.alloc(16 * 64 * 4), ol = d.alloc(64);
  bench("argmax (greedy)", [&](Enc& e) { arch::encode_argmax(e, logits, 1, arch::kVocab, oi, ov, s); }, arch::kVocab * 4.0);
  bench("argmax + gumbel (temperature sampling)", [&](Enc& e) { arch::encode_argmax(e, logits, 1, arch::kVocab, oi, ov, s, 0.8f, 7); });
  bench("argmax 6 rows (verify block)", [&](Enc& e) { arch::encode_argmax(e, logits, 6, arch::kVocab, oi, ov, s); });
  bench("topk k=64 + lse", [&](Enc& e) { arch::encode_topk(e, logits, 1, arch::kVocab, 64, 0.8f, ov, oi, ol, s); });
  Buffer lso = d.alloc((size_t)arch::kVocab * 16 * 4);
  bench("log_softmax 1 row", [&](Enc& e) { arch::encode_log_softmax(e, logits, lso, 1, arch::kVocab); }, arch::kVocab * 8.0);
  bench("log_softmax 16 rows", [&](Enc& e) { arch::encode_log_softmax(e, logits, lso, 16, arch::kVocab); }, arch::kVocab * 8.0 * 16, 50);

  printf("rmsnorm:\n");
  bench("rmsnorm 5120 x1", [&](Enc& e) { arch::encode_rmsnorm(e, xs, nw, xo, 1, 5120, 1e-6f); });
  bench("rmsnorm 1280 x1 (q_norm)", [&](Enc& e) { arch::encode_rmsnorm(e, xs, nw, xo, 1, 1280, 1e-6f); });
  bench("rmsnorm 512 x1 (kv_norm)", [&](Enc& e) { arch::encode_rmsnorm(e, xs, nw, xo, 1, 512, 1e-6f); });
  bench("rmsnorm 5120 x2048", [&](Enc& e) { arch::encode_rmsnorm(e, xs, nw, xo, 2048, 5120, 1e-6f); }, 2048 * 5120 * 8.0, 20);
  return 0;
}
