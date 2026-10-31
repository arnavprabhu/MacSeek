// bench_attn200k: attention-side cost of all 40 layers at 200K context.
//   decode: ms/token (kv write + compressor, indexer full scans for 2/8/14/20, candidate-restricted 24..36, top-k,
//           sparse attention x40) with a byte model and % of measured streaming bandwidth
//   prefill: one 2048-token chunk at depth 200K, per-component ms and achieved indexer TFLOP/s
// Synthetic state (random caches); never loads the model.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/env.hpp"
#include "core/metal.hpp"
#include "kernels/attn.hpp"

using namespace ms;
namespace A = ms::attn;

static uint64_t rs = 88172645463325252ull;
static inline uint32_t xr() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static uint16_t rh() {  // random half in about [-2, 2]
  _Float16 h = (_Float16)(((int)(xr() & 0xFFFF) - 32768) / 16384.0f);
  uint16_t u; memcpy(&u, &h, 2); return u;
}
static void fill_f32(Buffer b, float sd) { float* p = (float*)b.cpu(); for (size_t i = 0; i < b.size / 4; i++) p[i] = sd * (((int)(xr() & 0xFFFF) - 32768) / 32768.0f); }

static double measure_bw(Device& d) {
  size_t nb = 1ull << 30; Buffer X = d.alloc(nb), O = d.alloc(64);
  memset(X.cpu(), 1, nb);
  uint32_t n4 = (uint32_t)(nb / 16); double best = 1e9;
  for (int it = 0; it < 6; it++) {
    Enc e; e.set_pipeline("attn_probe_bw"); e.set_buffer(0, X); e.set_buffer(1, O); e.set_value(2, n4);
    e.dispatch_threads(20 * 1024 * 8, 1, 1, 1024); e.wait(); best = std::min(best, e.gpu_ms());
  }
  d.release(X);
  return nb / best / 1e6;  // GB/s
}

// Sustained timing: the work is repeated `reps` times inside ONE command buffer (short command buffers run at low
// GPU clocks — DVFS — and are not representative of a busy decode loop); min over `iters` buffers, per rep.
static int g_reps = 20;
template <class F> static double time_med(int iters, F&& f) {
  std::vector<double> t;
  for (int i = 0; i < iters; i++) { Enc e; for (int r = 0; r < g_reps; r++) f(e); e.wait(); t.push_back(e.gpu_ms() / g_reps); }
  std::sort(t.begin(), t.end());
  return t[0];  // min: least interference from other GPU work (a model process may be running)
}

int main(int argc, char** argv) {
  auto& d = Device::get();
  const int depth = (int)env_int("BENCH_DEPTH", 200000);
  const int nchunk = (int)env_int("BENCH_CHUNK", 2048);
  const bool do_prefill = env_int("BENCH_PREFILL", 1) != 0;
  g_reps = (int)env_int("BENCH_REPS", 20);
  double bw = measure_bw(d);
  printf("streaming read bandwidth (attn_probe_bw, 1 GiB): %.1f GB/s\n", bw);

  A::Options o = A::Options::from_env();
  o.max_chunk = nchunk;
  A::State st; A::init(st, o);
  // random caches to depth
  for (int L = 0; L < A::kLayers; L++) { uint16_t* p = (uint16_t*)st.win[L].cpu(); for (size_t i = 0; i < st.win[L].size / 2; i++) p[i] = rh(); }
  for (int i = 0; i < 4; i++) {
    uint16_t* c = (uint16_t*)st.ckv[i].cpu(); uint16_t* k = (uint16_t*)st.ikey[i].cpu();
    for (size_t j = 0; j < st.ckv[i].size / 2; j++) c[j] = rh();
    for (size_t j = 0; j < st.ikey[i].size / 2; j++) k[j] = rh();
  }
  const int N = std::max(nchunk, 1);
  Buffer q = d.alloc((size_t)N * 64 * 512 * 4), kv = d.alloc((size_t)N * 512 * 4), ckv = d.alloc((size_t)N * 512 * 4),
         cg = d.alloc((size_t)N * 512 * 4), iq = d.alloc((size_t)N * 32 * 128 * 4), iw = d.alloc((size_t)N * 32 * 4),
         out = d.alloc((size_t)N * 64 * 512 * 4), w512 = d.alloc(512 * 4), w128 = d.alloc(128 * 4), sinks = d.alloc(64 * 4),
         wk = d.alloc(128 * 512 * 2);
  fill_f32(q, 0.2f); fill_f32(kv, 1); fill_f32(ckv, 1); fill_f32(cg, 2); fill_f32(iq, 1); fill_f32(iw, 1); fill_f32(sinks, 1);
  for (int i = 0; i < 512; i++) ((float*)w512.cpu())[i] = 1;
  for (int i = 0; i < 128; i++) ((float*)w128.cpu())[i] = 1;
  { uint16_t* p = (uint16_t*)wk.cpu(); for (int i = 0; i < 128 * 512; i++) p[i] = 0x3c00 ^ (xr() & 0x8000) ? 0x3c00 : 0xbc00; }
  auto io_of = [&](int L) {
    A::LayerIO x; x.layer = L; x.q = q; x.kv = kv; x.kv_norm = w512; x.sinks = sinks; x.out = out; x.c_kv = ckv; x.c_gate = cg;
    x.c_norm = w512; x.idx_wk = wk; x.idx_knorm = w128; x.idx_q = iq; x.idx_w = iw; return x;
  };

  // ---------------- decode ----------------
  const int pos = depth;
  printf("\n== decode, 1 token at position %d (all 40 layers), sustained (%d reps per command buffer) ==\n", pos, g_reps);
  auto all = [&](Enc& e) { for (int L = 0; L < A::kLayers; L++) A::encode_layer(e, st, io_of(L), pos, 1); };
  time_med(3, all);
  double t_all = time_med(9, all);
  double t_kv = time_med(9, [&](Enc& e) {
    for (int L = 0; L < A::kLayers; L++) {
      A::encode_kv_window(e, st, L, kv, w512, pos, 1);
      if (A::is_kv_source(L)) A::encode_compress(e, st, L, io_of(L), pos | 1, 1);  // completing group (worst case)
      e.barrier();
    }
  });
  double t_idx_full = time_med(9, [&](Enc& e) { for (int L : {2, 8, 14, 20}) A::encode_index_decode(e, st, L, iq, iw, pos); });
  double t_idx_cand = time_med(9, [&](Enc& e) { for (int L : {24, 28, 32, 36}) A::encode_index_decode(e, st, L, iq, iw, pos); });
  double t_l20 = time_med(9, [&](Enc& e) { A::encode_index_decode(e, st, 20, iq, iw, pos); });
  double t_l2 = time_med(9, [&](Enc& e) { A::encode_index_decode(e, st, 2, iq, iw, pos); });
  double t_attn = time_med(9, [&](Enc& e) { for (int L = 0; L < A::kLayers; L++) A::encode_sparse(e, st, L, q, sinks, out, pos, 1); });
  // byte model (unique DRAM bytes)
  const double len2 = (pos + 1) / 2, len1 = pos + 1;
  const double b_idx_full = 3 * len2 * 256 + len1 * 256 + (3 * len2 + len1) * 4 * 2 /*scores w + r*/ + len1 / 8 * 4 * 2;
  const double b_idx_cand = 4 * (2048 * 8 * 256 + 2048 * 8 * 4 * 2);
  const double b_attn = 2 * 128 * 1024.0 + 38 * 640 * 1024.0 + 40 * (128 + 128) * 1024.0 /*q in, out*/ +
                        40 * (10 * 128 * 1024.0 * 2) /*split partials w+r*/;
  const double b_kv = 40 * (2048 + 1024) + 4 * (128 * 1024 + 4 * 2048);
  const double b_tot = b_idx_full + b_idx_cand + b_attn + b_kv;
  printf("%-44s %8s %10s %8s\n", "component", "ms", "MB moved", "% BW");
  auto row = [&](const char* n, double ms, double bytes) { printf("%-44s %8.3f %10.1f %7.1f%%\n", n, ms, bytes / 1e6, bytes / (ms * 1e-3) / (bw * 1e9) * 100); };
  row("kv window + compressor (40 layers)", t_kv, b_kv);
  row("indexer full scans 2,8,14 (100K) + 20 (200K)", t_idx_full, b_idx_full);
  row("  of which layer 2 (100K)", t_l2, len2 * 256 + len2 * 8);
  row("  of which layer 20 (200K + candidate blocks)", t_l20, len1 * 256 + len1 * 8);
  row("indexer candidate-restricted 24,28,32,36", t_idx_cand, b_idx_cand);
  row("sparse attention (40 layers, split-K)", t_attn, b_attn);
  row("TOTAL (one command buffer, all 40 layers)", t_all, b_tot);
  printf("sum of components: %.3f ms\n", t_kv + t_idx_full + t_idx_cand + t_attn);

  if (!do_prefill) return 0;
  // ---------------- prefill ----------------
  const int n = nchunk, start = depth;
  g_reps = 1;  // prefill command buffers are long enough to run at full clock
  printf("\n== prefill, chunk of %d tokens at depth %d (all 40 layers) ==\n", n, start);
  auto pall = [&](Enc& e) { for (int L = 0; L < A::kLayers; L++) A::encode_layer(e, st, io_of(L), start, n); };
  time_med(1, pall);
  double p_all = time_med(3, pall);
  double p_kv = time_med(3, [&](Enc& e) {
    for (int L = 0; L < A::kLayers; L++) {
      A::encode_kv_window(e, st, L, kv, w512, start, n);
      if (A::is_kv_source(L)) A::encode_compress(e, st, L, io_of(L), start, n);
      e.barrier();
      A::encode_window_commit(e, st, L, start, n);
    }
  });
  double p_idx[8]; const int idxL[8] = {2, 8, 14, 20, 24, 28, 32, 36};
  for (int i = 0; i < 8; i++) p_idx[i] = time_med(3, [&](Enc& e) { A::encode_index_prefill(e, st, idxL[i], iq, iw, start, n); });
  double p_attn = time_med(3, [&](Enc& e) { for (int L = 0; L < A::kLayers; L++) A::encode_sparse(e, st, L, q, sinks, out, start, n); });
  printf("%-48s %9s %9s\n", "component", "ms", "TFLOP/s");
  printf("%-48s %9.2f\n", "kv window + compressor + ring commit (40 layers)", p_kv);
  double idx_sum = 0;
  for (int i = 0; i < 8; i++) {
    int r = A::ratio_of(idxL[i]);
    double keys = (double)(start + n) / r;
    double fl = 2.0 * n * 32 * 128 * keys;
    char nm[64]; snprintf(nm, sizeof nm, "indexer layer %d (%s, %.0fK keys)", idxL[i], A::uses_cand(idxL[i]) ? "dense+mask" : "full", keys / 1e3);
    printf("%-48s %9.2f %9.2f\n", nm, p_idx[i], fl / (p_idx[i] * 1e-3) / 1e12);
    idx_sum += p_idx[i];
  }
  double attn_fl = 40.0 * n * 64 * 512 * 2 * 2 * (128 + 512) - 2.0 * n * 64 * 512 * 4 * 512;
  printf("%-48s %9.2f %9.2f\n", "sparse attention (40 layers)", p_attn, attn_fl / (p_attn * 1e-3) / 1e12);
  printf("%-48s %9.2f   (%.1f tok/s attention-side)\n", "TOTAL (one command buffer)", p_all, n / (p_all * 1e-3));
  printf("sum of components: %.2f ms (indexer %.2f)\n", p_kv + idx_sum + p_attn, idx_sum);
  return 0;
}
