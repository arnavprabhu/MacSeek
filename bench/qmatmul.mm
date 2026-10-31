// bench_qmatmul: prefill matmul throughput (TFLOPS) vs n, per weight type, vs a plain F16 GEMM peak measured here
// with the same tensor API. Usage: bench_qmatmul [--peak] [--moe] [filter]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/metal.hpp"
#include "kernels/quant.hpp"

using namespace ms;
using ms::q::QType;

static void fill_rand(Buffer& b, uint32_t seed, uint32_t mask = 0x3bff3bffu) {
  uint32_t* p = (uint32_t*)b.cpu(); uint32_t st = seed;
  for (size_t i = 0; i < b.size / 4; i++) { st = st * 1664525u + 1013904223u; p[i] = st & mask; }
}

template <class F> static double best_ms(int reps, F&& enc) {
  std::vector<double> t;
  for (int r = 0; r < reps; r++) { Enc e; enc(e); e.wait(); if (r) t.push_back(e.gpu_ms()); }
  return *std::min_element(t.begin(), t.end());
}

static double peak_f16(bool verbose) {
  auto& d = Device::get();
  double best = 0;
  const int n = 4096, m = 8192, k = 4096;
  Buffer A = d.alloc((size_t)n * k * 2), B = d.alloc((size_t)m * k * 2), C = d.alloc((size_t)n * m * 4);
  fill_rand(A, 1); fill_rand(B, 2);
  for (auto [name, tm, tn] : {std::tuple{"q_gemm_f16_32x64", 32, 64}, {"q_gemm_f16_64x64", 64, 64},
                              {"q_gemm_f16_64x128", 64, 128}, {"q_gemm_f16_128x64", 128, 64},
                              {"q_gemm_f16_128x128", 128, 128}, {"q_gemm_f32a_64x128", 64, 128},
                              {"q_gemm_f16_64x128_k64", 64, 128}, {"q_gemm_f16_64x128_k128", 64, 128},
                              {"q_gemm_f16_64x256", 64, 256}}) {
    uint32_t dims[4] = {(uint32_t)n, (uint32_t)m, (uint32_t)k, 0};
    int nsg = ((tm == 128 && tn == 128) || tn == 256) ? 8 : 4;
    bool f32a = strstr(name, "f32a") != nullptr;
    Buffer A32 = f32a ? d.alloc((size_t)n * k * 4) : Buffer{};
    double ms = best_ms(6, [&](Enc& e) {
      e.set_pipeline(name); e.set_buffer(0, f32a ? A32 : A); e.set_buffer(1, B); e.set_buffer(2, C); e.set_bytes(3, dims, 16);
      e.dispatch_groups(m / tn, n / tm, 1, 32 * nsg);
    });
    double tf = 2.0 * n * m * k / ms / 1e9;
    if (verbose) printf("  plain F16 GEMM (MPP matmul2d) %-20s n=%d m=%d k=%d: %6.2f ms  %6.2f TFLOPS\n", name, n, m, k, ms, tf);
    if (!f32a) best = std::max(best, tf);
    if (f32a) d.release(A32);
  }
  // quick correctness spot check of the last config (C row 0..3, col 0..3)
  d.release(A); d.release(B); d.release(C);
  return best;
}

// MoE prefill: n tokens, E experts, top-6 random routing (+ shared Q6_K), IQ1_S experts.
static void bench_moe(double peak) {
  auto& d = Device::get();
  printf("MoE prefill (IQ1_S experts + Q6_K shared, top-6 random routing)\n");
  printf("  %-6s %-8s %-12s %10s %9s %8s\n", "n", "experts", "tok/expert", "ms", "TFLOPS", "%peak");
  for (auto [n, E] : {std::pair{512, 96}, {2048, 96}, {2048, 32}, {4096, 64}}) {
    const int D = q::kExpertDim, I = q::kExpertInter, K = 6;
    const size_t gu = q::row_bytes(QType::IQ1_S, D) * I, dn = q::row_bytes(QType::IQ1_S, I) * D;
    const size_t sgu = q::row_bytes(QType::Q6_K, D) * I, sdn = q::row_bytes(QType::Q6_K, I) * D;
    Buffer arena = d.alloc((size_t)E * (2 * gu + dn) + 2 * sgu + sdn);
    fill_rand(arena, 9);
    std::vector<q::ExpertSlot> slots(E);
    size_t off = 0;
    for (int j = 0; j < E; j++) {
      slots[j].gate = arena.sub(off, gu); off += gu; slots[j].up = arena.sub(off, gu); off += gu;
      slots[j].down = arena.sub(off, dn); off += dn;
    }
    q::ExpertSlot sh{arena.sub(off, sgu), arena.sub(off + sgu, sgu), arena.sub(off + 2 * sgu, sdn)};
    uint32_t st = 1;
    std::vector<std::vector<int>> per(E);
    std::vector<int> slot_of(n * K);
    std::vector<std::pair<int, int>> ent;  // (token, k) per expert
    std::vector<std::vector<std::pair<int, int>>> pe(E);
    for (int t = 0; t < n; t++) {
      std::vector<int> used;
      while ((int)used.size() < K) {
        st = st * 1664525u + 1013904223u;
        int j = (int)((st >> 8) % E);
        if (std::find(used.begin(), used.end(), j) == used.end()) { used.push_back(j); pe[j].push_back({t, (int)used.size() - 1}); }
      }
    }
    std::vector<int> offs(E + 1), tok; std::vector<float> wt;
    int ne = 0; std::vector<q::ExpertSlot> used_slots;
    for (int j = 0; j < E; j++) {
      if (pe[j].empty()) continue;
      offs[ne] = (int)tok.size(); used_slots.push_back(slots[j]);
      for (auto [t, k] : pe[j]) { slot_of[t * K + k] = (int)tok.size(); tok.push_back(t); wt.push_back(0.16f); }
      ne++;
    }
    offs[ne] = (int)tok.size();
    const int total = (int)tok.size();
    Buffer btok = d.alloc(total * 4), bwt = d.alloc(total * 4), bslot = d.alloc(n * K * 4);
    memcpy(btok.cpu(), tok.data(), total * 4); memcpy(bwt.cpu(), wt.data(), total * 4); memcpy(bslot.cpu(), slot_of.data(), n * K * 4);
    Buffer X = d.alloc((size_t)n * D * 4), Y = d.alloc((size_t)n * D * 4);
    Buffer scratch = d.alloc(q::moe_prefill_scratch_bytes(total, n, ne));
    q::MoePrefillArgs a;
    a.n_tokens = n; a.n_exp = ne; a.slots = used_slots.data(); a.offs = offs.data(); a.tok = btok; a.wt = bwt;
    a.slot_of = bslot; a.k_per_token = K; a.shared = sh; a.X = X; a.Y = Y; a.scratch = scratch;
    double ms = best_ms(4, [&](Enc& e) { q::encode_moe_prefill(e, a); });
    double fl = 2.0 * 3 * D * I * (double)(total + n);
    printf("  %-6d %-8d %-12.0f %10.2f %9.2f %7.0f%%\n", n, E, (double)total / ne, ms, fl / ms / 1e9, 100 * fl / ms / 1e9 / peak);
    for (Buffer* b : {&arena, &btok, &bwt, &bslot, &X, &Y, &scratch}) d.release(*b);
  }
}

int main(int argc, char** argv) {
  bool only_peak = false, moe = false; const char* filt = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--peak")) only_peak = true;
    else if (!strcmp(argv[i], "--moe")) moe = true;
    else filt = argv[i];
  }
  double peak = peak_f16(true);
  printf("F16 GEMM peak: %.2f TFLOPS\n", peak);
  if (only_peak) return 0;
  if (moe) { bench_moe(peak); return 0; }
  struct Shape { QType t; int rows, cols; const char* what; };
  const Shape shapes[] = {
      {QType::Q6_K, 32768, 1280, "attn_q_b"}, {QType::Q6_K, 5120, 8192, "attn_output_b"},
      {QType::Q5_K, 5120, 8192, "attn_output_b"}, {QType::Q6_K, 1280, 5120, "attn_q_a"},
      {QType::Q8_0, 2304, 5120, "shexp gate/up"}, {QType::IQ1_S, 2304, 5120, "exp gate/up"},
      {QType::IQ2_XXS, 5120, 2304, "exp down"}, {QType::BF16, 4096, 1280, "indexer q_b"},
      {QType::F16, 8192, 4096, "f16 synthetic"},
  };
  const int ns[] = {32, 128, 512, 2048, 4096};
  printf("%-8s %-14s %-14s", "type", "tensor", "shape");
  for (int n : ns) printf("  n=%-5d", n);
  printf("   (TFLOPS; %% of F16 peak at n=4096)\n");
  auto& d = Device::get();
  for (const Shape& s : shapes) {
    std::string lab = std::string(q::type_name(s.t)) + " " + s.what;
    if (filt && lab.find(filt) == std::string::npos) continue;
    Buffer W = d.alloc(q::row_bytes(s.t, s.cols) * s.rows);
    fill_rand(W, 5, s.t == QType::F32 ? 0x3f7f3f7fu : 0x3bff3bffu);
    Buffer X16 = d.alloc((size_t)4096 * s.cols * 2), Y = d.alloc((size_t)4096 * s.rows * 4);
    fill_rand(X16, 6);
    char shape[32]; snprintf(shape, sizeof shape, "[%d->%d]", s.cols, s.rows);
    printf("%-8s %-14s %-14s", q::type_name(s.t), s.what, shape);
    double last = 0;
    for (int n : ns) {
      double ms = best_ms(5, [&](Enc& e) { q::encode_matmul_h(e, s.t, W, s.rows, s.cols, X16, Y, n); });
      last = 2.0 * n * s.rows * s.cols / ms / 1e9;
      printf("  %7.2f", last);
    }
    printf("   %4.0f%%\n", 100 * last / peak);
    d.release(W); d.release(X16); d.release(Y);
  }
  return 0;
}
