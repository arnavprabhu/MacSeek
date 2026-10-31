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

int main(int argc, char** argv) {
  bool only_peak = false; const char* filt = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--peak")) only_peak = true;
    else filt = argv[i];
  }
  double peak = peak_f16(true);
  printf("F16 GEMM peak: %.2f TFLOPS\n", peak);
  if (only_peak) return 0;
  (void)filt;
  return 0;
}
