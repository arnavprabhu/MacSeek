// bench_attn_mmprobe: matrix-unit throughput for the indexer GEMM shape (K=128, half in, f32 acc) via
// simdgroup_matrix vs Metal 4 tensor-op matmul2d, plus a streaming-read bandwidth probe.
#include "core/metal.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
int main() {
  auto& d = ms::Device::get();
  const uint32_t M = 32768, N = 16384;
  auto A = d.alloc((size_t)M * 128 * 2), B = d.alloc((size_t)N * 128 * 2), O = d.alloc(64u << 20);
  uint16_t* pa = (uint16_t*)A.cpu(); uint16_t* pb = (uint16_t*)B.cpu();
  for (size_t i = 0; i < (size_t)M * 128; i++) pa[i] = 0x3c00 ^ (i & 0x8000);
  for (size_t i = 0; i < (size_t)N * 128; i++) pb[i] = 0x3800;
  double flop = 2.0 * M * N * 128;
  auto run = [&](const char* name, uint32_t tm, uint32_t tn, bool mpp) {
    double best = 1e9;
    for (int it = 0; it < 5; it++) {
      ms::Enc e; e.set_pipeline(name); e.set_buffer(0, A); e.set_buffer(1, B); e.set_buffer(2, O);
      e.set_value(3, M); e.set_value(4, N);
      e.dispatch_groups(N / tn, M / tm, 1, 128); e.wait(); best = std::min(best, e.gpu_ms());
    }
    printf("%-20s %8.2f ms  %6.2f TFLOP/s\n", name, best, flop / best / 1e9);
    (void)mpp;
  };
  run("attn_probe_sg", 64, 64, false);
  run("attn_probe_mpp64", 64, 64, true);
  run("attn_probe_mpp128", 128, 64, true);
  // bandwidth: 1 GiB read
  size_t nb = 1ull << 30; auto X = d.alloc(nb); memset(X.cpu(), 1, nb);
  uint32_t n4 = nb / 16; double best = 1e9;
  for (int it = 0; it < 5; it++) {
    ms::Enc e; e.set_pipeline("attn_probe_bw"); e.set_buffer(0, X); e.set_buffer(1, O); e.set_value(2, n4);
    e.dispatch_threads(20 * 1024 * 8, 1, 1, 1024); e.wait(); best = std::min(best, e.gpu_ms());
  }
  printf("stream read 1 GiB: %.2f ms = %.1f GB/s\n", best, nb / best / 1e6);
}
