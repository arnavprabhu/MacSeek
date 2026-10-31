// bench_attn_cache_probe: read bandwidth vs working-set size (repeated passes over W bytes) -> on-chip cache capacity
#include "core/metal.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
int main() {
  auto& d = ms::Device::get();
  ms::Buffer X = d.alloc(256u << 20), O = d.alloc(64);
  memset(X.cpu(), 1, X.size);
  for (int mb : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256}) {
    size_t nb = (size_t)mb << 20; uint32_t n4 = (uint32_t)(nb / 16);
    const uint32_t reps = std::max(2, 2048 / mb);
    double best = 1e9;
    for (int it = 0; it < 3; it++) {
      ms::Enc e;
      struct { uint32_t n, r; } a{n4, reps};
      e.set_pipeline("attn_probe_bw_rep"); e.set_buffer(0, X); e.set_buffer(1, O); e.set_value(2, a);
      e.dispatch_threads(20 * 1024 * 4, 1, 1, 1024);
      e.wait(); best = std::min(best, e.gpu_ms());
    }
    printf("W = %4d MB: %7.1f GB/s\n", mb, nb * (double)reps / best / 1e6);
  }
}
