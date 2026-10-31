// bench_attn_cache_probe: read bandwidth vs working-set size (repeated passes over W bytes) -> on-chip cache capacity
#include "core/metal.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
int main() {
  auto& d = ms::Device::get();
  ms::Buffer X = d.alloc(256u << 20), O = d.alloc(64);
  memset(X.cpu(), 1, X.size);
  for (int mb : {2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256}) {
    size_t nb = (size_t)mb << 20; uint32_t n4 = (uint32_t)(nb / 16);
    const int passes = std::max(4, 1024 / mb);
    double best = 1e9;
    for (int it = 0; it < 4; it++) {
      ms::Enc e;
      for (int p = 0; p < passes; p++) {
        e.set_pipeline("attn_probe_bw"); e.set_buffer(0, X); e.set_buffer(1, O); e.set_value(2, n4);
        e.dispatch_threads(20 * 1024 * 8, 1, 1, 1024); e.barrier();
      }
      e.wait(); best = std::min(best, e.gpu_ms());
    }
    printf("W = %4d MB: %7.1f GB/s\n", mb, nb * (double)passes / best / 1e6);
  }
}
