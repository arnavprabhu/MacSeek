#include <cstdint>
#include "core/metal.hpp"
#include <cstdio>
int main() {
  auto& d = ms::Device::get();
  uint32_t n = 1 << 20;
  auto a = d.alloc(n * 4), b = d.alloc(n * 4), c = d.alloc(n * 4);
  float *pa = (float*)a.cpu(), *pb = (float*)b.cpu();
  for (uint32_t i = 0; i < n; i++) { pa[i] = i; pb[i] = 2 * i; }
  { ms::Enc e; e.set_pipeline("smoke_add"); e.set_buffer(0, a); e.set_buffer(1, b); e.set_buffer(2, c); e.set_value(3, n);
    e.dispatch_threads(n, 1, 1, 256); e.wait(); }
  float* pc = (float*)c.cpu();
  for (uint32_t i = 0; i < n; i++) if (pc[i] != 3.0f * i) { printf("FAIL at %u\n", i); return 1; }
  printf("smoke OK\n"); return 0;
}
