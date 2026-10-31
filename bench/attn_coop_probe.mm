// bench_attn_coop_probe: dumps the cooperative destination tensor layout of matmul2d tiles (4 simdgroups)
#include "core/metal.hpp"
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
  auto& d = ms::Device::get();
  auto A = d.alloc(128 * 128 * 2), B = d.alloc(128 * 128 * 2), o = d.alloc((128 * 256 + 128) * 4);
  for (const char* k : {"attn_probe_coop_64x128", "attn_probe_coop_64x64", "attn_probe_coop_128x64"}) {
    memset(o.cpu(), 0xff, o.size);
    { ms::Enc e; e.set_pipeline(k); e.set_buffer(0, A); e.set_buffer(1, B); e.set_buffer(2, o); e.dispatch_groups(1, 1, 1, 128); e.wait(); }
    int* p = (int*)o.cpu();
    printf("%s capacity %d\n", k, p[128 * 256]);
    for (int t : {0, 1, 2, 3, 4, 5, 8, 16, 31, 32, 64, 96}) {
      printf(" t%3d:", t);
      for (int i = 0; i < p[128 * 256 + t] && i < 256; i++) { int v = p[t * 256 + i]; if (v < 0) printf(" -"); else printf(" (%d,%d)", v / 4096, v % 4096); }
      printf("\n");
    }
  }
  {
    auto o8 = d.alloc((256 * 256 + 256) * 4);
    { ms::Enc e; e.set_pipeline("attn_probe_coop8_64x128"); e.set_buffer(0, o8); e.dispatch_groups(1, 1, 1, 256); e.wait(); }
    int* p = (int*)o8.cpu(); int bad = 0;
    printf("coop8 64x128 capacity %d\n", p[256 * 256]);
    for (int t = 0; t < 256; t++) for (int i = 0; i + 1 < p[256 * 256 + t]; i += 2) {
      int a = p[t * 256 + i], b = p[t * 256 + i + 1];
      if (a < 0 || b != a + 1 || (a % 4096) % 2) bad++;
    }
    printf("pairs (i, i+1) = (d even, d+1) same row: %s (%d bad)\n", bad ? "NO" : "yes", bad);
    for (int t : {0, 1, 2, 32, 64, 128, 224}) { printf(" t%3d:", t); for (int i = 0; i < p[256 * 256 + t] && i < 32; i++) { int v = p[t * 256 + i]; printf(" (%d,%d)", v / 4096, v % 4096); } printf("\n"); }
  }
}
