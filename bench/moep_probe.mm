// bench_moep_probe: dump cooperative-tensor layouts used by the MoE prefill register kernels.
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "core/metal.hpp"
#import <Metal/Metal.h>
using namespace ms;
int main(int argc, char** argv) {
  auto& d = Device::get();
  if (argc > 1 && !strcmp(argv[1], "occ")) {  // pipeline limits (register pressure shows as maxTotalThreadsPerThreadgroup < 1024)
    for (int i = 2; i < argc; i++) {
      id<MTLComputePipelineState> p = (__bridge id<MTLComputePipelineState>)d.pipeline(argv[i]);
      printf("%-40s maxThreads %4lu  width %lu\n", argv[i], (unsigned long)p.maxTotalThreadsPerThreadgroup,
             (unsigned long)p.threadExecutionWidth);
    }
    return 0;
  }
  if (argc > 1 && !strcmp(argv[1], "tg4")) {  // execution_simdgroups<4> 64x128 destination layout
    Buffer A = d.alloc(128 * 128 * 2), B = d.alloc(128 * 128 * 2), o4 = d.alloc((128 * 256 + 128) * 4);
    Enc e; e.set_pipeline("attn_probe_coop_64x128"); e.set_buffer(0, A); e.set_buffer(1, B); e.set_buffer(2, o4);
    e.dispatch_groups(1, 1, 1, 128); e.wait();
    const int* o = (const int*)o4.cpu();
    for (int t = 0; t < 128; t += (argc > 2 ? 1 : 9)) {
      printf(" T%03d cap %d:", t, o[128 * 256 + t]);
      for (int i = 0; i < std::min(o[128 * 256 + t], 256); i++) { int v = o[t * 256 + i]; if (v < 0) printf(" x"); else printf(" n%d/t%d", v % 4096, v / 4096); }
      printf("\n");
    }
    return 0;
  }
  Buffer X = d.alloc(128 * 64 * 2), out = d.alloc(32 * 256 * 4);
  for (const char* k : {"q_moep_probe_dst_64x32", "q_moep_probe_dst_64x16", "q_moep_probe_dst_32x64", "q_moep_probe_dst_128x16", "q_moep_probe_dst_48x32", "q_moep_probe_dst_16x64", "q_moep_probe_dst_64x64"}) {
    if (argc > 1 && !strstr(k, argv[1])) continue;
    Enc e; e.set_pipeline(k); e.set_buffer(0, X); e.set_buffer(1, out); e.dispatch_groups(1, 1, 1, 32); e.wait();
    const int* o = (const int*)out.cpu();
    printf("%s\n", k);
    for (int l = 0; l < 32; l++) {
      printf(" L%02d:", l);
      for (int i = 0; i < 256 && o[l * 256 + i] != -2; i++) {
        int v = o[l * 256 + i];
        if (v < 0) printf(" x"); else printf(" r%d/t%d", v / 4096, v % 4096);
      }
      printf("\n");
    }
  }
}
