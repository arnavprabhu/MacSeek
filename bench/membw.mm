// bench_membw: achievable GPU read bandwidth on this machine (decode roofline input).
// Usage: bench_membw [MiB=1024] [iters=20]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/metal.hpp"

using namespace ms;

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? atoi(argv[1]) : 1024;
  int iters = argc > 2 ? atoi(argv[2]) : 20;
  auto& d = Device::get();
  size_t bytes = mib << 20;
  Buffer src = d.alloc(bytes, "membw.src");
  Buffer dst = d.alloc(bytes, "membw.dst");
  Buffer out = d.alloc(64 << 20, "membw.out");
  { uint32_t* p = (uint32_t*)src.cpu(); for (size_t i = 0; i < bytes / 4; i++) p[i] = (uint32_t)i * 2654435761u; }
  printf("buffer %zu MiB, %d iters (median, GB/s = 1e9 B/s)\n", mib, iters);

  auto run = [&](const char* label, auto enc_fn, double bytes_moved) {
    for (int w = 0; w < 3; w++) { Enc e; enc_fn(e); e.wait(); }
    std::vector<double> t;
    for (int i = 0; i < iters; i++) { Enc e; enc_fn(e); e.wait(); t.push_back(e.gpu_ms()); }
    double ms = median(t), best = *std::min_element(t.begin(), t.end());
    printf("  %-34s %7.1f GB/s median  %7.1f GB/s best\n", label, bytes_moved / ms / 1e6, bytes_moved / best / 1e6);
  };

  uint32_t n4 = (uint32_t)(bytes / 16);
  for (uint32_t tgs : {1024u, 2560u, 5120u, 10240u}) {
    char lab[64]; snprintf(lab, sizeof lab, "read f4 grid-stride (%u x 256)", tgs);
    run(lab, [&](Enc& e) { e.set_pipeline("q_bw_read_f4"); e.set_buffer(0, src); e.set_buffer(1, out);
                            e.set_value(2, n4); e.dispatch_groups(tgs, 1, 1, 256); }, (double)bytes);
  }
  for (uint32_t chunk_kib : {64u, 256u, 1024u}) {
    uint32_t per_tg = chunk_kib * 1024 / 16, ntg = (uint32_t)(bytes / (chunk_kib * 1024));
    for (uint32_t tsz : {256u, 1024u}) {
      char lab[64]; snprintf(lab, sizeof lab, "read chunk %u KiB/TG, %u thr", chunk_kib, tsz);
      run(lab, [&](Enc& e) { e.set_pipeline("q_bw_read_chunk"); e.set_buffer(0, src); e.set_buffer(1, out);
                              e.set_value(2, per_tg); e.dispatch_groups(ntg, 1, 1, tsz); }, (double)bytes);
    }
  }
  run("copy f4 (read+write)", [&](Enc& e) { e.set_pipeline("q_bw_copy"); e.set_buffer(0, src); e.set_buffer(1, dst);
                                            e.dispatch_threads(n4, 1, 1, 256); }, 2.0 * bytes);
  return 0;
}
