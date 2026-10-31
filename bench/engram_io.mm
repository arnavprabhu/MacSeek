// Engram table I/O microbenchmarks on the real GGUF (per_layer_token_embd, 144 B rows, 103 GiB; reads ~1.5 GB).
//   bench_engram_io [--decode-iters N] [--batch-tokens N]
// Rows are uniformly random (engram hashes are) and never repeated; rows in page-cache-resident pages are skipped
// (mincore), and the row cache is disabled, so every number is SSD latency/throughput.
#include "core/timer.hpp"
#include "store/engram.hpp"
#include "store/memguard.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ms;

static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[std::min(v.size() - 1, (size_t)(p * v.size()))]; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int decode_iters = 100, batch_tokens = 256;
  for (int i = 1; i + 1 < argc; i++) {
    std::string a = argv[i];
    if (a == "--decode-iters") decode_iters = atoi(argv[++i]);
    else if (a == "--batch-tokens") batch_tokens = atoi(argv[++i]);
  }
  Gguf g(Gguf::default_model_path());
  const TensorInfo& t = g.at("per_layer_token_embd.weight");
  char* map = (char*)mmap(nullptr, g.file_size(t.split), PROT_READ, MAP_SHARED, g.fd(t.split), 0);
  const uint64_t NR = (uint64_t)t.ne[1];
  std::mt19937_64 rng(99);
  std::unordered_set<uint64_t> used_pages;
  int skipped = 0;
  auto cold_rows = [&](size_t n) {
    std::vector<uint64_t> r;
    while (r.size() < n) {
      uint64_t row = rng() % NR, off = t.offset + row * 144, pg = off / 16384, pg2 = (off + 143) / 16384;
      if (used_pages.count(pg) || used_pages.count(pg2)) continue;
      char v[2] = {0, 0};
      if (map != MAP_FAILED) { mincore(map + pg * 16384, (pg2 - pg + 1) * 16384, v); if ((v[0] | v[1]) & 1) { skipped++; continue; } }
      used_pages.insert(pg); used_pages.insert(pg2);
      r.push_back(row);
    }
    return r;
  };
  MemStatus m0 = mem_status();
  printf("mem before: %s\n", m0.str().c_str());
  uint64_t total = 0;
  printf("\n== decode: 48 rows/token (2 engram layers x 24), %d tokens per config, row cache off ==\n", decode_iters);
  printf("mode   QD |  p50_ms  p90_ms  p99_ms  max_ms | rows/s(seq tokens)\n");
  std::vector<uint8_t> out(48 * 144);
  for (auto mode : {EngramReadMode::Exact, EngramReadMode::Page4K, EngramReadMode::Page16K})
    for (int qd : {8, 16, 32}) {
      EngramOptions o; o.mode = mode; o.threads = qd; o.cache_bytes = 0;
      EngramFetcher f(g, o);
      { auto w = cold_rows(48); f.fetch_sync(w.data(), 48, out.data()); }   // warm-up threads
      f.reset_stats();
      std::vector<double> lat;
      Timer tt;
      for (int i = 0; i < decode_iters; i++) {
        auto rows = cold_rows(48);
        Timer t1;
        f.fetch_sync(rows.data(), rows.size(), out.data());
        lat.push_back(t1.ms());
      }
      double ms = tt.ms();
      total += f.stats().bytes_read;
      printf("%-5s %3d | %7.3f %7.3f %7.3f %7.3f | %9.0f\n", engram_mode_name(mode), qd, pct(lat, .5), pct(lat, .9), pct(lat, .99), pct(lat, 1),
             48.0 * decode_iters / (ms / 1e3));
    }
  printf("\n== prompt batch: %d tokens = %d rows in one async request, row cache off ==\n", batch_tokens, batch_tokens * 48);
  printf("mode   QD |   ms     rows/s    reads   MB_read  IOPS\n");
  for (auto mode : {EngramReadMode::Exact, EngramReadMode::Page4K, EngramReadMode::Page16K})
    for (int qd : {16, 32, 64}) {
      EngramOptions o; o.mode = mode; o.threads = qd; o.cache_bytes = 0;
      EngramFetcher f(g, o);
      auto rows = cold_rows((size_t)batch_tokens * 48);
      std::vector<uint8_t> bo(rows.size() * 144);
      Timer t1;
      auto tk = f.fetch(rows.data(), rows.size(), bo.data());
      tk.wait();
      double ms = t1.ms();
      auto s = f.stats();
      total += s.bytes_read;
      printf("%-5s %3d | %7.1f %9.0f %7llu %8.1f %6.0f\n", engram_mode_name(mode), qd, ms, rows.size() / (ms / 1e3), s.reads, s.bytes_read / 1e6,
             s.reads / (ms / 1e3));
    }
  // spot-check correctness of the last configuration against plain preads
  MemStatus m1 = mem_status();
  printf("\ntotal bytes read %.2f GB, rows skipped as resident %d\nmem after: %s\nswap grew: %s\n", total / 1e9, skipped, m1.str().c_str(),
         swap_grew(m0, m1) ? "YES" : "no");
  if (map != MAP_FAILED) munmap(map, g.file_size(t.split));
  return 0;
}
