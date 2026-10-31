// Expert I/O microbenchmarks on the real GGUF (reads <= ~8 GB per full run; RAM <= ~2.5 GB).
//   bench_experts_io [sections] [--iters N]     sections: lat conc stream cpu sim  (default: all)
// Page-cache hygiene: experts whose pages are resident in the unified page cache (mincore on a PROT_READ mapping)
// are never chosen, and no expert is read twice in one run, so every number is a real SSD read.
#import <Metal/Metal.h>

#include "core/timer.hpp"
#include "store/experts.hpp"
#include "store/memguard.hpp"

#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace ms;

struct Residency {   // mincore over read-only mappings of every split
  std::vector<std::pair<char*, size_t>> maps; std::vector<int> fds;
  explicit Residency(const Gguf& g) {
    for (int s = 0; s < g.n_splits(); s++) {
      void* p = mmap(nullptr, g.file_size(s), PROT_READ, MAP_SHARED, g.fd(s), 0);
      maps.push_back({p == MAP_FAILED ? nullptr : (char*)p, g.file_size(s)});
      fds.push_back(g.fd_nocache(s));
    }
  }
  ~Residency() { for (auto& m : maps) if (m.first) munmap(m.first, m.second); }
  double frac(int fd_nc, uint64_t off, uint64_t len) const {
    for (size_t s = 0; s < fds.size(); s++) if (fds[s] == fd_nc && maps[s].first) {
      uint64_t a = off / 16384 * 16384, b = (off + len + 16383) / 16384 * 16384;
      std::vector<char> v((b - a) / 16384);
      if (mincore(maps[s].first + a, b - a, v.data()) != 0) return 0;
      int r = 0; for (char c : v) r += c & 1;
      return (double)r / v.size();
    }
    return 0;
  }
};

struct Picker {
  const ExpertLayout& L; const Residency& R; std::mt19937_64 rng{1234}; std::set<uint32_t> used; int skipped = 0;
  Picker(const ExpertLayout& l, const Residency& r) : L(l), R(r) {}
  bool cold(int l, int e) {
    for (int k = 0; k < 3; k++) { auto& m = L.layers[l].mat[k]; if (R.frac(m.fd, m.base + (uint64_t)e * m.stride, m.len) > 0) return false; }
    return true;
  }
  std::vector<int> pick(int l, int n) {
    std::vector<int> out;
    while ((int)out.size() < n) {
      int e = (int)(rng() % L.n_experts);
      uint32_t k = (uint32_t)(l * L.n_experts + e);
      if (used.count(k)) continue;
      used.insert(k);
      if (!cold(l, e)) { skipped++; continue; }
      out.push_back(e);
    }
    return out;
  }
  int layer() { return (int)(rng() % L.n_layers); }
};

static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[std::min(v.size() - 1, (size_t)(p * v.size()))]; }
static double cpu_s() { rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::set<std::string> sec; int iters = 20;
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]); else sec.insert(a); }
  auto on = [&](const char* s) { return sec.empty() || sec.count(s); };
  Gguf g(Gguf::default_model_path());
  ExpertLayout L = ExpertLayout::from_gguf(g);
  Residency R(g);
  Picker P(L, R);
  MemStatus m0 = mem_status();
  printf("mem before: %s\n", m0.str().c_str());
  uint64_t total_read = 0;

  if (on("lat")) {
    printf("\n== ensure() latency, 6 cold misses in one layer (%d iters per config) ==\n", iters);
    printf("threads chunkKB align |  p50_ms  p90_ms  max_ms  mean_ms | eff_GB/s | ensure_us  cpu_us/miss\n");
    struct Cfg { int t; size_t chunk; uint32_t align; } cfgs[] = {
        {8, 1 << 20, 16384}, {16, 1 << 20, 16384}, {16, 0, 16384}, {16, 256 << 10, 16384}, {16, 1 << 20, 1}, {16, 0, 1}, {12, 512 << 10, 16384}};
    for (auto c : cfgs) {
      ExpertCacheOptions o; o.arena_bytes = 2ull << 30; o.io_threads = c.t; o.read_chunk = c.chunk; o.align = c.align;
      ExpertCache cache(L, o);
      std::vector<double> lat, call;
      uint64_t bytes = 0;
      // warm-up (thread start, page faults on first slot touches)
      { int l = P.layer(); auto ids = P.pick(l, 6); auto x = cache.ensure(l, ids); x.wait(); }
      cache.reset_stats();
      double c0 = cpu_s();
      for (int it = 0; it < iters; it++) {
        int l = P.layer();
        auto ids = P.pick(l, 6);
        Timer t;
        auto lease = cache.ensure(l, ids);
        double tc = t.ms();
        lease.wait();
        lat.push_back(t.ms()); call.push_back(tc * 1e3);
        bytes += 6 * L.expert_bytes(l);
      }
      double cpu = cpu_s() - c0;
      double sum = 0; for (double x : lat) sum += x;
      auto ts = cache.total_stats();
      total_read += ts.bytes_read;
      printf("%7d %7zu %5u | %7.2f %7.2f %7.2f %8.2f | %8.2f | %9.1f %12.1f\n", c.t, c.chunk >> 10, c.align, pct(lat, .5), pct(lat, .9),
             pct(lat, 1), sum / iters, bytes / (sum * 1e6), pct(call, .5), cpu * 1e6 / (6.0 * iters));
    }
  }

  if (on("conc")) {
    printf("\n== 16 concurrent clients, 1 cold miss each per ensure (pool threads = 16) ==\n");
    ExpertCacheOptions o; o.arena_bytes = 2ull << 30; o.io_threads = 16; o.read_chunk = 1 << 20;
    ExpertCache cache(L, o);
    const int C = 16, per = std::max(4, iters / 2);
    std::vector<std::vector<std::pair<int, int>>> work(C);
    for (int c = 0; c < C; c++) for (int i = 0; i < per; i++) { int l = P.layer(); work[c].push_back({l, P.pick(l, 1)[0]}); }
    std::vector<std::vector<double>> lat(C);
    Timer t;
    std::vector<std::thread> th;
    for (int c = 0; c < C; c++) th.emplace_back([&, c] {
      for (auto [l, e] : work[c]) { Timer tt; auto x = cache.ensure(l, &e, 1); x.wait(); lat[c].push_back(tt.ms()); }
    });
    for (auto& x : th) x.join();
    double ms = t.ms();
    auto ts = cache.total_stats();
    total_read += ts.bytes_read;
    std::vector<double> all; for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
    printf("%d experts, %.2f GB in %.1f ms: %.2f GB/s; per-ensure p50 %.2f ms p99 %.2f ms\n", C * per, ts.bytes_read / 1e9, ms,
           ts.bytes_read / (ms * 1e6), pct(all, .5), pct(all, .99));
  }

  if (on("stream")) {
    printf("\n== bulk stream (prefill): one layer, cold experts through the ring, CPU consumer ==\n");
    for (int ring : {8, 16}) {
      ExpertCacheOptions o; o.arena_bytes = 2ull << 30; o.io_threads = 16; o.read_chunk = 1 << 20; o.ring_slots = ring;
      ExpertCache cache(L, o);
      int l = 3 + ring;   // different layer per config
      auto ids = P.pick(l, 48);
      Timer t;
      auto st = cache.stream(l, ids.data(), (int)ids.size());
      for (size_t i = 0; i < st->items().size(); i++) { st->wait_ready(i); st->consumed(i); }
      double ms = t.ms();
      auto ls = cache.layer_stats(l);
      total_read += ls.stream_bytes;
      printf("ring %2d: %llu experts, %.2f GB in %.1f ms: %.2f GB/s (consumer stall %.1f ms)\n", ring, ls.streamed, ls.stream_bytes / 1e9, ms,
             ls.stream_bytes / (ms * 1e6), ls.stall_ms);
    }
  }

  if (on("cpu") || on("sim")) {
    printf("\n== bookkeeping cost + policy simulation (dry_io: no reads, no memory; real geometry, 40 GiB arena) ==\n");
    printf("   synthetic routing: per layer Zipf(s) over a random expert permutation, 6 distinct experts per token\n");
    for (double s : {0.6, 1.0}) {
      // trace
      const int T = 4000;
      std::mt19937_64 rng(7);
      std::vector<std::vector<double>> cdf(L.n_layers);
      std::vector<std::vector<int>> perm(L.n_layers);
      for (int l = 0; l < L.n_layers; l++) {
        perm[l].resize(L.n_experts);
        for (int e = 0; e < L.n_experts; e++) perm[l][e] = e;
        std::shuffle(perm[l].begin(), perm[l].end(), rng);
        double acc = 0;
        for (int e = 0; e < L.n_experts; e++) { acc += 1.0 / std::pow(e + 1, s); cdf[l].push_back(acc); }
        for (double& x : cdf[l]) x /= acc;
      }
      std::uniform_real_distribution<double> U(0, 1);
      std::vector<std::vector<std::vector<int>>> trace(T, std::vector<std::vector<int>>(L.n_layers));
      for (int t = 0; t < T; t++) for (int l = 0; l < L.n_layers; l++) {
        auto& v = trace[t][l];
        while (v.size() < 6) {
          int r = (int)(std::lower_bound(cdf[l].begin(), cdf[l].end(), U(rng)) - cdf[l].begin());
          int e = perm[l][std::min(r, L.n_experts - 1)];
          if (std::find(v.begin(), v.end(), e) == v.end()) v.push_back(e);
        }
      }
      // routing stats from the first half (train) for the static policy
      const char* tmp = getenv("TMPDIR");
      std::string sp = std::string(tmp ? tmp : "/tmp") + "/ms_bench_routing.txt";
      {
        FILE* f = fopen(sp.c_str(), "w");
        std::vector<double> cnt((size_t)L.n_layers * L.n_experts, 0);
        for (int t = 0; t < T / 2; t++) for (int l = 0; l < L.n_layers; l++) for (int e : trace[t][l]) cnt[(size_t)l * L.n_experts + e]++;
        for (int l = 0; l < L.n_layers; l++) { for (int e = 0; e < L.n_experts; e++) fprintf(f, "%.0f ", cnt[(size_t)l * L.n_experts + e]); fprintf(f, "\n"); }
        fclose(f);
      }
      struct PC { PolicyKind k; int sample; const char* name; } pcs[] = {
          {PolicyKind::LRU, 0, "lru"}, {PolicyKind::LFU, 0, "lfu-exact"}, {PolicyKind::LFU, 64, "lfu-s64"}, {PolicyKind::Static, 0, "static"}};
      for (auto pc : pcs) {
        PolicyKind pk = pc.k;
        ExpertCacheOptions o; o.arena_bytes = 40ull << 30; o.dry_io = true; o.policy = pk; o.io_threads = 1; o.lfu_sample = pc.sample;
        if (pk == PolicyKind::Static) o.stats_path = sp;
        ExpertCache cache(L, o);
        for (int t = 0; t < T / 2; t++) for (int l = 0; l < L.n_layers; l++) cache.ensure(l, trace[t][l]);   // warm
        cache.reset_stats();
        Timer tm;
        for (int t = T / 2; t < T; t++) for (int l = 0; l < L.n_layers; l++) cache.ensure(l, trace[t][l]);
        double us = tm.ms() * 1e3 / ((T / 2) * L.n_layers);
        auto ts = cache.total_stats();
        double miss_gb_tok = 0;
        for (int l = 0; l < L.n_layers; l++) miss_gb_tok += cache.layer_stats(l).misses * (double)L.expert_bytes(l);
        miss_gb_tok /= (T / 2) * 1e9;
        printf("zipf s=%.1f  %-9s: hit %.1f%%  miss bytes/token %.3f GB (-> %.1f ms/token @14 GB/s)  ensure(6) %.2f us/call  slots %d pinned %d\n",
               s, pc.name, 100 * ts.hit_rate(), miss_gb_tok, miss_gb_tok / 14.0 * 1e3, us, cache.n_slots(), cache.n_pinned());
      }
    }
  }
  MemStatus m1 = mem_status();
  printf("\ntotal expert bytes read: %.2f GB (cold picks skipped as page-cache resident: %d)\nmem after: %s\nswap grew: %s\n",
         total_read / 1e9, P.skipped, m1.str().c_str(), swap_grew(m0, m1) ? "YES" : "no");
  return 0;
}
