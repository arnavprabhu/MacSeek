// EngramFetcher tests: synthetic 144 B-row table (all read modes, dedupe, cache, async) + real rows from the GGUF.
#include "store/engram.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static inline uint8_t pat(uint64_t x) { x ^= x >> 7; x *= 0x9E3779B97F4A7C15ULL; return (uint8_t)(x >> 56); }

int main() {
  const uint32_t RB = 144; const uint64_t NR = 200000, BASE = 192;
  const char* tmp = getenv("TMPDIR");
  std::string path = std::string(tmp ? tmp : "/tmp") + "/ms_test_engram.bin";
  {
    std::vector<uint8_t> b(BASE + NR * RB);
    for (size_t i = 0; i < b.size(); i++) b[i] = pat(i);
    FILE* f = fopen(path.c_str(), "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f);
  }
  int fd = open(path.c_str(), O_RDONLY);
  fcntl(fd, F_NOCACHE, 1);
  auto check_rows = [&](const std::vector<uint64_t>& rows, const std::vector<uint8_t>& out) {
    for (size_t i = 0; i < rows.size(); i++)
      for (uint32_t j = 0; j < RB; j++)
        if (out[i * RB + j] != pat(BASE + rows[i] * RB + j)) { fprintf(stderr, "  row %zu (%llu) byte %u\n", i, (unsigned long long)rows[i], j); return false; }
    return true;
  };
  std::mt19937_64 rng(42);
  for (int mode = 0; mode < 3; mode++) {
    ms::EngramOptions o; o.threads = 8; o.mode = (ms::EngramReadMode)mode; o.cache_bytes = 1 << 20;
    ms::EngramFetcher f(fd, BASE, RB, NR, o);
    // decode-size request with duplicates and neighbours (rows sharing a 4K/16K block), plus the last row (EOF)
    std::vector<uint64_t> rows;
    for (int i = 0; i < 40; i++) rows.push_back(rng() % NR);
    rows.push_back(rows[3]); rows.push_back(rows[3]); rows.push_back(rows[10] + 1 < NR ? rows[10] + 1 : 0);
    rows.push_back(NR - 1); rows.push_back(0); rows.push_back(1); rows.push_back(113); rows.push_back(114);
    std::vector<uint8_t> out(rows.size() * RB);
    f.fetch_sync(rows.data(), rows.size(), out.data());
    CHECK(check_rows(rows, out), "mode %s decode rows", ms::engram_mode_name(o.mode));
    auto s1 = f.stats();
    CHECK(s1.rows == rows.size() && s1.unique < rows.size() && s1.cache_hits == 0, "dedupe: rows %llu unique %llu", s1.rows, s1.unique);
    // same rows again: all from cache, no reads
    std::fill(out.begin(), out.end(), 0);
    f.fetch_sync(rows.data(), rows.size(), out.data());
    auto s2 = f.stats();
    CHECK(check_rows(rows, out) && s2.reads == s1.reads && s2.cache_hits - s1.cache_hits == s1.unique, "cache hits");
    // batch (prompt) request, async
    std::vector<uint64_t> big;
    for (int i = 0; i < 20000; i++) big.push_back(rng() % NR);
    std::vector<uint8_t> bo(big.size() * RB);
    auto t = f.fetch(big.data(), big.size(), bo.data());
    t.wait();
    CHECK(t.ready() && t.latency_ms() >= 0 && check_rows(big, bo), "mode %s batch", ms::engram_mode_name(o.mode));
    auto s3 = f.stats();
    printf("mode %-5s: decode %zu rows -> %llu unique, %llu reads; batch 20000 rows -> %llu reads, %.1f MB read, %.2f ms\n",
           ms::engram_mode_name(o.mode), rows.size(), s1.unique, s1.reads, s3.reads - s2.reads, (s3.bytes_read - s2.bytes_read) / 1e6, t.latency_ms());
  }
  // no cache
  {
    ms::EngramOptions o; o.threads = 4; o.cache_bytes = 0;
    ms::EngramFetcher f(fd, BASE, RB, NR, o);
    std::vector<uint64_t> rows = {5, 5, 7};
    std::vector<uint8_t> out(3 * RB);
    f.fetch_sync(rows.data(), 3, out.data());
    f.fetch_sync(rows.data(), 3, out.data());
    CHECK(check_rows(rows, out) && f.stats().cache_hits == 0 && f.stats().reads == 4, "no-cache");
  }
  close(fd);
  unlink(path.c_str());

  // real table: a few rows vs plain pread
  std::string mp = ms::Gguf::default_model_path();
  if (ms::Gguf::exists(mp)) {
    ms::Gguf g(mp);
    const ms::TensorInfo& t = g.at("per_layer_token_embd.weight");
    for (int mode = 0; mode < 3; mode++) {
      ms::EngramOptions o; o.threads = 16; o.mode = (ms::EngramReadMode)mode; o.cache_bytes = 1 << 20;
      ms::EngramFetcher f(g, o);
      CHECK(f.row_bytes() == 144 && f.n_rows() == 768022850ull, "real geometry");
      std::vector<uint64_t> rows = {0, 1, 768022849ull, 16000057ull, 123456789ull};
      for (int i = 0; i < 43; i++) rows.push_back(rng() % f.n_rows());
      std::vector<uint8_t> out(rows.size() * 144), ref(144);
      f.fetch_sync(rows.data(), rows.size(), out.data());
      bool ok = true;
      for (size_t i = 0; i < rows.size(); i++) {
        ms::pread_full(g.fd(t.split), ref.data(), 144, t.offset + rows[i] * 144);
        ok &= memcmp(ref.data(), &out[i * 144], 144) == 0;
      }
      CHECK(ok, "real rows mode %s", ms::engram_mode_name(o.mode));
    }
  } else {
    printf("real-model part skipped\n");
  }
  printf("test_store_engram: %s\n", fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
