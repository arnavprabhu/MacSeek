#pragma once
// Engram table row fetcher: per_layer_token_embd (IQ4_NL [256 x 768022850], 144 B rows, 103 GiB) stays on SSD.
// Given global row ids (24 per engram layer per token, 2 engram layers -> 48 rows/token) it writes the packed row
// bytes (row i at dst + i*144) into a caller buffer (typically a shared MTLBuffer's contents()).
//   - dedupes rows within a request, serves repeats from a set-associative RAM row cache (<= 1 GiB, 8-way LRU)
//   - misses become preads on an IoPool (queue depth = threads): exact 144 B, or the containing 4 KiB / 16 KiB
//     aligned block(s) (neighbouring misses that share/abut blocks are merged into one read)
//   - async: fetch() returns a ticket immediately, so a prompt's rows can be read while layer 0 runs
#include "gguf/gguf.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace ms {

class IoPool;

enum class EngramReadMode { Exact = 0, Page4K = 1, Page16K = 2 };
const char* engram_mode_name(EngramReadMode m);

struct EngramOptions {
  int threads = 32;                     // queue depth (decode 48 rows: p50 0.32 ms @32 vs 0.37 @16)
  EngramReadMode mode = EngramReadMode::Exact;
  size_t cache_bytes = 512ull << 20;    // row cache (0 = off); SPEC: <= 1 GiB
  bool nocache = true;                  // F_NOCACHE fd (do not let 103 GiB of random pages churn the page cache)
  uint32_t max_merge = 64 << 10;        // page modes: max bytes of one merged read
  static EngramOptions from_env();      // MACSEEK_ENGRAM_THREADS, MACSEEK_ENGRAM_MODE=exact|4k|16k, MACSEEK_ENGRAM_CACHE_MB
};

struct EngramStats {
  uint64_t requests = 0, rows = 0, unique = 0, cache_hits = 0, reads = 0, bytes_read = 0;
  double fetch_ms = 0;                  // sum of request latencies (submit -> done)
};

class EngramFetcher {
 public:
  struct State;
  class Ticket {
   public:
    Ticket() = default;
    void wait();                        // block until all rows are in dst
    bool ready() const;
    double latency_ms() const;          // submit -> done (after wait)
   private:
    friend class EngramFetcher;
    std::shared_ptr<State> st_;
  };

  EngramFetcher(const Gguf& g, const EngramOptions& o = EngramOptions::from_env());
  EngramFetcher(int fd, uint64_t base, uint32_t row_bytes, uint64_t n_rows, const EngramOptions& o);   // tests
  ~EngramFetcher();

  Ticket fetch(const uint64_t* rows, size_t n, void* dst);   // async; dst must stay valid until wait()
  void fetch_sync(const uint64_t* rows, size_t n, void* dst) { fetch(rows, n, dst).wait(); }
  bool cached(uint64_t row, void* out);                      // row-cache lookup only

  uint32_t row_bytes() const { return row_bytes_; }
  uint64_t n_rows() const { return n_rows_; }
  const EngramOptions& options() const { return o_; }
  EngramStats stats() const;
  void reset_stats();
  void clear_cache();

 private:
  void init();
  void cache_put(uint64_t row, const char* data);
  int fd_ = -1;
  uint64_t base_ = 0, n_rows_ = 0;
  uint32_t row_bytes_ = 0;
  EngramOptions o_;
  std::unique_ptr<IoPool> pool_;
  // set-associative row cache
  static constexpr int kWays = 8, kStripes = 256;
  uint64_t n_sets_ = 0;
  std::vector<uint64_t> keys_;          // row + 1 (0 = empty)
  std::vector<uint32_t> ages_;
  std::vector<char> data_;
  std::unique_ptr<std::mutex[]> locks_;
  std::atomic<uint32_t> clock_{0};
  mutable std::mutex smu_;
  EngramStats st_;
};

}  // namespace ms
