#include "store/engram.hpp"

#include "core/env.hpp"
#include "store/io_pool.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <string>

namespace ms {

const char* engram_mode_name(EngramReadMode m) {
  switch (m) { case EngramReadMode::Exact: return "exact"; case EngramReadMode::Page4K: return "4k"; case EngramReadMode::Page16K: return "16k"; }
  return "?";
}

EngramOptions EngramOptions::from_env() {
  EngramOptions o;
  o.threads = (int)env_int("MACSEEK_ENGRAM_THREADS", o.threads);
  std::string m = env_str("MACSEEK_ENGRAM_MODE", "exact");
  o.mode = m == "4k" ? EngramReadMode::Page4K : m == "16k" ? EngramReadMode::Page16K : EngramReadMode::Exact;
  long mb = env_int("MACSEEK_ENGRAM_CACHE_MB", -1);
  if (mb >= 0) o.cache_bytes = (size_t)mb << 20;
  return o;
}

struct EngramFetcher::State {
  std::vector<std::pair<uint64_t, uint32_t>> sorted;   // (row, dst index)
  struct Miss { uint64_t row; uint32_t first, count; };   // run in `sorted`
  std::vector<Miss> miss;
  std::vector<std::pair<uint32_t, uint32_t>> chunk_miss;   // chunk -> [miss begin, end)
  char* dst = nullptr;
  std::mutex mu; std::condition_variable cv; bool done = false;
  std::chrono::steady_clock::time_point t0, t1;
};

EngramFetcher::EngramFetcher(const Gguf& g, const EngramOptions& o) : o_(o) {
  const TensorInfo& t = g.at("per_layer_token_embd.weight");
  fd_ = o.nocache ? g.fd_nocache(t.split) : g.fd(t.split);
  base_ = t.offset; row_bytes_ = (uint32_t)t.row_bytes(); n_rows_ = (uint64_t)t.ne[1];
  init();
}

EngramFetcher::EngramFetcher(int fd, uint64_t base, uint32_t rb, uint64_t nr, const EngramOptions& o)
    : fd_(fd), base_(base), n_rows_(nr), row_bytes_(rb), o_(o) { init(); }

void EngramFetcher::init() {
  pool_ = std::make_unique<IoPool>(o_.threads, std::max<uint32_t>(o_.max_merge, 32768) + 16384);
  if (o_.cache_bytes >= (size_t)kWays * row_bytes_) {
    uint64_t sets = o_.cache_bytes / ((uint64_t)kWays * row_bytes_);
    n_sets_ = 1; while (n_sets_ * 2 <= sets) n_sets_ *= 2;
    keys_.assign(n_sets_ * kWays, 0);
    ages_.assign(n_sets_ * kWays, 0);
    data_.resize(n_sets_ * kWays * row_bytes_);
  }
  locks_.reset(new std::mutex[kStripes]);
}

EngramFetcher::~EngramFetcher() { pool_.reset(); }

static inline uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }

bool EngramFetcher::cached(uint64_t row, void* out) {
  if (!n_sets_) return false;
  uint64_t set = mix(row) & (n_sets_ - 1);
  std::lock_guard<std::mutex> g(locks_[set % kStripes]);
  uint64_t* k = &keys_[set * kWays];
  for (int w = 0; w < kWays; w++)
    if (k[w] == row + 1) {
      memcpy(out, &data_[(set * kWays + w) * row_bytes_], row_bytes_);
      ages_[set * kWays + w] = ++clock_;
      return true;
    }
  return false;
}

void EngramFetcher::cache_put(uint64_t row, const char* d) {
  if (!n_sets_) return;
  uint64_t set = mix(row) & (n_sets_ - 1);
  std::lock_guard<std::mutex> g(locks_[set % kStripes]);
  uint64_t* k = &keys_[set * kWays];
  uint32_t* a = &ages_[set * kWays];
  int victim = 0;
  uint32_t now = ++clock_;
  for (int w = 0; w < kWays; w++) {
    if (k[w] == row + 1) { a[w] = now; return; }
    if (k[w] == 0) { victim = w; break; }
    if ((uint32_t)(now - a[w]) > (uint32_t)(now - a[victim])) victim = w;   // oldest (wrap-safe)
  }
  k[victim] = row + 1; a[victim] = now;
  memcpy(&data_[(set * kWays + victim) * row_bytes_], d, row_bytes_);
}

void EngramFetcher::clear_cache() {
  for (int i = 0; i < kStripes; i++) locks_[i].lock();
  std::fill(keys_.begin(), keys_.end(), 0);
  for (int i = 0; i < kStripes; i++) locks_[i].unlock();
}

EngramFetcher::Ticket EngramFetcher::fetch(const uint64_t* rows, size_t n, void* dst) {
  auto st = std::make_shared<State>();
  st->t0 = std::chrono::steady_clock::now();
  st->dst = (char*)dst;
  st->sorted.resize(n);
  for (size_t i = 0; i < n; i++) {
    if (rows[i] >= n_rows_) { fprintf(stderr, "engram: row %llu out of range\n", (unsigned long long)rows[i]); abort(); }
    st->sorted[i] = {rows[i], (uint32_t)i};
  }
  std::sort(st->sorted.begin(), st->sorted.end());
  const uint32_t RB = row_bytes_;
  uint64_t unique = 0, hits = 0;
  for (size_t i = 0; i < n;) {
    size_t j = i + 1;
    while (j < n && st->sorted[j].first == st->sorted[i].first) j++;
    unique++;
    char* first = st->dst + (size_t)st->sorted[i].second * RB;
    if (cached(st->sorted[i].first, first)) {
      hits++;
      for (size_t k = i + 1; k < j; k++) memcpy(st->dst + (size_t)st->sorted[k].second * RB, first, RB);
    } else {
      st->miss.push_back({st->sorted[i].first, (uint32_t)i, (uint32_t)(j - i)});
    }
    i = j;
  }
  auto req = std::make_shared<IoRequest>();
  const uint64_t A = o_.mode == EngramReadMode::Exact ? 1 : o_.mode == EngramReadMode::Page4K ? 4096 : 16384;
  for (size_t m = 0; m < st->miss.size();) {
    uint64_t off = base_ + st->miss[m].row * RB;
    uint64_t lo = off / A * A, hi = (off + RB + A - 1) / A * A;
    size_t me = m + 1;
    if (A > 1) {
      while (me < st->miss.size()) {
        uint64_t o2 = base_ + st->miss[me].row * RB;
        uint64_t lo2 = o2 / A * A, hi2 = (o2 + RB + A - 1) / A * A;
        if (lo2 > hi || hi2 - lo > o_.max_merge) break;
        hi = std::max(hi, hi2); me++;
      }
    }
    req->chunks.push_back({fd_, lo, (uint32_t)(hi - lo), nullptr, 0});
    st->chunk_miss.push_back({(uint32_t)m, (uint32_t)me});
    m = me;
  }
  {
    std::lock_guard<std::mutex> g(smu_);
    st_.requests++; st_.rows += n; st_.unique += unique; st_.cache_hits += hits; st_.reads += req->chunks.size();
  }
  Ticket t; t.st_ = st;
  if (req->chunks.empty()) { st->t1 = std::chrono::steady_clock::now(); st->done = true; return t; }
  State* sp = st.get();
  req->on_chunk = [this, sp, RB](IoRequest& r, size_t ci, const char* data, long long got) {
    const IoChunk& c = r.chunks[ci];
    if (got < (long long)c.len) {
      // tail of the file: only fail if a needed row is missing
      for (uint32_t m = sp->chunk_miss[ci].first; m < sp->chunk_miss[ci].second; m++)
        if (base_ + (sp->miss[m].row + 1) * RB > c.off + (uint64_t)std::max<long long>(got, 0)) { fprintf(stderr, "engram: short read\n"); abort(); }
    }
    for (uint32_t m = sp->chunk_miss[ci].first; m < sp->chunk_miss[ci].second; m++) {
      const auto& ms_ = sp->miss[m];
      const char* src = data + (base_ + ms_.row * RB - c.off);
      for (uint32_t k = ms_.first; k < ms_.first + ms_.count; k++) memcpy(sp->dst + (size_t)sp->sorted[k].second * RB, src, RB);
      cache_put(ms_.row, src);
    }
  };
  req->on_done = [this, st](IoRequest& r) {
    auto t1 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(smu_);
      st_.bytes_read += r.bytes.load();
      st_.fetch_ms += std::chrono::duration<double, std::milli>(t1 - st->t0).count();
    }
    std::lock_guard<std::mutex> g(st->mu);
    st->t1 = t1;
    st->done = true;
    st->cv.notify_all();
  };
  pool_->submit(std::move(req));
  return t;
}

void EngramFetcher::Ticket::wait() {
  if (!st_) return;
  std::unique_lock<std::mutex> lk(st_->mu);
  st_->cv.wait(lk, [&] { return st_->done; });
}
bool EngramFetcher::Ticket::ready() const { if (!st_) return true; std::lock_guard<std::mutex> g(st_->mu); return st_->done; }
double EngramFetcher::Ticket::latency_ms() const {
  if (!st_) return 0;
  std::lock_guard<std::mutex> g(st_->mu);
  return st_->done ? std::chrono::duration<double, std::milli>(st_->t1 - st_->t0).count() : -1;
}

EngramStats EngramFetcher::stats() const { std::lock_guard<std::mutex> g(smu_); return st_; }
void EngramFetcher::reset_stats() { std::lock_guard<std::mutex> g(smu_); st_ = EngramStats{}; }

}  // namespace ms
