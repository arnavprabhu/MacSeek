#include "store/weights.hpp"

#include "core/env.hpp"
#include "core/timer.hpp"
#include "store/io_pool.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ms {

static constexpr uint64_t kPage = 16384;
static inline uint64_t rdown(uint64_t x, uint64_t a) { return x / a * a; }
static inline uint64_t rup(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

WeightsOptions WeightsOptions::from_env() {
  WeightsOptions o;
  o.token_embd_resident = env_int("MACSEEK_EMBD_RESIDENT", 0) != 0;
  o.threads = (int)env_int("MACSEEK_WEIGHTS_THREADS", o.threads);
  o.mlock = env_int("MACSEEK_WEIGHTS_MLOCK", 0) != 0;
  return o;
}

bool Weights::is_streamed(const std::string& n) {
  return n.find("_exps.") != std::string::npos || n.rfind("per_layer_token_embd", 0) == 0;
}

static void wait_request(const IoRequestPtr& r, std::mutex& mu, std::condition_variable& cv, bool& done) {
  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&] { return done; });
  (void)r;
}

Weights::Weights(const Gguf& g, const WeightsOptions& o) {
  Timer timer;
  // 1. select + sort by (split, offset)
  std::vector<const TensorInfo*> sel;
  for (auto& t : g.tensors()) {
    if (is_streamed(t.name)) continue;
    if (!o.token_embd_resident && t.name == "token_embd.weight") continue;
    if (o.filter && !o.filter(t)) continue;
    sel.push_back(&t);
  }
  std::sort(sel.begin(), sel.end(), [](auto a, auto b) { return a->split != b->split ? a->split < b->split : a->offset < b->offset; });

  // 2. contiguous runs (merge gaps <= 256 KiB: cheaper to read through than to split a request)
  struct Run { int split; uint64_t lo, hi; size_t t0, t1; uint64_t arena_off; int buf; };
  std::vector<Run> runs;
  for (size_t i = 0; i < sel.size(); i++) {
    const TensorInfo* t = sel[i];
    if (!runs.empty() && runs.back().split == t->split && t->offset <= runs.back().hi + (256 << 10)) {
      runs.back().hi = std::max(runs.back().hi, t->offset + t->nbytes);
      runs.back().t1 = i + 1;
    } else {
      runs.push_back({t->split, t->offset, t->offset + t->nbytes, i, i + 1, 0, 0});
    }
  }
  // 3. arena layout: each run gets [rdown(lo), rup(hi)) at a 16K-aligned arena offset; runs packed into buffers
  size_t maxbuf = o.max_buffer_bytes ? o.max_buffer_bytes : std::min<size_t>(Device::get().max_buffer_length(), 16ull << 30);
  maxbuf = rdown(maxbuf, kPage);
  std::vector<std::pair<uint64_t, uint64_t>> buf_ranges;   // arena [start, end)
  uint64_t pos = 0;
  for (auto& r : runs) {
    uint64_t span = rup(r.hi, kPage) - rdown(r.lo, kPage);
    if (span > maxbuf) { fprintf(stderr, "weights: run of %llu B exceeds max buffer\n", (unsigned long long)span); abort(); }
    if (buf_ranges.empty() || pos + span - buf_ranges.back().first > maxbuf) buf_ranges.push_back({pos, pos});
    r.arena_off = pos; r.buf = (int)buf_ranges.size() - 1;
    pos += span;
    buf_ranges.back().second = pos;
  }
  arena_size_ = std::max<uint64_t>(pos, kPage);
  arena_ = (char*)mmap(nullptr, arena_size_, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (arena_ == MAP_FAILED) { perror("weights mmap"); abort(); }

  // 4. parallel reads
  auto req = std::make_shared<IoRequest>();
  for (auto& r : runs) {
    int fd = o.nocache ? g.fd_nocache(r.split) : g.fd(r.split);
    uint64_t flo = rdown(r.lo, kPage), fhi = std::min(rup(r.hi, kPage), g.file_size(r.split));
    for (uint64_t off = flo; off < fhi; off += o.read_chunk) {
      uint64_t len = std::min<uint64_t>(o.read_chunk, fhi - off);
      req->chunks.push_back({fd, off, (uint32_t)len, arena_ + r.arena_off + (off - flo), 0});
    }
  }
  std::mutex mu; std::condition_variable cv; bool done = false;
  req->on_done = [&](IoRequest&) { std::lock_guard<std::mutex> lk(mu); done = true; cv.notify_all(); };
  {
    IoPool pool(o.threads, kPage);
    if (req->chunks.empty()) done = true; else pool.submit(req);
    wait_request(req, mu, cv, done);
  }
  if (req->errors) { fprintf(stderr, "weights: %d read errors\n", req->errors.load()); abort(); }
  bytes_read_ = req->bytes;

  // 5. wrap + views
  for (auto& br : buf_ranges) {
    Buffer b = Device::get().wrap(arena_ + br.first, rup(br.second - br.first, kPage), "weights");
    bufs_.push_back(b);
  }
  for (auto& r : runs) {
    uint64_t flo = rdown(r.lo, kPage);
    for (size_t i = r.t0; i < r.t1; i++) {
      const TensorInfo* t = sel[i];
      WTensor w;
      w.name = t->name; w.type = t->type; w.n_dims = t->n_dims; w.nbytes = t->nbytes; w.info = t;
      for (int d = 0; d < 4; d++) w.ne[d] = t->ne[d];
      uint64_t aoff = r.arena_off + (t->offset - flo);
      w.buf = bufs_[r.buf].sub(aoff - buf_ranges[r.buf].first, t->nbytes);
      w.host = arena_ + aoff;
      bytes_ += t->nbytes;
      index_[w.name] = tensors_.size();
      tensors_.push_back(std::move(w));
    }
  }
  if (o.mlock) locked_ = mlock(arena_, arena_size_) == 0;
  load_ms_ = timer.ms();
}

Weights::~Weights() {
  for (auto& b : bufs_) Device::get().release(b);
  if (arena_) { if (locked_) munlock(arena_, arena_size_); munmap(arena_, arena_size_); }
}

const WTensor* Weights::find(const std::string& n) const { auto it = index_.find(n); return it == index_.end() ? nullptr : &tensors_[it->second]; }
const WTensor& Weights::at(const std::string& n) const {
  auto* w = find(n);
  if (!w) { fprintf(stderr, "weights: tensor '%s' not resident\n", n.c_str()); abort(); }
  return *w;
}

size_t Weights::repack(const RepackFn& fn, bool release_src) {
  size_t n = 0;
  for (auto& w : tensors_) {
    WTensor out = w;
    if (!fn(w, out)) continue;
    if (release_src && !w.repacked && (char*)w.host >= arena_ && (char*)w.host < arena_ + arena_size_) {
      uint64_t a = rup((uint64_t)w.host, kPage), b = rdown((uint64_t)w.host + w.nbytes, kPage);
      if (b > a) madvise((void*)a, b - a, MADV_FREE_REUSABLE);
    }
    out.repacked = true; out.info = w.info; out.name = w.name;
    w = out;
    n++;
  }
  return n;
}

// ---------------- TokenEmbd ----------------
TokenEmbd::TokenEmbd(const Gguf& g, const Weights* w, int threads) {
  t_ = &g.at("token_embd.weight");
  row_bytes_ = t_->row_bytes(); n_rows_ = t_->ne[1]; type_ = t_->type;
  resident_ = w ? w->find("token_embd.weight") : nullptr;
  fd_ = g.fd(t_->split);
  if (!resident_) pool_ = std::make_unique<IoPool>(threads, kPage);
}
TokenEmbd::~TokenEmbd() = default;

void TokenEmbd::fetch(const int32_t* toks, size_t n, void* dst) {
  char* d = (char*)dst;
  for (size_t i = 0; i < n; i++)
    if (toks[i] < 0 || toks[i] >= n_rows_) { fprintf(stderr, "token_embd: token %d out of range\n", toks[i]); abort(); }
  if (resident_) {
    for (size_t i = 0; i < n; i++) memcpy(d + i * row_bytes_, (char*)resident_->host + (uint64_t)toks[i] * row_bytes_, row_bytes_);
    return;
  }
  if (n <= 2) {
    for (size_t i = 0; i < n; i++) pread_full(fd_, d + i * row_bytes_, row_bytes_, t_->offset + (uint64_t)toks[i] * row_bytes_);
    return;
  }
  auto req = std::make_shared<IoRequest>();
  for (size_t i = 0; i < n; i++)
    req->chunks.push_back({fd_, t_->offset + (uint64_t)toks[i] * row_bytes_, (uint32_t)row_bytes_, d + i * row_bytes_, 0});
  std::mutex mu; std::condition_variable cv; bool done = false;
  req->on_done = [&](IoRequest&) { std::lock_guard<std::mutex> lk(mu); done = true; cv.notify_all(); };
  pool_->submit(req);
  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&] { return done; });
}

}  // namespace ms
