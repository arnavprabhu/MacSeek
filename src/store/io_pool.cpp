#include "store/io_pool.hpp"

#include "gguf/gguf.hpp"

#include <pthread.h>
#include <sys/mman.h>

#include <cstdio>
#include <cstdlib>

namespace ms {

IoPool::IoPool(int threads, size_t scratch_bytes) : scratch_(scratch_bytes) {
  if (threads < 1) threads = 1;
  for (int i = 0; i < threads; i++) workers_.emplace_back([this, i] { run(i); });
}

IoPool::~IoPool() {
  { std::lock_guard<std::mutex> g(mu_); stop_ = true; }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void IoPool::submit(IoRequestPtr r) {
  r->remaining.store(r->chunks.size());
  if (r->chunks.empty()) { if (r->on_done) r->on_done(*r); return; }
  int p = r->priority ? 1 : 0;
  size_t n = r->chunks.size();
  {
    std::lock_guard<std::mutex> g(mu_);
    r->next = 0; r->queued = true;
    q_[p].push_back(std::move(r));
  }
  (void)n;
  cv_.notify_one();
}

void IoPool::submit(std::vector<IoRequestPtr>& rs) {
  size_t chunks = 0;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& r : rs) {
      r->remaining.store(r->chunks.size());
      r->next = 0; r->queued = true;
      chunks += r->chunks.size();
      q_[r->priority ? 1 : 0].push_back(r);
    }
  }
  if (chunks) cv_.notify_one();   // chain wake-up: woken workers wake the next one while work remains
  rs.clear();
}

void IoPool::promote(IoRequest* r) {
  std::lock_guard<std::mutex> g(mu_);
  if (!r->queued || r->priority == 0) return;
  for (auto it = q_[1].begin(); it != q_[1].end(); ++it)
    if (it->get() == r) { IoRequestPtr p = *it; q_[1].erase(it); p->priority = 0; q_[0].push_back(std::move(p)); return; }
}

void IoPool::run(int id) {
  char name[32]; snprintf(name, sizeof name, "ms.io.%d", id);
  pthread_setname_np(name);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  // page-aligned scratch so uncached reads can DMA straight into it
  size_t sb = (scratch_ + 16383) & ~(size_t)16383;
  char* scratch = (char*)mmap(nullptr, sb, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (scratch == MAP_FAILED) { perror("io scratch"); abort(); }
  for (;;) {
    IoRequestPtr r; size_t idx; bool wake;
    {
      std::unique_lock<std::mutex> lk(mu_);
      idle_++;
      cv_.wait(lk, [&] { return stop_ || !q_[0].empty() || !q_[1].empty(); });
      idle_--;
      if (stop_ && q_[0].empty() && q_[1].empty()) { cv_.notify_all(); break; }
      auto& q = !q_[0].empty() ? q_[0] : q_[1];
      r = q.front();
      idx = r->next++;
      if (r->next == r->chunks.size()) { r->queued = false; q.pop_front(); }
      wake = idle_ > 0 && (!q_[0].empty() || !q_[1].empty());
    }
    if (wake) cv_.notify_one();   // pass the wake-up on (keeps the submitter's critical path at one wake)
    const IoChunk& c = r->chunks[idx];
    char* dst = c.dst ? c.dst : scratch;
    if (!c.dst && c.len > sb) { fprintf(stderr, "io: chunk %u > scratch %zu\n", c.len, sb); abort(); }
    long long got = pread_full(c.fd, dst, c.len, c.off);
    if (got < 0) { r->errors++; perror("io pread"); }
    else { r->bytes += (uint64_t)got; bytes_ += (uint64_t)got; }
    reads_++;
    if (r->on_chunk) r->on_chunk(*r, idx, dst, got);
    if (r->remaining.fetch_sub(1) == 1 && r->on_done) r->on_done(*r);
  }
  munmap(scratch, sb);
}

}  // namespace ms
