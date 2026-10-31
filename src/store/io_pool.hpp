#pragma once
// Thread pool for synchronous preads (one outstanding pread per worker => queue depth = threads).
// A request is a list of chunks (fd, offset, len, dst). Workers drain the FRONT request first (lowest latency for
// the oldest demand request), two priority levels (0 = demand, 1 = prefetch), low-priority requests can be promoted.
// dst == nullptr means "read into the worker's private scratch buffer" (then on_chunk must consume the bytes).
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ms {

struct IoChunk {
  int fd = -1;
  uint64_t off = 0;
  uint32_t len = 0;
  char* dst = nullptr;     // nullptr => worker scratch (len <= scratch size)
  uint32_t tag = 0;        // caller data (e.g. index of the first row served by this chunk)
};

struct IoRequest {
  std::vector<IoChunk> chunks;
  // optional, runs on the worker right after chunk i was read (data = dst or scratch, got = bytes read)
  std::function<void(IoRequest&, size_t i, const char* data, long long got)> on_chunk;
  // runs once on the worker that completed the last chunk
  std::function<void(IoRequest&)> on_done;
  int priority = 0;
  void* user = nullptr;
  // ---- set by the pool ----
  std::atomic<size_t> remaining{0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<int> errors{0};
  size_t next = 0;          // next chunk to hand out (pool mutex)
  bool queued = false;      // still in a pool queue (pool mutex)
};
using IoRequestPtr = std::shared_ptr<IoRequest>;

class IoPool {
 public:
  explicit IoPool(int threads, size_t scratch_bytes = 64 << 10);
  ~IoPool();
  IoPool(const IoPool&) = delete; IoPool& operator=(const IoPool&) = delete;
  void submit(IoRequestPtr r);       // r->priority selects the queue
  void submit(std::vector<IoRequestPtr>& rs);   // batch: one lock + one wake-up for all (cleared on return)
  void promote(IoRequest* r);        // move a queued low-priority request to the demand queue (no-op otherwise)
  int threads() const { return (int)workers_.size(); }
  uint64_t reads() const { return reads_.load(); }
  uint64_t bytes() const { return bytes_.load(); }

 private:
  void run(int id);
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<IoRequestPtr> q_[2];
  bool stop_ = false;
  int idle_ = 0;
  size_t scratch_;
  std::vector<std::thread> workers_;
  std::atomic<uint64_t> reads_{0}, bytes_{0};
};

}  // namespace ms
