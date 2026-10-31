#pragma once
// SSD-streamed MoE expert cache.
//
// Unit = one (layer, expert): 3 byte ranges (gate, up, down), expert e's matrix being the e-th contiguous slab
// (nbytes/n_experts) of each 3D *_exps tensor. Slots live in a page-aligned anonymous arena wrapped as no-copy shared
// MTLBuffers. Layers whose 3 matrices have the same aligned sizes form a SIZE CLASS (UD-IQ1_S: 3 classes:
// 3xIQ1_S 6.94 MB, 2xIQ1_S+IQ2_XXS 7.68 MB, 3xIQ2_XXS 9.16 MB); each class has its own slot pool, so no byte is
// wasted on padding small experts to the largest size. Inside a slot every matrix sits at (file offset mod 16 KiB)
// so 16 KiB-aligned reads land directly in place (no copy, no bounce).
//
// Concurrency model: one mutex for metadata (a few µs per ensure()), an IoPool of preads (F_NOCACHE) doing the I/O.
// A slot is never evicted while (a) it is loading, (b) a lease holds it (refcount; leases are released explicitly or
// by a command-buffer completion handler, i.e. after the GPU is done with it), or (c) it is pinned.
//
// GPU sync: every lease that has to wait for I/O gets a monotonically increasing ticket. io_event() (MTLSharedEvent)
// is signaled with the highest ticket whose lease AND all earlier leases are complete, so a command buffer can
// `encodeWaitForEvent:io_event value:lease.ticket` and be committed before the reads finish.
#include "core/metal.hpp"
#include "gguf/gguf.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ms {

class IoPool;
struct IoRequest;

struct ExpertMat { int fd = -1; uint64_t base = 0, stride = 0, file_size = 0; uint32_t len = 0; GType type = GType::F32; };
struct ExpertLayerLayout { ExpertMat mat[3]; };   // 0 = gate (w1), 1 = up (w3), 2 = down (w2)
struct ExpertLayout {
  int n_layers = 0, n_experts = 0;
  std::vector<ExpertLayerLayout> layers;
  static ExpertLayout from_gguf(const Gguf& g, bool nocache = true);
  uint64_t expert_bytes(int layer) const { auto& l = layers[layer]; return (uint64_t)l.mat[0].len + l.mat[1].len + l.mat[2].len; }
};

enum class PolicyKind { LRU, LFU, Static };
const char* policy_name(PolicyKind k);

// Pluggable eviction policy. All calls happen under the cache mutex. Keys are layer*n_experts+expert.
// The cache itself handles Empty slots, pinning and in-use protection; victim() only ranks candidates.
class EvictionPolicy {
 public:
  virtual ~EvictionPolicy() = default;
  virtual void init(const std::vector<int>& slot_class, int n_classes, int n_keys) = 0;
  virtual void on_insert(int slot, uint32_t key) = 0;   // slot now holds key (demand load or prefetch)
  virtual void on_access(int slot, uint32_t key) = 0;   // demand access (hit or miss)
  virtual void on_evict(int slot, uint32_t key) = 0;
  // pick a victim of class cls with evictable(slot) == true; -1 if none
  virtual int victim(int cls, bool (*evictable)(const void* ctx, int slot), const void* ctx) = 0;
};
std::unique_ptr<EvictionPolicy> make_policy(PolicyKind k, double lfu_decay, uint64_t lfu_period, int lfu_sample = 64);

struct ExpertCacheOptions {
  size_t arena_bytes = 4ull << 30;  // production ~38-42 GiB (see recommend_arena_bytes); tests keep it small
  int io_threads = 8;               // measured: 8 = same 6-miss latency as 16 at half the CPU (339 vs 766 µs/miss)
  size_t read_chunk = 1 << 20;      // split each matrix read into chunks (0 = one pread per matrix)
  uint32_t align = 16384;           // round file ranges out to this (power of 2 <= 16384; 1 = exact byte ranges)
  PolicyKind policy = PolicyKind::LRU;
  std::string stats_path;           // routing statistics file (Static policy; also splits the arena across classes)
  double pin_fraction = 0.85;       // Static: fraction of each class pool pinned to the most frequent experts
  bool preload_pinned = true;       // Static: read the pinned set at construction
  double lfu_decay = 0.5;           // LFU: multiply all counts by this every lfu_period demand accesses (aging)
  uint64_t lfu_period = 1 << 14;
  int lfu_sample = 64;              // LFU: victim = least frequent of N random candidates (0 = exact full scan)
  int ring_slots = 0;               // bulk-stream staging slots (0 = stream() unavailable); 12-24 recommended
  size_t max_buffer_bytes = 0;      // 0 = min(device maxBufferLength, 8 GiB)
  bool dry_io = false;              // policy simulation: no reads, slots become ready immediately
  bool prefault = true;             // touch every arena page at startup (parallel): no zero-fill faults on the miss
                                    // path (~2 µs per 16 KiB page = ~1 ms per expert otherwise); env MACSEEK_ARENA_PREFAULT
  bool wire = false;                // mlock the arena at startup (implies prefault; never compressed/paged out);
                                    // env MACSEEK_ARENA_MLOCK=1. Recommended in production once sized by memguard
  std::shared_ptr<EvictionPolicy> custom_policy;   // overrides `policy` (Static pinning still uses stats_path)
  static ExpertCacheOptions from_env();   // MACSEEK_ARENA_GB, MACSEEK_IO_THREADS, MACSEEK_READ_CHUNK_KB,
                                          // MACSEEK_IO_ALIGN, MACSEEK_POLICY=lru|lfu|static, MACSEEK_ROUTING_STATS,
                                          // MACSEEK_RING_SLOTS
};

struct ExpertView {
  Buffer mat[3];         // gate, up, down: bind mat[k].buf at mat[k].off (size = matrix bytes)
  uint64_t gpu[3] = {};  // GPU virtual addresses (gpuAddress + off) for argument-buffer / pointer-table kernels
  void* host[3] = {};
  GType type[3] = {};
  int slot = -1;
};

struct LayerStats {
  uint64_t hits = 0, misses = 0, inflight_hits = 0;   // inflight_hits: present but still loading (counted in hits)
  uint64_t prefetch_issued = 0, prefetch_used = 0, prefetch_wasted = 0, evictions = 0;
  uint64_t bytes_read = 0, streamed = 0, stream_bytes = 0;
  double stall_ms = 0;        // time CPU waits (ExpertLease::wait / stream wait_ready) blocked on I/O
  LayerStats& operator+=(const LayerStats& o);
  double hit_rate() const { return hits + misses ? (double)hits / (hits + misses) : 0; }
};

class ExpertCache;

// Result of ensure(): views in the order of the requested ids, holding a reference on each slot.
class ExpertLease {
 public:
  ExpertLease() = default;
  ExpertLease(ExpertLease&& o) noexcept { *this = std::move(o); }
  ExpertLease& operator=(ExpertLease&& o) noexcept;
  ExpertLease(const ExpertLease&) = delete; ExpertLease& operator=(const ExpertLease&) = delete;
  ~ExpertLease() { release(); }

  std::vector<ExpertView> views;
  int layer = -1;
  uint64_t ticket = 0;            // 0 = everything was already resident; else GPU waits io_event >= ticket
  bool ready() const;             // all slots loaded
  void wait();                    // CPU: block until all slots loaded (adds to the layer's stall_ms)
  void release();                 // drop the slot references now (GPU must be done with them)
  void release_after(void* cmdbuf);   // drop when this id<MTLCommandBuffer> completes (addCompletedHandler)
  // GPU-side wait helper: [cmdbuf encodeWaitForEvent:io_event value:ticket] (no-op if ticket == 0)
  void encode_wait(void* cmdbuf) const;

 private:
  friend class ExpertCache;
  ExpertCache* cache_ = nullptr;
  std::vector<int> slots_;
};

// Bulk stream (prefill): every expert of one layer needed by a chunk. Cached experts are served from the cache
// (one lease), the misses are streamed in file order through a ring of `ring_slots` staging slots while the GPU
// consumes them. Protocol per ring item i (in order):  wait ready(i) -> use -> consumed(i).
//   CPU:  wait_ready(i) ... consumed(i)
//   GPU:  encode_wait_ready(cb, i) ... kernels ... encode_consumed(cb, i)   (between compute encoders)
// Values are contiguous, so waiting/signalling only the last item of a group of g items also works.
// Do not mix CPU consumed() and GPU encode_consumed() within one cache.
class ExpertStream {
 public:
  struct Item { int expert; ExpertView view; bool cached; uint64_t seq; };   // seq: ring sequence (cached: 0)
  ~ExpertStream();
  int layer() const { return layer_; }
  const std::vector<Item>& items() const { return items_; }   // cached items first, then ring items (file order)
  size_t n_cached() const { return n_cached_; }
  ExpertLease& cached() { return lease_; }                     // lease for the cached items
  void wait_ready(size_t i);                                   // CPU wait for item i (cached: lease.wait())
  void consumed(size_t i);                                     // CPU: ring slot of item i may be reused
  void encode_wait_ready(void* cmdbuf, size_t i) const;
  void encode_consumed(void* cmdbuf, size_t i) const;
  void* ready_event() const;    // id<MTLSharedEvent>: value seq+1 => item ready
  void* consume_event() const;  // id<MTLSharedEvent>: value seq+1 => item consumed
 private:
  friend class ExpertCache;
  ExpertCache* cache_ = nullptr;
  int layer_ = -1;
  size_t n_cached_ = 0;
  std::vector<Item> items_;
  ExpertLease lease_;
};

class ExpertCache {
 public:
  ExpertCache(const ExpertLayout& layout, const ExpertCacheOptions& o = ExpertCacheOptions::from_env());
  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete; ExpertCache& operator=(const ExpertCache&) = delete;

  // Demand: make experts resident (misses read at high priority, in parallel). Returns immediately; use
  // lease.wait() (CPU) or lease.ticket / encode_wait (GPU). Blocks only if every slot of a class is in use.
  ExpertLease ensure(int layer, const int* ids, int n);
  ExpertLease ensure(int layer, const std::vector<int>& ids) { return ensure(layer, ids.data(), (int)ids.size()); }
  // Prefetch: low-priority loads for absent experts (skipped if no evictable slot). Returns number issued.
  int prefetch(int layer, const int* ids, int n);
  // Pin/unpin: pinned slots are never evicted (pin loads absent experts at low priority). Returns slots pinned.
  int pin(int layer, const int* ids, int n);
  void unpin(int layer, const int* ids, int n);
  bool contains(int layer, int expert) const;       // present (ready or loading)
  void drain();                                     // wait until no load is in flight

  std::unique_ptr<ExpertStream> stream(int layer, const int* ids, int n);

  // stats
  LayerStats layer_stats(int layer) const;
  LayerStats total_stats() const;
  void reset_stats();
  std::string stats_report() const;                 // per-layer table + totals
  // routing statistics: demand access counts per (layer, expert); file = n_layers lines of n_experts counts
  bool write_routing_stats(const std::string& path) const;
  static bool read_routing_stats(const std::string& path, int n_layers, int n_experts, std::vector<double>& out);

  // geometry
  struct ClassInfo { uint64_t slot_bytes; uint64_t region[3]; int n_slots; std::vector<int> layers; };
  const std::vector<ClassInfo>& classes() const { return classes_; }
  int layer_class(int layer) const { return layer_class_[layer]; }
  int n_slots() const { return (int)slots_.size(); }
  int n_pinned() const;
  uint64_t arena_bytes() const { return arena_bytes_; }
  std::vector<Buffer> buffers() const;              // all arena + ring MTLBuffers (residency set)
  void* io_event() const;                           // id<MTLSharedEvent>
  const ExpertLayout& layout() const { return L_; }
  const ExpertCacheOptions& options() const { return o_; }
  std::string describe() const;

 private:
  friend class ExpertLease;
  friend class ExpertStream;
  struct Impl;
  enum State : uint8_t { Empty, Loading, Ready };
  struct Slot {
    int cls = 0; uint32_t key = ~0u; State state = Empty;
    int ref = 0; bool pinned = false, prefetched = false, used = false;
    char* host = nullptr; Buffer buf; uint64_t gpu = 0;   // slot base
    std::vector<uint64_t> waiters;                       // lease tickets waiting for this slot
    IoRequest* req = nullptr;                            // in-flight request (for promotion)
    uint32_t data_off[3] = {};                           // matrix data offset within the slot
  };
  int take_slot_locked(int cls, bool block, std::unique_lock<std::mutex>& lk);
  void evict_locked(int s);
  void load_locked(int s, uint32_t key, int prio);
  void flush_submits();             // submit requests queued by load_locked (call after dropping mu_)
  void on_loaded(int s, uint64_t bytes);
  ExpertView view_locked(int s) const;
  void release_slots(const std::vector<int>& slots);
  void complete_tickets_locked();
  void stream_feeder();
  uint32_t key(int layer, int e) const { return (uint32_t)(layer * L_.n_experts + e); }

  ExpertLayout L_;
  ExpertCacheOptions o_;
  std::vector<ClassInfo> classes_;
  std::vector<int> layer_class_;
  std::vector<Slot> slots_;
  std::vector<int> map_;                  // key -> slot (-1)
  std::vector<std::vector<int>> free_;    // per class free (Empty) slots
  std::vector<LayerStats> stats_;
  std::vector<double> access_;            // demand access counts per key (routing stats output)
  mutable std::mutex mu_;
  std::condition_variable cv_;            // slot became ready / released
  int inflight_ = 0;
  std::vector<std::shared_ptr<IoRequest>> to_submit_;   // built under mu_, submitted after unlock (one wake-up)
  uint64_t arena_bytes_ = 0;
  std::unique_ptr<Impl> p_;
  std::unique_ptr<IoPool> pool_;
};

}  // namespace ms
