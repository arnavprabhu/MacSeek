#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "store/experts.hpp"

#include "core/env.hpp"
#include "core/timer.hpp"
#include "store/io_pool.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <thread>
#include <sstream>

namespace ms {

static constexpr uint64_t kPage = 16384;
static inline uint64_t rdown(uint64_t x, uint64_t a) { return x / a * a; }
static inline uint64_t rup(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

LayerStats& LayerStats::operator+=(const LayerStats& o) {
  hits += o.hits; misses += o.misses; inflight_hits += o.inflight_hits; prefetch_issued += o.prefetch_issued;
  prefetch_used += o.prefetch_used; prefetch_wasted += o.prefetch_wasted; evictions += o.evictions;
  bytes_read += o.bytes_read; streamed += o.streamed; stream_bytes += o.stream_bytes; stall_ms += o.stall_ms;
  return *this;
}

ExpertLayout ExpertLayout::from_gguf(const Gguf& g, bool nocache) {
  ExpertLayout L;
  L.n_layers = (int)g.get_i64("deepseek41.block_count", 0);
  L.n_experts = (int)g.get_i64("deepseek41.expert_count", 0);
  static const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
  for (int l = 0; l < L.n_layers; l++) {
    ExpertLayerLayout ll;
    for (int k = 0; k < 3; k++) {
      const TensorInfo& t = g.at("blk." + std::to_string(l) + "." + names[k] + ".weight");
      if (t.ne[2] != L.n_experts || t.nbytes % L.n_experts) { fprintf(stderr, "experts: bad tensor %s\n", t.name.c_str()); abort(); }
      ExpertMat& m = ll.mat[k];
      m.fd = nocache ? g.fd_nocache(t.split) : g.fd(t.split);
      m.base = t.offset; m.stride = t.nbytes / L.n_experts; m.len = (uint32_t)m.stride; m.type = t.type;
      m.file_size = g.file_size(t.split);
    }
    L.layers.push_back(ll);
  }
  return L;
}

ExpertCacheOptions ExpertCacheOptions::from_env() {
  ExpertCacheOptions o;
  double gb = env_f("MACSEEK_ARENA_GB", 0);
  if (gb > 0) o.arena_bytes = (size_t)(gb * (1ull << 30));
  o.io_threads = (int)env_int("MACSEEK_IO_THREADS", o.io_threads);
  long ck = env_int("MACSEEK_READ_CHUNK_KB", -1);
  if (ck >= 0) o.read_chunk = (size_t)ck << 10;
  o.align = (uint32_t)env_int("MACSEEK_IO_ALIGN", o.align);
  std::string p = env_str("MACSEEK_POLICY", "lru");
  o.policy = p == "lfu" ? PolicyKind::LFU : p == "static" ? PolicyKind::Static : PolicyKind::LRU;
  o.stats_path = env_str("MACSEEK_ROUTING_STATS", "");
  o.ring_slots = (int)env_int("MACSEEK_RING_SLOTS", o.ring_slots);
  o.prefault = env_int("MACSEEK_ARENA_PREFAULT", 1) != 0;
  o.wire = env_int("MACSEEK_ARENA_MLOCK", 0) != 0;
  return o;
}

struct ExpertCache::Impl {
  std::shared_ptr<EvictionPolicy> policy;
  id<MTLSharedEvent> io_ev, ready_ev, consume_ev;
  std::vector<std::pair<char*, size_t>> maps;
  std::vector<Buffer> bufs;
  // lease tickets
  uint64_t next_ticket = 1, base = 1, done = 0;
  std::deque<int> pending;          // remaining not-ready slots for tickets base, base+1, ...
  std::vector<uint8_t> pinset;      // Static: keys that get pinned when loaded
  std::vector<int> pin_quota, pin_count;
  // bulk-stream ring
  int K = 0;
  uint64_t ring_slot_bytes = 0;
  std::vector<char*> ring_host; std::vector<Buffer> ring_buf; std::vector<uint64_t> ring_gpu;
  std::mutex rmu; std::condition_variable rcv;
  struct Job { uint64_t seq; int layer, expert; };
  std::deque<Job> jobs;
  uint64_t ring_next = 0, ring_done = 0;   // seqs < ring_done are loaded
  std::vector<uint8_t> ring_flag;
  bool stop = false;
  std::thread feeder;
};

ExpertCache::ExpertCache(const ExpertLayout& layout, const ExpertCacheOptions& o) : L_(layout), o_(o), p_(new Impl) {
  if (o_.align == 0) o_.align = 1;
  if (o_.align > kPage || (o_.align & (o_.align - 1))) { fprintf(stderr, "experts: align must be a power of 2 <= 16384\n"); abort(); }
  const int NL = L_.n_layers, NE = L_.n_experts;
  // ---- size classes: exact per-layer regions (max over experts of the 16K-rounded span at offset mod 16K) ----
  std::map<std::vector<uint64_t>, int> cls_of;
  layer_class_.assign(NL, 0);
  for (int l = 0; l < NL; l++) {
    std::vector<uint64_t> key;
    for (int k = 0; k < 3; k++) {
      const ExpertMat& m = L_.layers[l].mat[k];
      uint64_t r = 0;
      for (int e = 0; e < NE; e++) r = std::max(r, rup((m.base + (uint64_t)e * m.stride) % kPage + m.len, kPage));
      key.push_back(r);
    }
    for (int k = 0; k < 3; k++) key.push_back((uint64_t)L_.layers[l].mat[k].type);
    auto it = cls_of.find(key);
    if (it == cls_of.end()) {
      ClassInfo ci; ci.region[0] = key[0]; ci.region[1] = key[1]; ci.region[2] = key[2];
      ci.slot_bytes = key[0] + key[1] + key[2]; ci.n_slots = 0;
      it = cls_of.emplace(key, (int)classes_.size()).first;
      classes_.push_back(ci);
    }
    layer_class_[l] = it->second;
    classes_[it->second].layers.push_back(l);
  }
  // ---- split the arena across classes ----
  const int C = (int)classes_.size();
  std::vector<double> rstats;
  bool have_stats = !o_.stats_path.empty() && read_routing_stats(o_.stats_path, NL, NE, rstats);
  if (!o_.stats_path.empty() && !have_stats) fprintf(stderr, "experts: could not read routing stats %s\n", o_.stats_path.c_str());
  uint64_t per_layer_bytes = 0;
  for (int l = 0; l < NL; l++) per_layer_bytes += classes_[layer_class_[l]].slot_bytes;
  if (!have_stats) {
    // equal slots per layer
    uint64_t spl = o_.arena_bytes / per_layer_bytes;
    for (auto& c : classes_) c.n_slots = (int)std::min<uint64_t>(spl * c.layers.size(), (uint64_t)NE * c.layers.size());
  } else {
    // floor of 16 slots per layer, then greedy by (count / bytes) over all (layer, expert)
    uint64_t budget = o_.arena_bytes;
    for (auto& c : classes_) {
      c.n_slots = (int)std::min<uint64_t>(16 * c.layers.size(), budget / c.slot_bytes);
      budget -= (uint64_t)c.n_slots * c.slot_bytes;
    }
    std::vector<std::pair<double, uint32_t>> rank;
    for (int l = 0; l < NL; l++)
      for (int e = 0; e < NE; e++) rank.push_back({rstats[(size_t)l * NE + e] / classes_[layer_class_[l]].slot_bytes, (uint32_t)(l * NE + e)});
    std::sort(rank.begin(), rank.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<int> taken(C, 0);
    for (auto& r : rank) {
      int c = layer_class_[r.second / NE];
      if (classes_[c].slot_bytes > budget) continue;
      if (taken[c] + classes_[c].n_slots >= (int)(NE * classes_[c].layers.size())) continue;
      taken[c]++; budget -= classes_[c].slot_bytes;
    }
    for (int c = 0; c < C; c++) classes_[c].n_slots += taken[c];
  }
  for (int c = 0; c < C; c++)
    if (classes_[c].n_slots < 12) {
      fprintf(stderr, "experts: arena %.2f GiB too small: class %d gets %d slots (need >= 12)\n", o_.arena_bytes / 1073741824.0, c, classes_[c].n_slots);
      abort();
    }
  // ---- allocate slots ----
  Device& dev = Device::get();
  size_t maxbuf = o_.max_buffer_bytes ? o_.max_buffer_bytes : std::min<size_t>(dev.max_buffer_length(), 8ull << 30);
  for (int c = 0; c < C; c++) {
    ClassInfo& ci = classes_[c];
    uint64_t per_buf = std::max<uint64_t>(1, maxbuf / ci.slot_bytes);
    for (int s0 = 0; s0 < ci.n_slots; s0 += (int)per_buf) {
      int ns = (int)std::min<uint64_t>(per_buf, ci.n_slots - s0);
      size_t bytes = (size_t)ns * ci.slot_bytes;
      char* mem = nullptr; Buffer b; uint64_t gbase = 0;
      if (!o_.dry_io) {   // dry_io (policy simulation): no memory at all
        mem = (char*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (mem == MAP_FAILED) { perror("experts mmap"); abort(); }
        p_->maps.push_back({mem, bytes});
        b = dev.wrap(mem, bytes, "expert_slots");
        p_->bufs.push_back(b);
        gbase = [(__bridge id<MTLBuffer>)b.buf gpuAddress];
      }
      for (int i = 0; i < ns; i++) {
        Slot s; s.cls = c; s.host = mem + (size_t)i * ci.slot_bytes;
        s.buf = b.sub((size_t)i * ci.slot_bytes, ci.slot_bytes); s.gpu = gbase + (uint64_t)i * ci.slot_bytes;
        slots_.push_back(std::move(s));
      }
      arena_bytes_ += bytes;
    }
  }
  bool wired = false;
  if (o_.wire && !o_.dry_io) {
    wired = true;
    for (auto& m : p_->maps) if (mlock(m.first, m.second) != 0) { perror("experts: mlock failed (continuing unwired)"); wired = false; break; }
  }
  if (o_.prefault && !o_.dry_io && !wired) {   // parallel first touch of every page
    std::vector<std::pair<char*, size_t>> parts;
    for (auto& m : p_->maps) for (size_t off = 0; off < m.second; off += 256ull << 20) parts.push_back({m.first + off, std::min<size_t>(256ull << 20, m.second - off)});
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; t++) th.emplace_back([&] {
      for (size_t i; (i = next++) < parts.size();) for (size_t o = 0; o < parts[i].second; o += kPage) parts[i].first[o] = 0;
    });
    for (auto& t : th) t.join();
  }
  free_.assign(C, {});
  for (int s = (int)slots_.size() - 1; s >= 0; s--) free_[slots_[s].cls].push_back(s);
  map_.assign((size_t)NL * NE, -1);
  stats_.assign(NL, LayerStats{});
  access_.assign((size_t)NL * NE, 0.0);
  id<MTLDevice> mdev = (__bridge id<MTLDevice>)dev.device();
  p_->io_ev = [mdev newSharedEvent];
  p_->io_ev.signaledValue = 0;
  // ---- policy ----
  if (o_.custom_policy) p_->policy = o_.custom_policy;
  else p_->policy = std::shared_ptr<EvictionPolicy>(make_policy(o_.policy, o_.lfu_decay, o_.lfu_period, o_.lfu_sample).release());
  std::vector<int> sc(slots_.size());
  for (size_t s = 0; s < slots_.size(); s++) sc[s] = slots_[s].cls;
  p_->policy->init(sc, C, NL * NE);
  // ---- I/O ----
  pool_ = std::make_unique<IoPool>(o_.io_threads, kPage);
  // ---- ring ----
  if (o_.ring_slots > 0 && !o_.dry_io) {
    p_->K = o_.ring_slots;
    for (auto& c : classes_) p_->ring_slot_bytes = std::max(p_->ring_slot_bytes, c.slot_bytes);
    size_t bytes = (size_t)p_->K * p_->ring_slot_bytes;
    char* mem = (char*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (mem == MAP_FAILED) { perror("ring mmap"); abort(); }
    p_->maps.push_back({mem, bytes});
    Buffer b = dev.wrap(mem, bytes, "expert_ring");
    p_->bufs.push_back(b);
    uint64_t gbase = [(__bridge id<MTLBuffer>)b.buf gpuAddress];
    for (int i = 0; i < p_->K; i++) {
      p_->ring_host.push_back(mem + (size_t)i * p_->ring_slot_bytes);
      p_->ring_buf.push_back(b.sub((size_t)i * p_->ring_slot_bytes, p_->ring_slot_bytes));
      p_->ring_gpu.push_back(gbase + (uint64_t)i * p_->ring_slot_bytes);
    }
    p_->ring_flag.assign(p_->K, 0);
    p_->ready_ev = [mdev newSharedEvent]; p_->ready_ev.signaledValue = 0;
    p_->consume_ev = [mdev newSharedEvent]; p_->consume_ev.signaledValue = 0;
    p_->feeder = std::thread([this] { stream_feeder(); });
  }
  // ---- static frequency-pinned set ----
  if (o_.policy == PolicyKind::Static && have_stats) {
    p_->pinset.assign((size_t)NL * NE, 0);
    p_->pin_quota.assign(C, 0); p_->pin_count.assign(C, 0);
    for (int c = 0; c < C; c++) {
      int quota = (int)(o_.pin_fraction * classes_[c].n_slots);
      p_->pin_quota[c] = quota;
      std::vector<std::pair<double, uint32_t>> r;
      for (int l : classes_[c].layers) for (int e = 0; e < NE; e++) if (rstats[(size_t)l * NE + e] > 0) r.push_back({rstats[(size_t)l * NE + e], key(l, e)});
      std::sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.first > b.first; });
      for (int i = 0; i < quota && i < (int)r.size(); i++) p_->pinset[r[i].second] = 1;
    }
    if (o_.preload_pinned) {
      for (int l = 0; l < NL; l++) {
        std::vector<int> ids;
        for (int e = 0; e < NE; e++) if (p_->pinset[key(l, e)]) ids.push_back(e);
        if (!ids.empty()) pin(l, ids.data(), (int)ids.size());
      }
    }
  }
}

ExpertCache::~ExpertCache() {
  { std::unique_lock<std::mutex> lk(mu_); cv_.wait(lk, [&] { return inflight_ == 0; }); }
  if (p_->feeder.joinable()) {
    { std::lock_guard<std::mutex> g(p_->rmu); p_->stop = true; }
    p_->rcv.notify_all();
    p_->feeder.join();
  }
  pool_.reset();   // joins workers (ring loads still queued finish first)
  for (auto& b : p_->bufs) Device::get().release(b);
  for (auto& m : p_->maps) munmap(m.first, m.second);
}

// ---------------- core (under mu_) ----------------
int ExpertCache::take_slot_locked(int cls, bool block, std::unique_lock<std::mutex>& lk) {
  auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    if (!free_[cls].empty()) { int s = free_[cls].back(); free_[cls].pop_back(); return s; }
    struct Ctx { const std::vector<Slot>* sl; } ctx{&slots_};
    int v = p_->policy->victim(cls, [](const void* c, int s) {
      const Slot& x = (*((const Ctx*)c)->sl)[s];
      return x.state == Ready && x.ref == 0 && !x.pinned;
    }, &ctx);
    if (v >= 0) { evict_locked(v); return v; }
    if (!block) return -1;
    if (!to_submit_.empty()) { lk.unlock(); flush_submits(); lk.lock(); continue; }   // loads we wait on must be queued
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
      fprintf(stderr, "experts: no evictable slot in class %d for 30 s (all %d slots in use/pinned/loading)\n", cls, classes_[cls].n_slots);
      abort();
    }
    cv_.wait_for(lk, std::chrono::milliseconds(50));
  }
}

void ExpertCache::evict_locked(int s) {
  Slot& x = slots_[s];
  if (x.key != ~0u) {
    LayerStats& st = stats_[x.key / L_.n_experts];
    st.evictions++;
    if (x.prefetched && !x.used) st.prefetch_wasted++;
    p_->policy->on_evict(s, x.key);
    map_[x.key] = -1;
  }
  x.key = ~0u; x.state = Empty; x.prefetched = x.used = false; x.pinned = false;
}

void ExpertCache::load_locked(int s, uint32_t k, int prio) {
  Slot& x = slots_[s];
  const int layer = (int)(k / L_.n_experts), e = (int)(k % L_.n_experts);
  const ClassInfo& ci = classes_[x.cls];
  x.key = k; x.state = Loading; map_[k] = s;
  if (!p_->pinset.empty() && p_->pinset[k] && p_->pin_count[x.cls] < p_->pin_quota[x.cls]) { x.pinned = true; p_->pin_count[x.cls]++; }
  static const bool cold_pf = env_int("MACSEEK_PF_COLD", 0) != 0;   // prefetches enter at the eviction end
  if (!x.pinned) { if (cold_pf && prio == 1) p_->policy->on_insert_cold(s, k); else p_->policy->on_insert(s, k); }
  inflight_++;
  auto req = std::make_shared<IoRequest>();
  uint64_t region = 0;
  const uint64_t A = o_.align, chunk = o_.read_chunk ? rup(o_.read_chunk, A) : (1ull << 40);
  for (int m = 0; m < 3; m++) {
    const ExpertMat& em = L_.layers[layer].mat[m];
    uint64_t off = em.base + (uint64_t)e * em.stride;
    x.data_off[m] = (uint32_t)(region + off % kPage);
    uint64_t lo = rdown(off, A), hi = std::min(rup(off + em.len, A), em.file_size);
    char* dst = x.host + region + (off % kPage) - (off - lo);
    for (uint64_t a = lo; a < hi; a += chunk) {
      uint64_t len = std::min(chunk, hi - a);
      req->chunks.push_back({em.fd, a, (uint32_t)len, dst + (a - lo), 0});
    }
    region += ci.region[m];
  }
  if (o_.dry_io) { inflight_--; x.state = Ready; return; }
  req->priority = prio;
  req->on_done = [this, s](IoRequest& r) { on_loaded(s, r.bytes.load()); };
  x.req = req.get();
  to_submit_.push_back(std::move(req));
}

void ExpertCache::flush_submits() {
  std::vector<std::shared_ptr<IoRequest>> v;
  { std::lock_guard<std::mutex> g(mu_); v.swap(to_submit_); }
  if (!v.empty()) pool_->submit(v);
}

void ExpertCache::complete_tickets_locked() {
  uint64_t before = p_->done;
  while (!p_->pending.empty() && p_->pending.front() == 0) { p_->pending.pop_front(); p_->done = p_->base++; }
  if (p_->done != before) p_->io_ev.signaledValue = p_->done;
}

void ExpertCache::on_loaded(int s, uint64_t bytes) {
  std::lock_guard<std::mutex> g(mu_);
  Slot& x = slots_[s];
  x.state = Ready; x.req = nullptr;
  stats_[x.key / L_.n_experts].bytes_read += bytes;
  for (uint64_t t : x.waiters) p_->pending[t - p_->base]--;
  x.waiters.clear();
  complete_tickets_locked();
  inflight_--;
  cv_.notify_all();
}

ExpertView ExpertCache::view_locked(int s) const {
  const Slot& x = slots_[s];
  ExpertView v; v.slot = s;
  const int layer = (int)(x.key / L_.n_experts);
  for (int m = 0; m < 3; m++) {
    const ExpertMat& em = L_.layers[layer].mat[m];
    v.mat[m] = x.buf.sub(x.data_off[m], em.len);
    v.gpu[m] = x.gpu + x.data_off[m];
    v.host[m] = x.host + x.data_off[m];
    v.type[m] = em.type;
  }
  return v;
}

ExpertLease ExpertCache::ensure(int layer, const int* ids, int n) {
  ExpertLease lease;
  lease.cache_ = this; lease.layer = layer;
  lease.views.reserve(n); lease.slots_.reserve(n);
  std::unique_lock<std::mutex> lk(mu_);
  LayerStats& st = stats_[layer];
  const int cls = layer_class_[layer];
  std::vector<int> waiting;
  for (int i = 0; i < n; i++) {
    uint32_t k = key(layer, ids[i]);
    access_[k] += 1;
    int s = map_[k];
    if (s >= 0) {
      Slot& x = slots_[s];
      st.hits++;
      if (x.state != Ready) {
        st.inflight_hits++;
        if (x.req && x.req->priority) {
          bool pending_submit = false;   // built but not yet handed to the pool: just raise its priority
          for (auto& r : to_submit_) if (r.get() == x.req) { r->priority = 0; pending_submit = true; break; }
          if (!pending_submit) pool_->promote(x.req);
        }
      }
      if (x.prefetched && !x.used) st.prefetch_used++;
    } else {
      s = take_slot_locked(cls, true, lk);
      // the lock may have been dropped while blocking: re-check the key
      if (map_[k] >= 0) { free_[cls].push_back(s); s = map_[k]; st.hits++; }
      else { st.misses++; load_locked(s, k, 0); }
    }
    Slot& x = slots_[s];
    x.used = true;
    x.ref++;
    if (!x.pinned) p_->policy->on_access(s, k);
    if (x.state != Ready && std::find(waiting.begin(), waiting.end(), s) == waiting.end()) waiting.push_back(s);
    lease.slots_.push_back(s);
    lease.views.push_back(view_locked(s));
  }
  if (!waiting.empty()) {
    lease.ticket = p_->next_ticket++;
    p_->pending.push_back((int)waiting.size());
    for (int s : waiting) slots_[s].waiters.push_back(lease.ticket);
  }
  lk.unlock();
  flush_submits();
  return lease;
}

int ExpertCache::prefetch(int layer, const int* ids, int n) {
  std::unique_lock<std::mutex> lk(mu_);
  int issued = 0;
  const int cls = layer_class_[layer];
  for (int i = 0; i < n; i++) {
    uint32_t k = key(layer, ids[i]);
    if (map_[k] >= 0) continue;
    int s = take_slot_locked(cls, false, lk);
    if (s < 0) break;
    load_locked(s, k, 1);
    slots_[s].prefetched = true; slots_[s].used = false;
    stats_[layer].prefetch_issued++;
    issued++;
  }
  lk.unlock();
  flush_submits();
  return issued;
}

int ExpertCache::pin(int layer, const int* ids, int n) {
  std::unique_lock<std::mutex> lk(mu_);
  int pinned = 0;
  const int cls = layer_class_[layer];
  for (int i = 0; i < n; i++) {
    uint32_t k = key(layer, ids[i]);
    int s = map_[k];
    if (s < 0) {
      s = take_slot_locked(cls, false, lk);
      if (s < 0) break;
      load_locked(s, k, 1);
    }
    if (!slots_[s].pinned) {
      slots_[s].pinned = true;
      p_->policy->on_evict(s, k);   // leaves the policy's candidate set
      if (!p_->pin_count.empty()) p_->pin_count[cls]++;
    }
    pinned++;
  }
  lk.unlock();
  flush_submits();
  return pinned;
}

void ExpertCache::unpin(int layer, const int* ids, int n) {
  std::lock_guard<std::mutex> g(mu_);
  for (int i = 0; i < n; i++) {
    int s = map_[key(layer, ids[i])];
    if (s >= 0 && slots_[s].pinned) {
      slots_[s].pinned = false;
      p_->policy->on_insert(s, slots_[s].key);
      if (!p_->pin_count.empty()) p_->pin_count[slots_[s].cls]--;
    }
  }
  cv_.notify_all();
}

int ExpertCache::pin_resident(int keep_unpinned) {
  std::lock_guard<std::mutex> g(mu_);
  int total = 0;
  for (int c = 0; c < (int)classes_.size(); c++) {
    int unpinned = 0, ready = 0;
    std::vector<std::pair<double, int>> cand;
    for (int s = 0; s < (int)slots_.size(); s++) {
      const Slot& x = slots_[s];
      if (x.cls != c || x.pinned) continue;
      unpinned++;
      if (x.state == Ready && x.key != ~0u) { cand.push_back({access_[x.key], s}); ready++; }
    }
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
    // streaming room per class: 3 prefill groups (see group_cap) for builds with many small size classes (GSQ-RCO Coder:
    // 8 classes of 1-14 layers); keep_unpinned applies as given for the 3-class UD-IQ1_S layout
    const int keep_c = classes_.size() <= 3 ? keep_unpinned
                     : std::max(3 * std::max(4, classes_[c].n_slots / 3) / 1, keep_unpinned * (int)classes_[c].layers.size() / std::max(1, L_.n_layers));
    int pc = 0;
    for (auto& pr : cand) {
      if (unpinned <= keep_c) break;
      Slot& x = slots_[pr.second];
      x.pinned = true; x.res_pin = true;
      p_->policy->on_evict(pr.second, x.key);   // leaves the policy's candidate set
      unpinned--; total++; pc++;
    }
    if (getenv("MACSEEK_VERBOSE")) fprintf(stderr, "pin_resident: class %d: %d unpinned (%d ready) -> pinned %d\n", c, unpinned + pc, ready, pc);
  }
  return total;
}

void ExpertCache::unpin_resident() {
  std::lock_guard<std::mutex> g(mu_);
  for (int s = 0; s < (int)slots_.size(); s++) {
    Slot& x = slots_[s];
    if (!x.res_pin) continue;
    x.res_pin = false; x.pinned = false;
    if (x.key != ~0u) p_->policy->on_insert(s, x.key);
  }
  cv_.notify_all();
}

bool ExpertCache::contains(int layer, int e) const { std::lock_guard<std::mutex> g(mu_); return map_[key(layer, e)] >= 0; }

int ExpertCache::n_pinned() const {
  std::lock_guard<std::mutex> g(mu_);
  int n = 0;
  for (auto& s : slots_) n += s.pinned;
  return n;
}

void ExpertCache::drain() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return inflight_ == 0; });
  lk.unlock();
  if (p_->K) {
    std::unique_lock<std::mutex> rl(p_->rmu);
    p_->rcv.wait(rl, [&] { return p_->stop || (p_->jobs.empty() && p_->ring_done == p_->ring_next); });
  }
}

void ExpertCache::release_slots(const std::vector<int>& sl) {
  std::lock_guard<std::mutex> g(mu_);
  for (int s : sl) slots_[s].ref--;
  cv_.notify_all();
}

std::vector<Buffer> ExpertCache::buffers() const { return p_->bufs; }
void* ExpertCache::io_event() const { return (__bridge void*)p_->io_ev; }

// ---------------- lease ----------------
ExpertLease& ExpertLease::operator=(ExpertLease&& o) noexcept {
  if (this != &o) {
    release();
    views = std::move(o.views); layer = o.layer; ticket = o.ticket; cache_ = o.cache_; slots_ = std::move(o.slots_);
    o.cache_ = nullptr; o.slots_.clear();
  }
  return *this;
}

bool ExpertLease::ready() const {
  if (!cache_) return true;
  std::lock_guard<std::mutex> g(cache_->mu_);
  for (int s : slots_) if (cache_->slots_[s].state != ExpertCache::Ready) return false;
  return true;
}

void ExpertLease::wait() {
  if (!cache_) return;
  std::unique_lock<std::mutex> lk(cache_->mu_);
  auto all = [&] { for (int s : slots_) if (cache_->slots_[s].state != ExpertCache::Ready) return false; return true; };
  if (all()) return;
  Timer t;
  cache_->cv_.wait(lk, all);
  cache_->stats_[layer].stall_ms += t.ms();
}

void ExpertLease::release() {
  if (!cache_) return;
  cache_->release_slots(slots_);
  cache_ = nullptr; slots_.clear();
}

void ExpertLease::release_after(void* cmdbuf) {
  if (!cache_) return;
  ExpertCache* c = cache_;
  std::vector<int> sl = slots_;
  [(__bridge id<MTLCommandBuffer>)cmdbuf addCompletedHandler:^(id<MTLCommandBuffer>) { c->release_slots(sl); }];
  cache_ = nullptr; slots_.clear();
}

void ExpertLease::encode_wait(void* cmdbuf) const {
  if (!cache_ || !ticket) return;
  [(__bridge id<MTLCommandBuffer>)cmdbuf encodeWaitForEvent:(__bridge id<MTLSharedEvent>)cache_->io_event() value:ticket];
}

// ---------------- bulk stream ----------------
std::unique_ptr<ExpertStream> ExpertCache::stream(int layer, const int* ids, int n) {
  if (!p_->K) { fprintf(stderr, "experts: stream() needs ring_slots > 0\n"); abort(); }
  auto st = std::unique_ptr<ExpertStream>(new ExpertStream);
  st->cache_ = this; st->layer_ = layer;
  std::vector<int> cached, miss;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (int i = 0; i < n; i++) (map_[key(layer, ids[i])] >= 0 ? cached : miss).push_back(ids[i]);
  }
  std::sort(miss.begin(), miss.end());
  miss.erase(std::unique(miss.begin(), miss.end()), miss.end());
  if (!cached.empty()) {
    st->lease_ = ensure(layer, cached);
    for (size_t i = 0; i < cached.size(); i++) st->items_.push_back({cached[i], st->lease_.views[i], true, 0});
  }
  st->n_cached_ = st->items_.size();
  const ClassInfo& ci = classes_[layer_class_[layer]];
  std::lock_guard<std::mutex> g(p_->rmu);
  for (int e : miss) {
    uint64_t seq = p_->ring_next++;
    int r = (int)(seq % p_->K);
    ExpertView v; v.slot = -1;
    uint64_t region = 0;
    for (int m = 0; m < 3; m++) {
      const ExpertMat& em = L_.layers[layer].mat[m];
      uint64_t doff = region + (em.base + (uint64_t)e * em.stride) % kPage;
      v.mat[m] = p_->ring_buf[r].sub(doff, em.len);
      v.gpu[m] = p_->ring_gpu[r] + doff;
      v.host[m] = p_->ring_host[r] + doff;
      v.type[m] = em.type;
      region += ci.region[m];
    }
    st->items_.push_back({e, v, false, seq});
    p_->jobs.push_back({seq, layer, e});
  }
  p_->rcv.notify_all();
  return st;
}

void ExpertCache::stream_feeder() {
  pthread_setname_np("ms.ring");
  const uint64_t K = (uint64_t)p_->K;
  for (;;) {
    Impl::Job j;
    {
      std::unique_lock<std::mutex> lk(p_->rmu);
      p_->rcv.wait(lk, [&] { return p_->stop || !p_->jobs.empty(); });
      if (p_->stop) return;
      j = p_->jobs.front(); p_->jobs.pop_front();
    }
    // wait until the ring slot's previous occupant (seq - K) has been consumed
    if (j.seq >= K) {
      while (p_->consume_ev.signaledValue < j.seq + 1 - K) {
        [p_->consume_ev waitUntilSignaledValue:j.seq + 1 - K timeoutMS:20];
        std::lock_guard<std::mutex> g(p_->rmu);
        if (p_->stop) return;
      }
    }
    const int r = (int)(j.seq % K);
    const ClassInfo& ci = classes_[layer_class_[j.layer]];
    auto req = std::make_shared<IoRequest>();
    uint64_t region = 0;
    const uint64_t A = o_.align, chunk = o_.read_chunk ? rup(o_.read_chunk, A) : (1ull << 40);
    for (int m = 0; m < 3; m++) {
      const ExpertMat& em = L_.layers[j.layer].mat[m];
      uint64_t off = em.base + (uint64_t)j.expert * em.stride;
      uint64_t lo = rdown(off, A), hi = std::min(rup(off + em.len, A), em.file_size);
      char* dst = p_->ring_host[r] + region + (off % kPage) - (off - lo);
      for (uint64_t a = lo; a < hi; a += chunk) req->chunks.push_back({em.fd, a, (uint32_t)std::min(chunk, hi - a), dst + (a - lo), 0});
      region += ci.region[m];
    }
    const uint64_t seq = j.seq; const int layer = j.layer;
    auto done = [this, seq, layer, K](uint64_t bytes) {
      {
        std::lock_guard<std::mutex> g(mu_);
        stats_[layer].streamed++; stats_[layer].stream_bytes += bytes;
      }
      std::lock_guard<std::mutex> g(p_->rmu);
      p_->ring_flag[seq % K] = 1;
      uint64_t before = p_->ring_done;
      while (p_->ring_done < p_->ring_next && p_->ring_flag[p_->ring_done % K]) { p_->ring_flag[p_->ring_done % K] = 0; p_->ring_done++; }
      if (p_->ring_done != before) p_->ready_ev.signaledValue = p_->ring_done;
      p_->rcv.notify_all();
    };
    if (o_.dry_io) { done(0); continue; }
    req->priority = 0;
    req->on_done = [done](IoRequest& rq) { done(rq.bytes.load()); };
    pool_->submit(std::move(req));
  }
}

ExpertStream::~ExpertStream() = default;

void ExpertStream::wait_ready(size_t i) {
  const Item& it = items_[i];
  if (it.cached) { lease_.wait(); return; }
  auto* p = cache_->p_.get();
  std::unique_lock<std::mutex> lk(p->rmu);
  if (p->ring_done > it.seq) return;
  Timer t;
  p->rcv.wait(lk, [&] { return p->ring_done > it.seq; });
  double ms = t.ms();
  lk.unlock();
  std::lock_guard<std::mutex> g(cache_->mu_);
  cache_->stats_[layer_].stall_ms += ms;
}

void ExpertStream::consumed(size_t i) {
  const Item& it = items_[i];
  if (it.cached) return;
  auto* p = cache_->p_.get();
  std::lock_guard<std::mutex> g(p->rmu);
  if (p->consume_ev.signaledValue < it.seq + 1) p->consume_ev.signaledValue = it.seq + 1;
}

void ExpertStream::encode_wait_ready(void* cb, size_t i) const {
  const Item& it = items_[i];
  if (it.cached) { lease_.encode_wait(cb); return; }
  [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:cache_->p_->ready_ev value:it.seq + 1];
}

void ExpertStream::encode_consumed(void* cb, size_t i) const {
  const Item& it = items_[i];
  if (it.cached) return;
  [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:cache_->p_->consume_ev value:it.seq + 1];
}

void* ExpertStream::ready_event() const { return (__bridge void*)cache_->p_->ready_ev; }
void* ExpertStream::consume_event() const { return (__bridge void*)cache_->p_->consume_ev; }

// ---------------- stats ----------------
LayerStats ExpertCache::layer_stats(int l) const { std::lock_guard<std::mutex> g(mu_); return stats_[l]; }
LayerStats ExpertCache::total_stats() const {
  std::lock_guard<std::mutex> g(mu_);
  LayerStats t;
  for (auto& s : stats_) t += s;
  return t;
}
void ExpertCache::reset_stats() { std::lock_guard<std::mutex> g(mu_); for (auto& s : stats_) s = LayerStats{}; }

std::string ExpertCache::stats_report() const {
  std::ostringstream os;
  char b[256];
  LayerStats t;
  std::lock_guard<std::mutex> g(mu_);
  os << "layer   hits  misses  hit%   pf_iss pf_used pf_waste  evict   read_GB  stream  stall_ms\n";
  for (int l = 0; l < L_.n_layers; l++) {
    const LayerStats& s = stats_[l];
    t += s;
    if (!(s.hits + s.misses + s.streamed + s.prefetch_issued)) continue;
    snprintf(b, sizeof b, "%5d %6llu %7llu %5.1f %8llu %7llu %8llu %6llu %9.3f %7llu %9.2f\n", l, (unsigned long long)s.hits,
             (unsigned long long)s.misses, 100 * s.hit_rate(), (unsigned long long)s.prefetch_issued, (unsigned long long)s.prefetch_used,
             (unsigned long long)s.prefetch_wasted, (unsigned long long)s.evictions, (s.bytes_read + s.stream_bytes) / 1e9,
             (unsigned long long)s.streamed, s.stall_ms);
    os << b;
  }
  snprintf(b, sizeof b, "total %6llu %7llu %5.1f %8llu %7llu %8llu %6llu %9.3f %7llu %9.2f\n", (unsigned long long)t.hits,
           (unsigned long long)t.misses, 100 * t.hit_rate(), (unsigned long long)t.prefetch_issued, (unsigned long long)t.prefetch_used,
           (unsigned long long)t.prefetch_wasted, (unsigned long long)t.evictions, (t.bytes_read + t.stream_bytes) / 1e9,
           (unsigned long long)t.streamed, t.stall_ms);
  os << b;
  return os.str();
}

std::string ExpertCache::describe() const {
  std::ostringstream os;
  char b[256];
  snprintf(b, sizeof b, "ExpertCache: arena %.2f GiB, %zu slots, policy %s, %d io threads, chunk %zu KiB, align %u, ring %d\n",
           arena_bytes_ / 1073741824.0, slots_.size(), o_.custom_policy ? "custom" : policy_name(o_.policy), o_.io_threads,
           o_.read_chunk >> 10, o_.align, p_->K);
  os << b;
  for (size_t c = 0; c < classes_.size(); c++) {
    const ClassInfo& ci = classes_[c];
    const ExpertLayerLayout& ll = L_.layers[ci.layers[0]];
    snprintf(b, sizeof b, "  class %zu: slot %llu B (regions %llu/%llu/%llu, types %s/%s/%s), %d layers, %d slots (%.1f/layer, %.1f%% of experts)\n",
             c, (unsigned long long)ci.slot_bytes, (unsigned long long)ci.region[0], (unsigned long long)ci.region[1],
             (unsigned long long)ci.region[2], type_name(ll.mat[0].type), type_name(ll.mat[1].type), type_name(ll.mat[2].type),
             (int)ci.layers.size(), ci.n_slots, (double)ci.n_slots / ci.layers.size(),
             100.0 * ci.n_slots / (ci.layers.size() * L_.n_experts));
    os << b;
  }
  return os.str();
}

bool ExpertCache::write_routing_stats(const std::string& path) const {
  std::lock_guard<std::mutex> g(mu_);
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;
  fprintf(f, "# MacSeek routing statistics: %d layers x %d experts, demand access counts\n", L_.n_layers, L_.n_experts);
  for (int l = 0; l < L_.n_layers; l++) {
    for (int e = 0; e < L_.n_experts; e++) fprintf(f, e ? " %.0f" : "%.0f", access_[(size_t)l * L_.n_experts + e]);
    fprintf(f, "\n");
  }
  fclose(f);
  return true;
}

bool ExpertCache::read_routing_stats(const std::string& path, int nl, int ne, std::vector<double>& out) {
  std::ifstream in(path);
  if (!in) return false;
  out.assign((size_t)nl * ne, 0.0);
  std::string line; int l = 0;
  while (std::getline(in, line) && l < nl) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    for (int e = 0; e < ne; e++) if (!(ss >> out[(size_t)l * ne + e])) return false;
    l++;
  }
  return l == nl;
}

}  // namespace ms
