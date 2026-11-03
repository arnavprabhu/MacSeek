#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "store/experts.hpp"

#include "core/env.hpp"
#include "core/timer.hpp"
#include "store/io_pool.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <climits>
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
  std::vector<int> pin_budget;      // Static: per pool, max regions pinned by the pinned set
  uint64_t tick = 0;                // demand accesses so far
  uint64_t pf_window = 0;           // unused prefetches younger than this many demand accesses are protected
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
  // ---- region size per matrix type: max over every (layer, expert) matrix of that type of the 16K-rounded span at
  //      offset mod 16K; one pool per distinct region size ----
  std::map<int, uint64_t> type_region;
  for (int l = 0; l < NL; l++)
    for (int k = 0; k < 3; k++) {
      const ExpertMat& m = L_.layers[l].mat[k];
      uint64_t r = 0;
      for (int e = 0; e < NE; e++) r = std::max(r, rup((m.base + (uint64_t)e * m.stride) % kPage + m.len, kPage));
      uint64_t& tr = type_region[(int)m.type];
      tr = std::max(tr, r);
    }
  std::map<uint64_t, int> pool_of;
  for (auto& tr : type_region) pool_of[tr.second] = 0;
  for (auto& po : pool_of) { po.second = (int)pools_.size(); PoolInfo pi; pi.region = po.first; pools_.push_back(pi); }
  for (auto& tr : type_region) pools_[pool_of[tr.second]].types.push_back((GType)tr.first);
  const int P = (int)pools_.size();
  if (P > 32) { fprintf(stderr, "experts: too many region sizes (%d)\n", P); abort(); }
  layer_pool_.assign((size_t)NL * 3, 0); layer_use_.assign((size_t)NL * P, 0); layer_mask_.assign(NL, 0);
  for (int l = 0; l < NL; l++)
    for (int k = 0; k < 3; k++) {
      const int pp = pool_of[type_region[(int)L_.layers[l].mat[k].type]];
      layer_pool_[(size_t)l * 3 + k] = pp; layer_use_[(size_t)l * P + pp]++; layer_mask_[l] |= 1u << pp;
      pools_[pp].demand += NE;
    }
  for (int pp = 0; pp < P; pp++) {
    int sum = 0, mx = 0;
    for (int l = 0; l < NL; l++) { sum += layer_use_[(size_t)l * P + pp]; mx = std::max<int>(mx, layer_use_[(size_t)l * P + pp]); }
    pools_[pp].mean_use = NL ? (double)sum / NL : 0; pools_[pp].max_use = mx;
  }
  // ---- split the arena across pools ----
  std::vector<double> rstats;
  bool have_stats = !o_.stats_path.empty() && read_routing_stats(o_.stats_path, NL, NE, rstats);
  if (!o_.stats_path.empty() && !have_stats) fprintf(stderr, "experts: could not read routing stats %s\n", o_.stats_path.c_str());
  uint64_t total_demand = 0;
  for (auto& pi : pools_) total_demand += pi.demand * pi.region;
  if (o_.arena_bytes >= total_demand) {
    for (auto& pi : pools_) pi.n_regions = (int)pi.demand;   // everything fits
  } else {
    std::vector<uint64_t> sel(P, 0);   // regions selected so far
    uint64_t budget = o_.arena_bytes;
    if (have_stats) {
      // floor of 16 experts per layer, then greedy by (count / bytes) over all (layer, expert)
      for (int pp = 0; pp < P; pp++) {
        uint64_t f = 0;
        for (int l = 0; l < NL; l++) f += 16ull * layer_use_[(size_t)l * P + pp];
        f = std::min<uint64_t>({f, pools_[pp].demand, budget / pools_[pp].region});
        sel[pp] = f; budget -= f * pools_[pp].region;
      }
      std::vector<std::pair<double, uint32_t>> rank;
      for (int l = 0; l < NL; l++)
        for (int e = 0; e < NE; e++)
          if (rstats[(size_t)l * NE + e] > 0) rank.push_back({rstats[(size_t)l * NE + e] / entry_bytes(l), key(l, e)});
      std::sort(rank.begin(), rank.end(), [](auto& x, auto& y) { return x.first > y.first; });
      for (auto& r : rank) {
        const int l = (int)(r.second / NE);
        const uint64_t eb = entry_bytes(l);
        if (eb > budget) continue;
        bool room = true;
        for (int pp = 0; pp < P; pp++) room &= sel[pp] + layer_use_[(size_t)l * P + pp] <= pools_[pp].demand;
        if (!room) continue;
        for (int pp = 0; pp < P; pp++) sel[pp] += layer_use_[(size_t)l * P + pp];
        budget -= eb;
      }
    }
    // the rest in proportion to the remaining demand (no stats: every pool holds the same fraction of its matrices)
    double rem = 0;
    for (int pp = 0; pp < P; pp++) rem += (double)(pools_[pp].demand - sel[pp]) * pools_[pp].region;
    for (int pp = 0; pp < P; pp++) {
      uint64_t extra = rem > 0 ? (uint64_t)((double)budget * (pools_[pp].demand - sel[pp]) / rem) : 0;
      pools_[pp].n_regions = (int)std::min<uint64_t>(pools_[pp].demand, sel[pp] + extra);
    }
  }
  for (auto& pi : pools_)
    if (pi.n_regions < 12 * pi.max_use) {
      fprintf(stderr, "experts: arena %.2f GiB too small: pool of %llu B regions gets %d regions (need >= %d)\n",
              o_.arena_bytes / 1073741824.0, (unsigned long long)pi.region, pi.n_regions, 12 * pi.max_use);
      abort();
    }
  // ---- allocate regions: per pool, MTLBuffers of whole regions ----
  Device& dev = Device::get();
  size_t maxbuf = o_.max_buffer_bytes ? o_.max_buffer_bytes : std::min<size_t>(dev.max_buffer_length(), 8ull << 30);
  for (int pp = 0; pp < P; pp++) {
    const PoolInfo& pi = pools_[pp];
    uint64_t per_buf = std::max<uint64_t>(1, maxbuf / pi.region);
    for (int r0 = 0; r0 < pi.n_regions; r0 += (int)per_buf) {
      int nr = (int)std::min<uint64_t>(per_buf, pi.n_regions - r0);
      size_t bytes = (size_t)nr * pi.region;
      char* mem = nullptr; Buffer b; uint64_t gbase = 0;
      if (!o_.dry_io) {   // dry_io (policy simulation): no memory at all
        mem = (char*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (mem == MAP_FAILED) { perror("experts mmap"); abort(); }
        p_->maps.push_back({mem, bytes});
        b = dev.wrap(mem, bytes, "expert_regions");
        p_->bufs.push_back(b);
        gbase = [(__bridge id<MTLBuffer>)b.buf gpuAddress];
      }
      for (int i = 0; i < nr; i++) {
        Region R;
        if (mem) { R.host = mem + (size_t)i * pi.region; R.buf = b.sub((size_t)i * pi.region, pi.region); R.gpu = gbase + (uint64_t)i * pi.region; }
        regions_.push_back(R); region_pool_.push_back(pp);
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
  free_reg_.assign(P, {});
  for (int r = (int)regions_.size() - 1; r >= 0; r--) free_reg_[region_pool_[r]].push_back(r);
  pinned_reg_.assign(P, 0); spec_reg_.assign(P, 0); ref_reg_.assign(P, 0);
  // entry ids: every resident expert owns 3 regions, so total regions / 3 bounds the number of residents
  slots_.resize(std::max<size_t>(1, regions_.size() / 3));
  for (int s = (int)slots_.size() - 1; s >= 0; s--) free_slots_.push_back(s);
  {
    long w = env_int("MACSEEK_PF_PROTECT", -1);   // demand accesses an unused prefetch stays protected (0 = off)
    p_->pf_window = w >= 0 ? (uint64_t)w : (uint64_t)slots_.size();
  }
  map_.assign((size_t)NL * NE, -1);
  stats_.assign(NL, LayerStats{});
  access_.assign((size_t)NL * NE, 0.0);
  id<MTLDevice> mdev = (__bridge id<MTLDevice>)dev.device();
  p_->io_ev = [mdev newSharedEvent];
  p_->io_ev.signaledValue = 0;
  // ---- policy: one class (the whole arena) ----
  if (o_.custom_policy) p_->policy = o_.custom_policy;
  else p_->policy = std::shared_ptr<EvictionPolicy>(make_policy(o_.policy, o_.lfu_decay, o_.lfu_period, o_.lfu_sample).release());
  p_->policy->init(std::vector<int>(slots_.size(), 0), 1, NL * NE);
  // ---- I/O ----
  pool_ = std::make_unique<IoPool>(o_.io_threads, kPage);
  // ---- ring (staging slots sized for the largest entry) ----
  if (o_.ring_slots > 0 && !o_.dry_io) {
    p_->K = o_.ring_slots;
    for (int l = 0; l < NL; l++) p_->ring_slot_bytes = std::max(p_->ring_slot_bytes, entry_bytes(l));
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
  // ---- static frequency-pinned set: globally most frequent (layer, expert) by count / bytes, until pin_fraction of
  //      every pool is pinned ----
  if (o_.policy == PolicyKind::Static && have_stats) {
    p_->pinset.assign((size_t)NL * NE, 0);
    p_->pin_budget.assign(P, 0);
    for (int pp = 0; pp < P; pp++) p_->pin_budget[pp] = (int)(o_.pin_fraction * pools_[pp].n_regions);
    std::vector<std::pair<double, uint32_t>> r;
    for (int l = 0; l < NL; l++)
      for (int e = 0; e < NE; e++)
        if (rstats[(size_t)l * NE + e] > 0) r.push_back({rstats[(size_t)l * NE + e] / entry_bytes(l), key(l, e)});
    std::stable_sort(r.begin(), r.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::vector<int> used(P, 0);
    for (auto& x : r) {
      const int l = (int)(x.second / NE);
      bool ok = true;
      for (int pp = 0; pp < P; pp++) ok &= used[pp] + layer_use_[(size_t)l * P + pp] <= p_->pin_budget[pp];
      if (!ok) continue;
      for (int pp = 0; pp < P; pp++) used[pp] += layer_use_[(size_t)l * P + pp];
      p_->pinset[x.second] = 1;
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

uint64_t ExpertCache::entry_bytes(int layer) const {
  uint64_t b = 0;
  for (int k = 0; k < 3; k++) b += pools_[layer_pool(layer, k)].region;
  return b;
}

double ExpertCache::cached_fraction() const {
  double f = 1;
  for (auto& pi : pools_) if (pi.demand) f = std::min(f, (double)pi.n_regions / pi.demand);
  return f;
}

int ExpertCache::reserve_regions(int pp, int keep) const {
  const PoolInfo& pi = pools_[pp];
  double r = std::ceil(std::max(0, keep) * std::max(1.0, pi.mean_use));
  r = std::max(r, 24.0 * pi.max_use);   // every layer can stream groups of >= 8
  return (int)std::min<double>(r, pi.n_regions);
}

int ExpertCache::unpinned_capacity_locked(int layer) const {
  const int P = (int)pools_.size();
  int cap = INT32_MAX;
  for (int pp = 0; pp < P; pp++) {
    const int u = layer_use_[(size_t)layer * P + pp];
    if (u) cap = std::min(cap, (pools_[pp].n_regions - pinned_reg_[pp]) / u);
  }
  return std::min(cap, (int)slots_.size());
}

int ExpertCache::unpinned_capacity(int layer) const { std::lock_guard<std::mutex> g(mu_); return unpinned_capacity_locked(layer); }

int ExpertCache::group_cap(int layer) const {
  std::lock_guard<std::mutex> g(mu_);
  return std::max(1, std::min(64, unpinned_capacity_locked(layer) / 3));
}

// ---------------- core (under mu_) ----------------
int ExpertCache::take_entry_locked(int layer, TakeMode mode, std::unique_lock<std::mutex>& lk) {
  auto t0 = std::chrono::steady_clock::now();
  const int P = (int)pools_.size();
  const uint8_t* use = &layer_use_[(size_t)layer * P];
  if (mode == kPrefetch)   // speculation cap: keep a demand margin of unpinned regions that are neither leased nor
    for (int pp = 0; pp < P; pp++) {   // unused prefetches (i.e. free or evictable for the next demand group)
      if (!use[pp]) continue;
      const int unp = pools_[pp].n_regions - pinned_reg_[pp];
      const int margin = std::min(unp / 4, 64 * pools_[pp].max_use);   // ~1 prefill group of 64 for any layer
      if (unp - spec_reg_[pp] - ref_reg_[pp] - use[pp] < margin) return -1;
    }
  for (;;) {
    uint32_t need = 0;
    for (int pp = 0; pp < P; pp++) if (free_reg_[pp].size() < use[pp]) need |= 1u << pp;
    if (!need && !free_slots_.empty()) {
      int s = free_slots_.back(); free_slots_.pop_back();
      Slot& x = slots_[s];
      for (int m = 0; m < 3; m++) { auto& fl = free_reg_[layer_pool(layer, m)]; x.reg[m] = fl.back(); fl.pop_back(); }
      return s;
    }
    // evict a victim that owns a region of a missing size (any victim if only entry ids ran out). Young unused
    // prefetches (issued < pf_window demand accesses ago) are spared: prefetch never evicts them, demand only as a last
    // resort (prefill prefetches layer l+1 before layer l's MoE runs; LRU would otherwise evict layer l's own
    // not-yet-used prefetches first).
    struct Ctx { const ExpertCache* c; uint32_t need; bool spare_pf; } ctx{this, need, p_->pf_window > 0};
    auto pred = [](const void* cp, int s) {
      const Ctx& c = *(const Ctx*)cp;
      const Slot& x = c.c->slots_[s];
      if (x.state != Ready || x.ref != 0 || x.pinned || x.key == ~0u) return false;
      if (c.spare_pf && x.prefetched && !x.used && c.c->p_->tick - x.pf_tick < c.c->p_->pf_window) return false;
      return !c.need || (c.c->layer_mask_[x.key / c.c->L_.n_experts] & c.need) != 0;
    };
    int v = p_->policy->victim(0, pred, &ctx);
    if (v < 0 && ctx.spare_pf && mode == kDemand) {
      // last resort: the YOUNGEST unused prefetch, highest key on ties (the speculation furthest ahead: layer l+1's
      // before the current layer's own pending prefetches, and within a layer the experts a prefill uses last, since
      // groups run in expert-id order). An LRU order would evict the current layer's next prefetches and cascade.
      uint64_t best = 0;
      for (int s = 0; s < (int)slots_.size(); s++) {
        const Slot& x = slots_[s];
        if (x.state != Ready || x.ref != 0 || x.pinned || x.key == ~0u || !x.prefetched || x.used) continue;
        if (need && !(layer_mask_[x.key / L_.n_experts] & need)) continue;
        if (v < 0 || x.pf_tick > best || (x.pf_tick == best && x.key > slots_[v].key)) { v = s; best = x.pf_tick; }
      }
    }
    if (v >= 0) { evict_locked(v); continue; }
    if (mode != kDemand) return -1;
    if (!to_submit_.empty()) { lk.unlock(); flush_submits(); lk.lock(); continue; }   // loads we wait on must be queued
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
      fprintf(stderr, "experts: no evictable entry for layer %d for 30 s (every region of a needed size in use/pinned/loading):\n", layer);
      for (int pp = 0; pp < P; pp++)
        fprintf(stderr, "  pool %d (%llu B): %d regions, %zu free, %d pinned, layer needs %d\n", pp, (unsigned long long)pools_[pp].region,
                pools_[pp].n_regions, free_reg_[pp].size(), pinned_reg_[pp], use[pp]);
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
  set_pinned_locked(s, false);
  set_spec_locked(s, false);
  for (int m = 0; m < 3; m++) if (x.reg[m] >= 0) { free_reg_[region_pool_[x.reg[m]]].push_back(x.reg[m]); x.reg[m] = -1; }
  x.key = ~0u; x.state = Empty; x.prefetched = x.used = false; x.res_pin = false;
  free_slots_.push_back(s);
}

void ExpertCache::set_pinned_locked(int s, bool v) {
  Slot& x = slots_[s];
  if (x.pinned == v) return;
  const bool spec = x.spec;
  if (spec) set_spec_locked(s, false);
  x.pinned = v;
  for (int m = 0; m < 3; m++) if (x.reg[m] >= 0) {
    pinned_reg_[region_pool_[x.reg[m]]] += v ? 1 : -1;
    if (x.ref > 0) ref_reg_[region_pool_[x.reg[m]]] += v ? -1 : 1;   // ref_reg_ counts unpinned leased entries
  }
  if (spec || (!v && x.prefetched && !x.used)) set_spec_locked(s, !v);
}

// spec = unpinned prefetch no demand access has used yet (counted per pool for the prefetch speculation cap)
void ExpertCache::set_spec_locked(int s, bool v) {
  Slot& x = slots_[s];
  if (x.spec == v) return;
  x.spec = v;
  for (int m = 0; m < 3; m++) if (x.reg[m] >= 0) spec_reg_[region_pool_[x.reg[m]]] += v ? 1 : -1;
}

bool ExpertCache::can_pin_locked(int layer, const std::vector<int>& budget) const {
  const int P = (int)pools_.size();
  for (int pp = 0; pp < P; pp++) if (pinned_reg_[pp] + layer_use_[(size_t)layer * P + pp] > budget[pp]) return false;
  return true;
}

void ExpertCache::load_locked(int s, uint32_t k, int prio) {
  Slot& x = slots_[s];
  const int layer = (int)(k / L_.n_experts), e = (int)(k % L_.n_experts);
  x.key = k; x.state = Loading; map_[k] = s;
  if (!p_->pinset.empty() && p_->pinset[k] && can_pin_locked(layer, p_->pin_budget)) set_pinned_locked(s, true);
  static const bool cold_pf = env_int("MACSEEK_PF_COLD", 0) != 0;   // prefetches enter at the eviction end
  if (!x.pinned) { if (cold_pf && prio == 1) p_->policy->on_insert_cold(s, k); else p_->policy->on_insert(s, k); }
  for (int m = 0; m < 3; m++) {
    const ExpertMat& em = L_.layers[layer].mat[m];
    x.data_off[m] = (uint32_t)((em.base + (uint64_t)e * em.stride) % kPage);
  }
  if (o_.dry_io) { x.state = Ready; return; }
  inflight_++;
  auto req = std::make_shared<IoRequest>();
  const uint64_t A = o_.align, chunk = o_.read_chunk ? rup(o_.read_chunk, A) : (1ull << 40);
  for (int m = 0; m < 3; m++) {
    const ExpertMat& em = L_.layers[layer].mat[m];
    uint64_t off = em.base + (uint64_t)e * em.stride;
    uint64_t lo = rdown(off, A), hi = std::min(rup(off + em.len, A), em.file_size);
    char* dst = regions_[x.reg[m]].host + (off % kPage) - (off - lo);
    for (uint64_t a = lo; a < hi; a += chunk) {
      uint64_t len = std::min(chunk, hi - a);
      req->chunks.push_back({em.fd, a, (uint32_t)len, dst + (a - lo), 0});
    }
  }
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
    const Region& R = regions_[x.reg[m]];
    v.mat[m] = R.buf.sub(x.data_off[m], em.len);
    v.gpu[m] = R.gpu + x.data_off[m];
    v.host[m] = R.host ? R.host + x.data_off[m] : nullptr;
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
  std::vector<int> waiting;
  for (int i = 0; i < n; i++) {
    uint32_t k = key(layer, ids[i]);
    access_[k] += 1;
    p_->tick++;
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
      s = take_entry_locked(layer, kDemand, lk);
      // the lock may have been dropped while blocking: re-check the key
      if (map_[k] >= 0) { evict_locked(s); s = map_[k]; st.hits++; }
      else { st.misses++; load_locked(s, k, 0); }
    }
    Slot& x = slots_[s];
    x.used = true;
    set_spec_locked(s, false);
    if (x.ref++ == 0 && !x.pinned) for (int m = 0; m < 3; m++) ref_reg_[region_pool_[x.reg[m]]]++;
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
  for (int i = 0; i < n; i++) {
    uint32_t k = key(layer, ids[i]);
    if (map_[k] >= 0) continue;
    int s = take_entry_locked(layer, kPrefetch, lk);
    if (s < 0) break;
    load_locked(s, k, 1);
    slots_[s].prefetched = true; slots_[s].used = false; slots_[s].pf_tick = p_->tick;
    if (!slots_[s].pinned) set_spec_locked(s, true);
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
  for (int i = 0; i < n; i++) {
    uint32_t k = key(layer, ids[i]);
    int s = map_[k];
    if (s < 0) {
      s = take_entry_locked(layer, kPin, lk);
      if (s < 0) break;
      load_locked(s, k, 1);
    }
    if (!slots_[s].pinned) {
      set_pinned_locked(s, true);
      p_->policy->on_evict(s, k);   // leaves the policy's candidate set
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
      set_pinned_locked(s, false);
      slots_[s].res_pin = false;
      p_->policy->on_insert(s, slots_[s].key);
    }
  }
  cv_.notify_all();
}

int ExpertCache::pin_resident(int keep_unpinned) {
  std::lock_guard<std::mutex> g(mu_);
  const int P = (int)pools_.size(), NE = L_.n_experts;
  std::vector<int> unp(P), floor_(P);
  for (int pp = 0; pp < P; pp++) { unp[pp] = pools_[pp].n_regions - pinned_reg_[pp]; floor_[pp] = reserve_regions(pp, keep_unpinned); }
  std::vector<std::pair<double, int>> cand;
  for (int s = 0; s < (int)slots_.size(); s++) {
    const Slot& x = slots_[s];
    if (!x.pinned && x.state == Ready && x.key != ~0u) cand.push_back({access_[x.key], s});
  }
  std::stable_sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
  int total = 0;
  for (auto& pr : cand) {
    Slot& x = slots_[pr.second];
    const uint8_t* use = &layer_use_[(size_t)(x.key / NE) * P];
    bool ok = true;
    for (int pp = 0; pp < P; pp++) ok &= unp[pp] - use[pp] >= floor_[pp];
    if (!ok) continue;
    for (int pp = 0; pp < P; pp++) unp[pp] -= use[pp];
    set_pinned_locked(pr.second, true); x.res_pin = true;
    p_->policy->on_evict(pr.second, x.key);   // leaves the policy's candidate set
    total++;
  }
  if (getenv("MACSEEK_VERBOSE")) {
    fprintf(stderr, "pin_resident: %zu ready unpinned -> pinned %d;", cand.size(), total);
    for (int pp = 0; pp < P; pp++) fprintf(stderr, " pool %d: %d/%d unpinned (floor %d)", pp, unp[pp], pools_[pp].n_regions, floor_[pp]);
    fprintf(stderr, "\n");
  }
  return total;
}

void ExpertCache::unpin_resident() {
  std::lock_guard<std::mutex> g(mu_);
  for (int s = 0; s < (int)slots_.size(); s++) {
    Slot& x = slots_[s];
    if (!x.res_pin) continue;
    x.res_pin = false;
    set_pinned_locked(s, false);
    if (x.key != ~0u) p_->policy->on_insert(s, x.key);
  }
  cv_.notify_all();
}

bool ExpertCache::contains(int layer, int e) const { std::lock_guard<std::mutex> g(mu_); return map_[key(layer, e)] >= 0; }

bool ExpertCache::is_pinned(int layer, int e) const {
  std::lock_guard<std::mutex> g(mu_);
  int s = map_[key(layer, e)];
  return s >= 0 && slots_[s].pinned;
}

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
  for (int s : sl) {
    Slot& x = slots_[s];
    if (--x.ref == 0 && !x.pinned) for (int m = 0; m < 3; m++) ref_reg_[region_pool_[x.reg[m]]]--;
  }
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
      region += pools_[layer_pool(layer, m)].region;
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
    auto req = std::make_shared<IoRequest>();
    uint64_t region = 0;
    const uint64_t A = o_.align, chunk = o_.read_chunk ? rup(o_.read_chunk, A) : (1ull << 40);
    for (int m = 0; m < 3; m++) {
      const ExpertMat& em = L_.layers[j.layer].mat[m];
      uint64_t off = em.base + (uint64_t)j.expert * em.stride;
      uint64_t lo = rdown(off, A), hi = std::min(rup(off + em.len, A), em.file_size);
      char* dst = p_->ring_host[r] + region + (off % kPage) - (off - lo);
      for (uint64_t a = lo; a < hi; a += chunk) req->chunks.push_back({em.fd, a, (uint32_t)std::min(chunk, hi - a), dst + (a - lo), 0});
      region += pools_[layer_pool(j.layer, m)].region;
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
  char b[320];
  uint64_t all = 0, data = 0;
  for (auto& pi : pools_) all += pi.demand * pi.region;
  for (int l = 0; l < L_.n_layers; l++) data += L_.expert_bytes(l) * (uint64_t)L_.n_experts;
  snprintf(b, sizeof b, "ExpertCache: arena %.2f GiB in %zu region pools (one eviction/pinning domain), %zu entry ids, policy %s, "
           "%d io threads, chunk %zu KiB, align %u, ring %d\n",
           arena_bytes_ / 1073741824.0, pools_.size(), slots_.size(), o_.custom_policy ? "custom" : policy_name(o_.policy),
           o_.io_threads, o_.read_chunk >> 10, o_.align, p_->K);
  os << b;
  for (size_t pp = 0; pp < pools_.size(); pp++) {
    const PoolInfo& pi = pools_[pp];
    std::string ty;
    for (GType t : pi.types) { if (!ty.empty()) ty += "+"; ty += type_name(t); }
    snprintf(b, sizeof b, "  pool %zu: region %llu B (%s), %d regions = %.2f GiB, demand %llu regions (%.2f GiB) -> %.1f%% cached, "
             "%.2f/entry avg (max %d)\n",
             pp, (unsigned long long)pi.region, ty.c_str(), pi.n_regions, (double)pi.n_regions * pi.region / 1073741824.0,
             (unsigned long long)pi.demand, (double)pi.demand * pi.region / 1073741824.0, 100.0 * pi.n_regions / std::max<uint64_t>(1, pi.demand),
             pi.mean_use, pi.max_use);
    os << b;
  }
  snprintf(b, sizeof b, "  expected cached: %.1f%% of all %d x %d experts (%.2f of %.2f GiB in regions; expert data %.2f GiB)\n",
           100.0 * cached_fraction(), L_.n_layers, L_.n_experts, arena_bytes_ / 1073741824.0, all / 1073741824.0, data / 1073741824.0);
  os << b;
  uint64_t rb = 0;
  std::string rs;
  for (size_t pp = 0; pp < pools_.size(); pp++) {
    int r = reserve_regions((int)pp, 256);
    rb += (uint64_t)r * pools_[pp].region;
    rs += (pp ? "/" : "") + std::to_string(r);
  }
  snprintf(b, sizeof b, "  pin_resident(256) keeps %s regions unpinned (%.2f GiB, %.1f%% of the arena): up to %.1f%% of all experts pinnable\n",
           rs.c_str(), rb / 1073741824.0, 100.0 * rb / std::max<uint64_t>(1, arena_bytes_),
           100.0 * cached_fraction() * (1.0 - (double)rb / std::max<uint64_t>(1, arena_bytes_)));
  os << b;
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
