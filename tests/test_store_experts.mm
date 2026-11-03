// ExpertCache tests on synthetic expert files (fast, exact data checks) + a few real experts from the GGUF.
// Covers: region pools (3 region sizes, mixed per-layer types), miss/hit/data, global LRU order across pools,
// in-use protection (leases and GPU command-buffer completion), prefetch accounting and prefetch-never-evicts-
// unused-prefetch, LFU aging, static frequency pinning, io_event GPU waits, bulk stream (CPU + GPU protocol),
// alignment / chunking variants, a 2-thread randomized ensure/prefetch/pin/pin_resident stress with byte checks
// against pread, and a prefill simulation (pin_resident + group_cap groups with two leases alive + next-layer
// prefetch, 192 experts per layer).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "core/metal.hpp"
#include "store/experts.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

static std::atomic<int> fails{0};
static std::mutex print_mu;
#define CHECK(c, ...) do { if (!(c)) { std::lock_guard<std::mutex> g_(print_mu); fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static inline uint8_t pat(uint64_t x) { x ^= x >> 7; x *= 0x9E3779B97F4A7C15ULL; return (uint8_t)(x >> 56); }

// layer specs: letter = matrix kind (A: IQ1_S, B: IQ2_XXS, C: IQ2_XS) for gate/up/down -> 3 region sizes
static const char* kSpec[] = {"AAA", "AAB", "BBA", "BBB", "CCA", "CCC"};
static const int NL = 6;

struct Synth { std::string path; int fd = -1, fd_plain = -1; ms::ExpertLayout L; };

static Synth make_synth(const char* name, int ne, uint32_t la, uint32_t lb, uint32_t lc) {
  Synth s;
  const char* tmp = getenv("TMPDIR");
  s.path = std::string(tmp ? tmp : "/tmp") + "/" + name;
  s.L.n_layers = NL; s.L.n_experts = ne;
  uint64_t off = 4096 + 96;
  for (int l = 0; l < NL; l++) {
    ms::ExpertLayerLayout ll;
    for (int k = 0; k < 3; k++) {
      ms::ExpertMat& m = ll.mat[k];
      char c = kSpec[l][k];
      m.len = c == 'A' ? la : c == 'B' ? lb : lc;
      m.type = c == 'A' ? ms::GType::IQ1_S : c == 'B' ? ms::GType::IQ2_XXS : ms::GType::IQ2_XS;
      m.stride = m.len; m.base = off;
      off += (uint64_t)m.len * ne + 96;
    }
    s.L.layers.push_back(ll);
  }
  std::vector<uint8_t> buf(off);
  for (uint64_t i = 0; i < off; i++) buf[i] = pat(i);
  FILE* f = fopen(s.path.c_str(), "wb"); fwrite(buf.data(), 1, buf.size(), f); fclose(f);
  s.fd = open(s.path.c_str(), O_RDONLY);
  fcntl(s.fd, F_NOCACHE, 1);
  s.fd_plain = open(s.path.c_str(), O_RDONLY);
  for (auto& ll : s.L.layers) for (auto& m : ll.mat) { m.fd = s.fd; m.file_size = off; }
  return s;
}

static bool check_view(const ms::ExpertLayout& L, int layer, int e, const ms::ExpertView& v) {
  for (int k = 0; k < 3; k++) {
    const ms::ExpertMat& m = L.layers[layer].mat[k];
    uint64_t off = m.base + (uint64_t)e * m.stride;
    const uint8_t* p = (const uint8_t*)v.host[k];
    if ((const void*)p != v.mat[k].cpu()) return false;
    if (v.mat[k].size != m.len || v.type[k] != m.type) return false;
    for (uint32_t j = 0; j < m.len; j++) if (p[j] != pat(off + j)) { fprintf(stderr, "  data mismatch L%d e%d mat%d byte %u\n", layer, e, k, j); return false; }
  }
  return true;
}

// compare against a plain pread of the file region
static bool check_view_pread(const Synth& S, int layer, int e, const ms::ExpertView& v) {
  for (int k = 0; k < 3; k++) {
    const ms::ExpertMat& m = S.L.layers[layer].mat[k];
    std::vector<char> ref(m.len);
    if (ms::pread_full(S.fd_plain, ref.data(), m.len, m.base + (uint64_t)e * m.stride) != m.len) return false;
    if (memcmp(ref.data(), v.host[k], m.len) != 0) { fprintf(stderr, "  pread mismatch L%d e%d mat%d\n", layer, e, k); return false; }
  }
  return true;
}

static uint64_t per_layer_bytes(const ms::ExpertLayout& L) {
  ms::ExpertCacheOptions o; o.arena_bytes = 1ull << 40; o.io_threads = 1; o.dry_io = true;   // geometry only
  ms::ExpertCache probe(L, o);
  uint64_t s = 0;
  for (int l = 0; l < L.n_layers; l++) s += probe.entry_bytes(l);
  return s;
}

// arena = every pool holds spl/NE of its matrices (same as "spl experts per layer" on average)
static ms::ExpertCacheOptions small_opts(uint64_t plb, int slots_per_layer) {
  ms::ExpertCacheOptions o;
  o.arena_bytes = plb * slots_per_layer;
  o.io_threads = 4;
  o.read_chunk = 16384;
  return o;
}

static void one(ms::ExpertCache& c, int layer, int e) { auto l = c.ensure(layer, &e, 1); l.wait(); }

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  Synth S = make_synth("ms_test_experts.bin", 32, 30016, 66016, 120032);
  const int NE = S.L.n_experts;
  const uint64_t plb = per_layer_bytes(S.L);
  id<MTLDevice> dev = (__bridge id<MTLDevice>)ms::Device::get().device();
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)ms::Device::get().queue();

  // ---- T1: pools, misses, hits, data ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 14));
    printf("%s", c.describe().c_str());
    CHECK(c.pools().size() == 3, "pools %zu", c.pools().size());
    const int pa = c.layer_pool(0, 0), pb = c.layer_pool(3, 0), pc = c.layer_pool(5, 0);
    CHECK(pa != pb && pb != pc && pa != pc, "pool map");
    CHECK(c.layer_pool(1, 2) == pb && c.layer_pool(2, 2) == pa && c.layer_pool(4, 0) == pc && c.layer_pool(4, 2) == pa, "mixed layers");
    CHECK(c.pools()[pa].region < c.pools()[pb].region && c.pools()[pb].region < c.pools()[pc].region, "region order");
    CHECK(c.pools()[pa].demand == 7u * NE && c.pools()[pb].demand == 6u * NE && c.pools()[pc].demand == 5u * NE, "demand");
    for (auto& pi : c.pools()) CHECK(std::abs((double)pi.n_regions - pi.demand * 14.0 / NE) <= 1.0, "proportional split %d of %llu", pi.n_regions, (unsigned long long)pi.demand);
    CHECK(std::abs(c.cached_fraction() - 14.0 / NE) < 0.01, "cached fraction %.3f", c.cached_fraction());
    for (auto& pi : c.pools()) CHECK(pi.region % 16384 == 0, "region size");
    int ids[3] = {1, 5, 31};
    auto l1 = c.ensure(0, ids, 3);
    CHECK(l1.ticket > 0, "miss lease has a ticket");
    l1.wait();
    for (int i = 0; i < 3; i++) CHECK(check_view(S.L, 0, ids[i], l1.views[i]), "data L0 e%d", ids[i]);
    auto st = c.layer_stats(0);
    CHECK(st.misses == 3 && st.hits == 0, "stats miss %llu hit %llu", st.misses, st.hits);
    auto l2 = c.ensure(0, ids, 3);
    CHECK(l2.ticket == 0 && l2.ready(), "hit lease");
    st = c.layer_stats(0);
    CHECK(st.hits == 3, "hits %llu", st.hits);
    for (int l = 1; l < NL; l++) {
      int id2 = 9 + l;
      auto l3 = c.ensure(l, &id2, 1); l3.wait();
      CHECK(check_view(S.L, l, id2, l3.views[0]), "data L%d (mixed pools)", l);
    }
    CHECK(c.layer_stats(5).bytes_read >= 3ull * 120032, "bytes_read");
    // io_event GPU wait: command buffer waits for a fresh miss lease
    int id3 = 20;
    auto l4 = c.ensure(2, &id3, 1);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    l4.encode_wait((__bridge void*)cb);
    [cb commit]; [cb waitUntilCompleted];
    CHECK(l4.ready() && ((__bridge id<MTLSharedEvent>)c.io_event()).signaledValue >= l4.ticket, "io_event signaled");
    CHECK(check_view(S.L, 2, 20, l4.views[0]), "data after GPU wait");
  }

  printf("-- T2\n");
  // ---- T2: global LRU order; an eviction only takes victims owning a missing region size ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 8));
    const int pa = c.layer_pool(0, 0), pb = c.layer_pool(3, 0);
    const int nA = c.pools()[pa].n_regions, nB = c.pools()[pb].n_regions;
    const int capA = nA / 3, capB = nB / 3;   // layer 0 = AAA, layer 3 = BBB
    printf("  A %d regions (layer 0 cap %d), B %d regions (layer 3 cap %d), %d entry ids\n", nA, capA, nB, capB, c.n_slots());
    CHECK(c.unpinned_capacity(0) == capA && c.unpinned_capacity(3) == capB, "unpinned capacity");
    for (int e = 0; e < capA; e++) one(c, 0, e);
    CHECK(c.layer_stats(0).evictions == 0, "no eviction while filling");
    one(c, 0, 0);                                  // touch 0 -> LRU is now 1
    one(c, 0, capA);
    CHECK(!c.contains(0, 1) && c.contains(0, 0) && c.contains(0, 2), "LRU victim");
    CHECK(c.layer_stats(0).evictions == 1, "evictions");
    for (int e = 0; e < capB; e++) one(c, 3, e);   // only B regions: layer 0 untouched
    int present = 0; for (int e = 0; e <= capA; e++) present += c.contains(0, e);
    CHECK(present == capA && c.layer_stats(0).evictions == 1, "B loads evicted A-only entries (%d present)", present);
    one(c, 1, 0);   // A,A,B
    if (nB % 3 == 0) CHECK(!c.contains(3, 0) && c.contains(3, 1), "B victim = LRU of the B users");
    if (nA % 3 < 2) CHECK(!c.contains(0, 2) && c.contains(0, 3), "A victim = LRU of the A users");
    else CHECK(c.layer_stats(0).evictions == 1, "free A regions used first");
  }

  printf("-- T3\n");
  // ---- T3: in-use protection (lease + GPU completion) ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 8));
    int held[2] = {2, 3};
    auto lh = c.ensure(0, held, 2); lh.wait();
    for (int i = 0; i < 80; i++) one(c, i & 1, 4 + (i % 28));   // churn pool A
    CHECK(c.layer_stats(0).evictions + c.layer_stats(1).evictions > 0, "churn evicts");
    CHECK(c.contains(0, 2) && c.contains(0, 3), "held entries evicted");
    CHECK(check_view(S.L, 0, 2, lh.views[0]) && check_view(S.L, 0, 3, lh.views[1]), "held data intact");
    id<MTLSharedEvent> gate = [dev newSharedEvent];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    [cb encodeWaitForEvent:gate value:1];
    lh.release_after((__bridge void*)cb);
    [cb commit];
    for (int i = 0; i < 80; i++) one(c, i & 1, 4 + ((i * 7) % 28));
    CHECK(c.contains(0, 2) && c.contains(0, 3), "entries evicted while GPU command in flight");
    gate.signaledValue = 1;
    [cb waitUntilCompleted];
    usleep(20000);   // completion handler runs on a Metal thread
    for (int i = 0; i < 120; i++) one(c, 0, 4 + (i % 28));
    CHECK(!c.contains(0, 2) || !c.contains(0, 3), "entries never released after GPU completion");
  }

  printf("-- T4\n");
  // ---- T4: prefetch accounting ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 8));
    int pf[2] = {7, 8};
    CHECK(c.prefetch(2, pf, 2) == 2, "prefetch issued");
    c.drain();
    one(c, 2, 7);
    for (int i = 0; i < 60; i++) one(c, 2 + (i & 1), 9 + (i % 23));
    auto st = c.layer_stats(2);
    CHECK(st.prefetch_issued == 2 && st.prefetch_used == 1 && st.prefetch_wasted == 1, "prefetch iss %llu used %llu wasted %llu",
          st.prefetch_issued, st.prefetch_used, st.prefetch_wasted);
  }

  {   // demand on a still-loading prefetch (promotion path)
    ms::ExpertCache c(S.L, small_opts(plb, 14));
    int pf[4] = {20, 21, 22, 23};
    c.prefetch(3, pf, 4);
    auto l = c.ensure(3, pf, 4); l.wait();
    bool ok = true;
    for (int i = 0; i < 4; i++) ok &= check_view(S.L, 3, pf[i], l.views[i]);
    auto st = c.layer_stats(3);
    CHECK(ok && st.hits == 4 && st.misses == 0 && st.prefetch_used == 4, "promoted prefetch: hits %llu inflight %llu", st.hits, st.inflight_hits);
  }

  {   // young unused prefetches: prefetch never evicts them, demand only as a last resort (youngest, highest key first);
      // speculation cap: unused prefetches fill at most (unpinned - margin) of a pool
    ms::ExpertCache c(S.L, small_opts(plb, 8));
    const int nB = c.pools()[c.layer_pool(3, 0)].n_regions;   // layer 3 = BBB
    int pf4[4] = {0, 1, 2, 3};
    CHECK(c.prefetch(3, pf4, 4) == 4, "prefetch 4");
    c.drain();
    std::vector<int> held; for (int e = 10; e < 10 + nB / 3 - 4; e++) held.push_back(e);
    auto lh = c.ensure(3, held); lh.wait();
    int pf2[2] = {30, 31};
    CHECK(c.prefetch(3, pf2, 2) == 0, "prefetch evicted a leased entry or an unused prefetch");
    lh.release();
    CHECK(c.prefetch(3, pf2, 2) == 2, "prefetch should evict the LRU used entries");
    c.drain();
    CHECK(c.contains(3, 0) && c.contains(3, 3) && !c.contains(3, 10) && !c.contains(3, 11) && c.contains(3, 12), "prefetch victims");
    std::vector<int> held2(held.begin() + 2, held.end());
    auto lh2 = c.ensure(3, held2); lh2.wait();
    one(c, 3, 25);   // nothing but unused prefetches is evictable: the youngest / highest key goes
    CHECK(c.contains(3, 25) && !c.contains(3, 31) && c.contains(3, 30) && c.contains(3, 0) && c.contains(3, 3), "demand last-resort victim");
    CHECK(c.layer_stats(3).prefetch_wasted == 1, "wasted %llu", c.layer_stats(3).prefetch_wasted);
    lh2.release();
    ms::ExpertCache c2(S.L, small_opts(plb, 8));
    std::vector<int> all; for (int e = 0; e < NE; e++) all.push_back(e);
    const int expect = (nB - std::min(nB / 4, 64 * 3)) / 3;
    CHECK(c2.prefetch(3, all.data(), NE) == expect, "speculation cap: expected %d prefetches", expect);
  }

  printf("-- T5\n");
  // ---- T5: LFU keeps a hot expert, LRU does not ----
  for (int pol = 0; pol < 2; pol++) {
    auto o = small_opts(plb, 8);
    o.policy = pol ? ms::PolicyKind::LFU : ms::PolicyKind::LRU;
    ms::ExpertCache c(S.L, o);
    for (int i = 0; i < 10; i++) one(c, 0, 0);
    for (int i = 0; i < 60; i++) one(c, 1, i % 32);
    if (pol) CHECK(c.contains(0, 0), "LFU evicted the hot expert");
    else CHECK(!c.contains(0, 0), "LRU should have evicted expert 0");
  }

  printf("-- T6\n");
  // ---- T6: static frequency-pinned set from a routing-statistics file ----
  {
    const char* tmp = getenv("TMPDIR");
    std::string sp = std::string(tmp ? tmp : "/tmp") + "/ms_test_routing.txt";
    FILE* f = fopen(sp.c_str(), "w");
    fprintf(f, "# test\n");
    for (int l = 0; l < NL; l++) { for (int e = 0; e < NE; e++) fprintf(f, "%d ", e < 3 ? 1000 - e : (e < 6 ? 1 : 0)); fprintf(f, "\n"); }
    fclose(f);
    auto o = small_opts(plb, 20);
    o.policy = ms::PolicyKind::Static; o.stats_path = sp; o.pin_fraction = 0.12;
    ms::ExpertCache c(S.L, o);
    c.drain();
    printf("%s", c.describe().c_str());
    const int P = (int)c.pools().size();
    std::vector<int> used(P, 0);
    std::vector<std::pair<int, int>> pinned;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (c.is_pinned(l, e)) {
      pinned.push_back({l, e});
      for (int k = 0; k < 3; k++) used[c.layer_pool(l, k)]++;
      CHECK(e < 6, "pinned a zero-count expert L%d e%d", l, e);
    }
    printf("  static: %zu pinned\n", pinned.size());
    CHECK((int)pinned.size() == c.n_pinned() && !pinned.empty(), "pinned %zu / %d", pinned.size(), c.n_pinned());
    for (int pp = 0; pp < P; pp++) CHECK(used[pp] <= (int)(0.12 * c.pools()[pp].n_regions), "pool %d pinned %d regions > budget", pp, used[pp]);
    for (int pp = 0; pp < P; pp++) CHECK(used[pp] >= (int)(0.12 * c.pools()[pp].n_regions) - 3, "pool %d budget unused (%d)", pp, used[pp]);
    CHECK(c.is_pinned(0, 0), "top expert (count/bytes) not pinned");
    for (int i = 0; i < 200; i++) one(c, i % NL, 6 + (i % 26));
    for (auto& pe : pinned) CHECK(c.contains(pe.first, pe.second) && c.is_pinned(pe.first, pe.second), "pinned L%d e%d evicted", pe.first, pe.second);
    std::vector<double> rs;
    CHECK(ms::ExpertCache::read_routing_stats(sp, NL, NE, rs) && rs[0] == 1000, "read stats");
    std::string wp = std::string(tmp ? tmp : "/tmp") + "/ms_test_routing_out.txt";
    CHECK(c.write_routing_stats(wp) && ms::ExpertCache::read_routing_stats(wp, NL, NE, rs), "write stats");
  }

  printf("-- T7\n");
  // ---- T7: alignment / chunking variants all return identical data ----
  for (uint32_t align : {1u, 4096u, 16384u}) for (size_t chunk : {(size_t)0, (size_t)16384, (size_t)(1 << 20)}) {
    auto o = small_opts(plb, 14); o.align = align; o.read_chunk = chunk;
    ms::ExpertCache c(S.L, o);
    bool ok = true;
    for (int l : {3, 4}) {
      int ids[4] = {0, 13, 30, 31};
      auto le = c.ensure(l, ids, 4); le.wait();
      for (int i = 0; i < 4; i++) ok &= check_view(S.L, l, ids[i], le.views[i]);
    }
    CHECK(ok, "align %u chunk %zu", align, chunk);
  }

  {   // wired arena (mlock) variant, and several small MTLBuffers per pool
    auto o = small_opts(plb, 14); o.wire = true; o.max_buffer_bytes = 1 << 20;
    ms::ExpertCache c(S.L, o);
    CHECK(c.buffers().size() > 6, "buffers %zu", c.buffers().size());
    bool ok = true;
    for (int l = 0; l < NL; l++) { int ids[2] = {4, 17}; auto le = c.ensure(l, ids, 2); le.wait(); ok &= check_view(S.L, l, 4, le.views[0]) && check_view(S.L, l, 17, le.views[1]); }
    CHECK(ok, "wired arena data");
  }

  printf("-- T8\n");
  // ---- T8: bulk stream, CPU consumer ----
  {
    auto o = small_opts(plb, 14); o.ring_slots = 4;
    ms::ExpertCache c(S.L, o);
    one(c, 1, 0); one(c, 1, 10);
    std::vector<int> all; for (int e = NE - 1; e >= 0; e--) all.push_back(e);
    auto st = c.stream(1, all.data(), (int)all.size());
    CHECK(st->items().size() == 32 && st->n_cached() == 2, "stream items %zu cached %zu", st->items().size(), st->n_cached());
    bool ok = true; int last = -1;
    for (size_t i = 0; i < st->items().size(); i++) {
      st->wait_ready(i);
      auto& it = st->items()[i];
      ok &= check_view(S.L, 1, it.expert, it.view);
      if (!it.cached) { ok &= it.expert > last; last = it.expert; }
      st->consumed(i);
    }
    CHECK(ok, "stream data / file order");
    CHECK(c.layer_stats(1).streamed == 30, "streamed %llu", c.layer_stats(1).streamed);
    st.reset();
    // second stream on another layer (largest entry) reuses the ring
    std::vector<int> some = {3, 4, 5, 6, 7, 8, 9, 11, 12};
    auto s2 = c.stream(5, some.data(), (int)some.size());
    for (size_t i = 0; i < s2->items().size(); i++) { s2->wait_ready(i); ok &= check_view(S.L, 5, s2->items()[i].expert, s2->items()[i].view); s2->consumed(i); }
    CHECK(ok, "second stream");
    c.drain();
  }

  printf("-- T9\n");
  // ---- T9: bulk stream, GPU consumer (encode_wait_ready / encode_consumed in one command buffer, ring wraps) ----
  {
    auto o = small_opts(plb, 14); o.ring_slots = 3;
    ms::ExpertCache c(S.L, o);
    one(c, 2, 5);
    NSError* err = nil;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void t_peek(device const uint* s [[buffer(0)]], device uint* o [[buffer(1)]], constant uint& i [[buffer(2)]],"
                     " uint t [[thread_position_in_grid]]) { o[i*4+t] = s[t]; }\n";
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"t_peek"] error:&err];
    CHECK(pso, "pso");
    std::vector<int> ids; for (int e = 0; e < NE; e++) ids.push_back(e);
    auto st = c.stream(2, ids.data(), (int)ids.size());
    id<MTLBuffer> out = [dev newBufferWithLength:NE * 16 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    for (size_t i = 0; i < st->items().size(); i++) {
      st->encode_wait_ready((__bridge void*)cb, i);
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:pso];
      const ms::Buffer& b = st->items()[i].view.mat[1];
      [ce setBuffer:(__bridge id<MTLBuffer>)b.buf offset:b.off atIndex:0];
      [ce setBuffer:out offset:0 atIndex:1];
      uint32_t ii = (uint32_t)i; [ce setBytes:&ii length:4 atIndex:2];
      [ce dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
      [ce endEncoding];
      st->encode_consumed((__bridge void*)cb, i);
    }
    st->cached().release_after((__bridge void*)cb);
    [cb commit]; [cb waitUntilCompleted];
    CHECK(cb.status == MTLCommandBufferStatusCompleted, "gpu stream cb status %ld", (long)cb.status);
    bool ok = true;
    for (size_t i = 0; i < st->items().size(); i++) {
      const ms::ExpertMat& m = S.L.layers[2].mat[1];
      uint64_t off = m.base + (uint64_t)st->items()[i].expert * m.stride;
      const uint8_t* g = (const uint8_t*)out.contents + i * 16;
      for (int j = 0; j < 16; j++) ok &= g[j] == pat(off + j);
    }
    CHECK(ok, "GPU saw the streamed bytes");
  }

  printf("-- T11\n");
  // ---- T11: randomized stress, 2 threads: ensure / prefetch / pin / unpin / pin_resident / unpin_resident; every view
  //      checked against pread while its lease is held; at the end every resident expert is checked ----
  for (int pol = 0; pol < 2; pol++) {
    auto o = small_opts(plb, 16);
    o.policy = pol ? ms::PolicyKind::LFU : ms::PolicyKind::LRU;
    ms::ExpertCache c(S.L, o);
    std::atomic<long> verified{0};
    auto worker = [&](int tid) {
      std::mt19937 rng(1234 + tid + 17 * pol);
      std::vector<std::pair<int, int>> mypins;
      for (int it = 0; it < 1500; it++) {
        const int r = rng() % 100, l = rng() % NL;
        if (r < 70) {
          int n = 1 + rng() % 6, ids[6];
          for (int i = 0; i < n; i++) ids[i] = rng() % NE;
          auto le = c.ensure(l, ids, n);
          le.wait();
          for (int i = 0; i < n; i++) CHECK(check_view_pread(S, l, ids[i], le.views[i]), "stress view L%d e%d", l, ids[i]);
          verified += n;
          if (rng() % 4 == 0) std::this_thread::yield();
        } else if (r < 85) {
          int n = 1 + rng() % 8, ids[8];
          for (int i = 0; i < n; i++) ids[i] = rng() % NE;
          c.prefetch(l, ids, n);
        } else if (r < 92) {
          if (mypins.size() < 3) { int e = rng() % NE; if (c.pin(l, &e, 1)) mypins.push_back({l, e}); }
          else { c.unpin(mypins[0].first, &mypins[0].second, 1); mypins.erase(mypins.begin()); }
        } else if (tid == 0 && r < 96) {
          c.pin_resident(4);
        } else if (tid == 0) {
          c.unpin_resident();
        }
      }
      for (auto& pe : mypins) c.unpin(pe.first, &pe.second, 1);
    };
    std::thread t1(worker, 1);
    worker(0);
    t1.join();
    c.unpin_resident();
    c.drain();
    CHECK(c.n_pinned() == 0, "pinned left %d", c.n_pinned());
    int resident = 0; bool ok = true;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (c.contains(l, e)) {
      resident++;
      auto le = c.ensure(l, &e, 1); le.wait();
      ok &= check_view_pread(S, l, e, le.views[0]);
    }
    CHECK(ok, "final resident data");
    auto ts = c.total_stats();
    printf("  %s: %ld views verified, %d resident at end (%d entry ids), %llu misses, %llu evictions, %llu prefetches\n", pol ? "lfu" : "lru",
           verified.load(), resident, c.n_slots(), (unsigned long long)ts.misses, (unsigned long long)ts.evictions, (unsigned long long)ts.prefetch_issued);
    CHECK(ts.evictions > 100, "stress did not evict");
  }

  printf("-- T12\n");
  // ---- T12: prefill simulation, 192 experts per layer: pin_resident between chunks, groups of group_cap(l) with two
  //      leases alive (released by GPU command-buffer completion), next-layer prefetch of every expert ----
  {
    Synth S2 = make_synth("ms_test_experts192.bin", 192, 2016, 18016, 34016);
    const int NE2 = 192;
    const uint64_t plb2 = per_layer_bytes(S2.L);
    for (int keep : {64, 256}) {
      ms::ExpertCacheOptions o = small_opts(plb2, NE2 / 2);   // half of all experts
      ms::ExpertCache c(S2.L, o);
      if (keep == 64) printf("%s", c.describe().c_str());
      std::vector<int> all(NE2); for (int e = 0; e < NE2; e++) all[e] = e;
      std::mt19937 rng(77 + keep);
      std::vector<std::pair<int, int>> pinned;
      int last_pinned = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (int chunk = 0; chunk < 4; chunk++) {
        const auto st0 = c.total_stats();
        for (int l = 0; l < NL; l++) {
          if (l == 0) c.prefetch(0, all.data(), NE2);
          if (l + 1 < NL) c.prefetch(l + 1, all.data(), NE2);
          std::vector<int> used;
          for (int e = 0; e < NE2; e++) if (rng() % 16) used.push_back(e);   // ~94% of the experts
          const int G = c.group_cap(l);
          CHECK(G >= 8 && 3 * G <= c.unpinned_capacity(l), "group_cap %d vs capacity %d (layer %d)", G, c.unpinned_capacity(l), l);
          const int ng = ((int)used.size() + G - 1) / G;
          auto gids = [&](int gi) { return std::vector<int>(used.begin() + gi * G, used.begin() + std::min((int)used.size(), (gi + 1) * G)); };
          std::vector<ms::ExpertLease> leases(ng);
          std::vector<id<MTLCommandBuffer>> cbs;
          leases[0] = c.ensure(l, gids(0));
          for (int gi = 0; gi < ng; gi++) {
            if (gi + 1 < ng) leases[gi + 1] = c.ensure(l, gids(gi + 1));
            if (l + 1 < NL) c.prefetch(l + 1, all.data(), NE2);   // top up the next layer as this one's prefetches get used
            leases[gi].wait();
            auto ids = gids(gi);
            bool ok = true;
            for (size_t j = 0; j < ids.size(); j++) ok &= check_view(S2.L, l, ids[j], leases[gi].views[j]);
            CHECK(ok, "prefill group data L%d g%d", l, gi);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            leases[gi].release_after((__bridge void*)cb);
            [cb commit];
            cbs.push_back(cb);
          }
          for (auto& pe : pinned) if (!(c.contains(pe.first, pe.second) && c.is_pinned(pe.first, pe.second))) { CHECK(false, "pinned L%d e%d evicted", pe.first, pe.second); break; }
          for (auto cb : cbs) [cb waitUntilCompleted];
        }
        usleep(5000);   // completion handlers
        const auto st1 = c.total_stats();
        const long loads = (long)(st1.misses - st0.misses + st1.prefetch_issued - st0.prefetch_issued);
        printf("  keep %d chunk %d: loads %ld (%.1f%% of %d experts; %d were pinned), prefetch used %llu / issued %llu / wasted %llu, G(l0..5) =",
               keep, chunk, loads, 100.0 * loads / (NL * NE2), NL * NE2, last_pinned, (unsigned long long)(st1.prefetch_used - st0.prefetch_used),
               (unsigned long long)(st1.prefetch_issued - st0.prefetch_issued), (unsigned long long)(st1.prefetch_wasted - st0.prefetch_wasted));
        for (int l = 0; l < NL; l++) printf(" %d", c.group_cap(l));
        printf("\n");
        CHECK(loads <= (long)((NL * NE2 - last_pinned) * (keep == 64 ? 1.05 : 1.12)), "chunk %d loads %ld with %d pinned", chunk, loads, last_pinned);
        int np = c.pin_resident(keep);
        last_pinned = c.n_pinned();
        CHECK(np > 0 || chunk > 0, "pin_resident pinned nothing");
        for (int l = 0; l < NL; l++) CHECK(c.unpinned_capacity(l) >= 24, "layer %d unpinned capacity %d", l, c.unpinned_capacity(l));
        pinned.clear();
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE2; e++) if (c.is_pinned(l, e)) pinned.push_back({l, e});
        CHECK((int)pinned.size() == last_pinned, "pinned list");
        if (keep == 64 && chunk >= 1) CHECK(last_pinned > c.n_slots() * 3 / 4, "pinned only %d of %d entry ids", last_pinned, c.n_slots());
      }
      // decode: release the resident pins, ensure 6 per layer
      c.unpin_resident();
      CHECK(c.n_pinned() == 0, "unpin_resident");
      for (int t = 0; t < 30; t++) for (int l = 0; l < NL; l++) {
        int ids[6]; for (int i = 0; i < 6; i++) ids[i] = rng() % NE2;
        auto le = c.ensure(l, ids, 6); le.wait();
        bool ok = true;
        for (int i = 0; i < 6; i++) ok &= check_view(S2.L, l, ids[i], le.views[i]);
        CHECK(ok, "decode data");
      }
      c.drain();
      printf("  keep %d: %.0f ms\n", keep, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    unlink(S2.path.c_str());
  }

  printf("-- T10\n");
  // ---- T10: real GGUF experts (mixed region pools, a few MB read) + describe() at a production arena (dry_io) ----
  std::string mp = ms::Gguf::default_model_path();
  if (ms::Gguf::exists(mp)) {
    ms::Gguf g(mp);
    ms::ExpertLayout L = ms::ExpertLayout::from_gguf(g);
    const int NE = L.n_experts;   // Unsloth UD-IQ1_S: 384 experts, IQ1_S + IQ2_XXS; GSQ-RCO Coder: 192, IQ1_M + IQ2_XXS + IQ2_XS
    {
      ms::ExpertCacheOptions o; o.arena_bytes = (size_t)(34.5 * 1073741824.0); o.dry_io = true; o.prefault = false; o.io_threads = 1;
      ms::ExpertCache c(L, o);
      printf("arena 34.5 GiB (dry): %s", c.describe().c_str());
      CHECK(c.pools().size() == (NE == 384 ? 2u : 3u), "real pools %zu", c.pools().size());
      if (NE == 192) {
        CHECK(c.cached_fraction() > 0.5 && c.cached_fraction() < 0.6, "Coder cached fraction %.3f", c.cached_fraction());
        bool sizes = true;
        for (auto& pi : c.pools()) sizes &= pi.region == 2605056 || pi.region == 3063808 || pi.region == 3440640;
        CHECK(sizes, "Coder region sizes");
      }
    }
    ms::ExpertCacheOptions o; o.arena_bytes = 2ull << 30; o.io_threads = 8; o.prefault = false;
    ms::ExpertCache c(L, o);
    printf("%s", c.describe().c_str());
    struct { int l, e; } want[] = {{0, 0}, {0, NE - 1}, {7, 5}, {17, 100}, {23, 42}, {39, NE - 1}, {39, 3}};
    for (auto w : want) {
      auto l = c.ensure(w.l, &w.e, 1); l.wait();
      bool ok = true;
      for (int k = 0; k < 3; k++) {
        const ms::ExpertMat& m = L.layers[w.l].mat[k];
        std::vector<char> ref(m.len);
        int fd = -1;
        for (int s = 0; s < g.n_splits(); s++) if (g.fd_nocache(s) == m.fd) fd = g.fd(s);
        ok &= ms::pread_full(fd, ref.data(), m.len, m.base + (uint64_t)w.e * m.stride) == m.len;
        ok &= memcmp(ref.data(), l.views[0].host[k], m.len) == 0;
        ok &= l.views[0].type[k] == m.type;
      }
      CHECK(ok, "real expert L%d e%d (%s/%s/%s)", w.l, w.e, ms::type_name(L.layers[w.l].mat[0].type), ms::type_name(L.layers[w.l].mat[1].type),
            ms::type_name(L.layers[w.l].mat[2].type));
    }
    printf("%s", c.stats_report().c_str());
  } else {
    printf("real-model part skipped (no model)\n");
  }

  unlink(S.path.c_str());
  printf("test_store_experts: %s\n", fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
