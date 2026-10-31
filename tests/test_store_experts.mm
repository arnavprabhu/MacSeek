// ExpertCache tests on a synthetic expert file (fast, exact data checks) + a few real experts from the GGUF.
// Covers: miss/hit/data, size classes, LRU order, in-use protection (leases and GPU command-buffer completion),
// prefetch accounting, LFU aging, static frequency pinning, io_event GPU waits, bulk stream (CPU + GPU protocol),
// alignment / chunking variants.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "core/metal.hpp"
#include "store/experts.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static inline uint8_t pat(uint64_t x) { x ^= x >> 7; x *= 0x9E3779B97F4A7C15ULL; return (uint8_t)(x >> 56); }

static const int NL = 4, NE = 32;
static const uint32_t LEN_A = 50016, LEN_B = 66016;   // multiples of 32 like GGUF slabs

struct Synth { std::string path; int fd = -1; ms::ExpertLayout L; };

static Synth make_synth() {
  Synth s;
  const char* tmp = getenv("TMPDIR");
  s.path = std::string(tmp ? tmp : "/tmp") + "/ms_test_experts.bin";
  s.L.n_layers = NL; s.L.n_experts = NE;
  uint64_t off = 4096 + 96;
  for (int l = 0; l < NL; l++) {
    ms::ExpertLayerLayout ll;
    for (int k = 0; k < 3; k++) {
      ms::ExpertMat& m = ll.mat[k];
      m.len = (l >= 2 && k == 2) ? LEN_B : LEN_A;
      m.stride = m.len; m.base = off; m.type = (l >= 2 && k == 2) ? ms::GType::IQ2_XXS : ms::GType::IQ1_S;
      off += (uint64_t)m.len * NE + 96;
    }
    s.L.layers.push_back(ll);
  }
  std::vector<uint8_t> buf(off);
  for (uint64_t i = 0; i < off; i++) buf[i] = pat(i);
  FILE* f = fopen(s.path.c_str(), "wb"); fwrite(buf.data(), 1, buf.size(), f); fclose(f);
  s.fd = open(s.path.c_str(), O_RDONLY);
  fcntl(s.fd, F_NOCACHE, 1);
  for (auto& ll : s.L.layers) for (auto& m : ll.mat) { m.fd = s.fd; m.file_size = off; }
  return s;
}

static bool check_view(const ms::ExpertLayout& L, int layer, int e, const ms::ExpertView& v) {
  for (int k = 0; k < 3; k++) {
    const ms::ExpertMat& m = L.layers[layer].mat[k];
    uint64_t off = m.base + (uint64_t)e * m.stride;
    const uint8_t* p = (const uint8_t*)v.host[k];
    if ((const void*)p != v.mat[k].cpu()) return false;
    if (v.mat[k].size != m.len) return false;
    for (uint32_t j = 0; j < m.len; j++) if (p[j] != pat(off + j)) { fprintf(stderr, "  data mismatch L%d e%d mat%d byte %u\n", layer, e, k, j); return false; }
  }
  return true;
}

static uint64_t per_layer_bytes(const ms::ExpertLayout& L) {
  ms::ExpertCacheOptions o; o.arena_bytes = 64 << 20; o.io_threads = 2;
  ms::ExpertCache probe(L, o);
  uint64_t s = 0;
  for (int l = 0; l < L.n_layers; l++) s += probe.classes()[probe.layer_class(l)].slot_bytes;
  return s;
}

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
  Synth S = make_synth();
  const uint64_t plb = per_layer_bytes(S.L);
  id<MTLDevice> dev = (__bridge id<MTLDevice>)ms::Device::get().device();
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)ms::Device::get().queue();

  // ---- T1: classes, misses, hits, data ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 14));
    printf("%s", c.describe().c_str());
    CHECK(c.classes().size() == 2, "classes %zu", c.classes().size());
    CHECK(c.layer_class(0) == c.layer_class(1) && c.layer_class(2) == c.layer_class(3) && c.layer_class(0) != c.layer_class(2), "class map");
    for (auto& ci : c.classes()) CHECK(ci.n_slots == 28, "slots %d", ci.n_slots);
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
    int id2 = 9;
    auto l3 = c.ensure(3, &id2, 1); l3.wait();
    CHECK(check_view(S.L, 3, 9, l3.views[0]), "data L3 (class B)");
    CHECK(c.layer_stats(3).bytes_read >= 2 * LEN_A + LEN_B, "bytes_read");
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
  // ---- T2: LRU order ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 14));   // 28 slots for layers {0,1}
    for (int e = 0; e < 28; e++) one(c, 0, e);
    one(c, 0, 0);                                  // touch 0 -> LRU is now 1
    one(c, 0, 28);
    CHECK(!c.contains(0, 1) && c.contains(0, 0) && c.contains(0, 2), "LRU victim");
    CHECK(c.layer_stats(0).evictions == 1, "evictions");
  }

  printf("-- T3\n");
  // ---- T3: in-use protection (lease + GPU completion) ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 14));
    int held[2] = {2, 3};
    auto lh = c.ensure(0, held, 2); lh.wait();
    for (int i = 0; i < 80; i++) one(c, i & 1, 4 + (i % 28));   // churn the class
    CHECK(c.contains(0, 2) && c.contains(0, 3), "held slots evicted");
    CHECK(check_view(S.L, 0, 2, lh.views[0]) && check_view(S.L, 0, 3, lh.views[1]), "held data intact");
    // hand the lease to a command buffer that is blocked on an event until we signal it
    id<MTLSharedEvent> gate = [dev newSharedEvent];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    [cb encodeWaitForEvent:gate value:1];
    lh.release_after((__bridge void*)cb);
    [cb commit];
    for (int i = 0; i < 80; i++) one(c, i & 1, 4 + ((i * 7) % 28));
    CHECK(c.contains(0, 2) && c.contains(0, 3), "slots evicted while GPU command in flight");
    gate.signaledValue = 1;
    [cb waitUntilCompleted];
    usleep(20000);   // completion handler runs on a Metal thread
    for (int i = 0; i < 120; i++) one(c, 0, 4 + (i % 28));
    CHECK(!c.contains(0, 2) || !c.contains(0, 3), "slots never released after GPU completion");
  }

  printf("-- T4\n");
  // ---- T4: prefetch accounting ----
  {
    ms::ExpertCache c(S.L, small_opts(plb, 14));
    int pf[2] = {7, 8};
    CHECK(c.prefetch(2, pf, 2) == 2, "prefetch issued");
    c.drain();
    one(c, 2, 7);
    for (int i = 0; i < 60; i++) one(c, 2 + (i & 1), 9 + (i % 23));
    auto st = c.layer_stats(2);
    CHECK(st.prefetch_issued == 2 && st.prefetch_used == 1 && st.prefetch_wasted == 1, "prefetch iss %llu used %llu wasted %llu",
          st.prefetch_issued, st.prefetch_used, st.prefetch_wasted);
  }

  printf("-- T5\n");
  // ---- T5: LFU keeps a hot expert, LRU does not ----
  for (int pol = 0; pol < 2; pol++) {
    auto o = small_opts(plb, 14);
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
    o.policy = ms::PolicyKind::Static; o.stats_path = sp; o.pin_fraction = 0.12;  // quota 4 per class of ~40
    ms::ExpertCache c(S.L, o);
    c.drain();
    printf("%s", c.describe().c_str());
    int pinned = c.n_pinned();
    int expect = 0;
    for (auto& ci : c.classes()) expect += std::min((int)(0.12 * ci.n_slots), 12);
    CHECK(pinned == expect, "pinned %d expected %d", pinned, expect);
    for (int i = 0; i < 200; i++) one(c, i % 4, 6 + (i % 26));
    // the 4 highest-count keys of each class (experts 0 and 1 of both layers) stay
    for (int l = 0; l < NL; l++) CHECK(c.contains(l, 0) && c.contains(l, 1), "pinned experts of layer %d evicted", l);
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
    int ids[4] = {0, 13, 30, 31};
    auto l = c.ensure(3, ids, 4); l.wait();
    bool ok = true;
    for (int i = 0; i < 4; i++) ok &= check_view(S.L, 3, ids[i], l.views[i]);
    CHECK(ok, "align %u chunk %zu", align, chunk);
  }

  {   // wired arena (mlock) variant
    auto o = small_opts(plb, 14); o.wire = true;
    ms::ExpertCache c(S.L, o);
    int ids[2] = {4, 17};
    auto l = c.ensure(0, ids, 2); l.wait();
    CHECK(check_view(S.L, 0, 4, l.views[0]) && check_view(S.L, 0, 17, l.views[1]), "wired arena data");
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
    // second stream on another layer reuses the ring
    std::vector<int> some = {3, 4, 5, 6, 7, 8, 9, 11, 12};
    auto s2 = c.stream(3, some.data(), (int)some.size());
    for (size_t i = 0; i < s2->items().size(); i++) { s2->wait_ready(i); ok &= check_view(S.L, 3, s2->items()[i].expert, s2->items()[i].view); s2->consumed(i); }
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

  printf("-- T10\n");
  // ---- T10: real GGUF experts (3 size classes, ~40 MB read) ----
  std::string mp = ms::Gguf::default_model_path();
  if (ms::Gguf::exists(mp)) {
    ms::Gguf g(mp);
    ms::ExpertLayout L = ms::ExpertLayout::from_gguf(g);
    ms::ExpertCacheOptions o; o.arena_bytes = 1ull << 30; o.io_threads = 8;
    ms::ExpertCache c(L, o);
    printf("%s", c.describe().c_str());
    CHECK(c.classes().size() == 3, "real classes %zu", c.classes().size());
    struct { int l, e; } want[] = {{0, 0}, {0, 383}, {25, 7}, {33, 100}, {39, 383}};
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
      CHECK(ok, "real expert L%d e%d", w.l, w.e);
    }
    printf("%s", c.stats_report().c_str());
  } else {
    printf("real-model part skipped (no model)\n");
  }

  unlink(S.path.c_str());
  printf("test_store_experts: %s\n", fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
