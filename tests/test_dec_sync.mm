// Decode handshake protocol (MACSEEK_DEC_SYNC, docs/modules/dec_sync.md), synthetic end to end:
//  A) the split MoE encodes are equivalent to the fused table MoE: SharedGu + AfterSharedGu bit-identical,
//     Shared + Routed equal up to summation order. Pieces run in separate encoders with fences and a real
//     signal -> CPU -> wait handshake in between (the shape used by the decode path), on random IQ1_S/IQ2_XXS/Q6_K/Q8_0
//     experts addressed through a GPU-address table.
//  B) the per-layer loop: 40 layers x tokens in ONE command buffer per token. GPU router -> rdy -> (gap work) -> go ->
//     table MoE, with the REAL ExpertCache on a synthetic expert file (small arena: misses, evictions, prefetch
//     pressure), random extra miss delays (simulated SSD), leases held until the token's command buffer completes.
//     The GPU checks every expert matrix it reads against the ids it routed (stale/evicted slot => bad), and that the
//     gap work it consumes is the current layer's (fence ordering); the CPU re-simulates the whole chain.
//     Timing: ms/token for event vs sgu-style overlap.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "core/env.hpp"
#include "core/metal.hpp"
#include "store/experts.hpp"
#include "quant_test_util.hpp"

using namespace ms;
namespace mq = ms::q;
using mq::QType;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void busy_us(double us) { double t = now_us(); while (now_us() - t < us) {} }
static uint32_t mixh(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

// ------------------------------------------------------------------------------------------------ A: kernel split
static void test_split() {
  const int D = mq::kExpertDim, I = mq::kExpertInter;
  struct Case { QType gu, down, sh; };
  const Case cases[] = {{QType::IQ1_S, QType::IQ1_S, QType::Q6_K}, {QType::IQ1_S, QType::IQ2_XXS, QType::Q8_0},
                        {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q6_K}};
  auto& dev = Device::get();
  id<MTLDevice> mdev = (__bridge id<MTLDevice>)dev.device();
  id<MTLSharedEvent> rdy = [mdev newSharedEvent], go = [mdev newSharedEvent];
  id<MTLFence> fa = [mdev newFence], fg = [mdev newFence];
  uint64_t ev = 0;
  for (const Case& c : cases) {
    std::vector<std::vector<uint8_t>> M(7 * 3);
    size_t total = 0;
    for (int e = 0; e < 7; e++)
      for (int j = 0; j < 3; j++) {
        const QType t = e == 6 ? c.sh : (j == 2 ? c.down : c.gu);
        M[e * 3 + j] = ref::random_blocks(t, (int64_t)I * D, 500 + e * 3 + j);
        total += M[e * 3 + j].size() + 64;
      }
    Buffer arena = dev.alloc(total + 4096, "arena");
    id<MTLBuffer> ab = (__bridge id<MTLBuffer>)arena.buf;
    Buffer tbl = dev.alloc(6 * 3 * 8, "tbl");
    mq::MoeDecodeArgs a;
    a.t_gate_up = c.gu; a.t_down = c.down; a.t_shared = c.sh; a.n_routed = 6; a.has_shared = true;
    size_t off = 16;
    Buffer sh[3];
    for (int e = 6; e >= 0; e--)   // shuffled placement
      for (int j = 0; j < 3; j++) {
        auto& v = M[e * 3 + j];
        memcpy((uint8_t*)arena.cpu() + off, v.data(), v.size());
        if (e == 6) sh[j] = arena.sub(off, v.size());
        else {
          ((uint64_t*)tbl.cpu())[e * 3 + j] = ab.gpuAddress + off;
          Buffer b = arena.sub(off, v.size());
          if (j == 0) a.slot[e].gate = b; else if (j == 1) a.slot[e].up = b; else a.slot[e].down = b;
        }
        off += (v.size() + 63) / 64 * 64 + 16;
      }
    a.shared = mq::ExpertSlot{sh[0], sh[1], sh[2]};
    auto x = qt::randvec(D, 91);
    auto y0 = qt::randvec(D, 92);
    std::vector<float> w = {0.31f, 0.22f, 0.17f, 0.12f, 0.1f, 0.08f};
    Buffer bx = qt::upload(x.data(), D * 4), bw = qt::upload(w.data(), 24);
    Buffer scratch = dev.alloc(mq::moe_decode_scratch_bytes());
    a.weights = bw; a.x = bx; a.scratch = scratch;
    std::vector<float> ys[4];
    for (int mode = 0; mode < 4; mode++) {
      Buffer by = qt::upload(y0.data(), D * 4);
      a.y = by;
      memset(scratch.cpu(), 0xff, scratch.size);
      Enc e;
      if (mode == 3) {   // slot-bound kernel (CPU-reference-checked by test_quant_moe)
        mq::encode_moe_decode(e, a);
        e.wait();
      } else if (mode == 0) {
        e.use_resources({arena});
        mq::encode_moe_decode_table(e, a, tbl);
        e.wait();
      } else {
        // the decode-path shape: [...] updateFence | signal rdy | waitFence piece1 updateFence | wait go | waitFence piece2
        const uint64_t v = ++ev;
        id<MTLComputeCommandEncoder> ce = (__bridge id<MTLComputeCommandEncoder>)e.encoder();
        [ce updateFence:fa];
        e.between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:rdy value:v]; });
        [(__bridge id<MTLComputeCommandEncoder>)e.encoder() waitForFence:fa];
        mq::encode_moe_decode_table_part(e, a, tbl, mode == 1 ? mq::MoePart::SharedGu : mq::MoePart::Shared);
        [(__bridge id<MTLComputeCommandEncoder>)e.encoder() updateFence:fg];
        e.between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:go value:v]; });
        [(__bridge id<MTLComputeCommandEncoder>)e.encoder() waitForFence:fg];
        e.use_resources({arena});
        mq::encode_moe_decode_table_part(e, a, tbl, mode == 1 ? mq::MoePart::AfterSharedGu : mq::MoePart::Routed);
        e.commit();
        while (rdy.signaledValue < v) {}
        go.signaledValue = v;
        e.wait();
      }
      ys[mode].assign((float*)by.cpu(), (float*)by.cpu() + D);
      dev.release(by);
    }
    const bool bit = memcmp(ys[0].data(), ys[1].data(), D * 4) == 0;
    double mx = 0, mag = 0, mx3 = 0;
    for (int i = 0; i < D; i++) {
      mx = std::max(mx, (double)std::fabs(ys[0][i] - ys[2][i])); mx3 = std::max(mx3, (double)std::fabs(ys[0][i] - ys[3][i]));
      mag = std::max(mag, (double)std::fabs(ys[0][i]));
    }
    printf("  split gu=%-7s down=%-7s shared=%-4s : sgu bit-identical %s | shared max|dy| %.2e | table vs slot kernel %.2e (|y| %.2e)\n",
           mq::type_name(c.gu), mq::type_name(c.down), mq::type_name(c.sh), bit ? "yes" : "NO", mx, mx3, mag);
    CHECK(mx3 <= 1e-5 * std::max(1.0, mag), "table MoE differs from the slot-bound MoE: %g", mx3);
    CHECK(bit, "SharedGu+AfterSharedGu differs from the fused table MoE");
    CHECK(mx <= 1e-5 * std::max(1.0, mag), "Shared+Routed differs beyond rounding: %g", mx);
    dev.release(arena); dev.release(tbl); dev.release(bx); dev.release(bw); dev.release(scratch);
  }
}

// ------------------------------------------------------------------------------------------------ B: decode loop
namespace {
constexpr int NL = 40, NE = 48;
constexpr uint32_t LEN = 16384;
struct SyncTArgs { uint32_t l, n_exp, iters, tok; };
uint32_t sig(uint32_t l, uint32_t e, uint32_t j) { return 0x5EED0000u ^ ((l * NE + e) * 3u + j); }
uint32_t word(uint32_t s, uint32_t w) { return w == 0 ? s : mixh(s + w); }

void ref_ids(uint32_t s, uint32_t l, int* ids) {
  for (int k = 0; k < 6; k++) {
    const uint32_t h = mixh(s + 0x9e3779b9u * (uint32_t)(k + 1) + l * 7919u);
    const uint32_t u = h & 0xffffu;
    const uint32_t u3 = (((u * u) >> 16) * u) >> 16;
    uint32_t id = (uint32_t)(((uint64_t)u3 * NE) >> 16);
    for (bool dup = true; dup;) { dup = false; for (int q = 0; q < k; q++) if ((uint32_t)ids[q] == id) { dup = true; id = (id + 1) % NE; } }
    ids[k] = (int)id;
  }
}
uint32_t ref_token(uint32_t s, uint32_t tok, std::vector<int>* all_ids) {
  for (uint32_t l = 0; l < NL; l++) {
    int ids[6]; ref_ids(s, l, ids);
    if (all_ids) all_ids->insert(all_ids->end(), ids, ids + 6);
    const uint32_t sh = mixh(s ^ (l * 0x1234567u) ^ tok);
    uint32_t acc = 0;
    for (int k = 0; k < 6; k++) for (int j = 0; j < 3; j++)
      acc ^= word(sig(l, ids[k], j), 1 + (s + (uint32_t)(k * 3 + j)) % 15u) + (uint32_t)(k * 3 + j);
    s = mixh(s ^ acc ^ sh);
  }
  return s;
}

struct Loop {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  id<MTLComputePipelineState> p_router, p_gap, p_moe;
  id<MTLSharedEvent> rdy, go;
  id<MTLFence> fa, fg;
  Buffer st, sh, rids, tbl, bad, junk;
  ExpertCache* cache;
  uint64_t ev = 0;
  uint32_t it_att = 0, it_gap = 0, it_moe = 0;
  std::mt19937 rng{1234};
  double miss_delay_max_us = 600;
  uint64_t misses = 0, demand = 0;
};

void dispatch(Loop& c, id<MTLComputeCommandEncoder> ce, id<MTLComputePipelineState> p, const SyncTArgs& a, bool moe) {
  [ce setComputePipelineState:p];
  [ce setBuffer:(__bridge id<MTLBuffer>)c.st.buf offset:0 atIndex:0];
  if (p == c.p_router) [ce setBuffer:(__bridge id<MTLBuffer>)c.rids.buf offset:0 atIndex:1];
  else [ce setBuffer:(__bridge id<MTLBuffer>)c.sh.buf offset:0 atIndex:1];
  [ce setBytes:&a length:sizeof a atIndex:2];
  [ce setBuffer:(__bridge id<MTLBuffer>)c.junk.buf offset:0 atIndex:3];
  if (moe) {
    [ce setBuffer:(__bridge id<MTLBuffer>)c.rids.buf offset:0 atIndex:4];
    [ce setBuffer:(__bridge id<MTLBuffer>)c.tbl.buf offset:(size_t)a.l * 18 * 8 atIndex:5];
    [ce setBuffer:(__bridge id<MTLBuffer>)c.bad.buf offset:0 atIndex:6];
  }
  [ce dispatchThreadgroups:MTLSizeMake(40, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// one token; mode 0 = event (current), 1 = overlap (gap work between rdy and go, fence-ordered, hits answered first)
double run_token(Loop& c, int mode, uint32_t tok, uint32_t s0, uint32_t* s_out, std::vector<int>& seen_ids) {
  *(uint32_t*)c.st.cpu() = s0;
  const uint64_t b0 = c.ev; c.ev += NL;
  const std::vector<Buffer> arena = c.cache->buffers();
  std::vector<id<MTLResource>> res;
  for (auto& b : arena) res.push_back((__bridge id<MTLBuffer>)b.buf);
  id<MTLCommandBuffer> cb = [c.q commandBufferWithUnretainedReferences];
  id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  auto reopen = [&] { [ce endEncoding]; ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; };
  for (uint32_t l = 0; l < NL; l++) {
    dispatch(c, ce, c.p_router, SyncTArgs{l, NE, c.it_att, tok}, false);
    if (mode == 0) {
      [ce endEncoding];
      [cb encodeSignalEvent:c.rdy value:b0 + 1 + l];
      [cb encodeWaitForEvent:c.go value:b0 + 1 + l];
      ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      dispatch(c, ce, c.p_gap, SyncTArgs{l, NE, c.it_gap, tok}, false);
    } else {
      [ce updateFence:c.fa];
      [ce endEncoding];
      [cb encodeSignalEvent:c.rdy value:b0 + 1 + l];
      ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      [ce waitForFence:c.fa];
      dispatch(c, ce, c.p_gap, SyncTArgs{l, NE, c.it_gap, tok}, false);
      [ce updateFence:c.fg];
      [ce endEncoding];
      [cb encodeWaitForEvent:c.go value:b0 + 1 + l];
      ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      [ce waitForFence:c.fg];
    }
    [ce useResources:res.data() count:res.size() usage:MTLResourceUsageRead];
    dispatch(c, ce, c.p_moe, SyncTArgs{l, NE, c.it_moe, tok}, true);
  }
  [ce endEncoding];
  const double t0 = now_us();
  [cb commit];
  std::vector<ExpertLease> leases;
  std::uniform_real_distribution<double> ud(0, 1);
  for (uint32_t l = 0; l < NL; l++) {
    const double tw = now_us();
    while (c.rdy.signaledValue < b0 + 1 + l)
      if (now_us() - tw > 5e6) { fprintf(stderr, "rdy lost at layer %u\n", l); abort(); }
    const int* ids = (const int*)c.rids.cpu();
    seen_ids.insert(seen_ids.end(), ids, ids + 6);
    ExpertLease lease = c.cache->ensure((int)l, ids, 6);
    uint64_t* t = (uint64_t*)c.tbl.cpu() + (size_t)l * 18;
    for (int k = 0; k < 6; k++) for (int j = 0; j < 3; j++) t[k * 3 + j] = lease.views[k].gpu[j];
    auto prefetch = [&] {   // eviction pressure: 2 random experts of the next layer
      if (l + 1 >= NL) return;
      int pf[2]; for (int& x : pf) x = (int)(c.rng() % NE);
      c.cache->prefetch((int)l + 1, pf, 2);
    };
    c.demand += 6;
    if (mode == 1 && !lease.ticket) {
      c.go.signaledValue = b0 + 1 + l;
      prefetch();
    } else {
      prefetch();
      if (lease.ticket) {
        c.misses++;
        busy_us(ud(c.rng) * c.miss_delay_max_us);   // simulated SSD latency on top of the (page-cached) reads
        lease.wait();
      }
      c.go.signaledValue = b0 + 1 + l;
    }
    leases.push_back(std::move(lease));
  }
  [cb waitUntilCompleted];
  const double t1 = now_us();
  CHECK(cb.status != MTLCommandBufferStatusError, "GPU error: %s", cb.error.localizedDescription.UTF8String);
  leases.clear();   // the GPU is done with every slot of this token
  *s_out = *(uint32_t*)c.st.cpu();
  return t1 - t0;
}
}  // namespace

static void test_loop(int slots_div, bool all_hit) {
  // synthetic expert file: NL x NE experts x 3 matrices of LEN bytes, word 0 = signature, words 1..15 = mixh(sig + w)
  const char* tmp = getenv("TMPDIR");
  const std::string path = std::string(tmp ? tmp : "/tmp") + "/ms_test_dec_sync.bin";
  ExpertLayout lay; lay.n_layers = NL; lay.n_experts = NE;
  const uint64_t mat_bytes = (uint64_t)LEN * NE;
  const uint64_t file_size = 16384 + (uint64_t)NL * 3 * mat_bytes;
  {
    std::vector<uint32_t> buf(file_size / 4, 0);
    for (int l = 0; l < NL; l++) {
      ExpertLayerLayout ll;
      for (int j = 0; j < 3; j++) {
        ExpertMat& m = ll.mat[j];
        m.base = 16384 + ((uint64_t)l * 3 + j) * mat_bytes; m.stride = LEN; m.len = LEN; m.type = GType::IQ1_S;
        for (int e = 0; e < NE; e++) {
          uint32_t* p = buf.data() + (m.base + (uint64_t)e * LEN) / 4;
          for (uint32_t w = 0; w < 16; w++) p[w] = word(sig(l, e, j), w);
        }
      }
      lay.layers.push_back(ll);
    }
    FILE* f = fopen(path.c_str(), "wb"); fwrite(buf.data(), 1, file_size, f); fclose(f);
  }
  const int fd = open(path.c_str(), O_RDONLY);
  fcntl(fd, F_NOCACHE, 1);
  for (auto& ll : lay.layers) for (auto& m : ll.mat) { m.fd = fd; m.file_size = file_size; }
  ExpertCacheOptions o;
  o.io_threads = 4; o.read_chunk = 16384; o.prefault = true;
  {
    ExpertCacheOptions p0 = o; p0.arena_bytes = 64 << 20;
    ExpertCache probe(lay, p0);
    // misses: a third of all experts (>= 240 leases per token); all_hit: everything, pre-loaded
    o.arena_bytes = probe.classes()[0].slot_bytes * (NL * NE / slots_div);
  }
  ExpertCache cache(lay, o);
  if (all_hit) {
    std::vector<int> all(NE); for (int e = 0; e < NE; e++) all[e] = e;
    for (int l = 0; l < NL; l++) { ExpertLease le = cache.ensure(l, all); le.wait(); }
    cache.reset_stats();
  }
  printf("  loop: %d layers x %d experts, arena %d slots (%.0f%% of experts), %s\n", NL, NE, cache.n_slots(),
         100.0 * cache.n_slots() / (NL * NE), cache.describe().c_str());

  auto& d = Device::get();
  Loop c;
  c.dev = (__bridge id<MTLDevice>)d.device(); c.q = (__bridge id<MTLCommandQueue>)d.queue();
  c.p_router = (__bridge id<MTLComputePipelineState>)d.pipeline("sync_t_router");
  c.p_gap = (__bridge id<MTLComputePipelineState>)d.pipeline("sync_t_gap");
  c.p_moe = (__bridge id<MTLComputePipelineState>)d.pipeline("sync_t_moe");
  c.rdy = [c.dev newSharedEvent]; c.go = [c.dev newSharedEvent];
  c.fa = [c.dev newFence]; c.fg = [c.dev newFence];
  c.st = d.alloc(256); c.sh = d.alloc(256); c.rids = d.alloc(256); c.tbl = d.alloc(NL * 18 * 8); c.bad = d.alloc(256);
  c.junk = d.alloc(4096);
  c.cache = &cache;
  // work calibration (stand-ins: attention side ~150 us, gap ~70 us, routed MoE ~120 us of GPU time per layer)
  auto calib = [&](id<MTLComputePipelineState> p, double us) {
    uint32_t it = 2048; double best = 1e9;
    for (int r = 0; r < 3; r++) {
      id<MTLCommandBuffer> cb = [c.q commandBuffer];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      SyncTArgs a{0, NE, it, 0};
      [ce setComputePipelineState:c.p_gap];
      [ce setBuffer:(__bridge id<MTLBuffer>)c.st.buf offset:0 atIndex:0];
      [ce setBuffer:(__bridge id<MTLBuffer>)c.junk.buf offset:0 atIndex:1];
      [ce setBytes:&a length:sizeof a atIndex:2];
      [ce setBuffer:(__bridge id<MTLBuffer>)c.junk.buf offset:0 atIndex:3];
      for (int i = 0; i < 10; i++) {
        [ce dispatchThreadgroups:MTLSizeMake(40, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
      best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / 10);
    }
    (void)p;
    return (uint32_t)std::max(1.0, it * us / best);
  };
  c.it_att = calib(c.p_router, 150); c.it_gap = calib(c.p_gap, 70); c.it_moe = calib(c.p_moe, 120);
  const int tokens = (int)env_int("DEC_SYNC_TOKENS", 24);
  std::vector<double> tm[2];
  uint64_t bad_sig = 0, bad_ord = 0, chain_bad = 0, ids_bad = 0;
  uint32_t s = 0xC0FFEEu;
  for (int t = 0; t < 2 * tokens; t++) {
    const int mode = t & 1;
    memset(c.bad.cpu(), 0, 16);
    std::vector<int> seen, want;
    uint32_t got;
    const double us = run_token(c, mode, (uint32_t)t, s, &got, seen);
    const uint32_t ref = ref_token(s, (uint32_t)t, &want);
    if (t >= 2) tm[mode].push_back(us);   // skip warm-up
    bad_sig += ((uint32_t*)c.bad.cpu())[0]; bad_ord += ((uint32_t*)c.bad.cpu())[1];
    chain_bad += got != ref; ids_bad += seen != want;
    s = got ^ (uint32_t)t;
  }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  auto mn = [](std::vector<double> v) { return *std::min_element(v.begin(), v.end()); };
  const LayerStats ts = cache.total_stats();
  printf("  loop[%s]: %d tokens/mode, demand hit rate %.2f, layers with a miss %.0f%%, evictions %llu, prefetch issued %llu\n",
         all_hit ? "all-hit" : "misses", tokens, ts.hit_rate(), 100.0 * c.misses / (c.demand / 6), (unsigned long long)ts.evictions,
         (unsigned long long)ts.prefetch_issued);
  printf("  loop: ms/token event min %.2f med %.2f | overlap min %.2f med %.2f | saved per handshake: min %.1f us, med %.1f us\n",
         mn(tm[0]) / 1e3, med(tm[0]) / 1e3, mn(tm[1]) / 1e3, med(tm[1]) / 1e3, (mn(tm[0]) - mn(tm[1])) / NL,
         (med(tm[0]) - med(tm[1])) / NL);
  printf("  loop: stale/evicted expert reads %llu, gap-ordering violations %llu, chain mismatches %llu, CPU id mismatches %llu\n",
         (unsigned long long)bad_sig, (unsigned long long)bad_ord, (unsigned long long)chain_bad, (unsigned long long)ids_bad);
  CHECK(bad_sig == 0, "GPU read a wrong/stale expert slot %llu times", (unsigned long long)bad_sig);
  CHECK(bad_ord == 0, "gap work not ordered %llu times", (unsigned long long)bad_ord);
  CHECK(chain_bad == 0 && ids_bad == 0, "chain/id mismatch");
  if (!all_hit) CHECK(ts.evictions > 0 && c.misses > 0, "test did not exercise misses/evictions");
  else CHECK(c.misses == 0, "all-hit config missed");
  close(fd); unlink(path.c_str());
}

int main() {
  @autoreleasepool {
    printf("A) split table MoE equivalence\n");
    test_split();
    printf("B) emulated decode loop\n");
    test_loop(3, false);
    test_loop(1, true);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
