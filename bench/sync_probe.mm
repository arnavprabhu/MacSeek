// GPU->CPU->GPU handshake latency probe for the decode router handoff (docs/modules/dec_sync.md has the results).
// One "token" = a chain of L=40 small dependent dispatches (sync_layer) in ONE command buffer; between layers the GPU
// publishes "ready", the CPU answers with the next layer's table word, the GPU continues. Every layer checks it read
// the value the CPU wrote for the previous layer's result (stale-read detector), and the CPU re-checks the whole chain.
// Per-handshake cost = (token time - no-handshake token time) / (L - 1). Modes run round-robin, best-of-N + median.
//
//   bench_sync_probe [tokens=30] [iters=64] [cpu_us=0] [gap_us=100]   main table
//   bench_sync_probe 1 64 0 0 vis        CPU store -> running kernel visibility matrix
//   bench_sync_probe 1 64 0 0 pingpong   CPU <-> running kernel ping-pong (latency, payload ordering)
//   bench_sync_probe 30 64 0 0 breakdown GPU timestamps: where the event round trip goes
//
// Modes:
//   nohs            no handshake, one encoder, barriers only (baseline; table precomputed)
//   nohs_serial     same, MTLDispatchTypeSerial encoder
//   split           encoder end/begin between layers, nothing else (encoder-split cost)
//   split_sig       + encodeSignalEvent each layer (no wait)
//   split_presat    + encodeWaitForEvent on an already-signaled value
//   event           CURRENT: signal rdy, wait go; CPU spins rdy.signaledValue, sets go.signaledValue
//   event_listen    same, CPU side in MTLSharedEventListener blocks (dispatch queue)
//   event_cbsig     CPU answers by committing a pre-encoded command buffer (2nd queue) that GPU-signals go
//   memflag_event   GPU publishes ready by an atomic store (no rdy event), CPU polls memory, GPU waits go event
//   gpupoll         ready by atomic store, CPU polls memory, answers with a memory go counter; a 1-simdgroup kernel
//                   spins on go (bounded, volatile coherent loads) and copies the table; _fb: + encodeWaitForEvent
//                   fallback (CPU signals the go event before the flag)
//   nohs_gap/event_gap  ~gap_us of GPU work (all cores) between the ready signal and the go wait: how much of the
//                   round trip can be hidden behind independent work (e.g. the shared expert)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dispatch/dispatch.h>
#include <libkern/OSCacheControl.h>
#include <mach/mach_time.h>
#include "core/metal.hpp"
using namespace ms;

namespace {
constexpr int L = 40;
int TS = 64;   // table stride in uint32 (256 B per layer; SP_TS=1 packs layers into shared cache lines)
uint32_t mixh(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
uint32_t cpu_tbl(uint32_t prev, int l) { return mixh(prev + 0x9e3779b9U * (uint32_t)(l + 1)); }
double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void busy_us(double us) { if (us <= 0) return; double t = now_us(); while (now_us() - t < us) {} }

struct SyncLayerArgs { uint32_t l, iters, ts, mode, check; };
struct SyncWaitArgs { uint32_t want, max_spin, n_words, stat_idx; };

enum Kind { NOHS, SPLIT, SPLIT_SIG, SPLIT_PRESAT, EVENT, EVENT_LISTEN, EVENT_CBSIG, MEMFLAG_EVENT, GPUPOLL };
struct Mode {
  const char* name; Kind kind;
  bool serial = false;
  bool fallback = false;  // gpupoll: encodeWaitForEvent after the spin
  bool gap = false;       // gap work between ready and go
};

struct Ctx {
  id<MTLDevice> dev; id<MTLCommandQueue> q, q2;
  id<MTLComputePipelineState> p_layer, p_wait, p_busy;
  Buffer out, tbl, tblg, flags, stat, bad, junk;
  id<MTLSharedEvent> rdy, go;
  id<MTLFence> f1, f2;
  MTLSharedEventListener* listener;
  uint64_t ev = 0;           // event value base (monotone across tokens)
  uint32_t iters = 64; double cpu_us = 0;
  uint32_t max_spin = 1u << 22;
  uint32_t gap_iters = 0;
  // breakdown
  id<MTLCounterSampleBuffer> sb = nil;
  std::vector<uint64_t> cpu_rdy, cpu_go;   // mach_absolute_time per layer
};

std::atomic<uint32_t>* rdy_flag(Ctx& c) { return (std::atomic<uint32_t>*)((char*)c.flags.cpu() + 0); }
std::atomic<uint32_t>* go_flag(Ctx& c) { return (std::atomic<uint32_t>*)((char*)c.flags.cpu() + 256); }

void enc_busy(Ctx& c, id<MTLComputeCommandEncoder> ce) {
  [ce setComputePipelineState:c.p_busy];
  [ce setBuffer:(__bridge id<MTLBuffer>)c.junk.buf offset:0 atIndex:0];
  [ce setBytes:&c.gap_iters length:4 atIndex:1];
  [ce dispatchThreadgroups:MTLSizeMake(160, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// Returns wall time of the token (commit -> completed) in us; *bad = stale-read count, *to = GPU spin timeouts.
double run_token(Ctx& c, const Mode& m, uint32_t* bad_out, uint32_t* timeouts_out, bool ts = false) {
  uint32_t* out = (uint32_t*)c.out.cpu();
  uint32_t* tbl = (uint32_t*)c.tbl.cpu();
  memset(out, 0, L * 4);
  memset(c.tbl.cpu(), 0, L * TS * 4);
  memset(c.stat.cpu(), 0, c.stat.size);
  *(uint32_t*)c.bad.cpu() = 0;
  const bool hs = m.kind == EVENT || m.kind == EVENT_LISTEN || m.kind == EVENT_CBSIG || m.kind == MEMFLAG_EVENT || m.kind == GPUPOLL;
  if (!hs) for (int l = 0; l < L; l++) tbl[l * TS] = 777u + l;   // no CPU answers: any value
  tbl[0] = cpu_tbl(12345u, 0);
  const uint64_t e0 = c.ev; c.ev += L + 1;
  if (m.kind == SPLIT_PRESAT) c.go.signaledValue = e0 + L + 1;
  id<MTLBuffer> bout = (__bridge id<MTLBuffer>)c.out.buf, btbl = (__bridge id<MTLBuffer>)c.tbl.buf,
                btg = (__bridge id<MTLBuffer>)c.tblg.buf, bfl = (__bridge id<MTLBuffer>)c.flags.buf,
                bst = (__bridge id<MTLBuffer>)c.stat.buf, bbad = (__bridge id<MTLBuffer>)c.bad.buf;
  id<MTLCommandBuffer> cb = [c.q commandBufferWithUnretainedReferences];
  const MTLDispatchType dt = m.serial ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent;
  auto open_enc = [&](int l) -> id<MTLComputeCommandEncoder> {
    if (!ts) return [cb computeCommandEncoderWithDispatchType:dt];
    MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.dispatchType = dt;
    pd.sampleBufferAttachments[0].sampleBuffer = c.sb;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * l;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * l + 1;
    return [cb computeCommandEncoderWithDescriptor:pd];
  };
  id<MTLComputeCommandEncoder> ce = open_enc(0);
  const bool gp = m.kind == GPUPOLL;
  const bool publish_mem = gp || m.kind == MEMFLAG_EVENT;
  for (int l = 0; l < L; l++) {
    SyncLayerArgs a{(uint32_t)l, c.iters, (uint32_t)TS, 0, hs ? 1u : 0u};
    id<MTLBuffer> tb = (gp && l > 0) ? btg : btbl;   // gpupoll: the spin kernel's copy (layer 0: written pre-commit)
    if (publish_mem && l + 1 < L) a.mode |= 2;
    [ce setComputePipelineState:c.p_layer];
    [ce setBuffer:bout offset:0 atIndex:0];
    [ce setBuffer:tb offset:0 atIndex:1];
    [ce setBytes:&a length:sizeof a atIndex:2];
    [ce setBuffer:bbad offset:0 atIndex:3];
    [ce setBuffer:bfl offset:0 atIndex:4];
    [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
    if (l + 1 == L) break;
    const uint64_t v = e0 + 1 + l;
    auto reopen = [&] { [ce endEncoding]; };
    switch (m.kind) {
      case NOHS: if (m.gap) enc_busy(c, ce); if (ts) { reopen(); ce = open_enc(l + 1); } break;
      case SPLIT: reopen(); ce = open_enc(l + 1); break;
      case SPLIT_SIG: reopen(); [cb encodeSignalEvent:c.rdy value:v]; ce = open_enc(l + 1); break;
      case SPLIT_PRESAT: reopen(); [cb encodeWaitForEvent:c.go value:v]; ce = open_enc(l + 1); break;
      case EVENT: case EVENT_LISTEN: case EVENT_CBSIG:
        // gap: independent work after the ready signal, ordered after layer l and before layer l+1 by MTLFences
        // (consecutive compute encoders over untracked buffers are NOT ordered by themselves)
        if (m.gap) [ce updateFence:c.f1];
        reopen(); [cb encodeSignalEvent:c.rdy value:v];
        if (m.gap) {
          ce = [cb computeCommandEncoderWithDispatchType:dt];
          [ce waitForFence:c.f1]; enc_busy(c, ce); [ce updateFence:c.f2]; [ce endEncoding];
        }
        [cb encodeWaitForEvent:c.go value:v];
        ce = open_enc(l + 1);
        if (m.gap) [ce waitForFence:c.f2];
        break;
      case MEMFLAG_EVENT: reopen(); [cb encodeWaitForEvent:c.go value:v]; ce = open_enc(l + 1); break;
      case GPUPOLL: {
        SyncWaitArgs w{(uint32_t)(l + 1), c.max_spin, 1u, (uint32_t)l};
        [ce setComputePipelineState:c.p_wait];
        [ce setBuffer:bfl offset:256 atIndex:0];
        [ce setBytes:&w length:sizeof w atIndex:1];
        [ce setBuffer:btbl offset:(size_t)(l + 1) * TS * 4 atIndex:2];
        [ce setBuffer:btg offset:(size_t)(l + 1) * TS * 4 atIndex:3];
        [ce setBuffer:bst offset:0 atIndex:4];
        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (m.fallback || ts) { reopen(); if (m.fallback) [cb encodeWaitForEvent:c.go value:v]; ce = open_enc(l + 1); }
        break;
      }
    }
  }
  [ce endEncoding];
  rdy_flag(c)->store(0); go_flag(c)->store(0);
  // event_cbsig: pre-encoded signal command buffers on the second queue
  std::vector<id<MTLCommandBuffer>> sigs;
  if (m.kind == EVENT_CBSIG)
    for (int l = 0; l + 1 < L; l++) {
      id<MTLCommandBuffer> s = [c.q2 commandBufferWithUnretainedReferences];
      [s encodeSignalEvent:c.go value:e0 + 1 + l];
      [s enqueue];
      sigs.push_back(s);
    }
  dispatch_semaphore_t done = nullptr;
  if (m.kind == EVENT_LISTEN) {
    done = dispatch_semaphore_create(0);
    for (int l = 0; l + 1 < L; l++) {
      Ctx* cp = &c;
      [c.rdy notifyListener:c.listener atValue:e0 + 1 + l block:^(id<MTLSharedEvent> ev, uint64_t val) {
        busy_us(cp->cpu_us);
        uint32_t* o = (uint32_t*)cp->out.cpu();
        ((uint32_t*)cp->tbl.cpu())[(l + 1) * TS] = cpu_tbl(o[l], l + 1);
        cp->go.signaledValue = val;
        if (l + 2 == L) dispatch_semaphore_signal(done);
      }];
    }
  }
  if (ts) { c.cpu_rdy.assign(L, 0); c.cpu_go.assign(L, 0); }
  const double t0 = now_us();
  [cb commit];
  if (hs && m.kind != EVENT_LISTEN) {
    for (int l = 0; l + 1 < L; l++) {
      const uint64_t v = e0 + 1 + l;
      if (m.kind == EVENT || m.kind == EVENT_CBSIG) { while (c.rdy.signaledValue < v) {} }
      else {
        const double tw = now_us();   // bounded: a lost flag aborts the probe
        while (rdy_flag(c)->load(std::memory_order_acquire) < (uint32_t)(l + 1))
          if (now_us() - tw > 2e6) { fprintf(stderr, "%s: ready flag lost at layer %d\n", m.name, l); abort(); }
      }
      if (ts) c.cpu_rdy[l] = mach_absolute_time();
      busy_us(c.cpu_us);
      tbl[(l + 1) * TS] = cpu_tbl(out[l], l + 1);
      if (ts) c.cpu_go[l] = mach_absolute_time();
      if (m.kind == GPUPOLL) {
        if (m.fallback) c.go.signaledValue = v;   // before the flag: the GPU's event wait is then already satisfied
        go_flag(c)->store((uint32_t)(l + 1), std::memory_order_release);
      } else if (m.kind == EVENT_CBSIG) {
        [sigs[l] commit];
      } else {
        c.go.signaledValue = v;
      }
    }
  }
  [cb waitUntilCompleted];
  const double t1 = now_us();
  if (done) dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  for (auto& s : sigs) [s waitUntilCompleted];
  if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "%s: GPU error %s\n", m.name, cb.error.localizedDescription.UTF8String); abort(); }
  if (c.go.signaledValue < e0 + L) c.go.signaledValue = e0 + L;
  if (c.rdy.signaledValue < e0 + L) c.rdy.signaledValue = e0 + L;
  *bad_out = *(uint32_t*)c.bad.cpu();
  *timeouts_out = ((uint32_t*)c.stat.cpu())[0];
  if (hs) {   // full chain check on the CPU
    uint32_t prev = 12345u, mism = 0;
    for (int l = 0; l < L; l++) {
      const uint32_t t = cpu_tbl(prev, l);
      uint32_t x = 0;
      for (uint32_t i = 0; i < 32; i++) { uint32_t v = prev ^ t ^ i; for (uint32_t k = 0; k < c.iters; k++) v = mixh(v + k); x ^= v; }
      if (out[l] != x) mism++;
      prev = out[l];
    }
    *bad_out += mism;
  }
  return t1 - t0;
}

// CPU store -> running kernel visibility matrix: kernel load kind x buffer CPU cache mode x CPU cache flush.
void vis_matrix(Ctx& c) {
  id<MTLComputePipelineState> pv = (__bridge id<MTLComputePipelineState>)Device::get().pipeline("sync_vis");
  const char* kn[] = {"atomic_load", "atomic_add0", "volatile", "coherent", "cas", "fence+load"};
  for (int wc = 0; wc < 2; wc++) for (int fl = 0; fl < 2; fl++) for (uint32_t km = 0; km < 6; km++) {
    MTLResourceOptions o = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked |
                           (wc ? MTLResourceCPUCacheModeWriteCombined : MTLResourceCPUCacheModeDefaultCache);
    id<MTLBuffer> fb = [c.dev newBufferWithLength:16384 options:o];
    id<MTLBuffer> ob = [c.dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
    memset(fb.contents, 0, 16384);
    uint32_t a[4] = {1, 2000000, km, 0};
    id<MTLCommandBuffer> cb = [c.q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:pv];
    [ce setBuffer:fb offset:0 atIndex:0]; [ce setBytes:a length:16 atIndex:1]; [ce setBuffer:ob offset:0 atIndex:2];
    [ce setBuffer:fb offset:0 atIndex:3];
    [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [ce endEncoding];
    [cb commit];
    busy_us(2000);
    const double ts = now_us();
    ((std::atomic<uint32_t>*)fb.contents)->store(1, std::memory_order_seq_cst);
    if (fl) sys_dcache_flush(fb.contents, 64);
    [cb waitUntilCompleted];
    const uint32_t* r = (const uint32_t*)ob.contents;
    printf("vis kernel=%-12s cpu_cache=%-13s flush=%d : %s spins=%u  store->done %.0f us\n", kn[km],
           wc ? "writecombined" : "default", fl, r[1] ? "SEEN " : "never", r[0], now_us() - ts);
  }
}

// CPU <-> running kernel ping-pong: round-trip latency and payload ordering in both directions.
void pingpong(Ctx& c) {
  id<MTLComputePipelineState> pp = (__bridge id<MTLComputePipelineState>)Device::get().pipeline("sync_pingpong");
  const char* kn[] = {"plain", "coherent", "atomic"};
  const int flush = getenv("PP_FLUSH") ? atoi(getenv("PP_FLUSH")) : 0;
  for (uint32_t kind = 0; kind < 3; kind++) for (int rep = 0; rep < 2; rep++) {
    Buffer fb = Device::get().alloc(65536);
    char* base = (char*)fb.cpu();
    auto* ping = (std::atomic<uint32_t>*)(base + 0);
    auto* pong = (std::atomic<uint32_t>*)(base + 4096);
    uint32_t* cp = (uint32_t*)(base + 8192);
    volatile uint32_t* gp = (volatile uint32_t*)(base + 12288);
    uint32_t* res = (uint32_t*)(base + 16384);
    const uint32_t n = 2000;
    uint32_t a[4] = {n, 20000000, 0, kind};
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)fb.buf;
    id<MTLCommandBuffer> cb = [c.q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:pp];
    [ce setBuffer:b offset:0 atIndex:0]; [ce setBuffer:b offset:4096 atIndex:1]; [ce setBuffer:b offset:8192 atIndex:2];
    [ce setBuffer:b offset:12288 atIndex:3]; [ce setBytes:a length:16 atIndex:4]; [ce setBuffer:b offset:16384 atIndex:5];
    [ce setBuffer:b offset:12288 atIndex:6]; [ce setBuffer:b offset:12288 atIndex:7];
    [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [ce endEncoding];
    [cb commit];
    std::vector<double> rt;
    uint32_t gbad = 0, lost = 0;
    busy_us(500);
    for (uint32_t k = 1; k <= n; k++) {
      cp[k % 16] = k * 2654435761u;
      if (flush) sys_dcache_flush(cp, 64);
      const double t = now_us();
      ping->store(k, std::memory_order_release);
      if (flush) sys_dcache_flush((void*)ping, 64);
      bool ok = true;
      while (pong->load(std::memory_order_acquire) < k) {
        if (flush) sys_dcache_flush((void*)pong, 64);
        if (now_us() - t > 200000) { ok = false; break; }
      }
      if (!ok) { lost++; break; }
      if (flush) sys_dcache_flush((void*)gp, 64);
      rt.push_back(now_us() - t);
      if (gp[k % 16] != k * 40503u) gbad++;
    }
    if (lost) ping->store(0xffffffffu);
    [cb waitUntilCompleted];
    std::sort(rt.begin(), rt.end());
    auto q = [&](double p) { return rt.empty() ? -1.0 : rt[(size_t)(p * (rt.size() - 1))]; };
    printf("pingpong gpu_store=%-8s rounds=%u  rt_us min %.2f p50 %.2f p99 %.2f max %.1f | cpu->gpu payload stale %u, gpu->cpu payload stale %u, lost %u, gpu timeouts %u\n",
           kn[kind], res[0], q(0), q(0.5), q(0.99), q(1), res[1], gbad, lost, res[2]);
    Device::get().release(fb);
  }
}

// GPU timestamps at encoder boundaries (one encoder per layer) correlated with CPU mach time: per handshake,
// X = GPU layer end -> CPU sees rdy, C = CPU work, Y = CPU sets go -> GPU starts next encoder.
void breakdown(Ctx& c, int tokens) {
  if (![c.dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { printf("no stage-boundary counters\n"); return; }
  id<MTLCounterSet> tsset = nil;
  for (id<MTLCounterSet> s in c.dev.counterSets) if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) tsset = s;
  if (!tsset) { printf("no timestamp counter set\n"); return; }
  MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
  d.counterSet = tsset; d.storageMode = MTLStorageModeShared; d.sampleCount = 2 * L;
  NSError* err = nil;
  c.sb = [c.dev newCounterSampleBufferWithDescriptor:d error:&err];
  if (!c.sb) { printf("sample buffer: %s\n", err.localizedDescription.UTF8String); return; }
  mach_timebase_info_data_t tb; mach_timebase_info(&tb);
  auto mt_ns = [&](uint64_t t) { return (double)t * tb.numer / tb.denom; };   // mach ticks -> ns (sampleTimestamps
                                                                              // and counter samples are ns here)
  Mode modes[] = {{"split(no events)", NOHS}, {"event", EVENT}, {"event_cbsig", EVENT_CBSIG}};
  for (auto& m : modes) {
    std::vector<double> X, Y, C, G;
    for (int t = 0; t < tokens; t++) {
      MTLTimestamp c0, g0, c1, g1;
      [c.dev sampleTimestamps:&c0 gpuTimestamp:&g0];
      uint32_t bad, to;
      run_token(c, m, &bad, &to, true);
      [c.dev sampleTimestamps:&c1 gpuTimestamp:&g1];
      NSData* r = [c.sb resolveCounterRange:NSMakeRange(0, 2 * L)];
      const MTLCounterResultTimestamp* s = (const MTLCounterResultTimestamp*)r.bytes;
      const double k = (double)(c1 - c0) / (double)(g1 - g0);
      if (t == 0 && getenv("BD_DEBUG")) printf("c0 %llu c1 %llu g0 %llu g1 %llu s0 %llu s1 %llu s79 %llu\n", c0, c1, g0, g1, s[0].timestamp, s[1].timestamp, s[2 * L - 1].timestamp);
      if (t == 0 && getenv("BD_DEBUG")) for (int l = 0; l < 4; l++) printf("  l%d start %llu end %llu cpu_rdy %llu cpu_go %llu\n", l, s[2*l].timestamp, s[2*l+1].timestamp, c.cpu_rdy[l], c.cpu_go[l]);
      auto g2c = [&](uint64_t g) { return (double)c0 + ((double)g - (double)g0) * k; };   // GPU ts -> CPU ns
      for (int l = 0; l + 1 < L; l++) {
        const double end_l = g2c(s[2 * l + 1].timestamp), start_n = g2c(s[2 * (l + 1)].timestamp);
        G.push_back((start_n - end_l) / 1e3);
        if (m.kind != NOHS) {
          X.push_back((mt_ns(c.cpu_rdy[l]) - end_l) / 1e3);
          C.push_back((mt_ns(c.cpu_go[l]) - mt_ns(c.cpu_rdy[l])) / 1e3);
          Y.push_back((start_n - mt_ns(c.cpu_go[l])) / 1e3);
        }
      }
    }
    auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto lo = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 10]; };
    printf("breakdown %-17s GPU gap end(l)->start(l+1): p10 %6.1f med %6.1f us | X gpu->cpu p10 %6.1f med %6.1f | "
           "cpu %4.1f | Y cpu->gpu p10 %6.1f med %6.1f\n", m.name, lo(G), med(G), lo(X), med(X), med(C), lo(Y), med(Y));
  }
}
}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    const int tokens = argc > 1 ? atoi(argv[1]) : 30;
    if (getenv("SP_TS")) TS = std::max(1, atoi(getenv("SP_TS")));
    Ctx c;
    c.iters = argc > 2 ? (uint32_t)atoi(argv[2]) : 64;
    c.cpu_us = argc > 3 ? atof(argv[3]) : 0;
    const double gap_us = argc > 4 ? atof(argv[4]) : 100;
    const char* sub = argc > 5 ? argv[5] : "";
    auto& d = Device::get();
    c.dev = (__bridge id<MTLDevice>)d.device();
    c.q = (__bridge id<MTLCommandQueue>)d.queue();
    c.q2 = [c.dev newCommandQueue];
    c.p_layer = (__bridge id<MTLComputePipelineState>)d.pipeline("sync_layer");
    c.p_wait = (__bridge id<MTLComputePipelineState>)d.pipeline("sync_wait_copy");
    c.p_busy = (__bridge id<MTLComputePipelineState>)d.pipeline("sync_busy");
    c.out = d.alloc(4096); c.tbl = d.alloc(L * TS * 4); c.tblg = d.alloc(L * TS * 4);
    c.flags = d.alloc(4096); c.stat = d.alloc(4096); c.bad = d.alloc(256); c.junk = d.alloc(4096);
    c.rdy = [c.dev newSharedEvent]; c.go = [c.dev newSharedEvent];
    c.f1 = [c.dev newFence]; c.f2 = [c.dev newFence];
    c.listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("sync_probe", DISPATCH_QUEUE_SERIAL)];
    if (!strcmp(sub, "vis")) { vis_matrix(c); return 0; }
    if (!strcmp(sub, "pingpong")) { pingpong(c); return 0; }
    if (!strcmp(sub, "breakdown")) { breakdown(c, tokens); return 0; }

    // GPU spin bound: ~5 ms of polling (calibrated on a spin that never succeeds)
    {
      SyncWaitArgs w{0xffffffffu, 1u << 18, 0, 0};
      double best = 1e9;
      for (int it = 0; it < 3; it++) {
        go_flag(c)->store(0);
        id<MTLCommandBuffer> cb = [c.q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:c.p_wait];
        [ce setBuffer:(__bridge id<MTLBuffer>)c.flags.buf offset:256 atIndex:0];
        [ce setBytes:&w length:sizeof w atIndex:1];
        [ce setBuffer:(__bridge id<MTLBuffer>)c.tbl.buf offset:0 atIndex:2];
        [ce setBuffer:(__bridge id<MTLBuffer>)c.tblg.buf offset:0 atIndex:3];
        [ce setBuffer:(__bridge id<MTLBuffer>)c.stat.buf offset:0 atIndex:4];
        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [ce endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e6);
      }
      c.max_spin = (uint32_t)std::min(4e9, 5000.0 / (best / (1 << 18)));
      printf("gpu spin: %.1f ns/poll -> bound %u polls (~5 ms)\n", best * 1e3 / (1 << 18), c.max_spin);
    }
    // gap work calibration: sync_busy (160 TGs x 256 threads) iterations for ~gap_us
    {
      c.gap_iters = 4096;
      double us = 1e9;
      for (int it = 0; it < 3; it++) {
        id<MTLCommandBuffer> cb = [c.q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        for (int i = 0; i < 10; i++) enc_busy(c, ce);
        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
        us = std::min(us, (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / 10);
      }
      c.gap_iters = (uint32_t)std::max(1.0, 4096 * gap_us / us);
      printf("gap work: %u iters ~ %.0f us\n", c.gap_iters, gap_us);
    }

    std::vector<Mode> modes = {
      {"nohs", NOHS}, {"nohs_serial", NOHS, true}, {"split", SPLIT}, {"split_sig", SPLIT_SIG},
      {"split_presat", SPLIT_PRESAT}, {"event", EVENT}, {"event_listen", EVENT_LISTEN}, {"event_cbsig", EVENT_CBSIG},
      {"memflag_event", MEMFLAG_EVENT}, {"gpupoll", GPUPOLL}, {"gpupoll_fb", GPUPOLL, false, true},
      {"nohs_gap", NOHS, false, false, true}, {"event_gap", EVENT, false, false, true},
    };
    if (getenv("SP_GAPONLY")) modes = {{"nohs", NOHS}, {"event", EVENT}, {"nohs_gap", NOHS, false, false, true}, {"event_gap", EVENT, false, false, true}};
    size_t gap_base = 0;
    for (size_t i = 0; i < modes.size(); i++) if (!strcmp(modes[i].name, "nohs_gap")) gap_base = i;
    struct Res { std::vector<double> t; uint64_t bad = 0, to = 0; };
    std::vector<Res> res(modes.size());
    for (size_t i = 0; i < modes.size(); i++) { uint32_t b, t; run_token(c, modes[i], &b, &t); }   // warm-up
    for (int r = 0; r < tokens; r++)
      for (size_t i = 0; i < modes.size(); i++) {
        uint32_t b = 0, t = 0;
        res[i].t.push_back(run_token(c, modes[i], &b, &t));
        res[i].bad += b; res[i].to += t;
      }
    auto pct = [](std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; };
    printf("\nL=%d layers (%d handshakes), iters=%u, cpu_us=%.1f, gap=%.0f us, %d tokens per mode (round-robin)\n",
           L, L - 1, c.iters, c.cpu_us, gap_us, tokens);
    printf("%-15s %10s %10s %11s %11s %7s %8s\n", "mode", "min_us", "med_us", "hs_min_us", "hs_med_us", "stale", "timeout");
    for (size_t i = 0; i < modes.size(); i++) {
      const size_t bi = modes[i].gap ? gap_base : 0;   // gap modes vs nohs_gap
      const double mn = pct(res[i].t, 0), md = pct(res[i].t, 0.5);
      printf("%-15s %10.1f %10.1f %11.2f %11.2f %7llu %8llu\n", modes[i].name, mn, md,
             (mn - pct(res[bi].t, 0)) / (L - 1), (md - pct(res[bi].t, 0.5)) / (L - 1),
             (unsigned long long)res[i].bad, (unsigned long long)res[i].to);
    }
  }
  return 0;
}
