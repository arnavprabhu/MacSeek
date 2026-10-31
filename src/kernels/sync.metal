// GPU<->CPU handshake probe/test kernels (bench/sync_probe.mm, tests/test_dec_sync.mm; docs/modules/dec_sync.md).
// Not used by the engine: the decode path keeps MTLSharedEvent handshakes and hides them behind independent GPU work
// (MACSEEK_DEC_SYNC=sgu|shared). Measured here: CPU stores reach a running kernel only via volatile coherent(device)
// loads, eviction-driven (tens of us to ms) and unordered across cache lines, in both directions => no GPU polling.
#include <metal_stdlib>
using namespace metal;

// Publish a monotone value to the CPU: flag = v (after all previous writes of this thread are visible).
kernel void sync_publish(device atomic_uint* flag [[buffer(0)]], constant uint& v [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_store_explicit(flag, v, memory_order_relaxed);
}

struct SyncWaitArgs {
  uint want;        // wait until *go >= want
  uint max_spin;    // bound (iterations of the poll loop); on timeout stat[0]++ and the table is NOT copied
  uint n_words;     // uint32 words to copy src -> dst after the wait (0 = none)
  uint stat_idx;    // stat[2 + stat_idx] = spins used (diagnostics)
};

// One simdgroup: lane 0 spins on go, then all lanes copy the CPU-written table words into dst.
// stat[0] = number of timeouts (atomic add), stat[2 + i] = spins for wait i.
// MEASURED (bench/sync_probe, M5 Pro): CPU stores reach a running kernel only through `volatile coherent(device)`
// loads (atomic loads / RMW / fences keep hitting a stale GPU cache line, sometimes for seconds), visibility is
// eviction-driven (round trip p50 40-300 us, p99 1-4 ms) and NOT ordered across cache lines (payload written before
// the flag is often stale). So this is a probe of a rejected design, not a production protocol.
kernel void sync_wait_copy(device coherent(device) uint* go [[buffer(0)]], constant SyncWaitArgs& a [[buffer(1)]],
                           device coherent(device) uint* src [[buffer(2)]], device uint* dst [[buffer(3)]],
                           device atomic_uint* stat [[buffer(4)]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  uint n = 0;
  bool ok = false;
  if (lane == 0) {
    for (; n < a.max_spin; n++)
      if (*(volatile device coherent(device) uint*)go >= a.want) { ok = true; break; }
    if (!ok) atomic_fetch_add_explicit(stat, 1u, memory_order_relaxed);
    atomic_store_explicit(stat + 2 + a.stat_idx, n, memory_order_relaxed);
  }
  ok = simd_broadcast((ushort)ok, (ushort)0) != 0;
  if (!ok) return;
  for (uint i = lane; i < a.n_words; i += 32) dst[i] = ((volatile device coherent(device) uint*)src)[i];
}

// probe-only: GPU busy work of ~iters rounds per thread in `groups` threadgroups (gap filler)
kernel void sync_busy(device uint* junk [[buffer(0)]], constant uint& iters [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
  uint v = tid;
  for (uint i = 0; i < iters; i++) v = v * 1664525u + 1013904223u + (v >> 13);
  if (v == 0x12345678u) junk[tid & 1023] = v;
}

// ---- probe-only kernel: one dependent "layer". out[l] = mix(out[l-1] ^ tbl[l*ts]) after `iters` rounds of work;
// checks that tbl[l*ts] equals the value the CPU must have written for out[l-1] (stale-read detector: bad[0]++).
// mode bit 0: read tbl with an atomic load (else a plain load). bit 1: publish flag = l+1 at the end (atomic store).
struct SyncLayerArgs { uint l, iters, ts, mode, check; };
static inline uint sync_mix(uint x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
kernel void sync_layer(device uint* out [[buffer(0)]], device atomic_uint* tbl [[buffer(1)]],
                       constant SyncLayerArgs& a [[buffer(2)]], device atomic_uint* bad [[buffer(3)]],
                       device atomic_uint* flag [[buffer(4)]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint prev = a.l ? out[a.l - 1] : 12345u;
  uint t;
  if (a.mode & 1) t = atomic_load_explicit(tbl + a.l * a.ts, memory_order_relaxed);
  else t = ((device uint*)tbl)[a.l * a.ts];
  if (a.check && lane == 0 && t != sync_mix(prev + 0x9e3779b9U * (a.l + 1))) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
  uint v = prev ^ t ^ lane;
  for (uint i = 0; i < a.iters; i++) v = sync_mix(v + i);
  v = simd_xor(v);
  if (lane == 0) {
    out[a.l] = v;
    if (a.mode & 2) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_store_explicit(flag, a.l + 1, memory_order_relaxed);
    }
  }
}

// ---- probe-only: CPU->running-kernel visibility variants. Spins (bounded) until *p >= want; out[0] = spins,
// out[1] = 1 if seen. mode: 0 relaxed atomic load, 1 atomic fetch_add(0), 2 volatile load, 3 coherent(device) load,
// 4 atomic exchange-free compare_exchange(0->0), 5 relaxed load + seq_cst fence per poll
kernel void sync_vis(device atomic_uint* p [[buffer(0)]], constant uint4& a [[buffer(1)]], device uint* out [[buffer(2)]],
                     device coherent(device) uint* pc [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint want = a.x, max_spin = a.y, mode = a.z;
  uint n = 0; bool ok = false;
  for (; n < max_spin; n++) {
    uint v;
    if (mode == 0) v = atomic_load_explicit(p, memory_order_relaxed);
    else if (mode == 1) v = atomic_fetch_add_explicit(p, 0u, memory_order_relaxed);
    else if (mode == 2) v = *(volatile device uint*)p;
    else if (mode == 3) v = *pc;
    else if (mode == 4) { uint e = 0; atomic_compare_exchange_weak_explicit(p, &e, 0u, memory_order_relaxed, memory_order_relaxed); v = e; }
    else { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); v = atomic_load_explicit(p, memory_order_relaxed); }
    if (v >= want) { ok = true; break; }
  }
  out[0] = n; out[1] = ok ? 1u : 0u;
}

// ---- probe-only: ping-pong between the CPU and one GPU thread inside ONE kernel.
// for k = 1..n: wait ping >= k (coherent loads, bounded); check CPU payload cp[k % 16] == k*2654435761 (coherent loads);
// write GPU payload gp[k % 16] = k*40503 (store kind a.w: 0 plain, 1 coherent, 2 atomic) ; fence; pong = k (atomic store)
// res[0] = rounds done, res[1] = CPU payload mismatches, res[2] = timeouts
kernel void sync_pingpong(device coherent(device) uint* ping [[buffer(0)]], device atomic_uint* pong [[buffer(1)]],
                          device coherent(device) uint* cp [[buffer(2)]], device uint* gp [[buffer(3)]],
                          constant uint4& a [[buffer(4)]], device uint* res [[buffer(5)]],
                          device coherent(device) uint* gpc [[buffer(6)]], device atomic_uint* gpa [[buffer(7)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint n = a.x, max_spin = a.y, kind = a.w;
  uint bad = 0, to = 0, k = 1;
  for (; k <= n; k++) {
    uint s = 0;
    while (*(volatile device coherent(device) uint*)ping < k && s < max_spin) s++;
    if (s >= max_spin) { to++; break; }
    if (((volatile device coherent(device) uint*)cp)[k % 16] != k * 2654435761u) bad++;
    const uint v = k * 40503u;
    if (kind == 0) gp[k % 16] = v;
    else if (kind == 1) gpc[k % 16] = v;
    else atomic_store_explicit(gpa + k % 16, v, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_store_explicit(pong, k, memory_order_relaxed);
  }
  res[0] = k - 1; res[1] = bad; res[2] = to;
}

// ---- tests/test_dec_sync.mm: emulated decode layer (router -> handshake -> gap work -> table MoE) ----
// All kernels: `groups` threadgroups x 256 threads burn `iters` rounds (stand-in for real work); thread 0 of
// threadgroup 0 does the protocol logic. Integer math only so the CPU reference matches exactly.
struct SyncTArgs { uint l, n_exp, iters, tok; };
static inline uint sync_t_burn(uint tid, uint iters) {
  uint v = tid;
  for (uint i = 0; i < iters; i++) v = v * 1664525u + 1013904223u + (v >> 13);
  return v;
}
static inline uint sync_t_sig(uint l, uint e, uint j, uint n_exp) { return 0x5EED0000u ^ ((l * n_exp + e) * 3u + j); }

// router: 6 distinct skewed ids from the state
kernel void sync_t_router(device uint* st [[buffer(0)]], device int* rids [[buffer(1)]], constant SyncTArgs& a [[buffer(2)]],
                          device uint* junk [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const uint v = sync_t_burn(tid, a.iters);
  if (v == 0x12345678u) junk[0] = v;
  if (tid != 0) return;
  const uint s = st[0];
  int ids[6];
  for (int k = 0; k < 6; k++) {
    const uint h = sync_mix(s + 0x9e3779b9u * (uint)(k + 1) + a.l * 7919u);
    const uint u = h & 0xffffu;
    const uint u3 = (((u * u) >> 16) * u) >> 16;   // skewed: hot experts (cache hits) dominate
    uint id = (uint)(((ulong)u3 * a.n_exp) >> 16);
    for (bool dup = true; dup;) { dup = false; for (int q = 0; q < k; q++) if ((uint)ids[q] == id) { dup = true; id = (id + 1) % a.n_exp; } }
    ids[k] = (int)id;
  }
  for (int k = 0; k < 6; k++) rids[k] = ids[k];
}

// gap work (shared-expert stand-in): sh = f(state)
kernel void sync_t_gap(device uint* st [[buffer(0)]], device uint* sh [[buffer(1)]], constant SyncTArgs& a [[buffer(2)]],
                       device uint* junk [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const uint v = sync_t_burn(tid, a.iters);
  if (v == 0x12345678u) junk[0] = v;
  if (tid != 0) return;
  sh[0] = sync_mix(st[0] ^ (a.l * 0x1234567u) ^ a.tok);
}

// routed MoE stand-in: expert matrices from the CPU-written table; checks every matrix signature against the router
// ids (bad[0]) and that the gap result belongs to this layer's state (bad[1]: ordering), folds expert data into state.
struct SyncTPtr { device const uint* m[3]; };
kernel void sync_t_moe(device uint* st [[buffer(0)]], device const uint* sh [[buffer(1)]], constant SyncTArgs& a [[buffer(2)]],
                       device uint* junk [[buffer(3)]], device const int* rids [[buffer(4)]],
                       device const SyncTPtr* tbl [[buffer(5)]], device atomic_uint* bad [[buffer(6)]],
                       uint tid [[thread_position_in_grid]]) {
  const uint v = sync_t_burn(tid, a.iters);
  if (v == 0x12345678u) junk[0] = v;
  if (tid != 0) return;
  const uint s = st[0];
  if (sh[0] != sync_mix(s ^ (a.l * 0x1234567u) ^ a.tok)) atomic_fetch_add_explicit(bad + 1, 1u, memory_order_relaxed);
  uint acc = 0;
  for (int k = 0; k < 6; k++)
    for (int j = 0; j < 3; j++) {
      device const uint* p = tbl[k].m[j];
      if (p[0] != sync_t_sig(a.l, (uint)rids[k], (uint)j, a.n_exp)) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
      acc ^= p[1 + (s + (uint)(k * 3 + j)) % 15u] + (uint)(k * 3 + j);
    }
  st[0] = sync_mix(s ^ acc ^ sh[0]);
}
