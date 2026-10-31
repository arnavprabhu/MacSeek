# Decode router handshake (MACSEEK_DEC_SYNC)

Table decode (src/model/ds41.mm, "Table decode") encodes one token as ONE command buffer. Each layer has one
GPU->CPU->GPU handshake: the router writes 6 ids, the GPU signals `rdy`, the CPU calls `ensure()` and writes the
experts' device addresses into `P.tbl[l]`, the CPU sets `go`, and the GPU runs the routed MoE. Measured on the model
(200K context, all hits forced): 44.1 ms/token, against 37.6 ms with no handshakes. That is about 160 us per
handshake.

## Measurements (bench/sync_probe.mm, M5 Pro, `taskpolicy -b`, a model job sharing the GPU)

The chain is 40 dependent one-simdgroup dispatches (39 handshakes). Per-handshake cost = (token - no-handshake token)
/ 39. The table gives min and median over 30 round-robin tokens. Every handshake mode also verifies that each layer read
the value the CPU wrote for the previous layer's GPU result.

| mode | per handshake min / med (us) | stale reads | notes |
|---|---|---|---|
| split (encoder end/begin only) | 0.1 / -0.3 | - | encoder splits are free |
| nohs_serial (serial dispatch type) | -0.5 / -0.7 | - | no difference |
| split_sig (encodeSignalEvent, nobody waits) | 11.6 / 11.9 | - | GPU-side cost of a signal (flush) |
| split_presat (wait on an already-signaled value) | 3.5 / 1.7 | - | a satisfied wait is ~free |
| **event (current)** | **44.3 / 63.4** | 0 | X 13-15 us GPU->CPU, Y 19-60 us CPU->GPU wake-up |
| event_listen (MTLSharedEventListener) | 91 / 107 | 0 | dispatch-queue hop: slower |
| event_cbsig (CPU commits a pre-encoded CB that GPU-signals go) | 91 / 95 | 0 | Y 42-58 us: slower |
| memflag_event (GPU atomic store -> CPU polls memory) | 46 / 58 | **122** | UNSAFE: CPU sees the flag before the data |
| gpupoll (kernel spins on a CPU-written word) | 668 / 1391 | **1196**, 44 timeouts | UNSAFE and slow |
| gpupoll_fb (+ event fallback) | 684 / 791 | **1778** | UNSAFE |
| **event_gap (50 us of independent GPU work between rdy and go)** | **7.7 / 9.2** | 0 | vs nohs with the same work |

Gap sweep (`SP_GAPONLY=1`): gap 25 us leaves 27 us per handshake exposed, 50 us leaves 9 us, 100 us leaves 7 us. With a
packed table (`SP_TS=1`, so layers share cache lines as P.tbl does), event mode had 0 stale reads: the event wait
invalidates the GPU caches.

Breakdown (`breakdown`, encoder-boundary GPU timestamps correlated with CPU time): GPU idle per handshake has median
39-78 us. X (layer end -> CPU sees `rdy`) is 13-15 us. Y (CPU sets `go` -> GPU starts the next encoder) is 19-63 us.
The CPU->GPU wake-up dominates. A wait whose value is already set costs ~2-4 us.

CPU <-> running-kernel memory (`vis`, `pingpong`):
- A running kernel sees CPU stores only through `volatile device coherent(device)` loads. Atomic loads, RMWs and
  fences keep hitting a stale GPU cache line, sometimes for seconds. `sys_dcache_flush` and write-combined CPU mappings
  do not help.
- Visibility is eviction-driven. The ping-pong round trip has p50 40-300 us and p99 1-4 ms.
- Nothing is ordered across cache lines, in either direction. A payload written before the flag was stale in ~45%
  (GPU->CPU) and ~1-5% (CPU->GPU) of the rounds. That holds even with atomic stores and seq_cst fences.
- So GPU-side polling and memory-flag signaling are rejected. Only events (or command-buffer boundaries) make CPU and
  GPU memory agree.

## What is implemented

`MACSEEK_DEC_SYNC=<mode>` sets the table decode path. The default is `event`.

| value | GPU sequence per layer | output |
|---|---|---|
| `event` / `0` (default, the old path) | attn side, router -> signal rdy -> wait go -> MoE (routed + shared fused) | reference |
| `sgu` / `1` | attn side, router, updateFence A -> signal rdy -> [waitFence A, **shared expert gate/up**, updateFence B] -> wait go -> [waitFence B, routed gate/up + down (routed + shared)] | **bit-identical** to event |
| `shared` / `2` | same, but the **whole shared expert** (gate/up + down, y += shared) runs in the gap; routed then does y += routed | same math, different float summation order |

- The shared expert depends only on the FFN input, not on the routing. Running it while the CPU resolves the routed
  experts means the GPU usually reaches the `go` wait after the CPU has already set it (a satisfied wait, ~2-4 us).
- Shared gate/up is 2 x 5120 x 2304 weights: 19 MB in Q6_K, 25 MB in Q8_0. That is ~60-100 us of GPU time, which covers
  X + CPU (~20-30 us on hits). `shared` adds the down projection, ~35-50 us more.
- The gap encoder needs **MTLFences**. Consecutive compute encoders over untracked buffers are not ordered by
  themselves (the probe saw encoder l+1 start before encoder l ended). The `go` wait only orders work encoded after it.
  So fence A orders the gap after the attention side, and fence B orders the routed MoE after the gap.
- In `sgu`/`shared`, the CPU handles an all-hit layer as: ensure -> write table -> set `go` -> next-layer prediction +
  prefetch. The prefetch is no longer on the GPU's critical path. On a miss the order stays as before: prefetch, then
  `lease.wait()`, then `go`.
- Kernels: `q::encode_moe_decode_table_part(e, a, table, MoePart::{SharedGu, AfterSharedGu, Shared, Routed})`. It
  reuses the existing `q_moe_*_t` kernels by rebasing the scratch to the shared row and setting n_routed = 0 for the
  shared-only pieces. No kernel changes.

Synthetic verification (tests/test_dec_sync.mm):
- (A) SharedGu + AfterSharedGu is bit-identical to the fused table MoE. Shared + Routed matches within 1e-7
  relative. The table MoE is bit-identical to the slot-bound MoE, which is CPU-checked. All three type combos pass,
  each run through a real signal/CPU/wait handshake.
- (B) Emulated 40-layer loop on the real ExpertCache: synthetic expert file, 1/3 arena (hit rate ~0.5, ~7.5K
  evictions), random prefetch pressure, random 0-600 us extra miss delays, leases held until the token's command
  buffer completes. Event and overlap modes alternate. The GPU checks every expert matrix signature against its own
  router ids and checks the gap result's ordering, and the CPU re-simulates the chain. Result: 0 stale/evicted reads,
  0 ordering violations, 0 chain mismatches.
- Timing in the emulation (noisy, shared GPU): all-hit saved 74-142 us per handshake, the miss config 30-90 us.

Expected on the model: most of the ~160 us per handshake is hidden when the gap work covers X + CPU. The probe shows
~8 us exposed with 50 us of gap work, against 44-63 us without. On all-hit tokens that is up to ~5 ms/token (44.1 ->
~39-40 ms). A/B it with `MACSEEK_DEC_SYNC=sgu`, then `shared`. Check that the greedy hash is unchanged for `sgu`.

## GPU-resolved expert map: considered, not implemented

The idea is to keep a GPU-visible (layer, expert) -> slot address map and let the GPU resolve hits itself. A safe
protocol exists:
- The CPU writes map entries only between command buffers. CPU writes during a running command buffer are not
  reliably visible or ordered (see above).
- An entry is published only for a Ready slot.
- Evicting a mapped slot clears its entry and **quarantines** the slot until every command buffer committed while it
  was mapped has completed (completion handler). Only then may the slot be reloaded. A GPU read of a stale entry then
  still sees the old, intact expert.

It does not remove the handshake, though. Encoded commands are static: every layer still needs an
`encodeWaitForEvent` in case of a miss. Letting the GPU skip that wait on all-hit layers needs a GPU-side decision,
and GPU polling of CPU memory is broken on this hardware. The remaining options are:
- a speculative token with indirect dispatches zeroed from the first miss on, then a replay from that layer's MoE
  (KV writes are position-indexed so the replay is idempotent, but re-encoding the rest of the token per miss costs
  ~19 restarts/token at a 0.9 hit rate), or
- having the CPU pre-signal `go`, which needs it to already know the ids.

The gap overlap captures most of the same win (the satisfied wait is ~2-4 us) without changing the cache. So the map
was not built. Revisit it only if the exposed per-handshake cost after `sgu`/`shared` is still significant.

## Risks / notes

- `shared` is not bit-identical. Gate it with the logit quality proxy, not the greedy hash.
- If the CPU path on a hit takes longer than the gap work (a slow `ensure`, a descheduled thread), the GPU waits as it
  did before. Nothing gets worse than `event`.
- Misses still wait for SSD I/O on the GPU (`go` is set after `lease.wait()`). The shared-expert overlap also hides
  ~60-130 us of each miss.
- The probe ran under `taskpolicy -b` (background QoS: E-cores) while a model job used the GPU. Absolute latencies on
  the real engine differ (160 us per handshake measured there), so take the ratios, not the absolute numbers.
