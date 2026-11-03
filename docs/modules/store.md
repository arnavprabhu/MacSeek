# Module: store (GGUF parser, resident weights, expert cache, engram fetcher, memory guard)

Files: `src/gguf/gguf.{hpp,cpp}`, `src/store/{io_pool,weights,experts(.mm),policy,engram,memguard}.*`,
tests `tests/test_gguf.cpp`, `tests/test_store_{weights,experts,engram}.mm`, benches `bench/{experts_io,engram_io,weights_load}.mm`.
Everything is written from scratch (C++20 / ObjC++). No mmap of model data on any hot path: all model I/O is
`pread` on `F_NOCACHE` fds through a thread pool, landing directly in page-aligned memory wrapped as no-copy shared
MTLBuffers.

## 1. GGUF parser (`ms::Gguf`)

```cpp
ms::Gguf g(ms::Gguf::default_model_path());   // any split path -> all 5 splits ($MACSEEK_MODEL overrides)
g.get_i64("deepseek41.block_count"); g.get_arr_str("tokenizer.ggml.tokens"); g.get_arr_i64("deepseek41.engram.token_map");
g.get_arr_f32("deepseek41.swiglu_clamp_exp"); const ms::KvValue* v = g.kv("...");   // typed: type/arr_type, i/u/f/s, ai/af/as/aa
const ms::TensorInfo& t = g.at("blk.0.attn_q_b.weight");  // type, n_dims, ne[4] (ggml order), nbytes, split, ABSOLUTE offset
g.fd(split) /* page-cached */; g.fd_nocache(split) /* F_NOCACHE + F_RDAHEAD 0 */; g.file_size(split); g.alignment();
ms::type_info(GType) / type_bytes(GType, n) / type_name(GType)   // all ggml types incl. IQ1_S 256/50, IQ2_XXS 256/66, IQ4_NL 32/18
ms::pread_full(fd, dst, len, off)                                  // EINTR/short-read safe
```
Merged metadata (first occurrence wins, insertion order kept), nested arrays supported, data offset =
align_up(header end, general.alignment (default 32)). Parse of all 5 headers (129280 tokens + 127741 merges + 129280
token_map) takes **~8 ms**. `test_gguf` checks all **1043/1043** tensors against `docs/tensors.txt` (name, type, dims,
nbytes, file, absolute offset) plus typed metadata spot checks.

## 2. Resident weights (`ms::Weights`, `ms::TokenEmbd`)

```cpp
ms::Weights w(g);                                  // WeightsOptions::from_env(): MACSEEK_EMBD_RESIDENT, _WEIGHTS_THREADS, _WEIGHTS_MLOCK
const ms::WTensor& q = w.at("blk.0.attn_q_b.weight");   // {name, type, n_dims, ne[4], nbytes, Buffer buf, void* host, info}
w.buffers();                                       // MTLBuffers (for a residency set)
w.repack([](const WTensor& src, WTensor& out) { ...; return true; });   // per-tensor hook; frees replaced arena pages
ms::TokenEmbd te(g, &w); te.fetch(tokens, n, dst); // n x 4200 B Q6_K rows
```
- Loads every tensor except `*_exps`, `per_layer_token_embd` and (by default) `token_embd`: **921 tensors, 6.154 GiB**.
- Tensors are grouped into contiguous file runs (gaps <= 256 KiB merged). Each run goes into the arena at a
  16 KiB-aligned position with `dst == file offset mod 16 KiB`, so every pread is 16 KiB-aligned on both sides and every
  tensor keeps its 32 B GGUF alignment. One anonymous mmap, wrapped into MTLBuffers of <= min(maxBufferLength, 16 GiB);
  a run never straddles two buffers.
- **token_embd is pread-on-demand (default).** Decode needs 1 row (4200 B) per token, issued right after sampling
  (~0.1 ms, cached fd: hot tokens stay in the *reclaimable* OS page cache); prefill batches rows in parallel. Making
  it resident would pin 518 MiB, i.e. ~70 expert slots (~1.3% of the arena), for a lookup that touches 0.0008% of it.
  `MACSEEK_EMBD_RESIDENT=1` switches it to resident (memcpy rows).
- **The arena is wired (`mlock`) before the reads by default** (`MACSEEK_WEIGHTS_MLOCK=0` to disable; RLIMIT_MEMLOCK is
  unlimited here). Measured: without it, macOS compressed ~4.8 GB of the freshly DMA-filled weights within a second
  (rss 1.39 GiB vs footprint 6.18 GiB). That's a silent decompression tax on the first GPU passes, and it comes back
  under pressure.

## 3. Expert cache (`ms::ExpertCache`)

```cpp
ms::ExpertLayout L = ms::ExpertLayout::from_gguf(g);  // per layer: {fd, base, stride, len, type} for gate/up/down
ms::ExpertCacheOptions o = ms::ExpertCacheOptions::from_env();
o.arena_bytes = ms::recommend_arena_bytes(planned_other);   // see §5
ms::ExpertCache cache(L, o);

// decode, per layer, after the router: misses are read in parallel at high priority; returns at once
ms::ExpertLease lease = cache.ensure(layer, ids, 6);
lease.encode_wait(cmdbuf);          // GPU waits io_event >= lease.ticket (ticket 0 = all hits, no wait)  -- or lease.wait() on CPU
lease.views[i].mat[k]               // ms::Buffer for gate(0)/up(1)/down(2); .gpu[k] = GPU address; .type[k]
lease.release_after(cmdbuf);        // slot refs dropped by the command buffer's completion handler (call before commit)

cache.prefetch(layer_next, predicted_ids, n);   // low priority; promoted to high if ensure() asks for it while queued
cache.pin(layer, ids, n); cache.unpin(...);      // never evicted
auto st = cache.stream(layer, ids, n);           // prefill bulk mode, see below
cache.layer_stats(l) / total_stats() / stats_report() / write_routing_stats(path)
```

**Region pools, one eviction domain.** Expert e's matrix is the e-th contiguous slab (nbytes/n_experts) of each 3D
tensor. Every matrix type gets a REGION size = max over all (layer, expert) matrices of that type of
`rup(file offset mod 16 KiB + len, 16 KiB)`; types with equal region sizes share a pool. A pool is an array of
fixed-size regions in page-aligned anonymous memory, wrapped as no-copy shared MTLBuffers of
<= min(maxBufferLength, 8 GiB) whole regions (`cache.buffers()` for a residency set). A matrix sits in its region at
`file offset mod 16 KiB`, so the 16 KiB-aligned read lands in place (no copy, no bounce buffer). The arena is split
across pools in proportion to their total demand (every expert of every layer), so every pool holds the same fraction
of its matrices; with a routing-statistics file: a floor of 16 experts per layer, then greedy count/byte selection,
then the rest by remaining demand. GSQ-RCO Coder at 34.5 GiB (`describe()`):

| pool | type | region | regions | GiB | demand | regions / entry (mean / max) |
|---|---|---|---|---|---|---|
| 0 | IQ1_M | 2,605,056 B | 6406 | 15.54 | 11520 | 1.50 / 3 |
| 1 | IQ2_XXS | 3,063,808 B | 4484 | 12.79 | 8064 | 1.05 / 3 |
| 2 | IQ2_XS | 3,440,640 B | 1922 | 6.16 | 3456 | 0.45 / 3 |

= 55.6% of all 40 x 192 experts (34.5 of 62.0 GiB in regions). UD-IQ1_S gets 2 pools (IQ1_S, IQ2_XXS).

An ENTRY is one resident (layer, expert) owning 3 regions (gate, up, down), possibly in different pools; entries are
what the policy, the key map, leases, `ExpertView::slot` and pinning see (`n_slots()` = total regions / 3 entry ids).
The policy has ONE class: global LRU / LFU / Static over the whole arena (no per-layer or per-size-class quotas). A
miss needing region sizes {a, b, c} takes free regions; if some size has none, policy victims that own a region of a
missing size are evicted (all 3 of their regions go back to their pools) until all are available. Replaces the old
per-size-class slot pools (Coder: 8 classes of 1-14 layers, each needing its own streaming room, so ~30% of the arena
could not be pinned during prefill).

With `prefault` (default on), every arena page is touched at startup by 8 threads, which keeps zero-fill page faults
(~2 µs per 16 KiB page, ~1 ms per expert) off the miss path. `wire` (`MACSEEK_ARENA_MLOCK=1`) mlocks the arena
instead, so it is never compressed or paged out.

**Safety (never evict what the GPU uses).** An entry can be evicted only if it is Ready, has refcount 0 and is not
pinned. `ensure()` takes a ref per returned view; refs are dropped by `release()` / the lease destructor /
`release_after(cmdbuf)` (Metal completion handler). If a needed region size has nothing evictable, `ensure()` blocks
(and aborts after 30 s with a per-pool message). Tested with a command buffer held open on an event: 80 churning misses
never evict its entries, and they become evictable after completion.

**Prefetch protection.** An unused prefetch is protected while young (fewer than `n_slots()` demand accesses since it
was issued; `MACSEEK_PF_PROTECT=<accesses>`, 0 = off): `prefetch()` never evicts it, `ensure()` only as a last resort,
and then the youngest / highest-key one (layer l+1's before the current layer's own pending prefetches; a prefill uses
a layer's experts in id order). Plain LRU evicted the current layer's next prefetches first and cascaded. Speculation
cap: `prefetch()` stops when fewer than `min(unpinned/4, 64 x max regions per entry)` unpinned regions of a needed size
would remain neither leased nor speculative (room for one demand group). Calling `prefetch(l+1, all)` again after each
of layer l's groups tops the next layer up as layer l's prefetches get used.

**Prefill pinning.** `pin_resident(keep)` pins Ready entries, most-accessed first, while every pool keeps at least
`reserve_regions(p, keep) = max(ceil(keep x max(1, mean regions of p per entry)), 24 x max per entry)` regions
unpinned (capped at the pool). ds41 default `MACSEEK_PF_PIN_FREE=256`: Coder 384/269/256 regions = 2.52 GiB (7.3% of
34.5 GiB) stay unpinned, so up to 51.5% of all experts stay pinned across chunks (old design: ~30% of the arena was
unpinnable). `group_cap(l) = clamp(unpinned capacity(l) / 3, 1, 64)` where unpinned capacity = min over the layer's
pools of unpinned regions / regions per entry: two groups leased at once always fit (Coder after pinning: 42 for
IQ1_M x3 and IQ2_XS/IQ2_XS/IQ1_M layers, 28-29 for 3-matrix IQ2_XXS / IQ2_XS layers; 64 before the first pin).

**I/O.** `IoPool`: N workers (default 8) each issue one blocking pread (queue depth = N). Two priority queues (demand,
prefetch), FIFO across requests, so the oldest demand request finishes first. A request = the 3 matrix ranges split
into `read_chunk` pieces (default 1 MiB). All requests built under the cache mutex are submitted after it is released,
with one wake-up (batched submit). Options: `align` (16384 default; 1 = exact ranges), `read_chunk` (0 = one pread per
matrix).

**GPU sync.** Leases that must wait get monotonically increasing tickets. `io_event()` (MTLSharedEvent) is signaled
with the highest ticket whose lease *and all earlier leases* are complete. So one command buffer can be encoded and
committed before the reads finish (`encodeWaitForEvent`). CPU `lease.wait()` waits per slot, so it never waits on
unrelated leases.

**Policies** (`PolicyKind` or a custom `EvictionPolicy` subclass via `o.custom_policy`; one class = the whole arena;
the cache's `evictable()` predicate also says which region sizes a victim must free): `LRU` (intrusive list, victim =
first evictable from the tail); `LFU` with aging (counts per (layer, expert) survive eviction, x`lfu_decay` every
`lfu_period` accesses; victim = least frequent of 64 random candidates, `lfu_sample=0` = exact scan); `Static` =
frequency-pinned set from a routing-statistics file (globally most frequent (layer, expert) by count / entry bytes
until `pin_fraction`=85% of every pool is pinned; preloaded at startup, the rest LRU).
Pinned slots leave the policy's candidate lists, so victim search stays O(1). Stats file format: one line per layer
(40), 384 whitespace-separated counts, `#` comments; `write_routing_stats()` dumps the observed demand counts in
the same format.

**Bulk stream (prefill).** `cache.stream(layer, ids, n)` -> `ExpertStream`. Cached experts are served from the cache
(one lease, `items()[0..n_cached)`). Misses go in file order through a ring of `ring_slots` staging slots (separate
memory, sized for the largest entry), loaded by a feeder thread on the same pool while the GPU consumes them. Protocol
per ring item i, in order: GPU `encode_wait_ready(cb,i)` -> kernels -> `encode_consumed(cb,i)` (between compute
encoders), or CPU `wait_ready(i)` / `consumed(i)`. The feeder overwrites a ring slot only after its previous occupant was
consumed. Values are contiguous, so a group of g items can wait on the last one and signal the last one. Tested with a
real Metal kernel reading 32 streamed experts through a 3-slot ring in a single command buffer. Rules: consume every
ring item, and don't mix CPU `consumed()` with GPU `encode_consumed()`.

**Stats per layer:** hits, misses, inflight_hits (present but still loading), prefetch issued/used/wasted, evictions,
bytes_read, streamed/stream_bytes, stall_ms (CPU time blocked in wait / wait_ready).

## 4. Engram fetcher (`ms::EngramFetcher`)

```cpp
ms::EngramFetcher ef(g);   // EngramOptions::from_env(): MACSEEK_ENGRAM_THREADS (QD), _MODE=exact|4k|16k, _CACHE_MB
auto tk = ef.fetch(rows, n, dst);   // async: row i -> dst + i*144 (dst = contents of a shared MTLBuffer)
tk.wait();                          // or ef.fetch_sync(...)
```
Rows (global ids into the [256 x 768022850] IQ4_NL table) are sorted and deduped. Repeats come from a set-associative
8-way LRU row cache (default 512 MiB, 256 lock stripes, no allocations). Misses become preads on the pool. In 4k/16k
modes, misses sharing or abutting a block are merged into one read (<= 64 KiB) and read into worker scratch, then
scattered. A prompt's rows can be fetched asynchronously while layer 0 runs (the hashes depend only on token ids).

## 5. Memory guard (`ms::mem_status`, `ms::recommend_arena_bytes`)

`mem_status()` gives total / free / inactive / speculative / purgeable / file-backed / active / wired / compressed
(host_statistics64), swap used/total (`vm.swapusage`), and this process's RSS and phys_footprint (TASK_VM_INFO).
`recommend_arena_bytes(planned_other, headroom = 8 GiB, cap)` = available - headroom - planned_other. Call it
**before** allocating the arena. `swap_grew(before, after)` is for run hygiene.

**Recommended arena (64 GB machine, nothing else big running):** 64 - 8 headroom - ~7 OS/apps/wired - 6.15 resident
weights - 2 prefill scratch - 1 state - 0.5 engram cache - 0.2 ring/misc ≈ **38-39 GiB**. Use
`min(recommend_arena_bytes(weights + scratch + state + engram_cache), 40 GiB)`, and check `vm.swapusage` before and
after a run.

## 6. Tests (ctest; all pass)

- `test_gguf`: 1043/1043 tensors match `docs/tensors.txt`; typed metadata (bool, f32, string, int/float/string
  arrays, u64 multipliers, 43-item compress_ratios, chat template).
- `test_store_experts`: synthetic expert files with 3 region sizes and mixed per-layer types (AAA, AAB, BBA, BBB, CCA,
  CCC), byte-exact checks. Covers: pool geometry / proportional split, miss/hit/data/stats, io_event GPU wait, global
  LRU order and size-targeted victims, in-use protection (held lease + a GPU command buffer blocked on an event),
  prefetch used/wasted, promotion, young-prefetch protection + last-resort victim + speculation cap, LFU keeps a hot
  expert where LRU does not, static pinning from a stats file (per-pool budgets, + stats write/read), alignment
  {1, 4K, 16K} x chunk {whole, 16K, 1M}, mlock-wired arena with many small MTLBuffers, stream with a CPU / GPU consumer,
  a 2-thread randomized ensure/prefetch/pin/unpin/pin_resident/unpin_resident stress (LRU and LFU; every view and every
  final resident checked against pread), and a prefill simulation with 192 experts/layer (4 chunks: group_cap groups
  with two leases alive released by command-buffer completion, next-layer prefetch, pin_resident between chunks:
  pinned entries never evicted, loads per chunk ~= unpinned experts). Plus 7 real experts of layers 0, 7, 17, 23, 39
  (mixed types) vs plain preads, and `describe()` at 34.5 GiB with dry_io.
- `test_store_engram`: synthetic table, all 3 modes. Covers: dedupe, cache hits (no reads on repeat), batch of
  20000 rows, last row at EOF, no-cache mode. Plus 48 real rows per mode vs plain preads.
- `test_store_weights`: subset load (layers 0-1 + globals, 0.96 GB, 2 MTLBuffers) byte-exact vs pread; repack hook;
  TokenEmbd rows; memguard.

## 7. Benchmarks (real SSD; page-cache hygiene: experts/rows resident in the page cache (mincore) are avoided and nothing is read twice)

All runs held the model lock (exclusive SSD). Raw outputs: `docs/modules/store-bench/*.txt`.

**Resident weights (`bench_weights_load`, 921 tensors, 6.154 GiB, 6.609 GB read incl. 16 KiB rounding).**
Caveat: every run happened right after llama.cpp oracle/baseline runs that mmap the model. A mincore check showed
**100% of the weight pages in the page cache**, and F_NOCACHE preads are served from cached pages. So these are *warm*
numbers (memory copy + VM cost), not SSD numbers. They are still valid for comparing the variants:

| config (warm page cache) | load | GB/s | after load |
|---|---|---|---|
| 8 thr x 8 MiB, no mlock | 986 ms | 6.70 | rss 1.39 GiB / footprint 6.18: **~4.8 GB compressed by macOS within 1 s** |
| 16 thr x 16 MiB, mlock after read | 1201 ms + 1179 ms mlock | 5.50 | rss 6.16 = footprint 6.18 (wired) |
| 8 thr x 8 MiB, prefault (touch) first | 2383 ms | 2.77 | rss 0.63: idle zero pages compressed, then refaulted |
| **8 thr x 8 MiB, mlock BEFORE read (default)** | **614 ms** (372 mlock + 242 read) | 10.8 | rss 6.18 = footprint 6.18, nothing compressed |

So wiring before the reads is both the fastest and the only variant that stays uncompressed. It is the default
(`MACSEEK_WEIGHTS_MLOCK=0` to disable). A cold start should be ~6.6 GB / ~11.5 GB/s sequential ≈ 0.6 s + 0.37 s mlock
≈ **1 s**, but that was not measured: the model lock was busy. `bench_weights_load` now prints the page-cache
residency first, so a cold run qualifies itself.

**Expert misses (`bench_experts_io lat`): 6 cold misses in one random layer, ensure() -> lease.wait() (12-20 iters per
config, 41-55 MB per call):**

| threads | chunk | align | p50 ms | p90 ms | eff GB/s | ensure() call | CPU / miss |
|---|---|---|---|---|---|---|---|
| 8 | 1 MiB | 16K | 3.65 | 3.82 | 12.3 | 4.4 µs | **339 µs** |
| 16 | 1 MiB | 16K | 3.75 | 4.07 | 11.6 | 3.6 µs | 766 µs |
| 16 | whole matrix | 16K | 4.00 | 4.56 | 11.6 | 3.5 µs | 1293 µs |
| 16 | 256 KiB | 16K | 3.67 | 4.33 | 12.6 | 4.3 µs | 630 µs |
| 16 | 1 MiB | exact (1) | 3.58 | 3.92 | 12.1 | 6.2 µs | 1084 µs |
| 16 | whole matrix | exact (1) | 3.88 | 4.85 | 11.0 | 5.1 µs | 2167 µs |
| 12 | 512 KiB | 16K | 4.29 | 6.00 | 10.2 | 3.8 µs | 682 µs |

- Latency is flat at ~3.6-3.8 ms for 6 misses (~12 GB/s; the SSD ceiling for this size is ~14-15 GB/s). Threads,
  chunk size and alignment don't change latency within noise. They do change **CPU**: 16 KiB-aligned 1 MiB chunks
  on 8 threads cost 339 µs per 7 MB miss, the same as a bare pread (~46 µs/MB, separate probe). Exact
  (unaligned) ranges cost 1.4-1.7x more CPU, and whole-matrix preads more again. Defaults: **8 threads, 1 MiB chunks,
  16 KiB alignment**.
- ensure() call time (lock + policy + request build + submit) went from 35-86 µs to 3.5-6 µs once the submit woke
  one worker instead of all 16 (chain wake-up: each woken worker wakes the next while work remains).
- 16 concurrent clients, 1 miss each (16 pool threads): 160 experts, 1.21 GB in 86.6 ms = **13.9 GB/s**; per-ensure
  p50 8.6 ms, p99 9.3 ms (queueing behind the 15 other clients).
- Bulk stream, 48 cold experts of one layer through the ring, CPU consumer: ring 8: **13.05 GB/s**; ring 16:
  12.2 GB/s (0.33 GB in 26-27 ms). One full layer (384 experts, 2.65-3.5 GB) would take ~0.2-0.27 s, i.e. ~9 s per
  prefill-chunk pass over all 40 layers when nothing is cached.
- Decode cost model: a token with m expert misses waits ~ m x 7.3 MB / 12 GB/s ≈ 0.6 ms per miss (when issued
  together per layer).

**Policy simulation (`bench_experts_io sim`, dry_io, real geometry, 40 GiB arena = 5640 slots, synthetic Zipf routing,
second half of a 4000-token trace measured after warming on the first half; static stats = first half):**

| routing | LRU | LFU exact | LFU sampled 64 | Static (85% pinned) |
|---|---|---|---|---|
| Zipf s=0.6 hit / miss GB per token | 52.7% / 0.852 | 58.2% / 0.752 | 58.2% / 0.753 | **62.3% / 0.686** |
| Zipf s=1.0 hit / miss GB per token | 75.8% / 0.436 | 79.0% / 0.377 | 79.0% / 0.378 | **81.9% / 0.329** |
| ensure(6) bookkeeping | 0.3-0.5 µs | 3.9-7.6 µs | 0.5-0.9 µs | 0.24-0.42 µs |

Synthetic routing, so only the ranking is meaningful. Frequency beats recency, and sampled LFU loses nothing
against exact LFU. Real routing statistics from the model decide the final policy.

**Engram rows (`bench_engram_io`, uniformly random cold rows, row cache off):**

Decode, 48 rows per token (100 tokens per config):

| mode | QD 8 p50 / p99 | QD 16 p50 / p99 | QD 32 p50 / p99 |
|---|---|---|---|
| exact 144 B | 0.531 / 0.590 ms | 0.371 / 0.422 | **0.322 / 0.409** |
| 4 KiB block | 0.472 / 0.532 | 0.343 / 0.418 | 0.316 / 0.432 |
| 16 KiB page | 0.591 / 2.945 | 0.376 / 4.496 | 0.329 / 0.391 |

Prompt batch, 256 tokens = 12288 rows in one async request:

| mode | QD 16 | QD 32 | QD 64 | bytes read |
|---|---|---|---|---|
| exact | 69.5 ms (177K rows/s) | 57.9 ms (212K) | **51.8 ms (237K rows/s)** | 1.8 MB |
| 4 KiB | 63.5 ms (194K) | 56.9 ms (216K) | 52.1 ms (236K) | 51.9 MB |
| 16 KiB | 69.8 ms (176K) | 58.4 ms (210K) | 52.9 ms (232K) | 202.7 MB |

- Decode target (< 1 ms) met with margin: **0.32 ms p50 / 0.41 ms p99 at QD 32**. Default: exact mode, 32
  threads. Small reads are IOPS-bound: ~210K IOPS at QD 32 and ~235K at QD 64, for 144 B and 16 KiB reads alike (more
  than the ~95K in SPEC §1, which came from a different probe). Exact 144 B reads move 100x fewer bytes than 16 KiB
  pages at the same latency, so exact is the default.
- Long prompts: 48 rows/token at ~237K rows/s ≈ **0.2 ms/token, about 40 s for a 200K-token prompt** before dedupe.
  This has to overlap with prefill compute: issue chunk k+1's rows while chunk k runs. Try QD 128 next.

## 8. Limits / caveats

- Expert data is served in GGUF block layout. A repacked expert layout would need a repack-on-load hook in
  `load_locked` (CPU cost per miss) or a pre-repacked expert file on SSD.
- `stream()` uses one global ring: streams are consumed in creation order (fine for layer-sequential prefill). Every ring
  item must be consumed, or later streams stall.
- `ensure()` blocks if a needed region size has nothing evictable. Size the arena so concurrent leases (<= 6 per layer x
  layers in flight, or two prefill groups of `group_cap`) are a small fraction of each pool.
- `group_cap()` counts every expert of a group against the unpinned room, but pinned (hit) experts need none: a
  caller could size groups by their non-pinned members (`is_pinned()`) for larger groups after pin_resident.
- dry_io mode (policy simulation) allocates nothing. Its numbers in §7 use a synthetic Zipf routing trace; real
  routing statistics must come from the model (`write_routing_stats`).
- The F_NOCACHE fds still serve pages already in the unified page cache (e.g. after an mmap-based oracle run), so cold
  numbers need the mincore filter used by the benches.

## 9. Next ideas

1. Feed real routing traces (from the oracle or engine) into `bench_experts_io sim` style replay to pick the policy
   and `pin_fraction`; Static + LFU-tail is likely best.
2. Prefetch with a predictor: run layer L+1's router on layer L's hidden state (or a small learned predictor), then
   `prefetch()`. Measure prefetch_used/wasted per layer.
3. Prefill stream -> cache admission: insert streamed experts into evictable slots when the policy rates them above the
   current victim (no extra reads).
4. Batch decode / speculative verify: `ensure()` the union of experts for k positions in one call (one ticket).
5. Expert-major file layout (a one-time rewrite of the 107 GiB of experts so each slot is one contiguous read)
   would turn 3 reads/miss into 1 and allow > 1 MiB sequential runs. Measure first: the chunked reads already reach
   12 GB/s.
6. Engram decode: issue the 48 rows right after sampling (the hashes need only token ids), overlapped with
   token_embd + layer 0; layer 1 waits on the ticket. That hides the ~0.3 ms fetch. Prefill: fetch chunk k+1's rows
   during chunk k, and try QD 64-128.
