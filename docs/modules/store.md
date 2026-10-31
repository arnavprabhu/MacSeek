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
- `mlock` is optional (`MACSEEK_WEIGHTS_MLOCK=1`, RLIMIT_MEMLOCK is unlimited here). Note from the bench: under memory
  pressure macOS compressed ~4.8 GB of freshly loaded weights within a second (`rss 1.39 GiB`, `footprint 6.18 GiB`).
  Pages the GPU reads every token stay hot in practice, but if decode shows compressor activity, turn mlock on.

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

**Slots and size classes.** A slot is gate+up+down of one (layer, expert). Expert e's matrix is the e-th contiguous
slab (nbytes/384) of each 3D tensor. Each matrix is placed in the slot at `file offset mod 16 KiB` inside a
16 KiB-rounded region, so the 16 KiB-aligned read lands in place (no copy, no bounce buffer). Region sizes are
computed exactly (max over the 384 experts) per layer. Layers with identical regions form a class with its own pool:

| class | layers | gate/up/down | data | slot | slots/layer @ 40 GiB |
|---|---|---|---|---|---|
| 0 | 0-19 (20) | IQ1_S/IQ1_S/IQ1_S | 6.912 MB | 6,979,584 B | 141 |
| 1 | 20-29, 35-38 (14) | IQ1_S/IQ1_S/IQ2_XXS | 7.649 MB | 7,716,864 B | 141 |
| 2 | 30-34, 39 (6) | IQ2_XXS x3 | 9.124 MB | 9,191,424 B | 141 |

Padding is <1% (vs 25% if every slot were 9.19 MB). At 40 GiB: 5640 slots = 36.7% of all experts. Without routing
stats the arena is split as equal slots per layer. With a routing-statistics file it is split greedily by
count/byte over all (layer, expert), with a floor of 16 slots per layer. Pools are split into MTLBuffers of
<= min(maxBufferLength, 8 GiB) whole slots (`cache.buffers()` gives them for a residency set). With
`prefault` (default on), every arena page is touched at startup by 8 threads, which keeps zero-fill page faults
(~2 µs per 16 KiB page, ~1 ms per expert) off the miss path.

**Safety (never evict what the GPU uses).** A slot can be evicted only if it is Ready, has refcount 0 and is not
pinned. `ensure()` takes a ref per returned view; refs are dropped by `release()` / the lease destructor /
`release_after(cmdbuf)` (Metal completion handler). If every slot of a class is held, `ensure()` blocks (and aborts
after 30 s with a message). Tested with a command buffer held open on an event: 80 churning misses never evict its slots,
and they become evictable after completion.

**I/O.** `IoPool`: N workers (default 12) each issue one blocking pread (queue depth = N). Two priority queues (demand,
prefetch), FIFO across requests, so the oldest demand request finishes first. A request = the 3 matrix ranges split
into `read_chunk` pieces (default 1 MiB). All requests built under the cache mutex are submitted after it is released,
with one wake-up (batched submit). Options: `align` (16384 default; 1 = exact ranges), `read_chunk` (0 = one pread per
matrix).

**GPU sync.** Leases that must wait get monotonically increasing tickets. `io_event()` (MTLSharedEvent) is signaled
with the highest ticket whose lease *and all earlier leases* are complete. So one command buffer can be encoded and
committed before the reads finish (`encodeWaitForEvent`). CPU `lease.wait()` waits per slot, so it never waits on
unrelated leases.

**Policies** (`PolicyKind` or a custom `EvictionPolicy` subclass via `o.custom_policy`): `LRU` (O(1) intrusive list
per class); `LFU` with aging (counts per (layer, expert) survive eviction, x`lfu_decay` every `lfu_period` accesses;
victim = least frequent of 64 random candidates, `lfu_sample=0` = exact scan); `Static` = frequency-pinned set from a
routing-statistics file (top `pin_fraction`=85% of each class's slots pinned and preloaded at startup, the rest LRU).
Pinned slots leave the policy's candidate lists, so victim search stays O(1). Stats file format: one line per layer
(40), 384 whitespace-separated counts, `#` comments; `write_routing_stats()` dumps the observed demand counts in
the same format.

**Bulk stream (prefill).** `cache.stream(layer, ids, n)` -> `ExpertStream`. Cached experts are served from the cache
(one lease, `items()[0..n_cached)`). Misses go in file order through a ring of `ring_slots` staging slots (separate
memory, sized for the largest class), loaded by a feeder thread on the same pool while the GPU consumes them. Protocol
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
- `test_store_experts`: synthetic 2-class expert file with byte-exact checks. Covers: miss/hit/data/stats, io_event
  GPU wait, LRU order, in-use protection (held lease + a GPU command buffer blocked on an event), prefetch
  used/wasted, LFU keeps a hot expert where LRU does not, static pinning from a stats file (+ stats write/read),
  alignment {1, 4K, 16K} x chunk {whole, 16K, 1M}, stream with a CPU consumer (cached + ring, file order, ring reuse
  across layers), stream with a GPU consumer (Metal kernel, 3-slot ring, one command buffer). Plus 5 real experts from
  all 3 classes vs plain preads.
- `test_store_engram`: synthetic table, all 3 modes. Covers: dedupe, cache hits (no reads on repeat), batch of
  20000 rows, last row at EOF, no-cache mode. Plus 48 real rows per mode vs plain preads.
- `test_store_weights`: subset load (layers 0-1 + globals, 0.96 GB, 2 MTLBuffers) byte-exact vs pread; repack hook;
  TokenEmbd rows; memguard.

## 7. Benchmarks (real SSD; page-cache hygiene: experts/rows resident in the page cache (mincore) are avoided and nothing is read twice)

BENCH_RESULTS

## 8. Limits / caveats

- Expert data is served in GGUF block layout. A repacked expert layout would need a repack-on-load hook in
  `load_locked` (CPU cost per miss) or a pre-repacked expert file on SSD.
- `stream()` uses one global ring: streams are consumed in creation order (fine for layer-sequential prefill). Every ring
  item must be consumed, or later streams stall.
- `ensure()` blocks if a class has no evictable slot. Size the arena so concurrent leases (<= 6 per layer x layers in
  flight) are a small fraction of each class.
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
6. Engram: map the 48 rows of the next token speculatively (all n-gram suffixes are known once the token is sampled)
   and issue them before the token's layer 0, which hides the ~0.3 ms decode fetch completely.
