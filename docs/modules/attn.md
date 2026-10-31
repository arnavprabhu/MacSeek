# Module `attn` — attention side of DeepSeek-V4.1-Flash on Metal

Branch `mod/attn`. Files:

| file | content |
|---|---|
| `src/kernels/attn.hpp`, `attn.mm` | host API: `State` (all attention state + scratch), `encode_layer`, building-block encoders, rope tables |
| `src/kernels/attn_common.h` | shared Metal helpers: exact e4m3 / e2m1 / ue8m0 rounding, order-preserving float keys, rope pair ops, simdgroup fragment layout |
| `src/kernels/attn_kv.metal` | window KV write (rmsnorm+rope+fp8 sim), compressor (softmax pool, norm, index key `wk` matvec + k_norm, rope, fp4 sims), partial-group state, ring commit, standalone quant sims |
| `src/kernels/attn_index.metal` | indexer: decode scorers (tensor-op full scan, tensor-op candidate-list, simdgroup fallback), exact radix top-k (one threadgroup), histogram compaction top-k, prefill tiled scorer (tensor ops) + per-query running top-k merge, candidate-block union |
| `src/kernels/attn_sparse.metal` | sparse attention with sinks, flash-style simdgroup-matrix kernel (decode split-K + combine; prefill fallback) |
| `src/kernels/attn_sparse_pf.metal` | prefill sparse attention, two-pass on the tensor units (scores → softmax → P·V) |
| `src/kernels/attn_sparse_pf2.metal` | prefill sparse attention, folded-softmax two-pass (per-tile max/sum in the score pass, no f32 score matrix / softmax dispatch), software-pipelined over 64-query blocks (default prefill path) |
| `src/kernels/attn_probe.metal` | probes (matmul throughput, streaming bandwidth, fragment layout) — not used by the engine |
| `tests/ref/attn_ref.{hpp,cpp}` | CPU reference of every piece (the math spec) |
| `tests/test_attn_units.mm`, `test_attn_stack.mm`, `test_attn_200k.mm`, `test_attn_pf_paths.mm`, `tests/attn_test_util.hpp` | ctests |
| `bench/attn200k.mm` (`BENCH_AB=1`: interleaved base/new prefill table), `attn_pfsplit` (per-kernel prefill split, A/B of single switches, query-block sweep, candidate-drift model), `attn_coop_probe` (cooperative-tensor layouts), `attn_cache_probe` (read BW vs working set) (+ `attn_prof`, `attn_pfprof`, `attn_mmprobe`, `attn_layout_probe`, `attn_dispatch_probe`) | benchmarks |

## API

```cpp
#include "kernels/attn.hpp"
namespace A = ms::attn;
A::State st;  A::init(st, A::Options::from_env());   // allocates everything once
A::LayerIO io;  io.layer = L;  io.q = ...;  io.kv = ...;  ...   // see below
A::encode_layer(enc, st, io, start, n);   // positions start..start+n-1; n == 1 -> decode path
```

`encode_layer` encodes the full attention side of layer `L` into the caller's `ms::Enc` (no commit/wait; it inserts
the barriers it needs and ends with one). Call it for L = 0..39 in order, for every token chunk. Prefill chunks of any
size (1..max_chunk) can be mixed with decode steps; results are the same as one-shot prefill (see tests).

`LayerIO` (all f32, n = tokens in the call):

| field | shape | from | needed for |
|---|---|---|---|
| `q` | [n][64][512] | `wq_b(q_norm(wq_a x))`, **pre-rope** | all |
| `kv` | [n][512] | `wkv x`, pre-norm | all |
| `kv_norm` | [512] | `attn_kv_a_norm` | all |
| `sinks` | [64] | `attn_sinks` | all |
| `out` | [n][64][512] | **output**: attention output after inverse rope (before the fp8 sim + `wo_a`) | all |
| `c_kv`, `c_gate` | [n][512] | `attn_compressor_kv x`, `attn_compressor_gate x` (gate: ratio 2 only) | kv sources 2, 8, 14, 20 |
| `c_norm` | [512] | `attn_compressor_norm` | kv sources |
| `idx_wk` | bf16 [128][512] | `indexer.attn_k` raw GGUF bytes | kv sources |
| `idx_knorm` | [128] | `indexer.k_norm` | kv sources |
| `idx_q` | [n][32][128] | `indexer.attn_q_b(qr)`, pre-rope | index sources 2,8,14,20,24,28,32,36 |
| `idx_w` | [n][32] | `indexer.proj(x)`, **unscaled** (the 128^-0.5·32^-0.5 = 1/64 is applied here) | index sources |

Layer roles are hard-coded helpers (`ratio_of`, `kv_source_of`, `index_source_of`, `is_cand_source`, `uses_cand`).
The fp8 sim of the attention output before `wo_a` (model.py 787) is **not** applied here; use
`A::encode_quant_sim(e, out, n*64*512, 0)` (bit-exact `act_quant` block 32 ue8m0) if the integration wants it.

Building blocks (also exposed): `encode_kv_window`, `encode_compress`, `encode_index_decode`,
`encode_index_prefill`, `encode_sparse`, `encode_window_commit`, `encode_quant_sim(mode 0 fp8 / 1 fp4-ue8m0-32 /
2 fp4-e4m3-16)`, `make_rope_table`, `persistent_state(st, pos)` (buffers + byte counts to snapshot for
save/load; restoring those bytes restores the state bit-exactly), `reset`.

Options / env switches (every optimization can be turned off; the slow path is the tested fallback):

| option | env | default | effect |
|---|---|---|---|
| `actq` | `MACSEEK_ACTQ` | 0 (`from_env`; struct default 1) | fp8/fp4 quant-dequant simulations as in the reference (0 = plain f32→f16; engine default, closer to the llama.cpp oracle) |
| `max_pos` | `MACSEEK_MAX_POS` | 262144 | rope tables and cache capacity |
| `max_chunk` | — (engine: `MACSEEK_CHUNK`) | 2048 (engine sets 4096) | largest prefill chunk |
| `key_chunk` | `MACSEEK_ATTN_KCHUNK` | 16384 | prefill indexer key chunk (bounds the score scratch) |
| `dec_fast_index` | `MACSEEK_ATTN_DEC_FAST` | 1 | tensor-op decode scorer + histogram/compaction top-k (0: simdgroup scorer + single-TG radix select) |
| `pf_cand_union` | `MACSEEK_ATTN_PF_UNION` | 1 | prefill layers 24..36 score only the union of the candidate blocks of each 2-query tile (0: dense + mask) |
| `pf_mpp` | `MACSEEK_ATTN_PF_MPP` | 1 | prefill sparse attention two-pass on tensor units (0: flash simdgroup kernel) |
| `pf_fold_softmax` | `MACSEEK_ATTN_PF_FOLD` | 1 | two-pass with the softmax folded into both passes (`attn_sparse_pf2.metal`); 0: 3-kernel two-pass (scores f32 → softmax → P·V) |
| `pf2_qblock` | `MACSEEK_ATTN_PF2_QBLOCK` | 64 | folded path: queries per pipelined block (2 scratch sets of ≈ 150 KB/query: 19 MB at 64) |
| `pf_qblock` | `MACSEEK_ATTN_PF_QBLOCK` | 512 | 3-kernel path: queries per two-pass block (scratch ≈ 0.3 MB/query) |
| `pf_score_reg` | `MACSEEK_ATTN_PF_SCORE_REG` | 1 | prefill indexer scorers (dense + candidate union) sum the heads in registers (0: threadgroup-memory epilogue) |
| `pf_merge_filter` | `MACSEEK_ATTN_PF_MFILTER` | 1 | running top-512 merge compacts only chunk elements above the running minimum (exact) |
| `pf_idx_qblock` | `MACSEEK_ATTN_PF_IDX_QBLOCK` | 256 | candidate-union layers: score/merge chain per block of 256 queries, 2 blocks in flight (0: whole chunk) |
| `dec_rows_split` | `MACSEEK_ATTN_DEC_SPLIT` | 64 | decode split-K rows per threadgroup |
| `dec_heads_tg`, `pf_heads_tg` | `MACSEEK_ATTN_DEC_HEADS`, `_PF_HEADS` | 8 | heads per flash threadgroup (16 = 256 threads, measured 2-3x slower) |
| `dec_mpp` | `MACSEEK_ATTN_DEC_MPP` | 0 | decode through the two-pass path (measured slower: 55 vs 29 µs/layer) |

## State layouts (owned by this module; allocated by `init`)

| buffer | layout | notes |
|---|---|---|
| `rope_plain`, `rope_yarn` | float2 [max_pos][32] (cos, sin) | plain: θ=1e4, no YaRN (layers 0,1). yarn: θ=1.6e5, YaRN factor 16 / orig 65536 / β 32,1 (layers 2..39, incl. compressed rows at position `j*ratio`, index keys and indexer queries). angle = fp32(pos)·fp32(freq) like `torch.outer`, cos/sin in double then rounded. 64 MiB each at 262144 |
| `win[L]` | half [128][512], slot = pos % 128 | window KV ring per layer (post rmsnorm, rope, fp8 sim) |
| `ckv[s]` | half [cap_s + 128][512], row j = group j | compressed KV per source s = 0..3 (layers 2, 8, 14, 20); cap = max_pos/ratio (131072 / 262144) |
| `ikey[s]` | half [cap_s + 128][128] | index keys (k_norm(wk·latent), rope, fp4 sim) |
| `cst[s]` | f32 [2][ratio][512] | compressor partial group: kv slots then gate-score slots, slot = pos % ratio |

Only rows `< (pos+1)/ratio` are meaningful. At 200K context the state is ckv+ikey 640 MB + rings 5 MB (+128 MB rope
tables). All scratch (decode + prefill) ≈ 0.5 GB at max_chunk 4096, dominated by the indexer score chunk S [n][16384] f32
(268 MB); the prefill attention scratch is 19 MB (folded path; the 3-kernel path needs 158 MB).

**Storage dtype and accuracy.** Caches are F16. With the quant simulation on (reference behaviour), every stored
value is an e4m3 (window, ue8m0 scale/32) or e2m1 (compressed KV with e4m3 scale/16; index keys with ue8m0
scale/32) value times its block scale; these have ≤ 5 significant bits and are **exactly representable in F16** for
|x| ≥ 2^-20 (F16 subnormal spacing is 2^-24) and |x| ≤ 65504, which normalized KV rows satisfy — i.e. F16 storage is
lossless relative to the reference. With `actq=0` F16 keeps 11 mantissa bits (the reference itself runs in BF16, 8 bits).
Packed fp8/fp4 storage would be equally exact and 2-4x smaller (see next ideas).

## Semantics (normative details)

- Window row: `fp8sim(rope(w ⊙ x·rsqrt(mean x²+1e-20)))`, rope = interleaved pairs on dims 448..511.
- Compressor ratio 2: per channel softmax over the 2 gate scores of the group (fp32), weighted sum of the 2 kv
  values; ratio 1: passthrough. Group j completes at token j·r+r-1; earlier tokens of an incomplete group live in
  `cst` across decode steps and chunk boundaries. Then rmsnorm(c_norm) → latent. Index key = fp4sim_e8m0_32(rope(
  k_norm(wk·latent))); cache row = fp4sim_e4m3_16(rope(latent)), both at rope position j·ratio.
- Indexer: q = fp4sim_e8m0_32(rope(q_raw)) per head; w = w_raw/64; score[t] = Σ_h relu(q_h·k_t)·w_h for t <
  (pos+1)/ratio. Top-512 among visible (and candidate) rows, ties → lower index, output sorted ascending, -1 padded.
- Layer 20: candidate blocks = top-2048 of block maxima over blocks of 8 with the newest (partial) block pinned
  (+inf); layers 24..36 select only inside those blocks (their own q/w, layer-20 keys).
- Sparse attention: rows = causal window (≤128) + top-k compressed rows; logits = q·k·512^-0.5 (q roped with the
  layer's table); softmax with the per-head sink in the denominator only; out = Σ p·row (K == V); inverse rope on
  dims 448..511.
- Quant sims bit-exact to `kernel.py` given the same fp32 input (`act_quant` round_scale; `fp4_act_quant` both scale
  types; RNE casts; `precise::divide` for the e4m3-scale path). Not reproduced: the reference's intermediate BF16
  roundings of activations (we keep f32), torch's fp32 `pow` in the frequency table (1-ulp level).

## Tests (all pass; `ctest`)

| test | what |
|---|---|
| `test_attn_units` | quant sims bit-exact on 1M values (incl. zero blocks, exact ties, 2^-30..2^12 scales); e4m3/e2m1 spot values; top-k: n=0, <k, =k, ties, -inf holes, 262144 elements, k=2048; decode indexer at pos 40000 (full, candidate source, candidate-restricted list mode, len<512) == CPU top-k exactly; sparse decode vs CPU at pos 5/127/1000/6000 with 0/7/512 compressed rows (rel rms ≈ 1.5e-4) |
| `test_attn_stack` | layers {0,2,3,20,21,24}, 600 tokens from 0: one-shot prefill, chunked prefill+decode (odd boundaries 1,1,131,1,1,1,122,…), pure decode — all vs CPU token-by-token reference (rel rms ≤ 2.6e-4, top-k lists identical 252/252); prefill-only chunking **bit-exact** vs one-shot; at depth (random 40K-position caches, 3 key chunks, candidate masking active): chunk of 96 vs CPU ref, chunked (33,1,62) vs one-shot |
| `test_attn_200k` | decode at pos 200000 and 262143 (max) for layers {0,2,3,20,21,24} vs CPU ref: top-512 identical, out rel rms ≈ 1.7e-4; ACTQ=0 path 250-prefill + 50 decode vs ref |

| `test_attn_pf_paths` | threshold-filtered merge == full merge == CPU top-k (5 key chunks, -inf holes, survivor overflow, threshold ties); at depth 40K (3 key chunks, candidate union): all prefill fast paths vs all off and each switch alone — top-k lists identical, outputs within 3e-4 rel rms (folded softmax; everything else bit-identical); pipelined block size 16 / 64 / 160 and index query-blocking 0 / 34 / 64 bit-identical; shallow chunk from position 0 (skipped all-invalid tiles) |

Fallback paths (`MACSEEK_ATTN_PF_MPP=0`, `MACSEEK_ATTN_DEC_FAST=0`, `MACSEEK_ATTN_PF_UNION=0`) pass the same tests. The register
epilogues assume a cooperative-tensor layout of `matmul2d` (probed with `bench_attn_coop_probe`); `init` verifies it with
`attn_idx_layout_check` / `attn_pf2_layout_check` and falls back to the threadgroup-memory kernels (stderr note) if it ever changes.
Accuracy vs the CPU reference (`test_attn_stack`, out rel rms one-shot from 0 / at depth): folded path 1.30e-4 / 1.78e-4,
3-kernel path 2.06e-4 / 2.00e-4 (the folded P = exp(S − m_tile) ≤ 1 keeps more half precision than the normalized P).

## Benchmarks (M5 Pro, 20-core GPU) — `build/bench_attn200k`

Decode numbers were measured with a `llama-bench` (model owner) running concurrently on the GPU (pessimistic);
timings are min over command buffers. Decode is timed **sustained** (20 tokens per command buffer): short command
buffers run at low GPU clocks (x10 layers: 132 µs/layer vs 29.6 µs/layer sustained) and are not representative.
Streaming read bandwidth (own probe, 1 GiB): **281-289 GB/s**. Dependent dispatch + barrier ≈ 4 µs.

Decode, 1 token at position 200000, all 40 layers (synthetic random caches):

| component | ms | MB moved | % of BW |
|---|---|---|---|
| kv window + compressor (40 layers) | 0.088 | 0.7 | — (latency) |
| indexer full scans 2, 8, 14 (100K keys) + 20 (200K keys + candidate blocks) | 0.90 | 132 | 51% |
| — layer 2 alone (100K) | 0.165 | 26 | 56% |
| — layer 20 alone (200K) | 0.36 | 53 | 51% |
| indexer candidate-restricted 24, 28, 32, 36 (16K keys each) | 0.23 | 17 | 26% |
| sparse attention, 40 layers (split-K + combine) | 1.17 | 140 | 42% |
| **total, one command buffer** | **2.51 ms/token** | 291 | 40% |

Prefill at depth 200000, all 40 layers, synthetic random state (`BENCH_AB=1 BENCH_CHUNK=n build/bench_attn200k`:
base = all prefill switches of this section off, new = defaults, toggled at run time on the same state and interleaved,
min over 3 rounds; idle GPU). TFLOP/s for the new path in brackets.

| component | n=2048 base | n=2048 new | n=4096 base | n=4096 new | speedup |
|---|---|---|---|---|---|
| kv window + compressor + ring commit | 0.98 | 0.98 | 1.82 | 1.85 | — |
| indexer layer 2 (= 8, 14; dense ~100K keys) | 118.4 | 58.1 | 242.3 | 117.7 (29 TF) | 2.06x |
| indexer layer 20 (dense ~200K keys + candidate blocks) | 242.1 | 121.5 | 497.1 | 244.2 (28 TF) | 2.04x |
| indexer layer 24 (= 28, 32, 36; candidate union, random = worst case) | 71.0 | 25.2 | 149.7 | 50.3 | 2.98x |
| indexer, all 8 index layers | 881 | 396 | 1823 | 799 | 2.28x |
| sparse attention, 40 layers | 726 | 515 | 1434 | 1024 (12.9 TF) | 1.40x |
| **total, one command buffer** | **1590** | **905** | **3238** | **1800** | **1.80x** |
| attention-side tok/s | 1288 | 2263 | 1265 | 2275 | |

Realistic overlap (neighbouring tokens pick mostly the same keys / blocks). Random state is the worst case for the candidate
union and for the row reuse of sparse attention. Two models, both in the benches: `BENCH_QCORR_PCT=95` (AR(1)-correlated
indexer inputs: 43% top-512 overlap of consecutive queries, union 3021 blocks per 2-query tile) gives n=4096 total
2967 → 1630 ms (1.82x; layer 24 117 → 41 ms, sparse attention 1284 → 797 ms). `BENCH_CAND_DRIFT=d` (bench_attn_pfsplit;
each query's 2048 candidate blocks = the previous query's with d replaced) gives layer 24 at n=4096: d=0 78 → 26 ms,
d=20 80 → 28 ms, d=100 84 → 29 ms, d≈all 126 → 43 ms.

What changed (each switchable, see Options; the old path is the fallback):

1. **Indexer score epilogue in registers** (`pf_score_reg`). The cooperative destination of `matmul2d(64, 128, 128)` on 4
   simdgroups has a fixed layout (probed: element i of lane l in simdgroup sg is row R + 8·bit2(i) + 32·bit5(i), col C +
   (i&3) + 32·bits3-4(i)); each thread adds its 2 heads in registers, a reduce-scatter over lane bits 1, 2, 4 sums 16 heads
   with 28 shuffles, and 1 KB of threadgroup memory adds the two head halves. The old epilogue wrote the whole 32 KB tile to
   threadgroup memory (1 threadgroup per core). Dense scorer 14 → 29-32 TFLOP/s; the candidate-union scorer (64x64 tiles)
   uses the same scheme.
2. **Threshold-filtered running top-k merge** (`pf_merge_filter`). Once a query's running list holds 512 entries, only chunk
   elements strictly above its minimum can enter (ties lose: they have higher indices). The chunk is compacted in index
   order into threadgroup memory (≤ 2048 survivors, else the full merge runs) and the exact select runs on 512 + survivors.
   Merge per 16K-key chunk 2.3 → 1.6 ms (n=4096), i.e. about one read of S.
3. **Folded-softmax two-pass attention** (`pf_fold_softmax`). Pass 1 (`attn_pf2_scores`, one 64-row tile per threadgroup)
   computes the tile's per-head max m_t and Σexp(S − m_t) in registers and writes P_t = exp(S − m_t) as half (≤ 1) plus
   (m_t, l_t); pass 2 (`attn_pf2_pv`) derives M = max(sink, m_t), L and α_t = exp(m_t − M)/L per head and accumulates
   α_t·(P_t·V_t). No f32 score matrix (671 MB per layer at n=4096), no softmax dispatch, tiles whose 64 rows are all invalid
   are skipped in both passes, gathers double-buffered. More accurate than before (P keeps more half precision).
4. **Software-pipelined small query blocks** (`pf2_qblock` = 64). Stage j dispatches qprep(j), scores(j−1), P·V(j−2) with one
   barrier per stage (two scratch sets). A 64-query block's gathered rows (40 MB), P and Qh are still in the on-chip cache
   (`bench_attn_cache_probe`: ~0.9-1.1 TB/s up to 32 MB working sets) when pass 2 re-reads them: sparse layer 30.3 →
   26.8 ms (random) / 24.9 → 20.4 ms (QCORR 95) vs 512-query blocks; scratch 158 MB → 19 MB.
5. **Query-blocked candidate-union chain** (`pf_idx_qblock` = 256): score + merge per 256-query block, two blocks in flight,
   so S rows are re-read from cache (layer 24: 55 → 50 ms). No gain on the dense layers (compute-bound), so not used there.

Measured and dropped (slower or no gain; numbers at n=4096): P·V over 128-dim slices on 4 simdgroups (64+64 accumulator
registers: 37 vs 32 ms per layer), on 8 simdgroups (`execution_simdgroups<8>`, single-buffered V: slower), α-scaled P
staged in threadgroup memory (no gain, 2x the half rounding of P: rel rms vs CPU 3.4e-4), f32 q read directly by the score
pass as an f32×f16 tensor op (slower, and its destination layout differs), window tiles read straight from device memory
(contiguous chunk rows; ±1%), 2 row chunks in flight per thread in pass 1 (±1%), larger key chunks for the dense indexer
(KC 16384 best), query-blocking the dense indexer chain (+1.5%), `pf_qblock` 1024 for the 3-kernel path (−1.5%).

History: decode 9.8 → 2.5 ms/token (indexer tensor ops + compaction top-k, 8-head TGs); prefill n=2048 3.2 s → 1.7 s
(candidate union, two-pass attention) → 0.9 s (this round). Matrix-unit probes: simdgroup_matrix 7.4 TFLOP/s, `matmul2d`
64x64 tiles 22 TFLOP/s measured next to a running llama-bench (the register-epilogue indexer reaches 29-32 on an idle GPU).

## Limits / caveats

- Decode attention is latency-bound (~29 µs/layer for 640 rows: 2 dispatches + 4 serial 16-row chunks per TG).
- The register epilogues depend on the (undocumented) cooperative-tensor layout of `matmul2d`; checked at init, fallback
  to the threadgroup-memory kernels if it changes.
- Sparse attention prefill (≈ 26 ms per 640-row layer at n=4096, 13 TFLOP/s) is limited by the row gathers through
  threadgroup memory: with all rows cache-resident (identical top-k lists) it runs at ~17 ms. The rows are still read
  twice (scores, P·V); a single pass needs the 64x512 f32 output per query on chip (128 KB), which does not fit (32 KB
  threadgroup memory; input cooperative tensors require single-simdgroup scope).
- Candidate-union cost depends on how much adjacent queries' candidate sets overlap (random test data is the worst case;
  see the drift numbers above).
- Exact-tie handling: if more elements tie at the k-th score than are needed, the fast decode path falls back to the
  exact single-TG select (deterministic lowest-index rule); prefill merges use the same rule per chunk.
- `encode_layer` uses shared scratch (topk, candidates, decode scores): layers must be encoded in order, one sequence
  per `State`.
- 16-head flash threadgroups and the decode two-pass path are slower (kept as options, not defaults).

## Next ideas (ranked)

1. Packed index keys (fp4 e2m1 + ue8m0/32 = 68 B instead of 256 B per key) — exact only with `actq=1`; with the engine's
   `actq=0` the keys are plain f16, so this needs an accuracy gate first. Tensor ops accept `metal_fp4_e2m1_format`.
2. Candidate union with 4-query tiles when the measured union is small (real text: union ≈ 2048 + 3·drift blocks), else
   2-query tiles — decided per tile on the GPU from `ucount`.
3. Fuse the dense indexer's threshold filter into the scorer (write only survivors, not the S matrix: ≈ 2 ms per 16K-key
   chunk, ~60 ms per 4096 chunk) and filter the layer-20 block merge (k = 2048) too (~6 ms).
4. Decode attention: fuse the split-K combine (last-threadgroup pattern) and keep Q prepped once per layer
   (≈ -0.3 ms/token); packed fp8 window rows halve row bytes.
5. Measure the real top-k / candidate overlap on model activations (the benches model it with `BENCH_QCORR_PCT` and
   `BENCH_CAND_DRIFT`) and retune `pf2_qblock` / `pf_idx_qblock` on it.
