# Decode GPU time: dispatch / barrier overhead and fusions (exp/dec-gpu)

Goal: cut the GPU time of one decode token (37.6 ms measured on the model at 200K context, all expert hits, no
handshakes; bandwidth floor ~29.5 ms for the ~8.4 GB read per token).

## What exists

| file | content |
|---|---|
| `src/model/dec_layer.hpp/.mm` | the decode (n = 1) per-layer encoder: `encode_dec_prologue` (engram rows + engram_kv), `encode_dec_attn_side` (hc boundary, projections, attention, wo_a/wo_b, hc boundary, router + prediction), `encode_dec_moe_table` (table MoE + rdy/go handshake, DEC_SYNC modes). `W`/`LayerW` moved here from ds41.mm. All switches in `DecOptions`. |
| `src/model/ds41.mm` | the table decode path calls the module (`MACSEEK_DEC_LAYER=1`, default). `MACSEEK_DEC_LAYER=0` = the previous inline lambda (kept unchanged). With all switches off the module emits the same dispatch sequence as the lambda. |
| `bench/decode_layer.mm` | `bench_decode_layer`: synthetic weights with the real per-layer shapes and types of all 40 layers (`tests/dec_synth.hpp`: Q6_K/Q5_K attention, Q6_K/Q8_0 shared experts, IQ1_S/IQ2_XXS routed experts per layer, BF16 compressor/indexer, F32 router/hc), attention state at position 200000, embed + engram prologue + 40 x (attn side + table MoE incl. the rdy/go events, go pre-signaled) + head, 20 tokens per command buffer, configs interleaved, best/median. `--breakdown` (time with each component removed), `--check` (bit-compare h/x/sub/logits/KV state against config 0), `--cpu US` (CPU thread answers each rdy like the model), `--layers N`. 1.65 GB of synthetic buffers. |
| `tests/test_dec_fuse.mm` | ctest: (1) `encode_matvec_rmsnorm` == rmsnorm -> matvec bit for bit (Q6_K, Q5_K, BF16, Q8_0, F32); (2) layers 1, 2, 5, 20, 24, 31 and chains 0-2, 12-14: every switch and combination vs the unfused sequence, bit-exact on h, x, sub, q, attention output, router ids/weights and the KV / compressed / index-key state; (3) run-to-run determinism. |
| `src/core/metal.*` | `ms::enc_stats()`: process-wide dispatch / barrier / stage / encoder counters. |

## Baseline (bench, M5 Pro, `taskpolicy -b`, best of N; other GPU jobs were running most of the time,
so single deltas below ~0.3 ms are noise)

`bench_decode_layer base`: **37.4-37.6 ms/token** (matches the 37.6 ms measured on the model), **784 dispatches, 595
barriers (all after >= 1 dispatch), 80 compute encoders per token** (2 encoder splits per layer for the handshake).

`--breakdown` (time with the component NOT encoded; delta = what it costs in the token):

| component | delta ms/token | floor (bytes at 285 GB/s) | note |
|---|---|---|---|
| routed MoE (6 experts x 40) | 8.98 | 6.3 | IQ1_S/IQ2_XXS decode ALU-bound (~70% of BW in situ) |
| attn_q_b | 5.38 | 4.8 | |
| shared expert | 5.20 | 4.5 | |
| attn_output_b | 5.03 | 4.8 | |
| attn_output_a | 4.11 | 3.9 | |
| sparse attention + combine | 2.06 | ~0.1 | ~51 us/layer, latency-bound (80 threadgroups) |
| all barriers removed (wrong output) | 1.94 | | upper bound of the whole dependency overhead |
| q_a + kv (+ compressor, indexer proj) | 1.71 | 1.2 | |
| indexer (8 layers) | 1.63 | 0.5 | ~200 us per index layer |
| router | 1.62 | 1.1 | |
| hc step into the FFN / into attention | 0.98 / 0.71 | 0.28 / 0.28 | |
| engram_kv (2 x 129 MB) | 0.85 | 0.9 | |
| handshake events + encoder splits | 0.72 | - | ~11 us GPU-side per rdy signal |
| q_a RMSNorm / kv window / fill(sub) | 0.58 / 0.52 / 0.34 | ~0 | each a stage of its own |
| next-layer prediction matvec | 0.47 | 0.3 | |

Probe (`probe=N`: N extra tiny dispatches per layer in an existing stage): +320 dispatches/token cost **0.00 ms**.
Concurrent dispatches are free; the dispatch count is not the overhead. Removing every barrier gains only 1.9 ms. Most of
the 8 ms above the floor is kernel efficiency: routed IQ MoE (+2.7), latency-bound attention (+3), hc steps (+1.1),
small matvecs/router (+1).

## Changes (all behind switches, all BIT-IDENTICAL, verified by test_dec_fuse and `bench --check` over 20 tokens x 40 layers)

| switch (env) | `MACSEEK_DEC_FUSE=1` sets | what |
|---|---|---|
| `MACSEEK_DEC_REORDER` | 1 | kv window write + compressor run in the q_a-norm stage (they only need the projections), fill(sub) in the router stage: -2 stages/layer |
| `MACSEEK_DEC_COMBINE` | 1 | sparse attention split-K combine done by the last threadgroup of each head group (`attn_sparse_lc_h8/h16`, device counter, combine arithmetic identical, heads interleaved for ILP): -1 dispatch, -1 stage/layer |
| `MACSEEK_DEC_SGU_ROUTER` | 1 | shared-expert gate/up next to the router (needs only x); the DEC_SYNC gap keeps the shared down projection. `=2`: the shared down also moves after the go wait, next to the routed gate/up (no gap work: fewest stages, but exposes the handshake on the model) |
| `MACSEEK_DEC_ENGRAM_BG` | 2 | layer 14's engram_kv (129 MB) split into chunks encoded right after the latency-critical dispatch of the indexer / sparse-attention (1) and hc (2) stages of layers 0..13; every chunk uses the full matrix's launch shape (`q::matvec_cfg`) |
| `MACSEEK_DEC_HC_SPLIT` | 2 | fused hc step with 2 threadgroups per dim chunk, each computing half of the 24 mix rows (same per-row arithmetic); h ping-pongs between `P.h` row 0 and row 1 (`dec_cur_h`) because split 1 still reads h_in while split 0 writes h_out |
| `MACSEEK_DEC_NORMQB` | 0 | q_a_norm RMSNorm fused into the x loads of attn_q_b / indexer q_b (`q::encode_matvec_rmsnorm`, replays arch_rmsnorm's reduction exactly; `MACSEEK_MV_RN_TPT` tiles per threadgroup). Bit-identical but SLOWER (+0.2..0.8 ms): every q_b threadgroup pays the norm prologue. Not in FUSE. |
| `MACSEEK_DEC_MOE_{GU,D}_{NR,RSG}` | - | table-MoE launch shapes (bit-identical for any value); sweep: the module defaults (gu 2x4, down 4x2) are best within noise |
| `MACSEEK_DEC_LAYER=0` | - | previous inline decode encoding (fallback) |

`MACSEEK_DEC_FUSE=1`: **789 dispatches (engram chunks +46, combine -40), 473 barriers per token (-122)**.

### Results (bench, interleaved, 15 reps, quietest run)

| config | best ms/token | median | barriers |
|---|---|---|---|
| base (current path) | 37.39 | 37.84 | 595 |
| reorder + combine + sgu_router 1 | 37.28 | 37.68 | 475 |
| **FUSE=1** (reorder, combine, sgu_router 1, engram_bg 2, hc_split 2) | **37.24** (-0.4%) | 37.66 (-0.5%) | 473 |
| FUSE=1 + sgu_router 2 (no gap work) | 37.10 (-0.8%) | 37.36 (-1.3%) | 473 |

The fusions remove 20% of the barriers but only ~0.2-0.5 ms: the per-token overhead is not dispatch / barrier bound
(see the probe and the no-barrier bound above). sgu_router 2 wins on GPU time but leaves no work between rdy and go;
on the model the handshake is then exposed (dec_sync.md: ~45-60 us per layer without gap work vs ~7-9 us with 50-100 us
of gap work), so FUSE uses 1. A/B on the model: `MACSEEK_DEC_FUSE=1` (greedy hash must be identical to FUSE=0), then
`MACSEEK_DEC_SGU_ROUTER=2`.

## Bug fixed: arch_hc_step in-place race (affects every decode path)

Decode calls the fused hc step with `h_out == h_in`. Every simdgroup of a threadgroup reads all of h_in for its mix rows
while simdgroup 0 writes h_out over the same addresses in the same loop: a slower simdgroup can read an already
updated element. The bench showed run-to-run different h / logits (max rel ~1e-3, and up to 1e2 after 40 layers) in
`--check` with 1 layer; `MACSEEK_ARCH_HC=rows` (separate kernels) was deterministic. Fix (default on,
`MACSEEK_ARCH_HC_SAFE=0` = old behaviour): simdgroup 0 writes h_out after a threadgroup barrier, recomputing h_new with
the same arithmetic. Outputs equal the race-free result of the old kernel; if the old path ever raced on the model, its
greedy outputs could differ from the fixed path in those tokens. test_arch_hc passes.

Also: with a pre-signaled go (bench/test), DEC_SYNC event mode (0) raced (no fences; on the model the CPU orders it).
The module now uses the fence pair for every mode (no cost).

## Negative results

- Background bandwidth work next to latency-bound stages does not hide: a cache warm-up read of N MB of attn_output_a
  next to the sparse attention cost its full DRAM time (+4.3 ms for 27 MB x 40) and gave o_a nothing; engram_kv chunks
  (engram_bg) are within noise.
- normqb (above); hc row splits 3 (+0.3-0.5 ms) and 2 (noise).
- MoE decode launch shapes: no better shape than the defaults (gu nr 1/2/4 x rsg 2/4/8, down nr 2/4 x rsg 1/2/4/8).

## Where the remaining time is (next steps, outside dispatch overhead)

1. Routed IQ1_S/IQ2_XXS MoE (+2.7 ms over the floor): ALU-bound decode; a faster block decoder changes per-lane
   summation order (not bit-identical).
2. Sparse attention ~51 us/layer and the indexer ~200 us per index layer (+3 ms): latency-bound; double-buffering the
   16-row gathers in attn_sparse (same arithmetic) or more splits (different numerics).
3. hc steps (~21 us each incl. stage, 80/token): the single-threadgroup tail (Sinkhorn + x normalization).
4. Handshake signals (0.72 ms): inherent to the per-layer router round trip.


## GSQ-RCO Coder at 262K (branch coder-dec)

`bench_decode_layer` defaults to `--layout coder` (tests/dec_synth.hpp `dsyn::layout()`): the per-layer types of
docs/tensors_coder.txt (routed gate=up / down per layer from IQ1_M / IQ2_XXS / IQ2_XS, 192 experts, F32 router
[192][5120], Q5_K shared / head / engram_kv / q_b / o_a / o_b, Q8_0 q_a / kv, Q5_K embedding), attention state at
position 260000 (max_pos 262144). `--layout ud` = the previous UD-IQ1_S layout (384 experts, pos 200000).
`--breakdown` also removes the output head (`head`) and prints MB/token and the floor at `--bw` (285 GB/s) per
component. test_dec_fuse also runs every fusion switch on Coder layers 1, 8, 23 and 12-14 (bit-identical).

Synthetic token (20 tokens/CB, go pre-signaled, under the model lock, best of 5-7):
**before 35.99 ms (median 36.02) -> after 34.71 ms (median 34.73), -3.6%**. Interleaved A/B (3 rounds, reps 7): before
35.81-36.34 best, after 34.53-34.76 (FUSE=1 config: 34.17-34.30). Bytes/token 8.03 GB -> floor 28.2 ms. The model
measured ~36.5 ms of GPU work per token before, matching the synthetic 36.0 ms.

| component | MB/token | floor ms | before delta ms | after delta ms |
|---|---|---|---|---|
| hcA | 85.4 | 0.300 | 0.683 | 0.746 |
| proj | 429.3 | 1.506 | 1.556 | 1.598 |
| norm | 0.0 | 0.000 | 0.141 | 0.140 |
| qb | 1237.3 | 4.341 | 4.688 | 4.500 |
| kvwin | 0.5 | 0.002 | 0.089 | 0.154 |
| index | 183.2 | 0.643 | 1.460 | 1.468 |
| sparse | 25.2 | 0.088 | 1.603 | 1.664 |
| oa | 922.7 | 3.238 | 3.669 | 3.272 |
| ob | 1153.4 | 4.047 | 4.169 | 4.108 |
| hcF | 85.2 | 0.299 | 0.667 | 0.711 |
| router | 157.3 | 0.552 | 0.688 | 0.709 |
| pred | 40.7 | 0.143 | 0.102 | 0.156 |
| fill | 0.0 | 0.000 | 0.019 | 0.062 |
| moe_shared | 973.2 | 3.415 | 4.022 | 3.974 |
| moe_routed | 2063.6 | 7.241 | 9.477 | 9.022 |
| barriers | 0.0 | 0.000 | 1.564 | 1.610 |
| events | 0.0 | 0.000 | 0.346 | 0.321 |
| engram | 216.3 | 0.759 | 0.714 | 0.757 |
| head | 456.1 | 1.600 | 1.571 | 1.532 |

(delta = base best minus best without the component; noise ~0.1 ms.)

Changes (each with an env switch to the old path):

| change | env (default) | effect | numerics |
|---|---|---|---|
| Q5_K decode-matvec launch rule: 1 row per simdgroup, K split to ~2 block iterations per simdgroup, 1-simdgroup tiles; short rows (q_b, 5 blocks): 4 rows x 2 simdgroups, no split | `MACSEEK_MV_CFG` (1; 0 = old rule) | q_b 87->95%, o_a (g8) 85->95%, o_b 93->97%, head 95.5->99.3%, engram_kv 96.6->99.0%, shexp gu 78->89%, down 84->87% of the in-process BW (302 GB/s; `bench_qmatvec --rules`) | K-split partial sums grouped differently (q_b: no split now) |
| IQ1_M dot: grid + delta folded into one 64 KB half table (`q_iq1m_gridd`, index = qs, full qh nibble << 8) | `MACSEEK_MOE_DEC_V` (1; 0 = old decoders in the MoE table kernels) | routed table part M/M 202.6 -> 193.3 us per layer | sum (g + delta) x instead of sum g x + delta sum x: same weights, different rounding |
| IQ2_XXS routed gate/up: 8 lanes per block; 4 rows x 4 simdgroups | `MACSEEK_MOE_DEC_V` (1); shape: `MACSEEK_MOE_GU_NR/RSG` | X/X 250.6 -> 237.5 us, X/M 241.5 -> 233.1 | per-lane grouping (8 vs 16 lanes); the shape is bit-identical |
| MoE down table kernel walks (expert, block) incrementally (no integer division per iteration) | - | included above | bit-identical |
| decode sparse attention next-chunk register prefetch | `MACSEEK_ATTN_DEC_PF` (0) | no gain: not default | bit-identical |

Tests: full ctest 24/24 pass; `bench_decode_layer --check base base fuse base` (Coder) and `--layout ud --check base
base` bit-identical run to run; test_quant_matvec IQ1_M max err <= 1.5e-8; test_quant_moe Q5_K-shared combos y err
<= 3.6e-7 (All == bound, SharedGu+AfterSharedGu bit-identical as before).

Measured and rejected (Coder, 262K):
- Q5_K decoder variants (8 lanes with 16-byte loads + x sums precomputed per block, 32 lanes, 2-block unrolled loop):
  with the tuned launch shape none beats the original 16-lane decoder (per-variant sweeps 93-95% vs 95%); an 8-lane
  Q5_K shared expert inside the MoE kernels: -1%.
- MoE decode launch shapes (`bench_qmatvec --moe-sweep gu,down,shared[,shared]`): flat within 1-2% except IQ2_XXS
  gate/up (taken). The shared-expert part runs at 84-88% for any shape.
- Routed IQ decode is bound by the table loads more than by FMAs: a probe without decode math (loads only) runs at
  91% of BW vs 77%; a lane-uniform grid index (no divergence) gains only 2%; a float IQ2_XXS grid (two table loads per
  8 weights instead of one) is 2x slower. Next lever: fewer table loads per weight (IQ2_XXS sign masks).
- Sparse attention split size (`MACSEEK_ATTN_DEC_SPLIT` 32/16: +0.9/+2 ms; 128: same; 224/320: +0.5/+1.2 ms), 16
  heads per threadgroup (+2.6 ms), hc step threadgroup 256/128 and row splits 2/4 (noise or slower). Sparse attention
  (~40 us/layer) is throughput-limited at 80 threadgroups of ~19 KB threadgroup memory: with 16 threadgroups
  (layers 0/1) it takes 14.5 us.
- Indexer at 262K: the tensor-op scorer streams keys at ~60% of BW; its threadgroup memory is full (Q 8 KB + head-sum
  tile 8 KB + 4096-bin histogram 16 KB) so the tile cannot be double-buffered; summing heads from the cooperative tensor
  (as the prefill scorer does) would change score rounding and so possibly the top-512 at ties.

Remaining gap to the floor (after): routed MoE +1.8 ms, sparse attention +1.6, barriers 1.6, indexer +0.8, hc steps
+0.85, shared expert +0.55.
