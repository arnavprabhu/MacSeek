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
