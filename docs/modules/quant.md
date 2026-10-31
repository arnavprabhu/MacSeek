# Module: quant — quantized matrix kernels (Metal)

Branch `mod/quant`. Files:

| file | content |
|---|---|
| `src/kernels/quant_types.hpp` | `ms::q::QType` (= ggml type ids), block geometry helpers (plain C++) |
| `src/kernels/quant.hpp`, `quant.mm` | host encode functions (`ms::q::encode_*`), launch heuristics, env overrides |
| `src/kernels/quant_grids.h` | GENERATED (`tools/gen_quant_grids.py <ggml-common.h> out`): IQ1_S / IQ2_XXS grid VALUES + derived GPU tables |
| `src/kernels/quant_common.h` | constant tables, unaligned-safe loads, `QMvArgs` |
| `src/kernels/quant_blocks.h` | per-type block decoders `QB_*` (one static interface, all kernels are templates over it) |
| `src/kernels/quant_tile.h` | tensor-API includes + `q_tile_deq` (dequantize a TN x 128 weight tile to half in threadgroup memory) |
| `src/kernels/quant_mv.metal` | `q_mv_*` decode matvec, `q_mvn_*` small-n matvec |
| `src/kernels/quant_moe.metal` | `q_moe_gu_*`, `q_moe_down_*` fused MoE decode |
| `src/kernels/quant_mm.metal` | `q_mm_*` prefill matmul (fused dequant tiles), `q_mm_{f16,bf16,f32}_64x128` dense, `q_deq_f16_*`, `q_f32_to_f16`, `q_gemm_f16_*` peak probe |
| `src/kernels/quant_moe_prefill.metal` | `q_moe_gather`, `q_moe_mm_gu_*`, `q_moe_mm_down_*`, `q_moe_reduce` |
| `src/kernels/quant_bench.metal` | `q_bw_*` bandwidth probes |
| `tests/ref/dequant.{hpp,cpp}` | CPU reference dequant for F32 F16 BF16 Q8_0 Q5_K Q6_K IQ1_S IQ2_XXS IQ4_NL MXFP4, `matvec_ref`, `random_blocks` |
| `tests/test_quant_{ref,matvec,matvec_n,matmul,moe,moe_prefill}.mm`, `tests/quant_test_util.hpp` | ctests |
| `bench/membw.mm`, `bench/qmatvec.mm`, `bench/qmatmul.mm` | microbenchmarks |
| `docs/dspark_tensors.txt` | tensor table of the dspark draft GGUF (same format as docs/tensors.txt; MXFP4 experts) |

No changes to `CMakeLists.txt` or `src/core`.

## API (namespace `ms::q`, all functions only encode; never commit/wait)

Weights are raw GGUF block rows: a `[in -> out]` tensor is `rows = out` rows of `cols = in` elements. Buffers are
`ms::Buffer` (buf + byte offset). Weight buffer offsets must be >= 4-byte aligned (GGUF tensor/expert offsets are);
activation buffers 16-byte aligned. `cols` must be a multiple of the block size (256 for K/IQ/dense types, 32 for Q8_0;
128 for the matmul paths). All real model shapes satisfy this.

```cpp
// decode n=1, F32 math. groups>1: block-diagonal (attn_output_a: groups=8, cols=4096, rows=8192, x has 32768)
void encode_matvec(Enc&, QType, W, rows, cols, x, y, bool accumulate=false, int groups=1, MvCfg cfg={});
// n = 1..16 columns X[n x cols(*groups)] -> Y[n x rows]; x staged as half; n >= 8 (groups==1) -> tensor-API tiles
void encode_matvec_n(Enc&, QType, W, rows, cols, X, Y, int n, bool accumulate=false, int groups=1);
// prefill: Y[n x rows] (+)= X[n x cols] . W^T. Converts X to half in a module-owned scratch (barrier before+after)
void encode_matmul(Enc&, QType, W, rows, cols, X, Y, int n, bool accumulate=false);
void encode_f32_to_f16(Enc&, X, X16, size_t count);              // convert once ...
void encode_matmul_h(Enc&, QType, W, rows, cols, X16, Y, int n, bool accumulate=false, int x_stride=0);  // ... reuse
// MoE decode: MoeDecodeArgs{t_gate_up, t_down, n_routed<=6, slot[6]{gate,up,down}, weights (device F32[6]),
//   has_shared, t_shared, shared{gate,up,down}, x, y (accumulated), scratch >= moe_decode_scratch_bytes()}
void encode_moe_decode(Enc&, const MoeDecodeArgs&);
// MoE prefill: MoePrefillArgs{types, n_tokens, n_exp, slots[n_exp] (host), offs[n_exp+1] (host), tok/wt (device,
//   expert-sorted), slot_of[n*k] (device), shared, X, Y (accumulated), scratch >= moe_prefill_scratch_bytes(total,n,n_exp)}
void encode_moe_prefill(Enc&, const MoePrefillArgs&);
```

Supported types: matvec / matvec_n / matmul: F32, F16, BF16, Q8_0, Q5_K, Q6_K, IQ1_S, IQ2_XXS. MoE: routed gate/up and
down IQ1_S or IQ2_XXS (any combination), shared Q6_K or Q8_0. IQ4_NL / MXFP4: CPU reference only (engram rows are 144 B
lookups; MXFP4 is only in the dspark draft).

Barriers: the encoder is concurrent (`MTLDispatchTypeConcurrent`). Functions insert the barriers they need internally
(MoE between phases; `encode_matmul` around its shared X16 scratch; the n >= 512 pre-dequant path around its shared W16
scratch). Callers must still `e.barrier()` between dependent ops of their own (e.g. matvec output -> next matvec input).

Env overrides (all optional, for tuning; defaults = measured best):
`MACSEEK_MV_NR/RSG/KS` (matvec launch), `MACSEEK_MVN_NR/SG`, `MACSEEK_MVN_MM_MIN` (n at which matvec_n uses tiles,
default 8, 0 = never), `MACSEEK_MM_TILE` (e.g. `128x64_b2`), `MACSEEK_MM_PREDEQ_N` (default 512, 0 = never),
`MACSEEK_MOE_{GU,D}_{NR,KS,RSG}`, `MACSEEK_MOEP_TM` (64|128).

## Kernel design / layouts

All kernels read the raw GGUF blocks directly (no repacking), so expert slots streamed from SSD are usable as-is.
GGUF blocks are only 2-byte aligned (34/50/66/176/210 B), so decoders use 16-bit loads (`q_ld16/q_ld32`).

Block decoders (`quant_blocks.h`): each type is a struct with `QK, BB, LPB (lanes per block), EPL (elements per lane),
X4, pos(), load_x(), dot(), deq()`:

| type | block | LPB x EPL | lane slice |
|---|---|---|---|
| F32/F16/BF16 | virtual 256 | 32 x 8 | two coalesced 4-vectors (elements 4s.., 128+4s..) |
| Q8_0 | 32 / 34 B | 4 x 8 | 8 consecutive int8 |
| Q6_K | 256 / 210 B | 16 x 16 | the 4 elements sharing one ql/qh byte column x 4 bytes: 128h + 32k + l0 + j |
| Q5_K | 256 / 176 B | 16 x 16 | 8 low-nibble + 8 high-nibble elements of one 64-chunk |
| IQ1_S | 256 / 50 B | 16 x 16 | 2 grid groups (16 contiguous elements) |
| IQ2_XXS | 256 / 66 B | 16 x 16 | 2 grid groups |

IQ grids: the GPU uses derived **half-valued** tables (IQ1_S: 2048 x 8 halfs = 32 KB constant; IQ2_XXS: 256 x 8 halfs
plus a 128 x 8 half sign-bit mask table, signs applied with one XOR per 2 elements). One 16-byte constant load gives 8
weights and half->float conversion is free; this beat int8 grids + char4->float conversion by ~30% (IQ1_S 59% -> 78%
of BW on a 33 MB matrix) and is exact (grid values are small integers). IQ1_S uses
`dot = dl * (sum(g*x) + delta * sum(x))` with sum(x) precomputed per x slice.

matvec (`q_mv<Q, NR>`): threadgroup = RSG row-simdgroups x KS K-split simdgroups; each simdgroup owns NR rows and keeps
its x slice in registers across them; 32/LPB block slots per simdgroup iteration; K-split partials reduced through
threadgroup memory. Heuristic (sweep-derived): K/IQ/dense types split K until each simdgroup runs ~1-2 block
iterations (memory-level parallelism beats x reuse; 256-thread TGs); IQ types are ALU-bound, so NR=4, KS<=2.

matvec_n (`q_mvn<Q, NR, NT>`): x chunk (all n columns) staged in threadgroup memory as half; weights dequantized once
per lane and dotted with every column. For n >= 8 the n FMAs/weight exceed the weight stream time and the tensor-API
tile path (32x64 tiles) is used instead.

MoE decode: `q_moe_gu` = one dispatch over 7 experts x 2304 rows; each simdgroup computes gate and up rows together
from the same x registers and writes `h = silu(min(g,10)) * clamp(u,-10,10) * w` into scratch[7][2304]. `q_moe_down` =
one dispatch over the 5120 output rows; the K space (routed expert, block) is flattened so all 6 experts' down
matrices stream in one pass, then the shared expert (different type), result accumulated into y. Each expert matrix is
its own buffer binding (slot can be anywhere in any buffer).

Prefill matmul (`q_mm<Q, TM, TN, NSG, NBUF>`): output tile TM tokens x TN weight rows; per K step of 128 the TG
dequantizes TN x 128 weights into threadgroup memory as half (`q_tile_deq`: works for every type because the lane
slices that cover a 128-element chunk are consecutive lanes), then `mpp::tensor_ops::matmul2d` (half x half -> float
cooperative accumulator, `multiply_accumulate`), double-buffered tiles (one barrier per K step). Activations must be
half (float left operands run at ~8 TFLOPS vs ~31). n >= 512: the whole weight is dequantized once to half (scratch,
bandwidth-bound) and the dense `q_mm_f16_64x128` GEMM runs at 92-98% of peak. Edge tiles rely on the tensor API's
bounds checking (any n, any rows).

MoE prefill: gather (expert-sorted token rows -> half Xg, shared expert rows appended) -> `q_moe_mm_gu` (two dequant
tiles + two accumulators per TG, fused clamp/SwiGLU/routing weight in the epilogue via
`cooperative_tensor::get_multidimensional_index`, H stored as half) -> `q_moe_mm_down` (Z f32 per entry) ->
`q_moe_reduce` (Y[t] += Z[shared t] + sum_k Z[slot_of[t][k]] in fixed order: deterministic, no atomics). Expert
weights are reached through a device pointer table (gpuAddress + offset) written into the scratch header, with
`useResource` for residency, so any number of experts in any buffers works.

## Tests (ctest, all pass)

| test | what | result |
|---|---|---|
| `test_quant_ref` | CPU dequant vs Homebrew libggml `to_float` (dlopen, test-only): 1M random elements per type for all 10 types + real rows pread from the model (IQ1_S gate/down, IQ2_XXS up/down, Q6_K, Q5_K, Q8_0, BF16, F32, IQ4_NL engram table, MXFP4 from dspark) | **bit-exact, all** |
| `test_quant_matvec` | GPU matvec vs dequant -> double dot, 29 cases = every real shape x type (incl. head 129280 rows, wo_a block-diag, accumulate, odd rows) | max err <= 2.4e-8 x sum\|w x\| (tol 2e-5) |
| `test_quant_matvec_n` | n = 2..16, all types, block-diag, accumulate | vs half-x ref <= 3e-8 (n<8); vs F32 <= 6e-5 |
| `test_quant_matmul` | tensor-API matmul, 11 cases (all types, odd n, accumulate) | vs half-rounded ref <= 3e-7; vs F32 ref <= 7e-5 |
| `test_quant_moe` | MoE decode, all 4 type combos, 0/3/6 routed, with/without shared, clamp-heavy case, 4-byte-aligned slot offsets in one arena | h err <= 5e-9, y err <= 1.4e-7 |
| `test_quant_moe_prefill` | grouped GEMM: multi-tile expert, 3 type combos, with/without shared | vs half-emulating ref <= 1.4e-5; vs F32 ref <= 1.9e-4 |

Errors are normalized: |gpu - ref| / sum_c |w_c x_c| (per output). The half staging of the tensor-API paths
(activations, weights, MoE H) costs ~3e-5..2e-4 of that scale vs the F32 path; the decode paths (n=1, MoE decode) are
F32 end-to-end (1e-8).

## Benchmarks (M5 Pro 20-core GPU, best-of-N GPU time, weights rotated over >= 512 MiB so every call streams DRAM)

Measured bandwidth (`bench_membw`, 1 GiB): chunked read **284 GB/s median / 293 best**, grid-stride float4 265-283,
copy 260-265 (r+w). %BW below is against the bandwidth measured in the same process run (~285-290).

Decode matvec (`bench_qmatvec`, default launch heuristic):

| type | tensor | shape | MB | us | %BW |
|---|---|---|---|---|---|
| Q6_K | attn_q_a | 5120->1280 | 5.38 | 21.6 | 86 |
| Q5_K | attn_q_a | 5120->1280 | 4.51 | 21.4 | 73 |
| Q6_K | attn_q_b | 1280->32768 | 34.4 | 126.6 | 94 |
| Q5_K | attn_q_b | 1280->32768 | 28.8 | 121.1 | 82 |
| Q6_K | attn_kv | 5120->512 | 2.15 | 10.3 | 72 |
| Q6_K | attn_output_a (g8) | 4096->8192 | 27.5 | 99.9 | 95 |
| Q6_K | attn_output_b | 8192->5120 | 34.4 | 128.1 | 93 |
| Q5_K | attn_output_b | 8192->5120 | 28.8 | 113.0 | 88 |
| Q6_K | shexp gate/up | 5120->2304 | 9.68 | 36.9 | 91 |
| Q6_K | shexp down | 2304->5120 | 9.68 | 41.6 | 80 |
| Q8_0 | shexp gate/up | 5120->2304 | 12.5 | 50.4 | 86 |
| Q8_0 | shexp down | 2304->5120 | 12.5 | 46.4 | 93 |
| Q6_K | output head | 5120->129280 | 543 | 1889 | 99 |
| Q6_K | engram_kv | 6144->25600 | 129 | 479 | 93 |
| F32 | router | 5120->384 | 7.86 | 31.6 | 86 |
| F32 | hc_fn | 20480->24 | 1.97 | 10.9 | 62 |
| BF16 | compressor_kv | 5120->512 | 5.24 | 25.1 | 72 |
| BF16 | indexer q_b | 1280->4096 | 10.5 | 41.1 | 88 |
| F16 | synthetic | 5120->5120 | 52.4 | 188.5 | 96 |
| IQ1_S | one expert gate/up | 5120->2304 | 2.30 | 14.5 | 55 |
| IQ1_S | one expert down | 2304->5120 | 2.30 | 14.7 | 54 |
| IQ2_XXS | one expert gate/up | 5120->2304 | 3.04 | 17.4 | 61 |
| IQ1_S | synthetic | 5120->32768 | 32.8 | 147 | 78 |
| IQ2_XXS | synthetic | 5120->32768 | 43.3 | 188 | 83 |

Small matrices (< ~5 MB) carry a ~5-8 us fixed cost (launch + barrier + ramp); IQ matvec is ALU-bound (~3 lane-ops
per weight budget at 1.56 bpw), so single experts sit at 55-60%. In the real decode path the experts go through the
fused MoE kernels:

MoE decode (`bench_qmatvec --moe`, 6 routed + shared, 2 dispatches + barrier):

| gate/up | down | shared | MB | us | GB/s | %BW |
|---|---|---|---|---|---|---|
| IQ1_S | IQ1_S | Q6_K | 70.5 | 301.8 | 234 | 81 |
| IQ1_S | IQ2_XXS | Q8_0 | 83.5 | 341.5 | 245 | 85 |
| IQ1_S | IQ2_XXS | Q6_K | 74.9 | 315.7 | 237 | 83 |
| IQ2_XXS | IQ2_XXS | Q6_K | 83.8 | 347.1 | 241 | 84 |

Prefill matmul (`bench_qmatmul`, TFLOPS; F16 GEMM peak measured with the same tensor API: **30.7-31.3 TFLOPS**):

| type | tensor | shape | n=32 | n=128 | n=512 | n=2048 | n=4096 | % peak @4096 |
|---|---|---|---|---|---|---|---|---|
| Q6_K | attn_q_b | 1280->32768 | 7.8 | 20.0 | 23.3 | 29.3 | 30.6 | 97 |
| Q6_K | attn_output_b | 8192->5120 | 6.0 | 17.2 | 23.6 | 28.1 | 28.9 | 92 |
| Q5_K | attn_output_b | 8192->5120 | 6.0 | 16.9 | 23.9 | 28.2 | 28.8 | 92 |
| Q6_K | attn_q_a | 5120->1280 | 3.5 | 9.3 | 26.8 | 29.2 | 29.3 | 93 |
| Q8_0 | shexp gate/up | 5120->2304 | 4.9 | 13.8 | 22.5 | 27.8 | 29.0 | 92 |
| IQ1_S | exp gate/up | 5120->2304 | 6.0 | 15.7 | 24.8 | 28.3 | 29.1 | 93 |
| IQ2_XXS | exp down | 2304->5120 | 6.7 | 17.0 | 26.7 | 29.7 | 30.6 | 98 |
| BF16 | indexer q_b | 1280->4096 | 7.9 | 23.2 | 28.0 | 31.2 | 31.7 | 101 |
| F16 | synthetic | 4096->8192 | 6.2 | 20.1 | 30.8 | 31.9 | 30.9 | 98 |

n >= 512 uses pre-dequant + dense GEMM; the fused dequant-tile path alone reaches 18-22 TFLOPS (60-70%) at large n.
Plain-GEMM probe variants: 64x128 tiles K-step 128 = 31.3 TF; 128x128 = 27; 64x64 = 22; float left operand = 7.9.

MoE prefill (`bench_qmatmul --moe`, IQ1_S routed + Q6_K shared, top-6 random routing):

| n | experts | tokens/expert | ms | TFLOPS | % peak |
|---|---|---|---|---|---|
| 512 | 96 | 32 | 31.6 | 8.0 | 26 |
| 2048 | 96 | 128 | 83.6 | 12.1 | 40 |
| 2048 | 32 | 384 | 73.5 | 13.8 | 45 |
| 4096 | 64 | 384 | 147.5 | 13.8 | 45 |

matvec_n (verify blocks): measured only while a llama-bench model run was saturating the GPU, so absolute numbers are
unreliable; relative to n=1 in the same conditions: n=2..4 costs 1.3-1.6x the n=1 time on K-quants (one weight read),
n>=8 goes to the tile path (~2.5x n=1 for 34 MB matrices at n=8..32). Re-measure on an idle machine:
`bench_qmatvec --n 4`.

## Known limits

- IQ1_S / IQ2_XXS matvec is ALU-bound (55-60% of BW for one 2.3 MB expert; 78-83% for big matrices; MoE decode 81-85%).
- Small matrices (attn_kv 2 MB, hc_fn 2 MB, compressor 5 MB) lose 25-40% to launch/ramp; fuse them with neighbours.
- matvec_n with n >= 5 is FP32-ALU-bound; the tile path for n >= 8 still dequantizes per call (no weight reuse across
  calls) and its 32-token tile wastes MMA at n = 8..16.
- MoE prefill reaches only 26-45% of the tensor peak: each TG re-dequantizes its 64 weight rows for every 64-token tile
  and `q_moe_mm_gu` holds two 16 KB tiles (one TG per core); experts with ~32 tokens fill half a 64-token tile.
- `encode_matmul` / pre-dequant paths use module-owned scratch buffers guarded by barriers: correct within one command
  buffer, but two command buffers encoded concurrently from different threads must not use them simultaneously.
- MoE prefill scratch header is host-written at encode time: the scratch must not be in use by an in-flight command
  buffer (double-buffer it if overlapping chunks).
- MoE decode takes routing weights from a device buffer but slot bindings from the host (needs the expert ids on the
  host, which the SSD-streaming expert cache needs anyway). A fully GPU-resolved path would use the pointer-table
  mechanism of the prefill kernels.
- No MXFP4 / IQ4_NL GPU kernels (draft model experts are MXFP4; add `QB_MXFP4` when the drafter is integrated: 17 B
  blocks, 4-bit e2m1 table + e8m0 scale, same decoder interface).

## Next optimization ideas

1. MoE prefill: per-expert pre-dequant (to half) when an expert has >= ~256 tokens, then dense GEMM; 128-token tiles for
   big experts; route the shared expert through `encode_matmul_h` (pre-dequant path); single-tile gate/up interleaving
   to free threadgroup memory for 2 TGs/core.
2. IQ1_S decode: process 4 grid groups per lane with 8-lane blocks only in the fused MoE kernel (more x reuse), or
   int8-quantized x with integer dot products; try `relaxed` half math for the grid dot (measured +3% only for
   half-x, with 4e-5 error: not adopted).
3. Fuse small decode matvecs (attn_kv + compressor kv/gate + indexer proj share x) into one dispatch with per-segment
   types; fuse rmsnorm into matvec x loads.
4. matvec_n: a 16-token MMA tile (`matmul2d_descriptor(16, ...)`) with occupancy-friendly single-buffered 8 KB tiles,
   or half FMA (`half4` dots) for n = 5..8.
5. Prefill: overlap tile dequant with MMA via warp specialization (producer simdgroups dequantize, consumers multiply)
   instead of the uniform dequant -> barrier -> MMA loop.
6. Persist pipelines in an MTLBinaryArchive (startup currently compiles the whole bundle; ~170 kernel variants).
