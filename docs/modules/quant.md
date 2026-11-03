# Module: quant — quantized matrix kernels (Metal)

Branch `mod/quant`. Files:

| file | content |
|---|---|
| `src/kernels/quant_types.hpp` | `ms::q::QType` (= ggml type ids), block geometry helpers (plain C++) |
| `src/kernels/quant.hpp`, `quant.mm` | host encode functions (`ms::q::encode_*`), launch heuristics, env overrides |
| `src/kernels/quant_grids.h` | GENERATED (`tools/gen_quant_grids.py <ggml-common.h> out`): IQ1_S (= IQ1_M) / IQ2_XXS / IQ2_XS grid VALUES + derived GPU tables |
| `src/kernels/quant_common.h` | constant tables, unaligned-safe loads, `QMvArgs` |
| `src/kernels/quant_blocks.h` | per-type block decoders `QB_*` (one static interface, all kernels are templates over it) |
| `src/kernels/quant_tile.h` | tensor-API includes + `q_tile_deq` (dequantize a TN x 128 weight tile to half in threadgroup memory) |
| `src/kernels/quant_mv.metal` | `q_mv_*` decode matvec, `q_mvn_*` small-n matvec |
| `src/kernels/quant_moe.metal` | `q_moe_gu_*`, `q_moe_down_*` fused MoE decode |
| `src/kernels/quant_mm.metal` | `q_mm_*` prefill matmul (fused dequant tiles), `q_mm_{f16,bf16,f32}_64x128` dense, `q_deq_f16_*`, `q_f32_to_f16`, `q_gemm_f16_*` peak probe |
| `src/kernels/quant_moe_prefill.metal` | `q_moe_gather`, `q_moe_rgu_*` / `q_moe_rdown_*` (register-dequant grouped GEMM, default), `q_moe_dgu` (dense shared expert), `q_moe_mm_gu_*` / `q_moe_mm_down_*` (previous threadgroup-tile path), `q_moe_reduce` |
| `src/kernels/quant_bench.metal` | `q_bw_*` bandwidth probes |
| `tests/ref/dequant.{hpp,cpp}` | CPU reference dequant for F32 F16 BF16 Q8_0 Q5_K Q6_K IQ1_S IQ2_XXS IQ1_M IQ2_XS IQ4_NL MXFP4, `matvec_ref`, `random_blocks` |
| `tests/test_quant_{ref,matvec,matvec_n,matmul,moe,moe_prefill}.mm`, `tests/quant_test_util.hpp` | ctests |
| `bench/membw.mm`, `bench/qmatvec.mm`, `bench/qmatmul.mm` | microbenchmarks |
| `docs/dspark_tensors.txt` | tensor table of the dspark draft GGUF (same format as docs/tensors.txt; MXFP4 experts) |

No changes to `CMakeLists.txt` or `src/core`.

## API (namespace `ms::q`, all functions only encode; never commit/wait)

Weights are raw GGUF block rows: a `[in -> out]` tensor is `rows = out` rows of `cols = in` elements. Buffers are
`ms::Buffer` (buf + byte offset). Weight buffer offsets must be >= 4-byte aligned (GGUF tensor/expert offsets are; the
IQ1_M and Q5_K decoders rely on it: their fields sit at 4-byte multiples of 56 / 176-byte blocks and use 32-bit loads);
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
// same with routed matrices from a device pointer table (6 x {gate, up, down} uint64), and its pieces
// (MoePart::SharedGu / AfterSharedGu / Shared / Routed / SharedDown; see quant.hpp for the identities)
void encode_moe_decode_table(Enc&, const MoeDecodeArgs&, const Buffer& table);
void encode_moe_decode_table_part(Enc&, const MoeDecodeArgs&, const Buffer& table, MoePart);
// MoE prefill: MoePrefillArgs{types, n_tokens, n_exp, slots[n_exp] (host), offs[n_exp+1] (host), tok/wt (device,
//   expert-sorted), slot_of[n*k] (device), shared, X, Y (accumulated), scratch >= moe_prefill_scratch_bytes(total,n,n_exp),
//   path = -1 (env; 1 register kernels, 0 previous tile kernels), shared_dense = -1 (env; 1 dense shared expert)}
// scratch = header | Xg | H | Z | W16 (47 MB: the shared expert's weights as half for the dense GEMMs)
void encode_moe_prefill(Enc&, const MoePrefillArgs&);
```

Supported types: matvec / matvec_n / matmul (incl. `q_deq_f16_*`): F32, F16, BF16, Q8_0, Q5_K, Q6_K, IQ1_S, IQ2_XXS,
IQ1_M, IQ2_XS. MoE (decode fused + table + parts, prefill register / tile paths, dense and grouped shared expert): routed
gate/up and down each any of IQ1_S, IQ2_XXS, IQ1_M, IQ2_XS (independently: UD-IQ1_S uses IQ1_S/IQ2_XXS, the
ISTA-DASLab GSQ-RCO-Coder model mixes IQ1_M/IQ2_XXS/IQ2_XS per layer), shared Q6_K, Q8_0 or Q5_K. MoE decode kernel
names are `q_moe_{gu,down}[_t]_<routed>_<shared>_r<NR>` (e.g. `q_moe_gu_t_iq1_m_q5_k_r2`), built by `moe_pair()` from the
two type names (every routed x shared pair and NR 1/2/4 instantiated). `matvec_rmsnorm` excludes the IQ types.
IQ4_NL / MXFP4: CPU reference only (engram rows are 144 B lookups; MXFP4 is only in the dspark draft).

Barriers: the encoder is concurrent (`MTLDispatchTypeConcurrent`). Functions insert the barriers they need internally
(MoE between phases; `encode_matmul` around its shared X16 scratch; the n >= 512 pre-dequant path around its shared W16
scratch). Callers must still `e.barrier()` between dependent ops of their own (e.g. matvec output -> next matvec input).

Env overrides (all optional, for tuning; defaults = measured best):
`MACSEEK_MV_NR/RSG/KS` (matvec launch), `MACSEEK_MVN_NR/SG`, `MACSEEK_MVN_MM_MIN` (n at which matvec_n uses tiles,
default 8, 0 = never), `MACSEEK_MM_TILE` (e.g. `128x64_b2`), `MACSEEK_MM_PREDEQ_N` (default 512, 0 = never),
`MACSEEK_MOE_{GU,D}_{NR,KS,RSG}`.
MoE prefill: `MACSEEK_MOEP_PATH` (1 = register-dequant kernels, default; 0 = previous threadgroup-tile kernels, with
`MACSEEK_MOEP_TM` 64|128), `MACSEEK_MOEP_SHDENSE` (1 = shared expert via half pre-dequant + dense GEMMs, default; 0 =
grouped kernels), launch shapes per token-tile class `MACSEEK_MOEP_RGU64/_RGU48/_RGU/_RGU16` (gate/up) and
`MACSEEK_MOEP_RD64/_RD48/_RD/_RD16` (down), format `TMxTNkTKsNSG` (instantiated set in quant_moe_prefill.metal; "0"
turns a class off, except the 32-token one), `MACSEEK_MOEP_SKIP` (PROFILING ONLY, wrong output: bit mask 1 gather,
2 routed gate/up, 4 shared gate/up, 8 routed down, 16 shared down, 32 reduce, 64 << c routed tile class c).

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
| IQ1_M | 256 / 56 B | 16 x 16 (8 x 32 in MoE decode gate/up) | 2 grid groups = one 16-element scale half |
| IQ2_XS | 256 / 74 B | 16 x 16 | 2 grid groups = one scale nibble |

IQ1_M (`QB_IQ1M_T<L>`): no separate d; the fp16 d is assembled from the top nibbles of the 4 scale words (2 dword
loads + 5 ALU ops); per 8-element group its own qh nibble (3 index bits + delta sign), per 16 elements a 3-bit scale.
`dot = d * sum_halves s_h * sum_g (grid_g . x_g + delta_g * sum(x_g))` with per-8 x sums precomputed in load_x. The MoE
decode gate/up kernels use 8 lanes per block (`QMoeGuDec<QB_IQ1M>`: one sub-block per lane, d/scale loads amortized
over 32 weights; MoE decode 75.7% -> 83.5% of BW); matvec keeps 16 (8 lanes lose on 2304-column rows: 9 blocks).
IQ2_XS (`QB_IQ2XS_T<L>`): 9-bit grid index into a 512 x 8 half table (8 KB constant, `Q_IQ2XS_GRID_HALF_VALUES`) and the
IQ2_XXS sign-mask table (7-bit index). Q5_K (shared expert of the GSQ model): scale/min pairs unpacked branch-free
from the 12 scale bytes (lanes of one block used to diverge on j < 4) and 32-bit loads (MoE decode with Q5_K shared
+4-5%).

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

MoE prefill: gather (expert-sorted token rows -> half Xg, shared expert rows appended) -> gate/up (fused
clamp/SwiGLU/routing weight epilogue, H stored as half) -> down (Z f32 per entry) -> `q_moe_reduce` (Y[t] += Z[shared t]
+ sum_k Z[slot_of[t][k]] in fixed order: deterministic, no atomics; tokens without an entry in this call are skipped).
Expert weights are reached through a device pointer table (gpuAddress + offset) written into the scratch header, with
`useResource` for residency, so any number of experts in any buffers works.

Routed experts (default path, `q_moe_rgu` / `q_moe_rdown`): every simdgroup works alone (`execution_simdgroup`
matmul2d) on TM tokens x TN weight rows and dequantizes its weights for each K step (32 or 64) straight into the
matmul's **right-input cooperative tensor** (registers): no threadgroup tiles, no barriers, so the ALU dequant of some
simdgroups overlaps the tensor-unit work of others (probe: MMA alone 28 TF on the padded work, dequant adds ~20%).
Cooperative layout (dumped on M5, guarded by the tests): lane L holds K quad q = (L&1) | ((L>>3)&1)<<1 and row rr =
((L>>1)&3) | ((L>>4)&1)<<2; element 4*(RB*g + rb) + c is k = 16g + 4q + c, row rr + 8rb. So each lane decodes runs of 4
consecutive weights of a row (`QDeqq`: IQ1_S / IQ2_XXS / IQ1_M / IQ2_XS share their loads per 32-weight sub-block; IQ2 signs by ALU
(7 bits + parity); the IQ2_XXS 4 KB grid is copied to threadgroup memory, the IQ2_XS 8 KB grid is read from the device
table (the copy cost more than it saved: 97.8 -> 95.9 ms); IQ1_M: d and scale words once per K step, per-lane shifts
hoisted, FMA dequant; generic `Q::deq4` for Q6_K / Q8_0 / Q5_K). gate/up keeps two
accumulators (gate, up) per simdgroup; the destination layout gives 4 consecutive rows of one token per 4 elements ->
one 8-byte H store. Token tiles: an expert with m tokens gets ceil(m/64) tiles (one weight dequantization per tile):
full 64-token tiles + the smallest of 64/48/32/16 that holds the remainder (padding 1.12x on the real distribution vs
1.52x for fixed 64). Launch shapes (TMxTN, K step): gate/up 64x16, 48x16, 32x32, 16x32 (K 64 for IQ1_S / IQ2_XS, 32 for
IQ2_XXS / IQ1_M: registers; IQ1_M at K 64 is 12% slower); down 64x32, 48x32, 32x64, 16x64 (K 32); 4 simdgroups per threadgroup. The config space is flat (+-1%).

Shared expert (all n tokens, default): gate and up are dequantized to half into W16 (concurrently with the gather),
then `q_moe_dgu` (dense 64x64 x2 tiles from device memory + SwiGLU epilogue), then down is dequantized into W16 and
`q_mm_f16_64x128` writes its Z rows (26-28 TF incl. the dequant).

## Tests (ctest, all pass)

| test | what | result |
|---|---|---|
| `test_quant_ref` | CPU dequant vs Homebrew libggml `to_float` (dlopen, test-only): 1M random elements per type for all 12 types (IQ1_M / IQ2_XS random only) + real rows pread from the model (IQ1_S gate/down, IQ2_XXS up/down, Q6_K, Q5_K, Q8_0, BF16, F32, IQ4_NL engram table, MXFP4 from dspark) | **bit-exact, all** |
| `test_quant_matvec` | GPU matvec vs dequant -> double dot, 39 cases = every real shape x type (incl. head 129280 rows, wo_a block-diag, accumulate, odd rows; IQ1_M / IQ2_XS expert shapes + 32768-row synthetic; Q5_K shared expert) | max err <= 2.4e-8 x sum\|w x\| (tol 2e-5) |
| `test_quant_matvec_n` | n = 2..16, all types, block-diag, accumulate | vs half-x ref <= 3e-8 (n<8); vs F32 <= 6e-5 |
| `test_quant_matmul` | tensor-API matmul, 19 cases (all types, odd n, accumulate, n >= 512 pre-dequant for IQ1_M / IQ2_XS / Q5_K / IQ2_XXS) | vs half-rounded ref <= 3e-7; vs F32 ref <= 7e-5 |
| `test_quant_moe` | MoE decode, 22 combos: UD-IQ1_S combos, all 9 gate/up x down combos of IQ1_M / IQ2_XXS / IQ2_XS with Q5_K shared, mixed old/new; 0/3/6 routed, with/without shared, clamp-heavy cases; 4-byte-aligned slot offsets in one arena. Each also through the table variant: All (== bound kernel, bit-identical), SharedGu+AfterSharedGu (bit-identical to All), Shared+Routed, SharedGu+SharedDown (bit-identical to Shared), +Routed | h err <= 5e-9, y err <= 3.6e-7 (Q5_K shared; 1.4e-7 otherwise) |
| `test_quant_moe_prefill` | grouped GEMM, 15 cases (7 with the GSQ types: gate/up type != down type, Q5_K shared dense and grouped, both kernel paths): multi-tile expert; explicit expert sizes 1, 2, 15-17, 31-33, 47-49, 63-65, 100..300 and one with 600 (> 512) tokens; Zipf(0.8) top-6 of 64; 4 type combos, with/without shared, both kernel paths and both shared paths; every case run twice -> bit-identical | vs half-emulating ref <= 4.4e-5; vs F32 ref <= 9.4e-4 (Q5_K shared cases; 1.9e-4 otherwise) |

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

MoE prefill, realistic (`bench_qmatmul --moe --real [--n 4096] [--group 64] [--zipf 0.8] [--types iq1|iq2]`): what
ds41.mm runs per layer: n = 4096 tokens, top-6 of 384 experts drawn from a Zipf(0.8) popularity (tokens/expert 11..1716,
median 33), 6 calls of 64 experts (one command buffer each), shared expert in the first; sum of GPU times, best of 5,
idle machine. Effective TFLOPS = 2 * 3 * 5120 * 2304 * (entries + n) / time. "before" = commit a0c9124.

| routed gate/up, down + shared | before ms | before TF | after ms | after TF | speedup |
|---|---|---|---|---|---|
| IQ1_S, IQ1_S + Q6_K | 194.6 | 10.4 | 97.0 | 20.9 | 2.01x |
| IQ2_XXS, IQ2_XXS + Q8_0 | 195.6 | 10.4 | 101.8 | 19.9 | 1.92x |
| IQ1_S + Q6_K, n = 2048 | 141.1 | 7.2 | 55.1 | 18.4 | 2.56x |

Per-phase (IQ1_S, after, `MACSEEK_MOEP_SKIP`): gather 2.5 ms, routed gate/up 55.7, shared gate/up 7.4, routed down
24.0, shared down 3.8, reduce 4.9 (6 calls). Routed gate/up runs at ~21 TF useful (25 TF on the padded tiles), down at
~24 TF; the 64-token tiles reach 23/26 TF, 16-token remainder tiles ~9.5 TF (dequant-bound: a full weight pass for <= 16
tokens).

MoE prefill (`bench_qmatmul --moe`, IQ1_S routed + Q6_K shared, top-6 uniform random routing):

| n | experts | tokens/expert | before ms | before TF | after ms | after TF | % peak |
|---|---|---|---|---|---|---|---|
| 512 | 96 | 32 | 31.1 | 8.2 | 13.2 | 19.2 | 61 |
| 2048 | 96 | 128 | 80.5 | 12.6 | 44.8 | 22.6 | 72 |
| 2048 | 32 | 384 | 71.9 | 14.1 | 42.5 | 23.9 | 76 |
| 4096 | 64 | 384 | 143.6 | 14.1 | 84.5 | 24.0 | 76 |

What did not help (measured, reverted): software-pipelined dequant (two register sets: spills, 1.5-2x slower); gate|up
in one 32x64 matmul (slower than two 32x32); half-precision dequant arithmetic (no gain, IQ2_XXS error 6e-6 -> 1.8e-4);
IQ1_S grid as 2-bit codes + ALU expansion, or a device-memory instead of constant table (no change); TM 96/128 tiles
(register spills / TN=8 layout differs); NSG 2 / 8 (+-1%).

### GSQ-RCO-Coder types (IQ1_M / IQ2_XS routed, Q5_K shared; 192 experts)

Measured on an idle machine (no other GPU job), same binary for old and new types, best of 3-5 interleaved runs.
Decode matvec (`bench_qmatvec`, default launch):

| type | tensor | shape | MB | us | GB/s | %BW |
|---|---|---|---|---|---|---|
| IQ1_S | one expert gate/up | 5120->2304 | 2.30 | 14.5 | 159 | 53 |
| IQ1_M | one expert gate/up | 5120->2304 | 2.58 | 15.9 | 162 | 54 |
| IQ1_S | one expert down | 2304->5120 | 2.30 | 14.5 | 159 | 53 |
| IQ1_M | one expert down | 2304->5120 | 2.58 | 16.2 | 159 | 53 |
| IQ2_XXS | one expert gate/up | 5120->2304 | 3.04 | 17.7 | 172 | 57 |
| IQ2_XS | one expert gate/up | 5120->2304 | 3.41 | 17.6 | 193 | 64 |
| IQ2_XXS | one expert down | 2304->5120 | 3.04 | 17.6 | 173 | 57 |
| IQ2_XS | one expert down | 2304->5120 | 3.41 | 19.7 | 173 | 57 |
| IQ1_S / IQ1_M | synthetic | 5120->32768 | 32.8 / 36.7 | 148.6 / 166.6 | 221 / 220 | 73 / 73 |
| IQ2_XXS / IQ2_XS | synthetic | 5120->32768 | 43.3 / 48.5 | 187.5 / 190.0 | 231 / 255 | 76 / 85 |
| Q5_K | shexp gate/up | 5120->2304 | 8.11 | 33.1 | 245 | 81 |
| Q5_K | shexp down | 2304->5120 | 8.11 | 31.8 | 255 | 85 |

MoE decode (`bench_qmatvec --moe [gu,down,shared]`, 6 routed + shared; GB/s is the stable measure, %BW is against the
bandwidth measured in each process, 281-300 GB/s):

| gate/up | down | shared | MB | us | GB/s | %BW |
|---|---|---|---|---|---|---|
| IQ1_S | IQ1_S | Q6_K | 70.50 | 294.6 | 239 | 79 |
| IQ1_S | IQ2_XXS | Q8_0 | 83.50 | 333.0 | 251 | 83 |
| IQ2_XXS | IQ2_XXS | Q6_K | 83.77 | 343.5 | 244 | 81 |
| IQ1_M | IQ1_M | Q6_K | 75.48 | 305.5 | 247 | 82 |
| IQ2_XS | IQ2_XS | Q6_K | 90.41 | 357.5 | 253 | 84 |
| **IQ1_M** | **IQ1_M** | **Q5_K** | 70.78 | 300.6 | 235 | 78 |
| **IQ2_XXS** | **IQ2_XXS** | **Q5_K** | 79.07 | 335.3 | 236 | 78 |
| **IQ2_XS** | **IQ2_XS** | **Q5_K** | 85.71 | 347.1 | 247 | 82 |
| **IQ1_M** | **IQ2_XXS** | **Q5_K** | 73.54 | 302.1 | 243 | 81 |
| **IQ2_XS** | **IQ1_M** | **Q5_K** | 80.73 | 342.3 | 236 | 78 |

With the same shared type the new routed types match or beat the old ones (IQ1_M/IQ1_M 247 vs IQ1_S/IQ1_S 239 GB/s;
IQ2_XS 253 vs IQ2_XXS 244). The Q5_K shared expert streams a little slower than Q6_K (Q5_K matvec 81-85% vs Q6_K 91%),
so the GSQ combos sit at 235-247 GB/s.

MoE prefill realistic (`bench_qmatmul --moe --real --experts 192 --n 4096 --types gu,down,shared`): top-6 of 192 experts,
Zipf(0.8) (tokens/expert 29..1933, median 69), 3 calls of 64 experts, shared expert in the first; best of 5.

| routed gate/up, down + shared | ms | effective TFLOPS | % F16 peak (30.8) |
|---|---|---|---|
| IQ1_S, IQ1_S + Q6_K | 90.6 | 22.4 | 73 |
| IQ2_XXS, IQ2_XXS + Q8_0 | 95.0 | 21.4 | 69 |
| **IQ1_M, IQ1_M + Q5_K** | 93.0 | 21.8 | 71 |
| **IQ2_XXS, IQ2_XXS + Q5_K** | 94.9 | 21.4 | 69 |
| **IQ2_XS, IQ2_XS + Q5_K** | 93.9 | 21.6 | 70 |
| **IQ1_M, IQ2_XXS + Q5_K** | 93.4 | 21.7 | 71 |
| **IQ2_XS, IQ1_M + Q5_K** | 93.2 | 21.8 | 71 |
| IQ1_M, IQ1_M + Q5_K, MACSEEK_MOEP_SHDENSE=0 | 104.8 | 19.4 | 63 |
| (reference: IQ1_S + Q6_K, 384 experts) | 97.3 | 20.9 | 68 |

Per phase (`MACSEEK_MOEP_SKIP`): routed gate/up IQ1_S 51.8 ms, IQ1_M 54.7 (was 70 at K 64, 58.2 before the
shift/FMA rewrite); routed down IQ1_S 22.6, IQ1_M 22.1; IQ2_XXS gate/up 56.2, IQ2_XS 54.8 (K 64), down 22.5 / 22.8.
Prefill matmul (`bench_qmatmul`) of the new types equals the old ones within noise (IQ1_M gate/up 7.9 / 16.1 / 24.9 /
28.5 / 29.4 TF at n = 32 / 128 / 512 / 2048 / 4096; IQ2_XS down 8.0 / 16.8 / 27.2 / 29.7 / 30.7).

Tuning that did not help (measured, reverted): IQ1_M 8 lanes in matvec (down 54% -> 41%: 9 blocks per row) and in MoE
decode down (no change); IQ1_M MoE prefill gate/up K 64 (+12% time); IQ2_XS grid copied to threadgroup memory in MoE
prefill (+2%); K-split for IQ2_XS matvec (-20%). 8-lane gate/up in MoE decode would also help IQ1_S (+3%, 239 -> 243 GB/s)
but changes UD-IQ1_S decode numerics (summation order), so it is not enabled for IQ1_S (`QMoeGuDec` in quant_moe.metal).

matvec_n (verify blocks): measured only while a llama-bench model run was saturating the GPU, so absolute numbers are
unreliable; relative to n=1 in the same conditions: n=2..4 costs 1.3-1.6x the n=1 time on K-quants (one weight read),
n>=8 goes to the tile path (~2.5x n=1 for 34 MB matrices at n=8..32). Re-measure on an idle machine:
`bench_qmatvec --n 4`.

## Known limits

- Metal bundle: 528 kernels (was 350 before IQ1_M / IQ2_XS / Q5_K-shared); uncached `newLibraryWithSource` 2.3 s
  (was 1.6 s; pipelines are created lazily).
- IQ1_M MoE prefill gate/up is ~5% behind IQ1_S per weight (routed gate/up at n = 4096): its dequant
  needs more registers (K 32 instead of 64; 54.7 vs 51.8 ms). matvec_n n = 2..7 with IQ2_XS is ~8% slower than IQ2_XXS (FP32 bound).

- IQ1_S / IQ2_XXS matvec is ALU-bound (55-60% of BW for one 2.3 MB expert; 78-83% for big matrices; MoE decode 81-85%).
- Small matrices (attn_kv 2 MB, hc_fn 2 MB, compressor 5 MB) lose 25-40% to launch/ramp; fuse them with neighbours.
- matvec_n with n >= 5 is FP32-ALU-bound; the tile path for n >= 8 still dequantizes per call (no weight reuse across
  calls) and its 32-token tile wastes MMA at n = 8..16.
- MoE prefill (realistic distribution) reaches ~65% of the tensor peak: every token tile re-dequantizes the expert's
  weights (one pass per <= 64 tokens), which is not fully hidden behind the tensor work (gate/up ~20% over its MMA
  time), remainder tiles pad to 16 rows, and each of the 6 per-layer calls pays a gather + Y read/modify/write (~7 ms
  of 97). The register path relies on the cooperative-tensor layouts dumped on M5 / macOS 27 (tests catch a change).
- MoE prefill scratch includes 47 MB for the shared expert's half weights (moe_prefill_scratch_bytes).
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

1. MoE prefill: (done: register-dequant kernels, 64/48/32/16 tile classes, dense shared expert.) Next: reduce + gather
   once per layer instead of per 64-expert call (needs a ds41.mm change: keep Z across calls; ~5 ms of 97); pre-dequant
   + dense GEMM only for experts with >= ~1000 tokens (estimated +1-2 ms); cheaper remainder tiles (two experts' <= 16
   token remainders cannot share a tile: different weights).
2. IQ1_S decode: process 4 grid groups per lane with 8-lane blocks only in the fused MoE kernel (more x reuse), or
   int8-quantized x with integer dot products; try `relaxed` half math for the grid dot (measured +3% only for
   half-x, with 4e-5 error: not adopted).
3. Fuse small decode matvecs (attn_kv + compressor kv/gate + indexer proj share x) into one dispatch with per-segment
   types; fuse rmsnorm into matvec x loads.
4. matvec_n: a 16-token MMA tile (`matmul2d_descriptor(16, ...)`) with occupancy-friendly single-buffered 8 KB tiles,
   or half FMA (`half4` dots) for n = 5..8.
5. Prefill matmul (`q_mm`, n < 512): use the MoE register-dequant scheme (right-input cooperative tensors, no barriers)
   instead of the uniform dequant -> barrier -> MMA loop.
6. Persist pipelines in an MTLBinaryArchive (startup currently compiles the whole bundle; 528 kernels, 2.3 s).

### Coder decode tuning (branch coder-dec, docs/modules/dec_gpu.md "GSQ-RCO Coder at 262K")

- Q5_K decode matvec launch rule (`MACSEEK_MV_CFG=1`, default; 0 = generic rule): nr 1, K split until each simdgroup runs
  ~2 block iterations, rsg 1; rows with <= 8 blocks: nr 4, rsg 2, no split. 95-99% of BW on the Coder shapes
  (`bench_qmatvec --rules` compares both rules interleaved; `q::matvec_cfg_rule`).
- IQ1_M `QB_IQ1M_T<L, V>`: V 1 (default) gathers grid + delta from `q_iq1m_gridd` (64 KB, generated by
  tools/gen_quant_grids.py); V 0 = previous dot. MoE decode table kernels with the old decoders: `MACSEEK_MOE_DEC_V=0`
  (pairs `iq1_m0_q5_k`, `iq2_xxs0_q5_k`). IQ2_XXS gate/up in the MoE decode uses 8 lanes; its table gate/up default
  launch is 4 rows x 4 simdgroups.
- `MoeDecodeArgs::gu_ks / d_ks` (K split per call), `bench_qmatvec --moe-sweep gu,down,shared[,shared]` (table part
  Routed / Shared shape sweep; `BENCH_NOSWEEP=1` = default shape only).
