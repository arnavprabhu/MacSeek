# MacSeek — bespoke inference engine for DeepSeek-V4.1-Flash-1017 (Unsloth UD-IQ1_S GGUF) on M5 Pro 64 GB

This file is the contract between modules. If you change an interface, update this file in the same change.

## 0. Goal and rules

- Maximize **prefill and decode speed at 200K context**, with no measurable quality loss vs the pinned baseline.
- Written from scratch: no llama.cpp/ggml code, no code from any earlier DS41 engine repo. Stock Homebrew llama.cpp
  (`/opt/homebrew/bin/llama-*`, `libllama`) is used ONLY as (a) the speed baseline and (b) the logits oracle.
  Reading upstream source for a *spec detail* (e.g. a tokenizer regex) is allowed; copying code is not.
- Principles: `~/.hermes/skills/software-development/local-llm-inference-engine/references/speed-principles.md`
  (roofline per phase, specialize hot paths with fallbacks, measured MoE placement, headroom, 3 gate layers).
- **Only one process may load the model at a time** (64 GB RAM, model is 233 GB). Module agents test kernels on
  synthetic data only. Model runs go through `tools/modellock.sh` (flock on `logs/.model.lock`).
- Every optimization is switchable via an env var / CLI flag, with the slow-but-correct path as fallback.

## 1. Hardware (measured)

- Apple M5 Pro, 18 CPU cores (6 super + 12 perf), 20-core GPU, Metal 4, MTLGPUFamilyApple10, tensor API (Metal 4
  `MetalPerformancePrimitives` matmul2d) available, bf16 available. recommendedMaxWorkingSetSize = 56.6 GB.
- SSD (F_NOCACHE pread): sequential 16 MB blocks 11.5 GB/s; random 2.3 MB reads 6.9 GB/s @1 thread, 14.1 @4,
  14.7 @8, 15.1 @16; random 16 KB: 15K IOPS @1 thread, ~95K IOPS @8-32 threads.
- RAM bandwidth (GPU, `bench_membw`, 1 GiB streaming read, idle machine): **~285 GB/s** sustained (chunked read 284
  median / 293 best; grid-stride float4 268-283; copy 260-265 GB/s counted as read+write). Use 285 GB/s as the decode
  roofline input. Note: GPU benchmarks are badly skewed while a model process (llama-bench) runs; take best-of-N.
- GPU matmul peak (`bench_qmatmul --peak`, Metal 4 tensor API MPP matmul2d, half x half -> float, 64x128 tiles,
  K-step 128): **~31 TFLOPS**; float activations as the left operand drop to ~8 TFLOPS (convert to half first).
  This is the prefill roofline input.

## 2. Model facts (from the GGUF; full tables in docs/tensors.txt and docs/metadata.txt)

Files (read-only, never copy): `~/Unsloth Models/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/8a6ec9cf118e785a362ea94ee123c61c70553be6/`
- `UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-0000{1..5}-of-00005.gguf` (main model, arch `deepseek41`)
- `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (arch `dflash`, native 3-layer block drafter, block 5) — later
- `mmproj-F16.gguf` (vision) — out of scope

`docs/tensors.txt` columns: name, type, dims (ggml order: dims[0] is the contiguous/input dim), nbytes, file, absolute
byte offset of the tensor data in that file.

Shapes: dim=5120, 40 layers, vocab 129280, 64 heads, head_dim 512 (rope part = LAST 64 dims), single shared KV
(MQA, kv_heads=1, K==V latent), q_lora 1280, o_groups 8, o_lora 1024, window 128, MoE 384 experts top-6 + 1 shared,
expert inter 2304, hc_mult 4, sinkhorn 20 iters, hc_eps 1e-6, norm_eps 1e-20, route_scale 1.5, swiglu_limit 10.
compress_ratios[layer] = 0 for layers 0,1; 2 for 2..19; 1 for 20..39. kv_source_layers [2,8,14,20] (layer L uses the
compressed KV of the nearest source <= L), index_source_layers [2,8,14,20,24,28,32,36] (others reuse the top-k
indices of the nearest index source <= L), candidate_source_layer 20 (candidate_topk_blocks 2048, block 8), indexer
32 heads x 128 dims, index_topk 512. RoPE: rope_head_dim 64, interleaved pairs (complex view of adjacent elements).
Layers with compress_ratio 0 use theta 10000 and NO YaRN; layers with ratio>0 use theta 160000 with YaRN
(factor 16, original 65536, beta_fast 32, beta_slow 1). Engram at layers 1 and 14.

Quant mix (UD-IQ1_S): experts IQ1_S (gate/up 34 layers, down 20 layers) and IQ2_XXS (down layers 20-39, gate/up
layers 30-34,39); attention Q6_K (layers 0-26) / Q5_K (27-39); shared expert Q6_K / Q8_0 (14-26); indexer +
compressor BF16; router, norms, hc_* F32; token_embd + output Q6_K; engram table `per_layer_token_embd` IQ4_NL
[256, 768022850] (103 GiB, 144 B per row); engram_kv Q6_K [6144, 25600]; engram_gate F32 [5120, 4].

Byte budget: resident dense read per decode token ~5.91 GiB (attention 3.86, shared expert 1.19, output 0.51,
router 0.29, hc 0.07); active experts ~1.68 GiB/token (6 x 40 slices); experts total 107.3 GiB; engram table 103 GiB.

## 3. Forward pass (normative; mirrors reference/inference/model.py — read it)

GGUF tensor name -> reference name (per layer `blk.L.`):
attn_norm, ffn_norm (RMSNorm 5120); attn_q_a [5120->1280] = wq_a; attn_q_a_norm = q_norm; attn_q_b [1280->32768]
= wq_b; attn_kv [5120->512] = wkv; attn_kv_a_norm = kv_norm; attn_sinks [64]; attn_output_a [4096->8192] = wo_a
(block-diagonal: group g in 0..7 uses output rows g*1024..g*1024+1023 and input = heads 8g..8g+7 of o, i.e. o[g*4096
.. +4095]); attn_output_b [8192->5120] = wo_b; attn_compressor_kv [5120->512], attn_compressor_gate [5120->512] (only
ratio>1 sources 2,8,14), attn_compressor_norm [512]; indexer.attn_q_b [1280->4096] = indexer.wq_b; indexer.proj
[5120->32] = weights_proj; indexer.attn_k [512->128] = indexer.wk; indexer.k_norm [128]; ffn_gate_inp [5120->384]
= gate.weight (F32); exp_probs_b.bias = gate.bias; exp_probs_b_vl.bias (vision only, ignore); ffn_{gate,up,down}_exps
= experts w1,w3,w2 (3D, expert is the outermost dim, each expert's matrix contiguous); ffn_*_shexp = shared expert;
hc_attn_fn [20480->24], hc_attn_base [24], hc_attn_scale [3] (same for hc_ffn_*); engram_kv [6144->25600] = wkv;
engram_gate [5120,4] = q_weight*k_weight product (one row of 5120 per hc copy — verify orientation against the
oracle); output_norm, output [5120->129280] (head), token_embd.

Per token stream h has shape [4][5120] (hc copies). Order of operations:
```
h = embed(tok) repeated 4x;  pre_mix = [1,0,0,0]
for L in 0..39:
  if L in {1,14}: h = engram(h, hashes[L])                        # model.py Engram.forward
  if L in {37,38,39}: main_hidden.append(mean over hc of h)         # dspark only
  # Block.forward
  attn_pre, attn_post, attn_comb = hc_mixes(h, hc_attn_*)           # rms over all 20480, F32 matvec 20480->24, sinkhorn
  x = attn_norm(sum_j pre_mix[j]*h[j]);  x = attention_L(x)
  h = attn_post[i]*x + sum_j attn_comb[j][i]*h[j]                  # hc_post: out copy i = post[i]*x + sum_j comb[j,i]*res[j]
  ffn_pre, ffn_post, ffn_comb = hc_mixes(h, hc_ffn_*)
  x = ffn_norm(sum_j attn_pre[j]*h[j]);  x = moe_L(x)               # NOTE: FFN uses attn_pre (lagged mixes)
  h = ffn_post[i]*x + sum_j ffn_comb[j][i]*h[j]
  pre_mix = ffn_pre
logits = output(output_norm(sum_j pre_mix[j]*h[j]))
```
hc_split_sinkhorn: pre = sigmoid(m[0:4]*s0+b[0:4]) + eps; post = 2*sigmoid(m[4:8]*s1+b[4:8]); comb[j][k] =
m[8+4j+k]*s2+b[8+4j+k] -> row softmax + eps -> col-normalize -> 19 x (row-normalize, col-normalize), each with +eps
in the denominator (kernel.py lines 406-474).

Attention (model.py 766-791): qr = q_norm(wq_a x); q = wq_b qr -> [64][512], rope on last 64 dims of each head;
kv = kv_norm(wkv x) -> [512], rope last 64, (reference quant-dequants kv to fp8 e4m3 / ue8m0 block 32: switchable,
see §6); window ring of 128 kv rows per layer. If ratio>0: compressed KV (from the layer's kv source) + indexer top-k
(from the layer's index source) are appended to the window keys. sparse_attn over the selected rows: scores =
q.k * 512^-0.5, softmax including a per-head sink logit attn_sinks[h] (added to the denominator only), out = sum p*kv
(K==V). Inverse-rope the last 64 dims of the output. Then o [64*512] -> 8 groups -> wo_a block-diag -> [8192] -> wo_b.
Compressor (ratio 2): kv=wkv x, score=wgate x (fp32), groups of 2 consecutive tokens pooled by softmax(score) over the
group per channel; ratio 1: norm(wkv x) directly (no gate). Output -> compressor_norm, rope at position
group_index*ratio with the compressed rope table, (fp4 quant-dequant block 16, e4m3 scales: switchable) -> cache.
Indexer (488-581): index keys = k_norm(wk(latent pre-rope)) with rope on last 64, fp4 sim; q = indexer.wq_b(qr) [32][128]
rope last 64, fp4 sim; weights = proj(x) * 128^-0.5 * 32^-0.5; score[t] = sum_h relu(q_h . k_t) * w_h; causal
visibility: position t visible iff t < (query_pos+1)//ratio. Layer 20 (candidate source) also computes candidate blocks
(blocks of 8 positions, block score = max, the block holding the newest visible position always kept, top 2048 blocks);
index layers 24..36 mask scores to those candidate blocks. top-512 indices per query.
MoE: scores = sqrt(softplus(x.Wg)); pick top-6 of scores+bias; weights = scores[top]/sum * 1.5; expert: g=w1 x,
u=w3 x, u=clamp(u,-10,10), g=min(g,10), y=w2(silu(g)*u*weight); plus shared expert (same clamps, weight 1).
Engram (328-365, engram.py): hashes from token ids only (compressed via deepseek41.engram.token_map; pad =
token_map[2]); for ngram n in 2..4 and head k in 0..7: rolling xor of tok[pos-s]*mult[layer][s] (s=0..n-1), %
prime -> + head_offset -> row of per_layer_token_embd (global offsets from deepseek41.ple.head_offsets /
head_vocab_sizes / layer_multipliers; layer 1 uses heads 0..23, layer 14 heads 24..47). 24 rows x 256 -> 6144 ->
engram_kv -> key [4][5120] + value [5120]; gate_j = sigmoid(signed_sqrt(clamp|dot|>=1e-6)) with dot = sum(h_j*w_j*key_j)
* rsqrt(mean h_j^2+eps) * rsqrt(mean key_j^2 + eps) / sqrt(5120); h_j += gate_j * value.

## 4. Code layout and interfaces

Language: C++20 + Objective-C++ (Metal host code in .mm), Metal Shading Language 4 kernels. There is NO offline Metal
toolchain on this machine (`xcrun metal` fails; do not install it). CMake bundles all `src/kernels/*.metal` (local
`#include "x.h"` inlined) into `build/macseek.metal` via `tools/metalc` which compile-checks it with the system compiler
(newLibraryWithSource, Metal 4.0, fast math); `ms::Device` compiles the bundle once at startup. Errors show at build time.
Kernel function names must be globally unique across .metal files (prefix with your module, e.g. `mv_`, `attn_`). Build: `cmake -S . -B build -G Ninja
-DCMAKE_BUILD_TYPE=Release && ninja -C build` (use ccache). Run builds under `taskpolicy -b` while a model job runs.

```
src/core/      metal.hpp/.mm   Device/queue/library/pipeline cache, Buffer (MTLBuffer + offset + size), Encoder helper
               arena.hpp/.cpp  page-aligned allocations wrapped as no-copy shared MTLBuffers; residency set
               timer.hpp, log.hpp, env.hpp (MACSEEK_* env flags)
src/gguf/      gguf.hpp/.cpp   mmap-free parser of all splits: metadata map, tensor table {name,type,dims,file,offset,nbytes}
src/store/     weights.*       loads resident (dense) tensors into arena buffers, optional repack per kernel layout;
                               TokenEmbd = token_embd rows by pread (not resident by default)
               experts.*       expert cache: slot arena (one slot = gate+up+down of one (layer,expert), per-size-class
                               pools), pread pool (F_NOCACHE, 8-16 threads), policy.cpp (LRU / sampled LFU with aging /
                               static frequency-pinned), stats, ensure/prefetch/pin, leases released by command-buffer
                               completion, io_event (MTLSharedEvent) GPU waits, bulk stream ring for prefill
               engram.*        engram row fetcher: rows -> packed 144 B rows (dedupe, 8-way row cache, async batches)
               io_pool.*       pread thread pool (2 priorities);  memguard.* free RAM / swap / arena sizing
                               (full API + bench numbers: docs/modules/store.md)
src/kernels/   *.metal         all GPU kernels;  kernels.hpp/.mm  host-side `encode_*` functions (one per kernel family)
src/model/     ds41.hpp/.mm    model state (KV rings, compressed caches, indexer caches, compressor states, engram hash
                               history), forward_decode(token), forward_prefill(tokens chunk), save/load state snapshot
src/tok/       tokenizer.*     GPT-2 byte-level BPE from GGUF vocab+merges, pre-tokenizer `joyai-llm`, special tokens
               chat.*          DeepSeek V4.1 chat formatting (hard-coded, matches the GGUF jinja template)
               json.*          ordered JSON value/parser/serializer (Python json.dumps-exact), shared with server
src/server/    http.*          OpenAI-compatible HTTP/1.1 + SSE server; generation via ms::server::Backend (docs/modules/tok.md)
src/sample/    sampler.*       greedy / temp / top-p (GPU argmax for greedy)
src/app/       main.cpp        `macseek` CLI: tokenize, detok, chat-format, serve; gen, bench, logits, snapshot (hooks)
tests/         unit tests per module (ctest), CPU references in tests/ref/*.cpp
tools/         gguf_dump.py, oracle/ (llama.cpp logits dumper), quality.py, bench scripts, modellock.sh
bench/         microbenchmarks (membw, matvec, matmul, attention)
```

### 4.1 Core conventions
- Activations: F32 in device buffers unless a kernel documents otherwise (F16/BF16 allowed internally for prefill tiles).
- Weight matrix W for [in -> out] in GGUF is `out` rows of `in` elements, each row a sequence of quant blocks.
  Kernels take raw GGUF block bytes (or a documented repacked layout produced by store/weights).
- Host encode functions have the shape
  `void encode_xxx(Enc& e, const Args& a)` where `Enc` wraps one `MTLComputeCommandEncoder` in the current command
  buffer; they never commit or wait. One command buffer per decode token (or per prefill chunk) — no per-op sync.
- `Buffer { id<MTLBuffer> buf; size_t off; size_t size; }` — kernels bind `buf` at `off`.

### 4.2 Kernel families (kernels agents)
Quant kernels live in `src/kernels/quant.hpp` (namespace `ms::q`, full docs: `docs/modules/quant.md`):
1. `encode_matvec(e, qtype, W, rows, cols, x, y, accumulate=false, groups=1)`  y[rows] (+)= W[rows x cols] . x  (decode,
   n=1, F32 math). `groups=8` = block-diagonal attn_output_a (group g reads x[g*cols..]).
2. `encode_matvec_n(e, qtype, W, rows, cols, X, Y, n, accumulate=false, groups=1)` n in 1..16; x staged as half;
   n >= 8 (groups==1) is routed to the tensor-API tile path.
3. `encode_matmul(e, qtype, W, rows, cols, X[n x cols], Y[n x rows], n, accumulate=false)` prefill (any n; tuned for
   n >= 32): Metal 4 tensor API (MPP matmul2d), half operands / F32 accumulate. `encode_f32_to_f16` +
   `encode_matmul_h` let one converted X16 feed several matmuls. n >= 512: weight dequantized once to half + dense GEMM.
4. `encode_moe_decode(e, MoeDecodeArgs{types, slot[6]{gate,up,down Buffers}, weights(device F32[6]), shared slot, x, y,
   scratch})`: 2 dispatches (fused gate+up+clamp+swiglu*w for all 7 experts; down for all experts accumulated into y).
5. `encode_moe_prefill(e, MoePrefillArgs{expert-sorted tok/wt, host offs, slot_of, slots, shared, X, Y, scratch})`:
   gather -> grouped fused gate/up GEMM -> grouped down GEMM -> deterministic reduce into Y (accumulates).
6. Elementwise/small: rmsnorm, rope (interleaved, table-driven), hc_mixes (F32 20480->24 matvec + sinkhorn), hc_pre,
   hc_post, router (sqrtsoftplus + bias top-6 + normalize), engram gate, compressor pool, fp8/fp4 quant-dequant sims.
7. Attention: indexer scoring (decode: split over positions; prefill: tiled q x k with causal mask), top-k (512 of up to
   200K; candidate-block select), sparse attention decode (64 heads x 512, rows = window ∪ selected, sink), sparse
   attention prefill (per query its own window + its own top-k).
Supported qtypes: F32, F16, BF16, Q8_0, Q5_K, Q6_K, IQ1_S, IQ2_XXS, IQ4_NL (engram rows, CPU or GPU dequant).
IQ1_S/IQ2_XXS need the iq1s_grid / iq2xxs_grid / ksigns tables: generate them from the published format description
(the grid values are data, not code — taking the table values is fine) and verify the CPU dequant against llama.cpp's
`llama-quantize`-free path: `tools/oracle/dequant_check` (links libggml from Homebrew, test-only).

### 4.3 Model state at 200K (all GPU-resident, small)
window KV: 40 x 128 x 512 F16; compressed KV: layers 2,8,14 -> 100K x 512; layer 20 -> 200K x 512 (F16; fp8/fp4 sim
values are exact in F16); index keys: F16 128-dim per compressed position for the 4 sources; compressor partial-group
state (f32); engram token history. Exact attention layouts, API and switches: docs/modules/attn.md (`ms::attn::State`,
`encode_layer`, `persistent_state()` lists the buffers/bytes to snapshot). `save_state(path)` / `load_state(path)` must round-trip
bit-exactly (enables the 1-minute loop: load 200K snapshot, decode 128, time a 2K prefill chunk).

## 5. Memory plan (64 GB)
OS/apps headroom >= 8 GB (never swap: check `sysctl vm.swapusage` before/after every run). Resident dense weights
~6.5 GB; token_embd stays mmap/pread (lookup only). Activations/scratch for prefill chunk <= 2 GB. State at 200K < 1 GB.
Expert slot arena = remainder, target ~38-42 GB (~40% of all experts; 40 GiB = 141 slots/layer = 36.7%), sized at
startup with `ms::recommend_arena_bytes(planned_other)` (available - 8 GiB headroom - what is still to be allocated).
Engram row cache <= 1 GB.

## 6. Correctness gates
- (A) kernels: every kernel has a CPU reference (tests/ref) and a ctest with abs/rel tolerances on random data and the
  real shapes above; quant dequant checked bit-exact against libggml's dequantize (test-only link).
- (B) algorithm: chunked prefill == one-shot prefill (within fp tolerance); prefill-then-decode == prefill of n+1;
  save/load state round-trip bit-exact; compressor partial groups across chunk boundaries.
- (C) model: `tools/quality.py` — fixed prompt set (tools/prompts/*.txt), compare full logits at every position
  vs the pinned baseline: top-1 agreement and mean KL. First vs the llama.cpp oracle (pins "baseline v0"), then each
  change vs the engine's own pinned baseline. Quick proxy per change; full gate (long-context needle at 200K, a few
  task prompts, sustained thermals: 10-min decode) every N wins / before commit.
- fp8/fp4 activation-quant simulation (reference behaviour) is a switch: `MACSEEK_ACTQ=0|1`; default = whichever
  matches the oracle best on the gate.

## 7. Speed loop (after integration)
Quick check (~1 min): load 200K snapshot -> decode 128 tokens (tok/s, ms/token split) -> 2048-token prefill chunk at depth
200K (tok/s) -> quality proxy on the small prompt set. Keep a change only if it wins beyond noise (>= 2 interleaved
runs); otherwise revert. Report every result as % of roofline:
decode roofline = resident_bytes/BW + expected_miss_bytes/SSD_BW (+ attention-at-200K bytes); prefill roofline = FLOPs/peak.
