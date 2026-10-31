# Module `arch` — model-specific small kernels

Hyper-connections (hc_post / hc_mixes + Sinkhorn / hc_pre + RMSNorm), MoE router, engram (CPU hashing, GPU gate,
IQ4_NL row dequant), token embedding (Q6_K row dequant), argmax / Gumbel / top-k sampling, log_softmax and RMSNorm.
Decode (1 token) and prefill (n tokens) variants. All kernels have CPU references and ctests.

Files: `src/kernels/arch.hpp|.mm` (host API, namespace `ms::arch`), `src/kernels/arch_{common.h,hc,router,engram,sample}.metal`
(Metal, all functions prefixed `arch_`), `src/model/engram_hash.hpp|.cpp` (CPU hashing), `tests/ref/arch_ref.*` (CPU
references), `tests/ref/arch_testutil.*` (test helpers: small pread of real tensors / GGUF metadata, skipped when the
model is absent), `tests/test_arch_*.{mm,cpp}`, `bench/arch.mm`, `tools/engram_hash_check.py`, `tools/arch_dequant_check.cpp`.

## Conventions

- `encode_*` never commits or waits. **Barriers:** each `encode_*` puts barriers between its *own* dependent dispatches.
  The caller puts `e.barrier()` between dependent `encode_*` calls (the encoder is concurrent).
- Activations are F32. Buffers must be 16-byte aligned (the decode hc step uses float4 loads).
- `arch::Scratch::create(max_tokens, max_sample_rows)` holds the partials and the zeroed atomic counters that the
  single-dispatch kernels use: the "last threadgroup to finish" does the tail work, then resets its counter. Each
  kernel family has its own counter slot. Two concurrent (no barrier between them) dispatches of the same family need
  two Scratch objects.
- Runtime switches live in `arch::options()`. They are read once from env and tests/bench may change them:
  `MACSEEK_ARCH_HC=fused|split|rows`, `MACSEEK_ARCH_HC_T=128|256|512|1024` (default 512),
  `MACSEEK_ARCH_HC_PREFILL=mma|mma8|rows`, `MACSEEK_ARCH_ROUTER_G` (default 96). The slow-but-simple paths stay
  available as fallbacks.

## API

### Hyper-connections: `encode_hc_step(e, HcStep, scratch)`
One sublayer boundary (SPEC §3), for n tokens:
```
h_new   = x_sub ? post[i]*x_sub + sum_j comb[j][i]*h_in[j] : h_in      (model.py hc_post orientation, -> h_out)
mix_out = fn ? hc_split_sinkhorn(fn . h_new * rsqrt(mean(h_new^2) + 1e-20), scale, base) : -
x_out   = rmsnorm(sum_j mix_prev.pre[j] * h_new[j]) * norm_w            (eps 1e-20)
```
- The mix layout is `[n][24]` = pre[4], post[4], comb[4][4] (comb[j][k] at 8+4j+k). `mix_prev` provides post/comb
  of the previous hc_mixes **and** the lagged pre, which is exactly SPEC's lag: the FFN step uses attn mixes (attn_post,
  attn_comb, attn_pre) and the next attention step uses ffn mixes. So two mix buffers ping-pong, and `mix_out` must
  not alias `mix_prev`.
- Layer 0: `mix_prev` = `fill_identity_mix` (pre = [1,0,0,0]), no `x_sub`. Final head: no `fn`,
  `norm_w = output_norm`. An empty `x_out` skips hc_pre + RMSNorm.
- Engram layers (1, 14) sit between layer L-1's FFN and layer L's attention, and take three calls:
  (A) `{x_sub = ffn out, mix_prev = ffn mixes of L-1, no fn, no x_out}` does hc_post only;
  (B) `encode_engram_gate(h)`;
  (C) `{no x_sub, mix_prev = the same ffn mixes (lagged pre), fn = hc_attn of L, x_out}` does mixes + pre + norm.
- `h_out` may equal `h_in`.
- Decode (n == 1), `fused` (default): **one dispatch** (5120/T groups; every simdgroup recomputes h_new in registers so
  the fn loads have no barrier in front of them, and the last group reduces partials in a fixed order, runs Sinkhorn on
  16 lanes with shuffles and normalizes x). `split` = the same kernel + a 1-group finalize (2 dispatches).
  `rows` = the prefill path with n=1 (slow, reference only).
- Prefill: `arch_hc_post_pre_n` (1 group per token: post + pre + norm + rstd) followed by the mixes GEMM on simdgroup
  matrices with the Sinkhorn epilogue fused in: 32-token tiles (n ≥ 32), 8-token tiles (8 ≤ n < 32), per-token rows
  kernel (n < 8). The last tile shifts back to stay in bounds.
- `encode_hc_mean(h, out, n)`: mean over the copies (dspark main_hidden).

### Router
- `encode_router_fused(x, W=ffn_gate_inp [384][5120] F32, bias=exp_probs_b, ids, w, scratch)`: decode. The F32
  matvec + sqrt(softplus) + top-6 of score+bias + `w = score/(sum+1e-20)*1.5` run in **one dispatch**. ids are int32
  [6] and w is f32 [6], descending by score+bias, ties go to the lowest id. Logits are also left in `scratch.router_logits`.
- `encode_router_topk(logits [n][384], bias, ids, w, n)`: any n (verify blocks and prefill, logits from a matmul).
- `encode_route_lists(ids, w, n, RouteLists)`: prefill grouped-GEMM tables. It is deterministic (ballot + prefix
  popcount, no atomics, so no zeroing is needed). It produces `counts[384]`, `offsets[385]`, and per sorted slot
  `entry` (= t*6+k), `tok`, `w`, plus `slot_of[t*6+k]` for the scatter back. The order is by expert, then by entry index.
- CPU readability: all buffers are shared storage, so `ids.cpu()` is valid once the GPU work that wrote it has
  completed. The expert cache needs ids in the middle of a token, so the integrator has to split the command buffer
  (or signal an MTLSharedEvent) after each layer's router. That is the only sync point this module implies.

### Engram
- CPU hashing, `ms::EngramHashParams::init(token_map, pad_token_id, layer_multipliers, head_vocab_sizes,
  head_offsets)` takes the raw GGUF arrays. `ms::EngramHasher` is a streaming hasher: `hash(tokens, n, out[n][48],
  mask)`. The output is 2 layers x 24 global row ids into `per_layer_token_embd`; layer 1 is columns 0..23 and
  layer 14 is 24..47, ordered by ngram (2,3,4) and then head 0..7. The state is the last 3 compressed tokens + pos,
  and it carries across prefill chunks and decode steps. `state()/set_state()` serve snapshots and speculative
  rollback; `hash_from(state, …)` hashes draft tokens without mutating anything; `set_history(last tokens, n, len)`
  rebuilds the state from token ids. `mask` false = dead token (image span): it blocks n-grams, as in engram.py.
- `encode_engram_gate(h [n][4][5120] in/out, kv [n][25600], w = engram_gate [4][5120], n, scratch, gate_out)`. kv is
  the engram_kv matvec output (quant module, Q6_K 6144→25600): key [4][5120], then value [5120]. One group of 1024
  threads per token.
- `encode_iq4nl_rows(src, row_stride, dst, n_rows)`: 144-byte IQ4_NL rows → F32 [n_rows][256]. The format is 8
  blocks of {f16 d, u8 qs[16]}: value j = d·kv[qs[j]&15], value j+16 = d·kv[qs[j]>>4], with kv = kvalues_iq4nl. For
  one token, pass its 24 rows in hash-column order and the output is the 6144-vector `embed(hash_ids).flatten(-2)`.

### Embedding
`encode_embed_q6k(rows, row_stride, h, n, copies=4)`: raw Q6_K token_embd rows (4200 B, from pread/mmap into any
shared buffer) → `h [n][copies][5120]`. copies = 4 does the hc expand.

### Sampling / logits
- `encode_argmax(logits [rows][V], rows, V, out_idx, out_val, scratch, temperature=0, seed=0)`: one dispatch (64
  groups + last-group merge), ties go to the lowest index. With `temperature > 0` it does Gumbel-max sampling, i.e.
  argmax(l/T + Gumbel(hash(i, seed, row))). That is exact multinomial sampling from softmax(l/T) over the full vocab
  and matches the reference `sample()`. It has the same cost as greedy.
- `encode_topk(logits, rows, V, k<=64, temperature, out_v [rows][64], out_i, out_lse, scratch)`: one dispatch. It
  returns the top-k raw logits (descending) and lse = logsumexp(l/T) over the full row, so the CPU knows the exact
  probabilities p_i = exp(v_i/T − lse) and the mass outside the k.
- `sample_top_p(vals, ids, k, lse, T, top_p, u)` (CPU): the nucleus is renormalized. If the cumulative mass inside
  the top-64 never reaches top_p, the sampler takes all 64 and truncates the rest (rare; the lse shows how much mass
  is outside).
- `encode_log_softmax(in, out, rows, V)` (in-place OK): full log-probs for the quality tool (top-1 + KL).

### RMSNorm
`encode_rmsnorm(x, w (optional), y, rows, dim, eps, in_stride, out_stride)`. One threadgroup per row by default; one
simdgroup per row when dim ≤ 128, or when dim ≤ 1024 and rows ≥ 16. Strides allow sub-vectors inside larger rows.

## Verified facts (from the GGUF + reference, see tests)

- `hc_*_fn` is stored as 24 rows of 20480 (dims [20480, 24]). Mixes = fn·flatten(h). The real blk.0 hc_attn scales
  are ~0.06 / 0.012 / 0.06 and the bases are large (e.g. -19.6), so the mixes are dominated by base.
- `engram_gate` [5120, 4] = 4 rows of 5120, one per hc copy. It is the product `q_weight*k_weight`: the reference only
  ever uses the product and the GGUF has no other engram q/k tensor. Real values: min −0.39, max 43.9, mean 0.008.
  This is still to be confirmed against the oracle at integration (gate C).
- Engram metadata (`tools/engram_hash_check.py`, stdlib + deterministic Miller–Rabin) matches
  `EngramLayout.from_args`. `head_vocab_sizes` = the 48 consecutive primes above 16,000,000, in [layer][ngram][head]
  order. `head_offsets` are **global** offsets into the 768,022,850-row table (per-layer cumsum + layer base
  384,006,168). The layer totals equal config `engram_num_embeddings`. `layer_multipliers` = [layer][shift] and equal
  `np.random.default_rng(10007*layer).integers(0, bound, 4)*2+1` (checked once with numpy in a scratch venv; the
  script runs this check automatically when numpy is installed). `token_map` max+1 = 99092, pad = token_map[2] = 2.
- The C++ hasher equals a literal port of `NgramHashState.forward` on 200 random sequences with prefill/decode
  splits (`tools/engram_hash_check.py --bin build/test_arch_engram_hash`). Golden values are embedded in the ctest.
- Q6_K and IQ4_NL dequant: the GPU matches the CPU reference bit-exactly, and the CPU reference matches libggml
  `to_float` bit-exactly on 50 token_embd rows, 50 output rows and 102 per_layer_token_embd rows
  (`tools/arch_dequant_check.cpp`, test-only, links Homebrew libggml-base; the build line is in the file header).

## Tests (`ctest`; real-data parts read < 10 MB via pread and are skipped if the model is absent)

| test | covers |
|---|---|
| test_arch_hc | decode fused (T = 128/256/512/1024) / split / rows, with/without post and mixes; 12-step chained decode vs the CPU chain; 300-dispatch race/determinism stress (bit-identical); prefill rows / mma8 / mma32 for n = 2, 7, 8, 31, 32, 37, 256, plus forced variants; real blk.0 hc_attn weights with real token embeddings |
| test_arch_router | top-6 for n = 1, 5, 2048 incl. the softplus threshold/tiny branches (near-ties accepted only if the key gap < 1e-5); route lists bit-exact vs CPU for n=2048; fused decode router for G = 24/48/96/384, synthetic and real blk.0 ffn_gate_inp + exp_probs_b; 300-dispatch stress |
| test_arch_engram | engram gate n = 1/3/33 with real engram_gate weights; IQ4_NL rows bit-exact (synthetic + real table rows incl. the first/last row); Q6_K embedding bit-exact (synthetic + real token_embd rows), 4 copies and 1 copy |
| test_arch_sample | argmax incl. ties, edges, chunk-boundary ties, multi-row, 300-dispatch stress; Gumbel sampling chi² vs softmax(l/T); top-64 + lse exact vs CPU (incl. ties); CPU top-p frequencies; log_softmax (incl. in place); rmsnorm dims 128…20480 strided |
| test_arch_engram_hash | streaming/chunked/masked hashing vs an independent whole-sequence implementation; state/hash_from/set_history round trips; real metadata (consecutive primes, global offsets, multipliers); golden hashes from the Python port. `--dump tokens split` mode is used by the Python cross-check |

Tolerances: the observed worst errors are < 3% of (atol 1e-5..2e-5, rtol 1e-5..2e-4). Mixes go through Sinkhorn in
F32 with fast-math exp/div.

## Bench (`build/bench_arch [N] [trials]`)

GPU time per dispatch, measured as N dependent dispatches separated by barriers in one command buffer (the shape of a
decode token), median of 7. Weights are distinct buffers (80 hc fn copies, 40 router matrices), so they stream from
DRAM as they would in a real token. M5 Pro, numbers are noisy ±20% from GPU clock states:

| kernel (decode) | per call | per token |
|---|---|---|
| floor: trivial dispatch + barrier | 1.0–3.8 µs | — |
| reference: plain stream read 2.0 MB / 7.9 MB | 8.9 / 30.4 µs | — |
| **hc_step fused, T=512** (post + mixes 2 MB F32 + Sinkhorn + pre + norm) | **12.3 µs** (178 GB/s eff.) | ×80 ≈ 1.0 ms |
| hc_step fused, cache-hot fn | 8.8 µs | |
| hc_step split (2 dispatches) | 14.3 µs | |
| hc_step final head (no mixes) | 5.1 µs | ×1 |
| **router_fused G=96** (7.9 MB F32 matvec + top-6) | **34.6 µs** (227 GB/s) | ×40 ≈ 1.4 ms |
| router_topk n=1 from logits | 2.2 µs | |
| engram_gate n=1 | 7.9 µs | ×2 |
| iq4nl_rows 24 rows | 1.2 µs | ×2 |
| embed_q6k n=1 → 4 copies | 1.7 µs | ×1 |
| argmax greedy (V=129280) / + Gumbel | 5.2 / 5.2 µs | ×1 |
| top-64 + lse | 95 µs | ×1 (top-p only) |
| log_softmax 1 row / 16 rows | 29 / 60 µs | quality tool |
| rmsnorm 5120 / 1280 / 512 (1 row) | 2.7 / 2.2 / 2.2 µs | |

The hc step and the router are at the stream-read reference for their sizes, so they are DRAM/latency bound, not
compute bound. The total arch cost per decode token is ≈ 2.5 ms (80 hc steps + 40 routers + small kernels).

Prefill (per token, chunk 256 / 2048): hc_step (post_pre + mma32) 1.16 / 1.34 µs, mma8 1.28 / 1.31, rows 1.80 /
2.39. The ×80 total is ≈ 0.1 ms/token. route_lists n=2048: 90 µs; router_topk n=2048: 16 µs; engram_gate n=64: 46 µs.

## Limits / follow-ups

- Prefill hc: `post_pre_n` is memory bound. It reads and writes h, [n][20480] F32 = 168 MB per 2048 chunk, before the
  GEMM reads h again. Fusing the post into the GEMM's A-tile load would save one pass (~0.4 ms per chunk per step).
- top-k (95 µs) does 64 rounds of threadgroup argmax. That is fine for top-p sampling. Temperature-only sampling
  should use Gumbel `encode_argmax` (5 µs).
- `encode_router_fused` handles 1 token only. Verify blocks (n ≤ 16) use the matvec_n module + `encode_router_topk`.
- The GPU engram gate has no token mask: image-span tokens must skip `encode_engram_gate` (text-only engine). The
  hasher supports masks.
- The reference casts dequantized engram rows to bf16 before `wkv`. That is not simulated: IQ4_NL values are passed
  as exact F32. Same for the reference's fp8 activation simulation inside the engram `Linear`.
- `rows` decode hc mode is a slow fallback (one threadgroup reads 2 MB; 138 µs). Use `fused` or `split`.
