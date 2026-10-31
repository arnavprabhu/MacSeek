# Integration design (coordinator) — decode / prefill schedule

## Decode token schedule (batch 1)

Routing is data-dependent and misses must be read from SSD by the CPU, so each layer has exactly one
GPU->CPU handoff (the router result). Everything else stays on the GPU in long command buffers.

```
CB_0:  embed row (CPU pread of token_embd row -> GPU dequant) ; engram rows were fetched by CPU before CB_0
       layer 0: hc_mixes(attn) ; hc_pre+norm ; attention ; hc_post ; hc_mixes(ffn) ; hc_pre+norm ;
                shared expert (no routing needed) ; router -> ids/weights in shared memory ; SIGNAL(L0)
       (speculative) router of layer 1 applied to the same normed x -> predicted ids -> SIGNAL(pred L1)
CPU:   spin on shared flag for L0 (no waitUntilCompleted) ; experts.ensure(0, ids) (hits = free; misses = parallel
       F_NOCACHE preads) ; experts.prefetch(1, predicted ids)
CB_1:  wait for slot fences ; moe_decode(layer 0, slots) ; hc_post ; layer 1 ... router ; SIGNAL(L1)
...    40 handoffs per token
CB_40: output head (Q6_K matvec) ; argmax / sampling on GPU ; token id to shared memory
```

- Cost of a handoff: commit + CPU spin + encode next CB ~ 20-60 us -> 40 x ~= 1-2.5 ms/token. Measure (`decode-prof`).
- Shared expert of layer L runs BEFORE the router handoff, so the GPU has work while the CPU checks the cache.
- Only routed experts wait on the CPU. On a full hit the handoff is the only overhead.
- Prefetch predictor: next layer's router on the current layer's normed FFN input (hidden state changes slowly
  through the residual stream). Measure accuracy (hit@6) before relying on it; switchable.

## Prefill chunk schedule (n tokens, n = 512..8192)

Per layer: attention for the whole chunk (matmul kernels), router for all n tokens -> CPU builds per-expert
token lists -> experts used by the chunk are streamed: cached ones run immediately, misses are streamed by the
bulk-stream ring (triple buffered) while the GPU runs grouped GEMMs on what is resident. Engram rows for the
whole prompt are fetched up front (prompt ids known) in parallel with layer 0.

Big chunks amortize expert streaming (each chunk touches ~all 384 experts per layer): measure tok/s vs chunk
size; at 200K depth the indexer prefill (quadratic part) scales with depth, so time chunk = attention-side +
expert-side, and overlap them where independent.

## State snapshot

`snapshot save` after prefilling the 200K corpus: window rings, compressed KV + index keys for source layers,
compressor partial groups, engram token history, position, last hc stream is NOT needed (per-token). Must be
bit-exact on reload; quick loop = load snapshot -> decode 128 -> 2048-token chunk at depth -> quality proxy.

## Memory at runtime
dense ~6.5 GB + state ~1 GB + scratch ~2 GB + expert arena (sized from free RAM with >= 8 GB headroom).
