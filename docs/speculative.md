# Speculative decoding with the dspark draft: measured, not built (2026-10-31)

Measurements:
- Draft acceptance (stock llama.cpp `--spec-type draft-dspark`, n_max 4, greedy, short prompts): 0.39
  (85/220), mean accepted length 2.5 tokens per verify pass (logs/spec/accept_n4.out).
- Expert union per layer over k consecutive decode tokens at 200K (real routing, logs/dbg/u.bin):
  k=1 6.0, 2 10.3, 3 14.0, 4 17.4, 5 20.5 -> a 4-token verify reads ~2.9x the routed-expert bytes of one token and
  causes ~2.9x the SSD misses.
- Current decode at 200K: ~56 ms/token = ~37 ms GPU + ~16 ms SSD stall (hit rate ~92% with a 33 GB arena).

Cost model for one verify pass (anchor + 3 drafts): GPU ~37 + 11 (extra expert bytes) + 3 (attention x4) ms,
SSD stall ~2.9 x 16 = ~46 ms, draft ~10 ms -> ~107 ms for ~2.3 accepted tokens = ~46 ms/token (1.2x). Keeping the
7.6 GB draft resident shrinks the expert arena by ~23% -> more misses -> ~1.1x. Not worth the complexity now; the SSD
miss term dominates. Revisit if the arena grows (more RAM) or misses get hidden better.
`macseek bench --verify K` measures the cost of K-token passes through the multi-token path.

## Exact dspark speculative decoding (branch coder-spec, state 2026-11-02, WIP)

Target: GSQ-RCO Coder IQ2_XXS. Drafter: `models/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (arch `dflash`, 3 DSV4
stages, block 5, 128 experts top-3 MXFP4 + Q8_0 dense, Markov head; conditioned on the mean-over-hc attention inputs of
target layers 37/38/39). Default path `<source tree>/models/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, env
`MACSEEK_DRAFT` overrides. Off by default: `MACSEEK_SPEC=K` or `--spec K` on `gen` / `bench` (K drafts per pass).

Pieces:
- `Model::verify(toks, k, next, logits)` / `verify_keep(m)` (src/model/ds41_verify.mm): k tokens through the decode
  table path at once. Every per-token value is bit-identical to `decode()`: hc steps, router, engram gate, attention
  kernels and routed MoE use the decode kernels per token; dense matvecs and the shared expert use `sv_*` k-token kernels
  (src/kernels/spec.metal) that reproduce q_mv's per-lane arithmetic (tests/test_spec_kernels: all bit-identical).
  Q5_K hoists raw block words laundered through a per-token runtime zero (fast-math otherwise re-associates the
  scale products across tokens); Q6_K hoisting is never bit-identical -> generic per-token dot.
  New window rows go to a staging chunk; `verify_keep` commits the kept rows and repairs compressor partial state;
  the engram hasher is rewound. Needs the default decode config (`Model::verify_supported`).
- `macseek spec-check --ids F [--n N --k K --reject]`: decode vs verify bit comparison of tokens AND logits.
  Results: code_python n48 k4, code_cpp n96 k5 --reject, needle 8K prefix n240 k3 --reject: 0 token / 0 logit-row diffs.
- `DSpark` (src/model/dspark.{hpp,mm}, kernels/dspark.metal): GPU drafter mirroring llama.cpp dflash.cpp
  graph_dsv41 + common/speculative.cpp sample_from_anchor (128-row feature ring, non-causal block attention with
  sinks, MXFP4 routed experts with GPU top-k, target output head, Markov chain). Not yet checked logit-by-logit vs
  llama.cpp; its acceptance matches llama.cpp's (below), so it is functionally right.
- `Model::spec_step` (greedy loop), `bench --spec K [--feed N] [--vexact 1,2,..] [--out-ids F]`, `gen --spec K --out-ids F`.
  The drafter's ~7.4 GiB comes out of the expert arena.

Measured:
- llama.cpp acceptance (greedy, 128 tokens, accepted/drafted): code_python n1 57/72, n2 75/110, n3 83/144, n4 88/174;
  code_cpp n1 14/18 .. n4 21/44 (stops at EOS early); chat n1 45/63, n3 67/141, n4 79/202; json: 0 (prompt ends in EOS).
- verify(k) at depth 259,900 (bench --vexact, HEADROOM 5, decode 45.4 ms/token): k=1 40.6 ms, k=2 62.5 (1.38x),
  k=3 82.9 (1.83x), k=4 111 (2.45x), k=6 170 (3.74x). The dense k-token kernels scale ~0.6-0.9 decode per extra token
  (q_b/o_a/o_b Q5_K k=4 = 2.8-3.1x decode; Q8_0 1.3x); routed MoE adds the union of the tokens' experts.
- Engine end to end, short prompt code_python: K=3 0.71 accepted per draft, 3.13 tokens/step, but 18.6 vs 20.7 tok/s
  plain (smaller arena, verify(4) ~2.8x decode); output ids identical to plain gen (cmp).
- 8K prefill + 4K decode (needle prompt, bench --feed 8192 --decode 4096): plain 25.4 tok/s (arena 40.6 GiB);
  spec K=2 27.0 tok/s (0.882 accepted/draft, 2.76 tok/step, step 102 ms = draft 8.3 + verify 93.8 incl. io stall 12.7,
  arena 32.1 GiB); spec K=3 20.9 tok/s (0.694, 3.08 tok/step, verify 137 incl. io stall 36.8, arena 30.3 GiB).
  OPEN BUG: the three runs' tok_hash differ (plain 2722d928.., K2 fdeaf336.., K3 53ca090c..), although spec-check on the
  same 8K prefix is exact for 240 tokens. Find the first diverging token with `bench --out-ids` (plain vs --spec 2)
  before trusting any spec speed number.
- 260K single run (before the arena fix): 8 tok/s - the drafter was not subtracted from the arena -> swap. Fixed
  (arena = recommendation before loading the drafter - drafter bytes); not re-measured.

Next steps: (1) the tok_hash divergence over 4K tokens; (2) cheaper verify: k-token routed MoE over the expert union,
faster k-token Q5_K/Q6_K (the dense slope is the main cost); (3) drafter cost 8-14 ms/step (head over the full vocab,
naive MXFP4 kernels); (4) adaptive K (K=2 is the only measured win so far: 1.06x at 8K); (5) re-measure at 260K
(prefill a feature tail with --feed), then decide defaults (keep MACSEEK_SPEC=0 unless a measured win at 260K);
(6) logit check of the drafter vs llama.cpp; ctest not run on this branch yet beyond test_spec_kernels.
