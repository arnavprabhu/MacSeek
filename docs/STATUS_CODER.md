# GSQ-RCO Coder support: status (2026-11-02, end of session)

Model: ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF (192 experts, IQ1_M / IQ2_XXS / IQ2_XS experts, Q5_K
dense/shared, BF16 routers, Engram as split 6 via symlink). Lives in models/ (gitignored). Draft for speculative
decoding: models/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf.

## Branches
- coder: integration branch (everything below merged except coder-spec).
- coder-quant (IQ1_M / IQ2_XS kernels, Q5_K shared), coder-pools (region-pool expert cache), coder-dec (decode kernels),
  coder-moep (merged gate|up MoE prefill GEMM): merged into coder.
- coder-spec: speculative decoding with the dspark drafter (work in progress, see docs/speculative.md there).

## Measured (M5 Pro 64 GB)
- Quality vs llama.cpp oracle (logs/oracle/v0, tools/gate8.sh): top-1 ~98.7-99.0%, KL mean ~0.0015.
- 262K needle (tools/prompts/long_needle_d50_260k, 259,960 tokens): PASS.
- Prefill: 363 tok/s full 260K at session start; 8K-chunk runs later reached ~490-515 tok/s.
  Quick check (tools/quick8k.sh: 8K prefill + 4K decode, ZALL off): prefill 516 tok/s, decode 24.2 tok/s.
- Decode at depth 259,900: ~23.1-23.7 tok/s (~37 ms GPU + 4-7 ms SSD stalls, hit rate ~98%).

## Open bug (why MACSEEK_PF_ZALL defaults to 0)
With MACSEEK_PF_ZALL=1 (layer-wide Z + one reduce) and the merged q_moe_rgu2 kernels, prompt `chat` gives a wrong
logit at position 20 (KL 20.6 vs oracle). ZALL=0 is correct; ZALL=1 with MACSEEK_MOEP_RGU48=48x16k32s4 or
RGU64=64x16k64s4 (unmerged) is also correct. Not caused by group order (MACSEEK_PF_ORDER), group size, pin-aware
groups, NODRAIN, DGU2, or a Z-row overflow into the shared rows (MACSEEK_PF_ZGAP). Kernel unit tests (incl. new
short-prompt cases) find merged == unmerged. Token 20 carries the largest MoE input in every late layer
(MACSEEK_DBG_XMAX=1: max|x| up to ~50) -> suspect a numerically fragile path for that token (e.g. half range in H /
the SwiGLU clamp, or an accumulation-order sensitive value), not a memory race. Next: dump layer-by-layer Y for token
20 with ZALL 0 vs 1 and find the first layer that diverges.

## Next steps
1. Fix the ZALL bug above, then re-enable ZALL (it was +11% prefill).
2. Finish / measure speculative decoding (coder-spec), merge if it wins at 260K.
3. Remaining decode gap to the bandwidth floor: routed MoE +1.8 ms, sparse attn +1.6, barriers 1.6, hc +0.85,
   indexer +0.8 (docs/modules/dec_gpu.md).
4. Full 262K benchmark at the end (snapshot + bench --decode, prefill timing of the whole prompt).
