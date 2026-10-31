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
