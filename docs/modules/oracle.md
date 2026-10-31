# Module: oracle (correctness oracle + measurement harness)

Test-only tooling. Stock Homebrew llama.cpp (0.8.0, build 11960, commit 8642541b2, ggml 0.28.0) is linked/run here
ONLY as the logits oracle and the speed baseline. No llama.cpp code enters the engine.

## 1. Working llama.cpp config (use for every dump and every baseline)

Default full Metal offload (`-ngl 99`, mmap) **fails**: `ggml_metal_synchronize: command buffer failed ... Insufficient
Memory (00000008)` after ~300 s (coordinator sanity run, `logs/baseline/llbench_sanity.log`, max RSS 44.8 GB).

Working config: **`-ngl 99` + all MoE expert tensors on CPU** (`--cpu-moe` / `-ncmoe 99`; in oracle_logits a tensor
buft override `\.ffn_(up|down|gate)_(ch|)exps` -> CPU), **mmap on** (default load mode). Placement it produces:
Metal 6.3 GiB (`MTL0_Mapped`, attention/shared experts/router/hc/indexer/compressor/head), CPU_Mapped 210.8 GiB
(experts 107 GiB + engram table `per_layer_token_embd` 103 GiB with llama.cpp's "lazy read" + token_embd).
`--load-mode none` is NOT an option (would copy 216 GiB into anonymous memory). For large batches llama.cpp offloads
the CPU-resident expert matmuls to the GPU (op offload; inferred from 59 s CPU user time in a 931 s prefill run), so
every prefill ubatch streams the experts through page faults (~55-75 s per pass, ~1-1.7 GB/s effective) — prefill speed is set by the ubatch size, not by compute.
Decode runs the experts on the CPU (threads matter; 12 is best).

Measurement wrapper: `tools/oracle/measured.sh <log> <cmd...>` = main checkout's `tools/modellock.sh` + `/usr/bin/time -l`
+ `sysctl vm.swapusage` and `vm_stat` (free pages, pageins) before/after. Always use it.

## 2. Tools

| path | what |
|---|---|
| `tools/oracle/oracle_logits.cpp` | logits dumper (links libllama). `tools/oracle/build.sh` -> `build/oracle/oracle_logits` |
| `tools/oracle/measured.sh` | lock + time + swap/pagein snapshots |
| `tools/oracle/bench_baseline.sh <label> [llama-bench args]` | llama-bench with the working config -> `logs/baseline/<label>.{log,jsonl}` |
| `tools/quality.py` | compare two dumps (or two dirs): KL(base‖test), top-1, top-5 overlap, prompt NLL; gates |
| `tools/needle_check.py run/grade` | long-context needle gate (llama.cpp run or grade any json with `gen_text`) |
| `tools/prompts/make_prompts.py` | (re)builds the 8 short prompts (.txt/.ids, manifest.json) |
| `tools/prompts/make_long.py` | (re)builds the long needle prompts (`--target`, `--depth`) |
| `tools/oracle/v0_manifest.json` | sha256 + config of the pinned v0 dumps |

### oracle_logits
```
build/oracle/oracle_logits -m SPLIT1 (--ids F.ids | --text F.txt | --prompt S | --list L) [--out F.logits]
   [--topk 64] [--last N] [--gen N] [--json F] [--ctx N] [--batch 2048] [--ubatch 512] [--threads 6]
   [--ngl 99] [--cpu-moe 1] [--mmap 1] [--fa auto] [--swa-full 0|1] [--ignore-eos] [--tokenize-only --dump-ids F] [--quiet]
```
- Text is tokenized with `add_special=false, parse_special=true` (prompt text carries its own `<｜begin▁of▁sentence｜>`).
- `--list L`: lines `IDS_FILE OUT_FILE`; one model load, KV cleared between jobs (load is ~25-30 s).
- `--tokenize-only` loads the vocab only (no weights, ~4 s, no lock needed).
- Every output gets `<out>.json`: config, load time, prefill time, per-chunk tok/s, generated ids/text, decode time.
- Determinism: two separate processes give **bit-identical** logits (v0 vs v0_rerun: KL 0, top-1 100% on all 2707
  positions), also with a different n_ctx (144-token chat run alone vs in the list).

### .logits format v1 (little-endian)
```
char magic[4] = "MSLG"; u32 version=1; u32 n_tokens; u32 n_vocab; u32 mode (0 full, 1 topk); u32 k;
u32 first_pos; u32 n_pos;                      // 32-byte header
i32 tokens[n_tokens]
mode 0: f32 logits[n_pos][n_vocab]             // row r = logits after token (first_pos + r), i.e. predicting r+1
mode 1: per position { f32 lse; f32 pad; i32 ids[k]; f32 logits[k] }   // ids sorted by logit desc,
                                               // lse = logsumexp over the FULL vocab -> p = exp(logit - lse)
```
Engine dumps must use the same format (`quality.write_dump()` is a reference writer). Full logits are 517 KB per
position: use `--topk 64` (520 B/position) for long prompts.

### quality.py
```
python3 tools/quality.py BASE.logits TEST.logits [--json out.json] [--per-pos]
python3 tools/quality.py --dir logs/oracle/v0 <test_dir> [--json out.json] [--max-mean-kl X] [--min-top1 Y]
```
Per position: log-softmax (f64), KL(base‖test), top-1 match, top-5 overlap; for full dumps also the next-token NLL of
the prompt under base and test (diagnostic). Summary = token-weighted mean over files; exit 1 if a gate fails.
Top-K dumps: KL approximated over the base's top-K set + one tail bucket (lower bound when the test side is full; may
over-estimate when both sides are top-K — compare like with like). Stdlib only (numpy is not installed; used
automatically if present). Speed: ~25 ms/position pure Python -> 67 s for the full v0 set.

## 3. Prompt set (`tools/prompts/`, token ids are the contract)

| name | tokens | content |
|---|---:|---|
| en_prose | 308 | original English narrative prose |
| code_python | 414 | CPython 3.14 `heapq.py` lines 133-175 |
| code_cpp | 449 | C++ lock-free ring buffer |
| math | 462 | LaTeX proof (recurrence bounds) |
| chat | 144 | multi-turn chat, DeepSeek V4.1 template (non-thinking, ends `<｜Assistant｜></think>`), via `reference/encoding/encoding.py` |
| chat_think | 173 | thinking mode (system "Reasoning Effort: 75", a completed `<think>…</think>` turn, ends `<｜Assistant｜><think>`) |
| cjk | 277 | Chinese, Japanese, Korean |
| json | 480 | nested JSON document |
| **long_needle_d50** | **199081** | git docs (`/opt/homebrew/share/doc/git-doc/*.adoc`) interleaved with Python stdlib sources, 24 docs, needle at token ~99542 (depth 0.500), chat-wrapped question at the end; answer `MAPLE-7314-QUARTZ` |
| long_needle_d50_32k | 31071 | same recipe, 32K target, needle at ~15375 (0.495) |

`.json` next to each long prompt holds the needle, answer, depth and doc list. Rebuild: `python3 tools/prompts/make_prompts.py`,
`python3 tools/prompts/make_long.py --target 199000 --depth 0.5` (deterministic: re-running reproduces the committed .ids byte-for-byte; the committed .ids are authoritative).

## 4. Pinned baseline v0 (oracle logits)

`logs/oracle/v0/<name>.logits` (+ `.json`), full F32 logits at every position, 8 prompts, 2707 positions, 1.3 GB.
sha256s in `tools/oracle/v0_manifest.json`. Config: ngl 99, cpu-moe, mmap, n_batch 2048, n_ubatch 512, threads 6,
fa auto, swa_full default (true), n_ctx = max_len + 16. Re-dump command:
```
tools/oracle/measured.sh logs/oracle/v0/run.log build/oracle/oracle_logits -m SPLIT1 --list logs/oracle/v0/list.txt --quiet
```
Timing (00:41-00:50, page cache partly warm from earlier runs): 504.6 s wall for all 8 (load 25 s; 46-87 s per
prompt — each prompt is one ubatch, i.e. one full expert stream from SSD). Swap unchanged (346 -> 347 MB).
Prompt NLL under the oracle (sanity): en_prose 1.355 (ppl 3.9), code_python 0.090, code_cpp 0.286, math 0.612,
json 0.561, cjk 1.749, chat 2.407, chat_think 2.400. Greedy continuation of `chat` is coherent ("A 230 GB memory-mapped
model can run on a 64 GB machine because mmap doesn't load the file into RAM up front—it maps …").

Noise floor: identical config -> bit-exact. (A different kernel/batch layout gives a nonzero floor — see §6.)

## 5. Speed baseline (stock llama.cpp, working config, M5 Pro 64 GB)

Depth-0 runs: swap stayed at ~346-351 MB (pre-existing, no growth), memory pressure normal (200K: see 5.2). The page cache never holds
the 107 GiB of experts, so every run is effectively "SSD-streaming warm": attention/dense weights cached, experts
mostly re-read (pageins 50-60 GB per prefill pass). Raw: `logs/baseline/*.log|*.jsonl`.

### 5.1 llama-bench, depth 0 (`tools/oracle/bench_baseline.sh`, flags `-ngl 99 -ncmoe 99`)

| run (log) | test | threads | n_ubatch | tok/s (samples) | wall |
|---|---|---:|---:|---|---:|
| probe_cpumoe (cold, 1st run) | pp128 / tg8 | 6 | 512 | 4.40 / 3.36 | 114 s |
| threads (r1) | pp512 / tg32 | 6 | 512 | 7.64 / 2.66 | 474 s total |
| threads (r1) | pp512 / tg32 | **12** | 512 | **8.75 / 2.67** | |
| threads (r1) | pp512 / tg32 | 18 | 512 | 7.95 / 1.15 | |
| d0_pp512_tg128 (r2) | **pp512** | 12 | 512 | **8.48 ± 0.99** (7.79, 9.18) | 293 s |
| d0_pp512_tg128 (r2) | **tg128** | 12 | 512 | **4.10 ± 0.75** (3.57, 4.63) | |
| d0_pp2048_ub (r2) | **pp2048** | 12 | 512 | **9.52 ± 0.07** (9.57, 9.46) | 931 s total |
| d0_pp2048_ub (r2) | pp2048 | 12 | 2048 | 27.80 ± 0.33 (27.57, 28.03) | |
| d0_pp4096_ub4096 (r1) | pp4096 | 12 | 4096 | 42.73 | 229 s |

Prefill time per ubatch pass is roughly constant (~55-75 s at ub 512-2048, ~96 s at 4096, ~130 s at 8192) because
each pass streams the experts from SSD through page faults; so llama.cpp prefill tok/s ≈ ubatch / pass time and
the default `-ub 512` badly understates what stock llama.cpp can do. Decode: CPU experts + mmap page faults,
2.7-4.6 tok/s (variance = page-cache state). t=18 oversubscribes (decode collapses to 1.15).

### 5.2 Long context (oracle_logits, same placement, `--batch 8192 --ubatch 8192 --threads 12 --swa-full 0`)

llama-bench `-d N` re-prefills the depth for every test and repetition (and needs `-ub 8192` to be usable), so the
long-context points were measured with ONE prefill each via oracle_logits, which logs tok/s per 8192-token chunk
(= prefill speed vs depth) and then decodes at that depth. Same model placement as llama-bench.

| run (log: logs/oracle/needle/, logs/baseline/) | prompt | prefill | per-chunk tok/s at depth | decode at end |
|---|---:|---|---|---|
| long_needle_d50_32k | 31,071 | 518.0 s = **59.98 tok/s** | 59.1 (0-8K), 67.3, 61.6, 52.1 (24-31K, partial chunk) | 2.44 tok/s (10 tok, needle answer) |
| long_needle_d50 | 199,081 | 3486.8 s (58.1 min) = **57.10 tok/s** | 62.9 @0, 60.7 @25K, 60.0 @66K, 58.3 @98K, 57.5 @123K, 56.3 @148K, 54.8 @164K, 54.1 @188K (last partial chunk 24.4) | **1.91 tok/s** (10 tok) |
| tg_at_32k | 31,071 | 526.0 s = **59.07 tok/s** | 61.4, 61.2, 61.9, (partial) | **2.62 tok/s** (64 tok, `--ignore-eos`) |

200K run conditions: started 01:48:38, wall 3517 s, swap 360 -> 870 MB (+510 MB, during the 200K context
allocation; ~49.6 GB wired by Metal buffers at 200K, ~10 GB of other apps' memory went to the compressor), max RSS
57.1 GB, pageins +181.9 M pages (16 KiB) ≈ 2.98 TB read = 25 passes x the full expert set (nothing stays cached).
Aftermath: swap kept rising to 1.96 GB in the 3 minutes after the run (other apps paging their memory back);
it did not grow during later runs (v0_faoff, tg_at_32k). Plan for this: a 200K llama.cpp run is at the memory edge.
Prefill slows only ~14% from depth 0 to 190K (DS41 attention is sparse/compressed), so it is dominated by expert
streaming at every depth.

**Long-context quality gate (needle) — PASS at 32K and 200K:** greedy output `MAPLE-7314-QUARTZ<｜end▁of▁sentence｜>`
for both. Engine gate: `python3 tools/needle_check.py grade <engine.json>` must also PASS (json needs `gen_text`).

### 5.3 The numbers to beat (stock llama.cpp, best config found)

| metric | llama.cpp |
|---|---|
| prefill, short prompt (pp512, default ub) | 8.5 tok/s |
| prefill, best ubatch (8192), depth 0-32K | 60 tok/s |
| prefill, 200K prompt end-to-end | 57.1 tok/s (58.1 min TTFT) |
| decode, depth 0 (tg128) | 4.1 tok/s (2.7-4.6) |
| decode, depth 32K | 2.6 tok/s (64 tokens) |
| decode, depth 200K | 1.9 tok/s (10-token sample) |


## 6. Notes for the engine / gate thresholds

- **Noise floor / gate calibration.** Same config, separate process: bit-exact (KL 0). Flash attention off vs on
  (`logs/oracle/v0_faoff`, a legitimate kernel change; same math, different accumulation order/precision):
  mean KL **0.0020**, median ~1e-4, p99 0.003-0.12 per prompt, max 0.21, top-1 **98.71%**, top-5 97.0%, prompt NLL
  moves by <= 0.006 (`logs/oracle/v0_vs_faoff.json`). Per prompt: chat 0.0063/99.3%, chat_think 0.0045/98.8%,
  cjk 0.0050/95.7%, code_cpp 0.00014/99.8%, code_python 0.00023/100%, en_prose 0.0052/96.8%, json 0.00056/99.0%,
  math 0.00084/99.1%. So an engine that matches llama.cpp "as well as llama.cpp matches itself across kernels" sits
  around mean KL ~0.002, top-1 ~98.5-99%. Suggested gate vs v0: mean KL <= 0.005 and top-1 >= 97% (flag anything
  past 2-3x the fa floor for inspection with `--per-pos`); the engine-vs-own-baseline gate for speed changes can be
  much tighter.
- The oracle is llama.cpp's deepseek41 implementation, not the reference PyTorch code. Whether it simulates the
  reference fp8 (KV) / fp4 (indexer, compressor) activation quantization is not visible from here; choose
  `MACSEEK_ACTQ` by which setting gets closer to v0 (SPEC §6).
- `llama_kv_cache_iswa: using full-size SWA cache` is llama.cpp's default (swa_full=1) for the short dumps; the long runs
  used swa_full=0 (needed to fit 200K). No logit effect is expected for a single sequence, not measured.
- Tokenization: the `.ids` files ARE the contract. `oracle_logits --tokenize-only --text F --dump-ids out.ids` gives
  llama.cpp's tokenization of any text in ~4 s (vocab-only load, no lock) for diffing the engine tokenizer.
- Long-context logits: not pinned (a full-logit 200K dump is 103 GB). If needed, a re-run with
  `--out X.logits --topk 64 --last 4096` (one more ~60 min run) gives top-64 + logsumexp for the final 4K positions.
- `dequant_check` (SPEC §4.2): `build/oracle/dequant_check --gguf <split with the tensor> --tensor blk.0.ffn_gate_exps.weight
  --row0 2304 --rows 2 --out x.f32 [--raw-out blocks.bin]` or `--type iq1_s --in blocks.bin --out x.f32` — libggml's
  `to_float` for any GGUF type (bit-exact reference for the engine's CPU dequant). Reads only the requested rows,
  no model lock needed. Verified file offsets agree with docs/tensors.txt.
- Measurement hygiene: macOS evicts most of the model's page cache between processes (free pages back to ~3M after
  a run), and the experts never fit, so every llama.cpp run here is "dense warm-ish, experts cold". Thermals were not
  controlled beyond running sequentially; the 58-minute 200K run shows a smooth ~14% decline, consistent with depth,
  not throttling cliffs.
