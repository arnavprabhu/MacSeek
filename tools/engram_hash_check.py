#!/usr/bin/env python3
"""Engram hash cross-check (stdlib only; numpy optional for the multiplier RNG check).

Re-derives everything reference/inference/engram.py computes from the config and compares it with the GGUF metadata:
  * primes (EngramLayout.from_args: layers [1,14], max_ngram 4, 8 heads, engram_vocab_size 16000000) vs
    deepseek41.ple.head_vocab_sizes, using a deterministic Miller-Rabin test (no sympy)
  * per-layer offsets (cumsum within a layer, as engram.py) + layer base vs deepseek41.ple.head_offsets (global)
  * layer row totals vs config engram_num_embeddings and the per_layer_token_embd row count
  * multipliers (np.random.default_rng(10007*layer).integers(...)*2+1) vs deepseek41.ple.layer_multipliers
  * token_map size / compressed vocab / pad id
Then hashes token sequences exactly like NgramHashState.forward (cache gather, cumulative blocking, rolling xor)
and, with --bin build/test_arch_engram_hash, compares the C++ streaming hasher (incl. chunked prefill + decode).

usage: tools/engram_hash_check.py [--gguf split1.gguf] [--bin build/test_arch_engram_hash] [--tokens 1,2,3]
"""
import argparse, os, random, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_dump  # noqa: E402

DEFAULT_GGUF = os.path.expanduser(
    "~/Unsloth Models/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/"
    "8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf")
LAYERS, MAX_NGRAM, HEADS, VOCAB = (1, 14), 4, 8, 16000000
NUM_EMB = (384006168, 384016682)       # config.json engram_num_embeddings
TABLE_ROWS = 768022850                 # per_layer_token_embd rows
COMPRESSED_VOCAB = 99092


def is_prime(n):
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:  # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def layout_primes():
    """EngramLayout.from_args: [layer][ngram-2][head]"""
    primes, seen = [], set()
    for _ in LAYERS:
        per_ngram = []
        for _ in range(MAX_NGRAM - 1):
            sizes, cur = [], VOCAB - 1
            for _ in range(HEADS):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def multipliers_numpy():
    try:
        import numpy as np
    except ImportError:
        return None
    bound = max(1, (np.iinfo(np.int64).max // COMPRESSED_VOCAB) // 2)
    rows = []
    for layer in LAYERS:
        g = np.random.default_rng(10007 * layer)
        rows.append([int(v) * 2 + 1 for v in g.integers(low=0, high=bound, size=(MAX_NGRAM,), dtype=np.int64)])
    return rows


class Hasher:
    """Literal port of NgramHashState (cache over absolute positions)."""

    DEAD = -1

    def __init__(self, token_map, pad_token_id, mult, primes, offsets):
        self.token_map, self.pad = token_map, token_map[pad_token_id]
        self.mult, self.primes, self.offsets = mult, primes, offsets  # primes/offsets flat [layer][24]
        self.cache = []

    def forward(self, input_ids, start_pos, token_mask=None):
        comp = [self.token_map[t] if (token_mask is None or token_mask[i]) else self.DEAD for i, t in enumerate(input_ids)]
        del self.cache[start_pos:]
        assert len(self.cache) == start_pos
        self.cache.extend(comp)
        out = []
        for pos in range(start_pos, start_pos + len(input_ids)):
            toks, blocked = [], False
            for shift in range(MAX_NGRAM):
                src = self.cache[max(pos - shift, 0)]
                blocked = blocked or pos < shift or src == self.DEAD
                toks.append(self.pad if blocked else src)
            row = []
            for l in range(len(LAYERS)):
                prods = [toks[s] * self.mult[l][s] for s in range(MAX_NGRAM)]
                rolling = prods[0]
                for i in range(1, MAX_NGRAM):
                    rolling ^= prods[i]
                    for k in range(HEADS):
                        c = (i - 1) * HEADS + k
                        row.append(rolling % self.primes[l][c] + self.offsets[l][c])
            out.append(row)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--bin", default=None, help="test_arch_engram_hash binary for the C++ cross-check")
    ap.add_argument("--tokens", default=None, help="comma-separated token ids to print hashes for")
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    kv, _, _ = gguf_dump.parse(a.gguf)
    tm = kv["deepseek41.engram.token_map"]
    hv = kv["deepseek41.ple.head_vocab_sizes"]
    ho = kv["deepseek41.ple.head_offsets"]
    lm = kv["deepseek41.ple.layer_multipliers"]
    pad_tok = kv["deepseek41.engram.pad_token_id"]
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")

    print("metadata vs engram.py derivation:")
    primes = layout_primes()
    flat = [p for layer in primes for ng in layer for p in ng]
    check("primes == head_vocab_sizes (order [layer][ngram][head])", flat == list(hv), f"first {flat[:3]} last {flat[-1]}")
    local = [[sum(layer_flat[:i]) for i in range(len(layer_flat))] for layer_flat in
             ([p for ng in layer for p in ng] for layer in primes)]
    totals = [sum(p for ng in layer for p in ng) for layer in primes]
    check("layer totals == engram_num_embeddings", tuple(totals) == NUM_EMB, str(totals))
    check("sum == per_layer_token_embd rows", sum(totals) == TABLE_ROWS)
    glob = local[0] + [totals[0] + o for o in local[1]]
    check("head_offsets == per-layer cumsum + layer base (global rows)", glob == list(ho))
    check("token_map size 129280, compressed vocab 99092", len(tm) == 129280 and max(tm) + 1 == COMPRESSED_VOCAB)
    check("pad = token_map[2]", tm[pad_tok] == 2, f"pad_token_id={pad_tok} -> {tm[pad_tok]}")
    mult = [list(lm[0:4]), list(lm[4:8])]
    check("multipliers odd", all(m % 2 == 1 for m in lm))
    bound = ((2**63 - 1) // COMPRESSED_VOCAB) // 2
    check("multipliers < 2*bound (no int64 overflow)", all(m < 2 * bound for m in lm))
    mn = multipliers_numpy()
    if mn is None:
        print("  skip multipliers RNG check (numpy not installed)")
    else:
        check("multipliers == default_rng(10007*layer) (order [layer][shift])", mn == mult, str(mn))

    prim2 = [flat[0:24], flat[24:48]]
    offs2 = [list(ho[0:24]), list(ho[24:48])]
    if a.tokens:
        h = Hasher(tm, pad_tok, mult, prim2, offs2)
        toks = [int(t) for t in a.tokens.split(",")]
        for p, row in enumerate(h.forward(toks, 0)):
            print(p, " ".join(map(str, row)))
    if a.bin:
        print(f"C++ cross-check vs {a.bin}:")
        rng = random.Random(1)
        bad = 0
        for trial in range(a.trials):
            n = rng.randint(1, 40)
            toks = [rng.randrange(129280) if rng.random() > 0.2 else rng.choice([0, 1, 2, 3, 128799]) for _ in range(n)]
            split = rng.randint(0, n)  # prefill chunk boundary; remaining tokens fed one by one (decode)
            h = Hasher(tm, pad_tok, mult, prim2, offs2)
            exp = h.forward(toks[:split], 0) if split else []
            for i in range(split, n):
                exp += h.forward([toks[i]], i)
            res = subprocess.run([a.bin, "--dump", ",".join(map(str, toks)), str(split)], capture_output=True, text=True)
            got = [list(map(int, ln.split()[1:])) for ln in res.stdout.strip().splitlines() if ln and ln[0].isdigit()]
            if got != exp:
                bad += 1
                if bad < 3:
                    print("  mismatch for", toks, "split", split)
        check(f"C++ hasher == engram.py port on {a.trials} random sequences", bad == 0)
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
