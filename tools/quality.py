#!/usr/bin/env python3
"""MacSeek quality proxy: compare two logit dumps (MSLG format, see docs/modules/oracle.md).

Per position (same token ids required): log-softmax of both, KL(base || test), top-1 agreement, top-5 overlap.
Full-logit dumps give exact KL. If either side is a top-K dump, KL is approximated over the base's top-K set plus one
tail bucket (KL_approx = sum_{v in S} p log p/q + P_tail log P_tail/Q_tail). By the data-processing inequality this is
a LOWER bound of the true KL when q is known exactly on S (test dump full). Where a token of S is missing from a
top-K test dump its q is estimated as min(test K-th prob, test tail mass / (V - K)), which can OVER-estimate KL
(synthetic check: exact 0.027 vs 0.044 with both sides top-64). Compare like with like when tracking a trend.

Stdlib only; uses numpy automatically when importable (much faster), else a C-iterator pure-Python path
(~30 ms/position for a 129280 vocab).

usage:
  quality.py BASE.logits TEST.logits [--json OUT.json] [--per-pos]
  quality.py --dir BASE_DIR TEST_DIR [--json OUT.json]       # all *.logits present in both dirs
  gates (exit 1 if violated): --max-mean-kl X  --min-top1 Y (fraction)
"""
import json, math, os, struct, sys
from array import array
from itertools import compress, repeat
from operator import ge, mul, sub

try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None

MAGIC = b"MSLG"


class Dump:
    """Reader for .logits files. rows are read lazily per position."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        hdr = self.f.read(32)
        if hdr[:4] != MAGIC:
            raise ValueError(f"{path}: bad magic")
        (self.version, self.n_tokens, self.n_vocab, self.mode, self.k, self.first_pos,
         self.n_pos) = struct.unpack("<7I", hdr[4:32])
        self.tokens = array("i")
        self.tokens.frombytes(self.f.read(4 * self.n_tokens))
        self.data_off = 32 + 4 * self.n_tokens
        self.row_bytes = 4 * self.n_vocab if self.mode == 0 else 8 + 8 * self.k

    def row(self, i):
        """i = index into stored positions (0..n_pos-1). Returns ('full', floats) or ('topk', lse, ids, logits)."""
        self.f.seek(self.data_off + i * self.row_bytes)
        b = self.f.read(self.row_bytes)
        if self.mode == 0:
            if np is not None:
                return ("full", np.frombuffer(b, dtype="<f4").astype(np.float64))
            a = array("f"); a.frombytes(b)
            return ("full", a)
        lse = struct.unpack("<f", b[:4])[0]
        ids = array("i"); ids.frombytes(b[8:8 + 4 * self.k])
        lg = array("f"); lg.frombytes(b[8 + 4 * self.k:])
        return ("topk", lse, ids, lg)


def write_dump(path, tokens, rows=None, topk_rows=None, k=0, first_pos=0, n_vocab=129280):
    """Writer for engine-side dumps. rows: list of float sequences (full); or topk_rows: list of (lse, ids, logits)."""
    with open(path, "wb") as f:
        n_pos = len(rows) if rows is not None else len(topk_rows)
        f.write(MAGIC + struct.pack("<7I", 1, len(tokens), n_vocab, 0 if rows is not None else 1, k, first_pos, n_pos))
        f.write(array("i", tokens).tobytes())
        if rows is not None:
            for r in rows:
                f.write(array("f", r).tobytes())
        else:
            for lse, ids, lg in topk_rows:
                f.write(struct.pack("<ff", lse, 0.0) + array("i", ids).tobytes() + array("f", lg).tobytes())


# ---------------- per-position metrics ----------------

def _lse_py(a):
    m = max(a)
    return m + math.log(math.fsum(map(math.exp, map(sub, a, repeat(m)))))


def _topn_py(a, n):
    import heapq
    vals = heapq.nlargest(n, a)
    t = vals[-1]
    idx = list(compress(range(len(a)), map(ge, a, repeat(t))))
    idx.sort(key=lambda i: (-a[i], i))
    return idx[:n]


def metrics_full(a, b, nxt=-1):
    """a = base logits, b = test logits (full rows), nxt = actual next token (-1 if none).
    Returns (kl, top1_match, top5_overlap, base_top1, test_top1, base_nll, test_nll)."""
    if np is not None:
        lsa = a.max() + np.log(np.exp(a - a.max()).sum())
        lsb = b.max() + np.log(np.exp(b - b.max()).sum())
        la, lb = a - lsa, b - lsb
        pa = np.exp(la)
        kl = float((pa * (la - lb)).sum())
        ta = np.argpartition(-a, 5)[:5]; tb = np.argpartition(-b, 5)[:5]
        a1 = int(np.argmax(a)); b1 = int(np.argmax(b))
        nl = (float(-la[nxt]), float(-lb[nxt])) if nxt >= 0 else (float("nan"), float("nan"))
        return (max(kl, 0.0), a1 == b1, len(set(ta.tolist()) & set(tb.tolist())), a1, b1) + nl
    lsa, lsb = _lse_py(a), _lse_py(b)
    # KL = sum pa*(a-b) - (lsa - lsb), pa = exp(a - lsa)
    s = math.fsum(map(mul, map(math.exp, map(sub, a, repeat(lsa))), map(sub, a, b)))
    kl = s - (lsa - lsb)
    a1 = a.index(max(a)); b1 = b.index(max(b))
    ta, tb = _topn_py(a, 5), _topn_py(b, 5)
    nl = (lsa - a[nxt], lsb - b[nxt]) if nxt >= 0 else (float("nan"), float("nan"))
    return (max(kl, 0.0), a1 == b1, len(set(ta) & set(tb)), a1, b1) + nl


def _as_topk(row, k):
    if row[0] == "topk":
        return row[1], list(row[2]), list(row[3])
    a = row[1]
    if np is not None:
        lse = float(a.max() + np.log(np.exp(a - a.max()).sum()))
        ids = np.argsort(-a, kind="stable")[:k].tolist()
        return lse, ids, [float(a[i]) for i in ids]
    lse = _lse_py(a)
    ids = _topn_py(a, k)
    return lse, ids, [a[i] for i in ids]


def metrics_topk(rb, rt, n_vocab, k):
    """Approximate metrics when at least one side is a top-K dump."""
    lse_b, ids_b, lg_b = _as_topk(rb, k)
    ids_b, lg_b = ids_b[:k], lg_b[:k]
    p = [math.exp(x - lse_b) for x in lg_b]
    if rt[0] == "full":
        b = rt[1]
        lse_t = float(b.max() + np.log(np.exp(b - b.max()).sum())) if np is not None else _lse_py(b)
        q = [math.exp(float(b[i]) - lse_t) for i in ids_b]
        t_top = _as_topk(rt, 5)[1]
    else:
        lse_t, ids_t, lg_t = rt[1], list(rt[2]), list(rt[3])
        qmap = {i: math.exp(x - lse_t) for i, x in zip(ids_t, lg_t)}
        q_kth = min(qmap.values())
        q_tail = max(1.0 - sum(qmap.values()), 0.0)
        q_fill = min(q_kth, q_tail / max(n_vocab - len(ids_t), 1))
        q = [qmap.get(i, q_fill) for i in ids_b]
        t_top = ids_t[:5]
    eps = 1e-12
    kl = sum(pi * (math.log(max(pi, eps)) - math.log(max(qi, eps))) for pi, qi in zip(p, q))
    pt, qt = max(1.0 - sum(p), eps), max(1.0 - sum(q), eps)
    kl += pt * math.log(pt / qt)
    return max(kl, 0.0), ids_b[0] == t_top[0], len(set(ids_b[:5]) & set(t_top[:5])), ids_b[0], t_top[0]


def compare(base_path, test_path, per_pos=False):
    B, T = Dump(base_path), Dump(test_path)
    if list(B.tokens) != list(T.tokens):
        raise SystemExit(f"token ids differ: {base_path} vs {test_path}")
    if B.n_vocab != T.n_vocab:
        raise SystemExit("n_vocab differs")
    # positions stored in both
    lo = max(B.first_pos, T.first_pos)
    hi = min(B.first_pos + B.n_pos, T.first_pos + T.n_pos)
    exact = B.mode == 0 and T.mode == 0
    k = B.k if B.mode == 1 else (T.k if T.mode == 1 else 0)
    kls, top1, top5, rows = [], 0, 0, []
    nll_b, nll_t, n_nll = 0.0, 0.0, 0
    for pos in range(lo, hi):
        rb, rt = B.row(pos - B.first_pos), T.row(pos - T.first_pos)
        nxt = B.tokens[pos + 1] if pos + 1 < B.n_tokens else -1
        if exact:
            kl, m1, o5, a1, b1, na, nb = metrics_full(rb[1], rt[1], nxt)
            if nxt >= 0:
                nll_b += na; nll_t += nb; n_nll += 1
        else:
            kl, m1, o5, a1, b1 = metrics_topk(rb, rt, B.n_vocab, k)
        kls.append(kl); top1 += m1; top5 += o5
        if per_pos:
            rows.append({"pos": pos, "kl": kl, "base_top1": a1, "test_top1": b1, "top5_overlap": o5, "next_tok": nxt})
    n = len(kls)
    s = sorted(kls)
    res = {
        "base": base_path, "test": test_path, "n_pos": n, "exact": exact, "topk": k,
        "mean_kl": sum(kls) / n if n else float("nan"),
        "median_kl": s[n // 2] if n else float("nan"),
        "p99_kl": s[min(n - 1, int(0.99 * n))] if n else float("nan"),
        "max_kl": s[-1] if n else float("nan"),
        "argmax_kl_pos": lo + kls.index(s[-1]) if n else -1,
        "top1_agree": top1 / n if n else float("nan"),
        "top5_overlap": top5 / (5 * n) if n else float("nan"),
        # next-token NLL of the prompt text itself (exact mode only): base vs test; diagnostic, not a gate
        "base_nll": nll_b / n_nll if n_nll else None, "test_nll": nll_t / n_nll if n_nll else None,
    }
    if per_pos:
        res["per_pos"] = rows
    return res


def human(name, r):
    return (f"{name:14s} n={r['n_pos']:6d} KL mean {r['mean_kl']:.5f} med {r['median_kl']:.5f} p99 {r['p99_kl']:.4f} "
            f"max {r['max_kl']:.4f}@{r['argmax_kl_pos']}  top1 {100 * r['top1_agree']:6.2f}%  top5 "
            f"{100 * r['top5_overlap']:6.2f}%"
            + (f"  nll {r['base_nll']:.3f}->{r['test_nll']:.3f}" if r.get("base_nll") is not None else "")
            + ("" if r["exact"] else "  (approx, top-%d)" % r["topk"]))


def main(argv):
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__); return 2
    opts = {}
    pos_args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--json", "--max-mean-kl", "--min-top1"):
            opts[a] = argv[i + 1]; i += 2
        elif a in ("--per-pos", "--dir"):
            opts[a] = True; i += 1
        else:
            pos_args.append(a); i += 1
    results = []
    if "--dir" in opts:
        bd, td = pos_args
        names = sorted(n for n in os.listdir(bd) if n.endswith(".logits") and os.path.exists(os.path.join(td, n)))
        for n in names:
            r = compare(os.path.join(bd, n), os.path.join(td, n), "--per-pos" in opts)
            r["name"] = n[:-7]; results.append(r)
            print(human(r["name"], r), flush=True)
    else:
        r = compare(pos_args[0], pos_args[1], "--per-pos" in opts)
        r["name"] = os.path.basename(pos_args[1])[:-7]; results.append(r)
        print(human(r["name"], r))
    tot = sum(r["n_pos"] for r in results)
    summary = {
        "n_files": len(results), "n_pos": tot,
        "mean_kl": sum(r["mean_kl"] * r["n_pos"] for r in results) / tot,
        "max_kl": max(r["max_kl"] for r in results),
        "top1_agree": sum(r["top1_agree"] * r["n_pos"] for r in results) / tot,
        "top5_overlap": sum(r["top5_overlap"] * r["n_pos"] for r in results) / tot,
        "numpy": np is not None,
    }
    if len(results) > 1:
        print(f"{'ALL':14s} n={tot:6d} KL mean {summary['mean_kl']:.5f} max {summary['max_kl']:.4f}  top1 "
              f"{100 * summary['top1_agree']:6.2f}%  top5 {100 * summary['top5_overlap']:6.2f}%")
    if "--json" in opts:
        with open(opts["--json"], "w") as f:
            json.dump({"summary": summary, "files": results}, f, indent=1)
    ok = True
    if "--max-mean-kl" in opts and summary["mean_kl"] > float(opts["--max-mean-kl"]):
        print(f"GATE FAIL: mean KL {summary['mean_kl']:.5f} > {opts['--max-mean-kl']}"); ok = False
    if "--min-top1" in opts and summary["top1_agree"] < float(opts["--min-top1"]):
        print(f"GATE FAIL: top1 {summary['top1_agree']:.4f} < {opts['--min-top1']}"); ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
