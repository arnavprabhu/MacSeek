#!/usr/bin/env python3
"""Build the ~200K-token long-context needle prompt (the 200K quality gate).

Corpus: machine-local real text, deterministic file order — git documentation (English technical prose,
/opt/homebrew/share/doc/git-doc/*.adoc) interleaved with Python 3.14 stdlib sources. Each document is preceded by a
"===== FILE: name =====" header. A NEEDLE sentence is inserted at the paragraph boundary closest to the requested depth
(fraction of corpus tokens), and the whole thing is wrapped in the DeepSeek V4.1 chat template (non-thinking mode)
with a final question. Output (tools/prompts/):
  long_needle_d<depth>[_<N>k].txt / .ids   prompt text and llama.cpp token ids (suffix when target < 190K)
  long_needle_d<depth>.json         {n_tokens, needle, answer, needle_token_pos, depth, files...}

usage: python3 tools/prompts/make_long.py [--target 199000] [--depth 0.5] [--model SPLIT1]
"""
import json, os, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from make_prompts import DEFAULT_MODEL, PYLIB, BOS  # noqa: E402

GITDOC = "/opt/homebrew/share/doc/git-doc"
GIT_FILES = ["user-manual.adoc", "gitcore-tutorial.adoc", "gittutorial.adoc", "gittutorial-2.adoc",
             "git-rebase.adoc", "gitworkflows.adoc", "git-merge.adoc", "git-commit.adoc", "gitrevisions.adoc",
             "git-reset.adoc", "git-checkout.adoc", "git-bisect.adoc", "git-stash.adoc", "git-worktree.adoc",
             "git-submodule.adoc", "git-cherry-pick.adoc", "git-tag.adoc", "git-fetch.adoc", "git-push.adoc",
             "git-clone.adoc", "git-log.adoc", "git-diff.adoc", "gitattributes.adoc", "gitignore.adoc",
             "githooks.adoc", "git-config.adoc", "git-branch.adoc", "git-remote.adoc", "git-blame.adoc"]
PY_FILES = ["heapq.py", "bisect.py", "textwrap.py", "fractions.py", "statistics.py", "functools.py",
            "difflib.py", "dataclasses.py", "shlex.py", "string/__init__.py", "csv.py", "calendar.py",
            "argparse.py", "contextlib.py", "graphlib.py", "colorsys.py", "random.py", "operator.py",
            # appended for the 262K prompt (documents after the 200K prompt's cut, so it is unchanged)
            "pathlib/_local.py", "typing.py", "inspect.py", "pickle.py", "tarfile.py", "zipfile/__init__.py",
            "subprocess.py", "threading.py", "logging/__init__.py", "enum.py", "ast.py", "decimal.py",
            "configparser.py", "email/message.py", "http/client.py", "tempfile.py", "pprint.py", "glob.py"]

NEEDLE = ("Important note for later: the access code for the Halvorsen observatory archive is "
          "MAPLE-7314-QUARTZ. Remember this code exactly.")
ANSWER = "MAPLE-7314-QUARTZ"
QUESTION = ("\n\nThe documents above contain one sentence that gives the access code for the Halvorsen "
            "observatory archive. What is that access code? Reply with the code only.")


def docs():
    gi, pi = 0, 0
    out = []
    # interleave: 2 git docs then 1 python file
    while gi < len(GIT_FILES) or pi < len(PY_FILES):
        for _ in range(2):
            if gi < len(GIT_FILES):
                p = os.path.join(GITDOC, GIT_FILES[gi]); gi += 1
                if os.path.exists(p):
                    out.append((os.path.basename(p), open(p, encoding="utf-8", errors="replace").read()))
        if pi < len(PY_FILES):
            p = os.path.join(PYLIB, PY_FILES[pi]); pi += 1
            if os.path.exists(p):
                out.append((PY_FILES[pi - 1], open(p, encoding="utf-8", errors="replace").read()))
    return out


def tokenize_text(tool, model, text, tmp):
    with open(tmp + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
    subprocess.run([tool, "-m", model, "--text", tmp + ".txt", "--tokenize-only", "--dump-ids", tmp + ".ids",
                    "--quiet"], check=True, stderr=subprocess.DEVNULL)
    d = open(tmp + ".ids", "rb").read()
    return list(struct.unpack("<%di" % (len(d) // 4), d))


def main():
    a = sys.argv[1:]
    target = int(a[a.index("--target") + 1]) if "--target" in a else 199000
    depth = float(a[a.index("--depth") + 1]) if "--depth" in a else 0.5
    model = a[a.index("--model") + 1] if "--model" in a else DEFAULT_MODEL
    tool = os.path.join(ROOT, "build", "oracle", "oracle_logits")
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "macseek_long")
    pieces = [f"===== FILE: {n} =====\n{t.strip()}\n\n" for n, t in docs()]
    # token count per piece (one tokenizer call on the whole thing, separated, is cheaper: count each piece once)
    counts = []
    for p in pieces:
        counts.append(len(tokenize_text(tool, model, p, tmp)))
    head = BOS + "<｜User｜>"
    tail_text = QUESTION + "<｜Assistant｜></think>"
    overhead = len(tokenize_text(tool, model, head + NEEDLE + "\n\n" + tail_text, tmp))
    budget = target - overhead
    chosen, ccounts, tot = [], [], 0
    for p, c in zip(pieces, counts):
        if tot + c > budget:
            # partial last piece: cut by characters proportionally, then re-measure
            frac = (budget - tot) / c
            cut = p[: int(len(p) * frac * 0.97)]
            cut = cut[: cut.rfind("\n") + 1]
            cc = len(tokenize_text(tool, model, cut, tmp))
            chosen.append(cut); ccounts.append(cc); tot += cc
            break
        chosen.append(p); ccounts.append(c); tot += c
    # needle position: the paragraph boundary ("\n\n") whose token offset (per-document token counts, linear
    # within a document) is closest to depth * tot; exact depth is re-measured below and recorded.
    full = "".join(chosen)
    starts, acc = [], 0
    for i, p in enumerate(chosen):
        starts.append((acc, len("".join(chosen[:i])), len(p), ccounts[i]))
        acc += ccounts[i]
    goal = depth * tot
    best_char, best_d = 0, 1e18
    for tok0, ch0, n_ch, n_tok in starts:
        if not (tok0 - n_tok <= goal <= tok0 + 2 * n_tok):
            continue
        p = full[ch0:ch0 + n_ch]
        j = p.find("\n\n")
        while j >= 0:
            est = tok0 + n_tok * (j + 2) / max(n_ch, 1)
            if abs(est - goal) < best_d:
                best_char, best_d = ch0 + j + 2, abs(est - goal)
            j = p.find("\n\n", j + 2)
    body = full[:best_char] + NEEDLE + "\n\n" + full[best_char:]
    text = head + body.rstrip() + tail_text
    name = f"long_needle_d{int(round(depth * 100)):02d}" + ("" if 190000 <= target < 250000 else f"_{target // 1000}k")
    tp, ip = os.path.join(HERE, name + ".txt"), os.path.join(HERE, name + ".ids")
    ids = tokenize_text(tool, model, text, tmp)
    with open(tp, "w", encoding="utf-8") as f:
        f.write(text)
    with open(ip, "wb") as f:
        f.write(struct.pack("<%di" % len(ids), *ids))
    # locate needle token position (prefix tokenization)
    pre = head + full[:best_char]
    needle_pos = len(tokenize_text(tool, model, pre, tmp))
    meta = {"name": name, "n_tokens": len(ids), "target": target, "depth": depth, "needle": NEEDLE,
            "answer": ANSWER, "question": QUESTION.strip(), "needle_token_pos_approx": needle_pos,
            "needle_depth_actual": needle_pos / len(ids), "n_docs": len(chosen),
            "docs": [p.split("\n", 1)[0] for p in chosen]}
    with open(os.path.join(HERE, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"{name}: {len(ids)} tokens, needle at ~{needle_pos} ({needle_pos / len(ids):.3f}), {len(chosen)} docs")


if __name__ == "__main__":
    main()
