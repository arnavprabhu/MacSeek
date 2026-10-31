#!/usr/bin/env python3
"""Long-context needle gate.

  needle_check.py run   [--prompt tools/prompts/long_needle_d50] [--gen 24] [--out-dir logs/oracle/needle]
        runs stock llama.cpp (build/oracle/oracle_logits, same config as the oracle dumps, swa_full=0) under the model
        lock, saves the json (prefill chunk tok/s across depth, generated text, decode tok/s) and grades it.
  needle_check.py grade RESULT.json [--prompt tools/prompts/long_needle_d50]
        grades any result json with a "gen_text" field (the engine can emit the same json).
PASS iff the expected answer string (from <prompt>.json "answer") appears in the generated text.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = "/Users/arnav/Desktop/Projects/MacSeek"


def arg(a, name, default):
    return a[a.index(name) + 1] if name in a else default


def grade(result_path, prompt):
    meta = json.load(open(prompt + ".json"))
    res = json.load(open(result_path))
    text = res.get("gen_text", "")
    ok = meta["answer"] in text
    print(f"needle {os.path.basename(prompt)} ({meta['n_tokens']} tokens, needle depth "
          f"{meta['needle_depth_actual']:.3f}): {'PASS' if ok else 'FAIL'}  generated: {text!r}")
    return ok


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "grade"):
        print(__doc__); return 2
    prompt = arg(a, "--prompt", os.path.join(ROOT, "tools", "prompts", "long_needle_d50"))
    if a[0] == "grade":
        return 0 if grade(a[1], prompt) else 1
    gen = arg(a, "--gen", "24")
    out_dir = arg(a, "--out-dir", os.path.join(MAIN, "logs", "oracle", "needle"))
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(prompt)
    model = open(os.path.join(MAIN, "logs", "baseline", "model_path.txt")).read().strip()
    rj, log = os.path.join(out_dir, name + ".json"), os.path.join(out_dir, name + ".log")
    cmd = [os.path.join(ROOT, "tools", "oracle", "measured.sh"), log,
           os.path.join(ROOT, "build", "oracle", "oracle_logits"), "-m", model, "--ids", prompt + ".ids",
           "--gen", gen, "--swa-full", "0", "--json", rj, "--quiet"]
    extra = a[a.index("--") + 1:] if "--" in a else []
    subprocess.run(cmd + extra, check=False)
    return 0 if grade(rj, prompt) else 1


if __name__ == "__main__":
    sys.exit(main())
