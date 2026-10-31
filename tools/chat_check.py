#!/usr/bin/env python3
"""Chat-format check: macseek chat-format vs DeepSeek's reference/encoding/encoding.py and the GGUF jinja template.

1. The official test vectors tests/data/chat/test_input_N.json -> test_output_N.txt (byte-exact).
2. N random conversations (system / multi-turn / thinking + chat / reasoning efforts / drop_thinking / tools with
   namespaces / tool calls with typed arguments / tool results incl. reordering / latest_reminder / mid-system /
   tasks / image blocks), each rendered by encoding.py and (if jinja2 is importable, e.g. a venv python) by the
   GGUF template (tokenizer.chat_template read from split 00001), compared with macseek chat-format --batch.

usage: [venv/bin/]python tools/chat_check.py [--n 2000] [--macseek build/macseek] [--gguf PATH]
"""
import argparse
import copy
import json
import os
import random
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "reference", "encoding"))
import encoding as E  # noqa: E402

DEFAULT_GGUF = os.path.expanduser(
    "~/Unsloth Models/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/"
    "8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf")


def gguf_chat_template(path):
    with open(path, "rb") as f:
        d = f.read(8 << 20)
    o = 4 + 4 + 8
    nkv = struct.unpack_from("<Q", d, o)[0]
    o += 8
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def rs(o):
        n = struct.unpack_from("<Q", d, o)[0]
        return d[o + 8:o + 8 + n], o + 8 + n

    for _ in range(nkv):
        k, o = rs(o)
        t = struct.unpack_from("<I", d, o)[0]
        o += 4
        if t == 8:
            v, o = rs(o)
            if k == b"tokenizer.chat_template":
                return v.decode()
        elif t == 9:
            et, n = struct.unpack_from("<IQ", d, o)
            o += 12
            if et == 8:
                for _ in range(n):
                    _, o = rs(o)
            else:
                o += sizes[et] * n
        else:
            o += sizes[t]
    return None


WORDS = ["hello", "world", "Paris", "天气", "北京", "🌍", "a\"quote\"", "back\\slash", "tab\there", "new\nline",
         "ctrl\x01char", "é", "<b>", "{json}", "  spaced  ", "", "数字123", "math: x<y", "end."]


def text(r, k=4):
    return " ".join(r.choice(WORDS) for _ in range(r.randint(0 if k == 0 else 1, k)))


def rand_value(r, depth=0):
    c = r.randint(0, 9 if depth < 2 else 5)
    if c == 0: return text(r)
    if c == 1: return r.randint(-1000, 10 ** 12)
    if c == 2: return r.choice([1.5, 0.1, 1e-07, 2.0, 1e16, 123456.789, -0.0, 3.0e-5, 1e22])
    if c == 3: return r.choice([True, False])
    if c == 4: return None
    if c == 5: return text(r, 2)
    if c <= 7: return [rand_value(r, depth + 1) for _ in range(r.randint(0, 3))]
    return {text(r, 1) or "k": rand_value(r, depth + 1) for _ in range(r.randint(0, 3))}


def rand_tool(r, i):
    fn = {"name": "tool_%d" % i, "description": text(r),
          "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": text(r)},
                                                            "n": {"type": "integer", "default": -1}},
                         "required": ["q"]}}
    if r.random() < 0.2:
        del fn["description"]
    tool = {"type": "function", "function": fn}
    k = r.random()
    if k < 0.15:
        tool["namespace"] = "ns%d" % i
    elif k < 0.25:
        tool["namespace"] = {"name": "space%d" % i, "description": "Namespace " + text(r, 2)}
    return tool


def rand_conv(r, for_template):
    msgs = []
    tools = [rand_tool(r, i) for i in range(r.randint(1, 3))] if r.random() < 0.4 else None
    if r.random() < 0.7 or tools:
        sys_content = text(r, 6)
        if r.random() < 0.1:
            sys_content = [{"type": "text", "text": text(r)}, {"type": "text", "text": text(r)}]
        msgs.append({"role": "system", "content": sys_content})
    if r.random() < 0.2:
        msgs.append({"role": "latest_reminder", "content": "2026-10-31," + text(r, 2)})
    nturns = r.randint(1, 4)
    call_id = 0
    for t in range(nturns):
        if r.random() < 0.12 and t > 0:
            msgs.append({"role": "system", "content": "mid-system " + text(r)})
        u = {"role": "user", "content": text(r, 8)}
        if r.random() < 0.1:
            u["content"] = [{"type": "text", "text": text(r)}, {"type": "image_url", "image_url": {"url": "x.png"}},
                            {"type": "text", "text": text(r)}]
        msgs.append(u)
        last = t == nturns - 1
        if last and r.random() < 0.5:
            break  # generation prompt case
        a = {"role": "assistant", "content": text(r, 6)}
        if r.random() < 0.7:
            a["reasoning_content"] = "reason: " + text(r, 6)
        if tools and r.random() < 0.6:
            calls = []
            for _ in range(r.randint(1, 3)):
                call_id += 1
                args = {"q": text(r, 2)}
                for _ in range(r.randint(0, 3)):
                    args[r.choice(["n", "flag", "data", "list", "x"])] = rand_value(r)
                tool = r.choice(tools)
                name = tool["function"]["name"]
                ns = tool.get("namespace")
                if isinstance(ns, dict):
                    ns = ns["name"]
                if ns and r.random() < 0.5:
                    name = ns + "::" + name
                tc = {"id": "call_%d" % call_id, "type": "function",
                      "function": {"name": name, "arguments": args if for_template else
                                   (json.dumps(args, ensure_ascii=False) if r.random() < 0.7 else args)}}
                calls.append(tc)
            a["tool_calls"] = calls
            if r.random() < 0.3:
                a["content"] = ""
            msgs.append(a)
            results = [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"r": text(r, 2)}, ensure_ascii=False)}
                       for c in calls]
            if r.random() < 0.4:
                r.shuffle(results)
            msgs.extend(results)
            if r.random() < 0.3 and t == nturns - 1:
                break  # generation after tool results
            a2 = {"role": "assistant", "content": text(r, 5)}
            if r.random() < 0.7:
                a2["reasoning_content"] = "after tools " + text(r, 3)
            msgs.append(a2)
        else:
            msgs.append(a)
    if r.random() < 0.08 and msgs[-1]["role"] == "user":
        msgs[-1]["task"] = r.choice(["action", "query", "title"])
    case = {"messages": msgs, "tools": tools,
            "thinking_mode": r.choice(["thinking", "chat"]),
            "reasoning_effort": r.choice([None, "low", "high", "max", 1, 37, 100] + (["medium"] if for_template else [])),
            "drop_thinking": r.choice([True, True, False])}
    return case


def enc_py(case):
    msgs = copy.deepcopy(case["messages"])
    if case["tools"]:
        msgs[0]["tools"] = case["tools"]  # first message is a system message whenever tools are used
    return E.encode_messages(msgs, thinking_mode=case["thinking_mode"], drop_thinking=case["drop_thinking"],
                             reasoning_effort=case["reasoning_effort"])


def make_jinja(tpl_src):
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(tpl_src)

    def render(case):
        kw = dict(messages=case["messages"], tools=case["tools"], add_generation_prompt=True,
                  thinking_mode=case["thinking_mode"], drop_thinking=case["drop_thinking"])
        if case["reasoning_effort"] is not None:
            kw["reasoning_effort"] = case["reasoning_effort"]
        return tpl.render(**kw)

    return render


def run_macseek(binary, cases, tag):
    reqs = []
    for c in cases:
        rq = {"messages": c["messages"], "thinking_mode": c["thinking_mode"], "drop_thinking": c["drop_thinking"]}
        if c["tools"]:
            rq["tools"] = c["tools"]
        if c["reasoning_effort"] is not None:
            rq["reasoning_effort"] = c["reasoning_effort"]
        reqs.append(rq)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "chat_check_%s.json" % tag)
    with open(path, "w") as f:
        json.dump(reqs, f, ensure_ascii=False)
    r = subprocess.run([binary, "chat-format", "--batch", "-f", path], capture_output=True)
    if r.returncode:
        sys.exit("macseek failed: " + r.stderr.decode())
    return json.loads(r.stdout.decode())


def compare(name, refs, outs, cases):
    bad = 0
    for i, (ref, out) in enumerate(zip(refs, outs)):
        if ref != out:
            bad += 1
            if bad <= 3:
                print("  [%s] case %d differs" % (name, i))
                if isinstance(out, dict) or isinstance(ref, dict):
                    print("    ref:", str(ref)[:300], "\n    got:", str(out)[:300])
                else:
                    k = next((j for j in range(min(len(ref), len(out))) if ref[j] != out[j]), min(len(ref), len(out)))
                    print("    ref: %r\n    got: %r" % (ref[max(0, k - 80):k + 80], out[max(0, k - 80):k + 80]))
                    print("    case:", json.dumps(cases[i], ensure_ascii=False)[:600])
    print("%-28s %d/%d identical" % (name, len(refs) - bad, len(refs)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--macseek", default=os.path.join(ROOT, "build", "macseek"))
    ap.add_argument("--gguf", default=os.environ.get("MACSEEK_GGUF", DEFAULT_GGUF))
    args = ap.parse_args()
    fails = 0

    # 1. official vectors
    vec_cases, vec_out = [], []
    for n in range(1, 100):
        fi = os.path.join(ROOT, "tests", "data", "chat", "test_input_%d.json" % n)
        if not os.path.exists(fi):
            break
        for c in E.load_cases(fi):
            msgs = c["messages"]
            vec_cases.append({"messages": msgs, "tools": None, "thinking_mode": c["thinking_mode"] or "chat",
                              "reasoning_effort": c["reasoning_effort"], "drop_thinking": True})
        vec_out.append(open(os.path.join(ROOT, "tests", "data", "chat", "test_output_%d.txt" % n)).read())
    fails += compare("official test vectors", vec_out, run_macseek(args.macseek, vec_cases, "vec"), vec_cases)

    # 2. random vs encoding.py
    r = random.Random(args.seed)
    cases = [rand_conv(r, for_template=False) for _ in range(args.n)]
    refs = []
    for c in cases:
        try:
            refs.append(enc_py(c))
        except Exception as e:  # reference rejects -> macseek must reject too
            refs.append({"error": str(e)})
    outs = run_macseek(args.macseek, cases, "py")
    outs = [({"error": o["error"]} if isinstance(o, dict) else o) for o in outs]
    # compare errors by presence only
    refs = [({"error": "x"} if isinstance(x, dict) else x) for x in refs]
    outs = [({"error": "x"} if isinstance(x, dict) else x) for x in outs]
    fails += compare("random vs encoding.py", refs, outs, cases)

    # 3. random vs GGUF jinja template
    try:
        render = make_jinja(gguf_chat_template(args.gguf))
    except ImportError:
        print("jinja2 not importable: skipping template comparison (use a venv with jinja2)")
        render = None
    if render:
        r = random.Random(args.seed + 1)
        cases = [rand_conv(r, for_template=True) for _ in range(args.n)]
        refs = []
        for c in cases:
            try:
                refs.append(render(c))
            except Exception:
                refs.append({"error": "x"})
        outs = [({"error": "x"} if isinstance(o, dict) else o) for o in run_macseek(args.macseek, cases, "jj")]
        fails += compare("random vs GGUF jinja", refs, outs, cases)
    print("ALL MATCH" if not fails else "MISMATCHES: %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
