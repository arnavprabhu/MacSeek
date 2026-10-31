#!/usr/bin/env python3
"""Tokenizer oracle check: macseek tokenize vs Homebrew llama-tokenize (vocab-only load of split 00001).

Builds a corpus (prose, code, JSON, CJK, emoji, whitespace, numbers, chat-formatted text, special-token edge
cases, random Unicode/invalid-UTF-8 fuzz, and a multi-MB concatenation of Python stdlib sources) in OUTDIR,
tokenizes every file with both tools and requires 100% id equality. Also writes <name>.ids (oracle output) for
the small files so tests/test_tok.cpp can re-check without llama-tokenize (--export-dir).

usage: tools/tok_oracle_check.py [--macseek build/macseek] [--gguf PATH] [--out DIR] [--export-dir tests/data/tok]
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import sysconfig
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model_path  # noqa: E402

DEFAULT_GGUF = model_path.default_model()

PROSE = """It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of
foolishness... "Don't you think so?" she asked. I'd've said no; they'll say yes, we're sure. O'Neill's CAN'T-stop
attitude -- e.g., Mr. Smith's 3rd attempt (at 9:45 a.m.) -- cost $1,234.56 (≈ €1.1k) & 12% more!!!
Naïve café owners in Zürich façade—“smart quotes”, ‘single’, «guillemets», and the em–dash… Ünïcödé ÀÉÎÕÜ ß ﬁ ﬂ.
Привет, как дела? Γειά σου Κόσμε! שלום עולם. مرحبا بالعالم. नमस्ते दुनिया। สวัสดีชาวโลก. ສະບາຍດີ. Xin chào thế giới.
"""

CJK = """深度求索公司于2023年成立。人工智能（AI）正在改变世界！我们的目标是：让每个人都能使用AGI。
日本語のテキスト：ひらがな、カタカナ、漢字が混ざっています。ｶﾀｶﾅ（半角）とＡＢＣ（全角）１２３。
한국어 텍스트도 포함합니다: 안녕하세요, 세계! 中文標點符號：「引號」『雙引號』〈書名號〉《書名》……——。
繁體中文與简体中文混合：龍龙、愛爱、國国。𠀀𠀁𪚥 (CJK Ext B), 〇一二三, 々〆ヶ, ㈱㍻, ⺀⿰⿱.
"""

EMOJI = ("Hello 👋🏽 world 🌍! Family: 👨‍👩‍👧‍👦, flags: 🇺🇸🇯🇵🏳️‍🌈, keycaps: 1️⃣#️⃣, hearts ❤️💙🧡, "
         "skin tones 👍🏻👍🏿, text-vs-emoji ☺︎☺️, ZWJ 🧑‍💻 👩🏽‍🚀, symbols ™©®℃№∑∫√∞≠≤≥±×÷→⇒↔♠♣♥♦★☆✓✗\n"
         "Math: 𝔸𝕓𝒞𝓓 𝑥² + 𝑦² = 𝑧², ∀x∈ℝ ∃y: x<y. Box: ┌─┬─┐│ │└─┴─┘ Braille ⠓⠑⠇⠇⠕\n")

NUMBERS = ("0 1 12 123 1234 12345 123456 1234567 3.14159 2.718e10 -42 +7 1,000,000 1_000_000 0x1F 0b1010 "
           "007 ٠١٢٣٤٥٦٧٨٩ ۰۱۲ ०१२३ ０１２３４ ¹²³ ½¾⅓ ⅠⅡⅫ ⅰⅱ ①②③ 𝟎𝟏𝟐 v1.2.3 10:30 2026-10-31 1e-5 "
           "99.99% $5 №1 x86_64 utf8 h264 3D 4K 5G abc123def 1a2b3c\n")

WHITESPACE = ("a  b   c\tdef\t\tghi\n\njk\r\nlm\r\r\nno \n  pq   \n\n\n   rs\u00a0tu\u3000vw\u2028xy\u2029z"
              "\x0b\x0c end   \t \n \t  \n\t\t\tindent\n    four\n        eight\n \u0085 nel \u200b zwsp "
              "\u2003emsp\u202f nnbsp \ufeff bom\n   \n  trailing   ")

SPECIAL_EDGE = ("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜><think>x</think>y<｜end▁of▁sentence｜>"
                "<｜User｜<｜User｜>> <<think>> </think></think><think><think> <｜｜DSML｜｜> ｜DSML｜｜DSML｜ "
                "<dsml:</dsml:<dsml:> <｜DSML｜ invoke name=\"f\"> <|EOT|> <｜place▁holder▁no▁7｜><｜place▁holder▁no▁77｜>"
                "<｜fim▁begin｜>def f():<｜fim▁hole｜>pass<｜fim▁end｜> <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                "<｜latest_reminder｜>x<｜System｜>y <｜deepseek_image｜> <|place_holder_mm_span_0442|> <｜Use r｜> "
                "<｜User｜ <think <｜end▁of▁sentence｜<｜end▁of▁sentence｜>\n")


def fuzz_text(seed, n):
    rnd = random.Random(seed)
    pools = [
        (0x20, 0x7E), (0x09, 0x0D), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF),
        (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF), (0x1E00, 0x1FFF), (0x2000, 0x206F), (0x2070, 0x22FF),
        (0x2460, 0x27BF), (0x2E80, 0x2FFF), (0x3000, 0x30FF), (0x3100, 0x33FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
        (0xA000, 0xA4CF), (0xAC00, 0xD7A3), (0xE000, 0xE0FF), (0xF900, 0xFAFF), (0xFE00, 0xFE6F), (0xFF00, 0xFFEF),
        (0xFFF0, 0xFFFD), (0x10000, 0x1007F), (0x1D400, 0x1D7FF), (0x1F300, 0x1FAFF), (0x20000, 0x2A6DF),
        (0x1B4E, 0x1B7F), (0x105C0, 0x105F3), (0x1CC00, 0x1CCF9), (0xE0000, 0xE007F), (0xF0000, 0xF00FF),
    ]
    out = []
    for _ in range(n):
        k = rnd.random()
        if k < 0.35:
            out.append(rnd.choice(["the", " cat", "  ", "\n", "\r\n", "!", "(", "'s", " 12", "345", "x", "\t", "—"]))
        else:
            lo, hi = rnd.choice(pools)
            cp = rnd.randint(lo, hi)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            out.append(chr(cp))
    return "".join(out)


def invalid_utf8_bytes(seed, n):
    rnd = random.Random(seed)
    junk = [b"\x80", b"\xbf", b"\xc0\x80", b"\xc1\xbf", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xed\xbf\xbf",
            b"\xf4\x90\x80\x80", b"\xf5\x80", b"\xff", b"\xfe", b"\xe4\xbd", b"\xf0\x9f\x98", b"\xc3",
            b"\xe2\x82\xac", b"\xf0\x9f\x98\x80", b"abc", b" ", b"\n", "中".encode(), b"\xef\xbf\xbd"]
    return b"".join(rnd.choice(junk) for _ in range(n))


def chat_texts():
    sys.path.insert(0, os.path.join(ROOT, "reference", "encoding"))
    import encoding as E  # DeepSeek reference encoder
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "tests", "data", "chat", "test_input_*.json"))):
        for c in E.load_cases(f):
            for tm in ("chat", "thinking"):
                try:
                    out.append(E.encode_messages(c["messages"], thinking_mode=c.get("thinking_mode") or tm,
                                                 reasoning_effort=c.get("reasoning_effort")))
                except Exception:
                    pass
    return "\n".join(out)


def read(path, limit=None):
    with open(path, "rb") as f:
        return f.read(limit) if limit else f.read()


def build_corpus(out):
    os.makedirs(out, exist_ok=True)
    files = {}

    def put(name, data, small=True):
        if isinstance(data, str):
            data = data.encode("utf-8")
        p = os.path.join(out, name)
        with open(p, "wb") as f:
            f.write(data)
        files[name] = (p, small)

    put("prose.txt", PROSE * 3 + "".join(open(os.path.join(sysconfig.get_paths()["stdlib"], "this.py")).read()))
    stdlib = sysconfig.get_paths()["stdlib"]
    put("code_python.txt", read(os.path.join(stdlib, "json", "decoder.py")) + read(os.path.join(stdlib, "textwrap.py")))
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True).stdout.strip()
    cpp = os.path.join(sdk, "usr", "include", "c++", "v1", "optional")
    put("code_cpp.txt", read(cpp, 20000) if os.path.exists(cpp) else b"int main(){return 0;}\n")
    js = glob.glob(os.path.join(ROOT, "web", "*.html"))
    put("code_js.txt", read(js[0], 20000) if js else b"const x = (a, b) => a ?? b; // js\n")
    put("json.txt", read(os.path.join(ROOT, "reference", "tokenizer.json"), 30000) +
        json.dumps(json.load(open(os.path.join(ROOT, "tests", "data", "chat", "test_input_3.json"))),
                   ensure_ascii=False, indent=2).encode())
    put("cjk.txt", CJK * 3)
    put("emoji.txt", EMOJI * 3)
    put("numbers.txt", NUMBERS * 3)
    put("whitespace.txt", WHITESPACE * 3)
    put("chat.txt", chat_texts())
    put("special_edge.txt", SPECIAL_EDGE * 2)
    put("fuzz_unicode.txt", fuzz_text(1234, 8000))
    put("fuzz_invalid_utf8.dat", invalid_utf8_bytes(99, 3000))
    # large: Python stdlib sources (several MB)
    big = []
    total = 0
    for p in sorted(glob.glob(os.path.join(stdlib, "**", "*.py"), recursive=True)):
        b = read(p)
        big.append(b)
        total += len(b)
        if total > 8 << 20:
            break
    put("large_stdlib.txt", b"".join(big), small=False)
    put("large_fuzz.txt", fuzz_text(777, 400000), small=False)
    return files


def run_ids(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("%s failed: %s" % (cmd[0], r.stderr.decode(errors="replace")[-2000:]))
    return json.loads(r.stdout.decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macseek", default=os.path.join(ROOT, "build", "macseek"))
    ap.add_argument("--gguf", default=os.environ.get("MACSEEK_GGUF", DEFAULT_GGUF))
    ap.add_argument("--out", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "macseek_tok_corpus"))
    ap.add_argument("--export-dir", default=None, help="write small corpus files + oracle .ids here")
    args = ap.parse_args()
    if "00001-of-" not in args.gguf:
        sys.exit("refusing: only use llama-tokenize on split 00001 (metadata only)")
    files = build_corpus(args.out)
    fails = 0
    for name, (path, small) in files.items():
        t0 = time.time()
        want = run_ids(["llama-tokenize", "-m", args.gguf, "-f", path, "--ids", "--log-disable", "--no-escape"])
        t1 = time.time()
        got = run_ids([args.macseek, "tokenize", "-m", args.gguf, "-f", path])
        t2 = time.time()
        ok = got == want
        size = os.path.getsize(path)
        print("%-24s %9d B %8d tok  %s   (oracle %.2fs, macseek %.2fs)" %
              (name, size, len(want), "OK" if ok else "MISMATCH", t1 - t0, t2 - t1))
        if not ok:
            fails += 1
            i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
            print("   first diff at %d: oracle %s | macseek %s" % (i, want[i:i + 8], got[i:i + 8]))
        if ok and small and args.export_dir:
            os.makedirs(args.export_dir, exist_ok=True)
            with open(os.path.join(args.export_dir, name), "wb") as f:
                f.write(read(path))
            with open(os.path.join(args.export_dir, name + ".ids"), "w") as f:
                f.write(" ".join(map(str, want)) + "\n")
    print("ALL MATCH" if not fails else "%d FILE(S) MISMATCH" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
