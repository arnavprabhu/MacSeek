#!/usr/bin/env python3
"""Build the fixed short prompt set for MacSeek's quality gate.

Writes tools/prompts/<name>.txt (exact text) and <name>.ids (raw little-endian int32 token ids as tokenized by
stock llama.cpp via build/oracle/oracle_logits --tokenize-only), plus manifest.json. Token ids are the contract:
the engine is fed the .ids, never the .txt. Re-running must reproduce byte-identical .ids (deterministic inputs).

usage: python3 tools/prompts/make_prompts.py [--model SPLIT1] [--tool build/oracle/oracle_logits]
"""
import json, os, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "reference", "encoding"))
import encoding as enc  # official DeepSeek-V4.1 chat encoder (stdlib only)

DEFAULT_MODEL = ("/Users/arnav/Unsloth Models/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/"
                 "8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf")
PYLIB = "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14"
BOS = enc.bos_token


def lines_of(path, start, end):
    with open(path, encoding="utf-8") as f:
        return "".join(f.readlines()[start:end])


PROSE = """The lighthouse keeper had kept a logbook for thirty-one years, and in all that time he had never missed an entry. Each evening, after he trimmed the wick and wound the clockwork that turned the great lens, he sat at the narrow desk beneath the stairs and wrote down the wind, the state of the sea, the ships that had passed, and anything else that seemed worth remembering. Most entries were short. "Wind NW, moderate. Two colliers southbound. Gulls restless." But some ran on for pages, especially in winter, when the storms came in off the Atlantic and the whole tower hummed like a struck bell.

When the lighthouse was finally automated, a young engineer from the coastal authority arrived to install the new equipment. She expected to find an old man resentful of the change, and she had rehearsed several polite ways of explaining that the decision had been made far above both their heads. Instead, he met her at the door with a pot of tea and a stack of logbooks tied with string.

"You'll want these," he said. "The machine will know when the light is on. It won't know why the herring moved inshore in the autumn of 1987, or which rocks the fog likes to hide behind."

She took the books back to the city and, out of curiosity more than duty, began to read them on the train. By the time she reached her station she had decided that the most valuable thing in the lighthouse had never been the lamp."""

CPP_CODE = """// Simple fixed-capacity ring buffer with a power-of-two size.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity_pow2)
        : mask_(capacity_pow2 - 1), data_(capacity_pow2) {
        static_assert(sizeof(T) <= 64, "keep elements small");
    }

    bool push(const T& value) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail > mask_) {
            return false;  // full
        }
        data_[head & mask_] = value;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        const size_t head = head_.load(std::memory_order_acquire);
        if (tail == head) {
            return std::nullopt;  // empty
        }
        T value = data_[tail & mask_];
        tail_.store(tail + 1, std::memory_order_release);
        return value;
    }

    size_t size() const {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }

private:
    const size_t mask_;
    std::vector<T> data_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

int main() {
    RingBuffer<uint32_t> rb(1024);
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 100000; ++i) {
        while (!rb.push(i)) {
            if (auto v = rb.pop()) sum += *v;
        }
    }
"""

MATH = r"""Problem. Let $a_1 = 1$ and $a_{n+1} = a_n + \frac{1}{a_n}$ for $n \ge 1$. Prove that $\sqrt{2n-1} \le a_n \le \sqrt{3n-2}$ for all $n \ge 1$, and deduce that $a_{100}$ lies strictly between $14$ and $18$.

Solution. Squaring the recurrence gives
$$a_{n+1}^2 = a_n^2 + 2 + \frac{1}{a_n^2}.$$
Since $a_n^2 > 0$, we immediately get $a_{n+1}^2 > a_n^2 + 2$. Summing this inequality from $k = 1$ to $n-1$ yields
$$a_n^2 > a_1^2 + 2(n-1) = 2n - 1 \quad (n \ge 2),$$
and equality holds for $n = 1$. Hence $a_n \ge \sqrt{2n-1}$.

For the upper bound, note that every term satisfies $a_n \ge 1$, so $\frac{1}{a_n^2} \le 1$ and therefore $a_{n+1}^2 \le a_n^2 + 3$. Summing again,
$$a_n^2 \le 1 + 3(n-1) = 3n - 2.$$
For $n = 100$ this gives $199 < a_{100}^2 \le 298$, so $14.1 < a_{100} < 17.3$, which proves the claim. In fact a sharper estimate follows from $\frac{1}{a_k^2} \le \frac{1}{2k-1}$:
$$a_n^2 \le 2n - 1 + \sum_{k=1}^{n-1} \frac{1}{2k-1} \approx 2n - 1 + \tfrac12 \ln n + 0.98,$$
so $a_{100} \approx 14.18$."""

CJK = """人工智能的发展离不开大量高质量的数据。近年来，大型语言模型在翻译、写作、编程和数学推理等任务上取得了显著进步，但它们的推理成本仍然很高。为了在个人电脑上运行这些模型，研究人员提出了量化、稀疏专家混合和投机解码等多种技术。量化通过降低权重的精度来减少内存占用；专家混合模型在每个词元只激活一小部分参数；而投机解码则利用一个小型草稿模型预测多个词元，再由大模型一次性验证。

春の京都は、桜の名所として世界中から観光客が訪れます。哲学の道を歩くと、疎水沿いに咲く桜が水面に映り、とても美しい景色が広がります。夕方になると、ライトアップされた寺院が幻想的な雰囲気を醸し出します。

한국어 문장도 포함합니다. 오늘은 날씨가 맑아서 공원에 산책을 나갔습니다. 아이들이 연을 날리고 있었고, 노인들은 벤치에 앉아 이야기를 나누고 있었습니다.

最后，我们用一个简单的问题来结束这段文字：如果一台电脑有六十四GB的统一内存，而模型文件有两百三十GB，应该如何设计推理引擎，才能既快又准确？"""

JSON_DOC = json.dumps({
    "id": "evt_01HZX9Q7M4K2",
    "type": "order.completed",
    "created": "2026-10-30T18:42:07Z",
    "livemode": False,
    "data": {
        "order": {
            "number": 48213,
            "currency": "EUR",
            "customer": {"name": "Ingrid Solberg", "email": "ingrid.solberg@example.no", "tier": "gold",
                         "address": {"line1": "Storgata 14", "city": "Tromsø", "postal_code": "9008", "country": "NO"}},
            "items": [
                {"sku": "CAM-X100-BLK", "title": "Mirrorless camera body", "qty": 1, "unit_price": 1299.0, "tax_rate": 0.25},
                {"sku": "LNS-35F18", "title": "35mm f/1.8 lens", "qty": 1, "unit_price": 449.5, "tax_rate": 0.25},
                {"sku": "SD-128-V60", "title": "128 GB SD card (V60)", "qty": 2, "unit_price": 39.9, "tax_rate": 0.25},
            ],
            
            "shipping": {"method": "express", "carrier": "Bring", "cost": 14.0, "eta_days": 2},
            "totals": {"subtotal": 1828.3, "discount": 129.9, "shipping": 14.0, "tax": 428.1, "grand_total": 2140.5},
            "payment": {"method": "card", "brand": "visa", "last4": "4242", "captured": True},
        },
    },
}, ensure_ascii=False, indent=1)


def chat_prompts():
    multi = [
        {"role": "system", "content": "You are a concise, careful assistant. Prefer short, correct answers."},
        {"role": "user", "content": "What is the difference between a process and a thread?"},
        {"role": "assistant", "content": "A process is an independent program in execution with its own address space, "
         "file descriptors and resources. A thread is a unit of execution inside a process: threads of the same process "
         "share memory and resources but each has its own stack and registers. Switching between threads is cheaper, "
         "but shared memory means you need synchronization (mutexes, atomics) to avoid data races."},
        {"role": "user", "content": "Give me a one-paragraph explanation of why a memory-mapped 230 GB model can still "
         "run on a 64 GB machine, and what the main performance risk is."},
    ]
    chat = enc.encode_messages(multi, thinking_mode="chat")
    think = enc.encode_messages([
        {"role": "user", "content": "A train leaves at 09:40 and arrives at 13:05, with a 25-minute stop halfway. "
         "If the average speed while moving was 96 km/h, how long is the route? Think it through, then answer."},
        {"role": "assistant", "reasoning_content": "Total elapsed time is 13:05 - 09:40 = 3 h 25 min. Subtract the "
         "25-minute stop: moving time is 3 h 0 min = 3 hours. Distance = 96 km/h x 3 h = 288 km.",
         "content": "The route is 288 km long."},
        {"role": "user", "content": "Now suppose the stop was 40 minutes instead, but the arrival time is unchanged. "
         "What average moving speed is needed?"},
    ], thinking_mode="thinking", drop_thinking=False)
    return chat, think


def build_set():
    chat, think = chat_prompts()
    return [
        ("en_prose", BOS + PROSE),
        ("code_python", BOS + lines_of(os.path.join(PYLIB, "heapq.py"), 132, 175)),
        ("code_cpp", BOS + CPP_CODE),
        ("math", BOS + MATH),
        ("chat", chat),
        ("chat_think", think),
        ("cjk", BOS + CJK),
        ("json", BOS + JSON_DOC),
    ]


def tokenize(tool, model, text_path, ids_path):
    subprocess.run([tool, "-m", model, "--text", text_path, "--tokenize-only", "--dump-ids", ids_path, "--quiet"],
                   check=True, stderr=subprocess.DEVNULL)
    data = open(ids_path, "rb").read()
    return list(struct.unpack("<%di" % (len(data) // 4), data))


def main():
    args = sys.argv[1:]
    model = args[args.index("--model") + 1] if "--model" in args else DEFAULT_MODEL
    tool = args[args.index("--tool") + 1] if "--tool" in args else os.path.join(ROOT, "build", "oracle", "oracle_logits")
    manifest = []
    for name, text in build_set():
        tp, ip = os.path.join(HERE, name + ".txt"), os.path.join(HERE, name + ".ids")
        with open(tp, "w", encoding="utf-8") as f:
            f.write(text)
        ids = tokenize(tool, model, tp, ip)
        manifest.append({"name": name, "n_tokens": len(ids), "txt": name + ".txt", "ids": name + ".ids",
                         "first_ids": ids[:8]})
        flag = "" if 64 <= len(ids) <= 512 else "   <-- outside 64..512"
        print(f"{name:12s} {len(ids):5d} tokens{flag}")
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump({"tokenizer": "llama.cpp 0.8.0 b11960 (vocab of the UD-IQ1_S GGUF), add_special=false, "
                   "parse_special=true", "prompts": manifest}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
