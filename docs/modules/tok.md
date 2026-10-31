# Module tok: tokenizer, chat format, detokenizer, CLI, HTTP server skeleton

Files: `src/tok/{tokenizer,chat,json}.{hpp,cpp}`, `src/tok/unicode_table.inc` (generated), `src/server/http.{hpp,cpp}`,
`src/app/main.cpp`, `tests/test_tok.cpp` (+ `tests/data/{tok,chat}`), `tools/{gen_unicode_table,tok_oracle_check,chat_check}.py`.

## Status (verified)

| what | result |
|---|---|
| ids vs Homebrew `llama-tokenize` 0.8.0 (b11960), `--no-escape`, parse_special on | **100% identical** on 15 files / 3.1M tokens: prose (multi-script), Python / C++ / JS code, JSON (tokenizer.json slice), CJK+kana+hangul, emoji (ZWJ, flags, skin tones, VS16), numbers (Arabic-Indic, fullwidth, superscripts, Roman), mixed whitespace (NBSP, U+3000, U+2028, VT/FF, NEL, BOM), chat-formatted text, special-token edge cases (adjacent / nested / partial), random Unicode fuzz (8K + 400K codepoints, incl. Unicode 16 additions, PUA, tags), invalid UTF-8 fuzz, **8.4 MB concatenated Python stdlib (2.28M tokens)** |
| ids with `--no-parse-special` | identical (special_edge, chat, prose) |
| detokenizer | `decode(encode(x)) == x` for every valid-UTF-8 corpus file; streaming decoder never emits partial UTF-8 |
| chat format vs DeepSeek `encoding.py` test vectors 1-5 (HF `encoding/tests`) | **5/5 byte-exact** |
| chat format vs `encoding.py`, 2000 random conversations | 2000/2000 identical |
| chat format vs GGUF jinja template (jinja2 3.1.6), 2000 random conversations | 2000/2000 identical |
| HTTP server + echo backend (ctest) | models, chat + completions, stream + non-stream, stop strings, max_tokens, keep-alive, 4xx |
| `ctest` `test_tok` | 174 checks pass, ~0.2 s |

Speed (M5 Pro, single thread, Release): load from split 00001 **26 ms**; encode **81 MB/s** on the 8.4 MB stdlib file
(22K tokens/ms: a 200K-token prompt takes ~10 ms), 29 MB/s on dense random Unicode, 100 MB/s on repeated code.
Worst cases (one 2 MB letter run / one 1 MB whitespace run) still 6-7 MB/s. Target was >= 5 MB/s.

## Tokenizer (`ms::tok::Tokenizer`)

```cpp
auto tk = ms::tok::Tokenizer::load_from_gguf(path, &err);   // private GGUF reader, tokenizer.* keys only
std::vector<int32_t> ids = tk->encode(text, /*parse_special=*/true, /*add_bos=*/false);
std::string_view bytes = tk->piece(id, /*special=*/true);    // raw bytes of one token
std::string s = tk->decode(ids);
ms::tok::StreamDecoder dec(*tk); out += dec.push(id); ... out += dec.flush();
```
- Swap point for the shared GGUF parser: fill `ms::tok::VocabData` (tokens, token_type, merges, bos/eos, chat
  template) from `src/gguf` and call `Tokenizer::init()`; `read_gguf_vocab()` is the private stand-in.
- `default_gguf_path()` = `$MACSEEK_GGUF` or the UD-IQ1_S split 00001 (5.8 MB; holds all metadata).
- The contract is "same ids as `llama_tokenize(parse_special=true, add_special=false)`", because the llama.cpp
  logits oracle (SPEC §6 gate C) tokenizes that way. Details that matter for exactness:
  1. **Special tokens** = token_type CONTROL (1278) | USER_DEFINED (5: `<think>`, `</think>`, `｜DSML｜`, `<dsml:`,
     `</dsml:`) | UNKNOWN. Partition like llama.cpp: tokens ordered by text length desc (same `std::sort` call on ids
     0..n), each token takes its leftmost non-overlapping occurrences inside still-raw text. Implemented as one trie
     scan collecting all occurrences + priority-ordered acceptance (O(text) not O(text x 1283)).
     `parse_special=false` skips CONTROL/UNKNOWN but still matches USER_DEFINED (llama.cpp/HF behaviour).
  2. **Invalid UTF-8**: each bad byte -> U+FFFD (llama.cpp's decoder rules: overlongs rejected, surrogates
     accepted, <= U+10FFFF).
  3. **Pre-tokenizer `joyai-llm`** = three regexes applied in sequence, each splitting the pieces of the previous
     one (`Isolated`): `\p{N}{1,3}`, `[一-龥぀-ゟ゠-ヿ]+`, and
     `[!-/:-@[-`{-~][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+`.
     Hand-written matcher (no std::regex) over llama.cpp's "collapsed" codepoint classes: ASCII is itself, other
     codepoints map to one of {White_Space, N, L, P, M, S, other}; `\s` is the C-locale set plus non-ASCII
     White_Space. Backtracking semantics of the alternatives were derived case by case (see `match3`).
  4. **Unicode tables**: `tools/gen_unicode_table.py` builds a two-stage 46 KB table from Python's `unicodedata`
     (16.0). llama.cpp's tables are Unicode 15.1, so the 5185 codepoints added in 16.0 are masked as unassigned
     (verified: with masking the class of every codepoint equals llama.cpp's). `--unicode16` disables it; HF
     `tokenizers` (Unicode 16) differs from llama.cpp only on those codepoints (2 tokens in the 400K fuzz).
  5. **BPE**: merges resolved to (left_id, right_id) -> (rank, merged_id) in an open-addressing table (all 127741
     merges have their parts and result in the vocab, checked at load). Lowest rank first, leftmost on ties
     (== llama.cpp's bigram queue). O(n^2) array merge for pieces <= 96 bytes, heap + linked list above.
     Per-call piece cache (pieces <= 48 bytes) gives most of the speed.
- `piece()`: NORMAL tokens are GPT-2 byte-decoded; CONTROL tokens return their text only with `special=true`;
  USER_DEFINED always return their text. `is_eog()` = eos (id 1) only; integration may extend.
- `StreamDecoder`: holds back an incomplete trailing UTF-8 sequence (max 3 bytes), replaces invalid bytes with
  U+FFFD, `flush()` releases the tail. Output chunks are always valid UTF-8 (safe for JSON/SSE).

## Chat format (`ms::chat`)

```cpp
ms::chat::Options o;                  // thinking_mode "thinking"|"chat", drop_thinking, reasoning_effort, add_generation_prompt, add_bos
ms::chat::format(messages, tools, response_format, o, prompt, &err);
ms::chat::format_request(openai_body, prompt, &err, &resolved);   // reads extensions, see below
ms::chat::Reply r = ms::chat::parse_reply(generated_text, thinking);   // reasoning / content / tool_calls
```
Hard-coded port of `reference/encoding/encoding.py` (normative; byte-exact on its vectors) covering: BOS,
`<｜System｜>` first/mid-conversation, `Reasoning Effort: N (range 1-100, ...)` line (thinking mode only;
low 50 / medium 62 / high 75 / max 100 / int 1..100, default high), multi-turn user/assistant, drop_thinking
(reasoning kept only after the last real user turn; disabled when any tools are present), `</think>` / `<think>`
generation headers, tool schemas (`## Tools` block, Python `json.dumps(ensure_ascii=False)`-exact JSON incl. float
repr and key order, namespaces `ns::name` and namespace descriptions), assistant tool calls in DSML
(`<｜DSML｜ calls>/invoke/parameter string="true|false"`, JSON-string arguments parsed like encoding.py), tool
results merged into user turns as `<tool_result>...</tool_result>` and reordered to the call order,
`latest_reminder`, tasks (`action`, `query`, ...), `wo_eos`, image blocks -> `<｜deepseek_image｜>`.
From the GGUF template on top: `reasoning_effort: "medium"`, top-level `tools`/`response_format` attached to a
leading system message (an empty one is inserted if needed), `thinking` as alias of `reasoning_content`, tool
definitions without the `function` wrapper, `add_generation_prompt=false`.
`format_request` reads (top level or in `chat_template_kwargs`): `thinking_mode`, `enable_thinking`, `thinking`
(bool), `reasoning_effort`, `drop_thinking`, `add_generation_prompt`. Default thinking_mode = "thinking"
(template default; encoding.py's CLI test harness uses "chat").
Not implemented: encoding.py's `context=` (prefix-cache continuation) parameter.

`ms::json` (`src/tok/json.hpp`): ordered-object JSON value, parser (\u surrogate pairs, NaN/Infinity) and a
serializer matching Python `json.dumps(ensure_ascii=False)`; also used by the server.

## HTTP server skeleton (`ms::server`)

`Server(tokenizer, backend, ServerConfig{host, port (0 = ephemeral), model_name, default_max_tokens})`,
`start()`, `stop()`, `wait()`. Dependency-free POSIX sockets, thread per connection, HTTP/1.1 keep-alive,
SSE via chunked transfer encoding, CORS `*`, one generation at a time (mutex around the backend).
- `GET /v1/models`, `GET /health`
- `POST /v1/chat/completions`: messages -> `chat::format_request` -> `encode` -> backend; non-stream returns
  `message.{content, reasoning_content (thinking mode), tool_calls}` + usage; stream sends `role` chunk,
  `reasoning_content` deltas until the `</think>` token, then `content` deltas, a final `tool_calls` delta if the
  reply contains a DSML block (content is held back only while its tail could still be the
  `\n\n<｜DSML｜ calls>` marker), `finish_reason` (`stop` | `length` | `tool_calls`), optional usage
  (`stream_options.include_usage`), `[DONE]`.
- `POST /v1/completions`: `prompt` string (special tokens parsed, no BOS added) or token-id array.
- Params: `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed` (passed through in
  `GenParams`), `stop` (string or list; streamed text never contains a prefix of a pending stop string).
- **Integration hook**: implement `ms::server::Backend::generate(prompt_ids, GenParams, emit)`: call `emit(tok)` for
  each sampled token (EOG included), stop when it returns false (client gone / stop string) -> `Cancelled`;
  return `Stop` after EOG, `Length` at `max_tokens`. `EchoBackend` is the test double.

## CLI (`build/macseek`)

```
macseek tokenize    [-m GGUF] (-p TEXT | -f FILE | --stdin) [--no-parse-special] [--bos] [--pieces] [--count] [--bench N]
macseek detok       [-m GGUF] (IDS... | -f FILE | --stdin) [--no-special] [--stream]
macseek chat-format [-m GGUF] (-f REQ.json | --stdin) [--thinking-mode M] [--effort E] [--no-drop-thinking] [--no-gen-prompt] [--ids] [--batch]
macseek serve       [-m GGUF] [--host H] [--port P] [--max-tokens N] --echo
macseek gen | bench | logits | snapshot      -> "not implemented yet (integration hook)"
```
`tokenize` prints `[id, id, ...]` exactly like `llama-tokenize --ids`. `serve` without `--echo` and the four hook
commands are marked `===== INTEGRATION HOOK` in `src/app/main.cpp`.

## Re-running the checks

```
ninja -C build && (cd build && ctest -R test_tok --output-on-failure)
python3 tools/tok_oracle_check.py [--export-dir tests/data/tok]   # needs llama-tokenize; split 00001 only (vocab-only load, ~0.2 s, 135 MB RSS)
<venv-with-jinja2>/bin/python tools/chat_check.py --n 2000        # encoding.py + GGUF jinja template
python3 tools/gen_unicode_table.py > src/tok/unicode_table.inc    # regenerate the category table
```
Note `llama-tokenize -f` applies escape processing unless `--no-escape` is given (the check script passes it).

## Known limits / notes for integration
- Equal-length special tokens that overlap in text are ordered by libc++'s `std::sort` exactly like llama.cpp
  (same comparator, same input order); with this vocab no such overlaps exist.
- `encode()` is const and thread-safe (per-call scratch/cache).
- EOG = eos (id 1) only, same as llama.cpp for this vocab. `<|EOT|>` (128805) is a CONTROL token but not EOG.
