#pragma once
// DeepSeek-V4.1 chat formatting, hard-coded (no jinja engine).
//
// Normative reference: reference/encoding/encoding.py (DeepSeek's encoder; byte-exact on its test vectors in
// tests/data/chat). Where the GGUF jinja template (tokenizer.chat_template, "Unsloth port of DS41") is more
// permissive it is followed too: reasoning_effort "medium" (=62), top-level tools/response_format attached to a
// leading system message (created empty if missing), `thinking` as an alias of `reasoning_content`, bare tool
// definitions without a "function" wrapper, add_generation_prompt=false.
#include <string>
#include <string_view>

#include "tok/json.hpp"

namespace ms::chat {

struct Options {
  std::string thinking_mode = "thinking";  // "thinking" | "chat"
  bool drop_thinking = true;               // drop reasoning of turns before the last user turn (off if tools)
  json::Value reasoning_effort;            // null -> "high"; "low"|"medium"|"high"|"max" or int in [1,100]
  bool add_generation_prompt = true;       // append <｜Assistant｜><think> / </think> after a final user turn
  bool add_bos = true;                     // prefix <｜begin▁of▁sentence｜>
};

// Render `messages` (OpenAI-style array). `tools` / `response_format` (top-level, may be null) are attached to the
// leading system message. Throws nothing; returns false and sets *err on invalid input.
bool format(const json::Value& messages, const json::Value& tools, const json::Value& response_format,
            const Options& opt, std::string& out, std::string* err = nullptr);

// Same as format() but takes an OpenAI chat-completions request body. Recognised extensions (top level or inside
// "chat_template_kwargs"): thinking_mode, enable_thinking, thinking (bool), reasoning_effort, drop_thinking,
// add_generation_prompt. `resolved` (optional) receives the options used.
bool format_request(const json::Value& body, std::string& out, std::string* err = nullptr,
                    Options* resolved = nullptr);

// Budget for a reasoning_effort value (returns -1 if invalid).
int effort_budget(const json::Value& effort);

// Parse a generated assistant completion (text that follows the generation prompt) into
// reasoning / content / OpenAI tool_calls. Lenient: a missing </think> means everything is reasoning (truncated),
// malformed DSML falls back to plain content.
struct Reply {
  std::string reasoning;
  std::string content;
  json::Value tool_calls = json::Value::array();  // [{"type":"function","function":{"name","arguments"}}]
};
Reply parse_reply(std::string_view text, bool thinking);

// Special strings
inline constexpr const char* kBos = "<｜begin▁of▁sentence｜>";
inline constexpr const char* kEos = "<｜end▁of▁sentence｜>";
inline constexpr const char* kThinkStart = "<think>";
inline constexpr const char* kThinkEnd = "</think>";
inline constexpr const char* kDsml = "｜DSML｜";

}  // namespace ms::chat
