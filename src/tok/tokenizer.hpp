#pragma once
// MacSeek tokenizer: GPT-2 byte-level BPE as stored in the GGUF (tokenizer.ggml.model = gpt2), pre-tokenizer
// `joyai-llm` (DeepSeek V3/V4 family), special/added tokens matched before BPE.
//
// Behaviour contract: ids are identical to llama.cpp's llama_tokenize(parse_special=true, add_special=false)
// (verified with Homebrew llama-tokenize, see docs/modules/tok.md), including
//  - special-token partition order (longest token text first, then leftmost, non-overlapping),
//  - invalid UTF-8 bytes -> U+FFFD, Unicode categories of llama.cpp's tables (Unicode 15.1),
//  - the three-regex split \p{N}{1,3} | [CJK]+ | <main regex> applied in sequence.
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ms::tok {

// GGUF tokenizer.ggml.token_type values (llama.cpp convention).
enum TokenType : uint8_t {
  kTokUndefined = 0,
  kTokNormal = 1,
  kTokUnknown = 2,
  kTokControl = 3,
  kTokUserDefined = 4,
  kTokUnused = 5,
  kTokByte = 6,
};

// Everything the tokenizer needs from the GGUF metadata. This is the swap point for the shared src/gguf parser:
// fill a VocabData from it and call Tokenizer::init().
struct VocabData {
  std::string model;  // tokenizer.ggml.model (must be "gpt2")
  std::string pre;    // tokenizer.ggml.pre (must be "joyai-llm" / "deepseek-v3" compatible)
  std::vector<std::string> tokens;
  std::vector<int32_t> types;
  std::vector<std::string> merges;
  int32_t bos = -1, eos = -1, pad = -1;
  bool add_bos = false, add_eos = false;
  std::string chat_template;
};

// Private minimal GGUF metadata reader (only the tokenizer.* keys). Reads split 00001 in ~20 ms.
bool read_gguf_vocab(const std::string& path, VocabData& out, std::string* err = nullptr);

class Tokenizer {
 public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  static std::unique_ptr<Tokenizer> load_from_gguf(const std::string& path, std::string* err = nullptr);
  bool init(VocabData data, std::string* err = nullptr);

  // Text -> ids. parse_special=false still matches user-defined added tokens (<think>, ｜DSML｜, ...) like llama.cpp.
  std::vector<int32_t> encode(std::string_view text, bool parse_special = true, bool add_bos = false) const;
  void encode_append(std::string_view text, std::vector<int32_t>& out, bool parse_special = true) const;

  // Raw bytes of one token. Normal tokens: byte-level decoded. Control tokens: their text if special=true else "".
  // User-defined tokens: their text. Unused/unknown/out of range: "".
  std::string_view piece(int32_t id, bool special = true) const;
  std::string decode(const int32_t* ids, size_t n, bool special = true) const;
  std::string decode(const std::vector<int32_t>& ids, bool special = true) const {
    return decode(ids.data(), ids.size(), special);
  }

  int32_t find_token(std::string_view text) const;  // exact token text -> id, -1 if none
  const std::string& token_text(int32_t id) const;  // stored (byte-level) text
  TokenType type(int32_t id) const;
  bool is_control(int32_t id) const { return type(id) == kTokControl; }
  bool is_eog(int32_t id) const;  // end of generation (eos)

  int32_t n_vocab() const;
  int32_t bos() const;
  int32_t eos() const;
  bool add_bos() const;
  const std::string& chat_template() const;
  const std::string& pre() const;

  // Pre-tokenizer only (for tests): split raw text (no special tokens) into pieces.
  std::vector<std::string> pretokenize(std::string_view text) const;

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

// Streaming detokenizer: push ids, get back only complete UTF-8. Bytes of an incomplete multi-byte sequence are
// held until completed; invalid bytes become U+FFFD. flush() emits whatever is pending (as U+FFFD).
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& t, bool special = true) : t_(t), special_(special) {}
  std::string push(int32_t id);
  std::string push_bytes(std::string_view bytes);
  std::string flush();
  void reset() { pending_.clear(); }
  size_t pending() const { return pending_.size(); }

 private:
  const Tokenizer& t_;
  bool special_;
  std::string pending_;
};

// Utilities shared with chat/server.
// Length of the longest prefix of s that is complete valid UTF-8 (stops before a truncated tail sequence).
size_t utf8_complete_prefix(std::string_view s);
// Replace invalid UTF-8 with U+FFFD (strict: no surrogates/overlongs).
std::string utf8_sanitize(std::string_view s);

// Default model path (split 00001 holds metadata + tokenizer). Overridable with env MACSEEK_GGUF.
std::string default_gguf_path();

}  // namespace ms::tok
