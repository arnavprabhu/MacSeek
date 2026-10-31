#pragma once
// Engram n-gram hashing (CPU), mirrors reference/inference/engram.py NgramHashState with the GGUF metadata:
//   deepseek41.engram.token_map [129280]   raw token id -> compressed id
//   deepseek41.engram.pad_token_id (2)     pad = token_map[pad_token_id]
//   deepseek41.ple.layer_multipliers [8]   [layer][shift], layer in {1, 14}, shift 0..3
//   deepseek41.ple.head_vocab_sizes [48]   primes, order [layer][ngram 2..4][head 0..7]
//   deepseek41.ple.head_offsets [48]       GLOBAL row offsets into per_layer_token_embd (layer 14 starts at 384006168)
// For each position p and engram layer l: tokens c[s] = compressed(tok[p-s]) for s = 0..3 (pad once a shift reaches
// before the sequence start or a dead token -- and for every larger shift), rolling = c[0]*m[l][0] ^ ... ^ c[n-1]*m[l][n-1]
// (uint64, cannot overflow), row[l][(n-2)*8 + k] = rolling % prime[l][n-2][k] + offset[l][n-2][k].
// The hasher is streaming: the state (last 3 compressed tokens) carries across prefill chunks and decode steps.
#include <cstdint>
#include <vector>

namespace ms {

struct EngramHashParams {
  static constexpr int kLayers = 2, kMaxNgram = 4, kHeads = 8, kCols = (kMaxNgram - 1) * kHeads;  // 24 per layer
  std::vector<int32_t> token_map;          // [vocab]
  int32_t pad_id = 2;                      // compressed pad id (token_map[pad_token_id])
  uint64_t mult[kLayers][kMaxNgram] = {};
  uint64_t prime[kLayers * kCols] = {};
  uint64_t offset[kLayers * kCols] = {};
  // Builds from the raw metadata arrays (as stored in the GGUF). Returns false on a size mismatch.
  bool init(const std::vector<int64_t>& token_map, int64_t pad_token_id, const std::vector<int64_t>& layer_multipliers,
            const std::vector<int64_t>& head_vocab_sizes, const std::vector<int64_t>& head_offsets);
};

class EngramHasher {
 public:
  static constexpr int32_t kNone = -2;   // history slot before the sequence start
  static constexpr int32_t kDead = -1;   // masked token (image span): blocks n-grams through it
  struct State {
    int32_t hist[3] = {kNone, kNone, kNone};  // hist[0] = previous token (compressed), hist[1] = the one before, ...
    int64_t pos = 0;                          // tokens consumed so far
  };

  explicit EngramHasher(const EngramHashParams* p) : p_(p) {}
  void reset() { st_ = State{}; }
  const State& state() const { return st_; }
  void set_state(const State& s) { st_ = s; }
  // Rebuild the state from the last tokens of an existing sequence (raw ids; up to 3 used) and its total length.
  void set_history(const int32_t* last_tokens, int n_last, int64_t total_len);

  // Hash n new tokens (raw ids). out: [n][2][24] uint32 global rows of per_layer_token_embd. mask (optional):
  // false = dead token (no engram, blocks n-grams). Advances the state.
  void hash(const int32_t* tokens, int n, uint32_t* out, const bool* mask = nullptr);
  // Same, without touching the state (e.g. draft tokens): starts from `from`, returns the advanced state.
  State hash_from(const State& from, const int32_t* tokens, int n, uint32_t* out, const bool* mask = nullptr) const;

  static constexpr int kRowsPerToken = EngramHashParams::kLayers * EngramHashParams::kCols;  // 48

 private:
  const EngramHashParams* p_;
  State st_;
};

}  // namespace ms
