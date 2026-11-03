#pragma once
// DeepSeek-V4.1-Flash forward pass (SPEC.md §3) on top of the module kernels.
// Decode: one command buffer per layer boundary (router -> CPU -> expert cache -> MoE), see docs/integration.md.
// Prefill: chunks of <= max_chunk tokens; MoE experts processed in groups that fit the expert cache.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "core/metal.hpp"
#include "model/config.hpp"

namespace ms {
class Gguf; class Weights; class TokenEmbd; class ExpertCache; class EngramFetcher; struct EngramHashParams; class EngramHasher;
}

namespace ms::ds41 {

struct ModelOptions {
  std::string model_path;      // any split; "" = Gguf::default_model_path()
  double arena_gb = 0;         // expert arena; 0 = auto (memguard, <= 40 GiB). env MACSEEK_ARENA_GB
  int max_chunk = 4096;        // prefill chunk (<= attn max_chunk)
  int max_logits_rows = 1024;  // rows of full logits a prefill call may return
  bool actq_out = true;        // fp8 sim of the attention output before wo_a (reference act_quant). env MACSEEK_ACTQ_OUT
  int moe_group = 64;          // prefill: experts per MoE group (must fit the cache comfortably). env MACSEEK_MOE_GROUP
  bool verbose = true;
  static ModelOptions from_env();
};

struct DecodeProfile {         // accumulated over decode steps (CPU wall times, ms; gpu_ms = sum of command-buffer GPU time)
  int tokens = 0;
  double total_ms = 0, gpu_wait_ms = 0, expert_ms = 0, encode_ms = 0, engram_ms = 0, gpu_ms = 0, io_stall_ms = 0;
  uint64_t io_bytes = 0;
  void reset() { *this = DecodeProfile{}; }
};
struct PrefillProfile {        // accumulated over prefill chunks
  int chunks = 0; long tokens = 0;
  double total_ms = 0, engram_ms = 0, main_gpu_ms = 0, main_wait_ms = 0, moe_ms = 0, moe_gpu_ms = 0, route_ms = 0;
  uint64_t io_bytes = 0; long experts_used = 0, groups = 0;
  void reset() { *this = PrefillProfile{}; }
  std::string str() const;
};

// In-memory copy of the sequence state (attention caches, compressor state, engram hasher, position): a server
// keeps one per conversation prefix to skip re-prefilling it.
struct MemState {
  int pos = 0;
  std::vector<char> hasher;
  std::vector<std::vector<char>> spans;
  size_t bytes() const { size_t b = hasher.size(); for (auto& s : spans) b += s.size(); return b; }
};

class Model {
 public:
  explicit Model(const ModelOptions& o = ModelOptions::from_env());
  ~Model();
  const Config& config() const { return cfg_; }
  int pos() const { return pos_; }
  void reset();

  // Append n tokens. logits_all: [n][vocab] F32 for every position (n <= max_logits_rows) or nullptr.
  // logits_last: [vocab] for the last position or nullptr. Returns the greedy next token after the last position.
  int32_t prefill(const int32_t* toks, int n, float* logits_all = nullptr, float* logits_last = nullptr);
  // Append one token; returns the greedy (temperature 0) or sampled next token. logits: [vocab] or nullptr.
  int32_t decode(int32_t tok, float* logits = nullptr, float temperature = 0.f, uint64_t seed = 0);

  bool save_state(const std::string& path) const;
  bool load_state(const std::string& path);
  std::shared_ptr<MemState> snapshot() const;
  bool restore(const MemState& s);
  int max_pos() const;
  // called after each prefill chunk with (tokens done, tokens total) of the current prefill() call
  void set_progress(std::function<void(int, int)> f) { progress_ = std::move(f); }

  // ---- speculative decoding (docs/speculative.md) ----
  // Exact multi-token verify: toks[0..k) (1 <= k <= 8) are appended at positions pos()..pos()+k-1 by kernels that
  // reproduce decode() bit for bit per token (same per-lane arithmetic; weights fetched once for all k).
  // next[j] = greedy token after toks[j]; logits (optional) [k][vocab]. The state advances by k provisionally:
  // verify_keep(m) keeps toks[0..m) (1 <= m <= k) and rolls the rest back exactly. Requires the default decode
  // configuration (MACSEEK_DEC_* defaults, MACSEEK_DEC_SYNC=shared); verify_supported() says why not otherwise.
  void verify(const int32_t* toks, int k, int32_t* next, float* logits = nullptr);
  void verify_keep(int m);
  static bool verify_supported(std::string* why = nullptr);
  // dspark target features of the last verify: [3][k][5120] (mean over hc copies of the attention input of layers
  // 37, 38, 39 for each verified token); valid until the next verify
  const float* verify_features() const;
  struct SpecProfile {
    long passes = 0, tokens = 0;   // verify passes, tokens fed
    double total_ms = 0, gpu_ms = 0, wait_ms = 0, expert_ms = 0, io_stall_ms = 0, keep_ms = 0, engram_ms = 0;
    uint64_t io_bytes = 0;
    void reset() { *this = SpecProfile{}; }
  };
  SpecProfile& spec_profile() { return sprof_; }

  DecodeProfile& profile() { return prof_; }
  PrefillProfile& prefill_profile() { return pprof_; }
  std::string stats_report() const;
  ExpertCache& experts() { return *cache_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  Config cfg_;
  ModelOptions o_;
  int pos_ = 0;
  DecodeProfile prof_;
  PrefillProfile pprof_;
  SpecProfile sprof_;
  const int32_t* next_toks_ = nullptr;   // prefill: the next chunk (engram row prefetch)
  int next_n_ = 0;
  long pinned_ = 0;
  double pf_busy_ms_ = 0, pf_span_ms_ = 0;   // last prefill call: GPU busy (union of CBs) vs first-start..last-end
  std::function<void(int, int)> progress_;   // experts pinned by a multi-chunk prefill (released at the next decode)
  int32_t forward(const int32_t* toks, int n, float* logits_all, float* logits_last, float temperature, uint64_t seed);
  std::unique_ptr<Gguf> g_;
  std::unique_ptr<Weights> w_;
  std::unique_ptr<TokenEmbd> te_;
  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<EngramFetcher> ef_;
  std::unique_ptr<EngramHashParams> ehp_;
  std::unique_ptr<EngramHasher> hasher_;
};

}  // namespace ms::ds41
