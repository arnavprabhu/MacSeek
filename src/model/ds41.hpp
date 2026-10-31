#pragma once
// DeepSeek-V4.1-Flash forward pass (SPEC.md §3) on top of the module kernels.
// Decode: one command buffer per layer boundary (router -> CPU -> expert cache -> MoE), see docs/integration.md.
// Prefill: chunks of <= max_chunk tokens; MoE experts processed in groups that fit the expert cache.
#include <cstdint>
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
  const int32_t* next_toks_ = nullptr;   // prefill: the next chunk (engram row prefetch)
  int next_n_ = 0;
  long pinned_ = 0;   // experts pinned by a multi-chunk prefill (released at the next decode)
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
