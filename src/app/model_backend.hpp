#pragma once
// Model-backed generation for `macseek serve`: prefix reuse of the sequence state, CPU sampling with the common
// OpenAI / llama.cpp parameters, live status for GET /v1/stats.
#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "model/ds41.hpp"
#include "server/http.hpp"
#include "tok/tokenizer.hpp"

namespace ms {

class ModelBackend : public server::Backend {
 public:
  ModelBackend(const tok::Tokenizer& t, std::string id);
  std::string model_id() const override { return id_; }
  server::Finish generate(const std::vector<int32_t>& prompt, const server::GenParams& p,
                          const std::function<bool(int32_t)>& emit) override;
  server::GenStats last_stats() const override;
  std::string status_json() const override;
  int max_context() const override { return max_ctx_; }

 private:
  int32_t sample(std::vector<float>& lg, const server::GenParams& p, const std::vector<int32_t>& gen);

  const tok::Tokenizer& tok_;
  std::string id_;
  std::unique_ptr<ds41::Model> m_;
  int max_ctx_ = 0;
  std::vector<int32_t> hist_;                // tokens in the live sequence state
  std::shared_ptr<ds41::MemState> snap_;     // state right after the last prompt's prefill
  std::vector<int32_t> snap_toks_;
  std::vector<int32_t> eog_;
  int32_t assistant_id_ = -1;
  std::mt19937_64 rng_{0x9E3779B97F4A7C15ull};
  std::vector<std::pair<float, int32_t>> cand_;
  std::vector<float> probs_;

  mutable std::mutex smu_;
  server::GenStats last_;
  std::string phase_ = "idle";               // idle | prefill | decode
  int pf_done_ = 0, pf_total_ = 0, gen_n_ = 0;
  double gen_ms_ = 0;
  long long requests_ = 0;
};

}  // namespace ms
