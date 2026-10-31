#include "app/model_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

#include "core/timer.hpp"
#include "store/experts.hpp"
#include "tok/json.hpp"

namespace ms {

using server::Finish;
using server::GenParams;

ModelBackend::ModelBackend(const tok::Tokenizer& t, std::string id) : tok_(t), id_(std::move(id)) {
  m_ = std::make_unique<ds41::Model>();
  max_ctx_ = m_->max_pos();
  assistant_id_ = tok_.find_token("<｜Assistant｜>");
  for (int32_t i = 0; i < tok_.n_vocab(); i++)
    if (tok_.is_eog(i)) eog_.push_back(i);
  m_->set_progress([this](int done, int total) {
    std::lock_guard<std::mutex> g(smu_);
    pf_done_ = done; pf_total_ = total;
  });
}

server::GenStats ModelBackend::last_stats() const {
  std::lock_guard<std::mutex> g(smu_);
  return last_;
}

std::string ModelBackend::status_json() const {
  std::lock_guard<std::mutex> g(smu_);
  json::Value r = json::Value::object();
  r.set("model", json::Value(id_));
  r.set("phase", json::Value(phase_));
  r.set("prefill_done", json::Value((long long)pf_done_));
  r.set("prefill_total", json::Value((long long)pf_total_));
  r.set("generated", json::Value((long long)gen_n_));
  r.set("decode_tok_s", json::Value(gen_ms_ > 0 ? gen_n_ / (gen_ms_ / 1e3) : 0.0));
  r.set("context_used", json::Value((long long)hist_.size()));
  r.set("context_max", json::Value((long long)max_ctx_));
  r.set("requests", json::Value(requests_));
  json::Value l = json::Value::object();
  l.set("prompt_n", json::Value((long long)last_.prompt_n));
  l.set("cache_n", json::Value((long long)last_.cached_n));
  l.set("prompt_ms", json::Value(last_.prompt_ms));
  l.set("predicted_n", json::Value((long long)last_.predicted_n));
  l.set("predicted_ms", json::Value(last_.predicted_ms));
  r.set("last", std::move(l));
  return r.dump();
}

static bool is_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {   // a is a prefix of b
  return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

int32_t ModelBackend::sample(std::vector<float>& lg, const GenParams& p, const std::vector<int32_t>& gen) {
  const int V = (int)lg.size();
  for (auto& [id, b] : p.logit_bias) if (id >= 0 && id < V) lg[id] += b;
  if (p.ignore_eos) for (int32_t e : eog_) lg[e] = -INFINITY;
  // repetition penalty (llama.cpp): over the last repeat_last_n tokens of the sequence (prompt + generated)
  if (p.repetition_penalty != 1.0 && p.repeat_last_n != 0) {
    const size_t n = p.repeat_last_n < 0 ? hist_.size() : std::min(hist_.size(), (size_t)p.repeat_last_n);
    std::vector<int32_t> seen(hist_.end() - n, hist_.end());
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    for (int32_t id : seen)
      if (id >= 0 && id < V) lg[id] = lg[id] > 0 ? lg[id] / (float)p.repetition_penalty : lg[id] * (float)p.repetition_penalty;
  }
  // presence / frequency penalties (OpenAI): over the tokens generated in this request
  if (p.presence_penalty != 0.0 || p.frequency_penalty != 0.0) {
    std::unordered_map<int32_t, int> cnt;
    for (int32_t id : gen) cnt[id]++;
    for (auto& [id, c] : cnt)
      if (id >= 0 && id < V) lg[id] -= (float)(p.frequency_penalty * c + p.presence_penalty);
  }
  int best = 0;
  for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
  if (p.temperature <= 0.0 || p.top_k == 1) return best;
  const float inv_t = 1.0f / (float)p.temperature;
  const float mx = lg[best];
  // candidates: top_k if set, else the top 4096 by logit (enough for any practical top_p / min_p); the full
  // vocabulary only when no truncation is requested
  const bool truncate = p.top_k > 0 || p.top_p < 1.0 || p.min_p > 0.0;
  cand_.clear();
  if (!truncate) {
    probs_.resize(V);
    double z = 0;
    for (int i = 0; i < V; i++) { probs_[i] = std::exp((lg[i] - mx) * inv_t); z += probs_[i]; }
    double r = std::uniform_real_distribution<double>(0, z)(rng_), acc = 0;
    for (int i = 0; i < V; i++) { acc += probs_[i]; if (acc >= r) return i; }
    return best;
  }
  int K = p.top_k > 0 ? std::min(p.top_k, V) : std::min(4096, V);
  cand_.reserve(V);
  for (int i = 0; i < V; i++) if (std::isfinite(lg[i])) cand_.push_back({lg[i], i});
  K = std::min<int>(K, (int)cand_.size());
  if (K <= 0) return best;
  std::partial_sort(cand_.begin(), cand_.begin() + K, cand_.end(), [](auto& a, auto& b) { return a.first > b.first; });
  cand_.resize(K);
  std::vector<double> pr(K);
  double z = 0;
  for (int i = 0; i < K; i++) { pr[i] = std::exp((cand_[i].first - mx) * inv_t); z += pr[i]; }
  for (auto& x : pr) x /= z;
  int keep = K;
  if (p.min_p > 0.0) {
    const double th = p.min_p * pr[0];
    for (int i = 1; i < keep; i++) if (pr[i] < th) { keep = i; break; }
  }
  if (p.top_p < 1.0) {
    double acc = 0;
    for (int i = 0; i < keep; i++) { acc += pr[i]; if (acc >= p.top_p) { keep = i + 1; break; } }
  }
  double tot = 0;
  for (int i = 0; i < keep; i++) tot += pr[i];
  double r = std::uniform_real_distribution<double>(0, tot)(rng_), acc = 0;
  for (int i = 0; i < keep; i++) { acc += pr[i]; if (acc >= r) return cand_[i].second; }
  return cand_[0].second;
}

Finish ModelBackend::generate(const std::vector<int32_t>& prompt, const GenParams& p,
                              const std::function<bool(int32_t)>& emit) {
  {
    std::lock_guard<std::mutex> g(smu_);
    requests_++; phase_ = "prefill"; pf_done_ = 0; pf_total_ = 0; gen_n_ = 0; gen_ms_ = 0;
    last_ = server::GenStats{};
    last_.prompt_n = (int)prompt.size();
  }
  auto set_phase = [&](const char* ph) { std::lock_guard<std::mutex> g(smu_); phase_ = ph; };
  if (prompt.empty()) { set_phase("idle"); return Finish::Error; }
  if ((int)prompt.size() >= max_ctx_) {
    fprintf(stderr, "serve: prompt of %zu tokens exceeds the context (%d)\n", prompt.size(), max_ctx_);
    set_phase("idle");
    return Finish::Error;
  }
  if (p.has_seed) rng_.seed(p.seed);
  // reuse the longest cached prefix: the live state (previous prompt + reply) or the snapshot taken right after the
  // previous prompt's prefill (chat templates drop earlier reasoning, so the live state often diverges)
  size_t live = is_prefix(hist_, prompt) && hist_.size() < prompt.size() ? hist_.size() : 0;
  size_t sn = snap_ && is_prefix(snap_toks_, prompt) && snap_toks_.size() < prompt.size() ? snap_toks_.size() : 0;
  size_t cached = 0;
  if (live > 0 && live >= sn) {
    cached = live;
  } else if (sn > 0 && m_->restore(*snap_)) {
    hist_ = snap_toks_;
    cached = sn;
  } else {
    m_->reset();
    hist_.clear();
  }
  const int V = m_->config().vocab;
  std::vector<float> lg(V);
  Timer tp;
  // Snapshot point: right after the last <｜Assistant｜> of the prompt. The next turn re-renders this assistant turn
  // with its reply (and without the generation prompt's <think>), so everything up to here is shared; the live
  // state (which also holds the generated reasoning) usually is not.
  size_t cut = 0;
  if (assistant_id_ >= 0)
    for (size_t i = prompt.size(); i-- > 0;) if (prompt[i] == assistant_id_) { cut = i + 1; break; }
  if (cut > cached && cut < prompt.size()) {
    m_->prefill(prompt.data() + cached, (int)(cut - cached));
    snap_ = m_->snapshot();
    snap_toks_.assign(prompt.begin(), prompt.begin() + cut);
    m_->prefill(prompt.data() + cut, (int)(prompt.size() - cut), nullptr, lg.data());
  } else {
    m_->prefill(prompt.data() + cached, (int)(prompt.size() - cached), nullptr, lg.data());
    if (cut == 0) { snap_ = m_->snapshot(); snap_toks_ = prompt; }
  }
  const double pf_ms = tp.ms();
  hist_ = prompt;
  {
    std::lock_guard<std::mutex> g(smu_);
    last_.cached_n = (int)cached;
    last_.prompt_ms = pf_ms;
    phase_ = "decode";
  }
  fprintf(stderr, "serve: prompt %zu tokens (%zu cached) prefill %.0f ms\n", prompt.size(), cached, pf_ms);
  std::vector<int32_t> gen;
  Timer td;
  Finish fin = Finish::Length;
  for (int i = 0; p.max_tokens < 0 || i < p.max_tokens; i++) {
    int32_t t = sample(lg, p, gen);
    gen.push_back(t);
    const bool eog = tok_.is_eog(t);
    {
      std::lock_guard<std::mutex> g(smu_);
      gen_n_ = (int)gen.size(); gen_ms_ = td.ms();
      last_.predicted_n = gen_n_; last_.predicted_ms = gen_ms_;
    }
    if (!emit(t)) { fin = Finish::Cancelled; break; }
    if (eog) { fin = Finish::Stop; break; }
    if ((int)hist_.size() + 1 >= max_ctx_) { fin = Finish::Length; break; }
    if (p.max_tokens >= 0 && i + 1 >= p.max_tokens) { fin = Finish::Length; break; }
    m_->decode(t, lg.data());
    hist_.push_back(t);
  }
  {
    std::lock_guard<std::mutex> g(smu_);
    last_.predicted_ms = td.ms();
    phase_ = "idle";
  }
  fprintf(stderr, "serve: generated %zu tokens in %.0f ms (%.2f tok/s)\n", gen.size(), td.ms(), gen.size() / (td.ms() / 1e3));
  return fin;
}

std::unique_ptr<server::Backend> make_model_backend(const tok::Tokenizer& t, const std::string& id) {
  return std::make_unique<ModelBackend>(t, id);
}

}  // namespace ms
