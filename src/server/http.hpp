#pragma once
// Minimal dependency-free OpenAI-compatible HTTP/1.1 server (skeleton).
//
//   GET  /v1/models             -> {"object":"list","data":[{"id":<model>,...}]}
//   GET  /health                -> {"status":"ok"}
//   POST /v1/chat/completions   -> chat.completion | SSE chat.completion.chunk stream ("stream": true)
//   POST /v1/completions        -> text_completion | SSE stream
//
// Generation is delegated to a Backend (integration implements it on top of the model; EchoBackend is a dummy
// used by tests and `macseek serve --echo`). The server owns prompt formatting (ms::chat), tokenization,
// incremental detokenization (UTF-8 safe), stop strings, EOG handling and reply parsing (reasoning / tool calls).
// One generation at a time (the backend is guarded by a mutex); connections are served on their own threads,
// keep-alive supported, streamed responses use chunked transfer encoding.
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tok/tokenizer.hpp"

namespace ms::server {

struct GenParams {
  int max_tokens = 1024;
  double temperature = 1.0;
  double top_p = 1.0;
  int top_k = 0;
  double min_p = 0.0;
  double presence_penalty = 0.0;    // OpenAI: logit -= presence * (count > 0), over generated tokens
  double frequency_penalty = 0.0;   // OpenAI: logit -= frequency * count, over generated tokens
  double repetition_penalty = 1.0;  // llama.cpp style over the last repeat_last_n tokens (prompt + generated)
  int repeat_last_n = 64;
  bool has_seed = false;
  uint64_t seed = 0;
  bool ignore_eos = false;
  std::vector<std::pair<int32_t, float>> logit_bias;
};

// Timings of the last generate() call (llama.cpp-style "timings" object in responses).
struct GenStats {
  int prompt_n = 0;      // prompt tokens
  int cached_n = 0;      // prompt tokens reused from the cached sequence state (not recomputed)
  double prompt_ms = 0;  // prefill wall time of the remaining prompt tokens
  int predicted_n = 0;
  double predicted_ms = 0;
};

enum class Finish { Stop, Length, Cancelled, Error };

// Generation backend interface (integration hook).
class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string model_id() const = 0;
  // Continue `prompt`. Call emit(token) for every sampled token (including a final EOG token, if any); stop as
  // soon as emit returns false (client gone / stop string / limit reached) and return Cancelled. Return Stop after
  // an EOG token, Length when p.max_tokens tokens were produced.
  virtual Finish generate(const std::vector<int32_t>& prompt, const GenParams& p,
                          const std::function<bool(int32_t)>& emit) = 0;
  virtual GenStats last_stats() const { return {}; }
  // live engine state for GET /v1/stats (JSON object text); "{}" if none
  virtual std::string status_json() const { return "{}"; }
  virtual int max_context() const { return 0; }
};

// Dummy backend: replies with the text of the last user turn (the tokens between the last <｜User｜> and the next
// control token), preceded by </think> when the prompt ends in <think>, followed by EOS.
class EchoBackend : public Backend {
 public:
  explicit EchoBackend(const tok::Tokenizer& t, std::string id = "macseek-echo") : t_(t), id_(std::move(id)) {}
  std::string model_id() const override { return id_; }
  Finish generate(const std::vector<int32_t>& prompt, const GenParams& p,
                  const std::function<bool(int32_t)>& emit) override;

 private:
  const tok::Tokenizer& t_;
  std::string id_;
};

struct HttpRequest {
  std::string method, path, query, version;
  std::map<std::string, std::string> headers;  // lower-case keys
  std::string body;
  std::string header(const std::string& k) const {
    auto it = headers.find(k);
    return it == headers.end() ? "" : it->second;
  }
};

struct ServerConfig {
  std::string host = "127.0.0.1";
  int port = 8080;  // 0 = ephemeral (see Server::port())
  std::string model_name;  // default: backend.model_id()
  int default_max_tokens = 1024;
  std::string ui_path;     // GET / serves this HTML file (read per request); "" = no UI
  size_t max_body_bytes = 64u << 20;
};

class Server {
 public:
  Server(const tok::Tokenizer& t, Backend& b, ServerConfig cfg);
  ~Server();
  bool start(std::string* err = nullptr);  // bind + listen + accept thread
  void stop();                             // stop accepting, wait for connections to finish
  void wait();                             // block until stop() (e.g. from a signal handler)
  int port() const { return port_; }

 private:
  void accept_loop();
  void serve_conn(int fd);
  bool handle(int fd, const HttpRequest& rq, bool keep_alive);
  bool handle_chat(int fd, const HttpRequest& rq, bool keep_alive);
  bool handle_completion(int fd, const HttpRequest& rq, bool keep_alive);

  const tok::Tokenizer& tok_;
  Backend& be_;
  ServerConfig cfg_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread accept_thr_;
  std::atomic<bool> stopping_{false};
  std::atomic<int> active_{0};
  std::atomic<uint64_t> req_counter_{0};
  std::mutex gen_mu_;
  std::mutex stop_mu_;
};

// Low-level helpers (exposed for tests).
bool read_request(int fd, std::string& buf, HttpRequest& rq, size_t max_body, std::string* err);
bool send_all(int fd, const void* p, size_t n);

}  // namespace ms::server
