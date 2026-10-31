#include "server/http.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>

#include "tok/chat.hpp"
#include "tok/json.hpp"

namespace ms::server {

using json::Value;

// ---------------------------------------------------------------------------------------------------------------
// Echo backend
// ---------------------------------------------------------------------------------------------------------------

Finish EchoBackend::generate(const std::vector<int32_t>& prompt, const GenParams& p,
                             const std::function<bool(int32_t)>& emit) {
  const int32_t user = t_.find_token("<｜User｜>");
  const int32_t think = t_.find_token("<think>");
  const int32_t think_end = t_.find_token("</think>");
  std::vector<int32_t> out;
  if (!prompt.empty() && prompt.back() == think && think_end >= 0) out.push_back(think_end);
  size_t s = 0;
  for (size_t i = prompt.size(); i-- > 0;)
    if (prompt[i] == user) { s = i + 1; break; }
  for (size_t i = s; i < prompt.size() && !t_.is_control(prompt[i]); i++) out.push_back(prompt[i]);
  out.push_back(t_.eos());
  int n = 0;
  for (int32_t id : out) {
    if (p.max_tokens >= 0 && n >= p.max_tokens) return Finish::Length;
    if (!emit(id)) return Finish::Cancelled;
    n++;
    if (t_.is_eog(id)) return Finish::Stop;
  }
  return Finish::Stop;
}

// ---------------------------------------------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------------------------------------------

bool send_all(int fd, const void* p, size_t n) {
  const char* c = (const char*)p;
  while (n) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    c += k;
    n -= (size_t)k;
  }
  return true;
}
static bool send_str(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

static std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}
static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

bool read_request(int fd, std::string& buf, HttpRequest& rq, size_t max_body, std::string* err) {
  auto fill = [&]() {
    char tmp[65536];
    ssize_t k;
    do { k = ::recv(fd, tmp, sizeof tmp, 0); } while (k < 0 && errno == EINTR);
    if (k <= 0) return false;
    buf.append(tmp, (size_t)k);
    return true;
  };
  size_t hdr_end;
  while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
    if (buf.size() > (1u << 20)) { if (err) *err = "header too large"; return false; }
    if (!fill()) { if (err) *err = buf.empty() ? "" : "connection closed in headers"; return false; }
  }
  std::string head = buf.substr(0, hdr_end);
  size_t le = head.find("\r\n");
  std::string line = head.substr(0, le);
  rq = HttpRequest();
  size_t a = line.find(' '), b = line.rfind(' ');
  if (a == std::string::npos || b == a) { if (err) *err = "bad request line"; return false; }
  rq.method = line.substr(0, a);
  std::string target = line.substr(a + 1, b - a - 1);
  rq.version = line.substr(b + 1);
  size_t qm = target.find('?');
  rq.path = target.substr(0, qm);
  if (qm != std::string::npos) rq.query = target.substr(qm + 1);
  size_t pos = le == std::string::npos ? head.size() : le + 2;
  while (pos < head.size()) {
    size_t e = head.find("\r\n", pos);
    if (e == std::string::npos) e = head.size();
    std::string h = head.substr(pos, e - pos);
    size_t c = h.find(':');
    if (c != std::string::npos) rq.headers[lower(trim(h.substr(0, c)))] = trim(h.substr(c + 1));
    pos = e + 2;
  }
  if (lower(rq.header("transfer-encoding")).find("chunked") != std::string::npos) {
    if (err) *err = "chunked request bodies not supported";
    return false;
  }
  size_t clen = 0;
  std::string cl = rq.header("content-length");
  if (!cl.empty()) clen = (size_t)strtoull(cl.c_str(), nullptr, 10);
  if (clen > max_body) { if (err) *err = "body too large"; return false; }
  size_t body_start = hdr_end + 4;
  while (buf.size() < body_start + clen)
    if (!fill()) { if (err) *err = "connection closed in body"; return false; }
  rq.body = buf.substr(body_start, clen);
  buf.erase(0, body_start + clen);
  return true;
}

static const char* status_text(int code) {
  switch (code) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    default: return "Error";
  }
}

static bool send_response(int fd, int code, const std::string& body, bool keep_alive,
                          const char* ctype = "application/json") {
  std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\n";
  h += std::string("Content-Type: ") + ctype + "\r\n";
  h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  h += "Access-Control-Allow-Origin: *\r\n";
  h += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  h += "\r\n";
  return send_str(fd, h) && send_str(fd, body);
}

static std::string error_json(const std::string& msg, const char* type = "invalid_request_error") {
  Value e = Value::object(), inner = Value::object();
  inner.set("message", Value(msg));
  inner.set("type", Value(type));
  e.set("error", std::move(inner));
  return e.dump();
}

// SSE over chunked transfer encoding
struct SseWriter {
  int fd;
  bool ok = true;
  bool begin(bool keep_alive) {
    std::string h =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\nTransfer-Encoding: chunked\r\n";
    h += keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n";
    return ok = send_str(fd, h);
  }
  bool chunk(const std::string& data) {
    if (!ok) return false;
    char len[32];
    snprintf(len, sizeof len, "%zx\r\n", data.size());
    ok = send_str(fd, std::string(len) + data + "\r\n");
    return ok;
  }
  bool event(const std::string& json) { return chunk("data: " + json + "\n\n"); }
  bool end() {
    if (!ok) return false;
    chunk("data: [DONE]\n\n");
    ok = ok && send_str(fd, "0\r\n\r\n");
    return ok;
  }
};

// ---------------------------------------------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------------------------------------------

Server::Server(const tok::Tokenizer& t, Backend& b, ServerConfig cfg) : tok_(t), be_(b), cfg_(std::move(cfg)) {
  if (cfg_.model_name.empty()) cfg_.model_name = be_.model_id();
}
Server::~Server() { stop(); }

bool Server::start(std::string* err) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) { if (err) *err = "socket() failed"; return false; }
  int one = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)cfg_.port);
  if (inet_pton(AF_INET, cfg_.host.c_str(), &addr.sin_addr) != 1) addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (::bind(listen_fd_, (sockaddr*)&addr, sizeof addr) != 0) {
    if (err) *err = "bind() failed on " + cfg_.host + ":" + std::to_string(cfg_.port) + ": " + strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::listen(listen_fd_, 64) != 0) {
    if (err) *err = "listen() failed";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  socklen_t len = sizeof addr;
  getsockname(listen_fd_, (sockaddr*)&addr, &len);
  port_ = ntohs(addr.sin_port);
  stopping_ = false;
  accept_thr_ = std::thread([this] { accept_loop(); });
  return true;
}

void Server::stop() {
  std::lock_guard<std::mutex> lk(stop_mu_);
  if (listen_fd_ < 0 && !accept_thr_.joinable()) return;
  stopping_ = true;
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (accept_thr_.joinable()) accept_thr_.join();
  for (int i = 0; i < 500 && active_.load() > 0; i++) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void Server::wait() {
  while (!stopping_) std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void Server::accept_loop() {
  while (!stopping_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (stopping_) break;
      if (errno == EINTR || errno == ECONNABORTED) continue;
      break;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    active_++;
    std::thread([this, fd] {
      serve_conn(fd);
      ::close(fd);
      active_--;
    }).detach();
  }
}

void Server::serve_conn(int fd) {
  std::string buf;
  while (!stopping_) {
    HttpRequest rq;
    std::string err;
    if (!read_request(fd, buf, rq, cfg_.max_body_bytes, &err)) {
      if (!err.empty()) send_response(fd, err == "body too large" ? 413 : 400, error_json(err), false);
      return;
    }
    std::string conn = lower(rq.header("connection"));
    bool keep_alive = rq.version == "HTTP/1.1" ? conn != "close" : conn == "keep-alive";
    if (!handle(fd, rq, keep_alive) || !keep_alive) return;
  }
}

bool Server::handle(int fd, const HttpRequest& rq, bool keep_alive) {
  if (rq.method == "OPTIONS") {
    std::string h =
        "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, "
        "OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nContent-Length: 0\r\n\r\n";
    return send_str(fd, h);
  }
  if (rq.path == "/health" || rq.path == "/v1/health") return send_response(fd, 200, "{\"status\": \"ok\"}", keep_alive);
  if (rq.method == "GET" && (rq.path == "/" || rq.path == "/index.html")) {
    std::string html;
    if (!cfg_.ui_path.empty()) {
      FILE* f = fopen(cfg_.ui_path.c_str(), "rb");
      if (f) { char b[65536]; size_t k; while ((k = fread(b, 1, sizeof b, f)) > 0) html.append(b, k); fclose(f); }
    }
    if (html.empty()) return send_response(fd, 404, error_json("no UI file (serve --ui PATH)"), keep_alive);
    std::string h = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: " +
                    std::to_string(html.size()) + (keep_alive ? "\r\nConnection: keep-alive" : "\r\nConnection: close") + "\r\n\r\n";
    return send_str(fd, h) && send_str(fd, html);
  }
  if (rq.path == "/v1/stats" || rq.path == "/stats") return send_response(fd, 200, be_.status_json(), keep_alive);
  if (rq.path == "/props" || rq.path == "/v1/props") {
    Value r = Value::object();
    r.set("model", Value(cfg_.model_name));
    r.set("n_ctx", Value((long long)be_.max_context()));
    r.set("default_max_tokens", Value((long long)cfg_.default_max_tokens));
    Value sp = Value::array();
    for (const char* k : {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
                          "repeat_last_n", "seed", "max_tokens", "max_completion_tokens", "stop", "logit_bias", "ignore_eos",
                          "stream", "stream_options.include_usage", "reasoning_effort", "thinking", "chat_template_kwargs",
                          "tools", "tool_choice", "response_format"})
      sp.push_back(Value(k));
    r.set("supported_params", std::move(sp));
    return send_response(fd, 200, r.dump(), keep_alive);
  }
  if (rq.path == "/v1/models" || rq.path == "/models") {
    if (rq.method != "GET") return send_response(fd, 405, error_json("use GET"), keep_alive);
    Value m = Value::object();
    m.set("id", Value(cfg_.model_name));
    m.set("object", Value("model"));
    m.set("created", Value((long long)time(nullptr)));
    m.set("owned_by", Value("macseek"));
    Value data = Value::array();
    data.push_back(std::move(m));
    Value r = Value::object();
    r.set("object", Value("list"));
    r.set("data", std::move(data));
    return send_response(fd, 200, r.dump(), keep_alive);
  }
  if (rq.path == "/v1/chat/completions" || rq.path == "/chat/completions") {
    if (rq.method != "POST") return send_response(fd, 405, error_json("use POST"), keep_alive);
    return handle_chat(fd, rq, keep_alive);
  }
  if (rq.path == "/v1/completions" || rq.path == "/completions") {
    if (rq.method != "POST") return send_response(fd, 405, error_json("use POST"), keep_alive);
    return handle_completion(fd, rq, keep_alive);
  }
  return send_response(fd, 404, error_json("not found: " + rq.path), keep_alive);
}

namespace {

GenParams parse_params(const Value& body, int default_max) {
  GenParams p;
  p.max_tokens = default_max;
  const Value* mt = body.find("max_completion_tokens");
  if (!mt || mt->is_null()) mt = body.find("max_tokens");
  if (mt && mt->is_number()) p.max_tokens = (int)mt->as_int();
  if (body["temperature"].is_number()) p.temperature = body["temperature"].as_double();
  if (body["top_p"].is_number()) p.top_p = body["top_p"].as_double();
  if (body["top_k"].is_number()) p.top_k = (int)body["top_k"].as_int();
  if (body["min_p"].is_number()) p.min_p = body["min_p"].as_double();
  if (body["presence_penalty"].is_number()) p.presence_penalty = body["presence_penalty"].as_double();
  if (body["frequency_penalty"].is_number()) p.frequency_penalty = body["frequency_penalty"].as_double();
  const Value* rp = body.find("repetition_penalty");
  if (!rp || !rp->is_number()) rp = body.find("repeat_penalty");
  if (rp && rp->is_number()) p.repetition_penalty = rp->as_double();
  if (body["repeat_last_n"].is_number()) p.repeat_last_n = (int)body["repeat_last_n"].as_int();
  if (body["seed"].is_number()) { p.seed = (uint64_t)body["seed"].as_int(); p.has_seed = true; }
  p.ignore_eos = body["ignore_eos"].as_bool(false);
  const Value& lb = body["logit_bias"];
  if (lb.is_object())
    for (const auto& kv : lb.obj())
      if (kv.second.is_number()) p.logit_bias.push_back({(int32_t)atoi(kv.first.c_str()), (float)kv.second.as_double()});
  return p;
}

std::vector<std::string> parse_stop(const Value& body) {
  std::vector<std::string> stop;
  const Value& s = body["stop"];
  if (s.is_string() && !s.str().empty()) stop.push_back(s.str());
  if (s.is_array())
    for (const Value& v : s.arr())
      if (v.is_string() && !v.str().empty()) stop.push_back(v.str());
  return stop;
}

Value timings_json(const GenStats& g) {
  Value t = Value::object();
  t.set("prompt_n", Value((long long)g.prompt_n));
  t.set("cache_n", Value((long long)g.cached_n));
  t.set("prompt_ms", Value(g.prompt_ms));
  const int pn = g.prompt_n - g.cached_n;
  t.set("prompt_per_second", Value(g.prompt_ms > 0 ? pn / (g.prompt_ms / 1e3) : 0.0));
  t.set("predicted_n", Value((long long)g.predicted_n));
  t.set("predicted_ms", Value(g.predicted_ms));
  t.set("predicted_per_second", Value(g.predicted_ms > 0 ? g.predicted_n / (g.predicted_ms / 1e3) : 0.0));
  return t;
}

const char* finish_str(Finish f, bool tool_calls) {
  if (tool_calls) return "tool_calls";
  switch (f) {
    case Finish::Length: return "length";
    case Finish::Error: return "error";
    default: return "stop";
  }
}

// Incremental text assembly with stop strings and a hold-back window so streamed text never contains (a prefix
// of) a stop string. `markers` are additional strings that end plain streaming (e.g. tool-call start).
struct TextStream {
  const tok::Tokenizer& t;
  tok::StreamDecoder dec;
  std::vector<std::string> stop;
  std::string text;     // full decoded text so far (truncated at a stop string)
  size_t emitted = 0;   // bytes of `text` already released
  bool stopped = false; // a stop string matched
  size_t hold = 0;
  TextStream(const tok::Tokenizer& tk, std::vector<std::string> st) : t(tk), dec(tk, true), stop(std::move(st)) {
    for (auto& s : stop) hold = std::max(hold, s.size());
  }
  // returns false when a stop string matched
  bool push(int32_t id) {
    text += dec.push(id);
    return check();
  }
  bool check() {
    size_t from = emitted > hold ? emitted - hold : 0;
    size_t best = std::string::npos;
    for (auto& s : stop) {
      size_t p = text.find(s, from);
      if (p < best) best = p;
    }
    if (best != std::string::npos) {
      text.resize(best);
      stopped = true;
      return false;
    }
    return true;
  }
  void finish() {
    if (!stopped) {
      text += dec.flush();
      check();
    }
  }
  // releasable prefix length: everything except a tail that could still become a stop string
  size_t safe_end(bool final) const {
    if (final || stopped) return text.size();
    size_t h = 0;
    for (auto& s : stop) h = std::max(h, partial_suffix(text, emitted, s));
    return utf8_floor(text, text.size() - h, emitted);
  }
  // longest k < pat.size() such that text[from..] ends with pat[0..k)
  static size_t partial_suffix(const std::string& text, size_t from, const std::string& pat) {
    size_t avail = text.size() - std::min(from, text.size());
    for (size_t k = std::min(pat.size() - 1, avail); k > 0; k--)
      if (text.compare(text.size() - k, k, pat, 0, k) == 0) return k;
    return 0;
  }
  static size_t utf8_floor(const std::string& text, size_t e, size_t lo) {
    while (e > lo && e < text.size() && ((unsigned char)text[e] & 0xC0) == 0x80) e--;
    return std::max(e, lo);
  }
};

}  // namespace

bool Server::handle_chat(int fd, const HttpRequest& rq, bool keep_alive) {
  Value body;
  std::string err;
  if (!json::parse(rq.body, body, &err) || !body.is_object())
    return send_response(fd, 400, error_json("invalid JSON body: " + err), keep_alive);
  chat::Options opt;
  std::string prompt;
  if (!chat::format_request(body, prompt, &err, &opt)) return send_response(fd, 400, error_json(err), keep_alive);
  const bool thinking = opt.thinking_mode == "thinking";
  std::vector<int32_t> ids = tok_.encode(prompt, true, false);
  GenParams gp = parse_params(body, cfg_.default_max_tokens);
  const bool stream = body["stream"].as_bool(false);
  const bool include_usage = body["stream_options"]["include_usage"].as_bool(false);

  const std::string id = "chatcmpl-" + std::to_string(++req_counter_);
  const long long created = (long long)time(nullptr);
  const int32_t think_end_id = tok_.find_token(chat::kThinkEnd);
  const std::string tc_marker = std::string("\n\n<") + chat::kDsml + " calls>";

  std::vector<std::string> stops = parse_stop(body);
  TextStream ts(tok_, stops);
  // Streaming state: phase 0 = reasoning (thinking mode, until </think>), 1 = content, 2 = tool-call block.
  int phase = thinking ? 0 : 1;
  size_t content_start = 0;  // byte offset in ts.text where content begins
  int n_gen = 0;
  SseWriter sse{fd};
  auto chunk_json = [&](const Value& delta, const Value& finish) {
    Value ch = Value::object();
    ch.set("id", Value(id));
    ch.set("object", Value("chat.completion.chunk"));
    ch.set("created", Value(created));
    ch.set("model", Value(cfg_.model_name));
    Value choice = Value::object();
    choice.set("index", Value(0));
    choice.set("delta", delta);
    choice.set("finish_reason", finish);
    Value choices = Value::array();
    choices.push_back(std::move(choice));
    ch.set("choices", std::move(choices));
    return ch.dump();
  };
  auto send_delta = [&](const char* key, const std::string& s) {
    if (s.empty()) return true;
    Value d = Value::object();
    d.set(key, Value(s));
    return sse.event(chunk_json(d, Value()));
  };
  // release newly safe text as reasoning/content deltas
  auto pump = [&](bool final) {
    size_t end = ts.safe_end(final);
    if (phase == 0) {
      if (end > ts.emitted) { send_delta("reasoning_content", ts.text.substr(ts.emitted, end - ts.emitted)); ts.emitted = end; }
      return;
    }
    if (phase == 1) {
      // hold back anything that could start the tool-call marker
      size_t m = ts.text.find(tc_marker, content_start);
      if (m != std::string::npos) {
        if (m > ts.emitted) send_delta("content", ts.text.substr(ts.emitted, m - ts.emitted));
        ts.emitted = m;
        phase = 2;
        return;
      }
      if (!final) {  // hold back a tail that could still become the tool-call marker
        size_t k = TextStream::partial_suffix(ts.text, ts.emitted, tc_marker);
        end = std::min(end, TextStream::utf8_floor(ts.text, ts.text.size() - k, ts.emitted));
      }
      if (end > ts.emitted) { send_delta("content", ts.text.substr(ts.emitted, end - ts.emitted)); ts.emitted = end; }
    }
  };

  if (stream) {
    if (!sse.begin(keep_alive)) return false;
    Value d = Value::object();
    d.set("role", Value("assistant"));
    d.set("content", Value(""));
    sse.event(chunk_json(d, Value()));
  }

  Finish fin;
  {
    std::lock_guard<std::mutex> lk(gen_mu_);
    fin = be_.generate(ids, gp, [&](int32_t tok) {
      if (tok_.is_eog(tok)) return true;  // do not render EOS; backend returns Stop
      n_gen++;
      if (thinking && phase == 0 && tok == think_end_id) {
        // close the reasoning part: everything before </think> is reasoning
        ts.text += ts.dec.flush();
        if (stream) pump(true);      // release the rest of the reasoning
        ts.text += chat::kThinkEnd;  // keep the full text parseable by chat::parse_reply
        ts.emitted = content_start = ts.text.size();
        phase = 1;
        return !stream || sse.ok;
      }
      bool go = ts.push(tok);
      if (stream) pump(false);
      if (stream && !sse.ok) return false;
      return go;
    });
  }
  ts.finish();
  if (ts.stopped && fin == Finish::Cancelled) fin = Finish::Stop;

  // final parse of the whole reply (reasoning / content / tool calls)
  chat::Reply rep = chat::parse_reply(ts.text, thinking);
  const bool has_tc = rep.tool_calls.size() > 0;

  if (stream) {
    pump(true);
    if (phase == 2) {
      if (has_tc) {
        Value tcs = Value::array();
        int idx = 0;
        for (const Value& tc : rep.tool_calls.arr()) {
          Value t = tc;
          t.set("index", Value(idx));
          t.set("id", Value("call_" + std::to_string(req_counter_.load()) + "_" + std::to_string(idx)));
          idx++;
          tcs.push_back(std::move(t));
        }
        Value d = Value::object();
        d.set("tool_calls", std::move(tcs));
        sse.event(chunk_json(d, Value()));
      } else {
        send_delta("content", ts.text.substr(ts.emitted));  // malformed DSML: plain content
      }
    }
    {
      Value ch = Value::object();
      ch.set("id", Value(id));
      ch.set("object", Value("chat.completion.chunk"));
      ch.set("created", Value(created));
      ch.set("model", Value(cfg_.model_name));
      Value choice = Value::object();
      choice.set("index", Value(0));
      choice.set("delta", Value::object());
      choice.set("finish_reason", Value(finish_str(fin, has_tc)));
      Value choices = Value::array();
      choices.push_back(std::move(choice));
      ch.set("choices", std::move(choices));
      ch.set("timings", timings_json(be_.last_stats()));
      sse.event(ch.dump());
    }
    if (include_usage) {
      Value u = Value::object();
      u.set("prompt_tokens", Value((long long)ids.size()));
      u.set("completion_tokens", Value(n_gen));
      u.set("total_tokens", Value((long long)ids.size() + n_gen));
      Value ch = Value::object();
      ch.set("id", Value(id));
      ch.set("object", Value("chat.completion.chunk"));
      ch.set("created", Value(created));
      ch.set("model", Value(cfg_.model_name));
      ch.set("choices", Value::array());
      ch.set("usage", std::move(u));
      sse.event(ch.dump());
    }
    return sse.end();
  }

  Value msg = Value::object();
  msg.set("role", Value("assistant"));
  msg.set("content", Value(rep.content));
  if (thinking) msg.set("reasoning_content", Value(rep.reasoning));
  if (has_tc) {
    Value tcs = Value::array();
    int idx = 0;
    for (const Value& tc : rep.tool_calls.arr()) {
      Value t = Value::object();
      t.set("id", Value("call_" + std::to_string(req_counter_.load()) + "_" + std::to_string(idx++)));
      t.set("type", Value("function"));
      t.set("function", tc["function"]);
      tcs.push_back(std::move(t));
    }
    msg.set("tool_calls", std::move(tcs));
  }
  Value choice = Value::object();
  choice.set("index", Value(0));
  choice.set("message", std::move(msg));
  choice.set("finish_reason", Value(finish_str(fin, has_tc)));
  Value choices = Value::array();
  choices.push_back(std::move(choice));
  Value u = Value::object();
  u.set("prompt_tokens", Value((long long)ids.size()));
  u.set("completion_tokens", Value(n_gen));
  u.set("total_tokens", Value((long long)ids.size() + n_gen));
  Value r = Value::object();
  r.set("id", Value(id));
  r.set("object", Value("chat.completion"));
  r.set("created", Value(created));
  r.set("model", Value(cfg_.model_name));
  r.set("choices", std::move(choices));
  r.set("usage", std::move(u));
  r.set("timings", timings_json(be_.last_stats()));
  return send_response(fd, 200, r.dump(), keep_alive);
}

bool Server::handle_completion(int fd, const HttpRequest& rq, bool keep_alive) {
  Value body;
  std::string err;
  if (!json::parse(rq.body, body, &err) || !body.is_object())
    return send_response(fd, 400, error_json("invalid JSON body: " + err), keep_alive);
  std::vector<int32_t> ids;
  const Value& pr = body["prompt"];
  if (pr.is_string()) {
    ids = tok_.encode(pr.str(), true, false);
  } else if (pr.is_array() && pr.size() > 0 && pr[0].is_number()) {
    for (const Value& v : pr.arr()) ids.push_back((int32_t)v.as_int());
  } else if (pr.is_array() && pr.size() == 1 && pr[0].is_string()) {
    ids = tok_.encode(pr[0].str(), true, false);
  } else {
    return send_response(fd, 400, error_json("prompt must be a string or an array of token ids"), keep_alive);
  }
  GenParams gp = parse_params(body, cfg_.default_max_tokens);
  const bool stream = body["stream"].as_bool(false);
  const std::string id = "cmpl-" + std::to_string(++req_counter_);
  const long long created = (long long)time(nullptr);
  TextStream ts(tok_, parse_stop(body));
  int n_gen = 0;
  SseWriter sse{fd};
  auto chunk_json = [&](const std::string& text, const Value& finish) {
    Value c = Value::object();
    c.set("index", Value(0));
    c.set("text", Value(text));
    c.set("logprobs", Value());
    c.set("finish_reason", finish);
    Value choices = Value::array();
    choices.push_back(std::move(c));
    Value r = Value::object();
    r.set("id", Value(id));
    r.set("object", Value("text_completion"));
    r.set("created", Value(created));
    r.set("model", Value(cfg_.model_name));
    r.set("choices", std::move(choices));
    return r;
  };
  auto pump = [&](bool final) {
    size_t end = ts.safe_end(final);
    if (end > ts.emitted) {
      sse.event(chunk_json(ts.text.substr(ts.emitted, end - ts.emitted), Value()).dump());
      ts.emitted = end;
    }
  };
  if (stream && !sse.begin(keep_alive)) return false;
  Finish fin;
  {
    std::lock_guard<std::mutex> lk(gen_mu_);
    fin = be_.generate(ids, gp, [&](int32_t tok) {
      if (tok_.is_eog(tok)) return true;
      n_gen++;
      bool go = ts.push(tok);
      if (stream) {
        pump(false);
        if (!sse.ok) return false;
      }
      return go;
    });
  }
  ts.finish();
  if (ts.stopped && fin == Finish::Cancelled) fin = Finish::Stop;
  if (stream) {
    pump(true);
    sse.event(chunk_json("", Value(finish_str(fin, false))).dump());
    return sse.end();
  }
  Value r = chunk_json(ts.text, Value(finish_str(fin, false)));
  Value u = Value::object();
  u.set("prompt_tokens", Value((long long)ids.size()));
  u.set("completion_tokens", Value(n_gen));
  u.set("total_tokens", Value((long long)ids.size() + n_gen));
  r.set("usage", std::move(u));
  r.set("timings", timings_json(be_.last_stats()));
  return send_response(fd, 200, r.dump(), keep_alive);
}

}  // namespace ms::server
