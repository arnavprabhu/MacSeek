// macseek CLI.
//
// Implemented here (module tok):
//   macseek tokenize    [-m GGUF] (-p TEXT | -f FILE | --stdin) [--no-parse-special] [--bos] [--pieces] [--count]
//                       [--bench N]
//   macseek detok       [-m GGUF] (IDS... | -f FILE | --stdin) [--no-special] [--stream]
//   macseek chat-format [-m GGUF] (-f REQUEST.json | --stdin) [--thinking-mode thinking|chat] [--effort E]
//                       [--no-drop-thinking] [--no-gen-prompt] [--ids] [--batch]
//   macseek serve       [-m GGUF] [--host H] [--port P] --echo        (echo backend; real backend: INTEGRATION HOOK)
// Integration hooks (filled by the integration step, see "INTEGRATION HOOK" markers):
//   macseek gen | bench | logits | snapshot | serve (with the model backend)
//
// -m defaults to $MACSEEK_GGUF or the UD-IQ1_S split 00001 (it holds the metadata + tokenizer, 5.8 MB).
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/timer.hpp"
#include "server/http.hpp"
#include "tok/chat.hpp"
#include "tok/json.hpp"
#include "tok/tokenizer.hpp"

using namespace ms;

namespace {

struct Args {
  std::vector<std::string> pos;
  std::vector<std::pair<std::string, std::string>> kv;
  std::vector<std::string> flags;
  bool flag(const char* f) const {
    for (auto& x : flags)
      if (x == f) return true;
    return false;
  }
  std::string get(const char* k, const std::string& d = "") const {
    for (auto& [a, b] : kv)
      if (a == k) return b;
    return d;
  }
  bool has(const char* k) const {
    for (auto& [a, b] : kv)
      if (a == k) return true;
    return false;
  }
};

// options that take a value
bool takes_value(const std::string& a) {
  static const char* kv[] = {"-m", "--model", "-p", "--prompt", "-f", "--file", "--bench", "--thinking-mode",
                             "--effort", "--host", "--port", "--max-tokens"};
  for (auto* k : kv)
    if (a == k) return true;
  return false;
}

Args parse_args(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i < argc; i++) {
    std::string s = argv[i];
    if (takes_value(s) && i + 1 < argc) {
      std::string k = s == "--model" ? "-m" : s == "--prompt" ? "-p" : s == "--file" ? "-f" : s;
      a.kv.emplace_back(k, argv[++i]);
    } else if (s.size() > 1 && s[0] == '-' && !(s[1] >= '0' && s[1] <= '9')) {
      a.flags.push_back(s);
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

bool read_file(const std::string& path, std::string& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::stringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

std::string read_stdin() {
  std::stringstream ss;
  ss << std::cin.rdbuf();
  return ss.str();
}

std::unique_ptr<tok::Tokenizer> load_tok(const Args& a) {
  std::string path = a.get("-m", tok::default_gguf_path());
  std::string err;
  Timer t;
  auto tk = tok::Tokenizer::load_from_gguf(path, &err);
  if (!tk) {
    fprintf(stderr, "error: %s\n", err.c_str());
    exit(1);
  }
  if (getenv("MACSEEK_VERBOSE")) fprintf(stderr, "tokenizer loaded in %.1f ms (%d tokens)\n", t.ms(), tk->n_vocab());
  return tk;
}

std::string input_text(const Args& a, bool* ok) {
  *ok = true;
  if (a.flag("--stdin")) return read_stdin();
  if (a.has("-f")) {
    std::string s;
    if (!read_file(a.get("-f"), s)) {
      fprintf(stderr, "error: cannot read %s\n", a.get("-f").c_str());
      *ok = false;
    }
    return s;
  }
  if (a.has("-p")) return a.get("-p");
  *ok = false;
  return "";
}

void print_ids(const std::vector<int32_t>& ids) {
  std::string s = "[";
  for (size_t i = 0; i < ids.size(); i++) {
    if (i) s += ", ";
    s += std::to_string(ids[i]);
  }
  s += "]\n";
  fwrite(s.data(), 1, s.size(), stdout);
}

int cmd_tokenize(const Args& a) {
  bool ok;
  std::string text = input_text(a, &ok);
  if (!ok) {
    fprintf(stderr, "usage: macseek tokenize [-m GGUF] (-p TEXT | -f FILE | --stdin) [--no-parse-special] "
                    "[--bos] [--pieces] [--count] [--bench N]\n");
    return 2;
  }
  auto tk = load_tok(a);
  const bool special = !a.flag("--no-parse-special");
  if (a.has("--bench")) {
    int n = atoi(a.get("--bench").c_str());
    double best = 1e30;
    size_t nt = 0;
    for (int i = 0; i < std::max(1, n); i++) {
      Timer t;
      auto ids = tk->encode(text, special, a.flag("--bos"));
      best = std::min(best, t.ms());
      nt = ids.size();
    }
    printf("bytes=%zu tokens=%zu best=%.2f ms  %.1f MB/s  %.0f tok/ms\n", text.size(), nt, best,
           text.size() / best / 1e3, nt / best);
    return 0;
  }
  auto ids = tk->encode(text, special, a.flag("--bos"));
  if (a.flag("--pieces")) {
    for (int32_t id : ids) {
      std::string p(tk->piece(id));
      printf("%6d -> '%s'\n", id, p.c_str());
    }
  } else if (!a.flag("--count")) {
    print_ids(ids);
  }
  if (a.flag("--count")) printf("Total number of tokens: %zu\n", ids.size());
  return 0;
}

int cmd_detok(const Args& a) {
  std::string src;
  if (a.flag("--stdin")) src = read_stdin();
  else if (a.has("-f")) {
    if (!read_file(a.get("-f"), src)) { fprintf(stderr, "error: cannot read %s\n", a.get("-f").c_str()); return 1; }
  } else {
    for (auto& p : a.pos) src += p + " ";
  }
  std::vector<int32_t> ids;
  const char* p = src.c_str();
  while (*p) {
    if ((*p >= '0' && *p <= '9') || *p == '-') {
      char* e;
      long v = strtol(p, &e, 10);
      ids.push_back((int32_t)v);
      p = e;
    } else {
      p++;
    }
  }
  auto tk = load_tok(a);
  const bool special = !a.flag("--no-special");
  if (a.flag("--stream")) {
    tok::StreamDecoder dec(*tk, special);
    for (int32_t id : ids) {
      std::string s = dec.push(id);
      printf("%6d -> '%s'%s\n", id, s.c_str(), dec.pending() ? " (pending)" : "");
    }
    std::string s = dec.flush();
    if (!s.empty()) printf("flush -> '%s'\n", s.c_str());
    return 0;
  }
  std::string out = tk->decode(ids, special);
  fwrite(out.data(), 1, out.size(), stdout);
  return 0;
}

int cmd_chat_format(const Args& a) {
  std::string src;
  if (a.flag("--stdin")) src = read_stdin();
  else if (!a.has("-f") || !read_file(a.get("-f"), src)) {
    fprintf(stderr, "usage: macseek chat-format (-f REQUEST.json | --stdin) [--thinking-mode thinking|chat] "
                    "[--effort E] [--no-drop-thinking] [--no-gen-prompt] [--ids]\n"
                    "  REQUEST: OpenAI chat request {messages, tools, ...}, an encoding.py test case, or a bare "
                    "message array\n");
    return 2;
  }
  json::Value body;
  std::string err;
  if (!json::parse(src, body, &err)) { fprintf(stderr, "error: bad JSON: %s\n", err.c_str()); return 1; }
  if (a.flag("--batch")) {  // array of request bodies -> JSON array of rendered strings / {"error": ...}
    if (!body.is_array()) { fprintf(stderr, "error: --batch expects a JSON array of requests\n"); return 1; }
    json::Value outs = json::Value::array();
    for (const json::Value& rq : body.arr()) {
      std::string out;
      if (chat::format_request(rq, out, &err)) outs.push_back(json::Value(out));
      else {
        json::Value e = json::Value::object();
        e.set("error", json::Value(err));
        outs.push_back(std::move(e));
      }
    }
    std::string s = outs.dump();
    fwrite(s.data(), 1, s.size(), stdout);
    return 0;
  }
  if (body.is_array()) {  // bare message list
    json::Value b = json::Value::object();
    b.set("messages", body);
    body = std::move(b);
  }
  // encoding.py test-case defaults: thinking_mode given at top level; CLI overrides
  if (a.has("--thinking-mode")) body.set("thinking_mode", json::Value(a.get("--thinking-mode")));
  if (a.has("--effort")) {
    std::string e = a.get("--effort");
    bool num = !e.empty() && e.find_first_not_of("0123456789") == std::string::npos;
    body.set("reasoning_effort", num ? json::Value((long long)atoll(e.c_str())) : json::Value(e));
  }
  if (a.flag("--no-drop-thinking")) body.set("drop_thinking", json::Value(false));
  if (a.flag("--no-gen-prompt")) body.set("add_generation_prompt", json::Value(false));
  std::string out;
  if (!chat::format_request(body, out, &err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
  if (a.flag("--ids")) {
    auto tk = load_tok(a);
    print_ids(tk->encode(out, true, false));
  } else {
    fwrite(out.data(), 1, out.size(), stdout);
  }
  return 0;
}

server::Server* g_server = nullptr;
void on_signal(int) {
  if (g_server) g_server->stop();
}

int cmd_serve(const Args& a) {
  auto tk = load_tok(a);
  std::unique_ptr<server::Backend> backend;
  if (a.flag("--echo")) {
    backend = std::make_unique<server::EchoBackend>(*tk);
  } else {
    // ===== INTEGRATION HOOK (serve): construct the model-backed ms::server::Backend here, e.g.
    //   backend = std::make_unique<ModelBackend>(model, sampler);
    fprintf(stderr, "serve: model backend not integrated yet; use --echo for the dummy echo backend\n");
    return 3;
  }
  server::ServerConfig cfg;
  cfg.host = a.get("--host", "127.0.0.1");
  cfg.port = atoi(a.get("--port", "8080").c_str());
  if (a.has("--max-tokens")) cfg.default_max_tokens = atoi(a.get("--max-tokens").c_str());
  server::Server srv(*tk, *backend, cfg);
  std::string err;
  if (!srv.start(&err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
  fprintf(stderr, "macseek: serving %s on http://%s:%d/v1\n", backend->model_id().c_str(), cfg.host.c_str(),
          srv.port());
  g_server = &srv;
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  srv.wait();
  g_server = nullptr;
  return 0;
}

int not_integrated(const char* cmd) {
  fprintf(stderr, "macseek %s: not implemented yet (integration hook)\n", cmd);
  return 3;
}

void usage() {
  fprintf(stderr,
          "usage: macseek <command> [options]\n"
          "  tokenize     text -> token ids (llama-tokenize compatible output)\n"
          "  detok        token ids -> text\n"
          "  chat-format  render a chat request with the DeepSeek V4.1 template\n"
          "  serve        OpenAI-compatible HTTP server (--echo: dummy backend)\n"
          "  gen | bench | logits | snapshot   (integration hooks)\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 2; }
  std::string cmd = argv[1];
  Args a = parse_args(argc, argv, 2);
  if (cmd == "tokenize") return cmd_tokenize(a);
  if (cmd == "detok") return cmd_detok(a);
  if (cmd == "chat-format") return cmd_chat_format(a);
  if (cmd == "serve") return cmd_serve(a);
  // ===== INTEGRATION HOOK (gen): load model, format/tokenize prompt (chat::format_request + Tokenizer::encode),
  //       prefill + decode, stream with tok::StreamDecoder.
  if (cmd == "gen") return not_integrated("gen");
  // ===== INTEGRATION HOOK (bench): prefill/decode speed loop (SPEC §7).
  if (cmd == "bench") return not_integrated("bench");
  // ===== INTEGRATION HOOK (logits): dump full logits per position for tools/quality.py (SPEC §6 gate C).
  if (cmd == "logits") return not_integrated("logits");
  // ===== INTEGRATION HOOK (snapshot): save/load model state snapshots (SPEC §4.3).
  if (cmd == "snapshot") return not_integrated("snapshot");
  if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
  fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
  usage();
  return 2;
}
