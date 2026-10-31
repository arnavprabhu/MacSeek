// tests for module tok: tokenizer (vs llama-tokenize oracle ids), detokenizer + streaming UTF-8, JSON,
// chat formatting (DeepSeek encoding.py test vectors), reply parsing, HTTP server with the echo backend.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/timer.hpp"
#include "server/http.hpp"
#include "tok/chat.hpp"
#include "tok/json.hpp"
#include "tok/tokenizer.hpp"

using namespace ms;

static int g_fail = 0, g_pass = 0;
#define CHECK(c)                                                          \
  do {                                                                    \
    if (c) g_pass++;                                                      \
    else { g_fail++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } \
  } while (0)
#define CHECK_EQ(a, b)                                                                              \
  do {                                                                                              \
    auto _a = (a); auto _b = (b);                                                                   \
    if (_a == _b) g_pass++;                                                                         \
    else { g_fail++; fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); }      \
  } while (0)

static std::string data_dir() {
  std::string f = __FILE__;
  return f.substr(0, f.rfind('/')) + "/data";
}
static bool slurp(const std::string& p, std::string& out) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  std::stringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}
static bool valid_utf8(const std::string& s) { return tok::utf8_sanitize(s) == s; }

// ---------------------------------------------------------------------------------------------------------------
static void test_json() {
  CHECK_EQ(json::py_float_repr(1.0), std::string("1.0"));
  CHECK_EQ(json::py_float_repr(0.1), std::string("0.1"));
  CHECK_EQ(json::py_float_repr(1e-05), std::string("1e-05"));
  CHECK_EQ(json::py_float_repr(0.0001), std::string("0.0001"));
  CHECK_EQ(json::py_float_repr(1e16), std::string("1e+16"));
  CHECK_EQ(json::py_float_repr(1e15), std::string("1000000000000000.0"));
  CHECK_EQ(json::py_float_repr(123456.789), std::string("123456.789"));
  CHECK_EQ(json::py_float_repr(-0.0), std::string("-0.0"));
  CHECK_EQ(json::py_float_repr(1.5e-07), std::string("1.5e-07"));
  CHECK_EQ(json::py_float_repr(1e22), std::string("1e+22"));
  json::Value v;
  std::string err;
  CHECK(json::parse(R"({"b": 1, "a": [true, null, 2.50, "x\u00e9\ud83d\ude00\n\u0001"], "c": {"z": -0, "y": 1E2}})", v, &err));
  CHECK_EQ(v.dump(), std::string("{\"b\": 1, \"a\": [true, null, 2.5, \"xé😀\\n\\u0001\"], \"c\": {\"z\": 0, \"y\": 100.0}}"));
  CHECK(!json::parse("{\"a\": }", v, &err));
  CHECK(!json::parse("[1, 2", v, &err));
}

// ---------------------------------------------------------------------------------------------------------------
static void test_tokenizer_oracle(const tok::Tokenizer& tk) {
  // tests/data/tok/<name> + <name>.ids were produced by Homebrew llama-tokenize (tools/tok_oracle_check.py)
  std::string dir = data_dir() + "/tok";
  DIR* d = opendir(dir.c_str());
  CHECK(d != nullptr);
  if (!d) return;
  int files = 0;
  while (dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() < 5 || n.substr(n.size() - 4) != ".ids") continue;
    std::string text, ids_s;
    std::string base = n.substr(0, n.size() - 4);
    CHECK(slurp(dir + "/" + base, text) && slurp(dir + "/" + n, ids_s));
    std::vector<int32_t> want;
    std::istringstream is(ids_s);
    for (int32_t x; is >> x;) want.push_back(x);
    std::vector<int32_t> got = tk.encode(text);
    bool ok = got == want;
    if (!ok) {
      size_t i = 0;
      while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
      fprintf(stderr, "  %s: mismatch at token %zu (got %zu want %zu tokens)\n", base.c_str(), i, got.size(),
              want.size());
    }
    CHECK(ok);
    // round trip (exact for valid UTF-8 input)
    if (valid_utf8(text)) CHECK(tk.decode(got) == text);
    files++;
  }
  closedir(d);
  CHECK(files >= 10);
  fprintf(stderr, "  oracle files checked: %d\n", files);
}

static void test_tokenizer_basics(const tok::Tokenizer& tk) {
  CHECK_EQ(tk.n_vocab(), 129280);
  CHECK_EQ(tk.bos(), 0);
  CHECK_EQ(tk.eos(), 1);
  CHECK_EQ(tk.find_token("<｜User｜>"), 128803);
  CHECK_EQ(tk.find_token("<think>"), 128821);
  CHECK_EQ(tk.find_token("｜DSML｜"), 128825);
  CHECK(tk.encode("").empty());
  CHECK_EQ(tk.encode("x", true, true).front(), 0);
  // control tokens need parse_special; user-defined (<think>) are always matched (llama.cpp semantics)
  auto a = tk.encode("<｜User｜><think>", false);
  CHECK(a.size() > 2 && a.back() == 128821);
  auto b = tk.encode("<｜User｜><think>", true);
  CHECK(b.size() == 2 && b[0] == 128803 && b[1] == 128821);
  // pre-tokenizer pieces
  auto p = tk.pretokenize("Hello world!! 12345 你好abc\n\n  x");
  std::vector<std::string> want = {"Hello", " world", "!!", " ", "123", "45", " ", "你好", "abc", "\n\n", " ", " x"};
  CHECK(p == want);
  // control tokens render only with special=true
  CHECK(tk.piece(1, true) == "<｜end▁of▁sentence｜>");
  CHECK(tk.piece(1, false).empty());
  CHECK(tk.piece(128821, false) == "<think>");  // user-defined: always rendered
  // invalid UTF-8 -> U+FFFD tokens, never crashes
  auto inv = tk.encode(std::string("a\xff\xc3", 3));
  CHECK(tk.decode(inv) == "a\xEF\xBF\xBD\xEF\xBF\xBD");
}

static void test_stream_decoder(const tok::Tokenizer& tk) {
  std::string text = "naïve 😀👨‍👩‍👧 中文テキスト ≠ €100 \xF0\x9F\x87\xBA\xF0\x9F\x87\xB8 done";
  auto ids = tk.encode(text);
  tok::StreamDecoder dec(tk);
  std::string acc;
  for (int32_t id : ids) {
    std::string s = dec.push(id);
    CHECK(valid_utf8(s));
    acc += s;
  }
  acc += dec.flush();
  CHECK(acc == text);
  // byte tokens of "€" (E2 82 AC) one at a time: nothing is released until the sequence is complete
  uint32_t b2u[256];
  for (int x = 0, n = 0; x < 256; x++) {
    bool direct = (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255);
    b2u[x] = direct ? (uint32_t)x : (uint32_t)(256 + n++);
  }
  std::vector<int32_t> bytes;
  for (unsigned char c : std::string("\xE2\x82\xAC")) {
    uint32_t cp = b2u[c];
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
    bytes.push_back(tk.find_token(s));
  }
  CHECK(bytes[0] >= 0 && bytes[1] >= 0 && bytes[2] >= 0);
  tok::StreamDecoder d2(tk);
  CHECK(d2.push(bytes[0]).empty());
  CHECK(d2.push(bytes[1]).empty());
  CHECK_EQ(d2.pending(), (size_t)2);
  CHECK(d2.push(bytes[2]) == "€");
  // truncated tail flushes as U+FFFD
  d2.push(bytes[0]);
  CHECK(d2.flush() == "\xEF\xBF\xBD");
  // invalid continuation -> replacement immediately, the following ASCII passes through
  CHECK(d2.push_bytes("\xE2" "A") == "\xEF\xBF\xBD" "A");
}

static void test_speed(const tok::Tokenizer& tk) {
  std::string text;
  if (!slurp(data_dir() + "/tok/code_python.txt", text)) return;
  std::string big;
  while (big.size() < (4u << 20)) big += text;
  Timer t;
  auto ids = tk.encode(big);
  double ms = t.ms();
  double mbps = big.size() / ms / 1e3;
  fprintf(stderr, "  speed: %.1f MB in %.1f ms = %.1f MB/s (%zu tokens)\n", big.size() / 1e6, ms, mbps, ids.size());
  CHECK(mbps >= 5.0);
}

// ---------------------------------------------------------------------------------------------------------------
static void test_chat_vectors() {
  // DeepSeek reference vectors (encoding/tests on the HF repo); encode_case(case, "chat") semantics
  for (int n = 1; n <= 5; n++) {
    std::string in, want;
    std::string base = data_dir() + "/chat/test_";
    CHECK(slurp(base + "input_" + std::to_string(n) + ".json", in));
    CHECK(slurp(base + "output_" + std::to_string(n) + ".txt", want));
    json::Value body;
    CHECK(json::parse(in, body));
    if (body.is_array()) {
      json::Value b = json::Value::object();
      b.set("messages", body);
      body = b;
    }
    if (!body.has("thinking_mode")) body.set("thinking_mode", json::Value("chat"));
    std::string out, err;
    bool ok = chat::format_request(body, out, &err);
    CHECK(ok);
    if (out != want) fprintf(stderr, "  chat vector %d differs: %s\n", n, err.c_str());
    CHECK(out == want);
  }
}

static void test_chat_misc() {
  json::Value msgs;
  json::parse(R"([{"role":"user","content":"hi"},{"role":"assistant","content":"yo","reasoning_content":"r1"},
                  {"role":"user","content":"again"}])",
              msgs);
  chat::Options o;
  std::string out;
  CHECK(chat::format(msgs, json::Value(), json::Value(), o, out));
  CHECK(out ==
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough "
        "the reasoning)\n\n<｜User｜>hi<｜Assistant｜></think>yo<｜end▁of▁sentence｜><｜User｜>again<｜Assistant｜><think>");
  o.drop_thinking = false;
  o.reasoning_effort = json::Value("medium");
  CHECK(chat::format(msgs, json::Value(), json::Value(), o, out));
  CHECK(out.find("Reasoning Effort: 62") != std::string::npos);
  CHECK(out.find("<｜Assistant｜><think>r1</think>yo") != std::string::npos);
  o = chat::Options();
  o.thinking_mode = "chat";
  o.add_generation_prompt = false;
  CHECK(chat::format(msgs, json::Value(), json::Value(), o, out));
  CHECK(out == "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>yo<｜end▁of▁sentence｜><｜User｜>again");
  o.reasoning_effort = json::Value(0);
  std::string err;
  CHECK(!chat::format(msgs, json::Value(), json::Value(), o, out, &err) && !err.empty());

  // reply parsing: round trip of a rendered tool call
  std::string reply =
      "plan it</think>Let me check.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter "
      "name=\"location\" string=\"true\">Beijing</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"days\" "
      "string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  chat::Reply r = chat::parse_reply(reply, true);
  CHECK(r.reasoning == "plan it");
  CHECK(r.content == "Let me check.");
  CHECK(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0]["function"]["name"].str() == "get_weather");
  CHECK(r.tool_calls[0]["function"]["arguments"].str() == "{\"location\": \"Beijing\", \"days\": 3}");
  chat::Reply r2 = chat::parse_reply("half a thought", true);
  CHECK(r2.reasoning == "half a thought" && r2.content.empty());
  chat::Reply r3 = chat::parse_reply("plain answer<｜end▁of▁sentence｜>", false);
  CHECK(r3.content == "plain answer" && r3.tool_calls.size() == 0);
}

// ---------------------------------------------------------------------------------------------------------------
// tiny HTTP client
struct Resp {
  int status = 0;
  std::string headers, body;
};
static int connect_to(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons((uint16_t)port);
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (connect(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return -1; }
  return fd;
}
static bool read_resp(int fd, std::string& buf, Resp& r) {
  auto fill = [&]() {
    char tmp[8192];
    ssize_t k = recv(fd, tmp, sizeof tmp, 0);
    if (k <= 0) return false;
    buf.append(tmp, (size_t)k);
    return true;
  };
  size_t he;
  while ((he = buf.find("\r\n\r\n")) == std::string::npos)
    if (!fill()) return false;
  r.headers = buf.substr(0, he);
  r.status = atoi(r.headers.c_str() + 9);
  buf.erase(0, he + 4);
  std::string lh = r.headers;
  for (char& c : lh) c = (char)tolower((unsigned char)c);
  if (lh.find("transfer-encoding: chunked") != std::string::npos) {
    r.body.clear();
    while (true) {
      size_t le;
      while ((le = buf.find("\r\n")) == std::string::npos)
        if (!fill()) return false;
      size_t n = strtoul(buf.c_str(), nullptr, 16);
      while (buf.size() < le + 2 + n + 2)
        if (!fill()) return false;
      r.body += buf.substr(le + 2, n);
      buf.erase(0, le + 2 + n + 2);
      if (n == 0) return true;
    }
  }
  size_t cl = 0;
  size_t p = lh.find("content-length:");
  if (p != std::string::npos) cl = strtoul(lh.c_str() + p + 15, nullptr, 10);
  while (buf.size() < cl)
    if (!fill()) return false;
  r.body = buf.substr(0, cl);
  buf.erase(0, cl);
  return true;
}
static Resp http(int port, const std::string& method, const std::string& path, const std::string& body = "") {
  Resp r;
  int fd = connect_to(port);
  if (fd < 0) return r;
  std::string req = method + " " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  server::send_all(fd, req.data(), req.size());
  std::string buf;
  read_resp(fd, buf, r);
  close(fd);
  return r;
}
// parse SSE body into json events
static std::vector<json::Value> sse_events(const std::string& body, bool* done) {
  std::vector<json::Value> ev;
  *done = false;
  size_t p = 0;
  while ((p = body.find("data: ", p)) != std::string::npos) {
    size_t e = body.find("\n\n", p);
    std::string d = body.substr(p + 6, e - p - 6);
    if (d == "[DONE]") *done = true;
    else ev.push_back(json::parse_or_null(d));
    p = e;
  }
  return ev;
}

static void test_server(const tok::Tokenizer& tk) {
  server::EchoBackend be(tk);
  server::ServerConfig cfg;
  cfg.port = 0;
  server::Server srv(tk, be, cfg);
  std::string err;
  CHECK(srv.start(&err));
  int port = srv.port();
  CHECK(port > 0);

  Resp m = http(port, "GET", "/v1/models");
  CHECK_EQ(m.status, 200);
  CHECK(json::parse_or_null(m.body)["data"][0]["id"].str() == "macseek-echo");
  CHECK_EQ(http(port, "GET", "/nope").status, 404);
  CHECK_EQ(http(port, "POST", "/v1/chat/completions", "{bad json").status, 400);
  CHECK_EQ(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"wizard","content":"x"}]})").status, 400);

  const std::string text = "Héllo 世界 😀 echo test";
  json::Value rq = json::Value::object();
  json::Value msgs = json::Value::array();
  json::Value um = json::Value::object();
  um.set("role", json::Value("user"));
  um.set("content", json::Value(text));
  msgs.push_back(um);
  rq.set("messages", msgs);
  rq.set("thinking_mode", json::Value("chat"));

  // non-stream chat
  Resp c = http(port, "POST", "/v1/chat/completions", rq.dump());
  CHECK_EQ(c.status, 200);
  json::Value cj = json::parse_or_null(c.body);
  CHECK(cj["object"].str() == "chat.completion");
  CHECK(cj["choices"][0]["message"]["content"].str() == text);
  CHECK(cj["choices"][0]["finish_reason"].str() == "stop");
  CHECK(cj["usage"]["completion_tokens"].as_int() > 0);

  // thinking mode: echo emits </think> first -> empty reasoning, content = text
  rq.set("thinking_mode", json::Value("thinking"));
  json::Value tj = json::parse_or_null(http(port, "POST", "/v1/chat/completions", rq.dump()).body);
  CHECK(tj["choices"][0]["message"]["content"].str() == text);
  CHECK(tj["choices"][0]["message"]["reasoning_content"].is_string());

  // stream chat
  rq.set("stream", json::Value(true));
  Resp s = http(port, "POST", "/v1/chat/completions", rq.dump());
  CHECK_EQ(s.status, 200);
  CHECK(s.headers.find("text/event-stream") != std::string::npos);
  bool done;
  auto ev = sse_events(s.body, &done);
  CHECK(done);
  std::string content, reasoning, fin;
  int n_content_deltas = 0;
  for (auto& e : ev) {
    const json::Value& d = e["choices"][0]["delta"];
    content += d["content"].as_string();
    n_content_deltas += !d["content"].as_string().empty();
    reasoning += d["reasoning_content"].as_string();
    if (e["choices"][0]["finish_reason"].is_string()) fin = e["choices"][0]["finish_reason"].str();
  }
  CHECK(content == text);
  CHECK(n_content_deltas >= 4);  // streamed incrementally, not in one final chunk
  CHECK(reasoning.empty());
  CHECK(fin == "stop");

  // stop string + max_tokens
  rq.set("stream", json::Value(false));
  rq.set("thinking_mode", json::Value("chat"));
  rq.set("stop", json::Value("世"));
  json::Value st = json::parse_or_null(http(port, "POST", "/v1/chat/completions", rq.dump()).body);
  CHECK(st["choices"][0]["message"]["content"].str() == "Héllo ");
  CHECK(st["choices"][0]["finish_reason"].str() == "stop");
  rq.erase("stop");
  rq.set("max_tokens", json::Value(2));
  json::Value ml = json::parse_or_null(http(port, "POST", "/v1/chat/completions", rq.dump()).body);
  CHECK(ml["choices"][0]["finish_reason"].str() == "length");
  CHECK(ml["usage"]["completion_tokens"].as_int() == 2);
  // stream with stop string: hold-back must never leak a prefix of the stop string
  rq.erase("max_tokens");
  rq.set("stream", json::Value(true));
  rq.set("stop", json::Value("echo"));
  ev = sse_events(http(port, "POST", "/v1/chat/completions", rq.dump()).body, &done);
  content.clear();
  for (auto& e : ev) content += e["choices"][0]["delta"]["content"].as_string();
  CHECK(content == "Héllo 世界 😀 ");

  // /v1/completions (echo backend has no <｜User｜> in a raw prompt -> echoes from the start)
  json::Value cr = json::Value::object();
  cr.set("prompt", json::Value("raw prompt text"));
  json::Value cc = json::parse_or_null(http(port, "POST", "/v1/completions", cr.dump()).body);
  CHECK(cc["object"].str() == "text_completion");
  CHECK(cc["choices"][0]["text"].str() == "raw prompt text");
  cr.set("stream", json::Value(true));
  ev = sse_events(http(port, "POST", "/v1/completions", cr.dump()).body, &done);
  std::string ctext;
  for (auto& e : ev) ctext += e["choices"][0]["text"].as_string();
  CHECK(done && ctext == "raw prompt text");

  // keep-alive: two requests on one connection
  int fd = connect_to(port);
  std::string req = "GET /health HTTP/1.1\r\nHost: x\r\n\r\nGET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n";
  server::send_all(fd, req.data(), req.size());
  std::string buf;
  Resp k1, k2;
  CHECK(read_resp(fd, buf, k1) && k1.status == 200 && k1.body.find("ok") != std::string::npos);
  CHECK(read_resp(fd, buf, k2) && k2.status == 200 && k2.body.find("macseek-echo") != std::string::npos);
  close(fd);

  srv.stop();
}

int main() {
  test_json();
  test_chat_vectors();
  test_chat_misc();
  std::string path = tok::default_gguf_path(), err;
  Timer t;
  auto tk = tok::Tokenizer::load_from_gguf(path, &err);
  if (!tk) {
    fprintf(stderr, "SKIP tokenizer tests: %s\n", err.c_str());
  } else {
    fprintf(stderr, "  tokenizer load: %.1f ms\n", t.ms());
    test_tokenizer_basics(*tk);
    test_tokenizer_oracle(*tk);
    test_stream_decoder(*tk);
    test_speed(*tk);
    test_server(*tk);
  }
  fprintf(stderr, "test_tok: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
