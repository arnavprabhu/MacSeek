// Model-backed CLI commands: gen, logits, snapshot, bench. Hooked from main.cpp.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/env.hpp"
#include "core/timer.hpp"
#include "model/ds41.hpp"
#include "store/experts.hpp"
#include "store/memguard.hpp"
#include "tok/tokenizer.hpp"

namespace {
std::string arg(int argc, char** argv, const char* k, const char* dflt = "") {
  for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1];
  return dflt;
}
bool flag(int argc, char** argv, const char* k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return true; return false; }
std::vector<int32_t> read_ids(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
  size_t n = (size_t)f.tellg() / 4; f.seekg(0);
  std::vector<int32_t> v(n); f.read((char*)v.data(), n * 4);
  return v;
}
void write_logits(const std::string& path, const std::vector<int32_t>& toks, const float* rows, int n_pos, int V, int first_pos) {
  std::ofstream f(path, std::ios::binary);
  uint32_t hdr[7] = {1, (uint32_t)toks.size(), (uint32_t)V, 0, 0, (uint32_t)first_pos, (uint32_t)n_pos};
  f.write("MSLG", 4); f.write((const char*)hdr, sizeof hdr);
  f.write((const char*)toks.data(), toks.size() * 4);
  f.write((const char*)rows, (size_t)n_pos * V * 4);
}
}  // namespace

// macseek logits --ids a.ids[,b.ids...] --out a.logits[,b.logits...]   (full logits at every position; n <= 1024 each)
static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> v; size_t a = 0;
  while (a <= s.size()) { size_t b = s.find(',', a); if (b == std::string::npos) b = s.size(); if (b > a) v.push_back(s.substr(a, b - a)); a = b + 1; }
  return v;
}
int cmd_logits(int argc, char** argv) {
  auto ins = split_csv(arg(argc, argv, "--ids")), outs = split_csv(arg(argc, argv, "--out"));
  if (ins.empty() || ins.size() != outs.size()) { fprintf(stderr, "logits: --ids and --out lists must match\n"); return 2; }
  ms::ds41::Model m;
  const int V = m.config().vocab;
  for (size_t i = 0; i < ins.size(); i++) {
    auto toks = read_ids(ins[i]);
    const int n = (int)toks.size();
    std::vector<float> lg((size_t)n * V);
    m.reset();
    ms::Timer t;
    m.prefill(toks.data(), n, lg.data());
    fprintf(stderr, "logits: %s %d tokens in %.2f s (%.1f tok/s)\n", ins[i].c_str(), n, t.ms() / 1e3, n / (t.ms() / 1e3));
    write_logits(outs[i], toks, lg.data(), n, V, 0);
  }
  if (flag(argc, argv, "--stats")) fprintf(stderr, "%s\n", m.stats_report().c_str());
  return 0;
}

// macseek snapshot --ids corpus.ids --save state.bin [--n N]
int cmd_snapshot(int argc, char** argv) {
  auto toks = read_ids(arg(argc, argv, "--ids"));
  int n = std::stoi(arg(argc, argv, "--n", "0"));
  if (n <= 0 || n > (int)toks.size()) n = (int)toks.size();
  ms::ds41::Model m;
  ms::Timer t;
  const int chunk = 8192;
  for (int i = 0; i < n; i += chunk) {
    int c = std::min(chunk, n - i);
    ms::Timer tc;
    m.prefill(toks.data() + i, c);
    fprintf(stderr, "snapshot: %d / %d tokens, chunk %.1f tok/s, avg %.1f tok/s\n", i + c, n, c / (tc.ms() / 1e3), (i + c) / (t.ms() / 1e3));
  }
  std::string path = arg(argc, argv, "--save");
  if (!m.save_state(path)) { fprintf(stderr, "save failed\n"); return 1; }
  fprintf(stderr, "snapshot: saved %s at pos %d (%.1f s)\n%s\n%s\n", path.c_str(), m.pos(), t.ms() / 1e3,
          m.prefill_profile().str().c_str(), m.stats_report().c_str());
  return 0;
}

// macseek bench --state s.bin --ids corpus.ids [--decode 128] [--prefill 2048] [--warm 8]
// Loads the snapshot, decodes N greedy tokens (after a few warm-up tokens), then reloads and times one prefill chunk
// of real corpus tokens at that depth. Prints one JSON line.
int cmd_bench(int argc, char** argv) {
  std::string st = arg(argc, argv, "--state");
  int nd = std::stoi(arg(argc, argv, "--decode", "128"));
  int np = std::stoi(arg(argc, argv, "--prefill", "2048"));
  int warm = std::stoi(arg(argc, argv, "--warm", "8"));
  auto corpus = read_ids(arg(argc, argv, "--ids"));
  ms::MemStatus m0 = ms::mem_status();
  ms::ds41::Model m;
  double load_s = 0;
  auto load = [&]() { ms::Timer t; if (!st.empty() && !m.load_state(st)) { fprintf(stderr, "load_state failed\n"); exit(1); } load_s = t.ms() / 1e3; };
  load();
  const int depth = m.pos();
  // decode: feed the corpus token at the current position, then follow greedy output
  int32_t tok = depth < (int)corpus.size() ? corpus[depth] : 1;
  for (int i = 0; i < warm; i++) tok = m.decode(tok);
  m.profile().reset();
  m.experts().reset_stats();
  ms::Timer td;
  uint64_t th = 1469598103934665603ull;   // FNV-1a over the generated ids: equal hash = identical greedy output
  std::string first;
  for (int i = 0; i < nd; i++) {
    tok = m.decode(tok);
    th = (th ^ (uint32_t)tok) * 1099511628211ull;
    if (i < 12) first += (i ? "," : "") + std::to_string(tok);
  }
  double dec_s = td.ms() / 1e3;
  auto& p = m.profile();
  ms::LayerStats es = m.experts().total_stats();
  // prefill chunk at depth
  double pf_s = 0;
  if (np > 0) {
    load();
    int s0 = std::min((int)corpus.size() - np, depth);
    if (s0 < 0) s0 = 0;
    ms::Timer tp;
    m.prefill(corpus.data() + s0, np);
    pf_s = tp.ms() / 1e3;
  }
  ms::MemStatus m1 = ms::mem_status();
  printf("{\"depth\": %d, \"decode_tokens\": %d, \"decode_tok_s\": %.3f, \"ms_per_token\": %.2f, \"gpu_wait_ms\": %.2f, "
         "\"expert_ms\": %.2f, \"engram_ms\": %.3f, \"encode_ms\": %.3f, \"gpu_ms\": %.2f, \"io_stall_ms\": %.2f, \"expert_hit_rate\": %.4f, \"miss_GB_per_token\": %.4f, "
         "\"prefill_tokens\": %d, \"prefill_tok_s\": %.2f, \"load_s\": %.2f, \"swap_grew\": %s, \"tok_hash\": \"%016llx\", \"first\": \"%s\", \"pf_issued\": %llu, \"pf_used\": %llu, \"pf_wasted\": %llu}\n",
         depth, nd, nd / dec_s, 1e3 * dec_s / nd, p.gpu_wait_ms / std::max(1, p.tokens), p.expert_ms / std::max(1, p.tokens),
         p.engram_ms / std::max(1, p.tokens), p.encode_ms / std::max(1, p.tokens), p.gpu_ms / std::max(1, p.tokens),
         p.io_stall_ms / std::max(1, p.tokens), es.hit_rate(),
         es.bytes_read / 1e9 / std::max(1, nd), np, np > 0 ? np / pf_s : 0.0, load_s, ms::swap_grew(m0, m1) ? "true" : "false", (unsigned long long)th, first.c_str(),
         (unsigned long long)es.prefetch_issued, (unsigned long long)es.prefetch_used, (unsigned long long)es.prefetch_wasted);
  if (flag(argc, argv, "--stats")) fprintf(stderr, "%s\n", m.stats_report().c_str());
  return 0;
}

// macseek gen -p "text" [-n 64] [--chat] [--temp T] [--ids file]
int cmd_gen(int argc, char** argv) {
  std::string err;
  auto tk = ms::tok::Tokenizer::load_from_gguf(ms::tok::default_gguf_path(), &err);
  if (!tk) { fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
  std::vector<int32_t> ids;
  std::string idf = arg(argc, argv, "--ids");
  if (!idf.empty()) ids = read_ids(idf);
  else {
    std::string p = arg(argc, argv, "-p");
    if (flag(argc, argv, "--chat")) p = "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>" + p + "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>";
    ids = tk->encode(p, true, false);
  }
  int n = std::stoi(arg(argc, argv, "-n", "64"));
  float temp = std::stof(arg(argc, argv, "--temp", "0"));
  bool ignore_eos = flag(argc, argv, "--ignore-eos");
  ms::ds41::Model m;
  // --state S: continue from a snapshot; the prompt is ids[pos..] (the snapshot holds ids[0..pos))
  std::string st = arg(argc, argv, "--state");
  size_t from = 0;
  if (!st.empty()) { if (!m.load_state(st)) { fprintf(stderr, "load_state failed\n"); return 1; } from = (size_t)m.pos(); }
  ms::Timer t;
  int32_t tok = m.prefill(ids.data() + from, (int)(ids.size() - from));
  double pf = t.ms() / 1e3;
  fprintf(stderr, "prefill: %zu tokens in %.2f s (%.1f tok/s)\n", ids.size() - from, pf, (ids.size() - from) / pf);
  std::string text;
  std::vector<double> win;  // tok/s per 64-token window (sustained-speed check)
  ms::Timer tw;
  ms::tok::StreamDecoder dec(*tk);
  ms::Timer td;
  int i = 0;
  for (; i < n; i++) {
    if (!ignore_eos && tk->is_eog(tok)) break;
    std::string s = dec.push(tok);
    text += s;
    if (!flag(argc, argv, "--quiet")) { fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); }
    tok = m.decode(tok, nullptr, temp, 1234 + i);
    if ((i + 1) % 64 == 0) { win.push_back(64 / (tw.ms() / 1e3)); tw.reset(); }
  }
  std::string s = dec.flush(); text += s;
  if (!flag(argc, argv, "--quiet")) fwrite(s.data(), 1, s.size(), stdout);
  printf("\n");
  fprintf(stderr, "decode: %d tokens, %.2f tok/s\n", i, i / (td.ms() / 1e3));
  if (!win.empty()) {
    std::string w; for (size_t k = 0; k < win.size(); k++) { char b[32]; snprintf(b, sizeof b, "%s%.2f", k ? "," : "", win[k]); w += b; }
    fprintf(stderr, "decode windows (tok/s per 64): %s\n", w.c_str());
  }
  std::string out = arg(argc, argv, "--out");
  if (!out.empty()) { std::ofstream f(out); f << text; }
  if (flag(argc, argv, "--stats")) fprintf(stderr, "%s\n", m.stats_report().c_str());
  return 0;
}

// macseek profile --ids corpus.ids [--n 8192] [--decode 64]: prefill n corpus tokens (MACSEEK_CHUNK sized chunks), then
// decode; prints the prefill and decode time split (where does the time go).
int cmd_profile(int argc, char** argv) {
  auto toks = read_ids(arg(argc, argv, "--ids"));
  int n = std::stoi(arg(argc, argv, "--n", "8192"));
  int nd = std::stoi(arg(argc, argv, "--decode", "64"));
  if (n > (int)toks.size()) n = (int)toks.size();
  ms::MemStatus m0 = ms::mem_status();
  ms::ds41::Model m;
  ms::Timer t;
  int32_t tok = m.prefill(toks.data(), n);
  fprintf(stderr, "%s\n", m.prefill_profile().str().c_str());
  m.experts().reset_stats();
  for (int i = 0; i < 4; i++) tok = m.decode(tok);
  m.profile().reset(); m.experts().reset_stats();
  ms::Timer td;
  for (int i = 0; i < nd; i++) tok = m.decode(tok);
  double s = td.ms() / 1e3;
  auto& p = m.profile();
  ms::LayerStats es = m.experts().total_stats();
  int k = std::max(1, p.tokens);
  fprintf(stderr, "decode @%d: %.2f tok/s | per token ms: total %.1f gpu %.1f gpu_wait %.1f expert(cpu path) %.1f io_stall %.1f "
          "encode %.2f engram %.2f | hit %.3f miss %.3f GB/token\n", m.pos(), nd / s, p.total_ms / k, p.gpu_ms / k,
          p.gpu_wait_ms / k, p.expert_ms / k, p.io_stall_ms / k, p.encode_ms / k, p.engram_ms / k, es.hit_rate(), p.io_bytes / 1e9 / k);
  ms::MemStatus m1 = ms::mem_status();
  fprintf(stderr, "mem: %s | swap grew: %s\n", m1.str().c_str(), ms::swap_grew(m0, m1) ? "YES" : "no");
  if (flag(argc, argv, "--stats")) fprintf(stderr, "%s\n", m.stats_report().c_str());
  return 0;
}
