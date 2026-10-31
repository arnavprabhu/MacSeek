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

// macseek logits --ids in.ids --out out.logits     (full logits at every position; n <= 1024)
int cmd_logits(int argc, char** argv) {
  auto toks = read_ids(arg(argc, argv, "--ids"));
  std::string out = arg(argc, argv, "--out");
  ms::ds41::Model m;
  const int V = m.config().vocab, n = (int)toks.size();
  std::vector<float> lg((size_t)n * V);
  ms::Timer t;
  m.prefill(toks.data(), n, lg.data());
  fprintf(stderr, "logits: %d tokens in %.2f s (%.1f tok/s)\n", n, t.ms() / 1e3, n / (t.ms() / 1e3));
  write_logits(out, toks, lg.data(), n, V, 0);
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
  fprintf(stderr, "snapshot: saved %s at pos %d (%.1f s)\n%s\n", path.c_str(), m.pos(), t.ms() / 1e3, m.stats_report().c_str());
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
  for (int i = 0; i < nd; i++) tok = m.decode(tok);
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
         "\"expert_ms\": %.2f, \"engram_ms\": %.3f, \"encode_ms\": %.3f, \"expert_hit_rate\": %.4f, \"miss_GB_per_token\": %.4f, "
         "\"prefill_tokens\": %d, \"prefill_tok_s\": %.2f, \"load_s\": %.2f, \"swap_grew\": %s}\n",
         depth, nd, nd / dec_s, 1e3 * dec_s / nd, p.gpu_wait_ms / std::max(1, p.tokens), p.expert_ms / std::max(1, p.tokens),
         p.engram_ms / std::max(1, p.tokens), p.encode_ms / std::max(1, p.tokens), es.hit_rate(),
         es.bytes_read / 1e9 / std::max(1, nd), np, np > 0 ? np / pf_s : 0.0, load_s, ms::swap_grew(m0, m1) ? "true" : "false");
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
  ms::ds41::Model m;
  ms::Timer t;
  int32_t tok = m.prefill(ids.data(), (int)ids.size());
  double pf = t.ms() / 1e3;
  fprintf(stderr, "prefill: %zu tokens in %.2f s (%.1f tok/s)\n", ids.size(), pf, ids.size() / pf);
  ms::tok::StreamDecoder dec(*tk);
  ms::Timer td;
  int i = 0;
  for (; i < n; i++) {
    if (tk->is_eog(tok)) break;
    std::string s = dec.push(tok);
    fwrite(s.data(), 1, s.size(), stdout); fflush(stdout);
    tok = m.decode(tok, nullptr, temp, 1234 + i);
  }
  std::string s = dec.flush(); fwrite(s.data(), 1, s.size(), stdout);
  printf("\n");
  fprintf(stderr, "decode: %d tokens, %.2f tok/s\n", i, i / (td.ms() / 1e3));
  if (flag(argc, argv, "--stats")) fprintf(stderr, "%s\n", m.stats_report().c_str());
  return 0;
}
