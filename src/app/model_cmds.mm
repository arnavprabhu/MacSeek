#include <algorithm>
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
    // --decode K: positions 0..K-1 by prefill, the rest token by token through decode() (teacher forcing). Checks the
    // decode kernels (hc step, router, decode attention, MoE table path) against the same oracle as prefill.
    const int dk = std::stoi(arg(argc, argv, "--decode", "0"));
    if (dk > 0 && dk < n) {
      m.prefill(toks.data(), dk, lg.data());
      for (int p = dk; p < n; p++) m.decode(toks[p], lg.data() + (size_t)p * V);
    } else {
      m.prefill(toks.data(), n, lg.data());
    }
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
  const int chunk = (int)std::max<long>(1, std::atol(getenv("MACSEEK_SNAP_PIECE") ? getenv("MACSEEK_SNAP_PIECE") : "32768"));
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
  std::vector<int32_t> gen_toks;   // greedy output (token source for --vexact past the end of the corpus)
  for (int i = 0; i < nd; i++) {
    tok = m.decode(tok);
    gen_toks.push_back(tok);
    th = (th ^ (uint32_t)tok) * 1099511628211ull;
    if (i < 12) first += (i ? "," : "") + std::to_string(tok);
  }
  double dec_s = td.ms() / 1e3;
  auto& p = m.profile();
  ms::LayerStats es = m.experts().total_stats();
  // --verify K: cost of K-token forward passes through the multi-token path right after decode (speculative-verify
  // cost model): 16 consecutive passes over the corpus continuation, warm expert cache. Printed to stderr.
  const int vk = std::stoi(arg(argc, argv, "--verify", "0"));
  if (vk > 0) {
    int at = m.pos();
    std::vector<double> ts;
    for (int r = 0; r < 16 && at + vk <= (int)corpus.size(); r++, at += vk) {
      ms::Timer tv;
      m.prefill(corpus.data() + at, vk);
      ts.push_back(tv.ms());
    }
    std::sort(ts.begin(), ts.end());
    if (!ts.empty()) fprintf(stderr, "verify: K=%d passes %zu median %.1f ms min %.1f ms max %.1f ms\n", vk, ts.size(), ts[ts.size() / 2], ts[0], ts.back());
  }
  // --vexact K: cost of K-token exact verify passes (Model::verify, the speculative path) on the corpus continuation,
  // all K tokens kept; 16 passes, warm cache. Printed to stderr next to the decode ms/token.
  for (const std::string& vxs : split_csv(arg(argc, argv, "--vexact", ""))) {
    const int vx = std::stoi(vxs);
    if (vx <= 0) continue;
    int at = m.pos();
    std::vector<double> ts;
    std::vector<int32_t> nx(vx);
    // tokens: the corpus continuation, or (past its end) the greedy output generated above, replayed
    const bool from_corpus = at + 18 * vx <= (int)corpus.size();
    std::vector<int32_t> src = from_corpus ? std::vector<int32_t>(corpus.begin() + at, corpus.begin() + at + 18 * vx) : gen_toks;
    m.spec_profile().reset();
    for (int r = 0, o = 0; r < 18 && o + vx <= (int)src.size() && m.pos() + vx <= m.max_pos(); r++, o += vx) {
      ms::Timer tv;
      m.verify(src.data() + o, vx, nx.data());
      m.verify_keep(vx);
      if (r >= 2) ts.push_back(tv.ms());   // first passes warm the verify buffers
      (void)at;
    }
    std::sort(ts.begin(), ts.end());
    auto& sp = m.spec_profile();
    if (!ts.empty())
      fprintf(stderr, "vexact: K=%d passes %zu median %.1f ms (%.2fx decode %.1f ms/token) min %.1f max %.1f | per pass: gpu %.1f wait %.1f "
              "expert %.1f io_stall %.1f keep %.2f engram %.2f io %.0f MB\n", vx, ts.size(), ts[ts.size() / 2],
              ts[ts.size() / 2] / (1e3 * dec_s / nd), 1e3 * dec_s / nd, ts[0], ts.back(), sp.gpu_ms / sp.passes,
              sp.wait_ms / sp.passes, sp.expert_ms / sp.passes, sp.io_stall_ms / sp.passes, sp.keep_ms / sp.passes,
              sp.engram_ms / sp.passes, sp.io_bytes / 1e6 / sp.passes);
  }
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

// macseek spec-check --ids prompt.ids [--state S] [--n 64] [--k 4] [--reject]
// Exactness of Model::verify: greedy-decode n tokens (recording logits), restore the state, then feed the same tokens
// through verify in passes of k (with --reject, keep fewer tokens per pass to exercise the rollback) and compare the
// greedy tokens and the logits bit for bit with decode().
int cmd_speccheck(int argc, char** argv) {
  auto ids = read_ids(arg(argc, argv, "--ids"));
  const int n = std::stoi(arg(argc, argv, "--n", "64"));
  const int k = std::stoi(arg(argc, argv, "--k", "4"));
  const bool reject = flag(argc, argv, "--reject");
  ms::ds41::Model m;
  const int V = m.config().vocab;
  std::string st = arg(argc, argv, "--state");
  size_t from = 0;
  if (!st.empty()) { if (!m.load_state(st)) { fprintf(stderr, "load_state failed\n"); return 1; } from = (size_t)m.pos(); }
  int32_t tok = m.prefill(ids.data() + from, (int)(ids.size() - from));
  auto snap = m.snapshot();
  std::vector<int32_t> seq{tok}, out;
  std::vector<float> lg((size_t)n * V);
  ms::Timer td;
  for (int i = 0; i < n; i++) { int32_t nx = m.decode(seq[i], lg.data() + (size_t)i * V); out.push_back(nx); seq.push_back(nx); }
  const double dms = td.ms() / n;
  if (!m.restore(*snap)) { fprintf(stderr, "restore failed\n"); return 1; }
  int i = 0, pass = 0, tok_bad = 0, row_bad = 0;
  double maxd = 0;
  std::vector<float> vl((size_t)k * V);
  std::vector<int32_t> nx(k);
  ms::Timer tv;
  while (i < n) {
    const int kk = std::min(k, n - i);
    m.verify(seq.data() + i, kk, nx.data(), vl.data());
    int keep = reject ? 1 + (pass * 7 + 3) % kk : kk;
    for (int j = 0; j < keep; j++) {
      if (nx[j] != out[i + j]) { tok_bad++; if (tok_bad <= 5) fprintf(stderr, "spec-check: token %d: verify %d decode %d\n", i + j, nx[j], out[i + j]); }
      const float* a = vl.data() + (size_t)j * V; const float* b = lg.data() + (size_t)(i + j) * V;
      if (memcmp(a, b, (size_t)V * 4)) {
        row_bad++;
        for (int c = 0; c < V; c++) maxd = std::max(maxd, (double)std::fabs(a[c] - b[c]));
      }
    }
    m.verify_keep(keep);
    i += keep; pass++;
  }
  printf("spec-check: n %d k %d%s passes %d | token mismatches %d | logit rows not bit-identical %d (max |diff| %.3g) | "
         "decode %.1f ms/token, verify %.1f ms/pass\n", n, k, reject ? " (reject)" : "", pass, tok_bad, row_bad, maxd, dms,
         tv.ms() / pass);
  return tok_bad || row_bad ? 1 : 0;
}
