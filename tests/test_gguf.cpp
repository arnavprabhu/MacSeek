// GGUF parser test: all 5 splits of the real model vs docs/tensors.txt (all 1043 tensors: name, type, dims, nbytes,
// file, absolute offset) + typed metadata spot checks. Skips (exit 0) if the model files are absent.
#include "gguf/gguf.hpp"
#include "core/timer.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static std::string repo_root() { std::string f = __FILE__; return f.substr(0, f.rfind("/tests/")); }

static std::string dims_str(const ms::TensorInfo& t) {
  std::string s = "[";
  for (int d = 0; d < t.n_dims; d++) { if (d) s += ", "; s += std::to_string(t.ne[d]); }
  return s + "]";
}

int main() {
  // type table sanity
  CHECK(ms::type_bytes(ms::GType::IQ1_S, 256) == 50, "IQ1_S size");
  CHECK(ms::type_bytes(ms::GType::IQ2_XXS, 256) == 66, "IQ2_XXS size");
  CHECK(ms::type_bytes(ms::GType::IQ4_NL, 256) == 144, "IQ4_NL size");
  CHECK(ms::type_bytes(ms::GType::Q6_K, 5120) == 4200, "Q6_K size");
  CHECK(ms::Gguf::expand_splits("/a/b-00002-of-00005.gguf").size() == 5, "expand");
  CHECK(ms::Gguf::expand_splits("/a/b.gguf").size() == 1, "expand single");

  std::string path = ms::Gguf::default_model_path();
  if (!ms::Gguf::exists(path)) { printf("test_gguf: model not found (%s), SKIP\n", path.c_str()); return 0; }
  ms::Timer t;
  ms::Gguf g(path);
  double ms_parse = t.ms();
  // two supported builds: Unsloth UD-IQ1_S (5 splits, 384 experts) and GSQ-RCO Coder (6 splits, 192 experts)
  const bool coder = g.n_splits() == 6;
  const char* table = coder ? "/docs/tensors_coder.txt" : "/docs/tensors.txt";
  CHECK(g.n_splits() == (coder ? 6 : 5), "splits %d", g.n_splits());
  CHECK(g.tensors().size() == 1043, "tensors %zu", g.tensors().size());
  CHECK(g.alignment() == 32, "align");

  // tensor table vs docs/tensors.txt
  std::ifstream in(repo_root() + table);
  CHECK(in.good(), "%s missing", table);
  std::string line; size_t n = 0, ok = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> col; std::stringstream ss(line); std::string c;
    while (std::getline(ss, c, '\t')) col.push_back(c);
    CHECK(col.size() == 6, "bad line %s", line.c_str());
    if (col.size() != 6) continue;
    n++;
    const ms::TensorInfo* ti = g.find(col[0]);
    CHECK(ti, "missing tensor %s", col[0].c_str());
    if (!ti) continue;
    bool good = true;
    good &= std::string(ms::type_name(ti->type)) == col[1];
    good &= dims_str(*ti) == col[2];
    good &= std::to_string(ti->nbytes) == col[3];
    good &= g.basename(ti->split) == col[4];
    good &= std::to_string(ti->offset) == col[5];
    CHECK(good, "mismatch %s: got %s %s %llu %s %llu", col[0].c_str(), ms::type_name(ti->type), dims_str(*ti).c_str(),
          (unsigned long long)ti->nbytes, g.basename(ti->split).c_str(), (unsigned long long)ti->offset);
    ok += good;
  }
  CHECK(n == 1043, "docs lines %zu", n);

  // metadata: typed scalars and arrays
  CHECK(g.get_str("general.architecture") == "deepseek41", "arch");
  CHECK(g.get_i64("deepseek41.block_count") == 40, "block_count");
  CHECK(g.get_i64("deepseek41.expert_count") == (coder ? 192 : 384), "experts");
  CHECK(g.kv("deepseek41.expert_weights_norm") && g.kv("deepseek41.expert_weights_norm")->type == ms::KvType::BOOL &&
        g.kv("deepseek41.expert_weights_norm")->u == 1, "bool");
  CHECK(g.get_f64("deepseek41.expert_weights_scale") == 1.5, "f32 scalar");
  CHECK(g.get_arr_str("tokenizer.ggml.tokens").size() == 129280, "tokens %zu", g.get_arr_str("tokenizer.ggml.tokens").size());
  CHECK(g.get_arr_str("tokenizer.ggml.merges").size() == 127741, "merges");
  CHECK(g.get_arr_i64("tokenizer.ggml.token_type").size() == 129280, "token_type");
  auto tm = g.get_arr_i64("deepseek41.engram.token_map");
  CHECK(tm.size() == 129280 && tm[7] == 7, "token_map");
  auto mult = g.get_arr_i64("deepseek41.ple.layer_multipliers");
  CHECK(mult.size() == 8 && mult[0] == 76632096046245LL && mult[7] == 82619226485591LL, "multipliers");
  auto ho = g.get_arr_i64("deepseek41.ple.head_offsets");
  CHECK(ho.size() == 48 && ho[1] == 16000057, "head_offsets");
  auto cr = g.get_arr_i64("deepseek41.attention.compress_ratios");
  CHECK(cr.size() == 43 && cr[0] == 0 && cr[2] == 2 && cr[39] == 1, "compress_ratios");
  auto clamp = g.get_arr_f32("deepseek41.swiglu_clamp_exp");
  CHECK(clamp.size() == 40 && clamp[0] == 10.0f, "f32 array");
  CHECK(g.get_i64("split.count") == (coder ? 6 : 5), "split.count");
  CHECK(g.get_str("tokenizer.chat_template").size() > 1000, "chat template");
  const auto* pe = g.find("per_layer_token_embd.weight");
  CHECK(pe && pe->row_bytes() == 144 && pe->ne[1] == 768022850 && pe->offset == (coder ? 416u : 192u), "engram table");

  printf("test_gguf: parsed %d splits, %zu tensors, %zu kv in %.1f ms; %zu/%zu tensor rows match docs/tensors.txt; %s\n",
         g.n_splits(), g.tensors().size(), g.keys().size(), ms_parse, ok, n, fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
