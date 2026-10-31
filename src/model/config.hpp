#pragma once
// DeepSeek-V4.1-Flash-1017 hyper-parameters and the derived per-layer plan (SPEC.md §2-3).
// Values are the released shapes; Config::validate() checks them against GGUF metadata at load time.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ms::ds41 {

struct Config {
  int n_layers = 40, dim = 5120, vocab = 129280, n_heads = 64, head_dim = 512, rope_dim = 64;
  int q_lora = 1280, o_groups = 8, o_lora = 1024, window = 128;
  int n_experts = 384, n_active = 6, inter = 2304;
  float route_scale = 1.5f, swiglu_limit = 10.0f;
  int hc = 4, sinkhorn_iters = 20;
  float hc_eps = 1e-6f, norm_eps = 1e-20f;
  int idx_heads = 32, idx_dim = 128, idx_topk = 512;
  int cand_source = 20, cand_topk_blocks = 2048, cand_block = 8;
  float rope_theta = 10000.f, compress_rope_theta = 160000.f, rope_factor = 16.f;
  int original_ctx = 65536; float beta_fast = 32.f, beta_slow = 1.f;
  std::vector<int> compress_ratio = std::vector<int>(40, 0);
  std::vector<int> kv_sources{2, 8, 14, 20}, index_sources{2, 8, 14, 20, 24, 28, 32, 36}, engram_layers{1, 14};
  std::vector<int> dspark_targets{37, 38, 39};
  int engram_heads = 8, engram_head_dim = 256, engram_max_ngram = 4;

  Config() { for (int l = 2; l < 20; l++) compress_ratio[l] = 2; for (int l = 20; l < 40; l++) compress_ratio[l] = 1; }

  // nearest source <= layer, or -1
  static int nearest(const std::vector<int>& src, int layer) { int r = -1; for (int s : src) if (s <= layer) r = s; return r; }
  bool is_kv_source(int l) const { for (int s : kv_sources) if (s == l) return true; return false; }
  bool is_index_source(int l) const { for (int s : index_sources) if (s == l) return true; return false; }
  bool is_engram(int l) const { for (int s : engram_layers) if (s == l) return true; return false; }
  int kv_source_of(int l) const { return compress_ratio[l] ? nearest(kv_sources, l) : -1; }
  int index_source_of(int l) const { return compress_ratio[l] ? nearest(index_sources, l) : -1; }
  bool uses_candidates(int l) const { return cand_source >= 0 && l > cand_source && is_index_source(l); }
  // RoPE: ratio-0 layers use plain theta 10000 without YaRN; ratio>0 layers use 160000 with YaRN.
  float theta_of(int l) const { return compress_ratio[l] ? compress_rope_theta : rope_theta; }
  bool yarn_of(int l) const { return compress_ratio[l] != 0; }
  // number of compressed rows a source layer holds at context length n
  int64_t compressed_rows(int l, int64_t n) const { return compress_ratio[l] ? n / compress_ratio[l] : 0; }
};

}  // namespace ms::ds41
