#pragma once
// Synthetic decode weights / state with the REAL DeepSeek-V4.1-Flash shapes and types per layer for bench/decode_layer.mm
// and tests/test_dec_fuse.mm. Two layouts (dsyn::layout()): Layout::UD = the Unsloth UD-IQ1_S GGUF (docs/tensors.txt,
// 384 experts, Q6_K/Q5_K attention), Layout::Coder = the ISTA-DASLab GSQ-RCO Coder GGUF (docs/tensors_coder.txt: 192
// experts, IQ1_M/IQ2_XXS/IQ2_XS routed per layer, Q5_K shared/head/engram_kv/q_b/o_a/o_b, Q8_0 q_a/kv, BF16 router
// converted to F32 at load). Values are random but valid blocks with scales chosen so the
// activations stay finite through 40 layers. Header-only (bench targets do not link macseek_ref).
#include <dispatch/dispatch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "core/metal.hpp"
#include "kernels/quant.hpp"
#include "model/config.hpp"
#include "model/dec_layer.hpp"

namespace dsyn {
using ms::Buffer;
using ms::q::QType;
using ms::ds41::W;
using ms::ds41::LayerW;

inline uint64_t sm64(uint64_t& s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
inline float unif(uint64_t& s) { return (float)((sm64(s) >> 40) * (1.0 / 16777216.0)) * 2.f - 1.f; }  // [-1, 1)
inline uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
inline uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

// Fill `rows` rows of `cols` elements of type t (raw GGUF block layout) at dst. rms ~ scale for dense / quant values.
inline void fill_rows(QType t, uint8_t* dst, int64_t rows, int64_t cols, uint64_t seed, float scale) {
  const size_t rb = ms::q::row_bytes(t, cols);
  const int64_t chunk = 64;
  dispatch_apply((size_t)((rows + chunk - 1) / chunk), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
    uint64_t s = seed * 0x100000001b3ull + ci * 7919 + 1;
    const int64_t r0 = (int64_t)ci * chunk, r1 = std::min(rows, r0 + chunk);
    for (int64_t r = r0; r < r1; r++) {
      uint8_t* p = dst + (size_t)r * rb;
      switch (t) {
        case QType::F32: for (int64_t i = 0; i < cols; i++) { float f = unif(s) * scale * 1.7f; memcpy(p + 4 * i, &f, 4); } break;
        case QType::F16: for (int64_t i = 0; i < cols; i++) { uint16_t h = f2h(unif(s) * scale * 1.7f); memcpy(p + 2 * i, &h, 2); } break;
        case QType::BF16: for (int64_t i = 0; i < cols; i++) { uint16_t h = f2bf(unif(s) * scale * 1.7f); memcpy(p + 2 * i, &h, 2); } break;
        default: {
          for (size_t i = 0; i + 8 <= rb; i += 8) { uint64_t v = sm64(s); memcpy(p + i, &v, 8); }
          for (size_t i = rb / 8 * 8; i < rb; i++) p[i] = (uint8_t)sm64(s);
          const size_t bb = ms::q::block_bytes(t);
          const int64_t nb = cols / ms::q::block_elems(t);
          for (int64_t b = 0; b < nb; b++) {
            uint8_t* k = p + b * bb;
            const float j = 0.75f + 0.5f * (unif(s) * 0.5f + 0.5f);
            uint16_t h;
            switch (t) {
              case QType::Q8_0: h = f2h(scale * j / 74.f); memcpy(k, &h, 2); break;
              case QType::Q6_K:
                for (int q = 0; q < 16; q++) k[192 + q] = (uint8_t)(int8_t)((int)(sm64(s) % 33) - 16);
                h = f2h(scale * j / 165.f); memcpy(k + 208, &h, 2); break;
              case QType::Q5_K:
                for (int q = 0; q < 12; q++) k[4 + q] &= 0x0F;
                h = f2h(scale * j / 128.f); memcpy(k, &h, 2); h = f2h(scale * j / 256.f); memcpy(k + 2, &h, 2); break;
              case QType::IQ1_S: h = f2h(scale * j / 8.f); memcpy(k, &h, 2); break;
              case QType::IQ2_XXS: h = f2h(scale * j / 40.f); memcpy(k, &h, 2); break;
              case QType::IQ2_XS: h = f2h(scale * j / 40.f); memcpy(k, &h, 2); break;
              case QType::IQ1_M: {  // fp16 d = top nibbles of the 4 scale words (bytes 48..55)
                h = f2h(scale * j / 8.f);
                for (int w = 0; w < 4; w++) {
                  uint16_t sw; memcpy(&sw, k + 48 + 2 * w, 2);
                  sw = (uint16_t)((sw & 0x0FFF) | (((h >> (4 * w)) & 0xF) << 12));
                  memcpy(k + 48 + 2 * w, &sw, 2);
                }
                break;
              }
              case QType::IQ4_NL: h = f2h(scale * j / 60.f); memcpy(k, &h, 2); break;
              default: break;
            }
          }
        }
      }
    }
  });
}

// Weight pool: buffers keyed by (name, type, rows, cols, copy) so 40 layers can share a few copies per tensor class.
struct Pool {
  std::map<std::tuple<std::string, int, int, int, int>, Buffer> m;
  size_t bytes = 0;
  uint64_t seed = 1;
  W get(const std::string& name, QType t, int rows, int cols, int copy, float scale) {
    auto key = std::make_tuple(name, (int)t, rows, cols, copy);
    auto it = m.find(key);
    W w; w.t = t; w.rows = rows; w.cols = cols;
    if (it != m.end()) { w.b = it->second; return w; }
    const size_t nb = ms::q::row_bytes(t, cols) * (size_t)rows;
    Buffer b = ms::Device::get().alloc(nb, name.c_str());
    fill_rows(t, (uint8_t*)b.cpu(), rows, cols, seed++, scale);
    bytes += nb;
    m[key] = b;
    w.b = b;
    return w;
  }
  // F32 vector around `center` (+- spread)
  W vec(const std::string& name, int n, int copy, float center, float spread) {
    auto key = std::make_tuple(name, 0, 1, n, copy);
    auto it = m.find(key);
    W w; w.t = QType::F32; w.rows = 1; w.cols = n;
    if (it != m.end()) { w.b = it->second; return w; }
    Buffer b = ms::Device::get().alloc((size_t)n * 4, name.c_str());
    uint64_t s = seed++ * 31 + 7;
    float* f = (float*)b.cpu();
    for (int i = 0; i < n; i++) f[i] = center + spread * unif(s);
    bytes += (size_t)n * 4;
    m[key] = b;
    w.b = b;
    return w;
  }
};

enum class Layout { UD, Coder };
// layout used by make_layer / make_pred / the per-layer type helpers (default: the UD-IQ1_S build, as before)
inline Layout& layout() { static Layout l = Layout::UD; return l; }
inline bool coder() { return layout() == Layout::Coder; }
inline int n_experts() { return coder() ? 192 : 384; }

// per-layer quant types. UD-IQ1_S GGUF:
inline QType attn_type(int l) { return l < 27 ? QType::Q6_K : QType::Q5_K; }
// GSQ-RCO Coder (docs/tensors_coder.txt): routed gate(=up) / down type per layer, M = IQ1_M, X = IQ2_XXS, S = IQ2_XS
inline QType coder_code(char c) { return c == 'M' ? QType::IQ1_M : c == 'X' ? QType::IQ2_XXS : QType::IQ2_XS; }
inline const char* coder_gu() { return "MMMMMMMMMMXXXMMXMSXSMMXXXXXSXXMXXXSSXXSS"; }
inline const char* coder_dn() { return "MMMMMMMXXMXMMMMMMMXMMMXSMXSMXMMMMSMMMMXS"; }
inline QType qa_type(int l) { return coder() ? QType::Q8_0 : attn_type(l); }     // attn_q_a and attn_kv
inline QType qb_type(int l) { return coder() ? QType::Q5_K : attn_type(l); }     // attn_q_b, attn_output_a/b
inline QType shexp_type(int l) {
  if (coder()) return QType::Q5_K;
  return (l >= 14 && l < 27) ? QType::Q8_0 : QType::Q6_K;
}
inline QType exp_gu_type(int l) {
  if (coder()) return coder_code(coder_gu()[l]);
  return ((l >= 30 && l <= 34) || l == 39) ? QType::IQ2_XXS : QType::IQ1_S;
}
inline QType exp_down_type(int l) {
  if (coder()) return coder_code(coder_dn()[l]);
  return l < 20 ? QType::IQ1_S : QType::IQ2_XXS;
}
inline QType head_type() { return coder() ? QType::Q5_K : QType::Q6_K; }    // output head and token_embd
inline QType engram_kv_type() { return coder() ? QType::Q5_K : QType::Q6_K; }

// Layer l's weights; tensors of the same class are shared between layers with the same `copy` (= l % copies).
inline LayerW make_layer(Pool& P, const ms::ds41::Config& cfg, int l, int copies) {
  const int c = l % copies;
  const QType ta = qa_type(l), tb = qb_type(l), ts = shexp_type(l);
  auto tn = [&](const char* s, QType t) { return std::string(s) + "_" + ms::q::type_name(t); };
  LayerW L;
  L.attn_norm = P.vec("attn_norm", 5120, c, 1.f, 0.1f);
  L.ffn_norm = P.vec("ffn_norm", 5120, c, 1.f, 0.1f);
  L.q_a = P.get(tn("q_a", ta), ta, 1280, 5120, c, 0.014f);
  L.q_a_norm = P.vec("q_a_norm", 1280, c, 1.f, 0.1f);
  L.q_b = P.get(tn("q_b", tb), tb, 32768, 1280, c, 0.028f);
  L.kv = P.get(tn("kv", ta), ta, 512, 5120, c, 0.014f);
  L.kv_norm = P.vec("kv_norm", 512, c, 1.f, 0.1f);
  L.o_a = P.get(tn("o_a", tb), tb, 8192, 4096, c, 0.016f);
  L.o_b = P.get(tn("o_b", tb), tb, 5120, 8192, c, 0.011f);
  L.sinks = P.vec("sinks", 64, c, 0.f, 1.f);
  if (cfg.is_kv_source(l)) {
    L.c_kv = P.get("c_kv", QType::BF16, 512, 5120, c, 0.014f);
    if (cfg.compress_ratio[l] == 2) L.c_gate = P.get("c_gate", QType::BF16, 512, 5120, c, 0.014f);
    L.c_norm = P.vec("c_norm", 512, c, 1.f, 0.1f);
    L.idx_k = P.get("idx_k", QType::BF16, 128, 512, c, 0.044f);
    L.idx_knorm = P.vec("idx_knorm", 128, c, 1.f, 0.1f);
  }
  if (cfg.is_index_source(l)) {
    L.idx_qb = P.get("idx_qb", QType::BF16, 4096, 1280, c, 0.028f);
    L.idx_proj = P.get("idx_proj", QType::BF16, 32, 5120, c, 0.014f);
  }
  L.gate_inp = P.get("gate_inp", QType::F32, n_experts(), 5120, c, 0.03f);
  L.bias = P.vec("exp_bias", n_experts(), c, 0.f, 0.1f);
  const std::string sn = ms::q::type_name(ts);
  L.sh_gate = P.get("sh_gate_" + sn, ts, 2304, 5120, c, 0.014f);
  L.sh_up = P.get("sh_up_" + sn, ts, 2304, 5120, c, 0.014f);
  L.sh_down = P.get("sh_down_" + sn, ts, 5120, 2304, c, 0.021f);
  L.hc_attn_fn = P.get("hc_attn_fn", QType::F32, 24, 20480, c, 0.007f);
  L.hc_attn_scale = P.vec("hc_attn_scale", 3, c, 1.f, 0.2f);
  L.hc_attn_base = P.vec("hc_attn_base", 24, c, 0.f, 0.5f);
  L.hc_ffn_fn = P.get("hc_ffn_fn", QType::F32, 24, 20480, c, 0.007f);
  L.hc_ffn_scale = P.vec("hc_ffn_scale", 3, c, 1.f, 0.2f);
  L.hc_ffn_base = P.vec("hc_ffn_base", 24, c, 0.f, 0.5f);
  if (cfg.is_engram(l)) {
    // one copy per engram layer (each is read once per token in the real model)
    L.engram_kv = P.get("engram_kv", engram_kv_type(), 25600, 6144, l, 0.0128f);
    L.engram_gate = P.get("engram_gate", QType::F32, 4, 5120, l, 0.014f);
  }
  return L;
}

// Q8_0 router copy for next-layer prediction (as Model builds it)
inline W make_pred(Pool& P, int l, int copies) { return P.get("pred_w", QType::Q8_0, n_experts(), 5120, l % copies, 0.03f); }

// random half values in [-a, a) into n halves
inline void fill_half(void* dst, size_t n, uint64_t seed, float a) {
  const size_t chunk = 1 << 20;
  dispatch_apply((n + chunk - 1) / chunk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
    uint64_t s = seed * 1000003ull + ci;
    uint16_t* d = (uint16_t*)dst;
    const size_t i1 = std::min(n, (ci + 1) * chunk);
    for (size_t i = ci * chunk; i < i1; i++) d[i] = f2h(unif(s) * a);
  });
}

// Attention state at position `pos`: window rings, compressed KV / index keys of every source up to pos.
inline void fill_attn_state(ms::attn::State& S, int pos, uint64_t seed) {
  using namespace ms::attn;
  for (int L = 0; L < kLayers; L++) fill_half(S.win[L].cpu(), (size_t)kWin * kD, seed + L, 1.f);
  const int srcs[4] = {2, 8, 14, 20};
  for (int i = 0; i < 4; i++) {
    const size_t rows = (size_t)pos / ratio_of(srcs[i]);
    fill_half(S.ckv[i].cpu(), rows * kD, seed + 100 + i, 1.f);
    fill_half(S.ikey[i].cpu(), rows * kID, seed + 200 + i, 1.f);
  }
}

}  // namespace dsyn
