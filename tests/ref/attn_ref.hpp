// CPU reference for the attention module (mirrors reference/inference/model.py + kernel.py semantics).
// Everything in fp32 (fp64 accumulation where cheap) — this is the math spec the GPU kernels are tested against.
#pragma once
#include <cstdint>
#include <vector>

namespace attnref {

constexpr int D = 512, RD = 64, H = 64, W = 128, IH = 32, ID = 128, TOPK = 512, CB = 2048, CBS = 8;

// ---- rope ----
std::vector<float> rope_freqs(bool yarn);                        // 32 frequencies
std::vector<float> rope_table(int max_pos, bool yarn);           // [max_pos][32][2] = (cos, sin)
void rope(float* x64, const float* cs32x2, bool inverse);        // interleaved pairs, in place

// ---- elementwise ----
void rmsnorm(float* x, const float* w, int n, float eps = 1e-20f);
float round_e4m3(float x);   // |x| <= 448
float round_e2m1(float x);   // |x| <= 6
void fp8_sim(float* x, int n);       // act_quant(block 32, ue8m0 scale), in place
void fp4_sim_e8m0(float* x, int n);  // fp4_act_quant(block 32, ue8m0), in place
void fp4_sim_e4m3(float* x, int n);  // fp4_act_quant(block 16, e4m3 scale), in place
inline float bf16_to_f32(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
uint16_t f32_to_f16(float f);
float f16_to_f32(uint16_t h);

// ---- layer metadata ----
inline int ratio_of(int L) { return L < 2 ? 0 : (L < 20 ? 2 : 1); }

// ---- window KV row: rmsnorm(kv_norm) -> rope(last 64, pos) -> fp8 sim ----
void window_row(const float* kv_raw, const float* norm_w, const float* cs, bool actq, float* out);

// ---- compressed KV source (layers 2, 8, 14: ratio 2; layer 20: ratio 1) ----
struct Source {
  int ratio = 2;
  std::vector<float> ckv;   // [rows][512] roped + fp4(e4m3,16)
  std::vector<float> ikey;  // [rows][128] roped + fp4(e8m0,32)
  std::vector<float> st_kv, st_sc;  // [ratio][512] partial group
  int rows() const { return (int)(ikey.size() / ID); }
};
void source_init(Source& s, int ratio);
// Feed one token at position p (compressor kv/gate outputs). Emits a row when its group completes.
void compress_token(Source& s, int p, const float* ckv_in, const float* cgate_in, const float* cnorm_w,
                    const uint16_t* wk_bf16 /*[128][512]*/, const float* knorm_w, const std::vector<float>& tab_yarn,
                    bool actq);

// ---- indexer ----
// prepared query: rope (pos) + fp4 sim; weights scaled by 128^-0.5*32^-0.5
void index_q(const float* q_raw, const float* w_raw, const float* cs, bool actq, float* q_out /*32*128*/, float* w_out);
// scores over keys[0..len): score[t] = sum_h relu(q_h.k_t) * w_h
void index_scores(const float* q, const float* w, const float* keys, int len, std::vector<float>& sc);
// top-k among finite scores (ties: lower index first), returned sorted ascending by index
std::vector<int> topk_sorted(const std::vector<float>& sc, int k);
// candidate blocks (layer 20): block max over 8, newest partial block pinned, top 2048 -> sorted block ids
std::vector<int> candidate_blocks(const std::vector<float>& sc, int len);

// ---- sparse attention (one query) ----
// q_raw [64][512] pre-rope; rows: list of row pointers (512 floats each); out [64][512] post inverse rope
void sparse_attn(const float* q_raw, const float* cs, const std::vector<const float*>& rows, const float* sinks,
                 float* out);

}  // namespace attnref
