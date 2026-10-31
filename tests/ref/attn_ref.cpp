#include "ref/attn_ref.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace attnref {

static inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

uint16_t f32_to_f16(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
float f16_to_f32(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

// precompute_freqs_cis(dim=64, ...): theta 10000 plain for ratio-0 layers; theta 160000 + YaRN(16, 65536, 32, 1)
std::vector<float> rope_freqs(bool yarn) {
  const int dim = RD;
  const double base = yarn ? 160000.0 : 10000.0;
  std::vector<float> f(dim / 2);
  for (int i = 0; i < dim / 2; i++) f[i] = (float)(1.0 / std::pow(base, (double)(float)(2 * i) / dim));
  if (yarn) {
    const double factor = 16, orig = 65536, beta_fast = 32, beta_slow = 1;
    auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(base)); };
    double low = std::max(std::floor(corr(beta_fast)), 0.0);
    double high = std::min(std::ceil(corr(beta_slow)), (double)dim - 1);
    for (int i = 0; i < dim / 2; i++) {
      float ramp = std::clamp((float)((i - low) / std::max(high - low, 1e-3)), 0.0f, 1.0f);
      float smooth = 1.0f - ramp;
      f[i] = f[i] / (float)factor * (1.0f - smooth) + f[i] * smooth;
    }
  }
  return f;
}

std::vector<float> rope_table(int max_pos, bool yarn) {
  auto f = rope_freqs(yarn);
  std::vector<float> t((size_t)max_pos * 64);
  for (int p = 0; p < max_pos; p++)
    for (int i = 0; i < 32; i++) {
      float a = (float)p * f[i];  // fp32 angle like torch.outer
      t[(size_t)p * 64 + 2 * i] = (float)std::cos((double)a);
      t[(size_t)p * 64 + 2 * i + 1] = (float)std::sin((double)a);
    }
  return t;
}

void rope(float* x, const float* cs, bool inverse) {
  for (int i = 0; i < 32; i++) {
    float c = cs[2 * i], s = inverse ? -cs[2 * i + 1] : cs[2 * i + 1];
    float a = x[2 * i], b = x[2 * i + 1];
    x[2 * i] = a * c - b * s;
    x[2 * i + 1] = a * s + b * c;
  }
}

void rmsnorm(float* x, const float* w, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float r = 1.0f / std::sqrt((float)(ss / n) + eps);
  for (int i = 0; i < n; i++) x[i] = w[i] * (x[i] * r);
}

static inline int floor_log2(float a) { return (int)((fbits(a) >> 23) & 0xFF) - 127; }
static inline int log2_ceil(float x) { uint32_t b = fbits(x); return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) ? 1 : 0); }
static inline float pow2(int e) { return bitsf((uint32_t)(e + 127) << 23); }

float round_e4m3(float x) {
  float a = std::fabs(x);
  int e = std::max(floor_log2(a), -6);
  float r = std::nearbyint(a * pow2(3 - e)) * pow2(e - 3);
  return std::copysign(r, x);
}
float round_e2m1(float x) {
  float a = std::fabs(x);
  int e = std::max(floor_log2(a), 0);
  float r = std::nearbyint(a * pow2(1 - e)) * pow2(e - 1);
  return std::copysign(r, x);
}

void fp8_sim(float* x, int n) {
  for (int b = 0; b < n; b += 32) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[b + i]));
    amax = std::max(amax, 1e-4f);
    int e = log2_ceil(amax * (1.0f / 448.0f));
    float s = pow2(e), inv = pow2(-e);
    for (int i = 0; i < 32; i++) x[b + i] = round_e4m3(std::clamp(x[b + i] * inv, -448.0f, 448.0f)) * s;
  }
}
void fp4_sim_e8m0(float* x, int n) {
  for (int b = 0; b < n; b += 32) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[b + i]));
    amax = std::max(amax, 6.0f * 0x1p-126f);
    int e = log2_ceil(amax * (1.0f / 6.0f));
    float s = pow2(e), inv = pow2(-e);
    for (int i = 0; i < 32; i++) x[b + i] = round_e2m1(std::clamp(x[b + i] * inv, -6.0f, 6.0f)) * s;
  }
}
void fp4_sim_e4m3(float* x, int n) {
  for (int b = 0; b < n; b += 16) {
    float amax = 0;
    for (int i = 0; i < 16; i++) amax = std::max(amax, std::fabs(x[b + i]));
    amax = std::max(amax, 6.0f * 0x1p-9f);
    float s = round_e4m3(std::min(amax / 6.0f, 448.0f));
    for (int i = 0; i < 16; i++) x[b + i] = round_e2m1(std::clamp(x[b + i] / s, -6.0f, 6.0f)) * s;
  }
}

void window_row(const float* kv_raw, const float* norm_w, const float* cs, bool actq, float* out) {
  memcpy(out, kv_raw, D * 4);
  rmsnorm(out, norm_w, D);
  rope(out + D - RD, cs, false);
  if (actq) fp8_sim(out, D);
}

void source_init(Source& s, int ratio) {
  s.ratio = ratio;
  s.ckv.clear(); s.ikey.clear();
  s.st_kv.assign((size_t)ratio * D, 0.0f);
  s.st_sc.assign((size_t)ratio * D, -INFINITY);
}

void compress_token(Source& s, int p, const float* ckv_in, const float* cgate_in, const float* cnorm_w,
                    const uint16_t* wk, const float* knorm_w, const std::vector<float>& tab, bool actq) {
  const int r = s.ratio;
  float lat[D];
  if (r == 1) {
    memcpy(lat, ckv_in, D * 4);
  } else {
    int slot = p % r;
    memcpy(&s.st_kv[(size_t)slot * D], ckv_in, D * 4);
    memcpy(&s.st_sc[(size_t)slot * D], cgate_in, D * 4);
    if ((p + 1) % r != 0) return;
    for (int c = 0; c < D; c++) {
      float m = -INFINITY;
      for (int j = 0; j < r; j++) m = std::max(m, s.st_sc[(size_t)j * D + c]);
      float e[8], sum = 0;
      for (int j = 0; j < r; j++) { e[j] = std::exp(s.st_sc[(size_t)j * D + c] - m); sum += e[j]; }
      float v = 0;
      for (int j = 0; j < r; j++) v += s.st_kv[(size_t)j * D + c] * (e[j] / sum);
      lat[c] = v;
    }
  }
  rmsnorm(lat, cnorm_w, D);
  const int row = p / r;
  const float* cs = &tab[(size_t)row * r * 64];
  // index key from the pre-rope latent
  float k[ID];
  for (int o = 0; o < ID; o++) {
    double acc = 0;
    for (int d = 0; d < D; d++) acc += (double)bf16_to_f32(wk[(size_t)o * D + d]) * lat[d];
    k[o] = (float)acc;
  }
  rmsnorm(k, knorm_w, ID);
  rope(k + ID - RD, cs, false);
  if (actq) fp4_sim_e8m0(k, ID);
  rope(lat + D - RD, cs, false);
  if (actq) fp4_sim_e4m3(lat, D);
  s.ikey.insert(s.ikey.end(), k, k + ID);
  s.ckv.insert(s.ckv.end(), lat, lat + D);
}

void index_q(const float* q_raw, const float* w_raw, const float* cs, bool actq, float* q, float* w) {
  memcpy(q, q_raw, IH * ID * 4);
  for (int h = 0; h < IH; h++) {
    rope(q + h * ID + ID - RD, cs, false);
    if (actq) fp4_sim_e8m0(q + h * ID, ID);
  }
  for (int h = 0; h < IH; h++) w[h] = w_raw[h] * (float)(std::pow(128.0, -0.5) * std::pow(32.0, -0.5));
}

void index_scores(const float* q, const float* w, const float* keys, int len, std::vector<float>& sc) {
  sc.assign(len, 0.0f);
  for (int t = 0; t < len; t++) {
    const float* k = keys + (size_t)t * ID;
    double s = 0;
    for (int h = 0; h < IH; h++) {
      double d = 0;
      for (int i = 0; i < ID; i++) d += (double)q[h * ID + i] * k[i];
      if (d > 0) s += d * w[h];
    }
    sc[t] = (float)s;
  }
}

static std::vector<int> topk_of(const std::vector<float>& sc, int k) {
  std::vector<int> id;
  for (int i = 0; i < (int)sc.size(); i++) if (sc[i] > -INFINITY) id.push_back(i);
  auto cmp = [&](int a, int b) { return sc[a] != sc[b] ? sc[a] > sc[b] : a < b; };
  if ((int)id.size() > k) { std::nth_element(id.begin(), id.begin() + k, id.end(), cmp); id.resize(k); }
  std::sort(id.begin(), id.end());
  return id;
}
std::vector<int> topk_sorted(const std::vector<float>& sc, int k) { return topk_of(sc, k); }

std::vector<int> candidate_blocks(const std::vector<float>& sc, int len) {
  int nb = (len + CBS - 1) / CBS;
  std::vector<float> bs(nb, -INFINITY);
  for (int t = 0; t < len; t++) bs[t / CBS] = std::max(bs[t / CBS], sc[t]);
  if (len > 0) bs[(len - 1) / CBS] = INFINITY;
  return topk_of(bs, CB);
}

void sparse_attn(const float* q_raw, const float* cs, const std::vector<const float*>& rows, const float* sinks,
                 float* out) {
  const double scale = 1.0 / std::sqrt((double)D);
  std::vector<double> s(rows.size());
  for (int h = 0; h < H; h++) {
    float q[D];
    memcpy(q, q_raw + (size_t)h * D, D * 4);
    rope(q + D - RD, cs, false);
    double m = sinks[h];
    for (size_t r = 0; r < rows.size(); r++) {
      double d = 0;
      for (int i = 0; i < D; i++) d += (double)q[i] * rows[r][i];
      s[r] = d * scale;
      m = std::max(m, s[r]);
    }
    double den = std::exp((double)sinks[h] - m);
    std::vector<double> o(D, 0.0);
    for (size_t r = 0; r < rows.size(); r++) {
      double p = std::exp(s[r] - m);
      den += p;
      for (int i = 0; i < D; i++) o[i] += p * rows[r][i];
    }
    float* y = out + (size_t)h * D;
    for (int i = 0; i < D; i++) y[i] = (float)(o[i] / den);
    rope(y + D - RD, cs, true);
  }
}

}  // namespace attnref
