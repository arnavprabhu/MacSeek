#include "ref/arch_ref.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ref_arch {

static constexpr int D = 5120, HC = 4, HCD = 20480, MIX = 24;

float half_to_float(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = std::ldexp((float)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp((float)(m | 1024), (int)e - 25);
  return s ? -v : v;
}

static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

void hc_split(const double* m, const float* scale, const float* base, float* mix, int iters, double eps) {
  for (int j = 0; j < 4; j++) mix[j] = (float)(sigmoid(m[j] * scale[0] + base[j]) + eps);
  for (int j = 0; j < 4; j++) mix[4 + j] = (float)(2.0 * sigmoid(m[4 + j] * scale[1] + base[4 + j]));
  double c[4][4];
  for (int j = 0; j < 4; j++)
    for (int k = 0; k < 4; k++) c[j][k] = m[8 + 4 * j + k] * scale[2] + base[8 + 4 * j + k];
  for (int j = 0; j < 4; j++) {  // row softmax + eps
    double mx = std::max({c[j][0], c[j][1], c[j][2], c[j][3]}), s = 0;
    for (int k = 0; k < 4; k++) { c[j][k] = std::exp(c[j][k] - mx); s += c[j][k]; }
    for (int k = 0; k < 4; k++) c[j][k] = c[j][k] / s + eps;
  }
  auto colnorm = [&] {
    for (int k = 0; k < 4; k++) {
      double s = c[0][k] + c[1][k] + c[2][k] + c[3][k];
      for (int j = 0; j < 4; j++) c[j][k] /= (s + eps);
    }
  };
  colnorm();
  for (int it = 1; it < iters; it++) {
    for (int j = 0; j < 4; j++) {
      double s = c[j][0] + c[j][1] + c[j][2] + c[j][3];
      for (int k = 0; k < 4; k++) c[j][k] /= (s + eps);
    }
    colnorm();
  }
  for (int j = 0; j < 4; j++)
    for (int k = 0; k < 4; k++) mix[8 + 4 * j + k] = (float)c[j][k];
}

void hc_mixes(const float* h, const float* fn, const float* scale, const float* base, float* mix, int iters,
              double norm_eps, double hc_eps) {
  double ss = 0;
  for (int i = 0; i < HCD; i++) ss += (double)h[i] * h[i];
  double rstd = 1.0 / std::sqrt(ss / HCD + norm_eps);
  double m[MIX];
  for (int r = 0; r < MIX; r++) {
    double a = 0;
    const float* f = fn + (size_t)r * HCD;
    for (int i = 0; i < HCD; i++) a += (double)f[i] * h[i];
    m[r] = a * rstd;
  }
  hc_split(m, scale, base, mix, iters, hc_eps);
}

void rmsnorm(const float* x, const float* w, float* y, int dim, double eps) {
  double ss = 0;
  for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
  double r = 1.0 / std::sqrt(ss / dim + eps);
  for (int i = 0; i < dim; i++) y[i] = (float)(x[i] * r * (w ? w[i] : 1.0f));
}

void hc_step(const float* h_in, const float* x_sub, const float* mix_prev, const float* fn, const float* scale,
             const float* base, const float* norm_w, float* h_out, float* mix_out, float* x_out) {
  std::vector<float> hn(h_in, h_in + HCD);
  if (x_sub) {
    // model.py hc_post: y[i] = post[i]*x + sum_j comb[j][i] * residual[j]
    for (int d = 0; d < D; d++)
      for (int i = 0; i < HC; i++) {
        double a = (double)mix_prev[4 + i] * x_sub[d];
        for (int j = 0; j < HC; j++) a += (double)mix_prev[8 + 4 * j + i] * h_in[j * D + d];
        hn[i * D + d] = (float)a;
      }
    std::memcpy(h_out, hn.data(), HCD * sizeof(float));
  }
  if (fn) hc_mixes(hn.data(), fn, scale, base, mix_out);
  if (!x_out) return;
  std::vector<float> xp(D);
  for (int d = 0; d < D; d++) {
    double a = 0;
    for (int j = 0; j < HC; j++) a += (double)mix_prev[j] * hn[j * D + d];
    xp[d] = (float)a;
  }
  rmsnorm(xp.data(), norm_w, x_out, D, 1e-20);
}

int g_n_experts = 384;

void router(const float* logits, const float* bias, int* ids, float* w, float route_scale) {
  const int E = g_n_experts;
  std::vector<double> sc(E), key(E);
  for (int e = 0; e < E; e++) {
    double x = logits[e];
    double sp = x > 20 ? x : std::log1p(std::exp(x));
    sc[e] = std::sqrt(sp);
    key[e] = (double)(float)((float)sc[e] + bias[e]);
  }
  std::vector<int> ord(E);
  std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return key[a] > key[b]; });
  double s = 0;
  for (int k = 0; k < 6; k++) s += sc[ord[k]];
  for (int k = 0; k < 6; k++) { ids[k] = ord[k]; w[k] = (float)(sc[ord[k]] / (s + 1e-20) * route_scale); }
}

void route_lists(const int* ids, const float* w, int n, std::vector<uint32_t>& counts, std::vector<uint32_t>& offsets,
                 std::vector<uint32_t>& entry, std::vector<uint32_t>& tok, std::vector<float>& lw,
                 std::vector<uint32_t>& slot_of) {
  const int E = g_n_experts, N = n * 6;
  counts.assign(E, 0);
  offsets.assign(E + 1, 0);
  entry.assign(N, 0);
  tok.assign(N, 0);
  lw.assign(N, 0);
  slot_of.assign(N, 0);
  for (int i = 0; i < N; i++) counts[ids[i]]++;
  for (int e = 0; e < E; e++) offsets[e + 1] = offsets[e] + counts[e];
  std::vector<uint32_t> fill(offsets.begin(), offsets.end() - 1);
  for (int i = 0; i < N; i++) {
    uint32_t p = fill[ids[i]]++;
    entry[p] = i; tok[p] = i / 6; lw[p] = w[i]; slot_of[i] = p;
  }
}

void engram_gate(float* h, const float* kv, const float* w, float* gate, double eps, double clamp) {
  const float* key = kv;
  const float* val = kv + HCD;
  for (int j = 0; j < HC; j++) {
    double hh = 0, kk = 0, hk = 0;
    for (int d = 0; d < D; d++) {
      double a = h[j * D + d], b = key[j * D + d];
      hh += a * a; kk += b * b; hk += a * w[j * D + d] * b;
    }
    double rstd = 1.0 / std::sqrt(hh / D + eps) / std::sqrt(kk / D + eps);
    double dot = hk * rstd / std::sqrt((double)D);
    double s = std::sqrt(std::max(std::fabs(dot), clamp));
    gate[j] = (float)sigmoid(std::copysign(s, dot));
  }
  for (int j = 0; j < HC; j++)
    for (int d = 0; d < D; d++) h[j * D + d] = (float)(h[j * D + d] + (double)gate[j] * val[d]);
}

static const int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void iq4nl_row(const uint8_t* src, float* dst, int n) {
  for (int b = 0; b < n / 32; b++) {
    const uint8_t* blk = src + b * 18;
    uint16_t hd;
    std::memcpy(&hd, blk, 2);
    float d = half_to_float(hd);
    for (int j = 0; j < 16; j++) {
      dst[b * 32 + j] = d * kIQ4NL[blk[2 + j] & 15];
      dst[b * 32 + j + 16] = d * kIQ4NL[blk[2 + j] >> 4];
    }
  }
}

void q6k_row(const uint8_t* src, float* dst, int n) {
  for (int b = 0; b < n / 256; b++) {
    const uint8_t* blk = src + b * 210;
    uint16_t hd;
    std::memcpy(&hd, blk + 208, 2);
    float d = half_to_float(hd);
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    float* y = dst + b * 256;
    for (int half = 0; half < 2; half++) {
      for (int l = 0; l < 32; l++) {
        int is = l / 16;
        int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      ql += 64; qh += 32; sc += 8; y += 128;
    }
  }
}

int argmax(const float* x, int n) {
  int b = 0;
  for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
  return b;
}

void topk(const float* x, int n, int k, float* v, uint32_t* idx) {
  std::vector<uint32_t> o(n);
  std::iota(o.begin(), o.end(), 0u);
  std::partial_sort(o.begin(), o.begin() + k, o.end(), [&](uint32_t a, uint32_t b) {
    return x[a] > x[b] || (x[a] == x[b] && a < b);
  });
  for (int i = 0; i < k; i++) { idx[i] = o[i]; v[i] = x[o[i]]; }
}

double logsumexp(const float* x, int n, double scale) {
  double m = -INFINITY;
  for (int i = 0; i < n; i++) m = std::max(m, x[i] * scale);
  double s = 0;
  for (int i = 0; i < n; i++) s += std::exp(x[i] * scale - m);
  return m + std::log(s);
}

void log_softmax(const float* x, float* y, int n) {
  double l = logsumexp(x, n);
  for (int i = 0; i < n; i++) y[i] = (float)(x[i] - l);
}

}  // namespace ref_arch
