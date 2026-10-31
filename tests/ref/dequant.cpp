// CPU reference dequantization. Written from the public block-format descriptions; expression order mirrors the
// format definitions so results are bit-exact with libggml (verified in tests/test_quant_ref.mm).
#include "ref/dequant.hpp"
#include <cmath>
#include <cstring>
#include <random>
#include "kernels/quant_grids.h"

namespace ref {

static const uint64_t kIQ1S[2048] = {Q_IQ1S_GRID_VALUES};
static const uint64_t kIQ2XXS[256] = {Q_IQ2XXS_GRID_VALUES};
static const uint8_t kSigns[128] = {Q_KSIGNS_IQ2XS_VALUES};
static const int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const int8_t kFP4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

float fp16_to_f32(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float)v; }
uint16_t f32_to_fp16(float f) { _Float16 v = (_Float16)f; uint16_t h; memcpy(&h, &v, 2); return h; }
float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

static inline uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

static float e8m0_half(uint8_t x) {  // 2^(x-128) (MXFP4 scale, already halved to match the doubled e2m1 table)
  uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
  float f; memcpy(&f, &bits, 4); return f;
}

static void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
  else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

void dequant_row(QType t, const void* srcv, float* y, int64_t n) {
  const uint8_t* s = (const uint8_t*)srcv;
  switch (t) {
    case QType::F32: memcpy(y, s, n * 4); return;
    case QType::F16: for (int64_t i = 0; i < n; i++) y[i] = fp16_to_f32(rd16(s + 2 * i)); return;
    case QType::BF16: for (int64_t i = 0; i < n; i++) y[i] = bf16_to_f32(rd16(s + 2 * i)); return;
    case QType::Q8_0:
      for (int64_t b = 0; b < n / 32; b++, s += 34) {
        const float d = fp16_to_f32(rd16(s));
        for (int j = 0; j < 32; j++) *y++ = (int8_t)s[2 + j] * d;
      }
      return;
    case QType::Q5_K:
      for (int64_t b = 0; b < n / 256; b++, s += 176) {
        const float d = fp16_to_f32(rd16(s)), mn = fp16_to_f32(rd16(s + 2));
        const uint8_t *sc = s + 4, *qh = s + 16, *ql = s + 48;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
          uint8_t a, m;
          scale_min_k4(is, sc, a, m); const float d1 = d * a, m1 = mn * m;
          scale_min_k4(is + 1, sc, a, m); const float d2 = d * a, m2 = mn * m;
          for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
          for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
      }
      return;
    case QType::Q6_K:
      for (int64_t b = 0; b < n / 256; b++, s += 210) {
        const float d = fp16_to_f32(rd16(s + 208));
        const uint8_t *ql = s, *qh = s + 128;
        const int8_t* sc = (const int8_t*)(s + 192);
        for (int h = 0; h < 2; h++, y += 128, ql += 64, qh += 32, sc += 8)
          for (int l = 0; l < 32; l++) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
          }
      }
      return;
    case QType::IQ1_S:
      for (int64_t b = 0; b < n / 256; b++, s += 50) {
        const float d = fp16_to_f32(rd16(s));
        const uint8_t* qs = s + 2;
        for (int ib = 0; ib < 8; ib++, qs += 4) {
          const uint16_t qh = rd16(s + 34 + 2 * ib);
          const float dl = d * (2 * ((qh >> 12) & 7) + 1);
          const float delta = qh & 0x8000 ? -0.125f : 0.125f;
          for (int l = 0; l < 4; l++, y += 8) {
            const int8_t* g = (const int8_t*)&kIQ1S[qs[l] | (((qh >> 3 * l) & 7) << 8)];
            for (int j = 0; j < 8; j++) y[j] = dl * (g[j] + delta);
          }
        }
      }
      return;
    case QType::IQ2_XXS:
      for (int64_t b = 0; b < n / 256; b++, s += 66) {
        const float d = fp16_to_f32(rd16(s));
        for (int ib = 0; ib < 8; ib++) {
          const uint8_t* a8 = s + 2 + 8 * ib;
          const uint32_t a1 = rd32(a8 + 4);
          const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
          for (int l = 0; l < 4; l++, y += 8) {
            const uint8_t* g = (const uint8_t*)&kIQ2XXS[a8[l]];
            const uint8_t sg = kSigns[(a1 >> 7 * l) & 127];
            for (int j = 0; j < 8; j++) y[j] = db * g[j] * (sg & (1 << j) ? -1.f : 1.f);
          }
        }
      }
      return;
    case QType::IQ4_NL:
      for (int64_t b = 0; b < n / 32; b++, s += 18, y += 32) {
        const float d = fp16_to_f32(rd16(s));
        for (int j = 0; j < 16; j++) { y[j] = d * kIQ4NL[s[2 + j] & 0xF]; y[j + 16] = d * kIQ4NL[s[2 + j] >> 4]; }
      }
      return;
    case QType::MXFP4:
      for (int64_t b = 0; b < n / 32; b++, s += 17, y += 32) {
        const float d = e8m0_half(s[0]);
        for (int j = 0; j < 16; j++) { y[j] = kFP4[s[1 + j] & 0xF] * d; y[j + 16] = kFP4[s[1 + j] >> 4] * d; }
      }
      return;
  }
}

void matvec_ref(QType t, const void* W, int rows, int cols, const float* x, float* y) {
  std::vector<float> w(cols);
  const size_t rb = ms::q::row_bytes(t, cols);
  for (int r = 0; r < rows; r++) {
    dequant_row(t, (const uint8_t*)W + rb * r, w.data(), cols);
    double acc = 0;
    for (int c = 0; c < cols; c++) acc += (double)w[c] * x[c];
    y[r] = (float)acc;
  }
}

std::vector<uint8_t> random_blocks(QType t, int64_t n, uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int be = ms::q::block_elems(t);
  const size_t bb = ms::q::block_bytes(t);
  const int64_t nb = n / be;
  std::vector<uint8_t> out(nb * bb);
  for (auto& v : out) v = (uint8_t)byte(rng);
  auto put16 = [&](uint8_t* p, uint16_t v) { memcpy(p, &v, 2); };
  auto h = [&](float f) { return f32_to_fp16(f); };
  for (int64_t b = 0; b < nb; b++) {
    uint8_t* p = out.data() + b * bb;
    float sc = scale * (0.5f + 0.5f * std::fabs(nd(rng)));
    switch (t) {
      case QType::F32: { float f = nd(rng) * scale * 10; memcpy(p, &f, 4); } break;
      case QType::F16: put16(p, h(nd(rng) * scale * 10)); break;
      case QType::BF16: { float f = nd(rng) * scale * 10; uint32_t u; memcpy(&u, &f, 4); put16(p, (uint16_t)(u >> 16)); } break;
      case QType::Q8_0: case QType::IQ4_NL: case QType::IQ1_S: case QType::IQ2_XXS:
        put16(p, h(t == QType::Q8_0 ? sc / 16 : sc)); break;
      case QType::Q5_K: put16(p, h(sc / 8)); put16(p + 2, h(sc / 4)); break;
      case QType::Q6_K: put16(p + 208, h(sc / 32)); break;
      case QType::MXFP4: p[0] = (uint8_t)(120 + byte(rng) % 8); break;
    }
  }
  return out;
}

}  // namespace ref
