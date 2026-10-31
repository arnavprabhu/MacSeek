#pragma once
// Shared helpers for the quant module tests / benches (header-only).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "core/metal.hpp"
#include "kernels/quant.hpp"
#include "ref/dequant.hpp"

namespace qt {
using ms::Buffer;
using ms::q::QType;

inline Buffer upload(const void* p, size_t n, const char* label = nullptr) {
  Buffer b = ms::Device::get().alloc(n, label);
  memcpy(b.cpu(), p, n);
  return b;
}
inline std::vector<float> randvec(size_t n, uint32_t seed, float scale = 1.f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, scale);
  std::vector<float> v(n);
  for (auto& f : v) f = nd(rng);
  return v;
}
// |sum w x| reference plus sum |w x| (error scale) per row, double precision, on dequantized weights.
inline void ref_matvec(QType t, const uint8_t* W, int rows, int cols, const float* x, std::vector<double>& y,
                       std::vector<double>& mag, int groups = 1) {
  y.assign(rows, 0); mag.assign(rows, 0);
  std::vector<float> w(cols);
  const size_t rb = ms::q::row_bytes(t, cols);
  const int rpg = rows / groups;
  for (int r = 0; r < rows; r++) {
    ref::dequant_row(t, W + rb * r, w.data(), cols);
    const float* xg = x + (size_t)(r / rpg) * cols;
    double s = 0, m = 0;
    for (int c = 0; c < cols; c++) { double p = (double)w[c] * xg[c]; s += p; m += std::fabs(p); }
    y[r] = s; mag[r] = m;
  }
}
// max over rows of |got - ref| / (mag + tiny); returns worst normalized error
inline double max_norm_err(const float* got, const std::vector<double>& ref, const std::vector<double>& mag,
                           int n, int* worst = nullptr) {
  double e = 0;
  for (int i = 0; i < n; i++) {
    double d = std::fabs(got[i] - ref[i]) / (mag[i] + 1e-12);
    if (!(d <= e)) { e = d; if (worst) *worst = i; }  // NaN-safe
  }
  return e;
}
}  // namespace qt
