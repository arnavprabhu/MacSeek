// Prefill matmul (tensor API) vs CPU reference. Two references per case:
//  (a) "half-exact": weights and activations rounded to half (the kernel's defined numerics), double accumulation
//  (b) full F32 dequant reference: documents the error the half staging adds vs the F32 decode path.
#include <cmath>
#include <cstdio>
#include <vector>
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;

static float r16(float v) { return (float)(_Float16)v; }

struct Case { QType t; int rows, cols, n; bool acc; const char* what; };

int main() {
  const Case cases[] = {
      {QType::Q6_K, 1280, 5120, 77, false, "attn_q_a"},   {QType::Q5_K, 1024, 1280, 160, true, "attn_q_b part"},
      {QType::Q6_K, 512, 5120, 64, false, "attn_kv"},     {QType::Q8_0, 2304, 5120, 33, false, "shexp gate"},
      {QType::Q6_K, 640, 8192, 40, false, "attn_out_b"},  {QType::IQ1_S, 2304, 5120, 50, false, "exp gate"},
      {QType::IQ2_XXS, 5120, 2304, 37, true, "exp down"}, {QType::IQ1_S, 300, 2304, 129, false, "odd rows"},
      {QType::BF16, 512, 5120, 48, false, "compressor"},  {QType::F16, 256, 1280, 70, false, "f16"},
      {QType::F32, 384, 5120, 36, false, "router f32"},
      {QType::IQ1_M, 2304, 5120, 50, false, "exp gate"},  {QType::IQ2_XS, 5120, 2304, 37, true, "exp down"},
      {QType::IQ1_M, 300, 2304, 129, true, "odd rows"},   {QType::IQ2_XS, 2304, 5120, 77, false, "exp up"},
      // n >= 512: whole-weight pre-dequant (q_deq_f16_*) + dense GEMM
      {QType::IQ1_M, 200, 2304, 520, false, "predeq"},    {QType::IQ2_XS, 136, 5120, 512, true, "predeq"},
      {QType::Q5_K, 160, 2304, 515, false, "predeq"},     {QType::IQ2_XXS, 128, 2304, 513, false, "predeq"},
  };
  int fails = 0;
  auto& dev = ms::Device::get();
  for (const Case& c : cases) {
    auto W = ref::random_blocks(c.t, (int64_t)c.rows * c.cols, 3 + c.rows);
    auto X = randvec((size_t)c.n * c.cols, 11 + c.n);
    auto Y0 = randvec((size_t)c.n * c.rows, 12);
    std::vector<float> wd((size_t)c.rows * c.cols);
    const size_t rb = ms::q::row_bytes(c.t, c.cols);
    for (int r = 0; r < c.rows; r++) ref::dequant_row(c.t, W.data() + rb * r, wd.data() + (size_t)r * c.cols, c.cols);
    Buffer bw = upload(W.data(), W.size()), bx = upload(X.data(), X.size() * 4), by = upload(Y0.data(), Y0.size() * 4);
    { ms::Enc e; ms::q::encode_matmul(e, c.t, bw, c.rows, c.cols, bx, by, c.n, c.acc); e.wait(); }
    const float* got = (const float*)by.cpu();
    double err_h = 0, err_f = 0;
    for (int t = 0; t < c.n; t++)
      for (int r = 0; r < c.rows; r++) {
        const float* w = wd.data() + (size_t)r * c.cols;
        const float* x = X.data() + (size_t)t * c.cols;
        double sh = 0, sf = 0, mag = 0;
        for (int k = 0; k < c.cols; k++) {
          sh += (double)(c.t == QType::F32 ? w[k] : r16(w[k])) * r16(x[k]);
          sf += (double)w[k] * x[k];
          mag += std::fabs((double)w[k] * x[k]);
        }
        double base = c.acc ? Y0[(size_t)t * c.rows + r] : 0;
        if (c.acc) mag += std::fabs(base);
        double g = got[(size_t)t * c.rows + r];
        err_h = std::max(err_h, std::fabs(g - (sh + base)) / (mag + 1e-12));
        err_f = std::max(err_f, std::fabs(g - (sf + base)) / (mag + 1e-12));
      }
    bool ok = err_h <= 1e-5 && err_f <= 2e-3;
    printf("  %-8s %-14s [%5d -> %5d] n=%-4d err(half-exact) %.2e  err(vs F32) %.2e  %s\n", ms::q::type_name(c.t),
           c.what, c.cols, c.rows, c.n, err_h, err_f, ok ? "ok" : "FAIL");
    fails += !ok;
    dev.release(bw); dev.release(bx); dev.release(by);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
