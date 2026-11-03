// GPU decode matvec vs CPU reference (dequant -> double dot) on the real model shapes, every weight type.
// Tolerance: |gpu - ref| <= tol * sum_c |w_c x_c| per row (F32 accumulation noise is ~1e-6 of that).
#include <cstdio>
#include <vector>
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;

struct Case { QType t; int rows, cols, groups; bool acc; const char* what; };

int main() {
  const Case cases[] = {
      {QType::Q6_K, 1280, 5120, 1, false, "attn_q_a"},       {QType::Q5_K, 1280, 5120, 1, false, "attn_q_a"},
      {QType::Q6_K, 32768, 1280, 1, false, "attn_q_b"},      {QType::Q5_K, 32768, 1280, 1, false, "attn_q_b"},
      {QType::Q6_K, 512, 5120, 1, false, "attn_kv"},         {QType::Q5_K, 512, 5120, 1, true, "attn_kv acc"},
      {QType::Q6_K, 8192, 4096, 8, false, "attn_output_a"},  {QType::Q5_K, 8192, 4096, 8, true, "attn_output_a acc"},
      {QType::Q6_K, 5120, 8192, 1, false, "attn_output_b"},  {QType::Q5_K, 5120, 8192, 1, false, "attn_output_b"},
      {QType::IQ1_S, 2304, 5120, 1, false, "exp gate"},      {QType::IQ1_S, 5120, 2304, 1, true, "exp down acc"},
      {QType::IQ2_XXS, 2304, 5120, 1, false, "exp gate"},    {QType::IQ2_XXS, 5120, 2304, 1, false, "exp down"},
      {QType::Q6_K, 2304, 5120, 1, false, "shexp gate"},     {QType::Q6_K, 5120, 2304, 1, false, "shexp down"},
      {QType::Q8_0, 2304, 5120, 1, false, "shexp gate"},     {QType::Q8_0, 5120, 2304, 1, true, "shexp down acc"},
      {QType::Q6_K, 129280, 5120, 1, false, "output head"},  {QType::Q6_K, 25600, 6144, 1, false, "engram_kv"},
      {QType::F32, 384, 5120, 1, false, "router"},           {QType::F32, 24, 20480, 1, false, "hc_fn"},
      {QType::BF16, 512, 5120, 1, false, "compressor_kv"},   {QType::BF16, 4096, 1280, 1, false, "indexer q_b"},
      {QType::BF16, 32, 5120, 1, false, "indexer proj"},     {QType::BF16, 128, 512, 1, false, "indexer k"},
      {QType::F16, 1280, 5120, 1, false, "f16 synthetic"},   {QType::F16, 7, 768, 1, true, "f16 odd rows"},
      {QType::Q6_K, 3, 256, 1, false, "tiny"},
      {QType::IQ1_M, 2304, 5120, 1, false, "exp gate"},      {QType::IQ1_M, 5120, 2304, 1, true, "exp down acc"},
      {QType::IQ2_XS, 2304, 5120, 1, false, "exp gate"},     {QType::IQ2_XS, 5120, 2304, 1, false, "exp down"},
      {QType::IQ1_M, 32768, 5120, 1, false, "big synthetic"}, {QType::IQ2_XS, 32768, 5120, 1, true, "big synth acc"},
      {QType::IQ1_M, 7, 768, 1, false, "iq1_m odd rows"},    {QType::IQ2_XS, 5, 512, 1, false, "iq2_xs odd rows"},
      {QType::Q5_K, 2304, 5120, 1, false, "shexp gate"},     {QType::Q5_K, 5120, 2304, 1, true, "shexp down acc"},
  };
  int fails = 0;
  for (const Case& c : cases) {
    auto W = ref::random_blocks(c.t, (int64_t)c.rows * c.cols, 7 + c.rows + c.cols);
    auto x = randvec((size_t)c.cols * c.groups, 99 + c.cols);
    auto y0 = randvec(c.rows, 5);
    std::vector<double> ry, mag;
    ref_matvec(c.t, W.data(), c.rows, c.cols, x.data(), ry, mag, c.groups);
    if (c.acc) for (int i = 0; i < c.rows; i++) { ry[i] += y0[i]; mag[i] += std::fabs(y0[i]); }
    Buffer bw = upload(W.data(), W.size()), bx = upload(x.data(), x.size() * 4), by = upload(y0.data(), y0.size() * 4);
    { ms::Enc e; ms::q::encode_matvec(e, c.t, bw, c.rows, c.cols, bx, by, c.acc, c.groups); e.wait(); }
    int worst = 0;
    double err = max_norm_err((float*)by.cpu(), ry, mag, c.rows, &worst);
    const double tol = 2e-5;
    bool ok = err <= tol;
    printf("  %-8s %-20s [%5d -> %6d] g%d  max_err %.2e %s\n", ms::q::type_name(c.t), c.what, c.cols, c.rows, c.groups,
           err, ok ? "ok" : "FAIL");
    if (!ok) { printf("     row %d gpu %.7g ref %.7g mag %.4g\n", worst, ((float*)by.cpu())[worst], ry[worst], mag[worst]); fails++; }
    auto& d = ms::Device::get(); d.release(bw); d.release(bx); d.release(by);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
