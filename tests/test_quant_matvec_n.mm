// matvec_n (n = 2..16 activation columns, one weight read) vs CPU reference, all types, incl. block-diagonal + acc.
#include <cstdio>
#include <vector>
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;

struct Case { QType t; int rows, cols, groups, n; bool acc; };

int main() {
  const Case cases[] = {
      {QType::Q6_K, 1280, 5120, 1, 2, false},   {QType::Q5_K, 1280, 5120, 1, 3, true},
      {QType::Q8_0, 2304, 5120, 1, 5, false},   {QType::IQ1_S, 2304, 5120, 1, 8, false},
      {QType::IQ2_XXS, 5120, 2304, 1, 16, true}, {QType::IQ1_S, 5120, 2304, 1, 13, false},
      {QType::Q6_K, 8192, 4096, 8, 6, false},   {QType::BF16, 512, 5120, 1, 4, false},
      {QType::F32, 24, 20480, 1, 5, false},     {QType::F16, 300, 1280, 1, 16, false},
      {QType::Q6_K, 32768, 1280, 1, 9, false},
  };
  int fails = 0;
  auto& dev = ms::Device::get();
  for (const Case& c : cases) {
    auto W = ref::random_blocks(c.t, (int64_t)c.rows * c.cols, 21 + c.rows);
    const size_t xs = (size_t)c.cols * c.groups;
    auto X = randvec(xs * c.n, 31 + c.n);
    auto Y0 = randvec((size_t)c.rows * c.n, 41);
    Buffer bw = upload(W.data(), W.size()), bx = upload(X.data(), X.size() * 4), by = upload(Y0.data(), Y0.size() * 4);
    { ms::Enc e; ms::q::encode_matvec_n(e, c.t, bw, c.rows, c.cols, bx, by, c.n, c.acc, c.groups); e.wait(); }
    double err = 0;
    for (int t = 0; t < c.n; t++) {
      std::vector<double> ry, mag;
      ref_matvec(c.t, W.data(), c.rows, c.cols, X.data() + xs * t, ry, mag, c.groups);
      if (c.acc) for (int i = 0; i < c.rows; i++) { ry[i] += Y0[(size_t)t * c.rows + i]; mag[i] += std::fabs(Y0[(size_t)t * c.rows + i]); }
      err = std::max(err, max_norm_err((float*)by.cpu() + (size_t)t * c.rows, ry, mag, c.rows));
    }
    bool ok = err <= 2e-5;
    printf("  %-8s [%5d -> %5d] g%d n=%-2d %s max_err %.2e %s\n", ms::q::type_name(c.t), c.cols, c.rows, c.groups, c.n,
           c.acc ? "acc" : "   ", err, ok ? "ok" : "FAIL");
    fails += !ok;
    dev.release(bw); dev.release(bx); dev.release(by);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
