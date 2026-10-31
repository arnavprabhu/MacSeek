// matvec_n (n = 2..16 activation columns, one weight read) vs CPU reference, all types, incl. block-diagonal + acc.
// The kernel stages x in threadgroup memory as half: checked tightly against a reference with x rounded to half,
// and loosely (documented error) against the pure F32 reference.
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
    double err = 0, errf = 0;
    std::vector<float> xh(X.size());
    for (size_t i = 0; i < X.size(); i++) xh[i] = (float)(_Float16)X[i];
    for (int t = 0; t < c.n; t++) {
      for (int hv = 0; hv < 2; hv++) {
        std::vector<double> ry, mag;
        ref_matvec(c.t, W.data(), c.rows, c.cols, (hv ? xh.data() : X.data()) + xs * t, ry, mag, c.groups);
        if (c.acc) for (int i = 0; i < c.rows; i++) { ry[i] += Y0[(size_t)t * c.rows + i]; mag[i] += std::fabs(Y0[(size_t)t * c.rows + i]); }
        double e = max_norm_err((float*)by.cpu() + (size_t)t * c.rows, ry, mag, c.rows);
        if (hv) err = std::max(err, e); else errf = std::max(errf, e);
      }
    }
    bool ok = err <= 2e-6 && errf <= 2e-4;
    printf("  %-8s [%5d -> %5d] g%d n=%-2d %s err(x half) %.2e  err(vs F32) %.2e %s\n", ms::q::type_name(c.t), c.cols,
           c.rows, c.groups, c.n, c.acc ? "acc" : "   ", err, errf, ok ? "ok" : "FAIL");
    fails += !ok;
    dev.release(bw); dev.release(bx); dev.release(by);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
