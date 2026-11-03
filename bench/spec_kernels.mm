// bench_spec_kernels: cost of the k-token decode-exact verify matvec (kernels/spec.hpp) vs the decode matvec, on the
// Coder model's hot dense shapes. Each measurement: 40 distinct weight copies (like 40 layers, no cache reuse) in one
// command buffer, best of 5. Run under the model lock (GPU timing).
#include <algorithm>
#include <cstdio>
#include <vector>
#include "kernels/spec.hpp"
#include "../tests/quant_test_util.hpp"

using namespace qt;
using ms::q::QType;

int main() {
  struct M { const char* name; QType t; int rows, cols, groups; };
  const M ms_[] = {{"q_b", QType::Q5_K, 32768, 1280, 1}, {"o_a", QType::Q5_K, 8192, 4096, 8},
                   {"o_b", QType::Q5_K, 5120, 8192, 1},  {"q_a", QType::Q8_0, 1280, 5120, 1},
                   {"head/16", QType::Q5_K, 8080, 5120, 1}};
  auto& dev = ms::Device::get();
  const int L = 40;
  for (const M& m : ms_) {
    std::vector<uint8_t> W(ms::q::row_bytes(m.t, m.cols) * (size_t)m.rows);
    { uint32_t r = 12345; for (auto& b : W) { r = r * 1664525u + 1013904223u; b = (uint8_t)((r >> 24) & 0x3B); } }
    std::vector<Buffer> ws;
    for (int l = 0; l < L; l++) ws.push_back(upload(W.data(), W.size()));
    const size_t xs = (size_t)m.cols * m.groups;
    auto X = randvec(xs * 8, 1);
    Buffer bx = upload(X.data(), X.size() * 4), by = dev.alloc((size_t)m.rows * 8 * 4);
    auto run = [&](int k, bool single) {
      double best = 1e9;
      for (int rep = 0; rep < 5; rep++) {
        ms::Enc e;
        for (int l = 0; l < L; l++) {
          if (single) for (int t = 0; t < k; t++) ms::q::encode_matvec(e, m.t, ws[l], m.rows, m.cols, bx.sub(t * xs * 4, xs * 4), by.sub((size_t)t * m.rows * 4, (size_t)m.rows * 4), false, m.groups);
          else ms::sv::encode_matvec_k(e, m.t, ws[l], m.rows, m.cols, bx, by, k, false, m.groups);
          e.barrier();
        }
        e.wait();
        best = std::min(best, e.gpu_ms());
      }
      return best;
    };
    const double b1 = run(1, true);
    const double bytes = (double)W.size() * L;
    printf("%-8s %s [%d -> %d]: decode %.3f ms (%.0f GB/s) | k:", m.name, ms::q::type_name(m.t), m.cols, m.rows, b1, bytes / b1 / 1e6);
    for (int k = 1; k <= 8; k++) printf(" %d=%.3f(%.2fx)", k, run(k, false), run(k, false) / b1);
    printf("\n");
    for (auto& b : ws) dev.release(b);
    dev.release(bx); dev.release(by);
  }
  return 0;
}
