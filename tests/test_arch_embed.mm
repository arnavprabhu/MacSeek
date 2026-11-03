// Token-embedding row dequant (Q6_K / Q5_K rows -> h [n][4][5120]) vs the CPU reference dequant.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/metal.hpp"
#include "kernels/arch.hpp"
#include "ref/dequant.hpp"

using namespace ms;

int main() {
  auto& d = Device::get();
  bool ok = true;
  const int n = 7, D = 5120, copies = 4;
  for (q::QType t : {q::QType::Q6_K, q::QType::Q5_K}) {
    const size_t rb = q::row_bytes(t, D);
    const uint32_t stride = (uint32_t)(rb + 16);   // rows need not be packed
    std::vector<uint8_t> blocks = ref::random_blocks(t, (int64_t)n * D, 1234u + (int)t, 0.02f);
    Buffer rows = d.alloc((size_t)n * stride), h = d.alloc((size_t)n * copies * D * 4);
    for (int i = 0; i < n; i++) memcpy((char*)rows.cpu() + (size_t)i * stride, blocks.data() + (size_t)i * rb, rb);
    { Enc e; arch::encode_embed(e, (int)t, rows, stride, h, n, copies); e.wait(); }
    std::vector<float> ref(D);
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
      ref::dequant_row(t, blocks.data() + (size_t)i * rb, ref.data(), D);
      for (int c = 0; c < copies; c++)
        for (int k = 0; k < D; k++) {
          const float g = ((const float*)h.cpu())[((size_t)i * copies + c) * D + k];
          maxerr = std::max(maxerr, (double)std::fabs(g - ref[k]) / (1e-6 + std::fabs(ref[k])));
        }
    }
    const bool o = maxerr < 1e-5;
    printf("embed %s: max rel err %.3g %s\n", q::type_name(t), maxerr, o ? "ok" : "FAIL");
    ok &= o;
  }
  printf(ok ? "test_arch_embed: ALL OK\n" : "test_arch_embed: FAILED\n");
  return ok ? 0 : 1;
}
