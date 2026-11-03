// Speculative-verify kernels (kernels/spec.hpp) must be BIT-identical per token to the decode kernels they mirror:
// k-token matvec vs k separate encode_matvec calls; k-token shared expert vs encode_moe_decode_table_part(Shared).
#include <cstdio>
#include <cstring>
#include <vector>
#include "kernels/spec.hpp"
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;

static int cmp_bits(const float* a, const float* b, size_t n, const char* what) {
  size_t bad = 0, first = (size_t)-1;
  for (size_t i = 0; i < n; i++)
    if (memcmp(a + i, b + i, 4)) { if (!bad) first = i; bad++; }
  if (bad) printf("    %s: %zu / %zu differ (first %zu: %.9g vs %.9g)\n", what, bad, n, first, a[first], b[first]);
  return bad ? 1 : 0;
}

int main() {
  struct Case { QType t; int rows, cols, groups; bool acc; };
  const Case cases[] = {
      {QType::Q8_0, 1280, 5120, 1, false},  {QType::Q8_0, 512, 5120, 1, false},   {QType::Q5_K, 32768, 1280, 1, false},
      {QType::Q5_K, 8192, 4096, 8, false},  {QType::Q5_K, 5120, 8192, 1, false},  {QType::Q5_K, 4096, 1280, 1, false},
      {QType::Q6_K, 5120, 8192, 1, true},   {QType::BF16, 512, 5120, 1, false},   {QType::BF16, 32, 5120, 1, false},
      {QType::F32, 192, 5120, 1, false},    {QType::F16, 300, 1280, 1, false},    {QType::Q5_K, 25600, 6144, 1, false},
      {QType::Q8_0, 192, 5120, 1, false},
  };
  int fails = 0;
  auto& dev = ms::Device::get();
  for (const Case& c : cases) {
    auto W = ref::random_blocks(c.t, (int64_t)c.rows * c.cols, 7 + c.rows);
    Buffer bw = upload(W.data(), W.size());
    const size_t xs = (size_t)c.cols * c.groups;
    for (int k : {1, 2, 3, 5, 6, 8}) {
      auto X = randvec(xs * k, 100 + k);
      auto Y0 = randvec((size_t)c.rows * k, 200 + k);
      Buffer bx = upload(X.data(), X.size() * 4), y1 = upload(Y0.data(), Y0.size() * 4), y2 = upload(Y0.data(), Y0.size() * 4);
      {
        ms::Enc e;
        for (int t = 0; t < k; t++)
          ms::q::encode_matvec(e, c.t, bw, c.rows, c.cols, bx.sub((size_t)t * xs * 4, xs * 4),
                               y1.sub((size_t)t * c.rows * 4, (size_t)c.rows * 4), c.acc, c.groups);
        ms::sv::encode_matvec_k(e, c.t, bw, c.rows, c.cols, bx, y2, k, c.acc, c.groups);
        e.wait();
      }
      char what[96];
      snprintf(what, sizeof what, "%s [%d -> %d] g%d k=%d", ms::q::type_name(c.t), c.cols, c.rows, c.groups, k);
      int f = cmp_bits((float*)y1.cpu(), (float*)y2.cpu(), (size_t)c.rows * k, what);
      if (!f) printf("  matvec_k %-48s bit-identical\n", what);
      fails += f;
      dev.release(bx); dev.release(y1); dev.release(y2);
    }
    dev.release(bw);
  }
  // shared expert
  for (QType ts : {QType::Q5_K, QType::Q6_K, QType::Q8_0}) {
    const int D = ms::q::kExpertDim, I = ms::q::kExpertInter;
    auto G = ref::random_blocks(ts, (int64_t)I * D, 1), U = ref::random_blocks(ts, (int64_t)I * D, 2),
         Dn = ref::random_blocks(ts, (int64_t)D * I, 3);
    Buffer bg = upload(G.data(), G.size()), bu = upload(U.data(), U.size()), bd = upload(Dn.data(), Dn.size());
    ms::q::ExpertSlot sh{bg, bu, bd};
    for (int k : {1, 3, 4, 6}) {
      auto X = randvec((size_t)D * k, 300 + k, 0.5f);
      auto Y0 = randvec((size_t)D * k, 400 + k);
      Buffer bx = upload(X.data(), X.size() * 4), y1 = upload(Y0.data(), Y0.size() * 4), y2 = upload(Y0.data(), Y0.size() * 4);
      Buffer scr = dev.alloc(ms::q::moe_decode_scratch_bytes() * k), h = dev.alloc((size_t)k * I * 4), tbl = dev.alloc(4096);
      Buffer wts = dev.alloc(64);
      {
        ms::Enc e;
        for (int t = 0; t < k; t++) {
          ms::q::MoeDecodeArgs m;
          m.t_gate_up = QType::IQ2_XXS; m.t_down = QType::IQ2_XXS; m.n_routed = 6; m.weights = wts;
          m.has_shared = true; m.t_shared = ts; m.shared = sh;
          m.x = bx.sub((size_t)t * D * 4, (size_t)D * 4); m.y = y1.sub((size_t)t * D * 4, (size_t)D * 4);
          m.scratch = scr.sub(ms::q::moe_decode_scratch_bytes() * t, ms::q::moe_decode_scratch_bytes());
          ms::q::encode_moe_decode_table_part(e, m, tbl, ms::q::MoePart::Shared);
        }
        e.barrier();
        ms::sv::encode_moe_shared_k(e, ts, sh, bx, h, y2, k);
        e.wait();
      }
      char what[64];
      snprintf(what, sizeof what, "shared %s k=%d", ms::q::type_name(ts), k);
      int f = cmp_bits((float*)y1.cpu(), (float*)y2.cpu(), (size_t)D * k, what);
      if (!f) printf("  moe_shared_k %-40s bit-identical\n", what);
      fails += f;
      dev.release(bx); dev.release(y1); dev.release(y2); dev.release(scr); dev.release(h); dev.release(tbl); dev.release(wts);
    }
    dev.release(bg); dev.release(bu); dev.release(bd);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
