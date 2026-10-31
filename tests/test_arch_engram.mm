// Engram gate (GPU) + IQ4_NL engram-row dequant + Q6_K embedding-row dequant vs CPU reference.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/metal.hpp"
#include "kernels/arch.hpp"
#include "ref/arch_ref.hpp"
#include "ref/arch_testutil.hpp"

using namespace ms;
using arch_test::close;

static Buffer up(const void* p, size_t n) {
  Buffer b = Device::get().alloc(n);
  memcpy(b.cpu(), p, n);
  return b;
}

// random block bytes with a finite, moderate f16 scale at the given offset within each block
static void rand_blocks(arch_test::Rng& r, uint8_t* p, size_t nblk, size_t bsize, size_t d_off) {
  for (size_t b = 0; b < nblk; b++) {
    for (size_t i = 0; i < bsize; i++) p[b * bsize + i] = (uint8_t)r.u32();
    uint16_t h = (uint16_t)((r.u32() & 0x8000) | ((uint16_t)(8 + r.u32() % 10) << 10) | (r.u32() & 1023));
    memcpy(p + b * bsize + d_off, &h, 2);
  }
}

static bool bitexact(const char* what, const float* a, const float* b, size_t n) {
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += memcmp(&a[i], &b[i], 4) != 0;
  printf("  %-44s %s (%zu / %zu differ)\n", what, bad ? "FAIL" : "ok   bit-exact", bad, n);
  return bad == 0;
}

int main() {
  arch_test::Rng r(7);
  auto& d = Device::get();
  auto s = arch::Scratch::create(64, 4);
  bool ok = true;

  // ---- engram gate ----
  std::vector<float> wg(4 * 5120);
  bool real_w = arch_test::read_tensor_f32("blk.1.engram_gate.weight", wg);
  if (!real_w) for (auto& v : wg) v = r.uniform(0.2f, 2.0f);
  else {
    double mn = 1e30, mx = -1e30, sum = 0;
    for (float v : wg) { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); sum += v; }
    printf("real blk.1.engram_gate [4][5120]: min %.4g max %.4g mean %.4g\n", mn, mx, sum / wg.size());
  }
  for (int n : {1, 3, 33}) {
    std::vector<float> h(n * 20480), kv(n * 25600);
    r.fill_normal(h.data(), h.size(), 1.0f);
    r.fill_normal(kv.data(), kv.size(), 0.5f);
    // make some tokens have strongly aligned / anti-aligned keys so the gate covers its range
    for (int t = 0; t < n; t++)
      for (int j = 0; j < 4; j++) {
        float a = (t + j) % 3 == 0 ? 3.0f : (t + j) % 3 == 1 ? -3.0f : 0.0f;
        for (int i = 0; i < 5120; i++) kv[t * 25600 + j * 5120 + i] += a * h[t * 20480 + j * 5120 + i];
      }
    Buffer bh = up(h.data(), h.size() * 4), bkv = up(kv.data(), kv.size() * 4), bw = up(wg.data(), wg.size() * 4);
    Buffer bg = d.alloc(n * 16);
    { Enc e; arch::encode_engram_gate(e, bh, bkv, bw, n, s, bg); e.wait(); }
    std::vector<float> rg(n * 4);
    for (int t = 0; t < n; t++) ref_arch::engram_gate(&h[t * 20480], &kv[t * 25600], wg.data(), &rg[t * 4]);
    printf("engram_gate n=%d (%s weights), gates[0..3] = %.4f %.4f %.4f %.4f\n", n, real_w ? "real" : "synthetic", rg[0],
           rg[1], rg[2], rg[3]);
    ok &= close("gate", (float*)bg.cpu(), rg.data(), rg.size(), 1e-5f, 1e-4f);
    ok &= close("h", (float*)bh.cpu(), h.data(), h.size(), 1e-5f, 1e-5f);
  }

  // ---- IQ4_NL rows ----
  {
    const int n_rows = 24 * 5, stride = 160;  // padded stride exercises row_stride
    std::vector<uint8_t> raw(n_rows * stride);
    for (int i = 0; i < n_rows; i++) rand_blocks(r, &raw[i * stride], 8, 18, 0);
    Buffer bs = up(raw.data(), raw.size()), bd = d.alloc(n_rows * 256 * 4);
    { Enc e; arch::encode_iq4nl_rows(e, bs, stride, bd, n_rows); e.wait(); }
    std::vector<float> ref(n_rows * 256);
    for (int i = 0; i < n_rows; i++) ref_arch::iq4nl_row(&raw[i * stride], &ref[i * 256], 256);
    printf("iq4nl_rows synthetic\n");
    ok &= bitexact("rows", (float*)bd.cpu(), ref.data(), ref.size());
    arch_test::TensorInfo t;
    if (arch_test::find_tensor("per_layer_token_embd.weight", t) && !arch_test::model_dir().empty()) {
      const uint64_t rows[6] = {0, 1, 16000057, 384006168, 500000000, 768022849};
      std::vector<uint8_t> rr(6 * 144);
      bool rd = true;
      for (int i = 0; i < 6; i++) rd &= arch_test::read_tensor_bytes(t, rows[i] * 144, 144, &rr[i * 144]);
      Buffer bs2 = up(rr.data(), rr.size()), bd2 = d.alloc(6 * 256 * 4);
      { Enc e; arch::encode_iq4nl_rows(e, bs2, 144, bd2, 6); e.wait(); }
      std::vector<float> ref2(6 * 256);
      for (int i = 0; i < 6; i++) ref_arch::iq4nl_row(&rr[i * 144], &ref2[i * 256], 256);
      printf("iq4nl_rows real per_layer_token_embd rows (read %s), row0[0..3] = %g %g %g %g\n", rd ? "ok" : "FAILED",
             ref2[0], ref2[1], ref2[2], ref2[3]);
      ok &= rd && bitexact("real rows", (float*)bd2.cpu(), ref2.data(), ref2.size());
    }
  }

  // ---- Q6_K embedding rows -> 4 hc copies ----
  {
    const int n = 5, stride = 4200;
    std::vector<uint8_t> raw(n * stride);
    for (int i = 0; i < n; i++) rand_blocks(r, &raw[i * stride], 20, 210, 208);
    arch_test::TensorInfo t;
    bool real = arch_test::find_tensor("token_embd.weight", t) && !arch_test::model_dir().empty();
    if (real) {  // rows 0..2 from the real table
      const int toks[3] = {0, 3000, 129279};
      for (int i = 0; i < 3; i++) real &= arch_test::read_tensor_bytes(t, (uint64_t)toks[i] * 4200, 4200, &raw[i * stride]);
    }
    Buffer bs = up(raw.data(), raw.size()), bh = d.alloc(n * 20480 * 4), b1 = d.alloc(n * 5120 * 4);
    {
      Enc e;
      arch::encode_embed_q6k(e, bs, stride, bh, n, 4);
      arch::encode_embed_q6k(e, bs, stride, b1, n, 1);
      e.wait();
    }
    std::vector<float> ref(n * 20480), ref1(n * 5120);
    for (int i = 0; i < n; i++) {
      ref_arch::q6k_row(&raw[i * stride], &ref1[i * 5120], 5120);
      for (int j = 0; j < 4; j++) memcpy(&ref[i * 20480 + j * 5120], &ref1[i * 5120], 5120 * 4);
    }
    printf("embed_q6k (%s rows 0..2 + synthetic)\n", real ? "real token_embd" : "synthetic");
    ok &= bitexact("h (4 copies)", (float*)bh.cpu(), ref.data(), ref.size());
    ok &= bitexact("x (1 copy)", (float*)b1.cpu(), ref1.data(), ref1.size());
  }
  printf(ok ? "test_arch_engram: ALL OK\n" : "test_arch_engram: FAILED\n");
  return ok ? 0 : 1;
}
