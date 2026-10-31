// Sampling + logits helpers: argmax (greedy, ties, multi-row), Gumbel sampling distribution, top-k + lse,
// log_softmax, rmsnorm variants, CPU top-p sampler.
#include <algorithm>
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

int main() {
  arch_test::Rng r(99);
  auto& d = Device::get();
  auto s = arch::Scratch::create(64, 16);
  bool ok = true;
  const int V = arch::kVocab;

  // ---- argmax, several rows, incl. ties and the max at the edges ----
  {
    const int rows = 6;
    std::vector<float> l(rows * V);
    r.fill_normal(l.data(), l.size(), 4.0f);
    l[0 * V + 0] = 100.f;                            // first element
    l[1 * V + V - 1] = 100.f;                        // last element
    l[2 * V + 5000] = 50.f; l[2 * V + 90000] = 50.f; // tie -> lowest index
    l[3 * V + 2019] = 60.f; l[3 * V + 2020] = 60.f;  // tie across threadgroup chunk boundary
    Buffer bl = up(l.data(), l.size() * 4), bi = d.alloc(rows * 4), bv = d.alloc(rows * 4);
    bool o = true;
    for (int rep = 0; rep < 3; rep++) {  // repeated dispatches (counter reset)
      { Enc e; arch::encode_argmax(e, bl, rows, V, bi, bv, s); e.wait(); }
      for (int i = 0; i < rows; i++) {
        int ref = ref_arch::argmax(&l[i * V], V);
        uint32_t got = ((uint32_t*)bi.cpu())[i];
        if ((int)got != ref || ((float*)bv.cpu())[i] != l[i * V + ref]) {
          printf("  argmax row %d: got %u ref %d\n", i, got, ref);
          o = false;
        }
      }
    }
    // stress: 300 single-dispatch argmaxes in one command buffer (counter reuse), all must agree
    const int R = 300;
    Buffer si = d.alloc(R * rows * 4), sv = d.alloc(R * rows * 4);
    { Enc e; for (int i = 0; i < R; i++) { arch::encode_argmax(e, bl, rows, V, si.sub(i * rows * 4, rows * 4), sv.sub(i * rows * 4, rows * 4), s); e.barrier(); } e.wait(); }
    for (int i = 0; i < R; i++)
      for (int j = 0; j < rows; j++) o &= ((uint32_t*)si.cpu())[i * rows + j] == (uint32_t)ref_arch::argmax(&l[j * V], V);
    printf("argmax rows=%d x3 + %d-dispatch stress: %s\n", rows, R, o ? "ok" : "FAIL");
    ok &= o;
  }

  // ---- Gumbel-max sampling: empirical distribution over a small vocab vs softmax(l/T) ----
  {
    const int Vs = 7, N = 20000;
    float T = 0.8f;
    std::vector<float> l = {1.0f, 2.0f, 0.5f, 2.5f, -1.0f, 1.5f, 0.0f};
    Buffer bl = up(l.data(), Vs * 4), bi = d.alloc(16 * 4), bv = d.alloc(16 * 4);
    std::vector<int> cnt(Vs, 0);
    const int per = 16;  // rows per dispatch (same logits, different rows -> different noise)
    std::vector<float> l16(per * Vs);
    for (int i = 0; i < per; i++) memcpy(&l16[i * Vs], l.data(), Vs * 4);
    Buffer bl16 = up(l16.data(), l16.size() * 4);
    for (int it = 0; it < N / per; it++) {
      { Enc e; arch::encode_argmax(e, bl16, per, Vs, bi, bv, s, T, 0x1234567ull + it * 7919ull); e.wait(); }
      for (int i = 0; i < per; i++) cnt[((uint32_t*)bi.cpu())[i]]++;
    }
    double lse = ref_arch::logsumexp(l.data(), Vs, 1.0 / T), chi2 = 0;
    printf("gumbel sampling T=%.1f, %d draws: p_emp vs p_ref:", T, N);
    for (int i = 0; i < Vs; i++) {
      double p = std::exp(l[i] / T - lse), e = p * N;
      chi2 += (cnt[i] - e) * (cnt[i] - e) / e;
      printf(" %.3f/%.3f", cnt[i] / (double)N, p);
    }
    bool o = chi2 < 22.5;  // chi2(6 dof) p=0.001
    printf("\n  chi2 = %.2f (6 dof, limit 22.5): %s\n", chi2, o ? "ok" : "FAIL");
    ok &= o;
  }

  // ---- top-k + lse ----
  {
    const int rows = 3, k = 64;
    std::vector<float> l(rows * V);
    r.fill_normal(l.data(), l.size(), 3.0f);
    for (int i = 0; i < 10; i++) l[1 * V + 4040 * i + 7] = 20.0f;  // ties across chunks
    Buffer bl = up(l.data(), l.size() * 4), bv = d.alloc(rows * 64 * 4), bi = d.alloc(rows * 64 * 4), bs = d.alloc(rows * 4);
    for (float T : {1.0f, 0.7f}) {
      { Enc e; arch::encode_topk(e, bl, rows, V, k, T, bv, bi, bs, s); e.wait(); }
      bool o = true;
      for (int row = 0; row < rows; row++) {
        std::vector<float> rv(k);
        std::vector<uint32_t> ri(k);
        ref_arch::topk(&l[row * V], V, k, rv.data(), ri.data());
        if (memcmp(ri.data(), (uint32_t*)bi.cpu() + row * 64, k * 4) || memcmp(rv.data(), (float*)bv.cpu() + row * 64, k * 4)) {
          printf("  topk row %d mismatch: first got %u ref %u\n", row, ((uint32_t*)bi.cpu())[row * 64], ri[0]);
          o = false;
        }
        float rl = (float)ref_arch::logsumexp(&l[row * V], V, 1.0 / T);
        char nm[48];
        snprintf(nm, sizeof nm, "lse row %d T=%.1f", row, T);
        o &= close(nm, (float*)bs.cpu() + row, &rl, 1, 1e-4f, 1e-5f);
      }
      printf("topk k=%d T=%.1f rows=%d: %s\n", k, T, rows, o ? "ok" : "FAIL");
      ok &= o;
    }
    // CPU top-p sampler on the result: frequencies vs renormalized nucleus probabilities
    const float* v0 = (float*)bv.cpu();
    const uint32_t* i0 = (uint32_t*)bi.cpu();
    float lse0 = ((float*)bs.cpu())[0];  // T = 0.7 from the last run
    std::vector<double> p(k);
    double cum = 0;
    int keep = k;
    for (int i = 0; i < k; i++) { p[i] = std::exp(v0[i] / 0.7 - lse0); cum += p[i]; if (cum >= 0.3) { keep = i + 1; break; } }
    std::vector<int> cnt(k, 0);
    const int N = 100000;
    bool in_nucleus = true;
    for (int it = 0; it < N; it++) {
      uint32_t tok = arch::sample_top_p(v0, i0, k, lse0, 0.7f, 0.3f, (it + 0.5) / N);
      int pos = int(std::find(i0, i0 + k, tok) - i0);
      if (pos >= keep) in_nucleus = false;
      else cnt[pos]++;
    }
    double tot = 0;
    for (int i = 0; i < keep; i++) tot += p[i];
    bool o = in_nucleus;
    for (int i = 0; i < keep; i++) o &= std::fabs(cnt[i] / (double)N - p[i] / tot) < 1e-3;
    printf("sample_top_p (T=0.7, p=0.3, nucleus %d tokens): %s\n", keep, o ? "ok" : "FAIL");
    ok &= o;
  }

  // ---- log_softmax ----
  {
    const int rows = 2;
    std::vector<float> l(rows * V), ref(rows * V);
    r.fill_normal(l.data(), l.size(), 5.0f);
    for (int i = 0; i < rows; i++) ref_arch::log_softmax(&l[i * V], &ref[i * V], V);
    Buffer bl = up(l.data(), l.size() * 4), bo = d.alloc(l.size() * 4);
    { Enc e; arch::encode_log_softmax(e, bl, bo, rows, V); e.barrier(); arch::encode_log_softmax(e, bl, bl, rows, V); e.wait(); }
    printf("log_softmax rows=%d\n", rows);
    ok &= close("out", (float*)bo.cpu(), ref.data(), ref.size(), 1e-4f, 1e-5f);
    ok &= close("in place", (float*)bl.cpu(), ref.data(), ref.size(), 1e-4f, 1e-5f);
  }

  // ---- rmsnorm variants ----
  for (int dim : {128, 512, 1280, 5120, 20480}) {
    const int rows = 37;
    const int in_stride = dim + 64;
    std::vector<float> x(rows * in_stride), w(dim), ref(rows * dim);
    r.fill_normal(x.data(), x.size(), 2.0f);
    for (auto& v : w) v = r.uniform(0.5f, 1.5f);
    for (int i = 0; i < rows; i++) ref_arch::rmsnorm(&x[i * in_stride], w.data(), &ref[i * dim], dim, 1e-6);
    Buffer bx = up(x.data(), x.size() * 4), bw = up(w.data(), dim * 4), by = d.alloc(rows * dim * 4);
    { Enc e; arch::encode_rmsnorm(e, bx, bw, by, rows, dim, 1e-6f, in_stride, dim); e.wait(); }
    char nm[48];
    snprintf(nm, sizeof nm, "rmsnorm dim=%d rows=%d (strided)", dim, rows);
    ok &= close(nm, (float*)by.cpu(), ref.data(), ref.size(), 1e-5f, 1e-5f);
  }
  printf(ok ? "test_arch_sample: ALL OK\n" : "test_arch_sample: FAILED\n");
  return ok ? 0 : 1;
}
