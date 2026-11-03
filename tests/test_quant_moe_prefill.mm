// encode_moe_prefill vs CPU reference: expert-sorted routing, experts in one arena at 4-byte-aligned offsets, shared
// expert, deterministic reduction. Routings: a popular expert spanning several token tiles; explicit expert sizes that
// hit every token-tile class boundary (1, 15..17, 31..33, 47..49, 63..65 tokens, ... and one expert with > 512 tokens);
// a skewed Zipf(0.8) top-6 routing. Both kernel paths (register dequant / threadgroup tiles) and both shared-expert
// paths (dense pre-dequant / grouped). Reference emulates the kernels' half staging (X, weights and H rounded to half);
// the error vs the pure F32 reference is reported too. Large cases check a token subset (the reference is O(n k D I)).
// Every case also runs twice and requires bit-identical outputs. The scratch is poisoned with NaN before every run and
// every H element of the checked entries is compared with the half-emulating reference (unwritten / misplaced H rows
// fail); the kCounts cases (192 experts, top-6, n 144..200) also require the merged gate|up kernels to match the
// two-matmul ones element by element.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <cstring>
#include <thread>
#include <vector>
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;
namespace mq = ms::q;

static float r16(float v) { return (float)(_Float16)v; }

struct Mat { QType t; int rows, cols; std::vector<uint8_t> q; };

// out[t][r] = sum_c W[r][c] * in[t][c] over a token subset; hw: round weights to half. Rows split over threads.
static void mm_ref(const Mat& m, const std::vector<std::vector<float>>& in, std::vector<std::vector<double>>& out,
                   std::vector<std::vector<double>>& mag, bool hw) {
  const size_t rb = mq::row_bytes(m.t, m.cols);
  out.assign(in.size(), std::vector<double>(m.rows, 0)); mag = out;
  if (in.empty()) return;
  const int nth = 10;
  std::vector<std::thread> th;
  for (int w = 0; w < nth; w++)
    th.emplace_back([&, w] {
      std::vector<float> wr(m.cols);
      for (int r = w; r < m.rows; r += nth) {
        ref::dequant_row(m.t, m.q.data() + rb * r, wr.data(), m.cols);
        if (hw) for (auto& v : wr) v = r16(v);
        for (size_t t = 0; t < in.size(); t++) {
          float s = 0, a = 0;
          const float* x = in[t].data();
          for (int c = 0; c < m.cols; c++) { float p = wr[c] * x[c]; s += p; a += std::fabs(p); }
          out[t][r] = s; mag[t][r] = a;
        }
      }
    });
  for (auto& t : th) t.join();
}

enum Routing { kPopular, kSizes, kZipf, kCounts };
struct Case { QType gu, dn, sh; int n, k, n_exp; bool shared; Routing rt; int path, sh_dense, n_check; };

// H offset inside the scratch (mirrors moe_header_bytes / the scratch layout in quant.mm: header | Xg | H | Z | W16)
static size_t a256(size_t v) { return (v + 255) & ~(size_t)255; }
static size_t h_offset(int total, int n, int ne) {
  const size_t tiles = (size_t)ne + total / 16 + n / 16 + 4;
  return a256(24 * (size_t)(ne + 1) + 4 * (size_t)(ne + 2) + 8 * tiles) + a256(((size_t)total + n) * mq::kExpertDim * 2);
}

int main() {
  const int D = mq::kExpertDim, I = mq::kExpertInter;
  const Case cases[] = {
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 80, 2, 5, true, kPopular, -1, -1, 0},
      {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q8_0, 70, 2, 4, true, kPopular, -1, -1, 0},
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q6_K, 40, 3, 4, false, kPopular, -1, -1, 0},
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 600, 6, 20, true, kSizes, -1, -1, 40},
      {QType::IQ2_XXS, QType::IQ1_S, QType::Q8_0, 600, 6, 20, true, kSizes, -1, 0, 40},
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q6_K, 256, 6, 64, true, kZipf, -1, -1, 40},
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 80, 2, 5, true, kPopular, 0, 0, 0},  // previous kernels
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q8_0, 256, 6, 64, true, kZipf, 0, 1, 40},
      // GSQ-RCO model types: gate/up and down types differ, Q5_K shared expert (dense and grouped), both paths
      {QType::IQ1_M, QType::IQ2_XXS, QType::Q5_K, 600, 6, 20, true, kSizes, -1, 1, 40},
      {QType::IQ2_XS, QType::IQ1_M, QType::Q5_K, 256, 6, 64, true, kZipf, -1, 0, 40},
      {QType::IQ2_XXS, QType::IQ2_XS, QType::Q5_K, 80, 2, 5, true, kPopular, 1, 0, 0},
      {QType::IQ1_M, QType::IQ1_M, QType::Q5_K, 80, 2, 5, true, kPopular, 1, 1, 0},
      {QType::IQ2_XS, QType::IQ2_XS, QType::Q5_K, 70, 2, 4, true, kPopular, 0, 0, 0},  // previous kernels
      {QType::IQ1_M, QType::IQ2_XS, QType::Q5_K, 256, 6, 64, true, kZipf, 0, 1, 40},
      {QType::IQ2_XS, QType::IQ2_XXS, QType::Q6_K, 40, 3, 4, false, kPopular, -1, -1, 0},
      // short prompts on the full 192-expert top-6 router (kCounts): few tokens per expert, so most experts are partial
      // 16/32-token tiles and a handful land in every other class incl. partial 48 / 64 tiles and 64 + remainder
      // (17..48, 49..64, 65..112, 113..128 tokens). Every token, every H element of every entry checked.
      {QType::IQ1_M, QType::IQ2_XXS, QType::Q5_K, 144, 6, 192, true, kCounts, -1, -1, 0},
      {QType::IQ2_XS, QType::IQ2_XS, QType::Q5_K, 144, 6, 192, true, kCounts, -1, -1, 0},
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q6_K, 160, 6, 192, true, kCounts, -1, -1, 0},
      {QType::IQ2_XXS, QType::IQ1_M, QType::Q5_K, 200, 6, 192, true, kCounts, -1, -1, 0},
  };
  // explicit expert sizes (kSizes): expert 0 takes every token (> 512), the rest straddle the tile classes
  const int sizes[] = {1, 1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 129, 200, 300};
  auto& dev = ms::Device::get();
  int fails = 0;
  const char* only = getenv("MOEP_TEST_CASES");  // optional comma-separated case indices (debugging)
  for (const Case& c : cases) {
    const int ci_ = (int)(&c - cases);
    if (only && !strstr((std::string(",") + only + ",").c_str(), (std::string(",") + std::to_string(ci_) + ",").c_str())) continue;
    std::mt19937 rng(42 + c.n + c.rt);
    std::vector<std::vector<int>> rt(c.n);  // routed experts per token (slot order)
    if (c.rt == kPopular) {  // slot 0 -> expert 0 for every token (multi-tile expert), other slots random distinct
      for (int t = 0; t < c.n; t++) {
        std::vector<int> perm(c.n_exp - 1);
        for (int j = 0; j < c.n_exp - 1; j++) perm[j] = j + 1;
        std::shuffle(perm.begin(), perm.end(), rng);
        rt[t].push_back(0);
        for (int k = 1; k < c.k; k++) rt[t].push_back(perm[k - 1]);
      }
    } else if (c.rt == kSizes) {
      for (int t = 0; t < c.n; t++) rt[t].push_back(0);
      int next = 0;
      for (int j = 1; j < c.n_exp; j++)  // spread each expert's tokens over the token range
        for (int q = 0; q < sizes[j - 1]; q++) {
          int t = next;
          while ((int)rt[t].size() >= c.k) t = (t + 1) % c.n;
          rt[t].push_back(j); next = (t + 7) % c.n;
        }
    } else if (c.rt == kCounts) {  // explicit per-expert token counts summing to n*k, each token gets k distinct experts
      std::vector<int> cnt(c.n_exp, 0);
      const int big[] = {120, 100, 70, 60, 50, 40, 33, 20, 17};  // 113..128, 65..112 (2), 49..64 (2), 17..48 (4)
      int left = c.n * c.k, j = 0;
      for (int b : big) { cnt[j++] = b; left -= b; }
      for (int q = 0; left > 0; q++, j = j + 1 < c.n_exp ? j + 1 : 9) {  // the rest 1..16 tokens (partial 16-tiles)
        const int v = std::min(left, q % 4 == 3 ? 9 + (q * 3) % 8 : 1 + (q * 5) % 8);
        if (cnt[j] + v > c.n) continue;
        cnt[j] += v; left -= v;
      }
      std::vector<int> order(c.n_exp);
      for (int q = 0; q < c.n_exp; q++) order[q] = q;
      std::shuffle(order.begin(), order.end(), rng);  // popular experts at random ids
      std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return cnt[x] > cnt[y]; });
      for (int e : order) {  // biggest first, onto the least-filled tokens (ties: random) that lack e
        std::vector<int> ts(c.n);
        for (int t = 0; t < c.n; t++) ts[t] = t;
        std::shuffle(ts.begin(), ts.end(), rng);
        std::stable_sort(ts.begin(), ts.end(), [&](int x, int y) { return rt[x].size() < rt[y].size(); });
        int got = 0;
        for (int t : ts) {
          if (got == cnt[e]) break;
          if ((int)rt[t].size() >= c.k || std::find(rt[t].begin(), rt[t].end(), e) != rt[t].end()) continue;
          rt[t].push_back(e); got++;
        }
      }
    } else {  // Zipf(0.8) popularity over a random expert permutation, k distinct experts per token
      std::vector<int> perm(c.n_exp);
      for (int j = 0; j < c.n_exp; j++) perm[j] = j;
      std::shuffle(perm.begin(), perm.end(), rng);
      std::vector<double> pw(c.n_exp);
      for (int r = 0; r < c.n_exp; r++) pw[perm[r]] = 1.0 / std::pow(r + 1.0, 0.8);
      std::uniform_real_distribution<double> ud(0, 1);
      for (int t = 0; t < c.n; t++) {
        std::vector<double> w = pw;
        for (int k = 0; k < c.k; k++) {
          double sum = 0; for (double v : w) sum += v;
          double u = ud(rng) * sum; int j = 0;
          for (; j < c.n_exp - 1; j++) { if (u < w[j]) break; u -= w[j]; }
          while (w[j] == 0) j = (j + 1) % c.n_exp;
          rt[t].push_back(j); w[j] = 0;
        }
      }
    }
    std::vector<std::vector<float>> rw(c.n);
    for (int t = 0; t < c.n; t++)
      for (int k = 0; k < (int)rt[t].size(); k++) rw[t].push_back(0.1f + 0.2f * (float)((t * 7 + k * 3) % 5) / 5);
    std::vector<int> offs(c.n_exp + 1, 0), tok, slot_of(c.n * c.k, -1);
    std::vector<float> wt;
    for (int j = 0; j < c.n_exp; j++) {
      offs[j] = (int)tok.size();
      for (int t = 0; t < c.n; t++)
        for (int k = 0; k < (int)rt[t].size(); k++)
          if (rt[t][k] == j) { slot_of[t * c.k + k] = (int)tok.size(); tok.push_back(t); wt.push_back(rw[t][k]); }
    }
    offs[c.n_exp] = (int)tok.size();
    const int total = (int)tok.size();
    int mmin = 1 << 30, mmax = 0;
    for (int j = 0; j < c.n_exp; j++) {
      mmin = std::min(mmin, offs[j + 1] - offs[j]); mmax = std::max(mmax, offs[j + 1] - offs[j]);
    }
    // weights in one arena
    const int ne_all = c.n_exp + (c.shared ? 1 : 0);
    std::vector<Mat> G, U, Dn;
    size_t bytes = 64;
    for (int j = 0; j < ne_all; j++) {
      bool sh = j == c.n_exp;
      QType tg = sh ? c.sh : c.gu, td = sh ? c.sh : c.dn;
      G.push_back({tg, I, D, ref::random_blocks(tg, (int64_t)I * D, 500 + 3 * j)});
      U.push_back({tg, I, D, ref::random_blocks(tg, (int64_t)I * D, 501 + 3 * j)});
      Dn.push_back({td, D, I, ref::random_blocks(td, (int64_t)I * D, 502 + 3 * j)});
      bytes += G.back().q.size() + U.back().q.size() + Dn.back().q.size() + 36;
    }
    Buffer arena = dev.alloc(bytes, "arena");
    size_t off = 4;
    auto place = [&](const std::vector<uint8_t>& v) {
      memcpy((uint8_t*)arena.cpu() + off, v.data(), v.size());
      Buffer b = arena.sub(off, v.size()); off += v.size() + 8; return b;
    };
    std::vector<mq::ExpertSlot> slots(c.n_exp);
    mq::ExpertSlot shared;
    for (int j = ne_all - 1; j >= 0; j--) {
      mq::ExpertSlot s; s.up = place(U[j].q); s.down = place(Dn[j].q); s.gate = place(G[j].q);
      if (j == c.n_exp) shared = s; else slots[j] = s;
    }
    auto X = randvec((size_t)c.n * D, 9);
    auto Y0 = randvec((size_t)c.n * D, 10);
    Buffer bx = upload(X.data(), X.size() * 4), by = upload(Y0.data(), Y0.size() * 4);
    Buffer btok = upload(tok.data(), tok.size() * 4), bwt = upload(wt.data(), wt.size() * 4);
    Buffer bslot = upload(slot_of.data(), slot_of.size() * 4);
    Buffer scratch = dev.alloc(mq::moe_prefill_scratch_bytes(total, c.n, c.n_exp));
    mq::MoePrefillArgs a;
    a.t_gate_up = c.gu; a.t_down = c.dn; a.t_shared = c.sh; a.n_tokens = c.n; a.n_exp = c.n_exp;
    a.slots = slots.data(); a.offs = offs.data(); a.tok = btok; a.wt = bwt; a.slot_of = bslot; a.k_per_token = c.k;
    a.has_shared = c.shared; a.shared = shared; a.X = bx; a.Y = by; a.scratch = scratch;
    a.path = c.path; a.shared_dense = c.sh_dense;
    // poison the whole scratch (NaN halfs / floats) so rows the kernels fail to write show up as NaN, not stale data
    memset(scratch.cpu(), 0xFF, scratch.size);
    { ms::Enc e; mq::encode_moe_prefill(e, a); e.wait(); }
    // run-to-run determinism: same inputs again -> bit-identical Y
    std::vector<float> y1((const float*)by.cpu(), (const float*)by.cpu() + (size_t)c.n * D);
    memcpy(by.cpu(), Y0.data(), Y0.size() * 4);
    memset(scratch.cpu(), 0xFF, scratch.size);
    { ms::Enc e; mq::encode_moe_prefill(e, a); e.wait(); }
    const bool same = memcmp(y1.data(), by.cpu(), y1.size() * 4) == 0;

    // tokens to check: all, or a subset (tokens of the smallest experts, tile-boundary tokens, random)
    std::vector<int> chk;
    if (c.n_check <= 0) {
      for (int t = 0; t < c.n; t++) chk.push_back(t);
    } else {
      std::vector<char> on(c.n, 0);
      for (int j = 0; j < c.n_exp; j++)
        if (offs[j + 1] - offs[j] <= 2) for (int i = offs[j]; i < offs[j + 1]; i++) on[tok[i]] = 1;
      for (int t : {0, 15, 16, 31, 32, 47, 48, 63, 64, 65, 127, 128, 511, 512, 513, c.n - 1}) if (t < c.n) on[t] = 1;
      std::uniform_int_distribution<int> ut(0, c.n - 1);
      for (int q = 0; q < c.n_check; q++) on[ut(rng)] = 1;
      for (int t = 0; t < c.n; t++) if (on[t]) chk.push_back(t);
    }
    std::vector<int> ci(c.n, -1);
    for (size_t q = 0; q < chk.size(); q++) ci[chk[q]] = (int)q;
    // reference (two flavours) for the checked tokens
    double err_h = 0, err_f = 0, err_hh = 0;  // err_hh: H elements of the checked entries vs the half-emulating ref
    long h_bad = 0, h_cnt = 0;
    const _Float16* Hg = (const _Float16*)((const uint8_t*)scratch.cpu() + h_offset(total, c.n, c.n_exp));
    const size_t nc = chk.size();
    std::vector<std::vector<double>> yh(nc, std::vector<double>(D)), yf = yh, ym = yh;
    for (size_t q = 0; q < nc; q++)
      for (int i = 0; i < D; i++) {
        const float v = Y0[(size_t)chk[q] * D + i];
        yh[q][i] = yf[q][i] = v; ym[q][i] = std::fabs(v);
      }
    for (int j = 0; j < ne_all; j++) {
      std::vector<int> toks, rows;  // rows: H row of each entry
      std::vector<float> ws;
      if (j == c.n_exp) for (int t : chk) { toks.push_back(t); ws.push_back(1.f); rows.push_back(total + t); }
      else
        for (int i = offs[j]; i < offs[j + 1]; i++)
          if (ci[tok[i]] >= 0) { toks.push_back(tok[i]); ws.push_back(wt[i]); rows.push_back(i); }
      if (toks.empty()) continue;
      for (int hv = 0; hv < 2; hv++) {
        std::vector<std::vector<float>> xin(toks.size(), std::vector<float>(D));
        for (size_t q = 0; q < toks.size(); q++)
          for (int i = 0; i < D; i++) { float v = X[(size_t)toks[q] * D + i]; xin[q][i] = hv ? r16(v) : v; }
        std::vector<std::vector<double>> g, gm, u, um, o, om;
        mm_ref(G[j], xin, g, gm, hv); mm_ref(U[j], xin, u, um, hv);
        std::vector<std::vector<float>> h(toks.size(), std::vector<float>(I));
        for (size_t q = 0; q < toks.size(); q++)
          for (int r = 0; r < I; r++) {
            double gg = std::min(g[q][r], 10.0), uu = std::max(-10.0, std::min(10.0, u[q][r]));
            float hh = (float)(gg / (1 + std::exp(-gg)) * uu * ws[q]);
            h[q][r] = hv ? r16(hh) : hh;
          }
        if (hv && c.path != 0) {  // every H element: |gpu - ref| vs the row's scale (half rounding ~5e-4 relative)
          for (size_t q = 0; q < toks.size(); q++) {
            double sc = 1e-6;
            for (int r = 0; r < I; r++) sc = std::max(sc, (double)std::fabs(h[q][r]));
            for (int r = 0; r < I; r++) {
              const double gv = (double)Hg[(size_t)rows[q] * I + r], e = std::fabs(gv - h[q][r]);
              const double rel = std::isfinite(gv) ? e / (std::fabs(h[q][r]) + 1e-2 * sc) : INFINITY;
              err_hh = std::max(err_hh, rel);
              h_cnt++;
              if (rel > 2e-2) {
                if (h_bad++ < 4)
                  printf("    H mismatch: expert %d token %d (entry row %d, m %d) r %d: gpu %g ref %g\n", j, toks[q],
                         rows[q], j < c.n_exp ? offs[j + 1] - offs[j] : c.n, r, gv, h[q][r]);
              }
            }
          }
        }
        mm_ref(Dn[j], h, o, om, hv);
        for (size_t q = 0; q < toks.size(); q++)
          for (int i = 0; i < D; i++) {
            (hv ? yh : yf)[ci[toks[q]]][i] += o[q][i];
            if (hv) ym[ci[toks[q]]][i] += om[q][i];
          }
      }
    }
    const float* got = (const float*)by.cpu();
    for (size_t q = 0; q < nc; q++)
      for (int i = 0; i < D; i++) {
        double gv = got[(size_t)chk[q] * D + i];
        err_h = std::max(err_h, std::fabs(gv - yh[q][i]) / (ym[q][i] + 1e-9));
        err_f = std::max(err_f, std::fabs(gv - yf[q][i]) / (ym[q][i] + 1e-9));
        if (!std::isfinite(gv)) err_h = err_f = INFINITY;
      }
    // kCounts: the merged gate|up kernels (q_moe_rgu2, default) vs the two-matmul q_moe_rgu for every tile class on
    // the same inputs: every H element must agree (same products; only accumulation order / half rounding differ)
    double err_ms = 0;
    long ms_bad = 0;
    if (c.rt == kCounts && c.path != 0) {
      std::vector<_Float16> h_m(Hg, Hg + (size_t)(total + c.n) * I);
      const bool iq1 = c.gu == QType::IQ1_S || c.gu == QType::IQ1_M;
      setenv("MACSEEK_MOEP_RGU64", iq1 ? "64x16k64s4" : "64x16k32s4", 1);
      setenv("MACSEEK_MOEP_RGU48", "48x16k32s4", 1);
      setenv("MACSEEK_MOEP_RGU", "32x32k32s4", 1);
      setenv("MACSEEK_MOEP_RGU16", "16x32k32s4", 1);
      memcpy(by.cpu(), Y0.data(), Y0.size() * 4);
      memset(scratch.cpu(), 0xFF, scratch.size);
      { ms::Enc e; mq::encode_moe_prefill(e, a); e.wait(); }
      for (const char* v : {"MACSEEK_MOEP_RGU64", "MACSEEK_MOEP_RGU48", "MACSEEK_MOEP_RGU", "MACSEEK_MOEP_RGU16"}) unsetenv(v);
      for (size_t i = 0; i < (size_t)total * I; i++) {
        const double x0 = (double)h_m[i], x1 = (double)Hg[i];
        const double d = std::fabs(x0 - x1) / (std::fabs(x1) + 1e-2);
        if (!(d <= 2e-2)) ms_bad++;
        else err_ms = std::max(err_ms, d);
      }
      printf("    merged vs two-matmul gate/up: H max diff %.2e, %ld bad\n", err_ms, ms_bad);
    }
    bool ok = err_h <= 1e-3 && err_f <= 5e-3 && same && h_bad == 0 && ms_bad == 0;
    static const char* rname[] = {"popular", "sizes", "zipf", "counts"};
    printf("  gu=%-7s down=%-7s shared=%-4s path=%2d shdense=%2d %-7s n=%d k=%d experts=%d entries=%d (m %d..%d) "
           "checked=%zu  err(half-emul) %.2e  err(vs F32) %.2e  H %.2e (%ld/%ld bad)  %s  %s\n",
           mq::type_name(c.gu), mq::type_name(c.dn), c.shared ? mq::type_name(c.sh) : "-", c.path, c.sh_dense,
           rname[c.rt], c.n, c.k, c.n_exp, total, mmin, mmax, nc, err_h, err_f, err_hh, h_bad, h_cnt,
           same ? "repeat bit-identical" : "REPEAT DIFFERS", ok ? "ok" : "FAIL");
    fails += !ok;
    for (Buffer* b : {&arena, &bx, &by, &btok, &bwt, &bslot, &scratch}) dev.release(*b);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
