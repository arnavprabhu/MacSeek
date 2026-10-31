// encode_moe_prefill vs CPU reference: expert-sorted routing, a popular expert spanning several token tiles, experts
// in one arena at 4-byte-aligned offsets, shared expert, deterministic reduction. Reference emulates the kernel's
// half staging (X, weights and H rounded to half); the error vs the pure F32 reference is reported too.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;
namespace mq = ms::q;

static float r16(float v) { return (float)(_Float16)v; }

struct Mat { QType t; int rows, cols; std::vector<uint8_t> q; };

// out[t][r] = sum_c W[r][c] * in[t][c] over a token subset; hw: round weights to half
static void mm_ref(const Mat& m, const std::vector<std::vector<float>>& in, std::vector<std::vector<double>>& out,
                   std::vector<std::vector<double>>& mag, bool hw) {
  std::vector<float> w(m.cols);
  const size_t rb = mq::row_bytes(m.t, m.cols);
  out.assign(in.size(), std::vector<double>(m.rows, 0)); mag = out;
  for (int r = 0; r < m.rows; r++) {
    ref::dequant_row(m.t, m.q.data() + rb * r, w.data(), m.cols);
    if (hw) for (auto& v : w) v = r16(v);
    for (size_t t = 0; t < in.size(); t++) {
      float s = 0, a = 0;
      const float* x = in[t].data();
      for (int c = 0; c < m.cols; c++) { float p = w[c] * x[c]; s += p; a += std::fabs(p); }
      out[t][r] = s; mag[t][r] = a;
    }
  }
}

struct Case { QType gu, dn, sh; int n, k, n_exp; bool shared; };

int main() {
  const int D = mq::kExpertDim, I = mq::kExpertInter;
  const Case cases[] = {
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 80, 2, 5, true},
      {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q8_0, 70, 2, 4, true},
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q6_K, 40, 3, 4, false},
  };
  auto& dev = ms::Device::get();
  int fails = 0;
  for (const Case& c : cases) {
    std::mt19937 rng(42 + c.n);
    // routing: slot 0 -> expert 0 for every token (multi-tile expert), other slots random distinct experts
    std::vector<std::vector<int>> rt(c.n);
    std::vector<std::vector<float>> rw(c.n);
    for (int t = 0; t < c.n; t++) {
      std::vector<int> perm(c.n_exp - 1);
      for (int j = 0; j < c.n_exp - 1; j++) perm[j] = j + 1;
      std::shuffle(perm.begin(), perm.end(), rng);
      rt[t].push_back(0);
      for (int k = 1; k < c.k; k++) rt[t].push_back(perm[k - 1]);
      for (int k = 0; k < c.k; k++) rw[t].push_back(0.1f + 0.2f * (float)((t * 7 + k * 3) % 5) / 5);
    }
    std::vector<int> offs(c.n_exp + 1, 0), tok, slot_of(c.n * c.k, -1);
    std::vector<float> wt;
    for (int j = 0; j < c.n_exp; j++) {
      offs[j] = (int)tok.size();
      for (int t = 0; t < c.n; t++)
        for (int k = 0; k < c.k; k++)
          if (rt[t][k] == j) { slot_of[t * c.k + k] = (int)tok.size(); tok.push_back(t); wt.push_back(rw[t][k]); }
    }
    offs[c.n_exp] = (int)tok.size();
    const int total = (int)tok.size();
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
    { ms::Enc e; mq::encode_moe_prefill(e, a); e.wait(); }

    // reference (two flavours)
    double err_h = 0, err_f = 0;
    std::vector<std::vector<double>> yh(c.n, std::vector<double>(D)), yf = yh, ym = yh;
    for (int t = 0; t < c.n; t++) for (int i = 0; i < D; i++) { yh[t][i] = yf[t][i] = Y0[(size_t)t * D + i]; ym[t][i] = std::fabs(Y0[(size_t)t * D + i]); }
    for (int j = 0; j < ne_all; j++) {
      std::vector<int> toks; std::vector<float> ws;
      if (j == c.n_exp) for (int t = 0; t < c.n; t++) { toks.push_back(t); ws.push_back(1.f); }
      else for (int i = offs[j]; i < offs[j + 1]; i++) { toks.push_back(tok[i]); ws.push_back(wt[i]); }
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
        mm_ref(Dn[j], h, o, om, hv);
        for (size_t q = 0; q < toks.size(); q++)
          for (int i = 0; i < D; i++) {
            (hv ? yh : yf)[toks[q]][i] += o[q][i];
            if (hv) ym[toks[q]][i] += om[q][i];
          }
      }
    }
    const float* got = (const float*)by.cpu();
    for (int t = 0; t < c.n; t++)
      for (int i = 0; i < D; i++) {
        double gv = got[(size_t)t * D + i];
        err_h = std::max(err_h, std::fabs(gv - yh[t][i]) / (ym[t][i] + 1e-9));
        err_f = std::max(err_f, std::fabs(gv - yf[t][i]) / (ym[t][i] + 1e-9));
      }
    bool ok = err_h <= 1e-3 && err_f <= 5e-3;
    printf("  gu=%-7s down=%-7s shared=%-4s n=%d k=%d experts=%d entries=%d  err(half-emul) %.2e  err(vs F32) %.2e  %s\n",
           mq::type_name(c.gu), mq::type_name(c.dn), c.shared ? mq::type_name(c.sh) : "-", c.n, c.k, c.n_exp, total,
           err_h, err_f, ok ? "ok" : "FAIL");
    fails += !ok;
    for (Buffer* b : {&arena, &bx, &by, &btok, &bwt, &bslot, &scratch}) dev.release(*b);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
