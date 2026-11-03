// encode_moe_decode vs CPU reference: 6 routed experts (+ shared) in one arena buffer at unaligned (4-byte) offsets,
// all type combinations used by the models (UD-IQ1_S: IQ1_S/IQ2_XXS + Q6_K/Q8_0; GSQ-RCO: every gate/up x down
// combination of IQ1_M / IQ2_XXS / IQ2_XS + Q5_K), clamp paths exercised with a large-x case. Also the pointer-table
// variants: encode_moe_decode_table (All) and the encode_moe_decode_table_part pieces (SharedGu + AfterSharedGu and
// SharedGu + SharedDown must be bit-identical to All / Shared; Shared + Routed matches the reference).
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "quant_test_util.hpp"

using namespace qt;
using ms::q::QType;
namespace mq = ms::q;

struct Case { QType gu, down, sh; int n_routed; bool shared; float xscale; };

static double silu_clamped(double g, double u, double w) {
  g = std::min(g, 10.0);
  u = std::max(-10.0, std::min(10.0, u));
  return g / (1.0 + std::exp(-g)) * u * w;
}

int main() {
  const int D = mq::kExpertDim, I = mq::kExpertInter;
  const QType R3[] = {QType::IQ1_M, QType::IQ2_XXS, QType::IQ2_XS};
  std::vector<Case> cases = {
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 6, true, 1.f},
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q8_0, 6, true, 1.f},
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q6_K, 6, true, 1.f},
      {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q6_K, 6, true, 1.f},
      {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q8_0, 3, false, 1.f},
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 0, true, 1.f},
      {QType::IQ1_S, QType::IQ1_S, QType::Q6_K, 6, true, 40.f},  // drives gate/up past the +-10 clamps
      {QType::IQ1_S, QType::IQ2_XXS, QType::Q5_K, 6, true, 1.f},
      {QType::IQ1_M, QType::IQ2_XS, QType::Q6_K, 6, true, 1.f},
      {QType::IQ2_XS, QType::IQ1_M, QType::Q8_0, 3, true, 1.f},
      {QType::IQ1_M, QType::IQ2_XXS, QType::Q5_K, 0, true, 1.f},
      {QType::IQ2_XS, QType::IQ2_XS, QType::Q5_K, 3, false, 1.f},
      {QType::IQ1_M, QType::IQ1_M, QType::Q5_K, 6, true, 40.f},
  };
  for (QType gu : R3) for (QType dn : R3) cases.push_back({gu, dn, QType::Q5_K, 6, true, 1.f});
  int fails = 0;
  auto& dev = ms::Device::get();
  for (const Case& c : cases) {
    const int n_exp = c.n_routed + (c.shared ? 1 : 0);
    std::vector<std::vector<uint8_t>> G(n_exp), U(n_exp), Dn(n_exp);
    std::vector<QType> tgu(n_exp), td(n_exp);
    size_t total = 0;
    for (int e = 0; e < n_exp; e++) {
      bool sh = e == c.n_routed;
      tgu[e] = sh ? c.sh : c.gu; td[e] = sh ? c.sh : c.down;
      G[e] = ref::random_blocks(tgu[e], (int64_t)I * D, 100 + e * 3);
      U[e] = ref::random_blocks(tgu[e], (int64_t)I * D, 101 + e * 3);
      Dn[e] = ref::random_blocks(td[e], (int64_t)I * D, 102 + e * 3);
      total += G[e].size() + U[e].size() + Dn[e].size() + 3 * 20;
    }
    // arena with 4-byte-aligned (not 16) offsets, in a shuffled order
    Buffer arena = dev.alloc(total + 64, "arena");
    mq::MoeDecodeArgs a;
    a.t_gate_up = c.gu; a.t_down = c.down; a.t_shared = c.sh; a.n_routed = c.n_routed; a.has_shared = c.shared;
    size_t off = 4;
    auto place = [&](const std::vector<uint8_t>& v) {
      memcpy((uint8_t*)arena.cpu() + off, v.data(), v.size());
      Buffer b = arena.sub(off, v.size());
      off += v.size() + 12;
      return b;
    };
    for (int e = n_exp - 1; e >= 0; e--) {
      mq::ExpertSlot s;
      s.down = place(Dn[e]); s.gate = place(G[e]); s.up = place(U[e]);
      if (e == c.n_routed) a.shared = s; else a.slot[e] = s;
    }
    auto x = randvec(D, 77, c.xscale);
    auto y0 = randvec(D, 78);
    std::vector<float> w = {0.31f, 0.22f, 0.17f, 0.12f, 0.1f, 0.08f};
    Buffer bx = upload(x.data(), D * 4), by = upload(y0.data(), D * 4), bw = upload(w.data(), w.size() * 4);
    Buffer scratch = dev.alloc(mq::moe_decode_scratch_bytes());
    a.weights = bw; a.x = bx; a.y = by; a.scratch = scratch;
    { ms::Enc enc; mq::encode_moe_decode(enc, a); enc.wait(); }

    // reference
    std::vector<double> yref(y0.begin(), y0.end()), ymag(D);
    for (int i = 0; i < D; i++) ymag[i] = std::fabs(y0[i]);
    int clamped = 0;
    double h_err = 0;
    const float* hgpu = (const float*)scratch.cpu();
    for (int e = 0; e < n_exp; e++) {
      std::vector<double> g, gm, u, um;
      ref_matvec(tgu[e], G[e].data(), I, D, x.data(), g, gm);
      ref_matvec(tgu[e], U[e].data(), I, D, x.data(), u, um);
      const double we = e == c.n_routed ? 1.0 : w[e];
      std::vector<float> h(I);
      for (int r = 0; r < I; r++) {
        h[r] = (float)silu_clamped(g[r], u[r], we);
        if (g[r] > 10 || std::fabs(u[r]) > 10) clamped++;
        double sc = std::fabs(we) * (gm[r] + um[r]) * 12 + 1e-6;  // |dh| <~ (|dg|+|du|) * max slope
        h_err = std::max(h_err, std::fabs(hgpu[e * I + r] - h[r]) / sc);
      }
      std::vector<double> yd, ydm;
      ref_matvec(td[e], Dn[e].data(), D, I, h.data(), yd, ydm);
      for (int i = 0; i < D; i++) { yref[i] += yd[i]; ymag[i] += ydm[i]; }
    }
    double err = max_norm_err((float*)by.cpu(), yref, ymag, D);
    const std::vector<float> y_bound((const float*)by.cpu(), (const float*)by.cpu() + D);

    // pointer-table variants: table[e] = {gate, up, down} device addresses
    Buffer tbl = dev.alloc(sizeof(uint64_t) * 3 * mq::kMaxRouted, "tbl");
    auto gaddr = [](const Buffer& b) { return (uint64_t)[(__bridge id<MTLBuffer>)b.buf gpuAddress] + b.off; };
    for (int e = 0; e < c.n_routed; e++) {
      uint64_t* t = (uint64_t*)tbl.cpu() + 3 * e;
      t[0] = gaddr(a.slot[e].gate); t[1] = gaddr(a.slot[e].up); t[2] = gaddr(a.slot[e].down);
    }
    using P = mq::MoePart;
    auto run_parts = [&](std::initializer_list<P> parts) {
      memcpy(by.cpu(), y0.data(), D * 4);
      ms::Enc enc;
      [(__bridge id<MTLComputeCommandEncoder>)enc.encoder() useResource:(__bridge id<MTLBuffer>)arena.buf
                                                                  usage:MTLResourceUsageRead];
      bool first = true;
      for (P p : parts) { if (!first) enc.barrier(); mq::encode_moe_decode_table_part(enc, a, tbl, p); first = false; }
      enc.wait();
      return std::vector<float>((const float*)by.cpu(), (const float*)by.cpu() + D);
    };
    const auto y_all = run_parts({P::All});
    const auto y_split = run_parts({P::SharedGu, P::AfterSharedGu});
    const auto y_sr = run_parts({P::Shared, P::Routed});
    const auto y_s = run_parts({P::Shared});
    const auto y_sgd = run_parts({P::SharedGu, P::SharedDown});
    const auto y_sgdr = run_parts({P::SharedGu, P::SharedDown, P::Routed});
    const double e_all = max_norm_err(y_all.data(), yref, ymag, D), e_sr = max_norm_err(y_sr.data(), yref, ymag, D),
                 e_sgdr = max_norm_err(y_sgdr.data(), yref, ymag, D);
    const bool all_eq_bound = y_all == y_bound, split_eq = y_split == y_all, sgd_eq = y_sgd == y_s;
    bool ok = err <= 5e-5 && h_err <= 2e-5 && e_all <= 5e-5 && e_sr <= 5e-5 && e_sgdr <= 5e-5 && split_eq && sgd_eq;
    printf("  gu=%-7s down=%-7s shared=%-4s routed=%d  clamped=%5d  h_err %.2e  y_err %.2e | table: All %.2e%s  "
           "SharedGu+AfterSharedGu %s  Shared+Routed %.2e  SharedGu+SharedDown %s  +Routed %.2e  %s\n",
           mq::type_name(c.gu), mq::type_name(c.down), c.shared ? mq::type_name(c.sh) : "-", c.n_routed, clamped,
           h_err, err, e_all, all_eq_bound ? " (== bound)" : "", split_eq ? "bit-identical" : "DIFFERS", e_sr,
           sgd_eq ? "== Shared" : "DIFFERS", e_sgdr, ok ? "ok" : "FAIL");
    fails += !ok;
    dev.release(arena); dev.release(bx); dev.release(by); dev.release(bw); dev.release(scratch); dev.release(tbl);
  }
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
