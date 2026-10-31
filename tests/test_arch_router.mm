// Router: sqrt(softplus) + bias top-6 + normalize*1.5 (decode fused matvec, generic logits path) and prefill lists.
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

static Buffer up(const void* p, size_t n) {
  Buffer b = Device::get().alloc(n);
  memcpy(b.cpu(), p, n);
  return b;
}

// Compare GPU ids/weights with the reference; a different pick is accepted only for a near-tie (key gap < tol).
static bool check_token(const float* logits, const float* bias, const int* gid, const float* gw, int t, int& ties) {
  int rid[6];
  float rw[6];
  ref_arch::router(logits, bias, rid, rw);
  bool same = true;
  for (int k = 0; k < 6; k++) same &= gid[k] == rid[k];
  if (same) {
    for (int k = 0; k < 6; k++)
      if (std::fabs(gw[k] - rw[k]) > 1e-5f + 1e-5f * std::fabs(rw[k])) {
        printf("  token %d weight %d: got %.8g ref %.8g\n", t, k, gw[k], rw[k]);
        return false;
      }
    return true;
  }
  auto key = [&](int e) { return std::sqrt(std::log1p(std::exp((double)logits[e]))) + bias[e]; };
  for (int k = 0; k < 6; k++)
    if (gid[k] != rid[k] && std::fabs(key(gid[k]) - key(rid[k])) > 1e-5) {
      printf("  token %d slot %d: got expert %d (key %.9g) ref %d (key %.9g)\n", t, k, gid[k], key(gid[k]), rid[k],
             key(rid[k]));
      return false;
    }
  ties++;
  return true;
}

int main() {
  arch_test::Rng r(42);
  auto& d = Device::get();
  auto s = arch::Scratch::create(64, 4);
  bool ok = true;

  // 1) generic top-k from logits, n tokens
  for (int n : {1, 5, 2048}) {
    std::vector<float> logits(n * 384), bias(384);
    r.fill_normal(logits.data(), logits.size(), 3.0f);
    for (int i = 0; i < n * 384; i += 97) logits[i] = 30.0f + r.normal();  // softplus threshold branch
    for (int i = 5; i < n * 384; i += 89) logits[i] = -25.0f;              // tiny-softplus branch
    r.fill_normal(bias.data(), 384, 0.3f);
    Buffer bl = up(logits.data(), logits.size() * 4), bb = up(bias.data(), 384 * 4);
    Buffer bid = d.alloc(n * 24), bw = d.alloc(n * 24);
    { Enc e; arch::encode_router_topk(e, bl, bb, bid, bw, n); e.wait(); }
    int ties = 0;
    bool o = true;
    for (int t = 0; t < n && o; t++)
      o &= check_token(&logits[t * 384], bias.data(), (int*)bid.cpu() + t * 6, (float*)bw.cpu() + t * 6, t, ties);
    printf("router_topk n=%d: %s (near-ties %d)\n", n, o ? "ok" : "FAIL", ties);
    ok &= o;

    if (n == 2048) {  // 2) routing lists from the GPU ids
      auto L = arch::RouteLists::create(n);
      { Enc e; arch::encode_route_lists(e, bid, bw, n, L); e.wait(); }
      std::vector<uint32_t> c, off, ent, tok, slot;
      std::vector<float> lw;
      ref_arch::route_lists((int*)bid.cpu(), (float*)bw.cpu(), n, c, off, ent, tok, lw, slot);
      bool lo = !memcmp(L.counts.cpu(), c.data(), 384 * 4) && !memcmp(L.offsets.cpu(), off.data(), 385 * 4) &&
                !memcmp(L.entry.cpu(), ent.data(), n * 24) && !memcmp(L.tok.cpu(), tok.data(), n * 24) &&
                !memcmp(L.w.cpu(), lw.data(), n * 24) && !memcmp(L.slot_of.cpu(), slot.data(), n * 24);
      printf("route_lists n=%d: %s (offsets[384]=%u)\n", n, lo ? "ok (bit-exact)" : "FAIL", ((uint32_t*)L.offsets.cpu())[384]);
      ok &= lo;
    }
  }

  // 3) fused decode matvec + top-k (synthetic and real layer-0 router)
  std::vector<float> W(384 * 5120), bias(384), x(5120);
  r.fill_normal(W.data(), W.size(), 0.02f);
  r.fill_normal(bias.data(), 384, 0.1f);
  bool real = false;
  for (int pass = 0; pass < 2; pass++) {
    if (pass == 1) {
      real = arch_test::read_tensor_f32("blk.0.ffn_gate_inp.weight", W) && arch_test::read_tensor_f32("blk.0.exp_probs_b.bias", bias);
      if (!real) { printf("real router check skipped (model not present)\n"); break; }
    }
    Buffer bW = up(W.data(), W.size() * 4), bb = up(bias.data(), 384 * 4), bx = d.alloc(5120 * 4);
    Buffer bid = d.alloc(24), bw = d.alloc(24);
    for (int G : {48, 96, 24, 384}) {
      arch::options().router_groups = G;
      int ties = 0;
      bool o = true;
      for (int it = 0; it < 20 && o; it++) {
        r.fill_normal(x.data(), 5120, 1.0f);
        memcpy(bx.cpu(), x.data(), 5120 * 4);
        { Enc e; arch::encode_router_fused(e, bx, bW, bb, bid, bw, s); e.wait(); }
        std::vector<float> logits(384);
        for (int e = 0; e < 384; e++) {
          double a = 0;
          for (int i = 0; i < 5120; i++) a += (double)W[e * 5120 + i] * x[i];
          logits[e] = (float)a;
        }
        if (it == 0) o &= arch_test::close("fused logits", (float*)s.router_logits.cpu(), logits.data(), 384, 1e-5f, 1e-4f);
        o &= check_token(logits.data(), bias.data(), (int*)bid.cpu(), (float*)bw.cpu(), it, ties);
      }
      printf("router_fused %s G=%d: %s (near-ties %d)\n", pass ? "real blk.0" : "synthetic", G, o ? "ok" : "FAIL", ties);
      ok &= o;
    }
    arch::options().router_groups = 96;
    // stress: 300 fused router dispatches in one command buffer, identical results
    const int R = 300;
    Buffer sid = d.alloc(R * 24), sw = d.alloc(R * 24);
    { Enc e; for (int i = 0; i < R; i++) { arch::encode_router_fused(e, bx, bW, bb, sid.sub(i * 24, 24), sw.sub(i * 24, 24), s); e.barrier(); } e.wait(); }
    int bad = 0;
    for (int i = 1; i < R; i++) bad += memcmp((char*)sid.cpu() + i * 24, sid.cpu(), 24) || memcmp((char*)sw.cpu() + i * 24, sw.cpu(), 24);
    printf("router_fused %d-dispatch stress: %s (%d differ)\n", R, bad ? "FAIL" : "ok", bad);
    ok &= bad == 0;
  }
  printf(ok ? "test_arch_router: ALL OK\n" : "test_arch_router: FAILED\n");
  return ok ? 0 : 1;
}
