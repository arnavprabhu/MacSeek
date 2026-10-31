// hc_step (hc_post + hc_mixes/Sinkhorn + hc_pre + RMSNorm): decode fused/split/rows and prefill MMA/rows vs CPU ref.
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

struct HcData {
  std::vector<float> fn, scale, base, norm_w;
};

static HcData synth_weights(arch_test::Rng& r) {
  HcData w;
  w.fn.resize(24 * 20480);
  r.fill_normal(w.fn.data(), w.fn.size(), 0.01f);
  w.scale = {r.uniform(0.5f, 2.f), r.uniform(0.5f, 2.f), r.uniform(0.5f, 2.f)};
  w.base.resize(24);
  r.fill_normal(w.base.data(), 24, 1.0f);
  w.norm_w.resize(5120);
  for (auto& v : w.norm_w) v = r.uniform(0.5f, 1.5f);
  return w;
}

static Buffer up(const std::vector<float>& v) {
  Buffer b = Device::get().alloc(v.size() * 4);
  memcpy(b.cpu(), v.data(), v.size() * 4);
  return b;
}

// Runs one hc_step for n tokens on GPU and CPU and compares.
static bool run_case(const char* name, const HcData& w, int n, bool post, bool mixes, arch_test::Rng& r,
                     const arch::Scratch& s, const std::vector<float>* h_given = nullptr, bool pre = true) {
  std::vector<float> h(n * 20480), xs(n * 5120), mp(n * 24);
  if (h_given) h = *h_given; else r.fill_normal(h.data(), h.size(), 0.7f);
  r.fill_normal(xs.data(), xs.size(), 1.0f);
  for (int t = 0; t < n; t++) {  // plausible prev mixes: pre in (0,1), post in (0,2), comb ~ doubly stochastic
    float* m = &mp[t * 24];
    for (int j = 0; j < 4; j++) m[j] = r.uniform(0.f, 1.f);
    for (int j = 0; j < 4; j++) m[4 + j] = r.uniform(0.f, 2.f);
    for (int j = 0; j < 16; j++) m[8 + j] = r.uniform(0.f, 0.5f);
  }
  Buffer bh = up(h), bxs = up(xs), bmp = up(mp), bfn = up(w.fn), bsc = up(w.scale), bba = up(w.base), bnw = up(w.norm_w);
  auto& d = Device::get();
  Buffer bho = d.alloc(h.size() * 4), bmo = d.alloc(n * 24 * 4), bxo = d.alloc(n * 5120 * 4);
  arch::HcStep a;
  a.h_in = bh; a.h_out = bho; a.x_sub = post ? bxs : Buffer{}; a.mix_prev = bmp;
  a.fn = mixes ? bfn : Buffer{}; a.scale = bsc; a.base = bba; a.mix_out = bmo; a.norm_w = bnw; a.x_out = pre ? bxo : Buffer{}; a.n = n;
  { Enc e; arch::encode_hc_step(e, a, s); e.wait(); }
  std::vector<float> rh(h.size()), rm(n * 24, 0.f), rx(n * 5120);
  for (int t = 0; t < n; t++)
    ref_arch::hc_step(&h[t * 20480], post ? &xs[t * 5120] : nullptr, &mp[t * 24], mixes ? w.fn.data() : nullptr,
                      w.scale.data(), w.base.data(), w.norm_w.data(), &rh[t * 20480], &rm[t * 24], pre ? &rx[t * 5120] : nullptr);
  printf("%s (n=%d post=%d mixes=%d pre=%d)\n", name, n, post, mixes, pre);
  bool ok = true;
  if (post) ok &= close("h_out", (float*)bho.cpu(), rh.data(), rh.size(), 1e-5f, 1e-5f);
  if (mixes) ok &= close("mix_out", (float*)bmo.cpu(), rm.data(), rm.size(), 2e-5f, 2e-4f);
  if (pre) ok &= close("x_out", (float*)bxo.cpu(), rx.data(), rx.size(), 2e-5f, 1e-4f);
  else {
    bool untouched = true;
    for (size_t i = 0; i < rx.size(); i++) untouched &= ((float*)bxo.cpu())[i] == 0.0f;
    printf("  %-44s %s\n", "x_out untouched (no pre)", untouched ? "ok" : "FAIL");
    ok &= untouched;
  }
  for (Buffer* b : {&bh, &bxs, &bmp, &bfn, &bsc, &bba, &bnw, &bho, &bmo, &bxo}) d.release(*b);
  return ok;
}

// Chain of K decode steps in one command buffer (counter reuse; ping-pong mixes), compared to the CPU chain.
static bool run_chain(const HcData& w, int K, arch_test::Rng& r, const arch::Scratch& s) {
  auto& d = Device::get();
  std::vector<float> h(20480), mix(24);
  r.fill_normal(h.data(), h.size(), 0.7f);
  arch::fill_identity_mix(mix.data());
  std::vector<std::vector<float>> xs(K, std::vector<float>(5120));
  for (auto& x : xs) r.fill_normal(x.data(), 5120, 1.0f);
  Buffer bh = up(h), bm[2] = {up(mix), d.alloc(96)}, bfn = up(w.fn), bsc = up(w.scale), bba = up(w.base), bnw = up(w.norm_w);
  std::vector<Buffer> bxs, bxo;
  for (int k = 0; k < K; k++) { bxs.push_back(up(xs[k])); bxo.push_back(d.alloc(5120 * 4)); }
  {
    Enc e;
    for (int k = 0; k < K; k++) {
      arch::HcStep a;
      a.h_in = bh; a.h_out = bh; a.x_sub = k ? bxs[k] : Buffer{}; a.mix_prev = bm[k & 1]; a.fn = bfn; a.scale = bsc;
      a.base = bba; a.mix_out = bm[(k + 1) & 1]; a.norm_w = bnw; a.x_out = bxo[k];
      arch::encode_hc_step(e, a, s);
      e.barrier();
    }
    e.wait();
  }
  std::vector<float> rh = h, rm = mix, rmo(24), rx(5120);
  bool ok = true;
  for (int k = 0; k < K; k++) {
    std::vector<float> hn(20480);
    ref_arch::hc_step(rh.data(), k ? xs[k].data() : nullptr, rm.data(), w.fn.data(), w.scale.data(), w.base.data(),
                      w.norm_w.data(), hn.data(), rmo.data(), rx.data());
    if (k) rh = hn;
    rm = rmo;
    char nm[64];
    snprintf(nm, sizeof nm, "chain step %d x_out", k);
    ok &= close(nm, (float*)bxo[k].cpu(), rx.data(), 5120, 1e-4f, 1e-3f);
  }
  ok &= close("chain final h", (float*)bh.cpu(), rh.data(), 20480, 1e-4f, 1e-3f);
  ok &= close("chain final mix", (float*)bm[K & 1].cpu(), rm.data(), 24, 1e-4f, 1e-3f);
  return ok;
}

// Race / determinism stress: R fused steps with identical inputs in one command buffer (barrier-separated, as in a
// token) must produce bit-identical outputs (fixed-order reductions; counter reuse across dispatches).
static bool stress(const HcData& w, int R, arch_test::Rng& r, const arch::Scratch& s) {
  auto& d = Device::get();
  std::vector<float> h(20480), xs(5120), mp(24);
  r.fill_normal(h.data(), h.size(), 0.7f);
  r.fill_normal(xs.data(), xs.size(), 1.0f);
  for (auto& v : mp) v = r.uniform(0.f, 0.5f);
  Buffer bh = up(h), bxs = up(xs), bmp = up(mp), bfn = up(w.fn), bsc = up(w.scale), bba = up(w.base), bnw = up(w.norm_w);
  Buffer out_x = d.alloc((size_t)R * 5120 * 4), out_m = d.alloc((size_t)R * 24 * 4), out_h = d.alloc((size_t)R * 20480 * 4);
  {
    Enc e;
    for (int i = 0; i < R; i++) {
      arch::HcStep a;
      a.h_in = bh; a.h_out = out_h.sub((size_t)i * 20480 * 4, 20480 * 4); a.x_sub = bxs; a.mix_prev = bmp; a.fn = bfn;
      a.scale = bsc; a.base = bba; a.norm_w = bnw;
      a.mix_out = out_m.sub((size_t)i * 96, 96); a.x_out = out_x.sub((size_t)i * 5120 * 4, 5120 * 4);
      arch::encode_hc_step(e, a, s);
      e.barrier();
    }
    e.wait();
  }
  int bad = 0;
  for (int i = 1; i < R; i++) {
    bad += memcmp((float*)out_x.cpu() + i * 5120, out_x.cpu(), 5120 * 4) != 0;
    bad += memcmp((float*)out_m.cpu() + i * 24, out_m.cpu(), 96) != 0;
    bad += memcmp((float*)out_h.cpu() + i * 20480, out_h.cpu(), 20480 * 4) != 0;
  }
  printf("stress: %d fused steps in one command buffer bit-identical: %s (%d differ)\n", R, bad ? "FAIL" : "ok", bad);
  return bad == 0;
}

int main() {
  arch_test::Rng r(1234);
  auto s = arch::Scratch::create(512, 4);
  HcData w = synth_weights(r);
  bool ok = true;
  const char* modes[] = {"decode fused", "decode split", "decode rows"};
  for (int m = 0; m < 3; m++) {
    arch::options().hc_mode = m;
    for (int T : {128, 256, 512, 1024}) {
      if (m != 0 && T != 256) continue;
      arch::options().hc_threads = T;
      char nm[64];
      snprintf(nm, sizeof nm, "%s T=%d", modes[m], T);
      ok &= run_case(nm, w, 1, true, true, r, s);
    }
    ok &= run_case(modes[m], w, 1, false, true, r, s);
    ok &= run_case(modes[m], w, 1, true, false, r, s);
    ok &= run_case(modes[m], w, 1, true, false, r, s, nullptr, false);  // post only (before engram)
  }
  arch::options().hc_mode = 0;
  arch::options().hc_threads = 512;
  ok &= run_chain(w, 12, r, s);
  for (int T : {128, 512, 1024}) { arch::options().hc_threads = T; ok &= stress(w, 300, r, s); }
  arch::options().hc_threads = 512;
  for (int n : {2, 7, 8, 31, 32, 37, 256}) ok &= run_case(n < 8 ? "prefill rows (auto)" : n < 32 ? "prefill mma8 (auto)" : "prefill mma32", w, n, true, true, r, s);
  arch::options().hc_prefill_mma8 = true;
  ok &= run_case("prefill mma8 (forced)", w, 37, true, true, r, s);
  arch::options().hc_prefill_mma8 = false;
  ok &= run_case("prefill mma no-post", w, 64, false, true, r, s);
  ok &= run_case("prefill post-only", w, 19, true, false, r, s, nullptr, false);
  arch::options().hc_prefill_rows = true;
  ok &= run_case("prefill rows", w, 37, true, true, r, s);
  arch::options().hc_prefill_rows = false;

  // Real hc weights of layer 0 (attention) + attn_norm, with h = real token embeddings (if the model is present).
  HcData rw;
  if (arch_test::read_tensor_f32("blk.0.hc_attn_fn.weight", rw.fn) && arch_test::read_tensor_f32("blk.0.hc_attn_scale.weight", rw.scale) &&
      arch_test::read_tensor_f32("blk.0.hc_attn_base.weight", rw.base) && arch_test::read_tensor_f32("blk.0.attn_norm.weight", rw.norm_w)) {
    printf("real blk.0 hc_attn: scale = %g %g %g, base[0..7] =", rw.scale[0], rw.scale[1], rw.scale[2]);
    for (int i = 0; i < 8; i++) printf(" %.3g", rw.base[i]);
    printf("\n");
    arch_test::TensorInfo te;
    arch_test::find_tensor("token_embd.weight", te);
    const int toks[8] = {0, 1, 2, 100, 1000, 12345, 99999, 129279};
    std::vector<float> h(8 * 20480), row(5120);
    std::vector<uint8_t> raw(4200);
    for (int t = 0; t < 8; t++) {
      arch_test::read_tensor_bytes(te, (uint64_t)toks[t] * 4200, 4200, raw.data());
      ref_arch::q6k_row(raw.data(), row.data(), 5120);
      for (int j = 0; j < 4; j++) memcpy(&h[t * 20480 + j * 5120], row.data(), 5120 * 4);
    }
    std::vector<float> h1(h.begin(), h.begin() + 20480);
    ok &= run_case("real decode fused", rw, 1, false, true, r, s, &h1);
    ok &= run_case("real prefill mma8", rw, 8, false, true, r, s, &h);
    std::vector<float> h32(32 * 20480);
    for (int t = 0; t < 32; t++) memcpy(&h32[t * 20480], &h[(t % 8) * 20480], 20480 * 4);
    ok &= run_case("real prefill mma32", rw, 32, false, true, r, s, &h32);
  } else {
    printf("real-weight check skipped (model not present)\n");
  }
  printf(ok ? "test_arch_hc: ALL OK\n" : "test_arch_hc: FAILED\n");
  return ok ? 0 : 1;
}
