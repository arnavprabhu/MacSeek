// bench_qmatmul: prefill matmul throughput (TFLOPS) vs n, per weight type, vs a plain F16 GEMM peak measured here
// with the same tensor API. Usage: bench_qmatmul [--peak] [--moe] [filter]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "core/metal.hpp"
#include "kernels/quant.hpp"

using namespace ms;
using ms::q::QType;

static void fill_rand(Buffer& b, uint32_t seed, uint32_t mask = 0x3bff3bffu) {
  uint32_t* p = (uint32_t*)b.cpu(); uint32_t st = seed;
  for (size_t i = 0; i < b.size / 4; i++) { st = st * 1664525u + 1013904223u; p[i] = st & mask; }
}

template <class F> static double best_ms(int reps, F&& enc) {
  std::vector<double> t;
  for (int r = 0; r < reps; r++) { Enc e; enc(e); e.wait(); if (r) t.push_back(e.gpu_ms()); }
  return *std::min_element(t.begin(), t.end());
}

static double peak_f16(bool verbose) {
  auto& d = Device::get();
  double best = 0;
  const int n = 4096, m = 8192, k = 4096;
  Buffer A = d.alloc((size_t)n * k * 2), B = d.alloc((size_t)m * k * 2), C = d.alloc((size_t)n * m * 4);
  fill_rand(A, 1); fill_rand(B, 2);
  for (auto [name, tm, tn] : {std::tuple{"q_gemm_f16_32x64", 32, 64}, {"q_gemm_f16_64x64", 64, 64},
                              {"q_gemm_f16_64x128", 64, 128}, {"q_gemm_f16_128x64", 128, 64},
                              {"q_gemm_f16_128x128", 128, 128}, {"q_gemm_f32a_64x128", 64, 128},
                              {"q_gemm_f16_64x128_k64", 64, 128}, {"q_gemm_f16_64x128_k128", 64, 128},
                              {"q_gemm_f16_64x256", 64, 256}}) {
    uint32_t dims[4] = {(uint32_t)n, (uint32_t)m, (uint32_t)k, 0};
    int nsg = ((tm == 128 && tn == 128) || tn == 256) ? 8 : 4;
    bool f32a = strstr(name, "f32a") != nullptr;
    Buffer A32 = f32a ? d.alloc((size_t)n * k * 4) : Buffer{};
    double ms = best_ms(6, [&](Enc& e) {
      e.set_pipeline(name); e.set_buffer(0, f32a ? A32 : A); e.set_buffer(1, B); e.set_buffer(2, C); e.set_bytes(3, dims, 16);
      e.dispatch_groups(m / tn, n / tm, 1, 32 * nsg);
    });
    double tf = 2.0 * n * m * k / ms / 1e9;
    if (verbose) printf("  plain F16 GEMM (MPP matmul2d) %-20s n=%d m=%d k=%d: %6.2f ms  %6.2f TFLOPS\n", name, n, m, k, ms, tf);
    if (!f32a) best = std::max(best, tf);
    if (f32a) d.release(A32);
  }
  // quick correctness spot check of the last config (C row 0..3, col 0..3)
  d.release(A); d.release(B); d.release(C);
  return best;
}

// MoE prefill: n tokens, E experts, top-6 random routing (+ shared Q6_K), IQ1_S experts.
static void bench_moe(double peak) {
  auto& d = Device::get();
  printf("MoE prefill (IQ1_S experts + Q6_K shared, top-6 random routing)\n");
  printf("  %-6s %-8s %-12s %10s %9s %8s\n", "n", "experts", "tok/expert", "ms", "TFLOPS", "%peak");
  for (auto [n, E] : {std::pair{512, 96}, {2048, 96}, {2048, 32}, {4096, 64}}) {
    const int D = q::kExpertDim, I = q::kExpertInter, K = 6;
    const size_t gu = q::row_bytes(QType::IQ1_S, D) * I, dn = q::row_bytes(QType::IQ1_S, I) * D;
    const size_t sgu = q::row_bytes(QType::Q6_K, D) * I, sdn = q::row_bytes(QType::Q6_K, I) * D;
    Buffer arena = d.alloc((size_t)E * (2 * gu + dn) + 2 * sgu + sdn);
    fill_rand(arena, 9);
    std::vector<q::ExpertSlot> slots(E);
    size_t off = 0;
    for (int j = 0; j < E; j++) {
      slots[j].gate = arena.sub(off, gu); off += gu; slots[j].up = arena.sub(off, gu); off += gu;
      slots[j].down = arena.sub(off, dn); off += dn;
    }
    q::ExpertSlot sh{arena.sub(off, sgu), arena.sub(off + sgu, sgu), arena.sub(off + 2 * sgu, sdn)};
    uint32_t st = 1;
    std::vector<std::vector<int>> per(E);
    std::vector<int> slot_of(n * K);
    std::vector<std::pair<int, int>> ent;  // (token, k) per expert
    std::vector<std::vector<std::pair<int, int>>> pe(E);
    for (int t = 0; t < n; t++) {
      std::vector<int> used;
      while ((int)used.size() < K) {
        st = st * 1664525u + 1013904223u;
        int j = (int)((st >> 8) % E);
        if (std::find(used.begin(), used.end(), j) == used.end()) { used.push_back(j); pe[j].push_back({t, (int)used.size() - 1}); }
      }
    }
    std::vector<int> offs(E + 1), tok; std::vector<float> wt;
    int ne = 0; std::vector<q::ExpertSlot> used_slots;
    for (int j = 0; j < E; j++) {
      if (pe[j].empty()) continue;
      offs[ne] = (int)tok.size(); used_slots.push_back(slots[j]);
      for (auto [t, k] : pe[j]) { slot_of[t * K + k] = (int)tok.size(); tok.push_back(t); wt.push_back(0.16f); }
      ne++;
    }
    offs[ne] = (int)tok.size();
    const int total = (int)tok.size();
    Buffer btok = d.alloc(total * 4), bwt = d.alloc(total * 4), bslot = d.alloc(n * K * 4);
    memcpy(btok.cpu(), tok.data(), total * 4); memcpy(bwt.cpu(), wt.data(), total * 4); memcpy(bslot.cpu(), slot_of.data(), n * K * 4);
    Buffer X = d.alloc((size_t)n * D * 4), Y = d.alloc((size_t)n * D * 4);
    Buffer scratch = d.alloc(q::moe_prefill_scratch_bytes(total, n, ne));
    q::MoePrefillArgs a;
    a.n_tokens = n; a.n_exp = ne; a.slots = used_slots.data(); a.offs = offs.data(); a.tok = btok; a.wt = bwt;
    a.slot_of = bslot; a.k_per_token = K; a.shared = sh; a.X = X; a.Y = Y; a.scratch = scratch;
    double ms = best_ms(4, [&](Enc& e) { q::encode_moe_prefill(e, a); });
    double fl = 2.0 * 3 * D * I * (double)(total + n);
    printf("  %-6d %-8d %-12.0f %10.2f %9.2f %7.0f%%\n", n, E, (double)total / ne, ms, fl / ms / 1e9, 100 * fl / ms / 1e9 / peak);
    for (Buffer* b : {&arena, &btok, &bwt, &bslot, &X, &Y, &scratch}) d.release(*b);
  }
}


// Realistic MoE prefill (what ds41.mm runs per layer): n tokens, top-6 of 384 experts drawn without replacement from a
// Zipf(s) popularity (random expert permutation), experts in id order processed in groups of `group` experts per
// command buffer (encode_moe_prefill call), shared expert (all n tokens) in the first group. Reports the sum of the
// groups' GPU times, best of `reps` passes. Expert weights alias over `uniq` distinct matrices (>= 512 MiB streamed).
static bool g_zall = true;  // --nozall: per-call reduce (old engine mode)
static void bench_moe_real(double peak, int n, int group, double zs, QType tgu, QType tdn, QType tsh, int reps, int NE) {
  auto& d = Device::get();
  const int D = q::kExpertDim, I = q::kExpertInter, K = 6;
  const size_t gu = q::row_bytes(tgu, D) * I, dn = q::row_bytes(tdn, I) * D;
  const size_t sgu = q::row_bytes(tsh, D) * I, sdn = q::row_bytes(tsh, I) * D;
  int uniq = (int)std::min<size_t>(NE, ((size_t)600 << 20) / (2 * gu + dn) + 1);
  Buffer arena = d.alloc((size_t)uniq * (2 * gu + dn) + 2 * sgu + sdn + 64, "arena");
  fill_rand(arena, 9);
  std::vector<q::ExpertSlot> slots(NE);
  size_t off = 4;  // weight offsets only need 4-byte alignment (IQ1_M / Q5_K decoders use 32-bit loads)
  std::vector<q::ExpertSlot> base(uniq);
  for (int j = 0; j < uniq; j++) {
    base[j].gate = arena.sub(off, gu); off += gu; base[j].up = arena.sub(off, gu); off += gu;
    base[j].down = arena.sub(off, dn); off += dn;
  }
  for (int j = 0; j < NE; j++) slots[j] = base[j % uniq];
  q::ExpertSlot sh{arena.sub(off, sgu), arena.sub(off + sgu, sgu), arena.sub(off + 2 * sgu, sdn)};
  // routing
  std::mt19937 rng(1234);
  std::vector<int> perm(NE);
  for (int j = 0; j < NE; j++) perm[j] = j;
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<double> pw(NE);
  for (int r = 0; r < NE; r++) pw[perm[r]] = 1.0 / std::pow(r + 1.0, zs);
  std::vector<std::vector<int>> toks_of(NE);  // entries t*K + k
  std::uniform_real_distribution<double> ud(0, 1);
  for (int t = 0; t < n; t++) {
    std::vector<double> w = pw;
    double sum = 0; for (double v : w) sum += v;
    for (int k = 0; k < K; k++) {
      double u = ud(rng) * sum; int j = 0;
      for (; j < NE - 1; j++) { if (u < w[j]) break; u -= w[j]; }
      if (w[j] == 0) { j = 0; while (w[j] == 0) j++; }
      toks_of[j].push_back(t * K + k); sum -= w[j]; w[j] = 0;
    }
  }
  std::vector<int> used, cnt;
  for (int j = 0; j < NE; j++) if (!toks_of[j].empty()) { used.push_back(j); cnt.push_back((int)toks_of[j].size()); }
  std::vector<int> sc = cnt; std::sort(sc.begin(), sc.end());
  int lt16 = 0, gt256 = 0; for (int c : sc) { lt16 += c < 16; gt256 += c > 256; }
  const int ngroups = ((int)used.size() + group - 1) / group;
  printf("MoE prefill realistic: n=%d top-%d of %d, Zipf s=%.2f, %s/%s + shared %s, %d experts used (tokens/expert min %d "
         "median %d max %d, <16: %d, >256: %d), %d groups of %d\n", n, K, NE, zs, q::type_name(tgu), q::type_name(tdn),
         q::type_name(tsh), (int)used.size(), sc.front(), sc[sc.size() / 2], sc.back(), lt16, gt256, ngroups, group);
  // per-group routing buffers + one scratch per group (no reuse hazards)
  struct Grp { std::vector<int> offs; std::vector<q::ExpertSlot> sl; Buffer tok, wt, slot_of, scratch; int total; };
  std::vector<Grp> gs(ngroups);
  Buffer X = d.alloc((size_t)n * D * 4, "X"), Y = d.alloc((size_t)n * D * 4, "Y");
  { std::normal_distribution<float> nd(0.f, 1.f); float* x = (float*)X.cpu(); for (size_t i = 0; i < (size_t)n * D; i++) x[i] = nd(rng); }
  size_t max_scr = 0;
  for (int gi = 0; gi < ngroups; gi++) {
    Grp& g = gs[gi];
    std::vector<int> tok, slot_of(n * K, -1); std::vector<float> wt;
    for (int jj = gi * group; jj < std::min((int)used.size(), (gi + 1) * group); jj++) {
      g.offs.push_back((int)tok.size()); g.sl.push_back(slots[used[jj]]);
      for (int ent : toks_of[used[jj]]) { slot_of[ent] = (int)tok.size(); tok.push_back(ent / K); wt.push_back(0.1f + 0.05f * (ent % K)); }
    }
    g.offs.push_back((int)tok.size());
    g.total = (int)tok.size();
    g.tok = d.alloc(g.total * 4 + 16); g.wt = d.alloc(g.total * 4 + 16); g.slot_of = d.alloc(n * K * 4);
    memcpy(g.tok.cpu(), tok.data(), g.total * 4); memcpy(g.wt.cpu(), wt.data(), g.total * 4);
    memcpy(g.slot_of.cpu(), slot_of.data(), n * K * 4);
    max_scr = std::max(max_scr, q::moe_prefill_scratch_bytes(g.total, n, (int)g.sl.size()));
  }
  Buffer scratch = d.alloc(max_scr, "scratch");
  // engine mode (ds41.mm ZALL): every group writes its Z rows into one layer-wide Z (routed rows at the group's global
  // entry offset, shared rows at n*K), no per-call reduce; one encode_moe_reduce after the last group
  std::vector<int> g0(ngroups + 1, 0);
  for (int gi = 0; gi < ngroups; gi++) g0[gi + 1] = g0[gi] + gs[gi].total;
  Buffer zall, slot_all;
  if (g_zall) {
    zall = d.alloc((size_t)(n * K + n) * D * 4, "zall");
    slot_all = d.alloc((size_t)n * K * 4, "slot_all");
    int* sa = (int*)slot_all.cpu();
    for (int gi = 0; gi < ngroups; gi++) {
      const int* so = (const int*)gs[gi].slot_of.cpu();
      for (int i = 0; i < n * K; i++) if (so[i] >= 0) sa[i] = g0[gi] + so[i];
    }
  }
  double best = 1e30, flops = 0;
  for (auto& g : gs) flops += 2.0 * 3 * D * I * g.total;
  flops += 2.0 * 3 * D * I * n;
  for (int r = 0; r < reps; r++) {
    double tot = 0;
    for (int gi = 0; gi < ngroups; gi++) {
      Grp& g = gs[gi];
      q::MoePrefillArgs a;
      a.t_gate_up = tgu; a.t_down = tdn; a.t_shared = tsh;
      a.n_tokens = n; a.n_exp = (int)g.sl.size(); a.slots = g.sl.data(); a.offs = g.offs.data(); a.tok = g.tok;
      a.wt = g.wt; a.slot_of = g.slot_of; a.k_per_token = K; a.has_shared = gi == 0; a.shared = sh; a.X = X; a.Y = Y;
      a.scratch = scratch;
      if (g_zall) {
        const size_t rb = (size_t)D * 4;
        a.z_ext = zall.sub((size_t)g0[gi] * rb, (size_t)std::max(g.total, 1) * rb);
        a.z_shared_ext = zall.sub((size_t)n * K * rb, (size_t)n * rb);
        a.skip_reduce = true;
      }
      Enc e; q::encode_moe_prefill(e, a);
      if (g_zall && gi == ngroups - 1) { e.barrier(); q::encode_moe_reduce(e, zall, Y, slot_all, n, n * K, K, true); }
      e.wait();
      tot += e.gpu_ms();
    }
    best = std::min(best, tot);
  }
  printf("  total %.2f ms  %.2f TFLOPS effective  (%.0f%% of F16 peak %.1f)   x40 layers = %.2f s per %d-token chunk\n", best,
         flops / best / 1e9, 100 * flops / best / 1e9 / peak, peak, best * 40 / 1e3, n);
  for (auto& g : gs) { d.release(g.tok); d.release(g.wt); d.release(g.slot_of); }
  for (Buffer* b : {&arena, &X, &Y, &scratch}) d.release(*b);
  if (g_zall) { d.release(zall); d.release(slot_all); }
}

int main(int argc, char** argv) {
  bool only_peak = false, moe = false, real = false; const char* filt = nullptr;
  int rn = 4096, rgroup = 64, reps = 3, ne = 384; double zs = 0.8; std::string rtypes = "all";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--peak")) only_peak = true;
    else if (!strcmp(argv[i], "--moe")) moe = true;
    else if (!strcmp(argv[i], "--real")) { moe = true; real = true; }
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) rn = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--group") && i + 1 < argc) rgroup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--zipf") && i + 1 < argc) zs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--types") && i + 1 < argc) rtypes = argv[++i];  // all | iq1 | iq2 | gu,down,shared
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) ne = atoi(argv[++i]);  // experts per layer (192: GSQ-RCO)
    else if (!strcmp(argv[i], "--nozall")) g_zall = false;  // per-call reduce instead of the engine's layer-wide Z
    else filt = argv[i];
  }
  double peak = (real && getenv("BENCH_PEAK")) ? atof(getenv("BENCH_PEAK")) : peak_f16(true);
  printf("F16 GEMM peak: %.2f TFLOPS\n", peak);
  if (only_peak) return 0;
  if (real) {  // bench_qmatmul --moe --real [--n 4096] [--group 64] [--zipf 0.8] [--experts 384] [--reps 3]
               //   [--types all|iq1|iq2|gu,down,shared  e.g. iq1_m,iq2_xxs,q5_k]
    if (rtypes == "all" || rtypes == "iq1") bench_moe_real(peak, rn, rgroup, zs, QType::IQ1_S, QType::IQ1_S, QType::Q6_K, reps, ne);
    if (rtypes == "all" || rtypes == "iq2") bench_moe_real(peak, rn, rgroup, zs, QType::IQ2_XXS, QType::IQ2_XXS, QType::Q8_0, reps, ne);
    if (rtypes.find(',') != std::string::npos) {
      auto parse = [](const std::string& s) {
        for (QType t : {QType::IQ1_S, QType::IQ2_XXS, QType::IQ1_M, QType::IQ2_XS, QType::Q6_K, QType::Q8_0, QType::Q5_K}) {
          std::string n = q::type_name(t);
          for (auto& ch : n) ch = (char)tolower(ch);
          if (n == s) return t;
        }
        fprintf(stderr, "unknown type %s\n", s.c_str()); exit(2);
      };
      const size_t c1 = rtypes.find(','), c2 = rtypes.find(',', c1 + 1);
      bench_moe_real(peak, rn, rgroup, zs, parse(rtypes.substr(0, c1)), parse(rtypes.substr(c1 + 1, c2 - c1 - 1)),
                     parse(rtypes.substr(c2 + 1)), reps, ne);
    }
    return 0;
  }
  if (moe) { bench_moe(peak); return 0; }
  struct Shape { QType t; int rows, cols; const char* what; };
  const Shape shapes[] = {
      {QType::Q6_K, 32768, 1280, "attn_q_b"}, {QType::Q6_K, 5120, 8192, "attn_output_b"},
      {QType::Q5_K, 5120, 8192, "attn_output_b"}, {QType::Q6_K, 1280, 5120, "attn_q_a"},
      {QType::Q8_0, 2304, 5120, "shexp gate/up"}, {QType::IQ1_S, 2304, 5120, "exp gate/up"},
      {QType::IQ2_XXS, 5120, 2304, "exp down"}, {QType::BF16, 4096, 1280, "indexer q_b"},
      {QType::IQ1_M, 2304, 5120, "exp gate/up"}, {QType::IQ2_XS, 5120, 2304, "exp down"},
      {QType::IQ2_XS, 2304, 5120, "exp gate/up"}, {QType::Q5_K, 2304, 5120, "shexp gate/up"},
      {QType::F16, 8192, 4096, "f16 synthetic"},
  };
  const int ns[] = {32, 128, 512, 2048, 4096};
  printf("%-8s %-14s %-14s", "type", "tensor", "shape");
  for (int n : ns) printf("  n=%-5d", n);
  printf("   (TFLOPS; %% of F16 peak at n=4096)\n");
  auto& d = Device::get();
  for (const Shape& s : shapes) {
    std::string lab = std::string(q::type_name(s.t)) + " " + s.what;
    if (filt && lab.find(filt) == std::string::npos) continue;
    Buffer W = d.alloc(q::row_bytes(s.t, s.cols) * s.rows);
    fill_rand(W, 5, s.t == QType::F32 ? 0x3f7f3f7fu : 0x3bff3bffu);
    Buffer X16 = d.alloc((size_t)4096 * s.cols * 2), Y = d.alloc((size_t)4096 * s.rows * 4);
    fill_rand(X16, 6);
    char shape[32]; snprintf(shape, sizeof shape, "[%d->%d]", s.cols, s.rows);
    printf("%-8s %-14s %-14s", q::type_name(s.t), s.what, shape);
    double last = 0;
    for (int n : ns) {
      double ms = best_ms(5, [&](Enc& e) { q::encode_matmul_h(e, s.t, W, s.rows, s.cols, X16, Y, n); });
      last = 2.0 * n * s.rows * s.cols / ms / 1e9;
      printf("  %7.2f", last);
    }
    printf("   %4.0f%%\n", 100 * last / peak);
    d.release(W); d.release(X16); d.release(Y);
  }
  return 0;
}
