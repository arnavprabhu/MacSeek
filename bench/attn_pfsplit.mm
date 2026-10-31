// bench_attn_pfsplit: per-kernel split of the attention-side prefill at depth (default 200000), chunk BENCH_CHUNK
// (default 4096). Each kernel group runs in its own command buffer (min of 3). Synthetic random state; no model.
// Replicates the dispatches of attn.mm (arg structs must match the .metal side).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <string>
#include <cstdlib>
#include <vector>
#include "core/env.hpp"
#include "core/metal.hpp"
#include "kernels/attn.hpp"
using namespace ms;
namespace A = ms::attn;

static uint64_t rs = 88172645463325252ull;
static inline uint32_t xr() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static uint16_t rh() { _Float16 h = (_Float16)(((int)(xr() & 0xFFFF) - 32768) / 16384.0f); uint16_t u; memcpy(&u, &h, 2); return u; }
static void fill_f32(Buffer b, float sd) { float* p = (float*)b.cpu(); for (size_t i = 0; i < b.size / 4; i++) p[i] = sd * (((int)(xr() & 0xFFFF) - 32768) / 32768.0f); }

static double tmin(const std::function<void(Enc&)>& f, int it = 3) {
  double b = 1e9;
  for (int i = 0; i < it; i++) { Enc e; f(e); e.wait(); b = std::min(b, e.gpu_ms()); }
  return b;
}

int main() {
  auto& d = Device::get();
  const int depth = (int)env_int("BENCH_DEPTH", 200000), n = (int)env_int("BENCH_CHUNK", 4096);
  A::Options o = A::Options::from_env(); o.max_chunk = n;
  A::State st; A::init(st, o);
  {  // the base (3-kernel) path needs pf_qblock-sized scratch; the folded path allocates only 2 x pf2_qblock
      const size_t qb = 2 * (size_t)std::min(512, n);  // also covers the folded path's 2 sets at pf2_qblock <= 512
      if (!st.pfS.size) st.pfS = d.alloc(qb * 64 * 640 * 4, "pfS");
      if (st.pfQ.size < qb * 64 * 512 * 2) { d.release(st.pfQ); st.pfQ = d.alloc(qb * 64 * 512 * 2, "pfQ"); }
      if (st.pfP.size < qb * 64 * 640 * 2) { d.release(st.pfP); st.pfP = d.alloc(qb * 64 * 640 * 2, "pfP"); }
    }
  if (st.pf_fold_ok) { d.release(st.pfML); st.pfML = d.alloc(2 * (size_t)std::min(512, n) * 10 * 64 * 8, "pfML"); }
  for (int L = 0; L < A::kLayers; L++) { uint16_t* p = (uint16_t*)st.win[L].cpu(); for (size_t i = 0; i < st.win[L].size / 2; i++) p[i] = rh(); }
  for (int i = 0; i < 4; i++) {
    uint16_t* c = (uint16_t*)st.ckv[i].cpu(); uint16_t* k = (uint16_t*)st.ikey[i].cpu();
    for (size_t j = 0; j < st.ckv[i].size / 2; j++) c[j] = rh();
    for (size_t j = 0; j < st.ikey[i].size / 2; j++) k[j] = rh();
  }
  Buffer q = d.alloc((size_t)n * 64 * 512 * 4), iq = d.alloc((size_t)n * 32 * 128 * 4), iw = d.alloc((size_t)n * 32 * 4),
         out = d.alloc((size_t)n * 64 * 512 * 4), sinks = d.alloc(64 * 4);
  fill_f32(q, 0.2f); fill_f32(iq, 1); fill_f32(iw, 1);
  {  // BENCH_QCORR = rho: consecutive tokens' indexer inputs follow an AR(1) process (real text: neighbouring queries pick
     // mostly the same keys / candidate blocks). 0 = independent random rows (worst case for the candidate union).
    const double rho = env_int("BENCH_QCORR_PCT", 0) / 100.0;
    if (rho > 0) {
      const double c = std::sqrt(1 - rho * rho);
      float* a = (float*)iq.cpu(); float* b = (float*)iw.cpu();
      for (size_t r = 1; r < iq.size / 4 / 4096; r++) for (int i = 0; i < 4096; i++) a[r * 4096 + i] = (float)(rho * a[(r - 1) * 4096 + i] + c * a[r * 4096 + i]);
      for (size_t r = 1; r < iw.size / 4 / 32; r++) for (int i = 0; i < 32; i++) b[r * 32 + i] = (float)(rho * b[(r - 1) * 32 + i] + c * b[r * 32 + i]);
    }
  } fill_f32(sinks, 1);
  const int start = depth;
  printf("prefill split: n=%d depth=%d key_chunk=%d\n", n, depth, o.key_chunk);
  // full index layers (fills topk / cand mask)
  for (int L : {2, 20, 24}) {
    double t = tmin([&](Enc& e) { A::encode_index_prefill(e, st, L, iq, iw, start, n); });
    printf("index layer %2d total            %9.2f ms\n", L, t);
  }
  if (env_int("BENCH_QB_SWEEP", 0)) {  // interleaved sweep of the two-pass query block (sparse layer 20 and 0), min of rounds
    const int qbs[5] = {512, 128, 64, 32, 16};
    double t[5][2];
    for (auto& x : t) x[0] = x[1] = 1e9;
    for (int rd = 0; rd < (int)env_int("BENCH_ROUNDS", 4); rd++)
      for (int i = 0; i < 5; i++) {
        st.opt.pf2_qblock = qbs[i];
        t[i][0] = std::min(t[i][0], tmin([&](Enc& e) { A::encode_sparse(e, st, 20, q, sinks, out, start, n); }, 1));
        t[i][1] = std::min(t[i][1], tmin([&](Enc& e) { A::encode_sparse(e, st, 0, q, sinks, out, start, n); }, 1));
      }
    for (int i = 0; i < 5; i++) printf("qblock %4d: sparse layer 20 %7.2f ms, layer 0 %6.2f ms\n", qbs[i], t[i][0], t[i][1]);
    return 0;
  }
  if (env_int("BENCH_AB", 0)) {  // interleaved A/B of the switchable paths (toggled at run time), min over rounds

    const int rounds = (int)env_int("BENCH_ROUNDS", 4);
    const bool reg_ok = st.pf_score_reg_ok, fold_ok = st.pf_fold_ok;
    // variants: 0 = all new paths off (baseline), 1 = all on, 2 = all on except the one named by BENCH_AB_OFF
    const int nv = 3;
    const char* off = getenv("BENCH_AB_OFF");
    double t[4][3];
    for (auto& x : t) x[0] = x[1] = x[2] = 1e9;
    const A::Options o0 = st.opt;
    for (int rd = 0; rd < rounds; rd++)
      for (int v = 0; v < nv; v++) {
        st.opt = o0;
        st.pf_score_reg_ok = v ? reg_ok : false;
        st.pf_fold_ok = v ? fold_ok : false;
        if (v == 0) { st.opt.pf_merge_filter = false; }
        if (v == 2 && off) {
          std::string f = off;
          if (f == "reg") st.pf_score_reg_ok = false;
          if (f == "fold") st.pf_fold_ok = false;
          if (f == "mfilter") st.opt.pf_merge_filter = false;
        }
        t[0][v] = std::min(t[0][v], tmin([&](Enc& e) { A::encode_index_prefill(e, st, 20, iq, iw, start, n); }, 1));
        t[1][v] = std::min(t[1][v], tmin([&](Enc& e) { A::encode_index_prefill(e, st, 24, iq, iw, start, n); }, 1));
        t[2][v] = std::min(t[2][v], tmin([&](Enc& e) { A::encode_sparse(e, st, 20, q, sinks, out, start, n); }, 1));
        t[3][v] = std::min(t[3][v], tmin([&](Enc& e) { A::encode_sparse(e, st, 0, q, sinks, out, start, n); }, 1));
      }
    st.opt = o0;
    const char* nm[4] = {"index layer 20", "index layer 24", "sparse layer 20 (640 rows)", "sparse layer 0 (128 rows)"};
    for (int i = 0; i < 4; i++) printf("A/B %-28s base %8.2f ms  new %8.2f ms (%.2fx)  new-without-%s %8.2f ms\n", nm[i], t[i][0], t[i][1], t[i][0] / t[i][1], off ? off : "-", t[i][2]);
    st.pf_score_reg_ok = reg_ok; st.pf_fold_ok = fold_ok;
    return 0;
  }
  // ---- prefill index internals for layer 20 (dense, cand source) ----
  struct IdxPfArgs { int start, n, ratio, kc0, kc, ld_s, use_mask, mask_words, cand_src, krows, qrows; };
  struct MergeArgs { int n, k, cnt, ld_s, base, first; };
  const int KC = o.key_chunk;
  for (int L : {2, 20}) {
    const int r = A::ratio_of(L), sl = A::src_slot(A::kv_source_of(L)), len_max = (start + n) / r;
    const bool cs = A::is_cand_source(L);
    double ts = tmin([&](Enc& e) {
      for (int kc0 = 0; kc0 < len_max; kc0 += KC) {
        int c = std::min(KC, len_max - kc0);
        IdxPfArgs a{start, n, r, kc0, c, KC, 0, st.mask_words, cs ? 1 : 0, st.cap[sl] + A::kRowPad, (n + 1) / 2 * 64};
        e.set_pipeline(st.pf_score_reg_ok ? "attn_idx_score_pf2" : "attn_idx_score_pf"); e.set_buffer(0, st.Qi); e.set_buffer(1, st.wq); e.set_buffer(2, st.ikey[sl]);
        e.set_buffer(3, st.S); e.set_buffer(4, st.cand_mask); e.set_buffer(5, st.BS); e.set_value(6, a);
        e.dispatch_groups((c + 127) / 128, (n + 1) / 2, 1, 128); e.barrier();
      } });
    auto mrg = [&](bool blocks) {
      return tmin([&](Enc& e) {
        for (int kc0 = 0; kc0 < len_max; kc0 += KC) {
          int c = std::min(KC, len_max - kc0);
          MergeArgs m = blocks ? MergeArgs{n, 2048, (c + 7) / 8, KC / 8, kc0 / 8, kc0 == 0} : MergeArgs{n, 512, c, KC, kc0, kc0 == 0};
          e.set_pipeline(!blocks && st.opt.pf_merge_filter ? "attn_topk_merge_f" : "attn_topk_merge"); e.set_buffer(0, blocks ? st.BS : st.S); e.set_buffer(1, blocks ? st.Bk : st.Rk);
          e.set_buffer(2, blocks ? st.Bi : st.Ri); e.set_buffer(3, blocks ? st.Bc : st.Rc); e.set_value(4, m);
          e.dispatch_groups(n, 1, 1, 1024); e.barrier();
        } });
    };
    double tm = mrg(false), tb = cs ? mrg(true) : 0;
    double fl = 2.0 * n * 32 * 128 * len_max;
    printf("layer %2d: score %8.2f ms (%5.1f TFLOP/s) | merge k=512 %8.2f ms | merge blocks k=2048 %8.2f ms\n", L, ts,
           fl / ts / 1e9, tm, tb);
  }
  {  // optional: overwrite the top-k lists (layer 24 result) to probe the memory-boundness of sparse attention
    const int tm = (int)env_int("BENCH_TOPK_MODE", 0);  // 1: identical lists (all rows cached)
    if (tm == 1) { int* t = (int*)st.topk.cpu(); for (int i = 0; i < n; i++) for (int j = 0; j < 512; j++) t[i * 512 + j] = j * 37; }
    // overlap of consecutive queries' lists
    const int* t = (const int*)st.topk.cpu(); double same = 0;
    for (int i = 1; i < n; i++) { std::vector<int> a(t + (i - 1) * 512, t + i * 512), b(t + i * 512, t + (i + 1) * 512);
      std::vector<int> c; std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c)); same += c.size(); }
    printf("top-k overlap of consecutive queries: %.1f%%\n", same / (n - 1) / 512 * 100);
  }
  {  // layer 24 (candidate union) split
    struct UnionArgs { int n, mask_words, cap; };
    struct IdxPfListArgs { int start, n, ratio, kc0, kc, ld_s, mask_words, cap, qrows; };
    struct MergeListArgs { int n, k, kc, ld_s, base, first, cap; };
    const int ntiles = (n + 1) / 2, cap = 2 * A::kCandBlocks, len_max = start + n;
    double tu = tmin([&](Enc& e) { UnionArgs a{n, st.mask_words, cap}; e.set_pipeline("attn_cand_union"); e.set_buffer(0, st.cand_mask);
      e.set_buffer(1, st.ulist); e.set_buffer(2, st.ucount); e.set_value(3, a); e.dispatch_groups(ntiles, 1, 1, 1024); });
    const int* uc = (const int*)st.ucount.cpu(); double usum = 0; int umax = 0;
    for (int i = 0; i < ntiles; i++) { usum += uc[i]; umax = std::max(umax, uc[i]); }
    const int nv_max = std::min(cap, (len_max + 7) / 8) * 8;
    double tsc = tmin([&](Enc& e) { for (int kc0 = 0; kc0 < nv_max; kc0 += KC) { const int c = std::min(KC, nv_max - kc0);
      IdxPfListArgs a{start, n, 1, kc0, c, KC, st.mask_words, cap, ntiles * 64};
      e.set_pipeline(st.pf_score_reg_ok ? "attn_idx_score_pf_list2" : "attn_idx_score_pf_list"); e.set_buffer(0, st.Qi); e.set_buffer(1, st.wq);
      e.set_buffer(2, st.ikey[3]); e.set_buffer(3, st.S); e.set_buffer(4, st.cand_mask); e.set_buffer(5, st.ulist); e.set_buffer(6, st.ucount);
      e.set_value(7, a); e.dispatch_groups((c + 63) / 64, ntiles, 1, 128); e.barrier(); } });
    double tml = tmin([&](Enc& e) { for (int kc0 = 0; kc0 < nv_max; kc0 += KC) { const int c = std::min(KC, nv_max - kc0);
      MergeListArgs m{n, 512, c, KC, kc0, kc0 == 0, cap}; e.set_pipeline(st.opt.pf_merge_filter ? "attn_topk_merge_list_f" : "attn_topk_merge_list"); e.set_buffer(0, st.S); e.set_buffer(1, st.Rk);
      e.set_buffer(2, st.Ri); e.set_buffer(3, st.Rc); e.set_buffer(4, st.ulist); e.set_buffer(5, st.ucount); e.set_value(6, m);
      e.dispatch_groups(n, 1, 1, 1024); e.barrier(); } });
    printf("layer 24: union %.2f ms (mean %.0f / max %d blocks per 2-query tile) | score %.2f ms | merge %.2f ms\n", tu, usum / ntiles, umax, tsc, tml);
  }
  // ---- sparse attention internals (one layer with 640 rows; one with 128) ----
  struct PfArgs { int start, nq, q0, chunk_start, ntop, R; float scale; };
  for (int L : {0, 20}) {
    const int r = A::ratio_of(L), ntop = r ? A::kTopk : 0, R = A::kWin + ntop;
    const Buffer& ckv = r ? st.ckv[A::src_slot(A::kv_source_of(L))] : st.win[L];
    const int QB = std::min(o.pf_qblock, n);
    double tt = tmin([&](Enc& e) { A::encode_sparse(e, st, L, q, sinks, out, start, n); });
    double t4[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; k++)
      t4[k] = tmin([&](Enc& e) {
        for (int q0 = 0; q0 < n; q0 += QB) {
          const int nb = std::min(QB, n - q0);
          PfArgs a{start + q0, nb, q0, start, ntop, R, 0.044f};
          if (k == 0) { e.set_pipeline("attn_pf_qprep"); e.set_buffer(0, q); e.set_buffer(1, st.rope_yarn); e.set_buffer(2, st.pfQ); e.set_value(3, a); e.dispatch_groups(nb, 1, 1, 128); }
          if (st.pf_fold_ok && k == 1) { e.set_pipeline("attn_pf2_scores"); e.set_buffer(0, st.pfQ); e.set_buffer(1, st.win[L]); e.set_buffer(2, st.chunk_kv); e.set_buffer(3, ckv); e.set_buffer(4, st.topk); e.set_buffer(5, st.pfP); e.set_buffer(6, st.pfML); e.set_value(7, a); e.dispatch_groups(R / 64, nb, 1, 128); }
          if (st.pf_fold_ok && k == 3) { e.set_pipeline("attn_pf2_pv"); e.set_buffer(0, st.pfP); e.set_buffer(1, st.win[L]); e.set_buffer(2, st.chunk_kv); e.set_buffer(3, ckv); e.set_buffer(4, st.topk); e.set_buffer(5, st.rope_yarn); e.set_buffer(6, out); e.set_buffer(7, st.pfML); e.set_buffer(8, sinks); e.set_value(9, a); e.dispatch_groups(512 / 64, nb, 1, 128); }
          if (st.pf_fold_ok && k != 0) { e.barrier(); continue; }
          if (k == 1) { e.set_pipeline("attn_pf_scores"); e.set_buffer(0, st.pfQ); e.set_buffer(1, st.win[L]); e.set_buffer(2, st.chunk_kv); e.set_buffer(3, ckv); e.set_buffer(4, st.topk); e.set_buffer(5, st.pfS); e.set_value(6, a); e.dispatch_groups(R / 64, nb, 1, 128); }
          if (k == 2) { e.set_pipeline("attn_pf_softmax"); e.set_buffer(0, st.pfS); e.set_buffer(1, sinks); e.set_buffer(2, st.pfP); e.set_value(3, a); e.dispatch_groups(nb, 1, 1, 1024); }
          if (k == 3) { e.set_pipeline("attn_pf_pv"); e.set_buffer(0, st.pfP); e.set_buffer(1, st.win[L]); e.set_buffer(2, st.chunk_kv); e.set_buffer(3, ckv); e.set_buffer(4, st.topk); e.set_buffer(5, st.rope_yarn); e.set_buffer(6, out); e.set_value(7, a); e.dispatch_groups(512 / 64, nb, 1, 128); }
          e.barrier();
        } });
    double fl = 2.0 * 2 * n * 64 * 512 * (double)R;
    printf("sparse layer %2d (R=%d): total %8.2f ms (%5.1f TFLOP/s) | qprep %6.2f scores %7.2f softmax %6.2f pv %7.2f\n", L, R,
           tt, fl / tt / 1e9, t4[0], t4[1], t4[2], t4[3]);
  }
  return 0;
}
