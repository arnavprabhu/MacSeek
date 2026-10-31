// Attention module host side: state allocation, rope tables, encode functions.
#include "kernels/attn.hpp"
#include <dispatch/dispatch.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "core/env.hpp"

namespace ms::attn {

// ---- argument structs (must match the .metal side) ----
struct KvArgs { int start, n, ring, actq; };
struct CompArgs { int start, n, ratio, j0, actq; };
struct StateArgs { int start, first, ratio; };
struct CommitArgs { int start, first; };
struct QuantArgs { uint32_t n; int mode; };
struct IdxDecArgs { int pos, len, nvirt, list_mode, actq, tiles_per_tg, cand_src; };
struct TopkArgs { int n, k, list_mode; };
struct IdxDec2Args { int pos, len, nvirt, actq, tiles_per_tg, cand_src, krows; };
struct CompactArgs { int n, k, cap, per_tg; };
struct FinalArgs { int n, k, list_mode; };
struct IdxQArgs { int start, n, actq; };
struct IdxPfArgs { int start, n, ratio, kc0, kc, ld_s, use_mask, mask_words, cand_src, krows, qrows; };
struct MergeArgs { int n, k, cnt, ld_s, base, first; };
struct UnionArgs { int n, mask_words, cap; };
struct IdxPfListArgs { int start, n, ratio, kc0, kc, ld_s, mask_words, cap, qrows; };
struct MergeListArgs { int n, k, kc, ld_s, base, first, cap; };
constexpr int kUnionCap = 2 * kCandBlocks;
struct FinArgs { int k, mode, mask_words; };
struct SparseArgs { int start, nq, chunk_start, ntop, splits, rows_per_split; float scale; };
struct CombArgs { int pos, splits; };
struct PfArgs { int start, nq, q0, chunk_start, ntop, R; float scale; };

Options Options::from_env() {
  Options o;
  o.actq = env_int("MACSEEK_ACTQ", 0) != 0;  // default 0: matches the llama.cpp oracle best (KL 0.003 vs 0.013)
  o.max_pos = (int)env_int("MACSEEK_MAX_POS", o.max_pos);
  o.key_chunk = (int)env_int("MACSEEK_ATTN_KCHUNK", o.key_chunk);
  o.dec_rows_split = (int)env_int("MACSEEK_ATTN_DEC_SPLIT", o.dec_rows_split);
  o.dec_heads_tg = (int)env_int("MACSEEK_ATTN_DEC_HEADS", o.dec_heads_tg);
  o.pf_heads_tg = (int)env_int("MACSEEK_ATTN_PF_HEADS", o.pf_heads_tg);
  o.dec_fast_index = env_int("MACSEEK_ATTN_DEC_FAST", 1) != 0;
  o.pf_cand_union = env_int("MACSEEK_ATTN_PF_UNION", 1) != 0;
  o.pf_mpp = env_int("MACSEEK_ATTN_PF_MPP", 1) != 0;
  o.dec_mpp = env_int("MACSEEK_ATTN_DEC_MPP", 0) != 0;
  o.pf_qblock = (int)env_int("MACSEEK_ATTN_PF_QBLOCK", o.pf_qblock);
  return o;
}

void make_rope_table(float* dst, int max_pos, bool yarn) {
  const int dim = kRD;
  const double base = yarn ? 160000.0 : 10000.0;
  float f[32];
  for (int i = 0; i < 32; i++) f[i] = (float)(1.0 / std::pow(base, (double)(float)(2 * i) / dim));
  if (yarn) {
    const double factor = 16, orig = 65536, beta_fast = 32, beta_slow = 1;
    auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(base)); };
    double low = std::max(std::floor(corr(beta_fast)), 0.0);
    double high = std::min(std::ceil(corr(beta_slow)), (double)dim - 1);
    for (int i = 0; i < 32; i++) {
      float ramp = std::clamp((float)((i - low) / std::max(high - low, 1e-3)), 0.0f, 1.0f);
      float smooth = 1.0f - ramp;
      f[i] = f[i] / (float)factor * (1.0f - smooth) + f[i] * smooth;
    }
  }
  const int blk = 4096;
  const float* fp = f;
  dispatch_apply((max_pos + blk - 1) / blk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
    int p1 = std::min(max_pos, (int)(b + 1) * blk);
    for (int p = (int)b * blk; p < p1; p++)
      for (int i = 0; i < 32; i++) {
        float a = (float)p * fp[i];
        dst[(size_t)p * 64 + 2 * i] = (float)std::cos((double)a);
        dst[(size_t)p * 64 + 2 * i + 1] = (float)std::sin((double)a);
      }
  });
}

void init(State& s, const Options& o) {
  auto& d = Device::get();
  s.opt = o;
  s.rope_plain = d.alloc((size_t)o.max_pos * 64 * 4, "attn.rope_plain");
  s.rope_yarn = d.alloc((size_t)o.max_pos * 64 * 4, "attn.rope_yarn");
  make_rope_table((float*)s.rope_plain.cpu(), o.max_pos, false);
  make_rope_table((float*)s.rope_yarn.cpu(), o.max_pos, true);
  for (int L = 0; L < kLayers; L++) s.win[L] = d.alloc((size_t)kWin * kD * 2, "attn.win");
  const int srcs[4] = {2, 8, 14, 20};
  for (int i = 0; i < 4; i++) {
    int r = ratio_of(srcs[i]);
    s.cap[i] = o.max_pos / r;
    s.ckv[i] = d.alloc((size_t)(s.cap[i] + kRowPad) * kD * 2, "attn.ckv");
    s.ikey[i] = d.alloc((size_t)(s.cap[i] + kRowPad) * kID * 2, "attn.ikey");
    s.cst[i] = d.alloc((size_t)2 * r * kD * 4, "attn.cstate");
  }
  const int nmax = o.max_chunk, nq2 = (nmax + 1) / 2 * 2;
  const int max_blocks = (o.max_pos + kCandBS - 1) / kCandBS;
  s.mask_words = (max_blocks + 31) / 32;
  s.chunk_kv = d.alloc((size_t)nmax * kD * 2, "attn.chunk_kv");
  const int max_splits = (kWin + kTopk + 15) / 16;
  s.part = d.alloc((size_t)max_splits * kH * kD * 4, "attn.part");
  s.ml = d.alloc((size_t)max_splits * kH * 8, "attn.ml");
  s.dec_scores = d.alloc((size_t)(o.max_pos + kRowPad) * 4, "attn.dec_scores");
  s.dec_bscores = d.alloc((size_t)(max_blocks + kRowPad) * 4, "attn.dec_bscores");
  s.topk = d.alloc((size_t)nmax * kTopk * 4, "attn.topk");
  s.dec_hist = d.alloc(2 * 4096 * 4, "attn.dec_hist");
  s.dec_ctl = d.alloc(2 * 4 * 4, "attn.dec_ctl");
  s.dec_sel = d.alloc(2 * 2048 * 4, "attn.dec_sel");
  s.dec_cand = d.alloc((size_t)2 * o.dec_cand_cap * 8, "attn.dec_cand");
  s.cand_list = d.alloc((size_t)kCandBlocks * 4, "attn.cand_list");
  s.cand_mask = d.alloc((size_t)nmax * s.mask_words * 4, "attn.cand_mask");
  s.Qi = d.alloc((size_t)nq2 * kIH * kID * 2, "attn.Qi");
  s.wq = d.alloc((size_t)nq2 * kIH * 4, "attn.wq");
  s.S = d.alloc((size_t)nmax * o.key_chunk * 4, "attn.S");
  s.BS = d.alloc((size_t)nmax * (o.key_chunk / 8) * 4, "attn.BS");
  s.Rk = d.alloc((size_t)nmax * kTopk * 4, "attn.Rk");
  s.Ri = d.alloc((size_t)nmax * kTopk * 4, "attn.Ri");
  s.Rc = d.alloc((size_t)nmax * 4, "attn.Rc");
  s.Bk = d.alloc((size_t)nmax * kCandBlocks * 4, "attn.Bk");
  s.Bi = d.alloc((size_t)nmax * kCandBlocks * 4, "attn.Bi");
  s.Bc = d.alloc((size_t)nmax * 4, "attn.Bc");
  if ((o.pf_mpp && nmax > 1) || o.dec_mpp) {
    const size_t qb = std::min(o.pf_qblock, nmax);
    s.pfQ = d.alloc(qb * kH * kD * 2, "attn.pfQ");
    s.pfS = d.alloc(qb * kH * (kWin + kTopk) * 4, "attn.pfS");
    s.pfP = d.alloc(qb * kH * (kWin + kTopk) * 2, "attn.pfP");
  }
  s.ulist = d.alloc((size_t)(nq2 / 2) * kUnionCap * 4, "attn.ulist");
  s.ucount = d.alloc((size_t)(nq2 / 2) * 4, "attn.ucount");
}

std::vector<StateSpan> persistent_state(const State& s, int pos) {
  std::vector<StateSpan> v;
  for (int L = 0; L < kLayers; L++) v.push_back({s.win[L], s.win[L].size});
  const int srcs[4] = {2, 8, 14, 20};
  for (int i = 0; i < 4; i++) {
    size_t rows = (size_t)pos / ratio_of(srcs[i]);
    v.push_back({s.ckv[i], rows * kD * 2});
    v.push_back({s.ikey[i], rows * kID * 2});
    v.push_back({s.cst[i], s.cst[i].size});
  }
  return v;
}

void reset(State& s) {
  for (int L = 0; L < kLayers; L++) memset(s.win[L].cpu(), 0, s.win[L].size);
  for (int i = 0; i < 4; i++) {
    memset(s.ckv[i].cpu(), 0, s.ckv[i].size);
    memset(s.ikey[i].cpu(), 0, s.ikey[i].size);
    memset(s.cst[i].cpu(), 0, s.cst[i].size);
  }
}

static const Buffer& tab_of(const State& s, int L) { return ratio_of(L) ? s.rope_yarn : s.rope_plain; }

void encode_kv_window(Enc& e, State& s, int L, const Buffer& kv, const Buffer& norm_w, int start, int n) {
  KvArgs a{start, n, n == 1 ? 1 : 0, s.opt.actq ? 1 : 0};
  e.set_pipeline("attn_kv_window");
  e.set_buffer(0, kv); e.set_buffer(1, norm_w); e.set_buffer(2, tab_of(s, L));
  e.set_buffer(3, n == 1 ? s.win[L] : s.chunk_kv); e.set_value(4, a);
  e.dispatch_groups(n, 1, 1, 128);
}

// compressor for kv-source layer L over tokens start..start+n-1 (no trailing barrier; see encode_layer)
static void compress_groups(Enc& e, State& s, int L, const LayerIO& io, int start, int n, bool& wrote_state) {
  const int r = ratio_of(L), sl = src_slot(L);
  const int j0 = start / r, j1 = (start + n) / r;
  wrote_state = false;
  if (j1 > j0) {
    CompArgs a{start, n, r, j0, s.opt.actq ? 1 : 0};
    e.set_pipeline("attn_compress");
    e.set_buffer(0, io.c_kv); e.set_buffer(1, r > 1 ? io.c_gate : io.c_kv); e.set_buffer(2, s.cst[sl]);
    e.set_buffer(3, io.c_norm); e.set_buffer(4, io.idx_wk); e.set_buffer(5, io.idx_knorm); e.set_buffer(6, s.rope_yarn);
    e.set_buffer(7, s.ckv[sl]); e.set_buffer(8, s.ikey[sl]); e.set_value(9, a);
    e.dispatch_groups(j1 - j0, 1, 1, 256);
  }
  const int first = std::max(start, j1 * r);  // trailing tokens of an incomplete group
  if (r > 1 && first < start + n) {
    if (j1 > j0) e.barrier();  // the group kernel may still read the state slots
    StateArgs a{start, first, r};
    e.set_pipeline("attn_compress_state");
    e.set_buffer(0, io.c_kv); e.set_buffer(1, io.c_gate); e.set_buffer(2, s.cst[sl]); e.set_value(3, a);
    e.dispatch_groups(start + n - first, 1, 1, 128);
    wrote_state = true;
  }
}

void encode_compress(Enc& e, State& s, int L, const LayerIO& io, int start, int n) {
  bool ws;
  compress_groups(e, s, L, io, start, n, ws);
  e.barrier();
}

void encode_index_decode(Enc& e, State& s, int L, const Buffer& iq, const Buffer& iw, int pos) {
  const int r = ratio_of(L), sl = src_slot(kv_source_of(L));
  const int len = (pos + 1) / r;
  const bool lm = uses_cand(L), cs = is_cand_source(L);
  const int nblocks = (len + kCandBS - 1) / kCandBS;
  const int nvirt = lm ? std::min(nblocks, kCandBlocks) * kCandBS : len;
  if (s.opt.dec_fast_index && nvirt > 0) {
    const int tpt = lm ? 4 : 16, tiles = (nvirt + 63) / 64;
    IdxDec2Args a{pos, len, nvirt, s.opt.actq ? 1 : 0, tpt, cs ? 1 : 0, s.cap[sl] + kRowPad};
    e.set_pipeline(lm ? "attn_idx_score_dec2_list" : "attn_idx_score_dec2");
    e.set_buffer(0, iq); e.set_buffer(1, iw); e.set_buffer(2, tab_of(s, L)); e.set_buffer(3, s.ikey[sl]);
    e.set_buffer(4, s.dec_scores);
    if (lm) e.set_buffer(5, s.cand_list); else e.set_buffer(5, s.dec_bscores);
    e.set_buffer(6, s.dec_hist.sub(0, 4096 * 4));
    e.set_buffer(7, s.dec_hist.sub(4096 * 4, 4096 * 4)); e.set_value(8, a);
    e.dispatch_groups((tiles + tpt - 1) / tpt, 1, 1, 128);
    e.barrier();
    const int per_tg = 4096;
    auto compact = [&](int p, const Buffer& sc, int n, int k) {
      CompactArgs c{n, k, s.opt.dec_cand_cap, per_tg};
      e.set_pipeline("attn_topk_compact");
      e.set_buffer(0, sc); e.set_buffer(1, s.dec_hist.sub((size_t)p * 4096 * 4, 4096 * 4));
      e.set_buffer(2, s.dec_ctl.sub((size_t)p * 16, 16)); e.set_buffer(3, s.dec_sel.sub((size_t)p * 2048 * 4, 2048 * 4));
      e.set_buffer(4, s.dec_cand.sub((size_t)p * s.opt.dec_cand_cap * 8, (size_t)s.opt.dec_cand_cap * 8)); e.set_value(5, c);
      e.dispatch_groups((n + per_tg - 1) / per_tg, 1, 1, 256);
    };
    auto fin = [&](int p, const Buffer& sc, int n, int k, const Buffer& out) {
      FinalArgs f{n, k, (p == 0 && lm) ? 1 : 0};
      e.set_pipeline("attn_topk_final");
      e.set_buffer(0, sc); e.set_buffer(1, s.dec_hist.sub((size_t)p * 4096 * 4, 4096 * 4));
      e.set_buffer(2, s.dec_ctl.sub((size_t)p * 16, 16)); e.set_buffer(3, s.dec_sel.sub((size_t)p * 2048 * 4, 2048 * 4));
      e.set_buffer(4, s.dec_cand.sub((size_t)p * s.opt.dec_cand_cap * 8, (size_t)s.opt.dec_cand_cap * 8));
      e.set_buffer(5, s.cand_list); e.set_buffer(6, out); e.set_value(7, f);
      e.dispatch_groups(1, 1, 1, 1024);
    };
    compact(0, s.dec_scores, nvirt, kTopk);
    if (cs) compact(1, s.dec_bscores, nblocks, kCandBlocks);
    e.barrier();
    fin(0, s.dec_scores, nvirt, kTopk, s.topk);
    if (cs) fin(1, s.dec_bscores, nblocks, kCandBlocks, s.cand_list);
    e.barrier();
    return;
  }
  if (nvirt > 0) {
    const int tpt = 16;
    IdxDecArgs a{pos, len, nvirt, lm ? 1 : 0, s.opt.actq ? 1 : 0, tpt, cs ? 1 : 0};
    int tiles = (nvirt + 63) / 64;
    e.set_pipeline("attn_idx_score_dec");
    e.set_buffer(0, iq); e.set_buffer(1, iw); e.set_buffer(2, tab_of(s, L)); e.set_buffer(3, s.ikey[sl]);
    e.set_buffer(4, s.cand_list); e.set_buffer(5, s.dec_scores); e.set_buffer(6, s.dec_bscores); e.set_value(7, a);
    e.dispatch_groups((tiles + tpt - 1) / tpt, 1, 1, 128);
    e.barrier();
  }
  {
    TopkArgs a{nvirt, kTopk, lm ? 1 : 0};
    e.set_pipeline("attn_topk_dec");
    e.set_buffer(0, s.dec_scores); e.set_buffer(1, s.cand_list); e.set_buffer(2, s.topk); e.set_value(3, a);
    e.dispatch_groups(1, 1, 1, 1024);
  }
  if (cs) {
    TopkArgs a{nblocks, kCandBlocks, 0};
    e.set_pipeline("attn_topk_dec");
    e.set_buffer(0, s.dec_bscores); e.set_buffer(1, s.cand_list); e.set_buffer(2, s.cand_list); e.set_value(3, a);
    e.dispatch_groups(1, 1, 1, 1024);
  }
  e.barrier();
}

void encode_index_prefill(Enc& e, State& s, int L, const Buffer& iq, const Buffer& iw, int start, int n) {
  const int r = ratio_of(L), sl = src_slot(kv_source_of(L));
  const bool um = uses_cand(L), cs = is_cand_source(L);
  const int len_max = (start + n) / r;
  const int KC = s.opt.key_chunk;
  {
    IdxQArgs a{start, n, s.opt.actq ? 1 : 0};
    e.set_pipeline("attn_idx_qprep");
    e.set_buffer(0, iq); e.set_buffer(1, iw); e.set_buffer(2, tab_of(s, L)); e.set_buffer(3, s.Qi);
    e.set_buffer(4, s.wq); e.set_value(5, a);
    e.dispatch_groups(n, 1, 1, 128);
    e.barrier();
  }
  auto merge = [&](const Buffer& src, const Buffer& k_, const Buffer& i_, const Buffer& c_, int k, int cnt, int ld, int base, int first) {
    MergeArgs a{n, k, cnt, ld, base, first};
    e.set_pipeline("attn_topk_merge");
    e.set_buffer(0, src); e.set_buffer(1, k_); e.set_buffer(2, i_); e.set_buffer(3, c_); e.set_value(4, a);
    e.dispatch_groups(n, 1, 1, 1024);
  };
  if (len_max == 0) {
    merge(s.S, s.Rk, s.Ri, s.Rc, kTopk, 0, KC, 0, 1);
    if (cs) merge(s.BS, s.Bk, s.Bi, s.Bc, kCandBlocks, 0, KC / 8, 0, 1);
    e.barrier();
  }
  if (um && s.opt.pf_cand_union && len_max > 0) {
    const int ntiles = (n + 1) / 2;
    {
      UnionArgs a{n, s.mask_words, kUnionCap};
      e.set_pipeline("attn_cand_union");
      e.set_buffer(0, s.cand_mask); e.set_buffer(1, s.ulist); e.set_buffer(2, s.ucount); e.set_value(3, a);
      e.dispatch_groups(ntiles, 1, 1, 1024);
      e.barrier();
    }
    const int nv_max = std::min(kUnionCap, (len_max + kCandBS - 1) / kCandBS) * kCandBS;
    for (int kc0 = 0; kc0 < nv_max; kc0 += KC) {
      const int c = std::min(KC, nv_max - kc0);
      IdxPfListArgs a{start, n, r, kc0, c, KC, s.mask_words, kUnionCap, ntiles * 64};
      e.set_pipeline("attn_idx_score_pf_list");
      e.set_buffer(0, s.Qi); e.set_buffer(1, s.wq); e.set_buffer(2, s.ikey[sl]); e.set_buffer(3, s.S);
      e.set_buffer(4, s.cand_mask); e.set_buffer(5, s.ulist); e.set_buffer(6, s.ucount); e.set_value(7, a);
      e.dispatch_groups((c + 63) / 64, ntiles, 1, 128);
      e.barrier();
      MergeListArgs m{n, kTopk, c, KC, kc0, kc0 == 0, kUnionCap};
      e.set_pipeline("attn_topk_merge_list");
      e.set_buffer(0, s.S); e.set_buffer(1, s.Rk); e.set_buffer(2, s.Ri); e.set_buffer(3, s.Rc);
      e.set_buffer(4, s.ulist); e.set_buffer(5, s.ucount); e.set_value(6, m);
      e.dispatch_groups(n, 1, 1, 1024);
      e.barrier();
    }
  }
  for (int kc0 = 0; kc0 < len_max && !(um && s.opt.pf_cand_union); kc0 += KC) {
    const int c = std::min(KC, len_max - kc0);
    IdxPfArgs a{start, n, r, kc0, c, KC, um ? 1 : 0, s.mask_words, cs ? 1 : 0, s.cap[sl] + kRowPad, (n + 1) / 2 * 64};
    e.set_pipeline("attn_idx_score_pf");
    e.set_buffer(0, s.Qi); e.set_buffer(1, s.wq); e.set_buffer(2, s.ikey[sl]); e.set_buffer(3, s.S);
    e.set_buffer(4, s.cand_mask); e.set_buffer(5, s.BS); e.set_value(6, a);
    e.dispatch_groups((c + 127) / 128, (n + 1) / 2, 1, 128);
    e.barrier();
    merge(s.S, s.Rk, s.Ri, s.Rc, kTopk, c, KC, kc0, kc0 == 0);
    if (cs) merge(s.BS, s.Bk, s.Bi, s.Bc, kCandBlocks, (c + 7) / 8, KC / 8, kc0 / 8, kc0 == 0);
    e.barrier();
  }
  {
    FinArgs a{kTopk, 0, s.mask_words};
    e.set_pipeline("attn_topk_finalize");
    e.set_buffer(0, s.Ri); e.set_buffer(1, s.Rc); e.set_buffer(2, s.topk); e.set_buffer(3, s.cand_mask); e.set_value(4, a);
    e.dispatch_groups(n, 1, 1, 1024);
  }
  if (cs) {
    FinArgs a{kCandBlocks, 1, s.mask_words};
    e.set_pipeline("attn_topk_finalize");
    e.set_buffer(0, s.Bi); e.set_buffer(1, s.Bc); e.set_buffer(2, s.topk); e.set_buffer(3, s.cand_mask); e.set_value(4, a);
    e.dispatch_groups(n, 1, 1, 1024);
  }
  e.barrier();
}

void encode_sparse(Enc& e, State& s, int L, const Buffer& q, const Buffer& sinks, const Buffer& out, int start, int n) {
  const int r = ratio_of(L);
  const int ntop = r ? kTopk : 0;
  const int R = kWin + ntop;
  const Buffer& ckv = r ? s.ckv[src_slot(kv_source_of(L))] : s.win[L];
  if (n == 1 && !(s.opt.dec_mpp && s.pfQ)) {
    const int rps = std::max(16, s.opt.dec_rows_split / 16 * 16);
    const int splits = (R + rps - 1) / rps;
    SparseArgs a{start, 1, start + 1, ntop, splits, rps, 1.0f / std::sqrt((float)kD)};
    const int hg = s.opt.dec_heads_tg == 16 ? 16 : 8;
    e.set_pipeline(hg == 16 ? "attn_sparse_h16" : "attn_sparse_h8");
    e.set_buffer(0, q); e.set_buffer(1, s.win[L]); e.set_buffer(2, s.win[L]); e.set_buffer(3, ckv);
    e.set_buffer(4, s.topk); e.set_buffer(5, sinks); e.set_buffer(6, tab_of(s, L)); e.set_buffer(7, out);
    e.set_buffer(8, s.part); e.set_buffer(9, s.ml); e.set_value(10, a);
    e.dispatch_groups(kH / hg, splits, 1, hg == 16 ? 256 : 128);
    e.barrier();
    CombArgs c{start, splits};
    e.set_pipeline("attn_sparse_combine");
    e.set_buffer(0, s.part); e.set_buffer(1, s.ml); e.set_buffer(2, sinks); e.set_buffer(3, tab_of(s, L));
    e.set_buffer(4, out); e.set_value(5, c);
    e.dispatch_groups(kH, 1, 1, 128);
  } else if (s.opt.pf_mpp && s.pfQ) {
    const int QB = std::min(s.opt.pf_qblock, s.opt.max_chunk);
    for (int q0 = 0; q0 < n; q0 += QB) {
      const int nb = std::min(QB, n - q0);
      PfArgs a{start + q0, nb, q0, start, ntop, R, 1.0f / std::sqrt((float)kD)};
      e.set_pipeline("attn_pf_qprep");
      e.set_buffer(0, q); e.set_buffer(1, tab_of(s, L)); e.set_buffer(2, s.pfQ); e.set_value(3, a);
      e.dispatch_groups(nb, 1, 1, 128);
      e.barrier();
      e.set_pipeline("attn_pf_scores");
      e.set_buffer(0, s.pfQ); e.set_buffer(1, s.win[L]); e.set_buffer(2, s.chunk_kv); e.set_buffer(3, ckv);
      e.set_buffer(4, s.topk); e.set_buffer(5, s.pfS); e.set_value(6, a);
      e.dispatch_groups(R / 64, nb, 1, 128);
      e.barrier();
      e.set_pipeline("attn_pf_softmax");
      e.set_buffer(0, s.pfS); e.set_buffer(1, sinks); e.set_buffer(2, s.pfP); e.set_value(3, a);
      e.dispatch_groups(nb, 1, 1, 1024);
      e.barrier();
      e.set_pipeline("attn_pf_pv");
      e.set_buffer(0, s.pfP); e.set_buffer(1, s.win[L]); e.set_buffer(2, s.chunk_kv); e.set_buffer(3, ckv);
      e.set_buffer(4, s.topk); e.set_buffer(5, tab_of(s, L)); e.set_buffer(6, out); e.set_value(7, a);
      e.dispatch_groups(kD / 64, nb, 1, 128);
      e.barrier();
    }
  } else {
    SparseArgs a{start, n, start, ntop, 1, R, 1.0f / std::sqrt((float)kD)};
    const int hg = s.opt.pf_heads_tg == 16 ? 16 : 8;
    e.set_pipeline(hg == 16 ? "attn_sparse_h16" : "attn_sparse_h8");
    e.set_buffer(0, q); e.set_buffer(1, s.win[L]); e.set_buffer(2, s.chunk_kv); e.set_buffer(3, ckv);
    e.set_buffer(4, s.topk); e.set_buffer(5, sinks); e.set_buffer(6, tab_of(s, L)); e.set_buffer(7, out);
    e.set_buffer(8, s.part); e.set_buffer(9, s.ml); e.set_value(10, a);
    e.dispatch_groups(kH / hg, 1, n, hg == 16 ? 256 : 128);
  }
  e.barrier();
}

void encode_window_commit(Enc& e, State& s, int L, int start, int n) {
  const int first = std::max(0, n - kWin);
  CommitArgs a{start, first};
  e.set_pipeline("attn_window_commit");
  e.set_buffer(0, s.chunk_kv); e.set_buffer(1, s.win[L]); e.set_value(2, a);
  e.dispatch_groups(n - first, 1, 1, 64);
  e.barrier();
}

void encode_quant_sim(Enc& e, const Buffer& x, uint32_t n, int mode) {
  QuantArgs a{n, mode};
  e.set_pipeline("attn_quant_sim");
  e.set_buffer(0, x); e.set_value(1, a);
  uint32_t threads = (n / 4 + 255) / 256 * 256;
  e.dispatch_groups(threads / 256, 1, 1, 256);
}

void encode_layer(Enc& e, State& s, const LayerIO& io, int start, int n) {
  const int L = io.layer;
  if (L < 0 || L >= kLayers || n < 1 || n > s.opt.max_chunk || start < 0 || start + n > s.opt.max_pos) {
    fprintf(stderr, "attn::encode_layer: bad call layer=%d start=%d n=%d (max_chunk %d, max_pos %d)\n", L, start, n,
            s.opt.max_chunk, s.opt.max_pos);
    abort();
  }
  encode_kv_window(e, s, L, io.kv, io.kv_norm, start, n);
  if (is_kv_source(L)) { bool ws; compress_groups(e, s, L, io, start, n, ws); }
  e.barrier();
  if (is_index_source(L)) {
    if (n == 1) encode_index_decode(e, s, L, io.idx_q, io.idx_w, start);
    else encode_index_prefill(e, s, L, io.idx_q, io.idx_w, start, n);
  }
  encode_sparse(e, s, L, io.q, io.sinks, io.out, start, n);
  if (n > 1) encode_window_commit(e, s, L, start, n);
}

}  // namespace ms::attn
