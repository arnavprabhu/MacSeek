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
struct IdxQArgs { int start, n, actq; };
struct IdxPfArgs { int start, n, ratio, kc0, kc, ld_s, use_mask, mask_words, cand_src, krows, qrows; };
struct MergeArgs { int n, k, cnt, ld_s, base, first; };
struct FinArgs { int k, mode, mask_words; };
struct SparseArgs { int start, nq, chunk_start, ntop, splits, rows_per_split; float scale; };
struct CombArgs { int pos, splits; };

Options Options::from_env() {
  Options o;
  o.actq = env_int("MACSEEK_ACTQ", 1) != 0;
  o.max_pos = (int)env_int("MACSEEK_MAX_POS", o.max_pos);
  o.key_chunk = (int)env_int("MACSEEK_ATTN_KCHUNK", o.key_chunk);
  o.dec_rows_split = (int)env_int("MACSEEK_ATTN_DEC_SPLIT", o.dec_rows_split);
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
  if (nvirt > 0) {
    const int tpt = 4;
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
  for (int kc0 = 0; kc0 < len_max; kc0 += KC) {
    const int c = std::min(KC, len_max - kc0);
    IdxPfArgs a{start, n, r, kc0, c, KC, um ? 1 : 0, s.mask_words, cs ? 1 : 0, s.cap[sl] + kRowPad, (n + 1) / 2 * 64};
    e.set_pipeline("attn_idx_score_pf");
    e.set_buffer(0, s.Qi); e.set_buffer(1, s.wq); e.set_buffer(2, s.ikey[sl]); e.set_buffer(3, s.S);
    e.set_buffer(4, s.cand_mask); e.set_buffer(5, s.BS); e.set_value(6, a);
    e.dispatch_groups((c + 63) / 64, (n + 1) / 2, 1, 128);
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
  if (n == 1) {
    const int rps = std::max(16, s.opt.dec_rows_split / 16 * 16);
    const int splits = (R + rps - 1) / rps;
    SparseArgs a{start, 1, start + 1, ntop, splits, rps, 1.0f / std::sqrt((float)kD)};
    e.set_pipeline("attn_sparse");
    e.set_buffer(0, q); e.set_buffer(1, s.win[L]); e.set_buffer(2, s.win[L]); e.set_buffer(3, ckv);
    e.set_buffer(4, s.topk); e.set_buffer(5, sinks); e.set_buffer(6, tab_of(s, L)); e.set_buffer(7, out);
    e.set_buffer(8, s.part); e.set_buffer(9, s.ml); e.set_value(10, a);
    e.dispatch_groups(4, splits, 1, 128);
    e.barrier();
    CombArgs c{start, splits};
    e.set_pipeline("attn_sparse_combine");
    e.set_buffer(0, s.part); e.set_buffer(1, s.ml); e.set_buffer(2, sinks); e.set_buffer(3, tab_of(s, L));
    e.set_buffer(4, out); e.set_value(5, c);
    e.dispatch_groups(kH, 1, 1, 128);
  } else {
    SparseArgs a{start, n, start, ntop, 1, R, 1.0f / std::sqrt((float)kD)};
    e.set_pipeline("attn_sparse");
    e.set_buffer(0, q); e.set_buffer(1, s.win[L]); e.set_buffer(2, s.chunk_kv); e.set_buffer(3, ckv);
    e.set_buffer(4, s.topk); e.set_buffer(5, sinks); e.set_buffer(6, tab_of(s, L)); e.set_buffer(7, out);
    e.set_buffer(8, s.part); e.set_buffer(9, s.ml); e.set_value(10, a);
    e.dispatch_groups(4, 1, n, 128);
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
