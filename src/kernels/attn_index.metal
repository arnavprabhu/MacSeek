// Attention module: indexer scoring (decode: simdgroup MMA split over many threadgroups; prefill: Metal 4 tensor-op
// matmul2d tiles), exact top-k selection (radix select in one threadgroup), candidate blocks.
#include "attn_common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ------------------------------------------------------------------------------------------------------------------
// threadgroup helpers (1024 threads = 32 simdgroups unless noted)
// exclusive prefix sum over the threadgroup; tmp needs 33 uints. Contains barriers: call from uniform control flow.
inline uint attn_scan_excl(uint v, threadgroup uint* tmp, ushort sg, ushort ln, ushort nsg, thread uint& total) {
  uint pre = simd_prefix_exclusive_sum(v);
  if (ln == 31) tmp[sg] = pre + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint x = ln < nsg ? tmp[ln] : 0u;
    uint px = simd_prefix_exclusive_sum(x);
    uint tot = simd_sum(x);
    tmp[ln] = px;
    if (ln == 0) tmp[32] = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = tmp[32];
  uint r = pre + tmp[sg];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// Exact top-k (largest keys; key 0 = invalid) over N elements read through `src`. Result (unordered) in
// out_key/out_idx[0..*out_cnt). Ties at the threshold resolve to the lowest element index (deterministic).
// TG memory: hist[2048], tmp[40]. Must be called by all 1024 threads.
template <typename Src>
inline void attn_select(thread const Src& src, int N, int k, threadgroup uint* hist, threadgroup uint* tmp,
                        threadgroup uint* out_key, threadgroup int* out_idx, threadgroup atomic_int* out_cnt,
                        ushort tid, ushort sg, ushort ln) {
  constexpr int NT = 1024;
  uint prefix = 0, pmask = 0;
  int rem = k;
  bool take_all = false;
  for (int ps = 0; ps < 3; ps++) {
    const int sh = ps == 0 ? 21 : (ps == 1 ? 10 : 0);
    const int nb = ps == 2 ? 1024 : 2048;
    for (int b = tid; b < 2048; b += NT) hist[b] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < N; i += NT) {
      uint kk = src.key(i);
      if (kk != 0u && (kk & pmask) == prefix)
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(kk >> sh) & uint(nb - 1)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int bpt = nb / NT;  // bins per thread, thread t owns the bpt bins just below nb - t*bpt (descending)
    uint s = 0;
    for (int j = 0; j < bpt; j++) s += hist[nb - 1 - (tid * bpt + j)];
    uint total;
    uint above = attn_scan_excl(s, tmp, sg, ln, NT / 32, total);
    if (ps == 0 && total <= uint(k)) { take_all = true; break; }
    if (above < uint(rem) && uint(rem) <= above + s) {
      uint acc = above;
      for (int j = 0; j < bpt; j++) {
        int b = nb - 1 - (tid * bpt + j);
        uint c = hist[b];
        if (acc + c >= uint(rem)) { tmp[34] = uint(b); tmp[35] = uint(rem) - acc; break; }
        acc += c;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= tmp[34] << sh;
    pmask |= uint(nb - 1) << sh;
    rem = int(tmp[35]);
  }
  if (tid == 0) atomic_store_explicit(out_cnt, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (take_all) {
    for (int i = tid; i < N; i += NT) {
      uint kk = src.key(i);
      if (kk != 0u) { int sl = atomic_fetch_add_explicit(out_cnt, 1, memory_order_relaxed); out_key[sl] = kk; out_idx[sl] = src.idx(i); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
  }
  const uint T = prefix;
  const bool all_ties = int(hist[T & 1023u]) == rem;
  for (int i = tid; i < N; i += NT) {
    uint kk = src.key(i);
    if (kk > T || (all_ties && kk == T)) {
      int sl = atomic_fetch_add_explicit(out_cnt, 1, memory_order_relaxed); out_key[sl] = kk; out_idx[sl] = src.idx(i);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!all_ties) {  // rare: more elements equal to the threshold than needed -> take the lowest indices, in order
    int base = atomic_load_explicit(out_cnt, memory_order_relaxed);
    int taken = 0;
    for (int i0 = 0; i0 < N && taken < rem; i0 += NT) {
      int i = i0 + tid;
      uint f = (i < N && src.key(i) == T) ? 1u : 0u;
      uint tot;
      uint ex = attn_scan_excl(f, tmp, sg, ln, NT / 32, tot);
      if (f && taken + int(ex) < rem) { out_key[base + taken + ex] = T; out_idx[base + taken + ex] = src.idx(i); }
      taken += int(tot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) atomic_store_explicit(out_cnt, base + rem, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ascending bitonic sort of a[0..m) (pads with INT_MAX up to the next power of two <= cap)
inline void attn_sort_int(threadgroup int* a, int m, int cap, ushort tid) {
  int P = 1;
  while (P < m) P <<= 1;
  for (int i = m + tid; i < P; i += 1024) a[i] = INT_MAX;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k2 = 2; k2 <= P; k2 <<= 1)
    for (int j = k2 >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < P; i += 1024) {
        int ixj = i ^ j;
        if (ixj > i) {
          int x = a[i], y = a[ixj];
          bool up = (i & k2) == 0;
          if ((x > y) == up) { a[i] = y; a[ixj] = x; }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// indexer query prep: rope(last 64 of each 128-dim head) + fp4 e8m0/32 sim; thread = (head, 32-dim block)
inline void attn_idx_prep_thread(device const float* qraw, device const float2* tab, int pos, int actq, ushort t,
                                 thread float* v) {
  int h = t >> 2, b = t & 3;
  device const float4* src = (device const float4*)(qraw + h * ATTN_ID + b * 32);
  for (int i = 0; i < 8; i++) { float4 x = src[i]; v[4 * i] = x.x; v[4 * i + 1] = x.y; v[4 * i + 2] = x.z; v[4 * i + 3] = x.w; }
  if (b >= 2) {
    device const float2* cs = tab + (size_t)pos * 32 + (b - 2) * 16;
    for (int i = 0; i < 16; i++) { float2 r = attn_rot(float2(v[2 * i], v[2 * i + 1]), cs[i]); v[2 * i] = r.x; v[2 * i + 1] = r.y; }
  }
  if (actq) {
    float m = 0;
    for (int i = 0; i < 32; i++) m = max(m, fabs(v[i]));
    int e = attn_log2_ceil(max(m, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    float s = attn_pow2(e), is = attn_pow2(-e);
    for (int i = 0; i < 32; i++) v[i] = attn_fp4e8_q(v[i], s, is);
  }
}

struct AttnIdxDecArgs {
  int pos;           // query position
  int len;           // visible compressed rows = (pos+1)/ratio
  int nvirt;         // virtual keys: len (full scan) or ncand*8 (candidate list mode)
  int list_mode;     // 1: virtual key v -> row blist[v/8]*8 + v%8
  int actq;
  int tiles_per_tg;  // 64-key tiles per threadgroup
  int cand_src;      // 1: also write block maxima (blocks of 8, newest pinned to +inf) to bscores
};

// Decode scoring. grid (ceil(ceil(nvirt/64)/tiles_per_tg)), 128 threads (4 simdgroups x 16 keys per tile).
kernel void attn_idx_score_dec(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                               device const float2* tab [[buffer(2)]], device const half* ikey [[buffer(3)]],
                               device const int* blist [[buffer(4)]], device float* scores [[buffer(5)]],
                               device float* bscores [[buffer(6)]], constant AttnIdxDecArgs& a [[buffer(7)]],
                               uint tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup half Qs[ATTN_IH * ATTN_ID];
  threadgroup float ws[ATTN_IH];
  {
    float v[32];
    attn_idx_prep_thread(qraw, tab, a.pos, a.actq, t, v);
    threadgroup half* d = Qs + (t >> 2) * ATTN_ID + (t & 3) * 32;
    for (int i = 0; i < 32; i++) d[i] = half(v[i]);
    if (t < 32) ws[t] = wraw[t] * (1.0f / 64.0f);  // 128^-0.5 * 32^-0.5
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qa[4][16];
  for (int ht = 0; ht < 4; ht++)
    for (int kk = 0; kk < 16; kk++) simdgroup_load(qa[ht][kk], Qs + ht * 8 * ATTN_ID + kk * 8, ATTN_ID);
  const ushort fm = attn_frag_row(ln), fn = attn_frag_col(ln);
  float wl[4];
  for (int ht = 0; ht < 4; ht++) wl[ht] = ws[ht * 8 + fm];
  const int nblocks = (a.len + 7) / 8;
  for (int tt = 0; tt < a.tiles_per_tg; tt++) {
    const int v0 = (int(tg) * a.tiles_per_tg + tt) * 64 + sg * 16;
    if (v0 >= a.nvirt) break;
    int base[2];
    bool ok[2];
    for (int kt = 0; kt < 2; kt++) {
      int v = v0 + kt * 8;
      if (a.list_mode) { int blk = v < a.nvirt ? blist[v >> 3] : -1; ok[kt] = blk >= 0; base[kt] = ok[kt] ? blk * 8 : 0; }
      else { ok[kt] = true; base[kt] = v; }
    }
    simdgroup_float8x8 acc[4][2];
    for (int ht = 0; ht < 4; ht++) for (int kt = 0; kt < 2; kt++) acc[ht][kt] = simdgroup_float8x8(0);
    for (int kk = 0; kk < 16; kk++) {
      for (int kt = 0; kt < 2; kt++) {
        simdgroup_half8x8 kb;
        simdgroup_load(kb, ikey + (size_t)base[kt] * ATTN_ID + kk * 8, ATTN_ID, ulong2(0, 0), true);
        for (int ht = 0; ht < 4; ht++) simdgroup_multiply_accumulate(acc[ht][kt], qa[ht][kk], kb, acc[ht][kt]);
      }
    }
    for (int kt = 0; kt < 2; kt++) {
      float2 s = 0;
      for (int ht = 0; ht < 4; ht++) {
        auto e = acc[ht][kt].thread_elements();
        s.x += max(float(e[0]), 0.0f) * wl[ht];
        s.y += max(float(e[1]), 0.0f) * wl[ht];
      }
      s += simd_shuffle_xor(s, 2);
      s += simd_shuffle_xor(s, 4);
      s += simd_shuffle_xor(s, 16);
      const int va = v0 + kt * 8 + fn;
      const int pa = base[kt] + fn;
      float sx = (ok[kt] && va < a.nvirt && pa < a.len) ? s.x : -INFINITY;
      float sy = (ok[kt] && va + 1 < a.nvirt && pa + 1 < a.len) ? s.y : -INFINITY;
      if ((ln & 0x16) == 0) {
        if (va < a.nvirt) scores[va] = sx;
        if (va + 1 < a.nvirt) scores[va + 1] = sy;
      }
      if (a.cand_src) {
        float bm = max(sx, sy);
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        int b = (v0 + kt * 8) >> 3;
        if (b == (a.len - 1) / 8) bm = INFINITY;
        if (ln == 0 && b < nblocks) bscores[b] = bm;
      }
    }
  }
}

struct AttnSrcScores {
  device const float* s;
  uint key(int i) const { return attn_fkey(s[i]); }
  int idx(int i) const { return i; }
};

struct AttnTopkArgs { int n; int k; int list_mode; };
// Decode top-k: one threadgroup (1024 threads). out[k]: selected rows sorted ascending, -1 padded.
// list_mode maps virtual index v -> blist[v/8]*8 + v%8 (sorted list => order preserved).
kernel void attn_topk_dec(device const float* scores [[buffer(0)]], device const int* blist [[buffer(1)]],
                          device int* out [[buffer(2)]], constant AttnTopkArgs& a [[buffer(3)]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  AttnSrcScores src{scores};
  attn_select(src, a.n, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  attn_sort_int(oidx, m, 2048, tid);
  for (int i = tid; i < a.k; i += 1024) {
    int v = i < m ? oidx[i] : -1;
    if (v >= 0 && a.list_mode) v = blist[v >> 3] * 8 + (v & 7);
    out[i] = v;
  }
}

// ------------------------------------------------------------------------------------------------------------------
// Prefill
struct AttnIdxQArgs { int start; int n; int actq; };
// grid (n), 128 threads -> Qi half [n][32][128], wq f32 [n][32] (scaled)
kernel void attn_idx_qprep(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                           device const float2* tab [[buffer(2)]], device half* Qi [[buffer(3)]],
                           device float* wq [[buffer(4)]], constant AttnIdxQArgs& a [[buffer(5)]],
                           uint q [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  float v[32];
  attn_idx_prep_thread(qraw + (size_t)q * ATTN_IH * ATTN_ID, tab, a.start + int(q), a.actq, t, v);
  device half4* d = (device half4*)(Qi + (size_t)q * ATTN_IH * ATTN_ID + (t >> 2) * ATTN_ID + (t & 3) * 32);
  for (int i = 0; i < 8; i++) d[i] = half4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  if (t < 32) wq[q * ATTN_IH + t] = wraw[q * ATTN_IH + t] * (1.0f / 64.0f);
}

struct AttnIdxPfArgs {
  int start, n, ratio;
  int kc0, kc;       // key chunk [kc0, kc0+kc)
  int ld_s;          // row stride of S (and 8*row stride of BS)
  int use_mask, mask_words;
  int cand_src;
  int krows, qrows;  // tensor extents (allocated rows) of ikey and Qi
};
// grid (ceil(kc/64), ceil(n/2)), 128 threads. S[q][t-kc0] = sum_h relu(q_h.k_t) w_h (or -inf if invisible/masked).
kernel void attn_idx_score_pf(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],
                              device half* ikey [[buffer(2)]], device float* S [[buffer(3)]],
                              device const uint* mask [[buffer(4)]], device float* BS [[buffer(5)]],
                              constant AttnIdxPfArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                              ushort t [[thread_index_in_threadgroup]]) {
  threadgroup float T[64 * 64];
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, a.krows));
  constexpr auto desc = matmul2d_descriptor(64, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, 64>(0, int(tg.y) * 64);
  auto mB = tB.template slice<ATTN_ID, 64>(0, a.kc0 + int(tg.x) * 64);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mB, cT);
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      int col = id[0], row = id[1];
      int q = int(tg.y) * 2 + row / 32;
      float w = q < a.n ? wq[q * ATTN_IH + (row & 31)] : 0.0f;
      T[row * 64 + col] = max(cT[i], 0.0f) * w;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ql = t >> 6, key = t & 63;
  const int q = int(tg.y) * 2 + ql;
  float s = 0;
  for (int h = 0; h < 32; h++) s += T[(ql * 32 + h) * 64 + key];
  const int tl = int(tg.x) * 64 + key, tabs = a.kc0 + tl;
  const int len = (a.start + q + 1) / a.ratio;
  bool vis = q < a.n && tabs < len && tl < a.kc;
  if (vis && a.use_mask) vis = (mask[(size_t)q * a.mask_words + (tabs >> 8)] >> ((tabs >> 3) & 31)) & 1u;
  float sc = vis ? s : -INFINITY;
  const bool live = q < a.n && tl < a.kc;
  if (live) S[(size_t)q * a.ld_s + tl] = sc;
  if (a.cand_src) {
    float bm = sc;
    bm = max(bm, simd_shuffle_xor(bm, 1)); bm = max(bm, simd_shuffle_xor(bm, 2)); bm = max(bm, simd_shuffle_xor(bm, 4));
    if (len > 0 && (tabs >> 3) == (len - 1) / 8) bm = INFINITY;
    if (live && (key & 7) == 0) BS[(size_t)q * (a.ld_s / 8) + (tl >> 3)] = bm;
  }
}

struct AttnSrcMerge {
  device const uint* rk;
  device const int* ri;
  int rc;
  device const float* s;
  int base;
  uint key(int i) const { return i < rc ? rk[i] : attn_fkey(s[i - rc]); }
  int idx(int i) const { return i < rc ? ri[i] : base + (i - rc); }
};
struct AttnMergeArgs {
  int n, k;
  int cnt;    // valid elements in this chunk row
  int ld_s;   // S row stride
  int base;   // element index of S[q][0]
  int first;  // 1: running list empty
};
// Running top-k per query: R[q] <- topk(R[q] U S[q][0..cnt)). grid (n), 1024 threads.
kernel void attn_topk_merge(device const float* S [[buffer(0)]], device uint* Rk [[buffer(1)]], device int* Ri [[buffer(2)]],
                            device int* Rc [[buffer(3)]], constant AttnMergeArgs& a [[buffer(4)]],
                            uint q [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  int rc = a.first ? 0 : Rc[q];
  AttnSrcMerge src{Rk + (size_t)q * a.k, Ri + (size_t)q * a.k, rc, S + (size_t)q * a.ld_s, a.base};
  attn_select(src, rc + a.cnt, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  for (int i = tid; i < m; i += 1024) { Rk[(size_t)q * a.k + i] = okey[i]; Ri[(size_t)q * a.k + i] = oidx[i]; }
  if (tid == 0) Rc[q] = m;
}

struct AttnFinArgs { int k; int mode; int mask_words; };  // mode 0: sorted index list [n][k] (-1 pad); 1: block bitmap
kernel void attn_topk_finalize(device const int* Ri [[buffer(0)]], device const int* Rc [[buffer(1)]],
                               device int* out [[buffer(2)]], device uint* mask [[buffer(3)]],
                               constant AttnFinArgs& a [[buffer(4)]], uint q [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup int arr[2048];
  int m = Rc[q];
  if (a.mode == 0) {
    for (int i = tid; i < m; i += 1024) arr[i] = Ri[(size_t)q * a.k + i];
    attn_sort_int(arr, m, 2048, tid);
    for (int i = tid; i < a.k; i += 1024) out[(size_t)q * a.k + i] = i < m ? arr[i] : -1;
  } else {
    threadgroup atomic_uint* bits = (threadgroup atomic_uint*)arr;  // up to 2048 words = 65536 blocks
    for (int i = tid; i < a.mask_words; i += 1024) atomic_store_explicit(&bits[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < m; i += 1024) {
      int b = Ri[(size_t)q * a.k + i];
      atomic_fetch_or_explicit(&bits[b >> 5], 1u << (b & 31), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < a.mask_words; i += 1024)
      mask[(size_t)q * a.mask_words + i] = atomic_load_explicit(&bits[i], memory_order_relaxed);
  }
}
