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
// grid (ceil(kc/128), ceil(n/2)), 128 threads. S[q][t-kc0] = sum_h relu(q_h.k_t) w_h (or -inf if invisible/masked).
// Tile: QT queries (QT*32 rows of Qi) x NK keys, tensor-op matmul, relu*w epilogue through threadgroup memory.
template <int QT, int NK>
inline void attn_idx_score_pf_impl(device half* Qi, device const float* wq, device half* ikey, device float* S,
                                   device const uint* mask, device float* BS, constant AttnIdxPfArgs& a, uint2 tg,
                                   ushort t, threadgroup float* T) {
  constexpr int M = QT * 32;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, a.krows));
  constexpr auto desc = matmul2d_descriptor(M, NK, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, M>(0, int(tg.y) * M);
  auto mB = tB.template slice<ATTN_ID, NK>(0, a.kc0 + int(tg.x) * NK);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mB, cT);
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      int col = id[0], row = id[1];
      int q = int(tg.y) * QT + row / 32;
      float w = q < a.n ? wq[q * ATTN_IH + (row & 31)] : 0.0f;
      T[row * NK + col] = max(cT[i], 0.0f) * w;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int o = t; o < QT * NK; o += 128) {
    const int ql = o / NK, key = o % NK;
    const int q = int(tg.y) * QT + ql;
    float s = 0;
    for (int h = 0; h < 32; h++) s += T[(ql * 32 + h) * NK + key];
    const int tl = int(tg.x) * NK + key, tabs = a.kc0 + tl;
    const int len = (a.start + q + 1) / a.ratio;
    bool vis = q < a.n && tabs < len && tl < a.kc;
    if (vis && a.use_mask) vis = (mask[(size_t)q * a.mask_words + (tabs >> 8)] >> ((tabs >> 3) & 31)) & 1u;
    float sc = vis ? s : -INFINITY;
    const bool live = q < a.n && tl < a.kc;
    if (live) S[(size_t)q * a.ld_s + tl] = sc;
    if (a.cand_src) {  // 8 consecutive keys = 8 consecutive lanes
      float bm = sc;
      bm = max(bm, simd_shuffle_xor(bm, 1)); bm = max(bm, simd_shuffle_xor(bm, 2)); bm = max(bm, simd_shuffle_xor(bm, 4));
      if (len > 0 && (tabs >> 3) == (len - 1) / 8) bm = INFINITY;
      if (live && (key & 7) == 0) BS[(size_t)q * (a.ld_s / 8) + (tl >> 3)] = bm;
    }
  }
}
#define ATTN_IDX_PF_KERNEL(NAME, QT, NK)                                                                          \
  kernel void NAME(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],                            \
                   device half* ikey [[buffer(2)]], device float* S [[buffer(3)]],                                 \
                   device const uint* mask [[buffer(4)]], device float* BS [[buffer(5)]],                          \
                   constant AttnIdxPfArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],             \
                   ushort t [[thread_index_in_threadgroup]]) {                                                     \
    threadgroup float T[QT * 32 * NK];                                                                             \
    attn_idx_score_pf_impl<QT, NK>(Qi, wq, ikey, S, mask, BS, a, tg, t, T);                                        \
  }
ATTN_IDX_PF_KERNEL(attn_idx_score_pf, 2, 128)  // (4 queries x 64 keys: -30%; 2 x 64: -3%)

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

// ------------------------------------------------------------------------------------------------------------------
// Decode fast path: tensor-op scoring (M = 32 heads, N = 64 keys, K = 128) + 4096-bin histogram of the top 12 key bits,
// then a grid-wide compaction and a small final select (attn_topk_compact / attn_topk_final).
struct AttnIdxDec2Args {
  int pos, len, nvirt, actq, tiles_per_tg, cand_src, krows;
};
// grid (ceil(ceil(nvirt/64)/tiles_per_tg)), 128 threads. hist/bhist: device uint[4096] (zeroed by attn_topk_final).
kernel void attn_idx_score_dec2(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                                device const float2* tab [[buffer(2)]], device half* ikey [[buffer(3)]],
                                device float* scores [[buffer(4)]], device float* bscores [[buffer(5)]],
                                device atomic_uint* hist [[buffer(6)]], device atomic_uint* bhist [[buffer(7)]],
                                constant AttnIdxDec2Args& a [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                                ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Qs[ATTN_IH * ATTN_ID];
  threadgroup float T[ATTN_IH * 64];
  threadgroup atomic_uint lh[4096];
  {
    float v[32];
    attn_idx_prep_thread(qraw, tab, a.pos, a.actq, t, v);
    threadgroup half* d = Qs + (t >> 2) * ATTN_ID + (t & 3) * 32;
    for (int i = 0; i < 32; i++) d[i] = half(v[i]);
    for (int b = t; b < 4096; b += 128) atomic_store_explicit(&lh[b], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Qs, dextents<int32_t, 2>(ATTN_ID, ATTN_IH));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, a.krows));
  constexpr auto desc = matmul2d_descriptor(ATTN_IH, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int nblocks = (a.len + 7) / 8;
  for (int tt = 0; tt < a.tiles_per_tg; tt++) {
    const int k0 = (int(tg) * a.tiles_per_tg + tt) * 64;
    if (k0 >= a.nvirt) break;
    auto mB = tB.template slice<ATTN_ID, 64>(0, k0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(mB), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(tA, mB, cT);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto id = cT.get_multidimensional_index(i);
        T[id[1] * 64 + id[0]] = max(cT[i], 0.0f) * (wraw[id[1]] * (1.0f / 64.0f));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64) {
      float s = 0;
      for (int h = 0; h < ATTN_IH; h++) s += T[h * 64 + t];
      const int key = k0 + t;
      const bool valid = key < a.len;
      float sc = valid ? s : -INFINITY;
      if (key < a.nvirt) scores[key] = sc;
      if (valid) atomic_fetch_add_explicit(&lh[attn_fkey(sc) >> 20], 1u, memory_order_relaxed);
      if (a.cand_src) {
        float bm = sc;
        bm = max(bm, simd_shuffle_xor(bm, 1)); bm = max(bm, simd_shuffle_xor(bm, 2)); bm = max(bm, simd_shuffle_xor(bm, 4));
        int b = key >> 3;
        if (b == (a.len - 1) / 8) bm = INFINITY;
        if ((t & 7) == 0 && b < nblocks) {
          bscores[b] = bm;
          atomic_fetch_add_explicit(&bhist[attn_fkey(bm) >> 20], 1u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int b = t; b < 4096; b += 128) {
    uint v = atomic_load_explicit(&lh[b], memory_order_relaxed);
    if (v) atomic_fetch_add_explicit(&hist[b], v, memory_order_relaxed);
  }
}

// threshold bin from a 4096-bin histogram (descending): largest b0 with count(bins >= b0) >= k (or 0 if total <= k).
// 256 threads, 16 bins each. Returns b0 and count of elements in bins > b0 via tmp.
inline void attn_hist_threshold(device const uint* hist, int k, threadgroup uint* tmp, ushort t, ushort sg, ushort ln,
                                thread uint& b0, thread uint& above_b0, thread uint& total) {
  uint c[16], s = 0;
  for (int j = 0; j < 16; j++) { c[j] = hist[4095 - (t * 16 + j)]; s += c[j]; }
  uint above = attn_scan_excl(s, tmp, sg, ln, 8, total);
  if (total <= uint(k)) { b0 = 0; above_b0 = 0; return; }
  if (above < uint(k) && uint(k) <= above + s) {
    uint acc = above;
    for (int j = 0; j < 16; j++) {
      if (acc + c[j] >= uint(k)) { tmp[34] = 4095 - (t * 16 + j); tmp[35] = acc; break; }
      acc += c[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  b0 = tmp[34]; above_b0 = tmp[35];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ctl layout (device uint): [0] count of definite picks (bins > b0) in sel, [1] count of candidates (bin == b0) in cand,
// [2] overflow flag. sel: int[k]; cand: uint2 (key, idx) [cap].
struct AttnCompactArgs { int n; int k; int cap; int per_tg; };
// grid (ceil(n/per_tg)), 256 threads
kernel void attn_topk_compact(device const float* scores [[buffer(0)]], device const uint* hist [[buffer(1)]],
                              device atomic_uint* ctl [[buffer(2)]], device int* sel [[buffer(3)]],
                              device uint2* cand [[buffer(4)]], constant AttnCompactArgs& a [[buffer(5)]],
                              uint tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint tmp[40];
  uint b0, above, total;
  attn_hist_threshold(hist, a.k, tmp, t, sg, ln, b0, above, total);
  const int i0 = int(tg) * a.per_tg, i1 = min(a.n, i0 + a.per_tg);
  for (int i = i0 + t; i < i1; i += 256) {
    uint kk = attn_fkey(scores[i]);
    if (kk == 0u) continue;
    uint bin = kk >> 20;
    if (bin > b0) {
      uint sl = atomic_fetch_add_explicit(&ctl[0], 1u, memory_order_relaxed);
      sel[sl] = i;
    } else if (bin == b0) {
      uint sl = atomic_fetch_add_explicit(&ctl[1], 1u, memory_order_relaxed);
      if (sl < uint(a.cap)) cand[sl] = uint2(kk, uint(i)); else atomic_store_explicit(&ctl[2], 1u, memory_order_relaxed);
    }
  }
}

struct AttnSrcCand {
  device const uint2* c;
  uint key(int i) const { return c[i].x; }
  int idx(int i) const { return int(c[i].y); }
};
struct AttnFinalArgs { int n; int k; int list_mode; };
// One threadgroup (1024): picks = sel[0..ctl0) + top (k - ctl0) of cand; sorted ascending; -1 padded; maps list mode;
// resets hist/ctl for the next use. Falls back to a full exact select over scores on candidate overflow or a
// partially-taken tie (keeps results deterministic).
kernel void attn_topk_final(device const float* scores [[buffer(0)]], device uint* hist [[buffer(1)]],
                            device uint* ctl [[buffer(2)]], device const int* sel [[buffer(3)]],
                            device const uint2* cand [[buffer(4)]], device const int* blist [[buffer(5)]],
                            device int* out [[buffer(6)]], constant AttnFinalArgs& a [[buffer(7)]],
                            ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint lhist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  threadgroup int flag;
  const int nsel = int(ctl[0]), ncand = int(ctl[1]);
  const bool overflow = ctl[2] != 0u;
  int m = 0;
  if (tid == 0) flag = 0;
  if (!overflow) {
    for (int i = tid; i < nsel; i += 1024) oidx[i] = sel[i];
    const int need = min(a.k, nsel + ncand) - nsel;
    AttnSrcCand src{cand};
    // select `need` of the candidates into okey/oidx after the definite picks (reuse lhist as histogram)
    threadgroup uint* ok2 = okey;
    threadgroup int* oi2 = oidx + nsel;
    if (need > 0) {
      attn_select(src, ncand, need, lhist, tmp, ok2, oi2, &ocnt, tid, sg, ln);
      // detect a partially taken tie at the threshold: if any unselected candidate equals the minimum selected key
      // we cannot know the lowest-index rule was respected (candidate order is arbitrary) -> full fallback.
      if (need < ncand) {
        if (tid == 0) { tmp[37] = 0xFFFFFFFFu; tmp[38] = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint mn = 0xFFFFFFFFu;
        for (int i = tid; i < need; i += 1024) mn = min(mn, ok2[i]);
        mn = simd_min(mn);
        if (ln == 0) atomic_fetch_min_explicit((threadgroup atomic_uint*)&tmp[37], mn, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint T = tmp[37];
        uint unsel = 0;  // candidates equal to T minus selected equal to T (>= 0)
        for (int i = tid; i < ncand; i += 1024) unsel += cand[i].x == T ? 1u : 0u;
        for (int i = tid; i < need; i += 1024) unsel -= ok2[i] == T ? 1u : 0u;
        unsel = simd_sum(unsel);
        if (ln == 0) atomic_fetch_add_explicit((threadgroup atomic_uint*)&tmp[38], unsel, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0 && tmp[38] != 0u) flag = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    m = nsel + max(need, 0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (overflow || flag) {
    AttnSrcScores src{scores};
    attn_select(src, a.n, a.k, lhist, tmp, okey, oidx, &ocnt, tid, sg, ln);
    m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  }
  attn_sort_int(oidx, m, 2048, tid);
  for (int i = tid; i < a.k; i += 1024) {
    int v = i < m ? oidx[i] : -1;
    if (v >= 0 && a.list_mode) v = blist[v >> 3] * 8 + (v & 7);
    out[i] = v;
  }
  for (int b = tid; b < 4096; b += 1024) hist[b] = 0;
  if (tid < 4) ctl[tid] = 0;
}

// ------------------------------------------------------------------------------------------------------------------
// Prefill, candidate-restricted index layers (24..36): score only the union of the candidate blocks of each 2-query
// tile (instead of all keys + mask).
struct AttnUnionArgs { int n; int mask_words; int cap; };  // cap = max blocks per union list (4096)
// grid (ceil(n/2)), 1024 threads: ulist[tile][0..ucount) = sorted union of the candidate blocks of queries 2t, 2t+1
kernel void attn_cand_union(device const uint* mask [[buffer(0)]], device int* ulist [[buffer(1)]],
                            device int* ucount [[buffer(2)]], constant AttnUnionArgs& a [[buffer(3)]],
                            uint tile [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint tmp[40];
  const int q0 = int(tile) * 2, q1 = min(q0 + 1, a.n - 1);
  int base = 0;
  for (int w0 = 0; w0 < a.mask_words; w0 += 1024) {
    int w = w0 + tid;
    uint bits = w < a.mask_words ? (mask[(size_t)q0 * a.mask_words + w] | mask[(size_t)q1 * a.mask_words + w]) : 0u;
    uint tot;
    uint ex = attn_scan_excl(popcount(bits), tmp, sg, ln, 32, tot);
    int o = base + int(ex);
    while (bits) {
      int b = ctz(bits);
      bits &= bits - 1;
      if (o < a.cap) ulist[(size_t)tile * a.cap + o] = w * 32 + b;
      o++;
    }
    base += int(tot);
  }
  if (tid == 0) ucount[tile] = min(base, a.cap);
}

struct AttnIdxPfListArgs {
  int start, n, ratio;
  int kc0, kc;       // virtual key chunk
  int ld_s, mask_words, cap, qrows;
};
// grid (kc/64, ceil(n/2)), 128 threads. Virtual key v of tile y -> position ulist[y][v/8]*8 + v%8.
kernel void attn_idx_score_pf_list(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],
                                   device const half* ikey [[buffer(2)]], device float* S [[buffer(3)]],
                                   device const uint* mask [[buffer(4)]], device const int* ulist [[buffer(5)]],
                                   device const int* ucount [[buffer(6)]], constant AttnIdxPfListArgs& a [[buffer(7)]],
                                   uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Kg[64 * ATTN_ID];
  threadgroup float T[64 * 64];
  const int nvirt = ucount[tg.y] * 8;
  const int v0 = a.kc0 + int(tg.x) * 64;
  if (v0 >= nvirt) return;
  {  // gather 8 blocks x 8 keys x 128 dims: thread t -> key t/2, half row (64 dims)
    int kl = t >> 1, part = t & 1, v = v0 + kl;
    threadgroup uint4* dst = (threadgroup uint4*)(Kg + kl * ATTN_ID + part * 64);
    if (v < nvirt) {
      int pos = ulist[(size_t)tg.y * a.cap + (v >> 3)] * 8 + (v & 7);
      device const uint4* src = (device const uint4*)(ikey + (size_t)pos * ATTN_ID + part * 64);
      for (int j = 0; j < 8; j++) dst[j] = src[j];
    } else {
      for (int j = 0; j < 8; j++) dst[j] = uint4(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kg, dextents<int32_t, 2>(ATTN_ID, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, 64>(0, int(tg.y) * 64);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, tB, cT);
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
  const int v = v0 + key, tl = v - a.kc0;
  if (q >= a.n || tl >= a.kc) return;
  bool vis = v < nvirt;
  if (vis) {
    int blk = ulist[(size_t)tg.y * a.cap + (v >> 3)];
    int pos = blk * 8 + (v & 7);
    int len = (a.start + q + 1) / a.ratio;
    vis = pos < len && ((mask[(size_t)q * a.mask_words + (blk >> 5)] >> (blk & 31)) & 1u);
  }
  S[(size_t)q * a.ld_s + tl] = vis ? s : -INFINITY;
}

struct AttnSrcMergeList {
  device const uint* rk;
  device const int* ri;
  int rc;
  device const float* s;
  int base;
  device const int* ul;
  uint key(int i) const { return i < rc ? rk[i] : attn_fkey(s[i - rc]); }
  int idx(int i) const { if (i < rc) return ri[i]; int v = base + (i - rc); return ul[v >> 3] * 8 + (v & 7); }
};
struct AttnMergeListArgs { int n, k, kc, ld_s, base, first, cap; };
// list-mode merge: chunk element count per query = clamp(ucount[q/2]*8 - base, 0, kc)
kernel void attn_topk_merge_list(device const float* S [[buffer(0)]], device uint* Rk [[buffer(1)]],
                                 device int* Ri [[buffer(2)]], device int* Rc [[buffer(3)]],
                                 device const int* ulist [[buffer(4)]], device const int* ucount [[buffer(5)]],
                                 constant AttnMergeListArgs& a [[buffer(6)]], uint q [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  const int tile = int(q) / 2;
  const int cnt = clamp(ucount[tile] * 8 - a.base, 0, a.kc);
  const int rc = a.first ? 0 : Rc[q];
  if (!a.first && cnt == 0) return;
  AttnSrcMergeList src{Rk + (size_t)q * a.k, Ri + (size_t)q * a.k, rc, S + (size_t)q * a.ld_s, a.base,
                       ulist + (size_t)tile * a.cap};
  attn_select(src, rc + cnt, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  for (int i = tid; i < m; i += 1024) { Rk[(size_t)q * a.k + i] = okey[i]; Ri[(size_t)q * a.k + i] = oidx[i]; }
  if (tid == 0) Rc[q] = m;
}


// Decode fast path, candidate-list mode (layers 24..36): virtual key v -> row blist[v/8]*8 + v%8. Keys of each 64-key
// tile (8 blocks) are gathered into threadgroup memory; histogram via device atomics. Same outputs as dec2.
kernel void attn_idx_score_dec2_list(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                                     device const float2* tab [[buffer(2)]], device const half* ikey [[buffer(3)]],
                                     device float* scores [[buffer(4)]], device const int* blist [[buffer(5)]],
                                     device atomic_uint* hist [[buffer(6)]], constant AttnIdxDec2Args& a [[buffer(8)]],
                                     uint tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Qs[ATTN_IH * ATTN_ID];
  threadgroup float T[ATTN_IH * 64];
  threadgroup half Kg[64 * ATTN_ID];
  {
    float v[32];
    attn_idx_prep_thread(qraw, tab, a.pos, a.actq, t, v);
    threadgroup half* d = Qs + (t >> 2) * ATTN_ID + (t & 3) * 32;
    for (int i = 0; i < 32; i++) d[i] = half(v[i]);
  }
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Qs, dextents<int32_t, 2>(ATTN_ID, ATTN_IH));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kg, dextents<int32_t, 2>(ATTN_ID, 64));
  constexpr auto desc = matmul2d_descriptor(ATTN_IH, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  for (int tt = 0; tt < a.tiles_per_tg; tt++) {
    const int k0 = (int(tg) * a.tiles_per_tg + tt) * 64;
    if (k0 >= a.nvirt) break;
    {  // gather: thread t -> key t/2, 64 dims
      int kl = t >> 1, part = t & 1, v = k0 + kl;
      int blk = v < a.nvirt ? blist[v >> 3] : -1;
      threadgroup uint4* dst = (threadgroup uint4*)(Kg + kl * ATTN_ID + part * 64);
      if (blk >= 0) {
        device const uint4* src = (device const uint4*)(ikey + (size_t)(blk * 8 + (v & 7)) * ATTN_ID + part * 64);
        for (int j = 0; j < 8; j++) dst[j] = src[j];
      } else {
        for (int j = 0; j < 8; j++) dst[j] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(tA, tB, cT);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto id = cT.get_multidimensional_index(i);
        T[id[1] * 64 + id[0]] = max(cT[i], 0.0f) * (wraw[id[1]] * (1.0f / 64.0f));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64) {
      float s = 0;
      for (int h = 0; h < ATTN_IH; h++) s += T[h * 64 + t];
      const int v = k0 + t;
      int blk = v < a.nvirt ? blist[v >> 3] : -1;
      int pos = blk * 8 + (v & 7);
      const bool valid = blk >= 0 && pos < a.len;
      float sc = valid ? s : -INFINITY;
      if (v < a.nvirt) scores[v] = sc;
      if (valid) atomic_fetch_add_explicit(&hist[attn_fkey(sc) >> 20], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
