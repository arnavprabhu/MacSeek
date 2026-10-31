// Attention module: sparse attention with per-head sink over window rows (ring / prefill chunk) + top-k compressed
// rows, single shared K==V latent (512 dims), 64 heads. simdgroup-matrix flash-style kernel.
#include "attn_common.h"

struct AttnSparseArgs {
  int start;           // absolute position of query 0
  int nq;
  int chunk_start;     // window positions >= chunk_start come from chunk rows (pos - chunk_start), else ring[pos%128]
  int ntop;            // compressed rows listed per query (0 for ratio-0 layers)
  int splits;          // > 1: write partials for attn_sparse_combine (decode split-K); 1: final output
  int rows_per_split;  // multiple of 16
  float scale;         // 512^-0.5
};

constant constexpr int ATTN_LDK = 520;  // padded row stride (halves) of the staged rows

// grid (64/(8*HT) head groups, splits, nq), 32*NSG threads. HT = head tiles of 8 per threadgroup; NSG simdgroups,
// simdgroup s owns latent dims [s*DPS, (s+1)*DPS) with DPS = 512/NSG. Per 16-row chunk: gather rows into threadgroup
// memory, S = Q K^T (partial over each simdgroup's dims, reduced through threadgroup memory), online softmax with
// running max/sum per head, O += P V. Per-thread state: Q fragments HT*DPS/8, O fragments HT*DPS/8.
template <int HT, int NSG>
inline void attn_sparse_impl(device const float* q, device const half* ring, device const half* chunk,
                             device const half* ckv, device const int* topk, device const float* sinks,
                             device const float2* tab, device float* out, device float* part, device float2* ml,
                             constant AttnSparseArgs& a, uint3 tg, ushort t, ushort sg, ushort ln,
                             threadgroup half* Ks, threadgroup float* Sp, threadgroup half* Ps, threadgroup float* As,
                             threadgroup int* vflag, threadgroup float* Ms, threadgroup float* Ls) {
  constexpr int HG = 8 * HT;       // heads per threadgroup
  constexpr int NT = 32 * NSG;     // threads
  constexpr int DPS = 512 / NSG;   // dims per simdgroup
  constexpr int KT = DPS / 8;      // 8-wide tiles per simdgroup
  constexpr int LPH = NT / HG;     // softmax lanes per head
  constexpr int RPL = 16 / LPH;    // rows per lane
  static_assert(LPH <= 16 && RPL >= 1, "softmax mapping");
  constexpr int TPR = NT / 16;     // gather threads per row
  constexpr int U4 = 512 / TPR / 8;  // uint4 per gather thread
  const int h0 = int(tg.x) * HG, split = int(tg.y), qi = int(tg.z);
  const int p = a.start + qi;
  // ---- stage Q (HG heads x 512, rope on dims 448..511) as half, keep in registers as 8x8 fragments ----
  {
    device const float4* qs = (device const float4*)(q + ((size_t)qi * ATTN_H + h0) * ATTN_D);
    for (int i = t; i < HG * 128; i += NT) {
      int hh = i >> 7, c4 = i & 127;
      float4 x = qs[i];
      if (c4 >= 112) {
        int pi = 2 * (c4 - 112);
        x.xy = attn_rot(x.xy, tab[(size_t)p * 32 + pi]);
        x.zw = attn_rot(x.zw, tab[(size_t)p * 32 + pi + 1]);
      }
      *(threadgroup half4*)(Ks + hh * ATTN_LDK + c4 * 4) = half4(x);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qa[HT][KT];
  for (int ht = 0; ht < HT; ht++)
    for (int kk = 0; kk < KT; kk++) simdgroup_load(qa[ht][kk], Ks + ht * 8 * ATTN_LDK + sg * DPS + kk * 8, ATTN_LDK);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 o[HT][KT];
  for (int ht = 0; ht < HT; ht++) for (int dt = 0; dt < KT; dt++) o[ht][dt] = simdgroup_float8x8(0);
  const ushort fm = attn_frag_row(ln), fn = attn_frag_col(ln);
  const int hh = t / LPH, c = (t % LPH) * RPL;  // softmax ownership: head hh, rows c..c+RPL-1 of each chunk
  float m_run = -INFINITY, l_run = 0;
  const int R = ATTN_W + a.ntop;
  const int r_begin = split * a.rows_per_split, r_end = min(R, r_begin + a.rows_per_split);
  for (int r0 = r_begin; r0 < r_end; r0 += 16) {
    {  // gather 16 rows: TPR threads per row
      int rr = t / TPR, pt = t % TPR, r = r0 + rr;
      device const half* src = nullptr;
      if (r < r_end) {
        if (r < ATTN_W) {
          int wp = p - (ATTN_W - 1) + r;
          if (wp >= 0) src = wp >= a.chunk_start ? chunk + (size_t)(wp - a.chunk_start) * ATTN_D : ring + (size_t)(wp % ATTN_W) * ATTN_D;
        } else {
          int id = topk[(size_t)qi * a.ntop + (r - ATTN_W)];
          if (id >= 0) src = ckv + (size_t)id * ATTN_D;
        }
      }
      if (pt == 0) vflag[rr] = src != nullptr;
      threadgroup uint4* dst = (threadgroup uint4*)(Ks + rr * ATTN_LDK + pt * U4 * 8);
      if (src) {
        device const uint4* s4 = (device const uint4*)(src + pt * U4 * 8);
        for (int j = 0; j < U4; j++) dst[j] = s4[j];
      } else {
        for (int j = 0; j < U4; j++) dst[j] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int any = 0;
    for (int i = 0; i < 16; i++) any |= vflag[i];
    if (any) {
      simdgroup_float8x8 sacc[HT][2];
      for (int ht = 0; ht < HT; ht++) for (int rt = 0; rt < 2; rt++) sacc[ht][rt] = simdgroup_float8x8(0);
      for (int kk = 0; kk < KT; kk++)
        for (int rt = 0; rt < 2; rt++) {
          simdgroup_half8x8 kb;
          simdgroup_load(kb, Ks + rt * 8 * ATTN_LDK + sg * DPS + kk * 8, ATTN_LDK, ulong2(0, 0), true);
          for (int ht = 0; ht < HT; ht++) simdgroup_multiply_accumulate(sacc[ht][rt], qa[ht][kk], kb, sacc[ht][rt]);
        }
      for (int ht = 0; ht < HT; ht++)
        for (int rt = 0; rt < 2; rt++) simdgroup_store(sacc[ht][rt], Sp + sg * (HG * 16) + ht * 8 * 16 + rt * 8, 16);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float sv[RPL];
      float mx = -INFINITY;
      for (int j = 0; j < RPL; j++) {
        float x = 0;
        for (int g = 0; g < NSG; g++) x += Sp[g * (HG * 16) + hh * 16 + c + j];
        sv[j] = vflag[c + j] ? x * a.scale : -INFINITY;
        mx = max(mx, sv[j]);
      }
      for (int sh = 1; sh < LPH; sh <<= 1) mx = max(mx, simd_shuffle_xor(mx, sh));
      float m_new = max(m_run, mx);
      float alpha = 1, ps = 0;
      float pv[RPL];
      for (int j = 0; j < RPL; j++) pv[j] = 0;
      if (m_new != -INFINITY) {
        alpha = exp(m_run - m_new);
        for (int j = 0; j < RPL; j++) { pv[j] = exp(sv[j] - m_new); ps += pv[j]; }
      }
      for (int sh = 1; sh < LPH; sh <<= 1) ps += simd_shuffle_xor(ps, sh);
      l_run = l_run * alpha + ps;
      m_run = m_new;
      for (int j = 0; j < RPL; j++) Ps[hh * 16 + c + j] = half(pv[j]);
      if ((t % LPH) == 0) As[hh] = alpha;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float al[HT];
      for (int ht = 0; ht < HT; ht++) al[ht] = As[ht * 8 + fm];
      simdgroup_half8x8 pa[HT][2];
      for (int ht = 0; ht < HT; ht++)
        for (int rt = 0; rt < 2; rt++) simdgroup_load(pa[ht][rt], Ps + ht * 8 * 16 + rt * 8, 16);
      for (int dt = 0; dt < KT; dt++) {
        for (int ht = 0; ht < HT; ht++) {
          o[ht][dt].thread_elements()[0] *= al[ht];
          o[ht][dt].thread_elements()[1] *= al[ht];
        }
        for (int rt = 0; rt < 2; rt++) {
          simdgroup_half8x8 vb;
          simdgroup_load(vb, Ks + rt * 8 * ATTN_LDK + sg * DPS + dt * 8, ATTN_LDK);
          for (int ht = 0; ht < HT; ht++) simdgroup_multiply_accumulate(o[ht][dt], pa[ht][rt], vb, o[ht][dt]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((t % LPH) == 0) { Ms[hh] = m_run; Ls[hh] = l_run; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ht = 0; ht < HT; ht++) {
    const int head = ht * 8 + fm, hgl = h0 + head;
    float mh = Ms[head], lh = Ls[head];
    if (a.splits == 1) {
      float sink = sinks[hgl];
      float mm = max(mh, sink);
      float em = exp(mh - mm);
      float f = em / (lh * em + exp(sink - mm));
      device float* dst = out + ((size_t)qi * ATTN_H + hgl) * ATTN_D;
      for (int dt = 0; dt < KT; dt++) {
        int d = sg * DPS + dt * 8 + fn;
        auto e = o[ht][dt].thread_elements();
        float2 v = float2(e[0], e[1]) * f;
        if (d >= ATTN_D - ATTN_RD) v = attn_rot_inv(v, tab[(size_t)p * 32 + (d - (ATTN_D - ATTN_RD)) / 2]);
        *(device float2*)(dst + d) = v;
      }
    } else {
      device float* dst = part + ((size_t)split * ATTN_H + hgl) * ATTN_D;
      for (int dt = 0; dt < KT; dt++) {
        int d = sg * DPS + dt * 8 + fn;
        auto e = o[ht][dt].thread_elements();
        *(device float2*)(dst + d) = float2(e[0], e[1]);
      }
      if (sg == 0 && (ln & 0x9) == 0) ml[split * ATTN_H + hgl] = float2(mh, lh);  // lanes with fn==0
    }
  }
}

#define ATTN_SPARSE_KERNEL(NAME, HT, NSG)                                                                         \
  kernel void NAME(device const float* q [[buffer(0)]], device const half* ring [[buffer(1)]],                     \
                   device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],                   \
                   device const int* topk [[buffer(4)]], device const float* sinks [[buffer(5)]],                  \
                   device const float2* tab [[buffer(6)]], device float* out [[buffer(7)]],                        \
                   device float* part [[buffer(8)]], device float2* ml [[buffer(9)]],                              \
                   constant AttnSparseArgs& a [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],           \
                   ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],         \
                   ushort ln [[thread_index_in_simdgroup]]) {                                                      \
    threadgroup half Ks[16 * ATTN_LDK];                                                                            \
    threadgroup float Sp[NSG * 8 * HT * 16];                                                                       \
    threadgroup half Ps[8 * HT * 16];                                                                              \
    threadgroup float As[8 * HT], Ms[8 * HT], Ls[8 * HT];                                                          \
    threadgroup int vflag[16];                                                                                     \
    attn_sparse_impl<HT, NSG>(q, ring, chunk, ckv, topk, sinks, tab, out, part, ml, a, tg, t, sg, ln, Ks, Sp, Ps,  \
                              As, vflag, Ms, Ls);                                                                  \
  }
ATTN_SPARSE_KERNEL(attn_sparse_h8, 1, 4)    // 8 heads, 128 threads
ATTN_SPARSE_KERNEL(attn_sparse_h16, 2, 8)   // 16 heads, 256 threads (64 dims per simdgroup)

struct AttnCombArgs { int pos; int splits; };
// grid (64 heads), 128 threads x 4 dims: merge split partials, sink, normalize, inverse rope -> out [64][512]
kernel void attn_sparse_combine(device const float* part [[buffer(0)]], device const float2* ml [[buffer(1)]],
                                device const float* sinks [[buffer(2)]], device const float2* tab [[buffer(3)]],
                                device float* out [[buffer(4)]], constant AttnCombArgs& a [[buffer(5)]],
                                uint h [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  float sink = sinks[h];
  float M = sink;
  for (int s = 0; s < a.splits; s++) M = max(M, ml[s * ATTN_H + h].x);
  float L = exp(sink - M);
  float4 acc = 0;
  for (int s = 0; s < a.splits; s++) {
    float2 v = ml[s * ATTN_H + h];
    float w = exp(v.x - M);
    L += w * v.y;
    acc += w * ((device const float4*)(part + ((size_t)s * ATTN_H + h) * ATTN_D))[t];
  }
  float4 o = acc / L;
  if (t >= 112) {
    int pi = 2 * (t - 112);
    o.xy = attn_rot_inv(o.xy, tab[(size_t)a.pos * 32 + pi]);
    o.zw = attn_rot_inv(o.zw, tab[(size_t)a.pos * 32 + pi + 1]);
  }
  ((device float4*)(out + (size_t)h * ATTN_D))[t] = o;
}

// ---- decode: sparse attention + split-K combine in ONE dispatch (MACSEEK_DEC_COMBINE) ----
// Every threadgroup runs attn_sparse_impl (partials as in the split path); the last threadgroup of each head group to
// finish (device atomic counter per head group, reset by that threadgroup) then merges the partials of its heads with
// exactly attn_sparse_combine's arithmetic (thread t of 128 owns dims 4t..4t+3 of each head) -> bit-identical output.
inline bool attn_last_group(device atomic_uint* counter, uint n_groups, ushort tid, threadgroup uint* flag) {
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint t = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    *flag = (t == n_groups - 1) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool last = *flag != 0;
  if (last) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return last;
}

#define ATTN_SPARSE_LC_KERNEL(NAME, HT, NSG)                                                                      \
  kernel void NAME(device const float* q [[buffer(0)]], device const half* ring [[buffer(1)]],                     \
                   device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],                   \
                   device const int* topk [[buffer(4)]], device const float* sinks [[buffer(5)]],                  \
                   device const float2* tab [[buffer(6)]], device float* out [[buffer(7)]],                        \
                   device float* part [[buffer(8)]], device float2* ml [[buffer(9)]],                              \
                   constant AttnSparseArgs& a [[buffer(10)]], device atomic_uint* cnt [[buffer(11)]],              \
                   coherent(device) device const float* part_c [[buffer(12)]],                                     \
                   coherent(device) device const float2* ml_c [[buffer(13)]],                                      \
                   uint3 tg [[threadgroup_position_in_grid]],                                                      \
                   ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],         \
                   ushort ln [[thread_index_in_simdgroup]]) {                                                      \
    threadgroup half Ks[16 * ATTN_LDK];                                                                            \
    threadgroup float Sp[NSG * 8 * HT * 16];                                                                       \
    threadgroup half Ps[8 * HT * 16];                                                                              \
    threadgroup float As[8 * HT], Ms[8 * HT], Ls[8 * HT];                                                          \
    threadgroup int vflag[16];                                                                                     \
    threadgroup uint flag;                                                                                         \
    attn_sparse_impl<HT, NSG>(q, ring, chunk, ckv, topk, sinks, tab, out, part, ml, a, tg, t, sg, ln, Ks, Sp, Ps,  \
                              As, vflag, Ms, Ls);                                                                  \
    if (!attn_last_group(cnt + tg.x, uint(a.splits), t, &flag)) return;                                           \
    if (t < 128) {  /* all heads of the group interleaved (independent loads in flight); per head the same    */  \
      constexpr int HG = 8 * HT; /* operations in the same order as attn_sparse_combine                       */  \
      const int hb = int(tg.x) * HG;                                                                               \
      float sink[HG], M[HG], L[HG];                                                                                \
      float4 acc[HG];                                                                                              \
      for (int hh = 0; hh < HG; hh++) { sink[hh] = sinks[hb + hh]; M[hh] = sink[hh]; }                             \
      for (int s = 0; s < a.splits; s++)                                                                           \
        for (int hh = 0; hh < HG; hh++) M[hh] = max(M[hh], ml_c[s * ATTN_H + hb + hh].x);                          \
      for (int hh = 0; hh < HG; hh++) { L[hh] = exp(sink[hh] - M[hh]); acc[hh] = 0; }                              \
      for (int s = 0; s < a.splits; s++)                                                                           \
        for (int hh = 0; hh < HG; hh++) {                                                                          \
          float2 v = ml_c[s * ATTN_H + hb + hh];                                                                   \
          float w = exp(v.x - M[hh]);                                                                              \
          L[hh] += w * v.y;                                                                                        \
          acc[hh] += w * ((coherent(device) device const float4*)(part_c + ((size_t)s * ATTN_H + hb + hh) * ATTN_D))[t]; \
        }                                                                                                          \
      for (int hh = 0; hh < HG; hh++) {                                                                            \
        float4 o = acc[hh] / L[hh];                                                                                \
        if (t >= 112) {                                                                                            \
          int pi = 2 * (t - 112);                                                                                  \
          o.xy = attn_rot_inv(o.xy, tab[(size_t)a.start * 32 + pi]);                                               \
          o.zw = attn_rot_inv(o.zw, tab[(size_t)a.start * 32 + pi + 1]);                                           \
        }                                                                                                          \
        ((device float4*)(out + (size_t)(hb + hh) * ATTN_D))[t] = o;                                               \
      }                                                                                                            \
    }                                                                                                              \
    if (t == 0) atomic_store_explicit(cnt + tg.x, 0u, memory_order_relaxed);                                      \
  }
ATTN_SPARSE_LC_KERNEL(attn_sparse_lc_h8, 1, 4)
ATTN_SPARSE_LC_KERNEL(attn_sparse_lc_h16, 2, 8)
