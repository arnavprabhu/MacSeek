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

// grid (4 head groups of 16, splits, nq), 128 threads (4 simdgroups; simdgroup s owns latent dims 128s..128s+127).
kernel void attn_sparse(device const float* q [[buffer(0)]], device const half* ring [[buffer(1)]],
                        device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                        device const int* topk [[buffer(4)]], device const float* sinks [[buffer(5)]],
                        device const float2* tab [[buffer(6)]], device float* out [[buffer(7)]],
                        device float* part [[buffer(8)]], device float2* ml [[buffer(9)]],
                        constant AttnSparseArgs& a [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
                        ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup half Ks[16 * ATTN_LDK];
  threadgroup float Sp[4 * 256];
  threadgroup half Ps[256];
  threadgroup float As[16];
  threadgroup int vflag[16];
  const int h0 = int(tg.x) * 16, split = int(tg.y), qi = int(tg.z);
  const int p = a.start + qi;
  // ---- stage Q (16 heads x 512, rope on dims 448..511) as half, keep in registers as 8x8 fragments ----
  {
    device const float4* qs = (device const float4*)(q + ((size_t)qi * ATTN_H + h0) * ATTN_D);
    for (int i = t; i < 16 * 128; i += 128) {
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
  simdgroup_half8x8 qa[2][16];
  for (int ht = 0; ht < 2; ht++)
    for (int kk = 0; kk < 16; kk++) simdgroup_load(qa[ht][kk], Ks + ht * 8 * ATTN_LDK + sg * 128 + kk * 8, ATTN_LDK);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 o[2][16];
  for (int ht = 0; ht < 2; ht++) for (int dt = 0; dt < 16; dt++) o[ht][dt] = simdgroup_float8x8(0);
  const ushort fm = attn_frag_row(ln), fn = attn_frag_col(ln);
  const int hh = t >> 3, c = (t & 7) * 2;  // softmax ownership: head hh, rows c, c+1 of each 16-row chunk
  float m_run = -INFINITY, l_run = 0;
  const int R = ATTN_W + a.ntop;
  const int r_begin = split * a.rows_per_split, r_end = min(R, r_begin + a.rows_per_split);
  for (int r0 = r_begin; r0 < r_end; r0 += 16) {
    {  // gather 16 rows: 8 threads per row, 64 halves each
      int rr = t >> 3, pt = t & 7, r = r0 + rr;
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
      threadgroup uint4* dst = (threadgroup uint4*)(Ks + rr * ATTN_LDK + pt * 64);
      if (src) {
        device const uint4* s4 = (device const uint4*)(src + pt * 64);
        for (int j = 0; j < 8; j++) dst[j] = s4[j];
      } else {
        for (int j = 0; j < 8; j++) dst[j] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int any = 0;
    for (int i = 0; i < 16; i++) any |= vflag[i];
    if (any) {
      simdgroup_float8x8 sacc[2][2];
      for (int ht = 0; ht < 2; ht++) for (int rt = 0; rt < 2; rt++) sacc[ht][rt] = simdgroup_float8x8(0);
      for (int kk = 0; kk < 16; kk++)
        for (int rt = 0; rt < 2; rt++) {
          simdgroup_half8x8 kb;
          simdgroup_load(kb, Ks + rt * 8 * ATTN_LDK + sg * 128 + kk * 8, ATTN_LDK, ulong2(0, 0), true);
          for (int ht = 0; ht < 2; ht++) simdgroup_multiply_accumulate(sacc[ht][rt], qa[ht][kk], kb, sacc[ht][rt]);
        }
      for (int ht = 0; ht < 2; ht++)
        for (int rt = 0; rt < 2; rt++) simdgroup_store(sacc[ht][rt], Sp + sg * 256 + ht * 8 * 16 + rt * 8, 16);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float s0 = 0, s1 = 0;
      for (int g = 0; g < 4; g++) { s0 += Sp[g * 256 + hh * 16 + c]; s1 += Sp[g * 256 + hh * 16 + c + 1]; }
      s0 = vflag[c] ? s0 * a.scale : -INFINITY;
      s1 = vflag[c + 1] ? s1 * a.scale : -INFINITY;
      float mx = max(s0, s1);
      mx = max(mx, simd_shuffle_xor(mx, 1)); mx = max(mx, simd_shuffle_xor(mx, 2)); mx = max(mx, simd_shuffle_xor(mx, 4));
      float m_new = max(m_run, mx);
      float alpha = 1, p0 = 0, p1 = 0;
      if (m_new != -INFINITY) { alpha = exp(m_run - m_new); p0 = exp(s0 - m_new); p1 = exp(s1 - m_new); }
      float ps = p0 + p1;
      ps += simd_shuffle_xor(ps, 1); ps += simd_shuffle_xor(ps, 2); ps += simd_shuffle_xor(ps, 4);
      l_run = l_run * alpha + ps;
      m_run = m_new;
      Ps[hh * 16 + c] = half(p0);
      Ps[hh * 16 + c + 1] = half(p1);
      if ((t & 7) == 0) As[hh] = alpha;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float al[2] = {As[fm], As[8 + fm]};
      simdgroup_half8x8 pa[2][2];
      for (int ht = 0; ht < 2; ht++)
        for (int rt = 0; rt < 2; rt++) simdgroup_load(pa[ht][rt], Ps + ht * 8 * 16 + rt * 8, 16);
      for (int dt = 0; dt < 16; dt++) {
        for (int ht = 0; ht < 2; ht++) {
          o[ht][dt].thread_elements()[0] *= al[ht];
          o[ht][dt].thread_elements()[1] *= al[ht];
        }
        for (int rt = 0; rt < 2; rt++) {
          simdgroup_half8x8 vb;
          simdgroup_load(vb, Ks + rt * 8 * ATTN_LDK + sg * 128 + dt * 8, ATTN_LDK);
          for (int ht = 0; ht < 2; ht++) simdgroup_multiply_accumulate(o[ht][dt], pa[ht][rt], vb, o[ht][dt]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float Ms[16], Ls[16];
  if ((t & 7) == 0) { Ms[hh] = m_run; Ls[hh] = l_run; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ht = 0; ht < 2; ht++) {
    const int head = ht * 8 + fm, hgl = h0 + head;
    float mh = Ms[head], lh = Ls[head];
    if (a.splits == 1) {
      float sink = sinks[hgl];
      float mm = max(mh, sink);
      float em = exp(mh - mm);
      float f = em / (lh * em + exp(sink - mm));
      device float* dst = out + ((size_t)qi * ATTN_H + hgl) * ATTN_D;
      for (int dt = 0; dt < 16; dt++) {
        int d = sg * 128 + dt * 8 + fn;
        auto e = o[ht][dt].thread_elements();
        float2 v = float2(e[0], e[1]) * f;
        if (d >= ATTN_D - ATTN_RD) v = attn_rot_inv(v, tab[(size_t)p * 32 + (d - (ATTN_D - ATTN_RD)) / 2]);
        *(device float2*)(dst + d) = v;
      }
    } else {
      device float* dst = part + ((size_t)split * ATTN_H + hgl) * ATTN_D;
      for (int dt = 0; dt < 16; dt++) {
        int d = sg * 128 + dt * 8 + fn;
        auto e = o[ht][dt].thread_elements();
        *(device float2*)(dst + d) = float2(e[0], e[1]);
      }
      if (sg == 0 && (ln & 0x9) == 0) ml[split * ATTN_H + hgl] = float2(mh, lh);  // lanes with fn==0
    }
  }
}

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
