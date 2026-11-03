// Speculative-decoding verify kernels (prefix sv_): K-token versions of the decode (n = 1) kernels that compute every
// token's outputs with exactly the decode kernel's per-lane arithmetic (same lane/row/K-split mapping, same
// accumulation order, same reductions), while each weight block is fetched once for all K tokens. Their outputs are
// bit-identical to K separate decode dispatches (tests/test_spec_kernels.mm), which keeps speculative greedy output
// identical to plain greedy decode.
#include "quant_blocks.h"

// Weight block decoded once per (row, block) and dotted with several tokens' x slices. dot() evaluates exactly the
// expression of Q::dot on the same decoded values, so each token's partial sums are bit-identical to Q::dot.
template <typename Q> struct SvW {   // generic: decode inside every dot (no reuse)
  device const uchar* b;
  inline void prep(device const uchar* b_, ushort) { b = b_; }
  inline float dot(ushort sub, thread const float4* xr, uint) const { return Q::dot(b, sub, xr); }
};
// Q5_K / Q6_K: only the integer unpacking is hoisted (exact); the float expression of Q::dot (incl. the scale products)
// is re-evaluated per token from the same source expression so fast-math contracts it the same way.
#ifndef SV_HOIST
#define SV_HOIST 1   // Q5_K: raw words hoisted (bit-identical, tested); Q6_K: generic
#endif
#if SV_HOIST
template <> struct SvW<QB_Q5K> {   // raw block words hoisted (loads only); all unpacking + float math per token
  uint ql0, ql1, qh0, qh1, dd, A, B, C; ushort j;
  inline void prep(device const uchar* b, ushort sub) {
    j = sub >> 2; const int l0 = 8 * (sub & 3);
    ql0 = q_ld32a(b + 48 + 32 * j + l0); ql1 = q_ld32a(b + 52 + 32 * j + l0);
    qh0 = q_ld32a(b + 16 + l0); qh1 = q_ld32a(b + 20 + l0);
    dd = q_ld32a(b); A = q_ld32a(b + 4); B = q_ld32a(b + 8); C = q_ld32a(b + 12);
  }
  inline float dot(ushort, thread const float4* xr, uint z) const {
    const uint ql0 = this->ql0 ^ z, ql1 = this->ql1 ^ z, qh0 = (this->qh0 ^ z) >> (2 * j), qh1 = (this->qh1 ^ z) >> (2 * j);
    float4 q[4]; float sm[4];
    q[0] = q_u8x4((ql0 & 0x0F0F0F0Fu) | ((qh0 & 0x01010101u) << 4));
    q[1] = q_u8x4((ql1 & 0x0F0F0F0Fu) | ((qh1 & 0x01010101u) << 4));
    q[2] = q_u8x4(((ql0 >> 4) & 0x0F0F0F0Fu) | ((qh0 & 0x02020202u) << 3));
    q[3] = q_u8x4(((ql1 >> 4) & 0x0F0F0F0Fu) | ((qh1 & 0x02020202u) << 3));
    const uint dd = this->dd ^ z, sh = 16 * (j & 1);
    const uint A = (this->A ^ z) >> sh, B = (this->B ^ z) >> sh, C = (this->C ^ z) >> sh;
    const uint sc = j < 2 ? (A & 0x3F3Fu) : ((C & 0x0F0Fu) | ((A >> 2) & 0x3030u));
    const uint mn = j < 2 ? (B & 0x3F3Fu) : (((C >> 4) & 0x0F0Fu) | ((B >> 2) & 0x3030u));
    const float d = float(as_type<half2>(dd).x), dm = float(as_type<half2>(dd).y);
    sm[0] = d * float(sc & 0xFF); sm[1] = dm * float(mn & 0xFF); sm[2] = d * float(sc >> 8); sm[3] = dm * float((mn >> 8) & 0xFF);
    const float4 one = 1.f;
    return sm[0] * (metal::dot(q[0], xr[0]) + metal::dot(q[1], xr[1])) - sm[1] * metal::dot(xr[0] + xr[1], one) +
           sm[2] * (metal::dot(q[2], xr[2]) + metal::dot(q[3], xr[3])) - sm[3] * metal::dot(xr[2] + xr[3], one);
  }
};
// Q6_K: hoisting (even raw words, laundered) is not bit-identical (measured): generic per-token Q::dot.
#endif
template <> struct SvW<QB_Q8_0> {
  float4 q0, q1; float d;
  inline void prep(device const uchar* b, ushort sub) {
    d = q_ldh(b); q0 = q_i8x4(q_ld32(b + 2 + 8 * sub)); q1 = q_i8x4(q_ld32(b + 6 + 8 * sub));
  }
  inline float dot(ushort, thread const float4* xr, uint) const { return d * (metal::dot(q0, xr[0]) + metal::dot(q1, xr[1])); }
};
template <> struct SvW<QB_BF16> {
  float4 w0, w1;
  inline void prep(device const uchar* b, ushort sub) {
    device const ushort4* p = (device const ushort4*)b;
    w0 = QB_BF16::cvt(p[sub]); w1 = QB_BF16::cvt(p[32 + sub]);
  }
  inline float dot(ushort, thread const float4* xr, uint) const { return metal::dot(w0, xr[0]) + metal::dot(w1, xr[1]); }
};
template <> struct SvW<QB_F16> {
  float4 w0, w1;
  inline void prep(device const uchar* b, ushort sub) {
    device const half4* p = (device const half4*)b;
    w0 = float4(p[sub]); w1 = float4(p[32 + sub]);
  }
  inline float dot(ushort, thread const float4* xr, uint) const { return metal::dot(w0, xr[0]) + metal::dot(w1, xr[1]); }
};
template <> struct SvW<QB_F32> {
  float4 w0, w1;
  inline void prep(device const uchar* b, ushort sub) {
    device const float4* p = (device const float4*)b;
    w0 = p[sub]; w1 = p[32 + sub];
  }
  inline float dot(ushort, thread const float4* xr, uint) const { return metal::dot(w0, xr[0]) + metal::dot(w1, xr[1]); }
};

// ---- matvec: K activation rows, mirrors q_mv (quant_mv.metal) ----
// x row t at x + t * x_stride (block-diagonal groups as in q_mv), y row t at y + t * y_stride.
template <typename Q, int NR, int NT>
kernel void sv_mvk(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                   constant QMvArgs& a [[buffer(3)]], threadgroup float* red [[threadgroup(0)]],
                   uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint zz = a.ksplit >> 16;   // runtime zero (ksplit <= 32), see SvW<QB_Q5K>::dot
  const uint row0 = (tg * rsg + rs) * NR;
  constexpr ushort BPI = 32 / Q::LPB;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK, n = a.n;
  device const float* xg = x + (row0 / a.rows_per_group) * a.x_group_stride;
  device const uchar* wr[NR];
  for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
  float acc[NR][NT];
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    SvW<Q> w[NR];
    for (int r = 0; r < NR; r++) w[r].prep(wr[r] + ib * Q::BB, sub);
    for (int t = 0; t < NT; t++) {
      if (t < (int)n) {
        float4 xr[Q::X4];
        Q::load_x(xg + t * a.x_stride + ib * Q::QK, sub, xr);
        for (int r = 0; r < NR; r++) acc[r][t] += w[r].dot(sub, xr, zz * t);
      }
    }
  }
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) acc[r][t] = simd_sum(acc[r][t]);
  if (ks > 1) {
    if (lane == 0)
      for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) red[((rs * NR + r) * NT + t) * ks + kk] = acc[r][t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++)
      for (int t = 0; t < NT; t++) {
        float s = 0;
        for (ushort k = 0; k < ks; k++) s += red[((rs * NR + r) * NT + t) * ks + k];
        acc[r][t] = s;
      }
  }
  if (lane == 0)
    for (int t = 0; t < NT; t++) {
      if (t >= (int)n) break;
      for (int r = 0; r < NR; r++) {
        const uint row = row0 + r;
        if (row < a.rows) {
          device float* yp = y + t * a.y_stride + row;
          *yp = a.accumulate ? *yp + acc[r][t] : acc[r][t];
        }
      }
    }
}

#define SV_MVK_INST1(Q, name, NR, NT)                                                                          \
  template [[host_name("sv_mvk_" name "_r" #NR "_n" #NT)]] kernel void sv_mvk<Q, NR, NT>(device const uchar*, \
      device const float*, device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);
#define SV_MVK_NR(Q, name, NT) SV_MVK_INST1(Q, name, 1, NT) SV_MVK_INST1(Q, name, 2, NT) SV_MVK_INST1(Q, name, 4, NT)
#define SV_MVK_INST(Q, name) SV_MVK_NR(Q, name, 2) SV_MVK_NR(Q, name, 4) SV_MVK_NR(Q, name, 8)
SV_MVK_INST(QB_F32, "f32")
SV_MVK_INST(QB_F16, "f16")
SV_MVK_INST(QB_BF16, "bf16")
SV_MVK_INST(QB_Q8_0, "q8_0")
SV_MVK_INST(QB_Q5K, "q5_k")
SV_MVK_INST(QB_Q6K, "q6_k")

// ---- shared expert, K tokens: mirrors the shared-expert branch of q_moe_gu_t / q_moe_down_t (quant_moe.metal) ----
struct SvMoeArgs {
  uint n;        // tokens
  uint dim;      // 5120
  uint inter;    // 2304
  uint rb_gu;    // shared gate/up row bytes
  uint rb_d;     // shared down row bytes
  uint ksplit;
};

static inline float sv_swiglu(float g, float u, float w) {   // == q_swiglu
  g = min(g, 10.f);
  u = clamp(u, -10.f, 10.f);
  return g / (1.f + exp(-g)) * u * w;
}

// h[t][row] = swiglu(gate[row] . x_t, up[row] . x_t, 1) for rows of the shared expert; x row t at x + t*dim,
// h row t at h + t*inter.
template <typename QS, int NR, int NT>
kernel void sv_moe_gu_s(device const float* x [[buffer(0)]], device float* h [[buffer(1)]],
                        constant SvMoeArgs& a [[buffer(2)]], device const uchar* G [[buffer(3)]],
                        device const uchar* U [[buffer(4)]], threadgroup float* red [[threadgroup(0)]],
                        uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint zz = a.ksplit >> 16;   // runtime zero (ksplit <= 32), see SvW<QB_Q5K>::dot
  const uint row0 = (tg * rsg + rs) * NR;
  constexpr ushort BPI = 32 / QS::LPB;
  const ushort sub = lane % QS::LPB, slot = lane / QS::LPB;
  const uint nb = a.dim / QS::QK, n = a.n;
  float ag[NR][NT], au[NR][NT];
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) { ag[r][t] = 0.f; au[r][t] = 0.f; }
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    SvW<QS> wg[NR], wu[NR];
    for (int r = 0; r < NR; r++) {
      const ulong off = (ulong)min(row0 + r, a.inter - 1) * a.rb_gu + ib * QS::BB;
      wg[r].prep(G + off, sub); wu[r].prep(U + off, sub);
    }
    for (int t = 0; t < NT; t++) {
      if (t < (int)n) {
        float4 xr[QS::X4];
        QS::load_x(x + t * a.dim + ib * QS::QK, sub, xr);
        for (int r = 0; r < NR; r++) {
          ag[r][t] += wg[r].dot(sub, xr, zz * t);
          au[r][t] += wu[r].dot(sub, xr, zz * t);
        }
      }
    }
  }
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) { ag[r][t] = simd_sum(ag[r][t]); au[r][t] = simd_sum(au[r][t]); }
  if (ks > 1) {
    if (lane == 0)
      for (int r = 0; r < NR; r++)
        for (int t = 0; t < NT; t++) {
          red[(((rs * NR + r) * NT + t) * 2 + 0) * ks + kk] = ag[r][t];
          red[(((rs * NR + r) * NT + t) * 2 + 1) * ks + kk] = au[r][t];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++)
      for (int t = 0; t < NT; t++) {
        float sg_ = 0, su = 0;
        for (ushort k = 0; k < ks; k++) {
          sg_ += red[(((rs * NR + r) * NT + t) * 2) * ks + k];
          su += red[(((rs * NR + r) * NT + t) * 2 + 1) * ks + k];
        }
        ag[r][t] = sg_; au[r][t] = su;
      }
  }
  if (lane == 0)
    for (int t = 0; t < NT; t++) {
      if (t >= (int)n) break;
      for (int r = 0; r < NR; r++) {
        const uint row = row0 + r;
        if (row < a.inter) h[t * a.inter + row] = sv_swiglu(ag[r][t], au[r][t], 1.f);
      }
    }
}

// y[t][row] += down[row] . h_t for the shared expert alone (q_moe_down_t with n_routed = 0).
template <typename QS, int NR, int NT>
kernel void sv_moe_down_s(device const float* h [[buffer(0)]], device float* y [[buffer(1)]],
                          constant SvMoeArgs& a [[buffer(2)]], device const uchar* D [[buffer(3)]],
                          threadgroup float* red [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                          ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint zz = a.ksplit >> 16;   // runtime zero (ksplit <= 32), see SvW<QB_Q5K>::dot
  const uint row0 = (tg * rsg + rs) * NR;
  constexpr ushort BPI = 32 / QS::LPB;
  const ushort sub = lane % QS::LPB, slot = lane / QS::LPB;
  const uint nb = a.inter / QS::QK, n = a.n;
  float acc[NR][NT];
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    SvW<QS> w[NR];
    for (int r = 0; r < NR; r++) w[r].prep(D + (ulong)min(row0 + r, a.dim - 1) * a.rb_d + ib * QS::BB, sub);
    for (int t = 0; t < NT; t++) {
      if (t < (int)n) {
        float4 xr[QS::X4];
        QS::load_x(h + t * a.inter + ib * QS::QK, sub, xr);
        for (int r = 0; r < NR; r++) acc[r][t] += w[r].dot(sub, xr, zz * t);
      }
    }
  }
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) acc[r][t] = simd_sum(acc[r][t]);
  if (ks > 1) {
    if (lane == 0)
      for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) red[((rs * NR + r) * NT + t) * ks + kk] = acc[r][t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++)
      for (int t = 0; t < NT; t++) {
        float s = 0;
        for (ushort k = 0; k < ks; k++) s += red[((rs * NR + r) * NT + t) * ks + k];
        acc[r][t] = s;
      }
  }
  if (lane == 0)
    for (int t = 0; t < NT; t++) {
      if (t >= (int)n) break;
      for (int r = 0; r < NR; r++) {
        const uint row = row0 + r;
        if (row < a.dim) y[t * a.dim + row] += acc[r][t];
      }
    }
}

#define SV_MOE_S_INST1(QS, name, NR, NT)                                                                              \
  template [[host_name("sv_moe_gu_s_" name "_r" #NR "_n" #NT)]] kernel void sv_moe_gu_s<QS, NR, NT>(                \
      device const float*, device float*, constant SvMoeArgs&, device const uchar*, device const uchar*,              \
      threadgroup float*, uint, ushort, ushort, ushort);                                                              \
  template [[host_name("sv_moe_down_s_" name "_r" #NR "_n" #NT)]] kernel void sv_moe_down_s<QS, NR, NT>(            \
      device const float*, device float*, constant SvMoeArgs&, device const uchar*, threadgroup float*, uint, ushort, \
      ushort, ushort);
#define SV_MOE_S_NR(QS, name, NT) SV_MOE_S_INST1(QS, name, 1, NT) SV_MOE_S_INST1(QS, name, 2, NT) SV_MOE_S_INST1(QS, name, 4, NT)
#define SV_MOE_S_INST(QS, name) SV_MOE_S_NR(QS, name, 2) SV_MOE_S_NR(QS, name, 4) SV_MOE_S_NR(QS, name, 8)
SV_MOE_S_INST(QB_Q6K, "q6_k")
SV_MOE_S_INST(QB_Q8_0, "q8_0")
SV_MOE_S_INST(QB_Q5K, "q5_k")
