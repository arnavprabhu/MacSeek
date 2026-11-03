// Per-type block decoders for the quant module. Every type exposes the same static interface so the matvec /
// matvec_n / matmul / MoE kernels are written once as templates:
//   QK   elements per (super)block      BB  bytes per block
//   LPB  lanes cooperating on one block EPL elements handled per lane  X4 float4 registers of x per lane
//   load_x(xb, sub, xr)  : load the x slice of lane `sub` (xb = x + block*QK) into xr[X4] (+ precomputed sums)
//   dot(b, sub, xr)      : partial dot of block b (lane sub) with xr
//   deq(b, sub, w)       : dequantized weights of lane sub, w[EPL/4] in the same element order as load_x
//   pos(sub, i)          : element index (0..QK-1) of lane-element i  (element order used by load_x / deq)
#pragma once
#include "quant_common.h"

// ---------------- dense: F32 / F16 / BF16 (virtual block of 256, lanes read 2 coalesced float4/half4) ----------------
struct QB_F32 {
  static constexpr constant int QK = 256, BB = 1024, LPB = 32, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return (i >> 2) * 128 + sub * 4 + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    xr[0] = *(device const float4*)(xb + 4 * sub); xr[1] = *(device const float4*)(xb + 128 + 4 * sub);
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    device const float4* p = (device const float4*)b;
    w[0] = p[sub]; w[1] = p[32 + sub];
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    device const float4* p = (device const float4*)b;
    return metal::dot(p[sub], xr[0]) + metal::dot(p[32 + sub], xr[1]);
  }
};
struct QB_F16 {
  static constexpr constant int QK = 256, BB = 512, LPB = 32, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return (i >> 2) * 128 + sub * 4 + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) { QB_F32::load_x(xb, sub, xr); }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    device const half4* p = (device const half4*)b;
    w[0] = float4(p[sub]); w[1] = float4(p[32 + sub]);
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    device const half4* p = (device const half4*)b;
    return metal::dot(float4(p[sub]), xr[0]) + metal::dot(float4(p[32 + sub]), xr[1]);
  }
};
struct QB_BF16 {
  static constexpr constant int QK = 256, BB = 512, LPB = 32, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return (i >> 2) * 128 + sub * 4 + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) { QB_F32::load_x(xb, sub, xr); }
  static inline float4 cvt(ushort4 u) { return as_type<float4>(uint4(u) << 16); }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    device const ushort4* p = (device const ushort4*)b;
    w[0] = cvt(p[sub]); w[1] = cvt(p[32 + sub]);
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    device const ushort4* p = (device const ushort4*)b;
    return metal::dot(cvt(p[sub]), xr[0]) + metal::dot(cvt(p[32 + sub]), xr[1]);
  }
};

// ---------------- Q8_0: {half d; int8 qs[32]} 34 B. 4 lanes per block, 8 elements each ----------------
struct QB_Q8_0 {
  static constexpr constant int QK = 32, BB = 34, LPB = 4, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return sub * 8 + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    xr[0] = *(device const float4*)(xb + 8 * sub); xr[1] = *(device const float4*)(xb + 8 * sub + 4);
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    float d = q_ldh(b);
    w[0] = d * q_i8x4(q_ld32(b + 2 + 8 * sub)); w[1] = d * q_i8x4(q_ld32(b + 6 + 8 * sub));
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    return q_ldh(b) * (metal::dot(q_i8x4(q_ld32(b + 2 + 8 * sub)), xr[0]) + metal::dot(q_i8x4(q_ld32(b + 6 + 8 * sub)), xr[1]));
  }  // 4 consecutive weights starting at element j (j % 4 == 0) of block b (MoE prefill cooperative-tensor dequant)
  static inline float4 deq4(device const uchar* b, uint j) { return q_ldh(b) * q_i8x4(q_ld32(b + 2 + j)); }
};

// ---------------- Q6_K: {ql[128]; qh[64]; int8 sc[16]; half d} 210 B. 16 lanes; lane = (half h, l0 = 4*(sub&7)),
// elements 128h + 32k + l0 + j (k,j in 0..3): the 4 elements that share one ql/qh byte column ----------------
struct QB_Q6K {
  static constexpr constant int QK = 256, BB = 210, LPB = 16, EPL = 16, X4 = 4;
  static inline ushort pos(ushort sub, ushort i) { return 128 * (sub >> 3) + 32 * (i >> 2) + 4 * (sub & 7) + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float* p = xb + 128 * (sub >> 3) + 4 * (sub & 7);
    for (int k = 0; k < 4; k++) xr[k] = *(device const float4*)(p + 32 * k);
  }
  static inline void quants(device const uchar* b, ushort sub, thread float4* q, thread float* s) {
    const int h = sub >> 3, l0 = 4 * (sub & 7);
    const uint lo = q_ld32(b + 64 * h + l0), hi = q_ld32(b + 64 * h + 32 + l0), qh = q_ld32(b + 128 + 32 * h + l0);
    device const char* sc = (device const char*)(b + 192 + 8 * h + (l0 >> 4));
    q[0] = q_u8x4((lo & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4)) - 32.f;
    q[1] = q_u8x4((hi & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2)) - 32.f;
    q[2] = q_u8x4(((lo >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u)) - 32.f;
    q[3] = q_u8x4(((hi >> 4) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2)) - 32.f;
    s[0] = sc[0]; s[1] = sc[2]; s[2] = sc[4]; s[3] = sc[6];
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    float4 q[4]; float s[4]; quants(b, sub, q, s);
    const float d = q_ldh(b + 208);
    for (int k = 0; k < 4; k++) w[k] = (d * s[k]) * q[k];
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    float4 q[4]; float s[4]; quants(b, sub, q, s);
    return q_ldh(b + 208) * (s[0] * metal::dot(q[0], xr[0]) + s[1] * metal::dot(q[1], xr[1]) +
                             s[2] * metal::dot(q[2], xr[2]) + s[3] * metal::dot(q[3], xr[3]));
  }  static inline float4 deq4(device const uchar* b, uint j) {
    const uint h = j >> 7, kk = (j >> 5) & 3, l = j & 31;
    const uint ql = q_ld32(b + 64 * h + 32 * (kk & 1) + l) >> (4 * (kk >> 1));
    const uint qh = q_ld32(b + 128 + 32 * h + l) >> (2 * kk);
    const float sc = float(((device const char*)b)[192 + 8 * h + (l >> 4) + 2 * kk]);
    return (q_ldh(b + 208) * sc) * (q_u8x4((ql & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4)) - 32.f);
  }
};

// ---------------- Q5_K: {half d, dmin; sc[12]; qh[32]; qs[128]} 176 B. 16 lanes; lane = (chunk j = sub>>2,
// l0 = 8*(sub&3)): elements 64j + l0 + i (low nibble) and 64j + 32 + l0 + i (high nibble), i in 0..7 ----------------
struct QB_Q5K {
  static constexpr constant int QK = 256, BB = 176, LPB = 16, EPL = 16, X4 = 4;
  static inline ushort pos(ushort sub, ushort i) { return 64 * (sub >> 2) + 8 * (sub & 3) + (i >> 3) * 32 + (i & 7); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float* p = xb + 64 * (sub >> 2) + 8 * (sub & 3);
    xr[0] = *(device const float4*)(p); xr[1] = *(device const float4*)(p + 4);
    xr[2] = *(device const float4*)(p + 32); xr[3] = *(device const float4*)(p + 36);
  }
  static inline void quants(device const uchar* b, ushort sub, thread float4* q, thread float* sm) {
    const int j = sub >> 2, l0 = 8 * (sub & 3);
    const uint ql0 = q_ld32(b + 48 + 32 * j + l0), ql1 = q_ld32(b + 52 + 32 * j + l0);
    const uint qh0 = q_ld32(b + 16 + l0) >> (2 * j), qh1 = q_ld32(b + 20 + l0) >> (2 * j);
    q[0] = q_u8x4((ql0 & 0x0F0F0F0Fu) | ((qh0 & 0x01010101u) << 4));
    q[1] = q_u8x4((ql1 & 0x0F0F0F0Fu) | ((qh1 & 0x01010101u) << 4));
    q[2] = q_u8x4(((ql0 >> 4) & 0x0F0F0F0Fu) | ((qh0 & 0x02020202u) << 3));
    q[3] = q_u8x4(((ql1 >> 4) & 0x0F0F0F0Fu) | ((qh1 & 0x02020202u) << 3));
    const float d = q_ldh(b), dm = q_ldh(b + 2);
    float s1, m1, s2, m2;
    q_scale_min_k4(2 * j, b + 4, s1, m1); q_scale_min_k4(2 * j + 1, b + 4, s2, m2);
    sm[0] = d * s1; sm[1] = dm * m1; sm[2] = d * s2; sm[3] = dm * m2;
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    float4 q[4]; float sm[4]; quants(b, sub, q, sm);
    w[0] = sm[0] * q[0] - sm[1]; w[1] = sm[0] * q[1] - sm[1]; w[2] = sm[2] * q[2] - sm[3]; w[3] = sm[2] * q[3] - sm[3];
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    float4 q[4]; float sm[4]; quants(b, sub, q, sm);
    const float4 one = 1.f;
    return sm[0] * (metal::dot(q[0], xr[0]) + metal::dot(q[1], xr[1])) - sm[1] * metal::dot(xr[0] + xr[1], one) +
           sm[2] * (metal::dot(q[2], xr[2]) + metal::dot(q[3], xr[3])) - sm[3] * metal::dot(xr[2] + xr[3], one);
  }  // 4 consecutive weights starting at element j (j % 4 == 0): MoE prefill grouped shared expert (SHDENSE=0)
  static inline float4 deq4(device const uchar* b, uint j) {
    const uint c = j >> 6, hi = (j >> 5) & 1, l = j & 31, is = 2 * c + hi;
    const uint ql = q_ld32(b + 48 + 32 * c + l) >> (4 * hi), qh = q_ld32(b + 16 + l) >> is;
    float sc, mn;
    q_scale_min_k4(is, b + 4, sc, mn);
    return (q_ldh(b) * sc) * q_u8x4((ql & 0x0F0F0F0Fu) | ((qh & 0x01010101u) << 4)) - q_ldh(b + 2) * mn;
  }
};

// ---------------- IQ1_S {half d; qs[32]; u16 qh[8]} 50 B and IQ2_XXS {half d; u16 qs[32]} 66 B.
// L lanes per block (16 -> 2 grid groups per lane, 8 -> 4 groups = one sub-block),
// half-valued grid tables (one uint4 constant load per 8 weights, half->float conversion is free) ----
template <int L>
struct QB_IQ1S_T {
  static constexpr constant int QK = 256, BB = 50, LPB = L, EPL = 256 / L, X4 = 256 / L / 4 + 1, G = 256 / L / 8;
  static inline ushort pos(ushort sub, ushort i) { return EPL * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + EPL * sub);
    float4 s = 0;
    for (int k = 0; k < EPL / 4; k++) { xr[k] = p[k]; s += xr[k]; }
    xr[EPL / 4] = float4(metal::dot(s, float4(1.f)), 0, 0, 0);
  }
  static inline uint qsbytes(device const uchar* b, int ib, int l0) {
    return G == 4 ? q_ld32(b + 2 + 4 * ib) : uint(q_ld16(b + 2 + 4 * ib + l0));
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = qsbytes(b, ib, l0);
    const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1), delta = (qh & 0x8000) ? -0.125f : 0.125f;
    for (int g = 0; g < G; g++) {
      const uint4 e = q_tbl4(q_iq1s_gridh, ((qs >> (8 * g)) & 0xFF) | (((qh >> (3 * (l0 + g))) & 7) << 8));
      w[2 * g] = dl * (float4(as_type<half4>(e.xy)) + delta);
      w[2 * g + 1] = dl * (float4(as_type<half4>(e.zw)) + delta);
    }
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = qsbytes(b, ib, l0);
    float s = 0;
    for (int g = 0; g < G; g++) {
      const uint4 e = q_tbl4(q_iq1s_gridh, ((qs >> (8 * g)) & 0xFF) | (((qh >> (3 * (l0 + g))) & 7) << 8));
      s += metal::dot(float4(as_type<half4>(e.xy)), xr[2 * g]) + metal::dot(float4(as_type<half4>(e.zw)), xr[2 * g + 1]);
    }
    const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1);
    return dl * (s + ((qh & 0x8000) ? -0.125f : 0.125f) * xr[EPL / 4].x);
  }  static inline float4 deq4(device const uchar* b, uint j) {
    const uint ib = j >> 5, gq = (j >> 3) & 3;
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = b[2 + 4 * ib + gq];
    const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1), delta = (qh & 0x8000) ? -0.125f : 0.125f;
    const uint2 e = ((constant uint2*)q_iq1s_gridh)[2 * (qs | (((qh >> (3 * gq)) & 7) << 8)) + ((j >> 2) & 1)];
    return dl * (float4(as_type<half4>(e)) + delta);
  }
};

template <int L>
struct QB_IQ2XXS_T {
  static constexpr constant int QK = 256, BB = 66, LPB = L, EPL = 256 / L, X4 = 256 / L / 4, G = 256 / L / 8;
  static inline ushort pos(ushort sub, ushort i) { return EPL * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + EPL * sub);
    for (int k = 0; k < EPL / 4; k++) xr[k] = p[k];
  }
  static inline uint idxbytes(device const uchar* b, int ib, int l0) {
    return G == 4 ? q_ld32(b + 2 + 8 * ib) : uint(q_ld16(b + 2 + 8 * ib + l0));
  }
  static inline uint4 sgrid(uint idx, uint a1, int l) {
    return q_tbl4(q_iq2xxs_gridh, idx) ^ q_tbl4(q_iq2_signh, (a1 >> (7 * l)) & 127);
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = idxbytes(b, ib, l0);
    const float db = q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
    for (int g = 0; g < G; g++) {
      const uint4 e = sgrid((idx >> (8 * g)) & 0xFF, a1, l0 + g);
      w[2 * g] = db * float4(as_type<half4>(e.xy)); w[2 * g + 1] = db * float4(as_type<half4>(e.zw));
    }
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = idxbytes(b, ib, l0);
    float s = 0;
    for (int g = 0; g < G; g++) {
      const uint4 e = sgrid((idx >> (8 * g)) & 0xFF, a1, l0 + g);
      s += metal::dot(float4(as_type<half4>(e.xy)), xr[2 * g]) + metal::dot(float4(as_type<half4>(e.zw)), xr[2 * g + 1]);
    }
    return q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f * s;
  }  static inline float4 deq4(device const uchar* b, uint j) {
    const uint ib = j >> 5, gq = (j >> 3) & 3, hf = (j >> 2) & 1;
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = b[2 + 8 * ib + gq];
    const uint2 e = ((constant uint2*)q_iq2xxs_gridh)[2 * idx + hf] ^ ((constant uint2*)q_iq2_signh)[2 * ((a1 >> (7 * gq)) & 127) + hf];
    return (q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f) * float4(as_type<half4>(e));
  }
};

// ---------------- IQ1_M {qs[32]; qh[16]; u16 sc[4]} 56 B, no separate d: fp16 d = top nibbles of the 4 scale words.
// Sub-block ib (32 elements) = 4 grid groups gg of 8; group gg: index qs[4ib+gg] | (3 bits of its qh nibble) << 8,
// delta sign = bit 3 of that nibble (qh byte 2ib + gg/2, nibble gg&1); scale 2*(3 bits) + 1 of sc[ib/2] at shift
// 6*(ib&1) + 3*(gg>>1) (one scale per 16 elements). Same iq1s grid as IQ1_S. L lanes per block (16: one scale and 2
// groups per lane; 8: one sub-block per lane). dot = dl * sum_g (grid_g . x_g + delta_g * sum(x_g)), per-8 x sums
// precomputed in load_x (xr[EPL/4] = group sums). ----
static inline float q_iq1m_d(device const uchar* b) {
  const uint s01 = q_ld32(b + 48), s23 = q_ld32(b + 52);
  const uint du = ((s01 >> 12) & 0xFu) | ((s01 >> 24) & 0xF0u) | ((s23 >> 4) & 0xF00u) | ((s23 >> 16) & 0xF000u);
  return float(as_type<half>(ushort(du)));
}
template <int L>
struct QB_IQ1M_T {
  static constexpr constant int QK = 256, BB = 56, LPB = L, EPL = 256 / L, X4 = 256 / L / 4 + 1, G = 256 / L / 8;
  static inline ushort pos(ushort sub, ushort i) { return EPL * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + EPL * sub);
    float4 s = 0;
    for (int g = 0; g < G; g++) {
      xr[2 * g] = p[2 * g]; xr[2 * g + 1] = p[2 * g + 1];
      s[g] = metal::dot(xr[2 * g] + xr[2 * g + 1], float4(1.f));
    }
    xr[EPL / 4] = s;
  }
  static inline uint qsbytes(device const uchar* b, int ib, int g0) {
    return G == 4 ? q_ld32(b + 4 * ib) : uint(q_ld16(b + 4 * ib + g0));
  }
  // lane geometry: sub-block ib, first group g0; qh nibbles of the sub-block (nibble gg = group gg); scale word
  static inline void geo(device const uchar* b, ushort sub, thread int& ib, thread int& g0, thread uint& qh,
                         thread uint& qs, thread uint& sc) {
    ib = (EPL * sub) >> 5; g0 = ((EPL * sub) & 31) >> 3;
    qh = q_ld16(b + 32 + 2 * ib); qs = qsbytes(b, ib, g0); sc = q_ld16(b + 48 + 2 * (ib >> 1)) >> (6 * (ib & 1));
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    int ib, g0; uint qh, qs, sc;
    geo(b, sub, ib, g0, qh, qs, sc);
    const float d = q_iq1m_d(b);
    for (int g = 0; g < G; g++) {
      const int gg = g0 + g;
      const uint nb = (qh >> (4 * gg)) & 15;
      const float dl = d * float(2 * ((sc >> (3 * (gg >> 1))) & 7) + 1), delta = (nb & 8) ? -0.125f : 0.125f;
      const uint4 e = q_tbl4(q_iq1s_gridh, ((qs >> (8 * g)) & 0xFF) | ((nb & 7) << 8));
      w[2 * g] = dl * (float4(as_type<half4>(e.xy)) + delta);
      w[2 * g + 1] = dl * (float4(as_type<half4>(e.zw)) + delta);
    }
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    int ib, g0; uint qh, qs, sc;
    geo(b, sub, ib, g0, qh, qs, sc);
    float s[2] = {0.f, 0.f};  // per scale half (16 elements)
    for (int g = 0; g < G; g++) {
      const int gg = g0 + g;
      const uint nb = (qh >> (4 * gg)) & 15;
      const uint4 e = q_tbl4(q_iq1s_gridh, ((qs >> (8 * g)) & 0xFF) | ((nb & 7) << 8));
      s[g >> 1] += metal::dot(float4(as_type<half4>(e.xy)), xr[2 * g]) + metal::dot(float4(as_type<half4>(e.zw)), xr[2 * g + 1]) +
                   ((nb & 8) ? -0.125f : 0.125f) * xr[EPL / 4][g];
    }
    float r = float(2 * ((sc >> (3 * (g0 >> 1))) & 7) + 1) * s[0];
    if (G == 4) r += float(2 * ((sc >> 3) & 7) + 1) * s[1];
    return q_iq1m_d(b) * r;
  }
  static inline float4 deq4(device const uchar* b, uint j) {
    const uint ib = j >> 5, gg = (j >> 3) & 3;
    const uint nb = (b[32 + 2 * ib + (gg >> 1)] >> (4 * (gg & 1))) & 15;
    const uint sc = q_ld16(b + 48 + 2 * (ib >> 1)) >> (6 * (ib & 1) + 3 * (gg >> 1));
    const float dl = q_iq1m_d(b) * float(2 * (sc & 7) + 1), delta = (nb & 8) ? -0.125f : 0.125f;
    const uint2 e = ((constant uint2*)q_iq1s_gridh)[2 * (b[4 * ib + gg] | ((nb & 7) << 8)) + ((j >> 2) & 1)];
    return dl * (float4(as_type<half4>(e)) + delta);
  }
};

// ---------------- IQ2_XS {half d; u16 qs[32]; scales[8]} 74 B. Group l of sub-block ib: q = qs[4ib + l], grid
// iq2xs_grid[q & 511] (8 x {8,25,43}), signs ksigns[q >> 9] (same sign table as IQ2_XXS); scale
// d * (0.5 + nibble) * 0.25, nibble (l >> 1) of scales[ib] (one scale per 16 elements). L lanes per block. ----
template <int L>
struct QB_IQ2XS_T {
  static constexpr constant int QK = 256, BB = 74, LPB = L, EPL = 256 / L, X4 = 256 / L / 4, G = 256 / L / 8;
  static inline ushort pos(ushort sub, ushort i) { return EPL * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + EPL * sub);
    for (int k = 0; k < EPL / 4; k++) xr[k] = p[k];
  }
  static inline uint4 sgrid(uint q) { return q_tbl4(q_iq2xs_gridh, q & 511) ^ q_tbl4(q_iq2_signh, q >> 9); }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    const int e0 = EPL * sub, ib = e0 >> 5, g0 = (e0 & 31) >> 3;
    const float d = q_ldh(b);
    const uint scb = b[66 + ib];
    for (int g = 0; g < G; g += 2) {
      const uint qq = q_ld32(b + 2 + 2 * (4 * ib + g0 + g));  // two groups sharing one scale nibble
      const float db = d * (0.5f + float((scb >> (2 * (g0 + g))) & 15)) * 0.25f;
      const uint4 e0v = sgrid(qq & 0xFFFF), e1v = sgrid(qq >> 16);
      w[2 * g] = db * float4(as_type<half4>(e0v.xy)); w[2 * g + 1] = db * float4(as_type<half4>(e0v.zw));
      w[2 * g + 2] = db * float4(as_type<half4>(e1v.xy)); w[2 * g + 3] = db * float4(as_type<half4>(e1v.zw));
    }
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    const int e0 = EPL * sub, ib = e0 >> 5, g0 = (e0 & 31) >> 3;
    const uint scb = b[66 + ib];
    float r = 0;
    for (int g = 0; g < G; g += 2) {
      const uint qq = q_ld32(b + 2 + 2 * (4 * ib + g0 + g));
      const uint4 e0v = sgrid(qq & 0xFFFF), e1v = sgrid(qq >> 16);
      const float s = metal::dot(float4(as_type<half4>(e0v.xy)), xr[2 * g]) + metal::dot(float4(as_type<half4>(e0v.zw)), xr[2 * g + 1]) +
                      metal::dot(float4(as_type<half4>(e1v.xy)), xr[2 * g + 2]) + metal::dot(float4(as_type<half4>(e1v.zw)), xr[2 * g + 3]);
      r += (0.5f + float((scb >> (2 * (g0 + g))) & 15)) * s;
    }
    return q_ldh(b) * 0.25f * r;
  }
  static inline float4 deq4(device const uchar* b, uint j) {
    const uint ib = j >> 5, gq = (j >> 3) & 3, hf = (j >> 2) & 1;
    const uint q = q_ld16(b + 2 + 2 * (4 * ib + gq));
    const uint2 e = ((constant uint2*)q_iq2xs_gridh)[2 * (q & 511) + hf] ^ ((constant uint2*)q_iq2_signh)[2 * (q >> 9) + hf];
    return (q_ldh(b) * (0.5f + float((b[66 + ib] >> (4 * (gq >> 1))) & 15)) * 0.25f) * float4(as_type<half4>(e));
  }
};

// production choice (bench_qmatvec: 16 lanes beat 8 on the expert shapes; half tables beat int8 + convert by ~30%)
typedef QB_IQ1S_T<16> QB_IQ1S;
typedef QB_IQ2XXS_T<16> QB_IQ2XXS;
typedef QB_IQ1M_T<16> QB_IQ1M;
typedef QB_IQ2XS_T<16> QB_IQ2XS;
