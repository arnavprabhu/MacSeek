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
  }
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
  }
};

// ---------------- IQ1_S: {half d; qs[32]; u16 qh[8]} 50 B. 16 lanes; lane = (sub-block ib = sub>>1, half hf):
// 16 contiguous elements 32ib + 16hf + i = 2 grid groups. xr[4].x holds sum(x) for the delta term ----------------
struct QB_IQ1S {
  static constexpr constant int QK = 256, BB = 50, LPB = 16, EPL = 16, X4 = 5;
  static inline ushort pos(ushort sub, ushort i) { return 16 * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + 16 * sub);
    xr[0] = p[0]; xr[1] = p[1]; xr[2] = p[2]; xr[3] = p[3];
    xr[4] = float4(metal::dot((xr[0] + xr[1]) + (xr[2] + xr[3]), float4(1.f)), 0, 0, 0);
  }
  static inline void grid(device const uchar* b, ushort sub, thread uint2& g0, thread uint2& g1, thread float& dl,
                          thread float& delta) {
    const int ib = sub >> 1, l = 2 * (sub & 1);
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = q_ld16(b + 2 + 4 * ib + 2 * (sub & 1));
    g0 = as_type<uint2>(q_iq1s_grid[(qs & 0xFF) | (((qh >> (3 * l)) & 7) << 8)]);
    g1 = as_type<uint2>(q_iq1s_grid[(qs >> 8) | (((qh >> (3 * l + 3)) & 7) << 8)]);
    dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1);
    delta = (qh & 0x8000) ? -0.125f : 0.125f;
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    uint2 g0, g1; float dl, delta; grid(b, sub, g0, g1, dl, delta);
    w[0] = dl * (q_i8x4(g0.x) + delta); w[1] = dl * (q_i8x4(g0.y) + delta);
    w[2] = dl * (q_i8x4(g1.x) + delta); w[3] = dl * (q_i8x4(g1.y) + delta);
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    uint2 g0, g1; float dl, delta; grid(b, sub, g0, g1, dl, delta);
    const float s = metal::dot(q_i8x4(g0.x), xr[0]) + metal::dot(q_i8x4(g0.y), xr[1]) +
                    metal::dot(q_i8x4(g1.x), xr[2]) + metal::dot(q_i8x4(g1.y), xr[3]);
    return dl * (s + delta * xr[4].x);
  }
};

// ---------------- IQ2_XXS: {half d; u16 qs[32]} 66 B. 16 lanes; lane = (ib = sub>>1, hf): 2 grid groups ----------------
struct QB_IQ2XXS {
  static constexpr constant int QK = 256, BB = 66, LPB = 16, EPL = 16, X4 = 4;
  static inline ushort pos(ushort sub, ushort i) { return 16 * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + 16 * sub);
    xr[0] = p[0]; xr[1] = p[1]; xr[2] = p[2]; xr[3] = p[3];
  }
  // signed grid bytes: g ^ m + (m & 1) negates the bytes selected by m (grid bytes are >= 8, so no carries)
  static inline void grid(device const uchar* b, ushort sub, thread uint2& s0, thread uint2& s1, thread float& db) {
    const int ib = sub >> 1, l = 2 * (sub & 1);
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = q_ld16(b + 2 + 8 * ib + 2 * (sub & 1));
    const ulong m0 = q_iq2_signmask[(a1 >> (7 * l)) & 127], m1 = q_iq2_signmask[(a1 >> (7 * l + 7)) & 127];
    const ulong g0 = q_iq2xxs_grid[idx & 0xFF], g1 = q_iq2xxs_grid[idx >> 8];
    s0 = as_type<uint2>((g0 ^ m0) + (m0 & 0x0101010101010101ul));
    s1 = as_type<uint2>((g1 ^ m1) + (m1 & 0x0101010101010101ul));
    db = q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    uint2 s0, s1; float db; grid(b, sub, s0, s1, db);
    w[0] = db * q_i8x4(s0.x); w[1] = db * q_i8x4(s0.y); w[2] = db * q_i8x4(s1.x); w[3] = db * q_i8x4(s1.y);
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    uint2 s0, s1; float db; grid(b, sub, s0, s1, db);
    return db * (metal::dot(q_i8x4(s0.x), xr[0]) + metal::dot(q_i8x4(s0.y), xr[1]) +
                 metal::dot(q_i8x4(s1.x), xr[2]) + metal::dot(q_i8x4(s1.y), xr[3]));
  }
};
