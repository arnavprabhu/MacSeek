// Shared helpers for the attention module (attn_*.metal). See docs/modules/attn.md.
#pragma once
#include "common.h"

// ---- model constants ----
constant constexpr int ATTN_D = 512;        // head dim == latent dim
constant constexpr int ATTN_RD = 64;        // rope dims (last 64)
constant constexpr int ATTN_H = 64;         // attention heads
constant constexpr int ATTN_W = 128;        // window ring
constant constexpr int ATTN_IH = 32;        // indexer heads
constant constexpr int ATTN_ID = 128;       // indexer head dim

// ---- simdgroup 8x8 fragment layout (verified by bench/attn_layout_probe): lane holds (row fm, cols fn, fn+1) ----
inline ushort attn_frag_row(ushort lane) { return ((lane >> 2) & 4) + ((lane >> 1) & 3); }
inline ushort attn_frag_col(ushort lane) { return ((lane >> 2) & 2) * 2 + (lane & 1) * 2; }

// ---- exact float formats used by the reference quant simulation (kernel.py act_quant / fp4_act_quant) ----
inline float attn_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }  // e in [-126, 127]
inline int attn_floor_log2(float a) { return int((as_type<uint>(a) >> 23) & 0xFF) - 127; }  // a >= 0 normal
// ceil(log2(x)) by bit manipulation (kernel.py fast_log2_ceil)
inline int attn_log2_ceil(float x) {
  uint b = as_type<uint>(x);
  return int((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
}
// round |x| <= 448 to the nearest e4m3fn value (ties to even), incl. subnormals (step 2^-9)
inline float attn_round_e4m3(float x) {
  float a = fabs(x);
  int e = max(attn_floor_log2(a), -6);
  float r = rint(a * attn_pow2(3 - e)) * attn_pow2(e - 3);
  return copysign(r, x);
}
// round |x| <= 6 to the nearest e2m1 value {0,.5,1,1.5,2,3,4,6} (ties to even)
inline float attn_round_e2m1(float x) {
  float a = fabs(x);
  int e = max(attn_floor_log2(a), 0);
  float r = rint(a * attn_pow2(1 - e)) * attn_pow2(e - 1);
  return copysign(r, x);
}
// block scale for fp8 act_quant with ue8m0 rounding: 2^ceil(log2(max(amax,1e-4)/448))
inline float attn_fp8_scale(float amax) { return attn_pow2(attn_log2_ceil(max(amax, 1e-4f) * (1.0f / 448.0f))); }
inline float attn_fp8_q(float x, float s, float inv_s) { return attn_round_e4m3(clamp(x * inv_s, -448.0f, 448.0f)) * s; }
// fp4 with ue8m0 scale (indexer q / keys, block 32)
inline float attn_fp4e8_scale(float amax) { return attn_pow2(attn_log2_ceil(max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f))); }
inline float attn_fp4e8_q(float x, float s, float inv_s) { return attn_round_e2m1(clamp(x * inv_s, -6.0f, 6.0f)) * s; }
// fp4 with e4m3 scale (compressed KV, block 16)
inline float attn_fp4e4_scale(float amax) { return attn_round_e4m3(min(precise::divide(max(amax, 6.0f * 0x1p-9f), 6.0f), 448.0f)); }
inline float attn_fp4e4_q(float x, float s) { return attn_round_e2m1(clamp(precise::divide(x, s), -6.0f, 6.0f)) * s; }

// ---- order-preserving float key: larger float -> larger uint. 0 is reserved for "invalid" (-inf / masked). ----
inline uint attn_fkey(float f) {
  uint u = as_type<uint>(f);
  uint k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (f == -INFINITY || isnan(f)) ? 0u : k;
}
inline float attn_unkey(uint k) {
  uint u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return as_type<float>(u);
}

// rope on one interleaved pair with table entry cs = (cos, sin)
inline float2 attn_rot(float2 x, float2 cs) { return float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x); }
inline float2 attn_rot_inv(float2 x, float2 cs) { return float2(x.x * cs.x + x.y * cs.y, -x.x * cs.y + x.y * cs.x); }
