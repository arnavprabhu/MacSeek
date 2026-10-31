// Attention module: prefill sparse attention on the tensor units (Metal 4 matmul2d), two-pass form per query block:
//   attn_pf_qprep   : Qh[q][h][:] = half(rope(q))
//   attn_pf_scores  : S[q][h][r] = scale * Qh[q][h] . row_r(q)   (64 heads x 64 rows tiles, K chunks of 64 dims)
//   attn_pf_softmax : P[q][h][r] = exp(S - m) / (sum exp(S - m) + exp(sink_h - m))   (half)
//   attn_pf_pv      : out[q][h][d] = inverse_rope(sum_r P[q][h][r] row_r(q)[d])        (64 heads x 64 dims tiles)
// Row r of query q: r < 128 -> window position p-127+r (chunk row if >= chunk_start, else ring), else topk[q][r-128].
#include "attn_common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct AttnPfArgs {
  int start;        // absolute position of query 0 of this block
  int nq;           // queries in this block
  int q0;           // index of query 0 of this block within the chunk (topk / chunk rows are chunk-indexed)
  int chunk_start;  // absolute position of chunk row 0
  int ntop;         // compressed rows per query (0 or 512)
  int R;            // rows per query = 128 + ntop (multiple of 64)
  float scale;
};

inline device const half* attn_pf_row(constant AttnPfArgs& a, int qb, int r, device const half* ring,
                                      device const half* chunk, device const half* ckv, device const int* topk) {
  const int p = a.start + qb;
  if (r < ATTN_W) {
    int wp = p - (ATTN_W - 1) + r;
    if (wp < 0) return nullptr;
    return wp >= a.chunk_start ? chunk + (size_t)(wp - a.chunk_start) * ATTN_D : ring + (size_t)(wp % ATTN_W) * ATTN_D;
  }
  int id = topk[(size_t)(a.q0 + qb) * a.ntop + (r - ATTN_W)];
  return id >= 0 ? ckv + (size_t)id * ATTN_D : nullptr;
}

// grid (nq), 128 threads: Qh half [nq][64][512] with rope on dims 448..511
kernel void attn_pf_qprep(device const float* q [[buffer(0)]], device const float2* tab [[buffer(1)]],
                          device half* Qh [[buffer(2)]], constant AttnPfArgs& a [[buffer(3)]],
                          uint qb [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  const int p = a.start + int(qb);
  device const float4* src = (device const float4*)(q + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D);
  device half4* dst = (device half4*)(Qh + (size_t)qb * ATTN_H * ATTN_D);
  for (int i = t; i < ATTN_H * 128; i += 128) {
    int c4 = i & 127;
    float4 x = src[i];
    if (c4 >= 112) {
      int pi = 2 * (c4 - 112);
      x.xy = attn_rot(x.xy, tab[(size_t)p * 32 + pi]);
      x.zw = attn_rot(x.zw, tab[(size_t)p * 32 + pi + 1]);
    }
    dst[i] = half4(x);
  }
}

// grid (R/64, nq), 128 threads. S f32 [nq][64][R]
kernel void attn_pf_scores(device half* Qh [[buffer(0)]], device const half* ring [[buffer(1)]],
                           device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                           device const int* topk [[buffer(4)]], device float* S [[buffer(5)]],
                           constant AttnPfArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Kt[64 * 64];
  threadgroup int vf[64];
  const int qb = int(tg.y), r0 = int(tg.x) * 64;
  const int rl = t >> 1, part = t & 1;
  device const half* src = attn_pf_row(a, qb, r0 + rl, ring, chunk, ckv, topk);
  if (part == 0) vf[rl] = src != nullptr;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qh, dextents<int32_t, 2>(ATTN_D, a.nq * ATTN_H));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kt, dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (int d0 = 0; d0 < ATTN_D; d0 += 64) {
    threadgroup uint4* dst = (threadgroup uint4*)(Kt + rl * 64 + part * 32);
    if (src) {
      device const uint4* s4 = (device const uint4*)(src + d0 + part * 32);
      for (int j = 0; j < 4; j++) dst[j] = s4[j];
    } else {
      for (int j = 0; j < 4; j++) dst[j] = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mA = tA.template slice<64, 64>(d0, qb * ATTN_H);
    op.run(mA, tB, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device float* Sq = S + (size_t)qb * ATTN_H * a.R;
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      int col = id[0], h = id[1];
      Sq[(size_t)h * a.R + r0 + col] = vf[col] ? cT[i] * a.scale : -INFINITY;
    }
  }
}

// grid (nq), 1024 threads (32 simdgroups x 2 heads). P half [nq][64][R]
kernel void attn_pf_softmax(device const float* S [[buffer(0)]], device const float* sinks [[buffer(1)]],
                            device half* P [[buffer(2)]], constant AttnPfArgs& a [[buffer(3)]],
                            uint qb [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort ln [[thread_index_in_simdgroup]]) {
  for (int hh = 0; hh < 2; hh++) {
    const int h = sg * 2 + hh;
    device const float* s = S + ((size_t)qb * ATTN_H + h) * a.R;
    device half* pp = P + ((size_t)qb * ATTN_H + h) * a.R;
    float m = sinks[h];
    for (int r = ln; r < a.R; r += 32) m = max(m, s[r]);
    m = simd_max(m);
    float l = 0;
    for (int r = ln; r < a.R; r += 32) l += exp(s[r] - m);
    l = simd_sum(l) + exp(sinks[h] - m);
    const float inv = 1.0f / l;
    for (int r = ln; r < a.R; r += 32) pp[r] = half(exp(s[r] - m) * inv);
  }
}

// grid (8, nq), 128 threads: out[q][h][d0..d0+63] for d0 = 64*tg.x
kernel void attn_pf_pv(device half* P [[buffer(0)]], device const half* ring [[buffer(1)]],
                       device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                       device const int* topk [[buffer(4)]], device const float2* tab [[buffer(5)]],
                       device float* out [[buffer(6)]], constant AttnPfArgs& a [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Vt[64 * 64];
  threadgroup float Ot[64 * 64];
  const int qb = int(tg.y), d0 = int(tg.x) * 64;
  const int rl = t >> 1, part = t & 1;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(P, dextents<int32_t, 2>(a.R, a.nq * ATTN_H));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt, dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (int r0 = 0; r0 < a.R; r0 += 64) {
    device const half* src = attn_pf_row(a, qb, r0 + rl, ring, chunk, ckv, topk);
    threadgroup uint4* dst = (threadgroup uint4*)(Vt + rl * 64 + part * 32);
    if (src) {
      device const uint4* s4 = (device const uint4*)(src + d0 + part * 32);
      for (int j = 0; j < 4; j++) dst[j] = s4[j];
    } else {
      for (int j = 0; j < 4; j++) dst[j] = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mA = tA.template slice<64, 64>(r0, qb * ATTN_H);
    op.run(mA, tB, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      Ot[id[1] * 64 + id[0]] = cT[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int p = a.start + qb;
  device float* oq = out + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D;
  for (int i = t; i < 64 * 32; i += 128) {  // (head, dim pair)
    int h = i >> 5, dp = (i & 31) * 2;
    float2 v = float2(Ot[h * 64 + dp], Ot[h * 64 + dp + 1]);
    int d = d0 + dp;
    if (d >= ATTN_D - ATTN_RD) v = attn_rot_inv(v, tab[(size_t)p * 32 + (d - (ATTN_D - ATTN_RD)) / 2]);
    *(device float2*)(oq + (size_t)h * ATTN_D + d) = v;
  }
}
