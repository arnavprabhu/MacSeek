// Attention module: prefill sparse attention, two-pass on the tensor units with the softmax folded into both passes
// (no f32 score matrix, no separate softmax dispatch):
//   attn_pf2_scores : per (64-row tile t, query): S = scale * Qh . rows^T (64 heads x 64 rows), per-head tile max m_t and
//                     sum l_t = sum exp(S - m_t); writes P_t = exp(S - m_t) (half, <= 1) and ML[q][t][h] = (m_t, l_t)
//   attn_pf2_pv     : per (64-dim slice, query): M = max(sink, m_t), L = exp(sink - M) + sum l_t exp(m_t - M);
//                     out = inverse_rope(sum_t (exp(m_t - M) / L) * (P_t . rows_t))
// Tiles whose 64 rows are all invalid (shallow depth, -1 top-k padding) are skipped in both passes.
// Row gathers are double-buffered (global loads of chunk k+1 are issued before the matmul of chunk k).
// Row r of query q: r < 128 -> window position p-127+r (chunk row if >= chunk_start, else ring), else topk[q][r-128].
#include "attn_common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct AttnPf2Args {
  int start;        // absolute position of query 0 of this block
  int nq;           // queries in this block
  int q0;           // index of query 0 of this block within the chunk
  int chunk_start;  // absolute position of chunk row 0
  int ntop;         // compressed rows per query (0 or 512)
  int R;            // rows per query = 128 + ntop (multiple of 64)
  float scale;
};

inline device const half* attn_pf2_rowp(constant AttnPf2Args& a, int qb, int r, device const half* ring,
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

// cooperative destination layout of matmul2d(64, 64, K) on 4 simdgroups (same family as attn_idx_score_pf2; checked at
// init): element i -> row = R + 8*((i>>2)&1) + 32*((i>>4)&1), col = C + (i&3) + 32*((i>>3)&1)
inline int attn_pf2_r(ushort sg, ushort l) { return ((l >> 1) & 3) + 4 * ((l >> 4) & 1) + 16 * (sg >> 1); }
inline int attn_pf2_c(ushort sg, ushort l) { return 4 * (l & 1) + 8 * ((l >> 3) & 1) + 16 * (sg & 1); }
inline int attn_pf2_ri(int i) { return 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1); }
inline int attn_pf2_ci(int i) { return (i & 3) + 32 * ((i >> 3) & 1); }

// grid (R/64, nq), 128 threads. P half [nq][64][R], ML float2 [nq][R/64][64].
// A = Qh (half, roped, from attn_pf_qprep). (An f32-A variant reading the raw q was tried: slower, and the f32-A
// destination layout differs from the one the epilogue assumes.)
inline void attn_pf2_scores_impl(device half* Qh, device const half* ring, device const half* chunk,
                                 device const half* ckv, device const int* topk, device half* P, device float2* ML,
                                 constant AttnPf2Args& a, uint2 tg, ushort t, ushort sg, ushort l,
                                 threadgroup half (&Kt)[2][64 * 64], threadgroup int* vf, threadgroup float (&red)[2][64],
                                 threadgroup int& anyv) {
  const int qb = int(tg.y), r0 = int(tg.x) * 64, nt = a.R / 64;
  const int rl = t >> 1, part = t & 1;
  device const half* src = attn_pf2_rowp(a, qb, r0 + rl, ring, chunk, ckv, topk);
  if (t == 0) anyv = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (part == 0) { vf[rl] = src != nullptr; if (src) anyv = 1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float2* mlq = ML + ((size_t)qb * nt + tg.x) * ATTN_H;
  if (!anyv) {  // whole tile invalid: P never read for it (pass 2 skips tiles with m = -inf)
    if (t < 64) mlq[t] = float2(-INFINITY, 0.0f);
    return;
  }
  const int arow0 = qb * ATTN_H;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qh, dextents<int32_t, 2>(ATTN_D, a.nq * ATTN_H));
  auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kt[0], dextents<int32_t, 2>(64, 64));
  auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kt[1], dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, arow0);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 32; ++i) cT[i] = 0;
  uint4 reg[4];
  auto load = [&](int d0) {
    if (src) { device const uint4* s4 = (device const uint4*)(src + d0 + part * 32); for (int j = 0; j < 4; j++) reg[j] = s4[j]; }
    else for (int j = 0; j < 4; j++) reg[j] = uint4(0);
  };
  load(0);
  for (int k = 0; k < ATTN_D / 64; k++) {
    threadgroup uint4* dst = (threadgroup uint4*)(Kt[k & 1] + rl * 64 + part * 32);
    for (int j = 0; j < 4; j++) dst[j] = reg[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 1 < ATTN_D / 64) load((k + 1) * 64);
    auto mA = tA.template slice<64, 64>(k * 64, arow0);
    if (k & 1) op.run(mA, tB1, cT); else op.run(mA, tB0, cT);
  }
  // per-head max over the tile's 64 rows (cols of the tile)
  const int R0 = attn_pf2_r(sg, l), C0 = attn_pf2_c(sg, l);
  float m4[4];
  #pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    float m = -INFINITY;
    #pragma unroll
    for (int x = 0; x < 8; x++) {
      const int i = (x & 3) | ((hh & 1) << 2) | ((x >> 2) << 3) | ((hh >> 1) << 4);
      const int col = C0 + attn_pf2_ci(i);
      float s = vf[col] ? cT[i] * a.scale : -INFINITY;
      cT[i] = s;
      m = max(m, s);
    }
    m = max(m, simd_shuffle_xor(m, 1));
    m = max(m, simd_shuffle_xor(m, 8));
    m4[hh] = m;
  }
  const bool wr = (l & 9) == 0;
  if (wr) for (int hh = 0; hh < 4; hh++) red[sg & 1][R0 + 8 * (hh & 1) + 32 * (hh >> 1)] = m4[hh];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int hh = 0; hh < 4; hh++) { const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1); m4[hh] = max(red[0][h], red[1][h]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device half* Pq = P + (size_t)qb * ATTN_H * a.R + r0;
  float s4[4];
  #pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1);
    float sum = 0;
    #pragma unroll
    for (int jx = 0; jx < 2; jx++) {
      half4 pv;
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int i = c | ((hh & 1) << 2) | (jx << 3) | ((hh >> 1) << 4);
        float e = exp(cT[i] - m4[hh]);  // -inf -> 0 (m finite: the tile has a valid row)
        sum += e;
        pv[c] = half(e);
      }
      *(device half4*)(Pq + (size_t)h * a.R + C0 + 32 * jx) = pv;
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 8);
    s4[hh] = sum;
  }
  if (wr) for (int hh = 0; hh < 4; hh++) red[sg & 1][R0 + 8 * (hh & 1) + 32 * (hh >> 1)] = s4[hh];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (wr && (sg & 1) == 0)
    for (int hh = 0; hh < 4; hh++) {
      const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1);
      mlq[h] = float2(m4[hh], red[0][h] + red[1][h]);
    }
}

kernel void attn_pf2_scores(device half* Qh [[buffer(0)]], device const half* ring [[buffer(1)]],
                            device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                            device const int* topk [[buffer(4)]], device half* P [[buffer(5)]],
                            device float2* ML [[buffer(6)]], constant AttnPf2Args& a [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half Kt[2][64 * 64];
  threadgroup int vf[64];
  threadgroup float red[2][64];
  threadgroup int anyv;
  attn_pf2_scores_impl(Qh, ring, chunk, ckv, topk, P, ML, a, tg, t, sg, l, Kt, vf, red, anyv);
}
// grid (8, nq), 128 threads: out[q][h][d0..d0+63] for d0 = 64*tg.x
kernel void attn_pf2_pv(device half* P [[buffer(0)]], device const half* ring [[buffer(1)]],
                        device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                        device const int* topk [[buffer(4)]], device const float2* tab [[buffer(5)]],
                        device float* out [[buffer(6)]], device const float2* ML [[buffer(7)]],
                        device const float* sinks [[buffer(8)]], constant AttnPf2Args& a [[buffer(9)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half Vt[2][64 * 64];
  threadgroup float al[10][64];  // alpha[tile][head] (R <= 640)
  threadgroup int tv[10];         // tile has a valid row
  const int qb = int(tg.y), d0 = int(tg.x) * 64, nt = a.R / 64;
  const int rl = t >> 1, part = t & 1;
  device const float2* mlq = ML + (size_t)qb * nt * ATTN_H;
  if (t < 64) {
    float M = sinks[t];
    for (int k = 0; k < nt; k++) M = max(M, mlq[k * ATTN_H + t].x);
    float L = exp(sinks[t] - M);
    for (int k = 0; k < nt; k++) { float2 v = mlq[k * ATTN_H + t]; L += v.y * exp(v.x - M); }
    const float inv = 1.0f / L;
    for (int k = 0; k < nt; k++) { float mk = mlq[k * ATTN_H + t].x; al[k][t] = mk == -INFINITY ? 0.0f : exp(mk - M) * inv; }
    if (t < nt) tv[t] = mlq[t * ATTN_H].x != -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(P, dextents<int32_t, 2>(a.R, a.nq * ATTN_H));
  auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt[0], dextents<int32_t, 2>(64, 64));
  auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt[1], dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto acc = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  auto tmp = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 32; ++i) acc[i] = 0;
  const int R0 = attn_pf2_r(sg, l);
  // valid tiles (uniform per tile: all heads share row validity)
  uint4 reg[4];
  auto load = [&](int k) {
    device const half* src = attn_pf2_rowp(a, qb, k * 64 + rl, ring, chunk, ckv, topk);
    if (src) { device const uint4* s4 = (device const uint4*)(src + d0 + part * 32); for (int j = 0; j < 4; j++) reg[j] = s4[j]; }
    else for (int j = 0; j < 4; j++) reg[j] = uint4(0);
  };
  auto next_valid = [&](int k) { while (k < nt && !tv[k]) k++; return k; };
  int k = next_valid(0);
  if (k < nt) load(k);
  int buf = 0;
  while (k < nt) {
    const int kn = next_valid(k + 1);
    threadgroup uint4* dst = (threadgroup uint4*)(Vt[buf] + rl * 64 + part * 32);
    for (int j = 0; j < 4; j++) dst[j] = reg[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kn < nt) load(kn);
    auto mA = tA.template slice<64, 64>(k * 64, qb * ATTN_H);
    if (buf) op.run(mA, tB1, tmp); else op.run(mA, tB0, tmp);
    #pragma unroll
    for (uint16_t i = 0; i < 32; ++i) acc[i] += al[k][R0 + attn_pf2_ri(i)] * tmp[i];
    buf ^= 1;
    k = kn;
  }
  const int C0 = attn_pf2_c(sg, l);
  const int p = a.start + qb;
  device float* oq = out + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D;
  #pragma unroll
  for (int g = 0; g < 8; g++) {  // groups of 4 consecutive dims: i = 4g .. 4g+3
    const int i = 4 * g, h = R0 + attn_pf2_ri(i), d = d0 + C0 + attn_pf2_ci(i);
    float4 v = float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
    if (d >= ATTN_D - ATTN_RD) {
      const int pi = (d - (ATTN_D - ATTN_RD)) / 2;
      v.xy = attn_rot_inv(v.xy, tab[(size_t)p * 32 + pi]);
      v.zw = attn_rot_inv(v.zw, tab[(size_t)p * 32 + pi + 1]);
    }
    *(device float4*)(oq + (size_t)h * ATTN_D + d) = v;
  }
}

// Variant of attn_pf2_pv with 128-dim slices (one 64x128 destination per threadgroup: P tiles are read half as often).
// grid (4, nq), 128 threads. Destination layout (as attn_idx_score_pf2): element i -> row = R + 8*((i>>2)&1) + 32*(i>>5),
// col = C + (i&3) + 32*((i>>3)&3). alpha computed per thread for its 4 heads.
kernel void attn_pf2_pv128(device half* P [[buffer(0)]], device const half* ring [[buffer(1)]],
                           device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                           device const int* topk [[buffer(4)]], device const float2* tab [[buffer(5)]],
                           device float* out [[buffer(6)]], device const float2* ML [[buffer(7)]],
                           device const float* sinks [[buffer(8)]], constant AttnPf2Args& a [[buffer(9)]],
                           uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half Vt[2][64 * 128];
  const int qb = int(tg.y), d0 = int(tg.x) * 128, nt = a.R / 64;
  const int rl = t >> 1, part = t & 1;
  device const float2* mlq = ML + (size_t)qb * nt * ATTN_H;
  const int R0 = attn_pf2_r(sg, l);
  float Mh[4], iL[4];
  #pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1);
    float M = sinks[h];
    for (int k = 0; k < nt; k++) M = max(M, mlq[k * ATTN_H + h].x);
    float L = exp(sinks[h] - M);
    for (int k = 0; k < nt; k++) { float2 v = mlq[k * ATTN_H + h]; L += v.y * exp(v.x - M); }  // v.x=-inf -> 0
    Mh[hh] = M; iL[hh] = 1.0f / L;
  }
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(P, dextents<int32_t, 2>(a.R, a.nq * ATTN_H));
  auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt[0], dextents<int32_t, 2>(128, 64));
  auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt[1], dextents<int32_t, 2>(128, 64));
  constexpr auto desc = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto acc = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  auto tmp = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 64; ++i) acc[i] = 0;
  uint4 reg[8];
  auto load = [&](int k) {  // row rl, dims d0 + part*64 .. +63
    device const half* src = attn_pf2_rowp(a, qb, k * 64 + rl, ring, chunk, ckv, topk);
    if (src) { device const uint4* s4 = (device const uint4*)(src + d0 + part * 64); for (int j = 0; j < 8; j++) reg[j] = s4[j]; }
    else for (int j = 0; j < 8; j++) reg[j] = uint4(0);
  };
  auto next_valid = [&](int k) { while (k < nt && mlq[k * ATTN_H].x == -INFINITY) k++; return k; };
  int k = next_valid(0);
  if (k < nt) load(k);
  int buf = 0;
  while (k < nt) {
    const int kn = next_valid(k + 1);
    threadgroup uint4* dst = (threadgroup uint4*)(Vt[buf] + rl * 128 + part * 64);
    for (int j = 0; j < 8; j++) dst[j] = reg[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kn < nt) load(kn);
    float al[4];
    #pragma unroll
    for (int hh = 0; hh < 4; hh++) al[hh] = exp(mlq[k * ATTN_H + R0 + 8 * (hh & 1) + 32 * (hh >> 1)].x - Mh[hh]) * iL[hh];
    auto mA = tA.template slice<64, 64>(k * 64, qb * ATTN_H);
    if (buf) op.run(mA, tB1, tmp); else op.run(mA, tB0, tmp);
    #pragma unroll
    for (uint16_t i = 0; i < 64; ++i) acc[i] += al[((i >> 2) & 1) | ((i >> 5) << 1)] * tmp[i];
    buf ^= 1;
    k = kn;
  }
  const int C0 = attn_pf2_c(sg, l);
  const int p = a.start + qb;
  device float* oq = out + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D;
  #pragma unroll
  for (int g = 0; g < 16; g++) {
    const int i = 4 * g, h = R0 + 8 * ((i >> 2) & 1) + 32 * (i >> 5), d = d0 + C0 + (i & 3) + 32 * ((i >> 3) & 3);
    float4 v = float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
    if (d >= ATTN_D - ATTN_RD) {
      const int pi = (d - (ATTN_D - ATTN_RD)) / 2;
      v.xy = attn_rot_inv(v.xy, tab[(size_t)p * 32 + pi]);
      v.zw = attn_rot_inv(v.zw, tab[(size_t)p * 32 + pi + 1]);
    }
    *(device float4*)(oq + (size_t)h * ATTN_D + d) = v;
  }
}

// Layout check for the pf2 kernels: out[t] = elements of the (64,64,64) NT multiply-accumulate and NN multiply
// destinations whose (row, col) differ from attn_pf2_r/c/ri/ci. grid 1, 128 threads.
template <bool TR, bool ACC>
inline int attn_pf2_check_one(threadgroup half* X, ushort sg, ushort l) {
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, TR, false,
                                            ACC ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  int bad = cT.get_capacity() == 32 ? 0 : 1000;
  for (uint16_t i = 0; i < cT.get_capacity() && i < 32; ++i) {
    if (!cT.is_valid_element(i)) { bad++; continue; }
    auto id = cT.get_multidimensional_index(i);
    if (int(id[1]) != attn_pf2_r(sg, l) + attn_pf2_ri(i) || int(id[0]) != attn_pf2_c(sg, l) + attn_pf2_ci(i)) bad++;
  }
  return bad;
}
kernel void attn_pf2_layout_check(device int* out [[buffer(0)]], ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half X[64 * 64];
  int bad = attn_pf2_check_one<true, true>(X, sg, l) + attn_pf2_check_one<false, false>(X, sg, l);
  {
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(128, 64));
    constexpr auto desc = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    if (cT.get_capacity() != 64) bad += 1000;
    for (uint16_t i = 0; i < cT.get_capacity() && i < 64; ++i) {
      if (!cT.is_valid_element(i)) { bad++; continue; }
      auto id = cT.get_multidimensional_index(i);
      if (int(id[1]) != attn_pf2_r(sg, l) + 8 * ((i >> 2) & 1) + 32 * (i >> 5) ||
          int(id[0]) != attn_pf2_c(sg, l) + (i & 3) + 32 * ((i >> 3) & 3)) bad++;
    }
  }
  out[sg * 32 + l] = bad;
}
