// Decode matvec (n = 1): y[r] (+)= W[r] . x  for raw GGUF block rows.
// Threadgroup = RSG row-simdgroups x KS k-split simdgroups (args.ksplit); each simdgroup owns NR consecutive rows and
// keeps its x slice in registers across them. Lanes are split into 32/LPB block slots; a slot's LPB lanes cooperate
// on one block, so each simdgroup iteration reads 32/LPB consecutive blocks of each of its rows.
#include "quant_blocks.h"

template <typename Q, int NR>
kernel void q_mv(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                 constant QMvArgs& a [[buffer(3)]], threadgroup float* red [[threadgroup(0)]],
                 uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint row0 = (tg * rsg + rs) * NR;
  constexpr ushort BPI = 32 / Q::LPB;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK;
  device const float* xg = x + (row0 / a.rows_per_group) * a.x_group_stride;
  device const uchar* wr[NR];
  for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
  float acc[NR];
  for (int r = 0; r < NR; r++) acc[r] = 0.f;
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    float4 xr[Q::X4];
    Q::load_x(xg + ib * Q::QK, sub, xr);
    for (int r = 0; r < NR; r++) acc[r] += Q::dot(wr[r] + ib * Q::BB, sub, xr);
  }
  for (int r = 0; r < NR; r++) acc[r] = simd_sum(acc[r]);
  if (ks > 1) {
    if (lane == 0) for (int r = 0; r < NR; r++) red[(rs * NR + r) * ks + kk] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++) { float s = 0; for (ushort k = 0; k < ks; k++) s += red[(rs * NR + r) * ks + k]; acc[r] = s; }
  }
  if (lane == 0)
    for (int r = 0; r < NR; r++) {
      const uint row = row0 + r;
      if (row < a.rows) y[row] = a.accumulate ? y[row] + acc[r] : acc[r];
    }
}

#define Q_MV_INST(Q, name)                                                                                     \
  template [[host_name("q_mv_" name "_r1")]] kernel void q_mv<Q, 1>(device const uchar*, device const float*,  \
      device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);                     \
  template [[host_name("q_mv_" name "_r2")]] kernel void q_mv<Q, 2>(device const uchar*, device const float*,  \
      device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);                     \
  template [[host_name("q_mv_" name "_r4")]] kernel void q_mv<Q, 4>(device const uchar*, device const float*,  \
      device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);

Q_MV_INST(QB_F32, "f32")
Q_MV_INST(QB_F16, "f16")
Q_MV_INST(QB_BF16, "bf16")
Q_MV_INST(QB_Q8_0, "q8_0")
Q_MV_INST(QB_Q5K, "q5_k")
Q_MV_INST(QB_Q6K, "q6_k")
Q_MV_INST(QB_IQ1S, "iq1_s")
Q_MV_INST(QB_IQ2XXS, "iq2_xxs")
Q_MV_INST(QB_IQ1M, "iq1_m")
Q_MV_INST(QB_IQ2XS, "iq2_xs")

// ---------------- decode matvec with the input RMSNorm fused into the x loads ----------------
// y = W . (x * rsqrt(mean(x^2) + eps) * nw)   (decode q_a_norm -> attn_q_b / indexer q_b).
// Every threadgroup computes the scale itself with exactly the reduction of arch_rmsnorm (threadgroup variant, one row
// of na.dim elements, na.nt threads): virtual thread v sums x[v], x[v + nt], ... ; simd_sum per 32 virtual threads;
// simd_sum of the per-simdgroup sums; rsqrt(sum / dim + eps). The normalized element is (x * r) * nw as in
// arch_rmsnorm, so the result is bit-identical to rmsnorm -> barrier -> q_mv. No block-diagonal groups.
struct QNormArgs { uint dim; float eps; uint nt; uint tiles_per_tg; };

// A threadgroup computes the scale once and then walks tiles_per_tg consecutive row tiles (rsg*NR rows each), so the
// norm prologue is amortized; each row's arithmetic is exactly q_mv's.
template <typename Q, int NR>
kernel void q_mv_rn(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                    constant QMvArgs& a [[buffer(3)]], device const float* nw [[buffer(4)]],
                    constant QNormArgs& na [[buffer(5)]], threadgroup float* red [[threadgroup(0)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float nred[32];
  {
    const ushort nvs = na.nt / 32;
    for (ushort v = sg; v < nvs; v += nsg) {
      const uint vt = uint(v) * 32 + lane;
      float ss = 0.0f;
      for (uint i = vt; i < na.dim; i += na.nt) ss += x[i] * x[i];
      ss = simd_sum(ss);
      if (lane == 0) nred[v] = ss;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rsum = lane < ushort(na.nt / 32) ? nred[lane] : 0.0f;
  rsum = simd_sum(rsum);
  const float rs = rsqrt(rsum / float(na.dim) + na.eps);

  const ushort ks = a.ksplit, kk = sg % ks, rs_ = sg / ks, rsg = nsg / ks;
  constexpr ushort BPI = 32 / Q::LPB;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK;
  const uint ntiles = (a.rows + rsg * NR - 1) / (rsg * NR);
  const uint t0 = tg * na.tiles_per_tg, t1 = min(ntiles, t0 + na.tiles_per_tg);
  for (uint tile = t0; tile < t1; tile++) {
    const uint row0 = (tile * rsg + rs_) * NR;
    device const uchar* wr[NR];
    for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.f;
    for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
      float4 xr[Q::X4], wn[Q::X4];
      Q::load_x(x + ib * Q::QK, sub, xr);
      Q::load_x(nw + ib * Q::QK, sub, wn);
      for (int k = 0; k < Q::X4; k++) xr[k] = xr[k] * rs * wn[k];
      for (int r = 0; r < NR; r++) acc[r] += Q::dot(wr[r] + ib * Q::BB, sub, xr);
    }
    for (int r = 0; r < NR; r++) acc[r] = simd_sum(acc[r]);
    if (ks > 1) {
      if (lane == 0) for (int r = 0; r < NR; r++) red[(rs_ * NR + r) * ks + kk] = acc[r];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (kk == 0) {
        for (int r = 0; r < NR; r++) { float s = 0; for (ushort k = 0; k < ks; k++) s += red[(rs_ * NR + r) * ks + k]; acc[r] = s; }
        if (lane == 0)
          for (int r = 0; r < NR; r++) {
            const uint row = row0 + r;
            if (row < a.rows) y[row] = a.accumulate ? y[row] + acc[r] : acc[r];
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (lane == 0) {
      for (int r = 0; r < NR; r++) {
        const uint row = row0 + r;
        if (row < a.rows) y[row] = a.accumulate ? y[row] + acc[r] : acc[r];
      }
    }
  }
}

#define Q_MVRN_INST1(Q, name, NR)                                                                                  \
  template [[host_name("q_mv_rn_" name "_r" #NR)]] kernel void q_mv_rn<Q, NR>(device const uchar*, device const float*, \
      device float*, constant QMvArgs&, device const float*, constant QNormArgs&, threadgroup float*, uint, ushort,      \
      ushort, ushort);
#define Q_MVRN_INST(Q, name) Q_MVRN_INST1(Q, name, 1) Q_MVRN_INST1(Q, name, 2) Q_MVRN_INST1(Q, name, 4)
Q_MVRN_INST(QB_F32, "f32")
Q_MVRN_INST(QB_F16, "f16")
Q_MVRN_INST(QB_BF16, "bf16")
Q_MVRN_INST(QB_Q8_0, "q8_0")
Q_MVRN_INST(QB_Q5K, "q5_k")
Q_MVRN_INST(QB_Q6K, "q6_k")

// ---------------- matvec_n: n in 2..NT activation columns share one weight read ----------------
// The threadgroup stages x[t][c0 .. c0+CK) for all n columns in threadgroup memory as half (CK = one simdgroup
// iteration = 32/LPB blocks), so every x element is fetched from L2 once per threadgroup instead of once per
// simdgroup-row; each lane dequantizes its EPL weights of NR rows once per block and dots them with all n columns.
// No K split (args.ksplit ignored); every row of a threadgroup must lie in the same block-diagonal group.
template <typename Q, int NR, int NT>
kernel void q_mvn(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                  constant QMvArgs& a [[buffer(3)]], threadgroup half* xs [[threadgroup(0)]],
                  uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]],
                  ushort tid [[thread_index_in_threadgroup]]) {
  constexpr ushort BPI = 32 / Q::LPB;
  constexpr int E4 = Q::EPL / 4, CK = BPI * Q::QK, CK4 = CK / 4;
  const uint row0 = (tg * nsg + sg) * NR;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK, n = a.n, nthr = 32 * nsg;
  device const float* xg = x + ((tg * nsg * NR) / a.rows_per_group) * a.x_group_stride;
  device const uchar* wr[NR];
  for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
  float acc[NR][NT];
  for (int r = 0; r < NR; r++) _Pragma("unroll") for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
  for (uint c0 = 0; c0 < a.cols; c0 += CK) {
    const uint lim4 = min((uint)CK4, (a.cols - c0) / 4);
    for (uint i = tid; i < n * CK4; i += nthr) {
      const uint t = i / CK4, j = i - t * CK4;
      if (j < lim4) ((threadgroup half4*)xs)[i] = half4(*(device const float4*)(xg + t * a.x_stride + c0 + 4 * j));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ib = c0 / Q::QK + slot;
    if (ib < nb) {
      float4 w[NR][E4];
      for (int r = 0; r < NR; r++) Q::deq(wr[r] + ib * Q::BB, sub, w[r]);
      threadgroup const half* xb = xs + slot * Q::QK;
      _Pragma("unroll") for (int t = 0; t < NT; t++) {
        if (t < (int)n) {
          float4 xr[E4];
          for (int k = 0; k < E4; k++) xr[k] = float4(*(threadgroup const half4*)(xb + t * CK + Q::pos(sub, 4 * k)));
          for (int r = 0; r < NR; r++) {
            float s = 0;
            for (int k = 0; k < E4; k++) s += metal::dot(w[r][k], xr[k]);
            acc[r][t] += s;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r = 0; r < NR; r++) _Pragma("unroll") for (int t = 0; t < NT; t++) acc[r][t] = simd_sum(acc[r][t]);
  if (lane < NT && lane < n) {
    for (int r = 0; r < NR; r++) {
      const uint row = row0 + r;
      float v = 0;
      _Pragma("unroll") for (int t = 0; t < NT; t++) if (t == lane) v = acc[r][t];
      if (row < a.rows) {
        device float* yp = y + lane * a.y_stride + row;
        *yp = a.accumulate ? *yp + v : v;
      }
    }
  }
}

#define Q_MVN_INST1(Q, name, NR, NT)                                                                          \
  template [[host_name("q_mvn_" name "_r" #NR "_n" #NT)]] kernel void q_mvn<Q, NR, NT>(device const uchar*, \
      device const float*, device float*, constant QMvArgs&, threadgroup half*, uint, ushort, ushort, ushort, ushort);
#define Q_MVN_INST(Q, name) \
  Q_MVN_INST1(Q, name, 1, 2) Q_MVN_INST1(Q, name, 1, 4) Q_MVN_INST1(Q, name, 1, 8) Q_MVN_INST1(Q, name, 1, 16) \
  Q_MVN_INST1(Q, name, 2, 2) Q_MVN_INST1(Q, name, 2, 4) Q_MVN_INST1(Q, name, 2, 8) Q_MVN_INST1(Q, name, 2, 16)

Q_MVN_INST(QB_F32, "f32")
Q_MVN_INST(QB_F16, "f16")
Q_MVN_INST(QB_BF16, "bf16")
Q_MVN_INST(QB_Q8_0, "q8_0")
Q_MVN_INST(QB_Q5K, "q5_k")
Q_MVN_INST(QB_Q6K, "q6_k")
Q_MVN_INST(QB_IQ1S, "iq1_s")
Q_MVN_INST(QB_IQ2XXS, "iq2_xxs")
Q_MVN_INST(QB_IQ1M, "iq1_m")
Q_MVN_INST(QB_IQ2XS, "iq2_xs")
