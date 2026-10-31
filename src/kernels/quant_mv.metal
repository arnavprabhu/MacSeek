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

// ---------------- matvec_n: n in 2..NT activation columns share one weight read ----------------
// Each lane dequantizes its EPL weights of NR rows once per block, then dots them with the x slices of all n columns.
template <typename Q, int NR, int NT>
kernel void q_mvn(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                  constant QMvArgs& a [[buffer(3)]], threadgroup float* red [[threadgroup(0)]],
                  uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint row0 = (tg * rsg + rs) * NR;
  constexpr ushort BPI = 32 / Q::LPB;
  constexpr int E4 = Q::EPL / 4;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK, n = a.n;
  device const float* xg = x + (row0 / a.rows_per_group) * a.x_group_stride;
  device const uchar* wr[NR];
  for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
  float acc[NR][NT];
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    float4 w[NR][E4];
    for (int r = 0; r < NR; r++) Q::deq(wr[r] + ib * Q::BB, sub, w[r]);
    for (int t = 0; t < NT; t++) {
      if (t < (int)n) {
        float4 xr[Q::X4];
        Q::load_x(xg + t * a.x_stride + ib * Q::QK, sub, xr);
        for (int r = 0; r < NR; r++) {
          float s = 0;
          for (int k = 0; k < E4; k++) s += metal::dot(w[r][k], xr[k]);
          acc[r][t] += s;
        }
      }
    }
  }
  for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) acc[r][t] = simd_sum(acc[r][t]);
  if (ks > 1) {
    if (lane == 0) for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) red[((rs * NR + r) * NT + t) * ks + kk] = acc[r][t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++) for (int t = 0; t < NT; t++) {
      float s = 0; for (ushort k = 0; k < ks; k++) s += red[((rs * NR + r) * NT + t) * ks + k]; acc[r][t] = s;
    }
  }
  if (lane < NT && lane < n) {
    for (int r = 0; r < NR; r++) {
      const uint row = row0 + r;
      float v = 0;
      for (int t = 0; t < NT; t++) if (t == lane) v = acc[r][t];
      if (row < a.rows) {
        device float* yp = y + lane * a.y_stride + row;
        *yp = a.accumulate ? *yp + v : v;
      }
    }
  }
}

#define Q_MVN_INST1(Q, name, NR, NT)                                                                          \
  template [[host_name("q_mvn_" name "_r" #NR "_n" #NT)]] kernel void q_mvn<Q, NR, NT>(device const uchar*, \
      device const float*, device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);
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
