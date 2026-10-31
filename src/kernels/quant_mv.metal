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
