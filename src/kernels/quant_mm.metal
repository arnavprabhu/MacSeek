// Prefill matmul probes / plain F16 GEMM (peak reference) using the Metal 4 tensor API (MetalPerformancePrimitives).
#include "quant_tile.h"

// C[n x m] = A[n x k] . B[m x k]^T  (A = activations half, B = weights half, both k-contiguous; C float m-contiguous)
// Threadgroup tile TM (tokens) x TN (weight rows), NSG simdgroups, K loop in chunks of TK.
template <int TM, int TN, int TK, int NSG>
kernel void q_gemm_f16(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], device float* C [[buffer(2)]],
                       constant uint4& dims [[buffer(3)]],  // n, m, k
                       uint2 tg [[threadgroup_position_in_grid]]) {
  const int n = dims.x, m = dims.y, k = dims.z;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A, dextents<int32_t, 2>(k, n));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)B, dextents<int32_t, 2>(k, m));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(m, n));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  const int r0 = tg.x * TN, t0 = tg.y * TM;
  for (int k0 = 0; k0 < k; k0 += TK) {
    auto a = tA.slice(k0, t0);
    auto b = tB.slice(k0, r0);
    op.run(a, b, cT);
  }
  auto c = tC.slice(r0, t0);
  cT.store(c);
}

template [[host_name("q_gemm_f16_64x64")]] kernel void q_gemm_f16<64, 64, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x128")]] kernel void q_gemm_f16<64, 128, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_128x64")]] kernel void q_gemm_f16<128, 64, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_128x128")]] kernel void q_gemm_f16<128, 128, 32, 8>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_32x64")]] kernel void q_gemm_f16<32, 64, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);

// same with F32 activations (left operand float x right half)
template <int TM, int TN, int TK, int NSG>
kernel void q_gemm_f32a(device const float* A [[buffer(0)]], device const half* B [[buffer(1)]], device float* C [[buffer(2)]],
                        constant uint4& dims [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int n = dims.x, m = dims.y, k = dims.z;
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tA((device float*)A, dextents<int32_t, 2>(k, n));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)B, dextents<int32_t, 2>(k, m));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(m, n));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  const int r0 = tg.x * TN, t0 = tg.y * TM;
  for (int k0 = 0; k0 < k; k0 += TK) {
    auto a = tA.slice(k0, t0);
    auto b = tB.slice(k0, r0);
    op.run(a, b, cT);
  }
  auto c = tC.slice(r0, t0);
  cT.store(c);
}
template [[host_name("q_gemm_f32a_64x128")]] kernel void q_gemm_f32a<64, 128, 32, 4>(device const float*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x128_k64")]] kernel void q_gemm_f16<64, 128, 64, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x128_k128")]] kernel void q_gemm_f16<64, 128, 128, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x256")]] kernel void q_gemm_f16<64, 256, 32, 8>(device const half*, device const half*,
    device float*, constant uint4&, uint2);

// ======================================================================================================== prefill
struct QMmArgs {
  uint n;          // tokens (rows of X16 / Y)
  uint rows;       // weight rows (output features)
  uint cols;       // K
  uint row_bytes;  // bytes per weight row
  uint x_stride;   // halves between token rows of X16
  uint y_stride;   // floats between token rows of Y
  uint accumulate;
};

// X F32 -> half (activations for the tensor-API matmuls); n4 = count / 4
kernel void q_f32_to_f16(device const float4* x [[buffer(0)]], device half4* y [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < n4) y[i] = half4(x[i]);
}

// Y[n x rows] (+)= X16[n x cols] . dequant(W)^T. TG tile: TM tokens x TN weight rows, K step 128, NBUF tile buffers.
template <typename Q, int TM, int TN, int NSG, int NBUF>
kernel void q_mm(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]],
                 constant QMmArgs& a [[buffer(3)]], threadgroup half* tiles [[threadgroup(0)]],
                 uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int TK = 128, TKP = 128;
  const uint r0 = tg.x * TN, t0 = tg.y * TM;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)X, dextents<int32_t, 2>(a.cols, a.n),
                                                              array<int32_t, 2>{1, (int32_t)a.x_stride});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.rows, a.n),
                                                               array<int32_t, 2>{1, (int32_t)a.y_stride});
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW0(tiles, dextents<int32_t, 2>(TK, TN));
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW0), float>();
  auto ty = tY.slice(r0, t0);
  if (a.accumulate) cT.load(ty);
  else for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  uint it = 0;
  for (uint k0 = 0; k0 < a.cols; k0 += TK, it++) {
    threadgroup half* tile = tiles + (NBUF > 1 ? (it & 1) : 0) * (TN * TKP);
    q_tile_deq<Q, TN, TKP, 32 * NSG>(W, a.row_bytes, a.rows, r0, k0, tile, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(tile, dextents<int32_t, 2>(TK, TN));
    auto xs = tX.slice(k0, t0);
    op.run(xs, tW, cT);
    if (NBUF == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  cT.store(ty);
}

// dense weights (F16 / BF16 / F32): right operand straight from device memory
template <typename WT, int TM, int TN, int NSG>
kernel void q_mm_dense(device const WT* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]],
                       constant QMmArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const uint r0 = tg.x * TN, t0 = tg.y * TM;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)X, dextents<int32_t, 2>(a.cols, a.n),
                                                              array<int32_t, 2>{1, (int32_t)a.x_stride});
  tensor<device WT, dextents<int32_t, 2>, tensor_inline> tW((device WT*)W, dextents<int32_t, 2>(a.cols, a.rows));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.rows, a.n),
                                                               array<int32_t, 2>{1, (int32_t)a.y_stride});
  constexpr auto desc = matmul2d_descriptor(TM, TN, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  auto ty = tY.slice(r0, t0);
  if (a.accumulate) cT.load(ty);
  else for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint k0 = 0; k0 < a.cols; k0 += 64) {
    auto xs = tX.slice(k0, t0);
    auto ws = tW.slice(k0, r0);
    op.run(xs, ws, cT);
  }
  cT.store(ty);
}

// Whole-matrix dequantization to half (prefill with large n: dequant once, then the dense tensor-API GEMM).
// One thread per lane-slice; grid = rows x (cols / EPL).
template <typename Q>
kernel void q_deq_f16(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint4& d [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {  // d = (rows, cols, row_bytes)
  const uint row = gid.y, g = gid.x;
  if (row >= d.x || g >= d.y / Q::EPL) return;
  const uint blk = g / Q::LPB;
  const ushort sub = g - blk * Q::LPB;
  float4 w[Q::EPL / 4];
  Q::deq(W + (ulong)row * d.z + (ulong)blk * Q::BB, sub, w);
  device half* o = out + (ulong)row * d.y + blk * Q::QK;
  for (int q = 0; q < Q::EPL / 4; q++) *(device half4*)(o + Q::pos(sub, 4 * q)) = half4(w[q]);
}
#define Q_DEQ_INST(Q, name) \
  template [[host_name("q_deq_f16_" name)]] kernel void q_deq_f16<Q>(device const uchar*, device half*, constant uint4&, uint2);
Q_DEQ_INST(QB_Q8_0, "q8_0")
Q_DEQ_INST(QB_Q5K, "q5_k")
Q_DEQ_INST(QB_Q6K, "q6_k")
Q_DEQ_INST(QB_IQ1S, "iq1_s")
Q_DEQ_INST(QB_IQ2XXS, "iq2_xxs")
Q_DEQ_INST(QB_IQ1M, "iq1_m")
Q_DEQ_INST(QB_IQ2XS, "iq2_xs")

#define Q_MM_INST1(Q, name, TM, TN, NSG, NBUF)                                                                   \
  template [[host_name("q_mm_" name "_" #TM "x" #TN "_b" #NBUF)]] kernel void q_mm<Q, TM, TN, NSG, NBUF>(      \
      device const uchar*, device const half*, device float*, constant QMmArgs&, threadgroup half*, uint2, ushort);
#define Q_MM_INST1S(Q, name, TM, TN, NSG, NBUF)                                                                  \
  template [[host_name("q_mm_" name "_" #TM "x" #TN "_b" #NBUF "_s" #NSG)]] kernel void q_mm<Q, TM, TN, NSG, NBUF>( \
      device const uchar*, device const half*, device float*, constant QMmArgs&, threadgroup half*, uint2, ushort);
#define Q_MM_INST(Q, name) \
  Q_MM_INST1(Q, name, 64, 64, 4, 2) Q_MM_INST1(Q, name, 128, 64, 4, 2) Q_MM_INST1(Q, name, 64, 128, 4, 1) \
  Q_MM_INST1(Q, name, 128, 128, 8, 1) Q_MM_INST1(Q, name, 32, 64, 4, 2) Q_MM_INST1(Q, name, 16, 64, 4, 2) \
  Q_MM_INST1(Q, name, 16, 32, 4, 1) Q_MM_INST1(Q, name, 32, 32, 4, 1) Q_MM_INST1S(Q, name, 16, 64, 2, 1) Q_MM_INST1(Q, name, 32, 64, 4, 1)

Q_MM_INST(QB_Q8_0, "q8_0")
Q_MM_INST(QB_Q5K, "q5_k")
Q_MM_INST(QB_Q6K, "q6_k")
Q_MM_INST(QB_IQ1S, "iq1_s")
Q_MM_INST(QB_IQ2XXS, "iq2_xxs")
Q_MM_INST(QB_IQ1M, "iq1_m")
Q_MM_INST(QB_IQ2XS, "iq2_xs")

#define Q_MMD_INST(WT, name)                                                                                     \
  template [[host_name("q_mm_" name "_64x128")]] kernel void q_mm_dense<WT, 64, 128, 4>(device const WT*,       \
      device const half*, device float*, constant QMmArgs&, uint2);
Q_MMD_INST(half, "f16")
Q_MMD_INST(bfloat, "bf16")
Q_MMD_INST(float, "f32")
