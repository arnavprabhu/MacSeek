// Throughput probes (bench/attn_mmprobe): simdgroup_matrix vs Metal 4 tensor-op matmul2d for the indexer shape
// (rows x 128) . (cols x 128)^T. Not used by the engine.
#include "attn_common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// 4 simdgroups, 64x64 output tile, K = 128 read straight from device memory.
kernel void attn_probe_sg(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                          device float* out [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  const uint r0 = tg.y * 64 + (sg >> 1) * 32, c0 = tg.x * 64 + (sg & 1) * 32;
  for (int k = 0; k < 128; k += 8) {
    simdgroup_half8x8 a[4], b[4];
    for (int i = 0; i < 4; i++) simdgroup_load(a[i], A + (r0 + i * 8) * 128 + k, 128);
    for (int j = 0; j < 4; j++) simdgroup_load(b[j], B + (c0 + j * 8) * 128 + k, 128, ulong2(0, 0), true);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  float s = 0;
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { auto e = acc[i][j].thread_elements(); s += max(e[0], 0.f) + max(e[1], 0.f); }
  s = simd_sum(s);
  if (ln == 0) out[(tg.y * 1024 + tg.x) * 4 + sg] = s;
}

template <int TM, int TN>
inline void probe_mpp(device half* A, device half* B, device float* out, uint M, uint N, uint2 tg) {
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(128, M));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(128, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 128, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<128, TM>(0, tg.y * TM);
  auto mB = tB.template slice<128, TN>(0, tg.x * TN);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mB, cT);
  float s = 0;
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) s += max(cT[i], 0.f);
  s = simd_sum(s);
  out[tg.y * 1024 + tg.x] = s;
}
kernel void attn_probe_mpp64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* out [[buffer(2)]],
                             constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
  probe_mpp<64, 64>(A, B, out, M, N, tg);
}
kernel void attn_probe_mpp128(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* out [[buffer(2)]],
                              constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
  probe_mpp<128, 64>(A, B, out, M, N, tg);
}

// streaming read bandwidth probe: each thread reads float4s, grid-stride
kernel void attn_probe_bw(device const float4* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                          uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  float4 s = 0;
  for (uint i = gid; i < n4; i += gsz) s += a[i];
  if (s.x == 1234.5f) out[0] = s.y + s.z + s.w;
}

// lane layout probe: load an 8x8 matrix whose element (r,c) = r*8+c, dump each lane's two thread_elements.
kernel void attn_probe_layout(device float* out [[buffer(0)]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup float m[64];
  for (int i = ln; i < 64; i += 32) m[i] = i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 a; simdgroup_load(a, m, 8);
  auto e = a.thread_elements();
  out[ln * 2] = e[0]; out[ln * 2 + 1] = e[1];
}

// cooperative destination layout probe: out[t * 256 + i] = row * 4096 + col (or -1), out[128*256 + t] = capacity
template <int TM, int TN>
inline void probe_coop_layout(device half* A, device half* B, device int* out, ushort t) {
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(128, TM));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(128, TN));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 128, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    int v = -1;
    if (cT.is_valid_element(i)) { auto id = cT.get_multidimensional_index(i); v = id[1] * 4096 + id[0]; }
    if (i < 256) out[t * 256 + i] = v;
  }
  out[128 * 256 + t] = cT.get_capacity();
}
kernel void attn_probe_coop_64x128(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device int* out [[buffer(2)]],
                                   ushort t [[thread_index_in_threadgroup]]) { probe_coop_layout<64, 128>(A, B, out, t); }
kernel void attn_probe_coop_64x64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device int* out [[buffer(2)]],
                                  ushort t [[thread_index_in_threadgroup]]) { probe_coop_layout<64, 64>(A, B, out, t); }
kernel void attn_probe_coop_128x64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device int* out [[buffer(2)]],
                                   ushort t [[thread_index_in_threadgroup]]) { probe_coop_layout<128, 64>(A, B, out, t); }
// 8-simdgroup NN 64x128 (K=64) destination layout: out[t * 256 + i] = row * 4096 + col, out[256*256 + t] = capacity
kernel void attn_probe_coop8_64x128(device int* out [[buffer(0)]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half X[64 * 128];
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(128, 64));
  constexpr auto desc = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    int v = -1;
    if (cT.is_valid_element(i)) { auto id = cT.get_multidimensional_index(i); v = id[1] * 4096 + id[0]; }
    if (i < 256) out[t * 256 + i] = v;
  }
  out[256 * 256 + t] = cT.get_capacity();
}

// cache-capacity probe: every thread re-reads its grid-stride share of a W-byte buffer `reps` times
kernel void attn_probe_bw_rep(device const float4* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint2& nr [[buffer(2)]],
                              uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  float4 s = 0;
  for (uint r = 0; r < nr.y; r++)
    for (uint i = gid; i < nr.x; i += gsz) s += a[(i + r * 4099u) % nr.x];
  if (s.x == 1234.5f) out[0] = s.y + s.z + s.w;
}
