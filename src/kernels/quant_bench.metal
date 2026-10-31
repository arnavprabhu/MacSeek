// Bandwidth probe kernels for bench/membw (q_ prefix: quant module).
#include "quant_common.h"

// v1: grid-stride float4 loads
kernel void q_bw_read_f4(device const float4* src [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]],
                         uint gsz [[threads_per_grid]]) {
  float4 acc = 0;
  for (uint i = gid; i < n4; i += gsz) acc += src[i];
  if (acc.x == 1234.5f) out[gid] = acc.y + acc.z + acc.w;  // practically never true; prevents DCE
}

// v2: each threadgroup streams one contiguous chunk, 4 x uint4 loads in flight per thread
kernel void q_bw_read_chunk(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                            constant uint& per_tg [[buffer(2)]],  // uint4 elements per threadgroup
                            uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint tsz [[threads_per_threadgroup]]) {
  device const uint4* p = src + (ulong)tg * per_tg;
  uint4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint i = tid; i + 3 * tsz < per_tg; i += 4 * tsz) {
    a0 ^= p[i]; a1 ^= p[i + tsz]; a2 ^= p[i + 2 * tsz]; a3 ^= p[i + 3 * tsz];
  }
  uint4 a = a0 ^ a1 ^ a2 ^ a3;
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u) out[tg * tsz + tid] = 1;
}

// v3: copy (read + write) for reference
kernel void q_bw_copy(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  dst[gid] = src[gid];
}
