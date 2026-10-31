// Tensor-API (MPP matmul2d) helpers shared by the prefill matmul and MoE prefill kernels.
#pragma once
#include "quant_blocks.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Dequantize rows [r0, r0+TN) x K range [k0, k0+128) of W into a half tile (row stride TKP). Each work unit is one
// lane-slice (EPL elements) of the Q decoder; for every supported block type the slices that cover a 128-element chunk
// are LPC = 128/EPL consecutive lane indices, so g = k0/EPL + s enumerates them.
template <typename Q, int TN, int TKP, int NT>
static inline void q_tile_deq(device const uchar* W, uint row_bytes, uint rows, uint r0, uint k0,
                              threadgroup half* tile, ushort tid) {
  constexpr int LPC = 128 / Q::EPL;
  for (int u = tid; u < TN * LPC; u += NT) {
    const int row = u / LPC, s = u - row * LPC;
    const uint g = k0 / Q::EPL + s, blk = g / Q::LPB;
    const ushort sub = g - blk * Q::LPB;
    device const uchar* b = W + (ulong)min(r0 + row, rows - 1) * row_bytes + (ulong)blk * Q::BB;
    float4 w[Q::EPL / 4];
    Q::deq(b, sub, w);
    threadgroup half* trow = tile + row * TKP + (int)(blk * Q::QK) - (int)k0;
    for (int q = 0; q < Q::EPL / 4; q++) *(threadgroup half4*)(trow + Q::pos(sub, 4 * q)) = half4(w[q]);
  }
}

