#include <metal_stdlib>
using namespace metal;
// Diagnostics only: plain float copy (dst[i] = src[i]).
kernel void dbg_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
