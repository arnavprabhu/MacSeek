// Small glue kernels used by the model orchestration (src/model). Prefix m_.
#include "common.h"

kernel void m_fill(device float4* y [[buffer(0)]], constant float& v [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i < n4) y[i] = float4(v);
}

// tmp[g][t][r] (g < G, t < n, r < R) -> out[t][g*R + r]   (grouped wo_a outputs -> [n][G*R])
kernel void m_ungroup(device const float4* tmp [[buffer(0)]], device float4* out [[buffer(1)]],
                      constant uint4& a [[buffer(2)]],  // x = n, y = G, z = R/4
                      uint3 gid [[thread_position_in_grid]]) {
  const uint r4 = gid.x, t = gid.y, g = gid.z;
  if (r4 >= a.z || t >= a.x || g >= a.y) return;
  out[(ulong)t * a.y * a.z + g * a.z + r4] = tmp[((ulong)g * a.x + t) * a.z + r4];
}
