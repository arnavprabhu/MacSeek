// MoE prefill: grouped GEMM over experts with the tensor API.
//   q_moe_gather : Xg[i] = half(X[tok[i]]) for the expert-sorted entries, then Xg[total + t] = half(X[t]) (shared)
//   q_moe_mm_gu  : per (expert, token tile, row tile): G = Xg_e . Wg^T, U = Xg_e . Wu^T  (two dequant tiles, two
//                  accumulators) -> H[i][r] = half(silu(min(G,10)) * clamp(U,-10,10) * wt[i])
//   q_moe_mm_down: Z[i] = H_e[i] . Wd^T  (F32)
//   q_moe_reduce : Y[t] += Z[total + t] + sum_k Z[slot_of[t][k]]   (deterministic order)
// Expert weights are reached through a device pointer table (gpuAddress + offset, written by the host into the
// scratch header); tiles map to experts through a host-written tile table.
#include "quant_tile.h"

struct QExpPtr { device const uchar* g; device const uchar* u; device const uchar* d; };

struct QMoePArgs {
  uint n;           // tokens
  uint total;       // routed entries (sum of expert token counts)
  uint dim;         // 5120
  uint inter;       // 2304
  uint rb_gu, rb_d; // row bytes of this dispatch's gate/up and down types
  uint k_per_token; // slot_of row length
};

kernel void q_moe_gather(device const float* X [[buffer(0)]], device half* Xg [[buffer(1)]],
                         device const int* tok [[buffer(2)]], constant QMoePArgs& a [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.y, c4 = gid.x;
  if (i >= a.total + a.n || c4 >= a.dim / 4) return;
  const uint src = i < a.total ? (uint)tok[i] : i - a.total;
  ((device half4*)(Xg + (ulong)i * a.dim))[c4] = half4(((device const float4*)(X + (ulong)src * a.dim))[c4]);
}

// tile table entry: x = expert index (into ptrs / offs), y = first token row of the tile within the expert
template <typename Q, int TM, int TN, int NSG>
kernel void q_moe_mm_gu(device const half* Xg [[buffer(0)]], device half* H [[buffer(1)]],
                        constant QMoePArgs& a [[buffer(2)]], device const float* wt [[buffer(3)]],
                        device const QExpPtr* ptrs [[buffer(4)]], device const uint* offs [[buffer(5)]],
                        device const uint2* tiles [[buffer(6)]], threadgroup half* tw [[threadgroup(0)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int TK = 128;
  const uint2 te = tiles[tg.y];
  const uint e = te.x, base = offs[e] + te.y, m = offs[e + 1] - base;
  const uint r0 = tg.x * TN;
  device const uchar* Wg = ptrs[e].g;
  device const uchar* Wu = ptrs[e].u;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(Xg + (ulong)base * a.dim),
                                                              dextents<int32_t, 2>(a.dim, min(m, (uint)TM)));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  threadgroup half* tg_g = tw;
  threadgroup half* tg_u = tw + TN * TK;
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(tg_g, dextents<int32_t, 2>(TK, TN));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(tg_u, dextents<int32_t, 2>(TK, TN));
  auto cG = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tG), float>();
  auto cU = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tU), float>();
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) if (cG.is_valid_element(i)) { cG[i] = 0; cU[i] = 0; }
  for (uint k0 = 0; k0 < a.dim; k0 += TK) {
    q_tile_deq<Q, TN, TK, 32 * NSG>(Wg, a.rb_gu, a.inter, r0, k0, tg_g, tid);
    q_tile_deq<Q, TN, TK, 32 * NSG>(Wu, a.rb_gu, a.inter, r0, k0, tg_u, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto xs = tX.slice(k0, 0);
    op.run(xs, tG, cG);
    op.run(xs, tU, cU);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
    if (!cG.is_valid_element(i)) continue;
    auto idx = cG.get_multidimensional_index(i);  // [0] = weight row within tile, [1] = token within tile
    const uint r = r0 + idx[0], t = idx[1];
    if (r >= a.inter || t >= m) continue;
    float g = min(cG[i], 10.f), u = clamp(cU[i], -10.f, 10.f);
    const float w = base + t < a.total ? wt[base + t] : 1.f;
    H[(ulong)(base + t) * a.inter + r] = half(g / (1.f + exp(-g)) * u * w);
  }
}

template <typename Q, int TM, int TN, int NSG>
kernel void q_moe_mm_down(device const half* H [[buffer(0)]], device float* Z [[buffer(1)]],
                          constant QMoePArgs& a [[buffer(2)]], device const QExpPtr* ptrs [[buffer(4)]],
                          device const uint* offs [[buffer(5)]], device const uint2* tiles [[buffer(6)]],
                          threadgroup half* tw [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int TK = 128;
  const uint2 te = tiles[tg.y];
  const uint e = te.x, base = offs[e] + te.y, m = min(offs[e + 1] - base, (uint)TM);
  const uint r0 = tg.x * TN;
  device const uchar* Wd = ptrs[e].d;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(H + (ulong)base * a.inter),
                                                              dextents<int32_t, 2>(a.inter, m));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tZ(Z + (ulong)base * a.dim, dextents<int32_t, 2>(a.dim, m));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW0(tw, dextents<int32_t, 2>(TK, TN));
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW0), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  uint it = 0;
  for (uint k0 = 0; k0 < a.inter; k0 += TK, it++) {
    threadgroup half* tile = tw + (it & 1) * (TN * TK);
    q_tile_deq<Q, TN, TK, 32 * NSG>(Wd, a.rb_d, a.dim, r0, k0, tile, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(tile, dextents<int32_t, 2>(TK, TN));
    auto xs = tX.slice(k0, 0);
    op.run(xs, tW, cT);
  }
  auto zs = tZ.slice(r0, 0);
  cT.store(zs);
}

kernel void q_moe_reduce(device const float* Z [[buffer(0)]], device float* Y [[buffer(1)]],
                         constant QMoePArgs& a [[buffer(2)]], device const int* slot_of [[buffer(3)]],
                         constant uint& has_shared [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const uint t = gid.y, c4 = gid.x;
  if (t >= a.n || c4 >= a.dim / 4) return;
  float4 s = has_shared ? ((device const float4*)(Z + (ulong)(a.total + t) * a.dim))[c4] : float4(0);
  for (uint k = 0; k < a.k_per_token; k++) {
    const int i = slot_of[t * a.k_per_token + k];
    if (i >= 0) s += ((device const float4*)(Z + (ulong)i * a.dim))[c4];
  }
  device float4* y = (device float4*)(Y + (ulong)t * a.dim) + c4;
  *y += s;
}

#define Q_MOEP_INST(Q, name)                                                                                      \
  template [[host_name("q_moe_mm_gu_" name)]] kernel void q_moe_mm_gu<Q, 64, 64, 4>(device const half*,           \
      device half*, constant QMoePArgs&, device const float*, device const QExpPtr*, device const uint*,           \
      device const uint2*, threadgroup half*, uint2, ushort);                                                      \
  template [[host_name("q_moe_mm_down_" name)]] kernel void q_moe_mm_down<Q, 64, 64, 4>(device const half*,       \
      device float*, constant QMoePArgs&, device const QExpPtr*, device const uint*, device const uint2*,          \
      threadgroup half*, uint2, ushort);
Q_MOEP_INST(QB_IQ1S, "iq1_s")
Q_MOEP_INST(QB_IQ2XXS, "iq2_xxs")
Q_MOEP_INST(QB_Q6K, "q6_k")
Q_MOEP_INST(QB_Q8_0, "q8_0")
