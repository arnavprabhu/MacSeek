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

// ---- per-simdgroup register path: every simdgroup dequantizes its own weight sub-tile straight into the matmul's
// right-input cooperative tensor (no threadgroup memory, no barriers), so the ALU dequant of some simdgroups overlaps
// the tensor-unit work of others. The cooperative layout holds runs of 4 consecutive K elements of one weight row
// (verified by tests), dequantized with Q::deq4.
// Lane layout of the right-input cooperative tensor of matmul2d<(TM, TN, TK, transpose_right), execution_simdgroup>
// (dumped on M5, checked by test_quant_moe_prefill): lane L owns K quad q = (L&1) | ((L>>3)&1)<<1 and row
// rr = ((L>>1)&3) | ((L>>4)&1)<<2; element i = 4*(RB*g + rb) + c holds k = 16g + 4q + c, row = rr + 8rb (RB = TN/8).
// q_deqq<Q, NG>: the NG quads k = k0 + 16g + 4q (g < NG) of one weight row as half4.
// grid tables used by the cooperative dequant (device copies of the constant tables: divergent per-lane lookups are
// much cheaper through the regular load path than through constant memory)
struct QMoeTbl { device const uint2* g1; threadgroup const uint2* g2; };
// tables: tbl = device copy (uint2 units: IQ1_S half grid [0, 4096) | IQ2_XXS half grid [4096, 4608) | IQ2 sign masks
// [4608, 4864) | IQ2_XS half grid [4864, 5888)). g1 = the device table (IQ1_S / IQ1_M grid at 0, IQ2_XS grid at
// kQTblIQ2XS); IQ2_XXS kernels copy their 4 KB grid into threadgroup memory first (g2; lower-latency divergent
// lookups: -8% MoE time). For IQ2_XS the 8 KB copy costs more than it saves (bench_qmatmul --real: 97.8 -> 95.9 ms
// reading the device table).
constant uint kQTblIQ2XXS = 4096, kQTblIQ2XS = 4864;
template <typename Q, int NSG>
static inline QMoeTbl q_moe_tbl(device const uint2* tbl, threadgroup uint2* tgt, ushort sg, ushort lane) {
  if (is_same<Q, QB_IQ2XXS>::value) {
    for (uint i = sg * 32 + lane; i < 512; i += 32 * NSG) tgt[i] = tbl[kQTblIQ2XXS + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return QMoeTbl{tbl, tgt};
}
template <typename Q, int NG> struct QDeqq {
  static inline void run(device const uchar* row, uint k0, uint q, thread half4* out, thread const QMoeTbl& tb) {
    for (int g = 0; g < NG; g++) {
      const uint k = k0 + 16 * g + 4 * q;
      out[g] = half4(Q::deq4(row + (ulong)(k / Q::QK) * Q::BB, k % Q::QK));
    }
  }
};
template <int NG> struct QDeqq<QB_IQ1S, NG> {  // two quads per 32-element sub-block share d / qh / qs
  static inline void run(device const uchar* row, uint k0, uint q, thread half4* out, thread const QMoeTbl& tb) {
    for (int g2 = 0; g2 < NG; g2 += 2) {
      const uint k = k0 + 16 * g2;
      device const uchar* b = row + (ulong)(k >> 8) * 50;
      const uint ib = (k >> 5) & 7;
      const uint qh = q_ld16(b + 34 + 2 * ib), qs = q_ld32(b + 2 + 4 * ib);
      const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1), delta = (qh & 0x8000) ? -0.125f : 0.125f;
      for (int h = 0; h < 2 && g2 + h < NG; h++) {
        const uint gq = 2 * h + (q >> 1);
        const uint2 e = tb.g1[2 * (((qs >> (8 * gq)) & 0xFF) | (((qh >> (3 * gq)) & 7) << 8)) + (q & 1)];
        out[g2 + h] = half4(dl * (float4(as_type<half4>(e)) + delta));
      }
    }
  }
};
template <int NG> struct QDeqq<QB_IQ2XXS, NG> {
  static inline void run(device const uchar* row, uint k0, uint q, thread half4* out, thread const QMoeTbl& tb) {
    for (int g2 = 0; g2 < NG; g2 += 2) {
      const uint k = k0 + 16 * g2;
      device const uchar* b = row + (ulong)(k >> 8) * 66;
      const uint ib = (k >> 5) & 7;
      const uint idx = q_ld32(b + 2 + 8 * ib), a1 = q_ld32(b + 6 + 8 * ib);
      const float db = q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
      for (int h = 0; h < 2 && g2 + h < NG; h++) {
        const uint gq = 2 * h + (q >> 1);
        // 7 sign bits + parity -> half sign-bit masks (ALU beats a second table lookup)
        const uint s7 = (a1 >> (7 * gq)) & 127;
        const uint b = ((s7 | ((popcount(s7) & 1) << 7)) >> (4 * (q & 1))) & 15;
        const uint2 sm = uint2(((b & 1) << 15) | ((b & 2) << 30), ((b & 4) << 13) | ((b & 8) << 28));
        const uint2 e = tb.g2[2 * ((idx >> (8 * gq)) & 0xFF) + (q & 1)] ^ sm;  // threadgroup copy of the grid
        out[g2 + h] = half4(db * float4(as_type<half4>(e)));
      }
    }
  }
};

// IQ1_M: a K step (TK | 256) lies in one block: d and the scale words once; per 32-element sub-block one qs / qh
// load (fields at 4-byte multiples: aligned 32-bit loads given >= 4-byte aligned weight offsets)
template <int NG> struct QDeqq<QB_IQ1M, NG> {
  static inline void run(device const uchar* row, uint k0, uint q, thread half4* out, thread const QMoeTbl& tb) {
    device const uchar* b = row + (ulong)(k0 >> 8) * 56;
    const uint s01 = q_ld32a(b + 48), s23 = q_ld32a(b + 52);
    const float d = q_iq1m_dw(s01, s23);
    const uint qo = q >> 1;  // the lane's group within each 16-element half: group gq = 2h + qo
    for (int g2 = 0; g2 < NG; g2 += 2) {
      const uint ib = ((k0 + 16 * g2) >> 5) & 7;
      const uint qs = q_ld32a(b + 4 * ib) >> (8 * qo), qh = uint(q_ld16(b + 32 + 2 * ib)) >> (4 * qo);
      const uint sc = ((ib & 4) ? s23 : s01) >> (16 * ((ib >> 1) & 1) + 6 * (ib & 1));
      for (int h = 0; h < 2 && g2 + h < NG; h++) {
        const uint nb = (qh >> (8 * h)) & 15;
        const float dl = d * float(2 * ((sc >> (3 * h)) & 7) + 1), dd = (nb & 8) ? -0.125f * dl : 0.125f * dl;
        const uint2 e = tb.g1[2 * (((qs >> (16 * h)) & 0xFF) | ((nb & 7) << 8)) + (q & 1)];
        out[g2 + h] = half4(fma(float4(as_type<half4>(e)), dl, dd));
      }
    }
  }
};
// IQ2_XS: signs by ALU (7 bits + parity, as IQ2_XXS), grid from the device table
template <int NG> struct QDeqq<QB_IQ2XS, NG> {
  static inline void run(device const uchar* row, uint k0, uint q, thread half4* out, thread const QMoeTbl& tb) {
    device const uchar* b = row + (ulong)(k0 >> 8) * 74;
    const float d = q_ldh(b);
    for (int g2 = 0; g2 < NG; g2 += 2) {
      const uint ib = ((k0 + 16 * g2) >> 5) & 7;
      const uint scb = b[66 + ib];
      for (int h = 0; h < 2 && g2 + h < NG; h++) {
        const uint gq = 2 * h + (q >> 1);
        const uint qv = q_ld16(b + 2 + 2 * (4 * ib + gq)), s7 = qv >> 9;
        const uint bb = ((s7 | ((popcount(s7) & 1) << 7)) >> (4 * (q & 1))) & 15;
        const uint2 sm = uint2(((bb & 1) << 15) | ((bb & 2) << 30), ((bb & 4) << 13) | ((bb & 8) << 28));
        const uint2 e = tb.g1[kQTblIQ2XS + 2 * (qv & 511) + (q & 1)] ^ sm;
        const float db = d * (0.5f + float((scb >> (4 * h)) & 15)) * 0.25f;
        out[g2 + h] = half4(db * float4(as_type<half4>(e)));
      }
    }
  }
};

template <typename Q, int TN, int TK, typename CW>
static inline void q_coop_deq(thread CW& cw, device const uchar* W, uint row_bytes, uint r0, uint k0, ushort lane,
                              thread const QMoeTbl& tb) {
  static_assert(TN % 16 == 0 && TK % 32 == 0, "layout verified for TN in {16, 32, 64}, TK in {32, 64}");
  constexpr int RB = TN / 8, NG = TK / 16;
  const uint q = (lane & 1) | ((lane >> 2) & 2), rr = ((lane >> 1) & 3) | ((lane >> 2) & 4);
  for (int rb = 0; rb < RB; rb++) {
    half4 v[NG];
    QDeqq<Q, NG>::run(W + (ulong)(r0 + rr + 8 * rb) * row_bytes, k0, q, v, tb);
    for (int g = 0; g < NG; g++) {
      const int i = 4 * (RB * g + rb);
      cw[i] = v[g].x; cw[i + 1] = v[g].y; cw[i + 2] = v[g].z; cw[i + 3] = v[g].w;
    }
  }
}

// gate/up: simdgroup tile = TM tokens x TN rows of gate AND up (two accumulators), K step TK.
template <typename Q, int TM, int TN, int TK, int NSG>
kernel void q_moe_rgu(device const half* Xg [[buffer(0)]], device half* H [[buffer(1)]],
                      constant QMoePArgs& a [[buffer(2)]], device const float* wt [[buffer(3)]],
                      device const QExpPtr* ptrs [[buffer(4)]], device const uint* offs [[buffer(5)]],
                      device const uint2* tiles [[buffer(6)]], device const uint2* tbl [[buffer(7)]],
                      threadgroup uint2* tgt [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const QMoeTbl tb = q_moe_tbl<Q, NSG>(tbl, tgt, sg, lane);
  const uint2 te = tiles[tg.y];
  const uint e = te.x, base = offs[e] + te.y, m = min(offs[e + 1] - base, (uint)TM);
  const uint r0 = (tg.x * NSG + sg) * TN;
  if (r0 >= a.inter) return;
  device const uchar* Wg = ptrs[e].g;
  device const uchar* Wu = ptrs[e].u;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(Xg + (ulong)base * a.dim),
                                                              dextents<int32_t, 2>(a.dim, m));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto wg = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto wu = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cG = op.template get_destination_cooperative_tensor<decltype(tX), decltype(wg), float>();
  auto cU = op.template get_destination_cooperative_tensor<decltype(tX), decltype(wu), float>();
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) if (cG.is_valid_element(i)) { cG[i] = 0; cU[i] = 0; }
  for (uint k0 = 0; k0 < a.dim; k0 += TK) {
    q_coop_deq<Q, TN, TK>(wg, Wg, a.rb_gu, r0, k0, lane, tb);
    q_coop_deq<Q, TN, TK>(wu, Wu, a.rb_gu, r0, k0, lane, tb);
    auto xs = tX.slice(k0, 0);
    op.run(xs, wg, cG);
    op.run(xs, wu, cU);
  }
  // destination layout: elements 4j..4j+3 are 4 consecutive weight rows of one token (one 8-byte H store)
  for (uint16_t i = 0; i < cG.get_capacity(); i += 4) {
    auto idx = cG.get_multidimensional_index(i);  // [0] = weight row within tile, [1] = token within tile
    const uint r = r0 + idx[0], t = idx[1];
    if (t >= m || r >= a.inter) continue;
    const float w = base + t < a.total ? wt[base + t] : 1.f;
    const float4 g = min(float4(cG[i], cG[i + 1], cG[i + 2], cG[i + 3]), 10.f);
    const float4 u = clamp(float4(cU[i], cU[i + 1], cU[i + 2], cU[i + 3]), -10.f, 10.f);
    const half4 h = half4(g / (1.f + exp(-g)) * u * w);
    auto id3 = cG.get_multidimensional_index(i + 3);
    if (id3[0] == idx[0] + 3 && id3[1] == idx[1]) {
      *(device half4*)(H + (ulong)(base + t) * a.inter + r) = h;
    } else {  // layout guard (never taken on M5): scalar stores
      for (int c = 0; c < 4; c++) {
        auto ic = cG.get_multidimensional_index(i + c);
        H[(ulong)(base + ic[1]) * a.inter + r0 + ic[0]] = h[c];
      }
    }
  }
}

// gate|up merged into ONE matmul per K step (TM tokens x 2*TNH coop rows: rows [0, TNH) = gate rows r0.., rows
// [TNH, 2*TNH) = up rows r0..): the X tile is read once per K step instead of once per matrix. Destination layout (M5,
// dumped by bench_moep_probe, guarded below and by the tests): element 8*(NB*tg + cb) + 4*s + c holds row 16*cb + a + c,
// token 16*tg + t0 + 8*s (NB = TN/16 row blocks), so the up partner of a gate element i sits at i + TNH/2 in the same lane.
template <typename Q, int TN, int TK, typename CW>
static inline void q_coop_deq2(thread CW& cw, device const uchar* Wg, device const uchar* Wu, uint row_bytes, uint r0, uint k0,
                               ushort lane, thread const QMoeTbl& tb) {
  static_assert(TN % 16 == 0 && TK % 32 == 0, "layout verified for TN in {32, 64}, TK in {32, 64}");
  constexpr int RB = TN / 8, NG = TK / 16;
  const uint q = (lane & 1) | ((lane >> 2) & 2), rr = ((lane >> 1) & 3) | ((lane >> 2) & 4);
  for (int rb = 0; rb < RB; rb++) {
    half4 v[NG];
    device const uchar* W = rb < RB / 2 ? Wg : Wu;
    const int rh = rb < RB / 2 ? rb : rb - RB / 2;
    QDeqq<Q, NG>::run(W + (ulong)(r0 + rr + 8 * rh) * row_bytes, k0, q, v, tb);
    for (int g = 0; g < NG; g++) {
      const int i = 4 * (RB * g + rb);
      cw[i] = v[g].x; cw[i + 1] = v[g].y; cw[i + 2] = v[g].z; cw[i + 3] = v[g].w;
    }
  }
}
template <typename Q, int TM, int TNH, int TK, int NSG>
kernel void q_moe_rgu2(device const half* Xg [[buffer(0)]], device half* H [[buffer(1)]],
                       constant QMoePArgs& a [[buffer(2)]], device const float* wt [[buffer(3)]],
                       device const QExpPtr* ptrs [[buffer(4)]], device const uint* offs [[buffer(5)]],
                       device const uint2* tiles [[buffer(6)]], device const uint2* tbl [[buffer(7)]],
                       threadgroup uint2* tgt [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int TN = 2 * TNH;
  const QMoeTbl tb = q_moe_tbl<Q, NSG>(tbl, tgt, sg, lane);
  const uint2 te = tiles[tg.y];
  const uint e = te.x, base = offs[e] + te.y, m = min(offs[e + 1] - base, (uint)TM);
  const uint r0 = (tg.x * NSG + sg) * TNH;
  if (r0 >= a.inter) return;
  device const uchar* Wg = ptrs[e].g;
  device const uchar* Wu = ptrs[e].u;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(Xg + (ulong)base * a.dim),
                                                              dextents<int32_t, 2>(a.dim, m));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto w = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(w), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint k0 = 0; k0 < a.dim; k0 += TK) {
    q_coop_deq2<Q, TN, TK>(w, Wg, Wu, a.rb_gu, r0, k0, lane, tb);
    auto xs = tX.slice(k0, 0);
    op.run(xs, w, cT);
  }
  constexpr int P = TNH / 2;  // up partner offset
  for (uint16_t i = 0; i < cT.get_capacity(); i += 4) {
    auto idx = cT.get_multidimensional_index(i);  // [0] = coop row, [1] = token within tile
    if (idx[0] >= TNH) continue;
    const uint r = r0 + idx[0], t = idx[1];
    if (t >= m || r >= a.inter) continue;
    const float w8 = base + t < a.total ? wt[base + t] : 1.f;
    auto id3 = cT.get_multidimensional_index(i + 3);
    auto ip = cT.get_multidimensional_index(i + P);
    auto ip3 = cT.get_multidimensional_index(i + P + 3);
    if (id3[0] == idx[0] + 3 && id3[1] == idx[1] && ip[0] == idx[0] + TNH && ip[1] == idx[1] && ip3[0] == idx[0] + TNH + 3 &&
        ip3[1] == idx[1]) {
      const float4 g = min(float4(cT[i], cT[i + 1], cT[i + 2], cT[i + 3]), 10.f);
      const float4 u = clamp(float4(cT[i + P], cT[i + P + 1], cT[i + P + 2], cT[i + P + 3]), -10.f, 10.f);
      *(device half4*)(H + (ulong)(base + t) * a.inter + r) = half4(g / (1.f + exp(-g)) * u * w8);
    } else {  // layout guard (never taken on M5): poison the rows so the tests fail loudly
      for (int c = 0; c < 4; c++) H[(ulong)(base + t) * a.inter + r + c] = half(NAN);
    }
  }
}

// ---- dense shared expert: weights pre-dequantized to half (W16: gate rows, then up rows; or down), tensor-API GEMM
// straight from device memory (the q_mm_f16 shape, 92-98% of peak). gate/up with the SwiGLU epilogue into H rows
// [total, total + n); down via q_mm_f16_64x128 into Z rows [total, total + n).
kernel void q_moe_dgu(device const half* Xg [[buffer(0)]], device half* H [[buffer(1)]], constant QMoePArgs& a [[buffer(2)]],
                      device const half* W16 [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr int TM = 64, TN = 64, TK = 64;
  const uint t0 = tg.y * TM, r0 = tg.x * TN;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(Xg + (ulong)a.total * a.dim),
                                                              dextents<int32_t, 2>(a.dim, a.n));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tG((device half*)W16, dextents<int32_t, 2>(a.dim, a.inter));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tU((device half*)(W16 + (ulong)a.inter * a.dim),
                                                              dextents<int32_t, 2>(a.dim, a.inter));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cG = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tG), float>();
  auto cU = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tU), float>();
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) if (cG.is_valid_element(i)) { cG[i] = 0; cU[i] = 0; }
  for (uint k0 = 0; k0 < a.dim; k0 += TK) {
    auto xs = tX.slice(k0, t0);
    auto gs = tG.slice(k0, r0);
    auto us = tU.slice(k0, r0);
    op.run(xs, gs, cG);
    op.run(xs, us, cU);
  }
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
    if (!cG.is_valid_element(i)) continue;
    auto idx = cG.get_multidimensional_index(i);
    const uint r = r0 + idx[0], t = t0 + idx[1];
    if (r >= a.inter || t >= a.n) continue;
    float g = min(cG[i], 10.f), u = clamp(cU[i], -10.f, 10.f);
    H[(ulong)(a.total + t) * a.inter + r] = half(g / (1.f + exp(-g)) * u);
  }
}

// dense shared gate/up as ONE 64 x 128 GEMM per tile: W16 holds gate and up interleaved in blocks of 64 rows
// (block j = gate rows 64j.. then up rows 64j..), so a 128-row slice is [gate | up] of the same 64 H rows. Destination
// layout of matmul2d(64, 128) on 4 simdgroups (M5, bench_moep_probe tg4; guarded below and by the tests): a thread
// holds columns {c, c+32, c+64, c+96} (+0..3) of its tokens, the up partner of gate element i at i + 16.
kernel void q_moe_dgu2(device const half* Xg [[buffer(0)]], device half* H [[buffer(1)]], constant QMoePArgs& a [[buffer(2)]],
                       device const half* W16 [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr int TM = 64, TN = 128, TK = 64;
  const uint t0 = tg.y * TM, b0 = tg.x * TN;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(Xg + (ulong)a.total * a.dim),
                                                              dextents<int32_t, 2>(a.dim, a.n));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tW((device half*)W16, dextents<int32_t, 2>(a.dim, 2 * a.inter));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint k0 = 0; k0 < a.dim; k0 += TK) {
    auto xs = tX.slice(k0, t0);
    auto ws = tW.slice(k0, b0);
    op.run(xs, ws, cT);
  }
  for (uint16_t i = 0; i < cT.get_capacity(); i += 4) {
    if (!cT.is_valid_element(i)) continue;
    auto idx = cT.get_multidimensional_index(i);  // [0] = column within the 128 (gate < 64 <= up), [1] = token
    if (idx[0] >= 64) continue;
    const uint r = tg.x * 64 + idx[0], t = t0 + idx[1];
    if (t >= a.n || r >= a.inter) continue;
    auto id3 = cT.get_multidimensional_index(i + 3);
    auto ip = cT.get_multidimensional_index(i + 16);
    auto ip3 = cT.get_multidimensional_index(i + 19);
    if (id3[0] == idx[0] + 3 && id3[1] == idx[1] && ip[0] == idx[0] + 64 && ip[1] == idx[1] && ip3[0] == idx[0] + 67 &&
        ip3[1] == idx[1]) {
      const float4 g = min(float4(cT[i], cT[i + 1], cT[i + 2], cT[i + 3]), 10.f);
      const float4 u = clamp(float4(cT[i + 16], cT[i + 17], cT[i + 18], cT[i + 19]), -10.f, 10.f);
      *(device half4*)(H + (ulong)(a.total + t) * a.inter + r) = half4(g / (1.f + exp(-g)) * u);
    } else {  // layout guard (never taken on M5): poison so the tests fail loudly
      for (int c = 0; c < 4; c++) H[(ulong)(a.total + t) * a.inter + r + c] = half(NAN);
    }
  }
}

template <typename Q, int TM, int TN, int TK, int NSG>
kernel void q_moe_rdown(device const half* H [[buffer(0)]], device float* Z [[buffer(1)]],
                        constant QMoePArgs& a [[buffer(2)]], device const QExpPtr* ptrs [[buffer(4)]],
                        device const uint* offs [[buffer(5)]], device const uint2* tiles [[buffer(6)]],
                        device const uint2* tbl [[buffer(7)]], threadgroup uint2* tgt [[threadgroup(0)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
  const QMoeTbl tb = q_moe_tbl<Q, NSG>(tbl, tgt, sg, lane);
  const uint2 te = tiles[tg.y];
  const uint e = te.x, base = offs[e] + te.y, m = min(offs[e + 1] - base, (uint)TM);
  const uint r0 = (tg.x * NSG + sg) * TN;
  if (r0 >= a.dim) return;
  device const uchar* Wd = ptrs[e].d;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)(H + (ulong)base * a.inter),
                                                              dextents<int32_t, 2>(a.inter, m));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tZ(Z + (ulong)base * a.dim, dextents<int32_t, 2>(a.dim, m));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto wd = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(wd), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint k0 = 0; k0 < a.inter; k0 += TK) {
    q_coop_deq<Q, TN, TK>(wd, Wd, a.rb_d, r0, k0, lane, tb);
    auto xs = tX.slice(k0, 0);
    op.run(xs, wd, cT);
  }
  auto zs = tZ.slice(r0, 0);
  cT.store(zs);
}

#define Q_MOER_GU1(Q, name, TM, TN, TK, NSG, PS)                                                                \
  template [[host_name("q_moe_rgu_" name "_" #TM "x" #TN "k" #TK "s" #NSG PS)]] kernel void                         \
  q_moe_rgu<Q, TM, TN, TK, NSG>(device const half*, device half*, constant QMoePArgs&, device const float*,      \
                                   device const QExpPtr*, device const uint*, device const uint2*, device const uint2*, \
                                   threadgroup uint2*, uint2, ushort, ushort);
#define Q_MOER_DN1(Q, name, TM, TN, TK, NSG, PS)                                                                \
  template [[host_name("q_moe_rdown_" name "_" #TM "x" #TN "k" #TK "s" #NSG PS)]] kernel void                       \
  q_moe_rdown<Q, TM, TN, TK, NSG>(device const half*, device float*, constant QMoePArgs&, device const QExpPtr*, \
                                     device const uint*, device const uint2*, device const uint2*, threadgroup uint2*,   \
                                     uint2, ushort, ushort);
#define Q_MOER_GU2(Q, name, TM, TN, TK, NSG)                                                                      \
  template [[host_name("q_moe_rgu2_" name "_" #TM "x" #TN "k" #TK "s" #NSG)]] kernel void                          \
  q_moe_rgu2<Q, TM, TN, TK, NSG>(device const half*, device half*, constant QMoePArgs&, device const float*,     \
                                  device const QExpPtr*, device const uint*, device const uint2*, device const uint2*, \
                                  threadgroup uint2*, uint2, ushort, ushort);
#define Q_MOER_GU(Q, name, TM, TN, TK, NSG) Q_MOER_GU1(Q, name, TM, TN, TK, NSG, "")
#define Q_MOER_DN(Q, name, TM, TN, TK, NSG) Q_MOER_DN1(Q, name, TM, TN, TK, NSG, "")
#define Q_MOER_ALL(Q, name)                                                                                         \
  Q_MOER_GU(Q, name, 64, 16, 64, 4) Q_MOER_GU(Q, name, 48, 16, 64, 4) Q_MOER_GU(Q, name, 32, 32, 64, 4)              \
  Q_MOER_GU(Q, name, 16, 32, 64, 4) Q_MOER_GU(Q, name, 64, 16, 32, 4) Q_MOER_GU(Q, name, 48, 16, 32, 4)              \
  Q_MOER_GU(Q, name, 32, 32, 32, 4) Q_MOER_GU(Q, name, 16, 32, 32, 4)                                               \
  Q_MOER_DN(Q, name, 64, 32, 32, 4) Q_MOER_DN(Q, name, 48, 32, 32, 4) Q_MOER_DN(Q, name, 32, 64, 32, 4)              \
  Q_MOER_DN(Q, name, 16, 64, 32, 4)                                                                                   \
  Q_MOER_GU2(Q, name, 64, 16, 32, 4) Q_MOER_GU2(Q, name, 48, 16, 32, 4) Q_MOER_GU2(Q, name, 32, 32, 32, 4)           \
  Q_MOER_GU2(Q, name, 16, 32, 32, 4) Q_MOER_GU2(Q, name, 64, 16, 64, 4) Q_MOER_GU2(Q, name, 48, 16, 64, 4)           \
  Q_MOER_GU2(Q, name, 32, 32, 64, 4) Q_MOER_GU2(Q, name, 16, 32, 64, 4)
Q_MOER_ALL(QB_IQ1S, "iq1_s")
Q_MOER_ALL(QB_IQ2XXS, "iq2_xxs")
Q_MOER_ALL(QB_IQ1M, "iq1_m")
Q_MOER_ALL(QB_IQ2XS, "iq2_xs")
// shared expert through the grouped kernels (MACSEEK_MOEP_SHDENSE=0)
Q_MOER_GU(QB_Q6K, "q6_k", 32, 32, 32, 4) Q_MOER_GU2(QB_Q6K, "q6_k", 32, 32, 32, 4) Q_MOER_DN(QB_Q6K, "q6_k", 32, 64, 32, 4)
Q_MOER_GU(QB_Q8_0, "q8_0", 32, 32, 32, 4) Q_MOER_GU2(QB_Q8_0, "q8_0", 32, 32, 32, 4) Q_MOER_DN(QB_Q8_0, "q8_0", 32, 64, 32, 4)
Q_MOER_GU(QB_Q5K, "q5_k", 32, 32, 32, 4) Q_MOER_GU2(QB_Q5K, "q5_k", 32, 32, 32, 4) Q_MOER_DN(QB_Q5K, "q5_k", 32, 64, 32, 4)


kernel void q_moe_reduce(device const float* Z [[buffer(0)]], device float* Y [[buffer(1)]],
                         constant QMoePArgs& a [[buffer(2)]], device const int* slot_of [[buffer(3)]],
                         constant uint& has_shared [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const uint t = gid.y, c4 = gid.x;
  if (t >= a.n || c4 >= a.dim / 4) return;
  float4 s = has_shared ? ((device const float4*)(Z + (ulong)(a.total + t) * a.dim))[c4] : float4(0);
  bool any = has_shared != 0;
  for (uint k = 0; k < a.k_per_token; k++) {
    const int i = slot_of[t * a.k_per_token + k];
    if (i >= 0) { s += ((device const float4*)(Z + (ulong)i * a.dim))[c4]; any = true; }
  }
  if (!any) return;  // token has no expert in this call: Y unchanged (saves the Y read + write)
  device float4* y = (device float4*)(Y + (ulong)t * a.dim) + c4;
  *y += s;
}

#define Q_MOEP_INST1(Q, name, TM)                                                                                 \
  template [[host_name("q_moe_mm_gu_" name "_m" #TM)]] kernel void q_moe_mm_gu<Q, TM, 64, 4>(device const half*,  \
      device half*, constant QMoePArgs&, device const float*, device const QExpPtr*, device const uint*,           \
      device const uint2*, threadgroup half*, uint2, ushort);                                                      \
  template [[host_name("q_moe_mm_down_" name "_m" #TM)]] kernel void q_moe_mm_down<Q, TM, 64, 4>(                 \
      device const half*, device float*, constant QMoePArgs&, device const QExpPtr*, device const uint*,           \
      device const uint2*, threadgroup half*, uint2, ushort);
#define Q_MOEP_INST(Q, name) Q_MOEP_INST1(Q, name, 64) Q_MOEP_INST1(Q, name, 128)
Q_MOEP_INST(QB_IQ1S, "iq1_s")
Q_MOEP_INST(QB_IQ2XXS, "iq2_xxs")
Q_MOEP_INST(QB_Q6K, "q6_k")
Q_MOEP_INST(QB_Q8_0, "q8_0")
Q_MOEP_INST(QB_IQ1M, "iq1_m")
Q_MOEP_INST(QB_IQ2XS, "iq2_xs")
Q_MOEP_INST(QB_Q5K, "q5_k")
