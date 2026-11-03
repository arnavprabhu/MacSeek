// Fused MoE decode (one token): routed experts (IQ1_S / IQ2_XXS / IQ1_M / IQ2_XS) + shared expert (Q6_K / Q8_0 / Q5_K).
//   q_moe_gu  : h[e][r] = silu(min(gate_e[r].x, 10)) * clamp(up_e[r].x, -10, 10) * w_e    (all experts, one dispatch)
//   q_moe_down: y[r] += sum_e down_e[r] . h[e]                                            (all experts, one dispatch)
// Expert e < n_routed uses the routed type QR and buffers gate[e]/up[e]/down[e]; e == n_routed is the shared expert
// (type QS, weight 1). Every expert's matrix is a separate binding (buffer + byte offset) so slots can live anywhere.
#include "quant_blocks.h"

struct QMoeArgs {
  uint n_routed;     // routed experts (0..6)
  uint has_shared;   // 1 if the shared expert is present
  uint dim;          // 5120 (gate/up cols, down rows)
  uint inter;        // 2304 (gate/up rows, down cols)
  uint rb_gu_r, rb_d_r, rb_gu_s, rb_d_s;  // row bytes: routed gate/up, routed down, shared gate/up, shared down
  uint ksplit;       // simdgroups splitting K per threadgroup
  uint tiles;        // threadgroups per expert (q_moe_gu)
};

// decoder used for the routed experts' gate/up rows in the fused MoE decode kernels (lane count per block may differ
// from the matvec choice: gate and up share the x registers, so wider lanes win there; bench_qmatvec --moe)
template <typename Q> struct QMoeGuDec { typedef Q T; };
template <> struct QMoeGuDec<QB_IQ1M> { typedef QB_IQ1M_T<8> T; };

template <typename T>
static inline T q_pick6(uint e, T a0, T a1, T a2, T a3, T a4, T a5) {
  return e == 0 ? a0 : e == 1 ? a1 : e == 2 ? a2 : e == 3 ? a3 : e == 4 ? a4 : a5;
}

static inline float q_swiglu(float g, float u, float w) {
  g = min(g, 10.f);
  u = clamp(u, -10.f, 10.f);
  return g / (1.f + exp(-g)) * u * w;
}

// gate+up rows [row0, row0+NR) of one expert; returns per-row (g, u) partial sums reduced over the simdgroup
template <typename Q, int NR>
static inline void q_gu_rows(device const uchar* G, device const uchar* U, uint row_bytes, uint rows, uint row0,
                             device const float* x, uint cols, ushort kk, ushort ks, ushort lane,
                             thread float* ag, thread float* au) {
  constexpr ushort BPI = 32 / Q::LPB;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = cols / Q::QK;
  for (int r = 0; r < NR; r++) { ag[r] = 0.f; au[r] = 0.f; }
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    float4 xr[Q::X4];
    Q::load_x(x + ib * Q::QK, sub, xr);
    for (int r = 0; r < NR; r++) {
      const ulong off = (ulong)min(row0 + r, rows - 1) * row_bytes + ib * Q::BB;
      ag[r] += Q::dot(G + off, sub, xr);
      au[r] += Q::dot(U + off, sub, xr);
    }
  }
  for (int r = 0; r < NR; r++) { ag[r] = simd_sum(ag[r]); au[r] = simd_sum(au[r]); }
}

template <typename QR, typename QS, int NR>
kernel void q_moe_gu(device const float* x [[buffer(0)]], device float* h [[buffer(1)]],
                     constant QMoeArgs& a [[buffer(2)]], device const float* wts [[buffer(3)]],
                     device const uchar* g0 [[buffer(4)]], device const uchar* g1 [[buffer(5)]],
                     device const uchar* g2 [[buffer(6)]], device const uchar* g3 [[buffer(7)]],
                     device const uchar* g4 [[buffer(8)]], device const uchar* g5 [[buffer(9)]],
                     device const uchar* u0 [[buffer(10)]], device const uchar* u1 [[buffer(11)]],
                     device const uchar* u2 [[buffer(12)]], device const uchar* u3 [[buffer(13)]],
                     device const uchar* u4 [[buffer(14)]], device const uchar* u5 [[buffer(15)]],
                     device const uchar* gs [[buffer(16)]], device const uchar* us [[buffer(17)]],
                     threadgroup float* red [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint e = tg / a.tiles;
  const uint row0 = ((tg % a.tiles) * rsg + rs) * NR;
  float ag[NR], au[NR];
  const bool shared = e >= a.n_routed;
  if (!shared) {
    q_gu_rows<typename QMoeGuDec<QR>::T, NR>(q_pick6(e, g0, g1, g2, g3, g4, g5), q_pick6(e, u0, u1, u2, u3, u4, u5), a.rb_gu_r, a.inter, row0,
                      x, a.dim, kk, ks, lane, ag, au);
  } else {
    q_gu_rows<QS, NR>(gs, us, a.rb_gu_s, a.inter, row0, x, a.dim, kk, ks, lane, ag, au);
  }
  if (ks > 1) {
    if (lane == 0) for (int r = 0; r < NR; r++) {
      red[((rs * NR + r) * 2 + 0) * ks + kk] = ag[r]; red[((rs * NR + r) * 2 + 1) * ks + kk] = au[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++) {
      float sg_ = 0, su = 0;
      for (ushort k = 0; k < ks; k++) { sg_ += red[((rs * NR + r) * 2) * ks + k]; su += red[((rs * NR + r) * 2 + 1) * ks + k]; }
      ag[r] = sg_; au[r] = su;
    }
  }
  if (lane == 0) {
    const float w = shared ? 1.f : wts[e];
    for (int r = 0; r < NR; r++) {
      const uint row = row0 + r;
      if (row < a.inter) h[e * a.inter + row] = q_swiglu(ag[r], au[r], w);
    }
  }
}

// down: rows [row0, row0+NR) of y; K space = (routed expert, block) pairs flattened, then the shared expert.
template <typename QR, typename QS, int NR>
kernel void q_moe_down(device const float* h [[buffer(0)]], device float* y [[buffer(1)]],
                       constant QMoeArgs& a [[buffer(2)]],
                       device const uchar* d0 [[buffer(4)]], device const uchar* d1 [[buffer(5)]],
                       device const uchar* d2 [[buffer(6)]], device const uchar* d3 [[buffer(7)]],
                       device const uchar* d4 [[buffer(8)]], device const uchar* d5 [[buffer(9)]],
                       device const uchar* ds [[buffer(10)]],
                       threadgroup float* red [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint row0 = (tg * rsg + rs) * NR;
  float acc[NR];
  for (int r = 0; r < NR; r++) acc[r] = 0.f;
  {
    constexpr ushort BPI = 32 / QR::LPB;
    const ushort sub = lane % QR::LPB, slot = lane / QR::LPB;
    const uint nb = a.inter / QR::QK, nu = nb * a.n_routed;
    for (uint u = kk * BPI + slot; u < nu; u += ks * BPI) {
      const uint e = u / nb, ib = u - e * nb;
      device const uchar* D = q_pick6(e, d0, d1, d2, d3, d4, d5);
      float4 xr[QR::X4];
      QR::load_x(h + e * a.inter + ib * QR::QK, sub, xr);
      for (int r = 0; r < NR; r++)
        acc[r] += QR::dot(D + (ulong)min(row0 + r, a.dim - 1) * a.rb_d_r + ib * QR::BB, sub, xr);
    }
  }
  if (a.has_shared) {
    constexpr ushort BPI = 32 / QS::LPB;
    const ushort sub = lane % QS::LPB, slot = lane / QS::LPB;
    const uint nb = a.inter / QS::QK;
    device const float* hs = h + a.n_routed * a.inter;
    for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
      float4 xr[QS::X4];
      QS::load_x(hs + ib * QS::QK, sub, xr);
      for (int r = 0; r < NR; r++)
        acc[r] += QS::dot(ds + (ulong)min(row0 + r, a.dim - 1) * a.rb_d_s + ib * QS::BB, sub, xr);
    }
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
      if (row < a.dim) y[row] += acc[r];
    }
}

#define Q_MOE_GU_INST(QR, QS, name, NR)                                                                            \
  template [[host_name("q_moe_gu_" name "_r" #NR)]] kernel void q_moe_gu<QR, QS, NR>(                             \
      device const float*, device float*, constant QMoeArgs&, device const float*, device const uchar*,           \
      device const uchar*, device const uchar*, device const uchar*, device const uchar*, device const uchar*,    \
      device const uchar*, device const uchar*, device const uchar*, device const uchar*, device const uchar*,    \
      device const uchar*, device const uchar*, device const uchar*, threadgroup float*, uint, ushort, ushort, ushort);
#define Q_MOE_DOWN_INST(QR, QS, name, NR)                                                                          \
  template [[host_name("q_moe_down_" name "_r" #NR)]] kernel void q_moe_down<QR, QS, NR>(                         \
      device const float*, device float*, constant QMoeArgs&, device const uchar*, device const uchar*,           \
      device const uchar*, device const uchar*, device const uchar*, device const uchar*, device const uchar*,    \
      threadgroup float*, uint, ushort, ushort, ushort);
#define Q_MOE_INST(QR, QS, name) \
  Q_MOE_GU_INST(QR, QS, name, 1) Q_MOE_GU_INST(QR, QS, name, 2) Q_MOE_GU_INST(QR, QS, name, 4) \
  Q_MOE_DOWN_INST(QR, QS, name, 1) Q_MOE_DOWN_INST(QR, QS, name, 2) Q_MOE_DOWN_INST(QR, QS, name, 4)

// every routed type x shared type (names "<routed>_<shared>", built by moe_pair in quant.mm)
#define Q_MOE_SHARED(M, QR, nr) M(QR, QB_Q6K, nr "_q6_k") M(QR, QB_Q8_0, nr "_q8_0") M(QR, QB_Q5K, nr "_q5_k")
#define Q_MOE_PAIRS(M) \
  Q_MOE_SHARED(M, QB_IQ1S, "iq1_s") Q_MOE_SHARED(M, QB_IQ2XXS, "iq2_xxs") Q_MOE_SHARED(M, QB_IQ1M, "iq1_m") \
  Q_MOE_SHARED(M, QB_IQ2XS, "iq2_xs")
Q_MOE_PAIRS(Q_MOE_INST)


// ---- pointer-table variants (decode without host re-binding): routed expert e's matrices come from tbl[e]
// (device addresses, written by the CPU before the GPU reaches this dispatch). Callers must make the expert arena
// resident (useResource) in the encoder.
struct QDecPtr { device const uchar* g; device const uchar* u; device const uchar* d; };

template <typename QR, typename QS, int NR>
kernel void q_moe_gu_t(device const float* x [[buffer(0)]], device float* h [[buffer(1)]],
                       constant QMoeArgs& a [[buffer(2)]], device const float* wts [[buffer(3)]],
                       device const QDecPtr* tbl [[buffer(4)]],
                       device const uchar* gs [[buffer(16)]], device const uchar* us [[buffer(17)]],
                       threadgroup float* red [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint e = tg / a.tiles;
  const uint row0 = ((tg % a.tiles) * rsg + rs) * NR;
  float ag[NR], au[NR];
  const bool shared = e >= a.n_routed;
  if (!shared) q_gu_rows<typename QMoeGuDec<QR>::T, NR>(tbl[e].g, tbl[e].u, a.rb_gu_r, a.inter, row0, x, a.dim, kk, ks, lane, ag, au);
  else q_gu_rows<QS, NR>(gs, us, a.rb_gu_s, a.inter, row0, x, a.dim, kk, ks, lane, ag, au);
  if (ks > 1) {
    if (lane == 0) for (int r = 0; r < NR; r++) {
      red[((rs * NR + r) * 2 + 0) * ks + kk] = ag[r]; red[((rs * NR + r) * 2 + 1) * ks + kk] = au[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk != 0) return;
    for (int r = 0; r < NR; r++) {
      float sg_ = 0, su = 0;
      for (ushort k = 0; k < ks; k++) { sg_ += red[((rs * NR + r) * 2) * ks + k]; su += red[((rs * NR + r) * 2 + 1) * ks + k]; }
      ag[r] = sg_; au[r] = su;
    }
  }
  if (lane == 0) {
    const float w = shared ? 1.f : wts[e];
    for (int r = 0; r < NR; r++) {
      const uint row = row0 + r;
      if (row < a.inter) h[e * a.inter + row] = q_swiglu(ag[r], au[r], w);
    }
  }
}

template <typename QR, typename QS, int NR>
kernel void q_moe_down_t(device const float* h [[buffer(0)]], device float* y [[buffer(1)]],
                         constant QMoeArgs& a [[buffer(2)]], device const QDecPtr* tbl [[buffer(4)]],
                         device const uchar* ds [[buffer(10)]],
                         threadgroup float* red [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                         ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint row0 = (tg * rsg + rs) * NR;
  float acc[NR];
  for (int r = 0; r < NR; r++) acc[r] = 0.f;
  {
    constexpr ushort BPI = 32 / QR::LPB;
    const ushort sub = lane % QR::LPB, slot = lane / QR::LPB;
    const uint nb = a.inter / QR::QK, nu = nb * a.n_routed;
    for (uint u = kk * BPI + slot; u < nu; u += ks * BPI) {
      const uint e = u / nb, ib = u - e * nb;
      device const uchar* D = tbl[e].d;
      float4 xr[QR::X4];
      QR::load_x(h + e * a.inter + ib * QR::QK, sub, xr);
      for (int r = 0; r < NR; r++)
        acc[r] += QR::dot(D + (ulong)min(row0 + r, a.dim - 1) * a.rb_d_r + ib * QR::BB, sub, xr);
    }
  }
  if (a.has_shared) {
    constexpr ushort BPI = 32 / QS::LPB;
    const ushort sub = lane % QS::LPB, slot = lane / QS::LPB;
    const uint nb = a.inter / QS::QK;
    device const float* hs = h + a.n_routed * a.inter;
    for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
      float4 xr[QS::X4];
      QS::load_x(hs + ib * QS::QK, sub, xr);
      for (int r = 0; r < NR; r++)
        acc[r] += QS::dot(ds + (ulong)min(row0 + r, a.dim - 1) * a.rb_d_s + ib * QS::BB, sub, xr);
    }
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
      if (row < a.dim) y[row] += acc[r];
    }
}

#define Q_MOE_T_INST(QR, QS, name, NR)                                                                             \
  template [[host_name("q_moe_gu_t_" name "_r" #NR)]] kernel void q_moe_gu_t<QR, QS, NR>(                         \
      device const float*, device float*, constant QMoeArgs&, device const float*, device const QDecPtr*,         \
      device const uchar*, device const uchar*, threadgroup float*, uint, ushort, ushort, ushort);                 \
  template [[host_name("q_moe_down_t_" name "_r" #NR)]] kernel void q_moe_down_t<QR, QS, NR>(                     \
      device const float*, device float*, constant QMoeArgs&, device const QDecPtr*, device const uchar*,         \
      threadgroup float*, uint, ushort, ushort, ushort);
#define Q_MOE_T_ALL(QR, QS, name) Q_MOE_T_INST(QR, QS, name, 1) Q_MOE_T_INST(QR, QS, name, 2) Q_MOE_T_INST(QR, QS, name, 4)
Q_MOE_PAIRS(Q_MOE_T_ALL)
