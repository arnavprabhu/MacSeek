// Shared Metal helpers for the quant module (all kernels prefixed q_).
#pragma once
#include "common.h"
#include "quant_grids.h"

// half-valued tables (16-byte aligned so one entry = one uint4 load)
constant uint q_iq1s_gridh[8192] __attribute__((aligned(16))) = {Q_IQ1S_GRID_HALF_VALUES};
constant uint q_iq2xxs_gridh[1024] __attribute__((aligned(16))) = {Q_IQ2XXS_GRID_HALF_VALUES};
constant uint q_iq2xs_gridh[2048] __attribute__((aligned(16))) = {Q_IQ2XS_GRID_HALF_VALUES};
constant uint q_iq2_signh[512] __attribute__((aligned(16))) = {Q_IQ2_SIGNH_VALUES};
// IQ1_M decode (V1): grid + delta (+1/8 for entries 0..2047, -1/8 for 2048..4095), index = qs | qh nibble << 8
constant uint q_iq1m_gridd[16384] __attribute__((aligned(16))) = {Q_IQ1M_GRIDD_HALF_VALUES};
static inline uint4 q_tbl4(constant uint* t, uint i) { return ((constant uint4*)t)[i]; }

// ---- unaligned-safe loads: GGUF blocks are only 2-byte aligned (34/50/66/176/210-byte blocks) ----
static inline uint q_ld32(device const uchar* p) {
  device const ushort* s = (device const ushort*)p;
  return uint(s[0]) | (uint(s[1]) << 16);
}
// 32-bit load for fields at a 4-byte multiple of a block whose size is a multiple of 4 (IQ1_M 56 B, Q5_K 176 B):
// aligned given the documented >= 4-byte aligned weight buffer offsets
static inline uint q_ld32a(device const uchar* p) { return *(device const uint*)p; }
static inline ushort q_ld16(device const uchar* p) { return *(device const ushort*)p; }
static inline float q_ldh(device const uchar* p) { return float(*(device const half*)p); }
static inline float4 q_u8x4(uint v) { return float4(as_type<uchar4>(v)); }
static inline float4 q_i8x4(uint v) { return float4(as_type<char4>(v)); }

// Q5_K / Q4_K 6-bit scale + min unpack (sub-block j in 0..7) from the 12 scale bytes
static inline void q_scale_min_k4(int j, device const uchar* q, thread float& d, thread float& m) {
  if (j < 4) { d = float(q[j] & 63); m = float(q[j + 4] & 63); }
  else { d = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); m = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

// Arguments shared by the matvec family (decode n=1 and verify n<=16).
struct QMvArgs {
  uint rows;            // total output rows
  uint cols;            // input elements per row
  uint row_bytes;       // bytes per weight row
  uint rows_per_group;  // block-diagonal: rows in each group (== rows when groups == 1)
  uint x_group_stride;  // elements between the inputs of consecutive groups
  uint accumulate;      // y += W x instead of y = W x
  uint ksplit;          // simdgroups splitting K inside a threadgroup
  uint n;               // number of activation columns (matvec_n)
  uint x_stride;        // elements between activation columns
  uint y_stride;        // elements between output columns
};
