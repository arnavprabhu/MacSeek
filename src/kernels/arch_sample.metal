// Sampling helpers over the 129280 vocab: argmax (greedy, optional Gumbel noise = exact temperature sampling),
// top-k (k <= 64) with the full-vocab log-sum-exp (for CPU top-p), log_softmax (quality tool), rmsnorm variants.
#include "arch_common.h"

struct ArchSampleParams {
  uint V;          // vocab (row length)
  uint n_groups;   // threadgroups per row
  uint chunk;      // elements per threadgroup (n_groups * chunk >= V)
  float inv_temp;  // argmax with gumbel / top-k lse: logits are scaled by inv_temp
  uint seed_lo, seed_hi;
  uint gumbel;     // argmax: add Gumbel(0,1) noise to logits*inv_temp
  uint k;          // top-k: k <= 64
};

inline uint arch_hash32(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
// Uniform (0,1) per (index, seed); Gumbel(0,1) = -log(-log(u)).
inline float arch_gumbel(uint i, uint s0, uint s1) {
  uint x = arch_hash32(i * 0x9E3779B9u ^ s0);
  x = arch_hash32(x ^ (s1 * 0x85EBCA6Bu + 0x632BE5ABu));
  float u = (float(x >> 8) + 0.5f) * (1.0f / 16777216.0f);
  return -log(-log(u));
}

// (value, index) argmax combine: larger value wins, ties -> lower index.
inline void arch_sg_argmax(thread float& v, thread uint& i) {
  float m = simd_max(v);
  uint c = simd_min(v == m ? i : 0xffffffffu);
  v = m; i = c;
}
inline void arch_tg_argmax(thread float& v, thread uint& i, threadgroup float* sv, threadgroup uint* si,
                           ushort sg, ushort lane, ushort nsg) {
  arch_sg_argmax(v, i);
  if (lane == 0) { sv[sg] = v; si[sg] = i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = lane < nsg ? sv[lane] : -FLT_MAX;
  i = lane < nsg ? si[lane] : 0xffffffffu;
  arch_sg_argmax(v, i);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Greedy / Gumbel argmax, single dispatch: grid (n_groups, rows) x T. part: [rows][n_groups] float2 (value, index
// bits); out_idx/out_val [rows]; counter [rows] (zero-initialized; reset by the kernel).
kernel void arch_argmax(device const float* logits [[buffer(0)]],
                        coherent(device) device float2* part [[buffer(1)]],
                        device uint* out_idx [[buffer(2)]],
                        device float* out_val [[buffer(3)]],
                        device atomic_uint* counter [[buffer(4)]],
                        constant ArchSampleParams& p [[buffer(5)]],
                        uint2 gp [[threadgroup_position_in_grid]],
                        uint2 tid2 [[thread_position_in_threadgroup]],
                        uint2 T2 [[threads_per_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup uint flag;
  const ushort tid = ushort(tid2.x), T = ushort(T2.x);
  const ushort nsg = T / 32;
  const uint g = gp.x, row = gp.y;
  device const float* l = logits + (ulong)row * p.V;
  uint b = g * p.chunk, e = min(p.V, b + p.chunk);
  float bv = -FLT_MAX;
  uint bi = 0xffffffffu;
  for (uint i = b + tid; i < e; i += T) {
    float v = p.gumbel ? l[i] * p.inv_temp + arch_gumbel(i, p.seed_lo + row * 0x2545F491u, p.seed_hi) : l[i];
    if (v > bv) { bv = v; bi = i; }
  }
  arch_tg_argmax(bv, bi, sv, si, sg, lane, nsg);
  if (tid == 0) part[row * p.n_groups + g] = float2(bv, as_type<float>(bi));
  if (!arch_last_group(counter + row, p.n_groups, tid, &flag)) return;
  bv = -FLT_MAX; bi = 0xffffffffu;
  for (uint k = tid; k < p.n_groups; k += T) {
    float2 q = part[row * p.n_groups + k];
    uint qi = as_type<uint>(q.y);
    if (q.x > bv || (q.x == bv && qi < bi)) { bv = q.x; bi = qi; }
  }
  arch_tg_argmax(bv, bi, sv, si, sg, lane, nsg);
  if (tid == 0) { out_idx[row] = bi; out_val[row] = bv; }
  arch_reset_counter(counter + row, tid);
}

// log-sum-exp combine of (max, sum) pairs
inline void arch_lse_combine(thread float& m, thread float& s, float m2, float s2) {
  float M = max(m, m2);
  s = s * exp(m - M) + s2 * exp(m2 - M);
  m = M;
}
inline void arch_sg_lse(thread float& m, thread float& s) {
  for (ushort o = 16; o > 0; o >>= 1) arch_lse_combine(m, s, simd_shuffle_xor(m, o), simd_shuffle_xor(s, o));
}
inline void arch_tg_lse(thread float& m, thread float& s, threadgroup float* sm, threadgroup float* ss, ushort sg,
                        ushort lane, ushort nsg) {
  arch_sg_lse(m, s);
  if (lane == 0) { sm[sg] = m; ss[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = lane < nsg ? sm[lane] : -FLT_MAX;
  s = lane < nsg ? ss[lane] : 0.0f;
  arch_sg_lse(m, s);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Extract the top-k (descending, ties -> lower index) of up to 4*T (value, index) pairs held 4 per thread.
template <typename PV, typename PI>
inline void arch_tg_topk_extract(thread float* v, thread uint* ix, uint k, threadgroup float* sv, threadgroup uint* si,
                                 ushort tid, ushort sg, ushort lane, ushort nsg, PV out_v, PI out_i) {
  for (uint r = 0; r < k; r++) {
    float bv = -FLT_MAX;
    uint bi = 0xffffffffu;
    for (uint q = 0; q < 4; q++)
      if (v[q] > bv || (v[q] == bv && ix[q] < bi)) { bv = v[q]; bi = ix[q]; }
    arch_tg_argmax(bv, bi, sv, si, sg, lane, nsg);
    for (uint q = 0; q < 4; q++)
      if (ix[q] == bi) { v[q] = -FLT_MAX; ix[q] = 0xffffffffu; }
    if (tid == 0) { out_v[r] = bv; out_i[r] = bi; }
  }
}

// Top-k (k <= 64) of each row + log-sum-exp of logits*inv_temp over the whole row, single dispatch.
// grid (n_groups, rows) x T with chunk <= 4*T and n_groups*64 <= 4*T. part_v/part_i: [rows][n_groups][64],
// part_ms: [rows][n_groups] float2; out_v/out_i: [rows][64] (raw logits, descending); out_lse: [rows].
kernel void arch_topk(device const float* logits [[buffer(0)]],
                      coherent(device) device float* part_v [[buffer(1)]],
                      coherent(device) device uint* part_i [[buffer(2)]],
                      coherent(device) device float2* part_ms [[buffer(3)]],
                      device float* out_v [[buffer(4)]],
                      device uint* out_i [[buffer(5)]],
                      device float* out_lse [[buffer(6)]],
                      device atomic_uint* counter [[buffer(7)]],
                      constant ArchSampleParams& p [[buffer(8)]],
                      uint2 gp [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]],
                      uint2 T2 [[threads_per_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float sv[32], sm2[32];
  threadgroup uint si[32];
  threadgroup uint flag;
  const ushort tid = ushort(tid2.x), T = ushort(T2.x);
  const ushort nsg = T / 32;
  const uint g = gp.x, row = gp.y;
  device const float* l = logits + (ulong)row * p.V;
  uint b = g * p.chunk, e = min(p.V, b + p.chunk);
  float v[4];
  uint ix[4];
  float m = -FLT_MAX, s = 0.0f;
  for (uint q = 0; q < 4; q++) {
    uint i = b + tid + q * T;
    bool ok = i < e;
    v[q] = ok ? l[i] : -FLT_MAX;
    ix[q] = ok ? i : 0xffffffffu;
    if (ok) arch_lse_combine(m, s, v[q] * p.inv_temp, 1.0f);
  }
  arch_tg_lse(m, s, sv, sm2, sg, lane, nsg);
  if (tid == 0) part_ms[row * p.n_groups + g] = float2(m, s);
  ulong pb = ((ulong)row * p.n_groups + g) * 64;
  arch_tg_topk_extract(v, ix, p.k, sv, si, tid, sg, lane, nsg, part_v + pb, part_i + pb);
  if (!arch_last_group(counter + row, p.n_groups, tid, &flag)) return;
  // merge: candidates n_groups * k
  const uint nc = p.n_groups * p.k;
  for (uint q = 0; q < 4; q++) {
    uint c = tid + q * T;
    bool ok = c < nc;
    uint gg = ok ? c / p.k : 0, rr = ok ? c - gg * p.k : 0;
    ulong o = ((ulong)row * p.n_groups + gg) * 64 + rr;
    v[q] = ok ? part_v[o] : -FLT_MAX;
    ix[q] = ok ? part_i[o] : 0xffffffffu;
  }
  m = -FLT_MAX; s = 0.0f;
  for (uint k = tid; k < p.n_groups; k += T) {
    float2 q = part_ms[row * p.n_groups + k];
    arch_lse_combine(m, s, q.x, q.y);
  }
  arch_tg_lse(m, s, sv, sm2, sg, lane, nsg);
  if (tid == 0) out_lse[row] = m + log(s);
  arch_tg_topk_extract(v, ix, p.k, sv, si, tid, sg, lane, nsg, out_v + row * 64, out_i + row * 64);
  arch_reset_counter(counter + row, tid);
}

// out[r][i] = in[r][i] - logsumexp(in[r]). grid: rows x T (one threadgroup per row). out may alias in.
kernel void arch_log_softmax(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& V [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             ushort tid [[thread_position_in_threadgroup]],
                             ushort T [[threads_per_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float sm[32], ss[32];
  const ushort nsg = T / 32;
  device const float* x = in + (ulong)row * V;
  float m = -FLT_MAX, s = 0.0f;
  for (uint i = tid; i < V; i += T) {
    float v = x[i];
    if (v > m) { s = s * exp(m - v) + 1.0f; m = v; }
    else s += exp(v - m);
  }
  arch_tg_lse(m, s, sm, ss, sg, lane, nsg);
  float lse = m + log(s);
  device float* y = out + (ulong)row * V;
  for (uint i = tid; i < V; i += T) y[i] = x[i] - lse;
}

struct ArchNormParams {
  uint n_rows;
  uint dim;
  float eps;
  uint has_weight;
  uint in_stride;    // floats between rows of x
  uint out_stride;   // floats between rows of y
};

// y = x * rsqrt(mean(x^2) + eps) * w. One threadgroup per row. y may alias x.
kernel void arch_rmsnorm(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant ArchNormParams& p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_position_in_threadgroup]],
                         ushort T [[threads_per_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device const float* xr = x + (ulong)row * p.in_stride;
  device float* yr = y + (ulong)row * p.out_stride;
  float ss = 0.0f;
  for (uint i = tid; i < p.dim; i += T) ss += xr[i] * xr[i];
  ss = arch_tg_sum(ss, scratch, sg, lane, T / 32);
  float r = rsqrt(ss / float(p.dim) + p.eps);
  for (uint i = tid; i < p.dim; i += T) yr[i] = xr[i] * r * (p.has_weight ? w[i] : 1.0f);
}

// Same, one simdgroup per row (small dims, e.g. 128/512): grid ceil(n_rows/8) x 256.
kernel void arch_rmsnorm_sg(device const float* x [[buffer(0)]],
                            device const float* w [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant ArchNormParams& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  uint row = tg * 8 + sg;
  if (row >= p.n_rows) return;
  device const float* xr = x + (ulong)row * p.in_stride;
  device float* yr = y + (ulong)row * p.out_stride;
  float ss = 0.0f;
  for (uint i = lane; i < p.dim; i += 32) ss += xr[i] * xr[i];
  ss = simd_sum(ss);
  float r = rsqrt(ss / float(p.dim) + p.eps);
  for (uint i = lane; i < p.dim; i += 32) yr[i] = xr[i] * r * (p.has_weight ? w[i] : 1.0f);
}

// Bandwidth reference for bench_arch: out[g] = sum of a contiguous slice (float4 loads, 8 per thread in flight).
kernel void arch_bench_read(device const float4* x [[buffer(0)]], device float* out [[buffer(1)]],
                            constant uint& n4 [[buffer(2)]], uint g [[threadgroup_position_in_grid]],
                            ushort tid [[thread_position_in_threadgroup]], ushort T [[threads_per_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  uint base = g * T * 8;
  float4 a = 0;
  for (uint k = 0; k < 8; k++) { uint i = base + k * T + tid; if (i < n4) a += x[i]; }
  float s = simd_sum(a.x + a.y + a.z + a.w);
  if (lane == 0 && s == 12345.678f) out[g * 32 + sg] = s;   // keep the loads alive
}
