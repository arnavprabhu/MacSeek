// Shared helpers for the arch_* kernels (model-specific small kernels: hyper-connections, router, engram, embedding,
// sampling, rmsnorm). See docs/modules/arch.md.
#pragma once
#include "common.h"
#include <metal_simdgroup_matrix>

constant constexpr uint ARCH_DIM = 5120;          // model dim
constant constexpr uint ARCH_HC = 4;              // hyper-connection copies
constant constexpr uint ARCH_HCD = 20480;         // ARCH_HC * ARCH_DIM
constant constexpr uint ARCH_MIX = 24;            // (2 + hc) * hc mix rows
constant constexpr uint ARCH_PART_STRIDE = 32;    // floats per threadgroup partial record

inline float arch_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Threadgroup-wide sum. scratch: >= 32 floats. All threads must call. Returns the total in every thread.
inline float arch_tg_sum(float v, threadgroup float* scratch, ushort sg, ushort lane, ushort nsg) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = lane < nsg ? scratch[lane] : 0.0f;
  r = simd_sum(r);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// Sinkhorn on a 4x4 matrix held by 16 lanes (lane & 15 = j*4 + k, j = row, k = column). Lanes 16..31 compute a mirror
// copy (shuffle_xor with 1,2,4,8 never crosses the 16-lane halves), so two matrices can be processed per simdgroup.
// Mirrors kernel.py hc_split_sinkhorn_kernel: row softmax + eps, col-normalize, then (iters-1) x (row, col) normalize,
// each normalization dividing by (sum + eps).
inline float arch_sinkhorn4(float c, int iters, float eps) {
  float m = max(c, simd_shuffle_xor(c, 1));
  m = max(m, simd_shuffle_xor(m, 2));
  float e = exp(c - m);
  float s = e + simd_shuffle_xor(e, 1);
  s += simd_shuffle_xor(s, 2);
  c = e / s + eps;
  float cs = c + simd_shuffle_xor(c, 4);
  cs += simd_shuffle_xor(cs, 8);
  c = c / (cs + eps);
  for (int it = 1; it < iters; it++) {
    float rs = c + simd_shuffle_xor(c, 1);
    rs += simd_shuffle_xor(rs, 2);
    c = c / (rs + eps);
    cs = c + simd_shuffle_xor(c, 4);
    cs += simd_shuffle_xor(cs, 8);
    c = c / (cs + eps);
  }
  return c;
}

// Computes the 24 hc mix outputs (pre[4], post[4], comb[4][4]) from raw projections. Called by all 32 lanes of one
// simdgroup; lanes 0..15 / 16..31 handle two independent tokens (raw0 / raw1 point to 24 raw dot products each, rstd0
// / rstd1 the stream rms scale; out0 / out1 the 24 outputs; pass the same pointers twice for one token).
template <typename PIn, typename POut>
inline void arch_hc_split(PIn raw, float rstd, POut out, device const float* scale, device const float* base,
                          ushort lane, int iters, float eps, bool write) {
  uint l = lane & 15;
  float m = raw[8 + l] * rstd * scale[2] + base[8 + l];
  float c = arch_sinkhorn4(m, iters, eps);
  if (!write) return;
  out[8 + l] = c;
  if (l < 4) out[l] = arch_sigmoid(raw[l] * rstd * scale[0] + base[l]) + eps;
  else if (l < 8) out[l] = 2.0f * arch_sigmoid(raw[l] * rstd * scale[1] + base[l]);
}

// Device-scope "last threadgroup done" ticket. Call after this threadgroup's device writes; returns true in every
// thread of the threadgroup that finished last (that threadgroup then sees all other groups' writes).
inline bool arch_last_group(device atomic_uint* counter, uint n_groups, ushort tid, threadgroup uint* flag) {
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint t = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    *flag = (t == n_groups - 1) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool last = *flag != 0;
  if (last) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return last;
}
inline void arch_reset_counter(device atomic_uint* counter, ushort tid) {
  if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}
