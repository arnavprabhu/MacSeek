#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_engram.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_engram.metal
// Engram block (GPU part) and row dequant kernels: IQ4_NL engram table rows, Q6_K token_embd rows.
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_common.h
// Shared helpers for the arch_* kernels (model-specific small kernels: hyper-connections, router, engram, embedding,
// sampling, rmsnorm). See docs/modules/arch.md.

// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/common.h
// Shared Metal helpers. Included by every .metal file.

#include <metal_stdlib>
using namespace metal;
#line 5 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_common.h"
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
#line 3 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_engram.metal"

constant constexpr uint ARCH_ENG_KV = 25600;   // engram_kv output: key [4][5120] then value [5120]

struct ArchEngramParams {
  uint n_tokens;
  float eps;       // norm_eps (1e-20)
  float clamp;     // 1e-6
  uint pad;
};

// gate_j = sigmoid(signed_sqrt(max(|dot_j|, clamp))), dot_j = sum_d h_j*w_j*key_j * rsqrt(mean h_j^2 + eps) *
// rsqrt(mean key_j^2 + eps) / sqrt(5120);  h_j += gate_j * value.  w = engram_gate [4][5120] (= q_weight*k_weight).
// h [n][4][5120] updated in place; kv [n][25600]; gate_out [n][4] (diagnostic). grid: n x 1024 threads.
kernel void arch_engram_gate(device float* h [[buffer(0)]],
                             device const float* kv [[buffer(1)]],
                             device const float* w [[buffer(2)]],
                             device float* gate_out [[buffer(3)]],
                             constant ArchEngramParams& p [[buffer(4)]],
                             uint t [[threadgroup_position_in_grid]],
                             ushort tid [[thread_position_in_threadgroup]],
                             ushort T [[threads_per_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32][12];
  threadgroup float gate[4];
  const ushort nsg = T / 32;
  device float* hp = h + (ulong)t * ARCH_HCD;
  device const float* key = kv + (ulong)t * ARCH_ENG_KV;
  device const float* val = key + ARCH_HCD;
  float hh[4] = {0, 0, 0, 0}, kk[4] = {0, 0, 0, 0}, hk[4] = {0, 0, 0, 0};
  for (uint d = tid; d < ARCH_DIM; d += T) {
    for (uint j = 0; j < 4; j++) {
      float a = hp[j * ARCH_DIM + d], b = key[j * ARCH_DIM + d];
      hh[j] += a * a;
      kk[j] += b * b;
      hk[j] += a * w[j * ARCH_DIM + d] * b;
    }
  }
  for (uint j = 0; j < 4; j++) {
    float a = simd_sum(hh[j]), b = simd_sum(kk[j]), c = simd_sum(hk[j]);
    if (lane == 0) { scratch[sg][j] = a; scratch[sg][4 + j] = b; scratch[sg][8 + j] = c; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4) {
    float a = 0, b = 0, c = 0;
    for (uint s = 0; s < nsg; s++) { a += scratch[s][tid]; b += scratch[s][4 + tid]; c += scratch[s][8 + tid]; }
    float rstd = rsqrt(a / float(ARCH_DIM) + p.eps) * rsqrt(b / float(ARCH_DIM) + p.eps);
    float dot = c * rstd * rsqrt(float(ARCH_DIM));
    float s = sqrt(max(fabs(dot), p.clamp));
    float gv = arch_sigmoid(dot < 0.0f ? -s : s);
    gate[tid] = gv;
    gate_out[t * 4 + tid] = gv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g0 = gate[0], g1 = gate[1], g2 = gate[2], g3 = gate[3];
  for (uint d = tid; d < ARCH_DIM; d += T) {
    float v = val[d];
    hp[d] += g0 * v;
    hp[ARCH_DIM + d] += g1 * v;
    hp[2 * ARCH_DIM + d] += g2 * v;
    hp[3 * ARCH_DIM + d] += g3 * v;
  }
}

constant float arch_kvalues_iq4nl[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                         1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};

// IQ4_NL rows -> F32. A row = 256 values = 8 blocks of {f16 d; u8 qs[16]} (144 B); value j (<16) of a block =
// d*kv[qs[j] & 15], value j+16 = d*kv[qs[j] >> 4]. src rows are row_stride bytes apart (even). dst [n_rows][256].
// grid: n_rows*128 threads.
kernel void arch_iq4nl_rows(device const uchar* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant uint2& nr_stride [[buffer(2)]],   // (n_rows, row_stride bytes)
                            uint i [[thread_position_in_grid]]) {
  uint row = i >> 7, rem = i & 127, b = rem >> 4, j = rem & 15;
  if (row >= nr_stride.x) return;
  device const uchar* blk = src + (ulong)row * nr_stride.y + b * 18;
  float d = float(*(device const half*)blk);
  uchar q = blk[2 + j];
  device float* o = dst + (ulong)row * 256 + b * 32;
  o[j] = d * arch_kvalues_iq4nl[q & 15];
  o[j + 16] = d * arch_kvalues_iq4nl[q >> 4];
}

struct ArchEmbedParams {
  uint n_tokens;
  uint row_stride;   // bytes between consecutive source rows (>= 4200, even)
  uint copies;       // output copies per token (4 = hc expand, 1 = plain)
  uint dim;          // 5120 (multiple of 256)
};

// Q6_K rows -> F32, replicated `copies` times: h[t][c][0..dim). Block (256 values, 210 B): ql[128], qh[64],
// int8 scales[16], f16 d. grid: n_tokens * (dim/256) * 64 threads (each thread produces 4 values).
kernel void arch_embed_q6k(device const uchar* rows [[buffer(0)]],
                           device float* h [[buffer(1)]],
                           constant ArchEmbedParams& p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
  const uint nb = p.dim / 256;
  uint t = i / (nb * 64);
  if (t >= p.n_tokens) return;
  uint rem = i - t * nb * 64, b = rem >> 6, r = rem & 63, hf = r >> 5, l = r & 31;
  device const uchar* blk = rows + (ulong)t * p.row_stride + b * 210;
  device const uchar* ql = blk + 64 * hf;
  device const uchar* qh = blk + 128 + 32 * hf;
  device const char* sc = (device const char*)(blk + 192) + 8 * hf;
  float d = float(*(device const half*)(blk + 208));
  uint is = l / 16;
  int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
  int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
  int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
  int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
  float y1 = d * float(sc[is + 0]) * float(q1);
  float y2 = d * float(sc[is + 2]) * float(q2);
  float y3 = d * float(sc[is + 4]) * float(q3);
  float y4 = d * float(sc[is + 6]) * float(q4);
  uint yb = b * 256 + 128 * hf + l;
  device float* o = h + (ulong)t * p.copies * p.dim;
  for (uint c = 0; c < p.copies; c++) {
    device float* oc = o + c * p.dim + yb;
    oc[0] = y1; oc[32] = y2; oc[64] = y3; oc[96] = y4;
  }
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_hc.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_hc.metal
// Hyper-connection kernels: hc_post (expand), hc_mixes (rms + F32 20480->24 projection + Sinkhorn), hc_pre (collapse)
// fused with the following RMSNorm. Layout: h [n][4][5120] F32, mix [n][24] = pre[4], post[4], comb[4][4] (comb[j][k]
// at 8 + 4j + k), fn [24][20480] F32 (GGUF hc_*_fn as stored), x [n][5120].
//
// One "step" at a sublayer boundary (see SPEC 3):
//   h_new = has_post ? post[i]*x_sub + sum_j comb[j][i]*h[j] : h          (mix_prev supplies post/comb)
//   mix_out = has_mixes ? hc_split_sinkhorn(fn . h_new * rsqrt(mean(h_new^2) + norm_eps)) : -
//   x_out = rmsnorm(sum_j pre_lag[j] * h_new[j]) * norm_w                    (pre_lag = mix_prev[0:4], lagged mixes)
#line 10 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_hc.metal"

struct ArchHcParams {
  uint n_groups;    // decode: threadgroups in the step grid (5120 / threads_per_group)
  uint has_post;
  uint has_mixes;
  uint n_tokens;    // prefill
  float norm_eps;
  float hc_eps;
  int iters;
  uint fused;       // decode: 1 = last threadgroup runs the tail (single dispatch), 0 = separate finalize dispatch
  uint has_pre;     // 0: skip hc_pre + RMSNorm (x_out untouched), e.g. the post-only step before engram
  uint safe_inplace;  // decode step: write h_out only after every simdgroup has read h_in (h_out == h_in race fix)
  uint row_splits;    // decode step: threadgroups per dim chunk, each computing 24/row_splits mix rows (h_out != h_in)
};

// ---- tail shared by the fused step and the finalize kernel (one threadgroup) ----
template <typename PPart, typename PX>
inline void arch_hc_tail(PPart part, PX x_out, device const float* norm_w,
                         device float* mix_out, device const float* scale, device const float* base,
                         constant ArchHcParams& p, ushort tid, ushort T, ushort sg, ushort lane,
                         threadgroup float* red) {
  const ushort nsg = T / 32;
  for (uint v = sg; v < 26; v += nsg) {   // fixed-order parallel reduction of the per-group partials
    float s = 0.0f;
    for (uint k = lane; k < p.n_groups; k += 32) s += part[k * ARCH_PART_STRIDE + v];
    s = simd_sum(s);
    if (lane == 0) red[v] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rstd_h = rsqrt(red[24] / float(ARCH_HCD) + p.norm_eps);
  float rstd_x = rsqrt(red[25] / float(ARCH_DIM) + p.norm_eps);
  if (p.has_mixes && sg == 0) arch_hc_split(red, rstd_h, mix_out, scale, base, lane, p.iters, p.hc_eps, lane < 16);
  if (p.has_pre)
    for (uint d = tid; d < ARCH_DIM; d += T) x_out[d] = x_out[d] * rstd_x * norm_w[d];
}

// Decode step: grid = 5120 / T threadgroups (T in {128, 256, 512, 1024}); group g owns dims [g*T, g*T + T).
// Every simdgroup recomputes h_new for the whole chunk in registers (cheap) so its fn loads have no barrier in
// front of them; simdgroup 0 also writes h_out / x_out and the sums. Buffers must be 16-byte aligned.
kernel void arch_hc_step(device const float* h_in [[buffer(0)]],
                         device float* h_out [[buffer(1)]],
                         device const float* x_sub [[buffer(2)]],
                         device const float* mix_prev [[buffer(3)]],
                         device const float* fn [[buffer(4)]],
                         device const float* scale [[buffer(5)]],
                         device const float* base [[buffer(6)]],
                         device const float* norm_w [[buffer(7)]],
                         coherent(device) device float* x_out [[buffer(8)]],
                         device float* mix_out [[buffer(9)]],
                         coherent(device) device float* part [[buffer(10)]],
                         device atomic_uint* counter [[buffer(11)]],
                         constant ArchHcParams& p [[buffer(12)]],
                         uint tgi [[threadgroup_position_in_grid]],
                         ushort tid [[thread_position_in_threadgroup]],
                         ushort T [[threads_per_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  threadgroup uint flag;
  const ushort nsg = T / 32;
  // row_splits > 1: threadgroup tgi = dim chunk g, split sp; split sp computes mix rows [r0, r1) of the chunk (each row's
  // partial is the same arithmetic as with one threadgroup per chunk); split 0 also does h_out / x_out / the sums.
  const uint S = max(1u, p.row_splits), g = tgi / S, sp = tgi - g * S;
  const uint r0 = sp * ARCH_MIX / S, r1 = (sp + 1) * ARCH_MIX / S;
  const bool main_split = sp == 0;
  const uint nv = T / 4;                 // float4 per copy in this chunk
  const uint d0v = g * nv;               // chunk start in float4 units
  float pre[4], post[4], comb[16];
  for (uint i = 0; i < 4; i++) { pre[i] = mix_prev[i]; post[i] = mix_prev[4 + i]; }
  for (uint i = 0; i < 16; i++) comb[i] = mix_prev[8 + i];
  device const float4* h4 = (device const float4*)h_in;
  device const float4* x4 = (device const float4*)x_sub;
  device const float4* f4 = (device const float4*)fn;
  constexpr uint MAXQ = 6;               // rows per simdgroup <= 24 / 4
  float acc[MAXQ] = {0, 0, 0, 0, 0, 0};
  float ssh = 0.0f, ssx = 0.0f;
  for (uint c = lane; c < nv; c += 32) {
    const uint dv = d0v + c;
    float4 hn[4];
    for (uint j = 0; j < 4; j++) hn[j] = h4[j * (ARCH_DIM / 4) + dv];
    if (p.has_post) {
      float4 xs = x4[dv];
      float4 r[4];
      for (uint i = 0; i < 4; i++) {
        float4 a = post[i] * xs;
        for (uint j = 0; j < 4; j++) a += comb[4 * j + i] * hn[j];
        r[i] = a;
      }
      for (uint i = 0; i < 4; i++) hn[i] = r[i];
    }
    if (p.has_mixes) {
      for (uint q = 0; q < MAXQ; q++) {
        uint r = r0 + sg + q * nsg;
        if (r >= r1) break;
        float a = 0.0f;
        for (uint j = 0; j < 4; j++) a += dot(f4[r * (ARCH_HCD / 4) + j * (ARCH_DIM / 4) + dv], hn[j]);
        acc[q] += a;
      }
    }
    if (sg == 0 && main_split) {
      if (p.has_post && !p.safe_inplace)
        for (uint i = 0; i < 4; i++) ((device float4*)h_out)[i * (ARCH_DIM / 4) + dv] = hn[i];
      float4 xp = pre[0] * hn[0] + pre[1] * hn[1] + pre[2] * hn[2] + pre[3] * hn[3];
      if (p.has_pre) ((coherent(device) device float4*)x_out)[dv] = xp;
      ssh += dot(hn[0], hn[0]) + dot(hn[1], hn[1]) + dot(hn[2], hn[2]) + dot(hn[3], hn[3]);
      ssx += dot(xp, xp);
    }
  }
  // h_out usually aliases h_in (decode updates h in place) and every simdgroup reads all of h_in above: simdgroup 0
  // must not overwrite an element before the others have read it. Safe mode: wait for all simdgroups, then simdgroup 0
  // recomputes h_new (same arithmetic -> same values) and stores it. (Unsafe legacy mode: MACSEEK_ARCH_HC_SAFE=0.)
  if (p.has_post && p.safe_inplace) {
    threadgroup_barrier(mem_flags::mem_none);
    if (sg == 0 && main_split)
      for (uint c = lane; c < nv; c += 32) {
        const uint dv = d0v + c;
        float4 hn[4];
        for (uint j = 0; j < 4; j++) hn[j] = h4[j * (ARCH_DIM / 4) + dv];
        float4 xs = x4[dv];
        for (uint i = 0; i < 4; i++) {
          float4 a = post[i] * xs;
          for (uint j = 0; j < 4; j++) a += comb[4 * j + i] * hn[j];
          ((device float4*)h_out)[i * (ARCH_DIM / 4) + dv] = a;
        }
      }
  }
  if (p.has_mixes) {
    for (uint q = 0; q < MAXQ; q++) {
      uint r = r0 + sg + q * nsg;
      if (r >= r1) break;
      float v = simd_sum(acc[q]);
      if (lane == 0) part[g * ARCH_PART_STRIDE + r] = v;
    }
  }
  if (sg == 0 && main_split) {
    ssh = simd_sum(ssh);
    ssx = simd_sum(ssx);
    if (lane == 0) { part[g * ARCH_PART_STRIDE + 24] = ssh; part[g * ARCH_PART_STRIDE + 25] = ssx; }
  }
  if (!p.fused) return;
  if (!arch_last_group(counter, p.n_groups * S, tid, &flag)) return;
  arch_hc_tail(part, x_out, norm_w, mix_out, scale, base, p, tid, T, sg, lane, red);
  arch_reset_counter(counter, tid);
}

// Split-mode tail: one threadgroup (any size >= 32), after arch_hc_step with fused = 0 and a barrier.
kernel void arch_hc_step_finalize(device const float* part [[buffer(0)]],
                                  device float* x_out [[buffer(1)]],
                                  device const float* norm_w [[buffer(2)]],
                                  device float* mix_out [[buffer(3)]],
                                  device const float* scale [[buffer(4)]],
                                  device const float* base [[buffer(5)]],
                                  constant ArchHcParams& p [[buffer(6)]],
                                  ushort tid [[thread_position_in_threadgroup]],
                                  ushort T [[threads_per_threadgroup]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  arch_hc_tail(part, x_out, norm_w, mix_out, scale, base, p, tid, T, sg, lane, red);
}

// Prefill step part 1: one threadgroup per token. hc_post (optional) + hc_pre + RMSNorm; writes rstd_h[t] =
// rsqrt(mean(h_new^2) + eps) for the mixes kernel. h_out may alias h_in.
kernel void arch_hc_post_pre_n(device const float* h_in [[buffer(0)]],
                               device float* h_out [[buffer(1)]],
                               device const float* x_sub [[buffer(2)]],
                               device const float* mix_prev [[buffer(3)]],
                               device const float* norm_w [[buffer(4)]],
                               device float* x_out [[buffer(5)]],
                               device float* rstd_h [[buffer(6)]],
                               constant ArchHcParams& p [[buffer(7)]],
                               uint t [[threadgroup_position_in_grid]],
                               ushort tid [[thread_position_in_threadgroup]],
                               ushort T [[threads_per_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const ushort nsg = T / 32;
  device const float* hi = h_in + (ulong)t * ARCH_HCD;
  device float* ho = h_out + (ulong)t * ARCH_HCD;
  device const float* mp = mix_prev + (ulong)t * ARCH_MIX;
  device float* xo = x_out + (ulong)t * ARCH_DIM;
  float pre[4], post[4], comb[16];
  for (uint i = 0; i < 4; i++) { pre[i] = mp[i]; post[i] = mp[4 + i]; }
  for (uint i = 0; i < 16; i++) comb[i] = mp[8 + i];
  float ssh = 0.0f, ssx = 0.0f;
  for (uint d = tid; d < ARCH_DIM; d += T) {
    float hn[4];
    for (uint j = 0; j < 4; j++) hn[j] = hi[j * ARCH_DIM + d];
    if (p.has_post) {
      float xs = x_sub[(ulong)t * ARCH_DIM + d];
      float r[4];
      for (uint i = 0; i < 4; i++) {
        float acc = post[i] * xs;
        for (uint j = 0; j < 4; j++) acc += comb[4 * j + i] * hn[j];
        r[i] = acc;
      }
      for (uint i = 0; i < 4; i++) { hn[i] = r[i]; ho[i * ARCH_DIM + d] = r[i]; }
    }
    float xp = pre[0] * hn[0] + pre[1] * hn[1] + pre[2] * hn[2] + pre[3] * hn[3];
    if (p.has_pre) xo[d] = xp;
    ssh += hn[0] * hn[0] + hn[1] * hn[1] + hn[2] * hn[2] + hn[3] * hn[3];
    ssx += xp * xp;
  }
  ssh = arch_tg_sum(ssh, scratch, sg, lane, nsg);
  ssx = arch_tg_sum(ssx, scratch, sg, lane, nsg);
  float rx = rsqrt(ssx / float(ARCH_DIM) + p.norm_eps);
  if (tid == 0) rstd_h[t] = rsqrt(ssh / float(ARCH_HCD) + p.norm_eps);
  if (p.has_pre)
    for (uint d = tid; d < ARCH_DIM; d += T) xo[d] = xo[d] * rx * norm_w[d];
}

// Prefill mixes, simdgroup-matrix GEMM [n x 20480] . fn^T [20480 x 24] with the Sinkhorn epilogue.
// One threadgroup = TOK = 8*NA tokens x 24 rows, KS simdgroups split K (reduced through threadgroup memory).
// Requires n >= TOK: the last tile is shifted back to [n-TOK, n) and only writes its new tokens.
template <uint NA, uint KS>
inline void arch_hc_mixes_mma_body(device const float* h, device const float* fn, device const float* scale,
                                   device const float* base, device const float* rstd_h, device float* mix_out,
                                   constant ArchHcParams& p, uint tile, ushort sg, ushort lane, threadgroup float* tg) {
  constexpr uint TOK = 8 * NA;
  const uint n = p.n_tokens;
  const uint wfrom = tile * TOK;
  const uint t0 = (wfrom + TOK > n) ? n - TOK : wfrom;
  simdgroup_float8x8 acc[NA][3];
  for (uint a = 0; a < NA; a++)
    for (uint r = 0; r < 3; r++) acc[a][r] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const uint kb = sg * (ARCH_HCD / KS), ke = kb + ARCH_HCD / KS;
  device const float* hp = h + (ulong)t0 * ARCH_HCD;
  for (uint k = kb; k < ke; k += 8) {
    simdgroup_float8x8 b0, b1, b2;
    simdgroup_load(b0, fn + 0 * 8 * ARCH_HCD + k, ARCH_HCD, ulong2(0, 0), true);
    simdgroup_load(b1, fn + 1 * 8 * ARCH_HCD + k, ARCH_HCD, ulong2(0, 0), true);
    simdgroup_load(b2, fn + 2 * 8 * ARCH_HCD + k, ARCH_HCD, ulong2(0, 0), true);
    for (uint a = 0; a < NA; a++) {
      simdgroup_float8x8 am;
      simdgroup_load(am, hp + (ulong)a * 8 * ARCH_HCD + k, ARCH_HCD);
      simdgroup_multiply_accumulate(acc[a][0], am, b0, acc[a][0]);
      simdgroup_multiply_accumulate(acc[a][1], am, b1, acc[a][1]);
      simdgroup_multiply_accumulate(acc[a][2], am, b2, acc[a][2]);
    }
  }
  threadgroup float* my = tg + sg * TOK * 24;
  for (uint a = 0; a < NA; a++)
    for (uint r = 0; r < 3; r++) simdgroup_store(acc[a][r], my + a * 8 * 24 + r * 8, 24);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg * 32 + lane; i < TOK * 24; i += KS * 32) {
    float v = tg[i];
    for (uint q = 1; q < KS; q++) v += tg[q * TOK * 24 + i];
    tg[i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint pair = sg; pair < TOK / 2; pair += KS) {
    const uint lt = pair * 2 + (lane >> 4);
    const uint t = t0 + lt;
    arch_hc_split(tg + lt * 24, rstd_h[t], mix_out + (ulong)t * ARCH_MIX, scale, base, lane, p.iters, p.hc_eps,
                  t >= wfrom);
  }
}

// 8 tokens / threadgroup, 4 simdgroups (128 threads). n >= 8.
kernel void arch_hc_mixes_mma(device const float* h [[buffer(0)]], device const float* fn [[buffer(1)]],
                              device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                              device const float* rstd_h [[buffer(4)]], device float* mix_out [[buffer(5)]],
                              constant ArchHcParams& p [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float tg[4 * 8 * 24];
  arch_hc_mixes_mma_body<1, 4>(h, fn, scale, base, rstd_h, mix_out, p, tile, sg, lane, tg);
}

// 32 tokens / threadgroup, 8 simdgroups (256 threads). n >= 32.
kernel void arch_hc_mixes_mma32(device const float* h [[buffer(0)]], device const float* fn [[buffer(1)]],
                                device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                                device const float* rstd_h [[buffer(4)]], device float* mix_out [[buffer(5)]],
                                constant ArchHcParams& p [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
                                ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float tg[8 * 32 * 24];
  arch_hc_mixes_mma_body<4, 8>(h, fn, scale, base, rstd_h, mix_out, p, tile, sg, lane, tg);
}

// Mixes, one threadgroup (256 threads) per token: any n (fallback / small n / decode split path).
kernel void arch_hc_mixes_rows(device const float* h [[buffer(0)]],
                               device const float* fn [[buffer(1)]],
                               device const float* scale [[buffer(2)]],
                               device const float* base [[buffer(3)]],
                               device const float* rstd_h [[buffer(4)]],
                               device float* mix_out [[buffer(5)]],
                               constant ArchHcParams& p [[buffer(6)]],
                               uint t [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort T [[threads_per_threadgroup]]) {
  threadgroup float raw[24];
  const ushort nsg = T / 32;
  device const float4* hv = (device const float4*)(h + (ulong)t * ARCH_HCD);
  for (uint r = sg; r < ARCH_MIX; r += nsg) {
    device const float4* fr = (device const float4*)(fn + r * ARCH_HCD);
    float acc = 0.0f;
    for (uint c = lane; c < ARCH_HCD / 4; c += 32) acc += dot(fr[c], hv[c]);
    acc = simd_sum(acc);
    if (lane == 0) raw[r] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) arch_hc_split(raw, rstd_h[t], mix_out + (ulong)t * ARCH_MIX, scale, base, lane, p.iters, p.hc_eps,
                             lane < 16);
}

// dspark helper: mean over the hc copies. out[t][d] = mean_j h[t][j][d]. grid: n*5120 threads.
kernel void arch_hc_mean(device const float* h [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n * ARCH_DIM) return;
  uint t = i / ARCH_DIM, d = i - t * ARCH_DIM;
  device const float* hp = h + (ulong)t * ARCH_HCD + d;
  out[i] = 0.25f * (hp[0] + hp[ARCH_DIM] + hp[2 * ARCH_DIM] + hp[3 * ARCH_DIM]);
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_router.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_router.metal
// MoE router: scores = sqrt(softplus(logit)); top-6 of (scores + bias); weights = score / (sum + 1e-20) * route_scale.
// Plus prefill routing lists (per-expert token lists for grouped GEMM), deterministic (no atomics).
#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_router.metal"

constant constexpr uint ARCH_NEXP = 384;
constant constexpr uint ARCH_TOPK = 6;
constant constexpr uint ARCH_EPL = ARCH_NEXP / 32;   // experts per lane

struct ArchRouterParams {
  uint n_tokens;
  uint n_groups;      // fused decode: threadgroups
  float route_scale;  // 1.5
  uint pad;
};

inline float arch_log1p_pos(float y) {   // y >= 0
  if (y < 1e-2f) return y * (1.0f - y * (0.5f - y * (1.0f / 3.0f - 0.25f * y)));
  return log(1.0f + y);
}
// torch.nn.functional.softplus (beta 1, threshold 20) followed by sqrt
inline float arch_sqrt_softplus(float x) {
  float sp = x > 20.0f ? x : (x > 0.0f ? x + arch_log1p_pos(exp(-x)) : arch_log1p_pos(exp(x)));
  return sqrt(sp);
}

// One simdgroup selects the top-6 experts of one token. Ties: lowest expert id. Writes ids/weights (descending key).
template <typename PL>
inline void arch_topk6(PL logits, device const float* bias, float route_scale, ushort lane,
                       device int* ids, device float* wts) {
  float sc[ARCH_EPL], key[ARCH_EPL];
  for (uint i = 0; i < ARCH_EPL; i++) {
    uint e = lane + 32 * i;
    sc[i] = arch_sqrt_softplus(logits[e]);
    key[i] = sc[i] + bias[e];
  }
  int sel_e[ARCH_TOPK];
  float sel_w[ARCH_TOPK];
  float sum = 0.0f;
  for (uint k = 0; k < ARCH_TOPK; k++) {
    float bk = -FLT_MAX, bs = 0.0f;
    int be = 0x7fffffff;
    for (uint i = 0; i < ARCH_EPL; i++)
      if (key[i] > bk) { bk = key[i]; bs = sc[i]; be = int(lane + 32 * i); }
    float m = simd_max(bk);
    int win = simd_min(bk == m ? be : 0x7fffffff);
    for (uint i = 0; i < ARCH_EPL; i++)
      if (int(lane + 32 * i) == win) key[i] = -FLT_MAX;
    float ws = simd_shuffle(bs, ushort(win & 31));
    sel_e[k] = win;
    sel_w[k] = ws;
    sum += ws;
  }
  float inv = route_scale / (sum + 1e-20f);
  if (lane < ARCH_TOPK) {
    // registers are indexed with a uniform loop to avoid dynamic indexing
    for (uint k = 0; k < ARCH_TOPK; k++)
      if (lane == k) { ids[k] = sel_e[k]; wts[k] = sel_w[k] * inv; }
  }
}

// logits [n][384] -> ids [n][6] int32, weights [n][6]. grid: ceil(n/4) x 128 threads (one simdgroup per token).
kernel void arch_router_topk(device const float* logits [[buffer(0)]],
                             device const float* bias [[buffer(1)]],
                             device int* ids [[buffer(2)]],
                             device float* wts [[buffer(3)]],
                             constant ArchRouterParams& p [[buffer(4)]],
                             uint tg [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  uint t = tg * 4 + sg;
  if (t >= p.n_tokens) return;
  arch_topk6(logits + (ulong)t * ARCH_NEXP, bias, p.route_scale, lane, ids + t * ARCH_TOPK, wts + t * ARCH_TOPK);
}

// Decode: fused F32 matvec x[5120] . W[384][5120] + top-6 in one dispatch (last threadgroup selects).
// grid: n_groups threadgroups (384 % (n_groups) == 0), any T multiple of 32. Also writes logits [384].
kernel void arch_router_fused(device const float* x [[buffer(0)]],
                              device const float* W [[buffer(1)]],
                              device const float* bias [[buffer(2)]],
                              coherent(device) device float* logits [[buffer(3)]],
                              device int* ids [[buffer(4)]],
                              device float* wts [[buffer(5)]],
                              device atomic_uint* counter [[buffer(6)]],
                              constant ArchRouterParams& p [[buffer(7)]],
                              uint g [[threadgroup_position_in_grid]],
                              ushort tid [[thread_position_in_threadgroup]],
                              ushort T [[threads_per_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup uint flag;
  const ushort nsg = T / 32;
  const uint rows = ARCH_NEXP / p.n_groups;
  device const float4* xv = (device const float4*)x;
  for (uint r = sg; r < rows; r += nsg) {
    uint e = g * rows + r;
    device const float4* wr = (device const float4*)(W + (ulong)e * ARCH_DIM);
    float acc = 0.0f;
    for (uint c = lane; c < ARCH_DIM / 4; c += 32) acc += dot(wr[c], xv[c]);
    acc = simd_sum(acc);
    if (lane == 0) logits[e] = acc;
  }
  if (!arch_last_group(counter, p.n_groups, tid, &flag)) return;
  if (sg == 0) arch_topk6(logits, bias, p.route_scale, lane, ids, wts);
  arch_reset_counter(counter, tid);
}

// Prefill routing lists, pass 1: counts[e] = #entries (t,k) with ids[t*6+k] == e. grid: 48 x 256 (simdgroup/expert).
kernel void arch_route_count(device const int* ids [[buffer(0)]],
                             device uint* counts [[buffer(1)]],
                             constant ArchRouterParams& p [[buffer(2)]],
                             uint tg [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const int e = int(tg * 8 + sg);
  const uint N = p.n_tokens * ARCH_TOPK;
  uint c = 0;
  for (uint i = lane; i < N; i += 32) c += (ids[i] == e) ? 1u : 0u;
  c = simd_sum(c);
  if (lane == 0) counts[e] = c;
}

// Pass 2: offsets[e] (exclusive scan, offsets[384] = total) and stable per-expert lists ordered by entry index:
// list_entry[pos] = t*6+k, list_tok[pos] = t, list_w[pos] = weight, slot_of[t*6+k] = pos. grid: 48 x 256.
kernel void arch_route_scatter(device const int* ids [[buffer(0)]],
                               device const float* wts [[buffer(1)]],
                               device const uint* counts [[buffer(2)]],
                               device uint* offsets [[buffer(3)]],
                               device uint* list_entry [[buffer(4)]],
                               device uint* list_tok [[buffer(5)]],
                               device float* list_w [[buffer(6)]],
                               device uint* slot_of [[buffer(7)]],
                               constant ArchRouterParams& p [[buffer(8)]],
                               uint tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const int e = int(tg * 8 + sg);
  const uint N = p.n_tokens * ARCH_TOPK;
  uint off = 0;
  for (uint i = lane; i < uint(e); i += 32) off += counts[i];
  off = simd_sum(off);
  if (lane == 0) {
    offsets[e] = off;
    if (e == int(ARCH_NEXP) - 1) offsets[ARCH_NEXP] = off + counts[e];
  }
  uint basepos = off;
  const ulong below = (1ul << lane) - 1ul;
  for (uint i0 = 0; i0 < N; i0 += 32) {
    uint i = i0 + lane;
    bool m = i < N && ids[i] == e;
    ulong bits = ulong(simd_vote::vote_t(simd_ballot(m)));
    if (m) {
      uint pos = basepos + popcount(bits & below);
      list_entry[pos] = i;
      list_tok[pos] = i / ARCH_TOPK;
      list_w[pos] = wts[i];
      slot_of[i] = pos;
    }
    basepos += popcount(bits);
  }
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_sample.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_sample.metal
// Sampling helpers over the 129280 vocab: argmax (greedy, optional Gumbel noise = exact temperature sampling),
// top-k (k <= 64) with the full-vocab log-sum-exp (for CPU top-p), log_softmax (quality tool), rmsnorm variants.
#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/arch_sample.metal"

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
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_index.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_index.metal
// Attention module: indexer scoring (decode: simdgroup MMA split over many threadgroups; prefill: Metal 4 tensor-op
// matmul2d tiles), exact top-k selection (radix select in one threadgroup), candidate blocks.
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_common.h
// Shared helpers for the attention module (attn_*.metal). See docs/modules/attn.md.

#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_common.h"

// ---- model constants ----
constant constexpr int ATTN_D = 512;        // head dim == latent dim
constant constexpr int ATTN_RD = 64;        // rope dims (last 64)
constant constexpr int ATTN_H = 64;         // attention heads
constant constexpr int ATTN_W = 128;        // window ring
constant constexpr int ATTN_IH = 32;        // indexer heads
constant constexpr int ATTN_ID = 128;       // indexer head dim

// ---- simdgroup 8x8 fragment layout (verified by bench/attn_layout_probe): lane holds (row fm, cols fn, fn+1) ----
inline ushort attn_frag_row(ushort lane) { return ((lane >> 2) & 4) + ((lane >> 1) & 3); }
inline ushort attn_frag_col(ushort lane) { return ((lane >> 2) & 2) * 2 + (lane & 1) * 2; }

// ---- exact float formats used by the reference quant simulation (kernel.py act_quant / fp4_act_quant) ----
inline float attn_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }  // e in [-126, 127]
inline int attn_floor_log2(float a) { return int((as_type<uint>(a) >> 23) & 0xFF) - 127; }  // a >= 0 normal
// ceil(log2(x)) by bit manipulation (kernel.py fast_log2_ceil)
inline int attn_log2_ceil(float x) {
  uint b = as_type<uint>(x);
  return int((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
}
// round |x| <= 448 to the nearest e4m3fn value (ties to even), incl. subnormals (step 2^-9)
inline float attn_round_e4m3(float x) {
  float a = fabs(x);
  int e = max(attn_floor_log2(a), -6);
  float r = rint(a * attn_pow2(3 - e)) * attn_pow2(e - 3);
  return copysign(r, x);
}
// round |x| <= 6 to the nearest e2m1 value {0,.5,1,1.5,2,3,4,6} (ties to even)
inline float attn_round_e2m1(float x) {
  float a = fabs(x);
  int e = max(attn_floor_log2(a), 0);
  float r = rint(a * attn_pow2(1 - e)) * attn_pow2(e - 1);
  return copysign(r, x);
}
// block scale for fp8 act_quant with ue8m0 rounding: 2^ceil(log2(max(amax,1e-4)/448))
inline float attn_fp8_scale(float amax) { return attn_pow2(attn_log2_ceil(max(amax, 1e-4f) * (1.0f / 448.0f))); }
inline float attn_fp8_q(float x, float s, float inv_s) { return attn_round_e4m3(clamp(x * inv_s, -448.0f, 448.0f)) * s; }
// fp4 with ue8m0 scale (indexer q / keys, block 32)
inline float attn_fp4e8_scale(float amax) { return attn_pow2(attn_log2_ceil(max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f))); }
inline float attn_fp4e8_q(float x, float s, float inv_s) { return attn_round_e2m1(clamp(x * inv_s, -6.0f, 6.0f)) * s; }
// fp4 with e4m3 scale (compressed KV, block 16)
inline float attn_fp4e4_scale(float amax) { return attn_round_e4m3(min(precise::divide(max(amax, 6.0f * 0x1p-9f), 6.0f), 448.0f)); }
inline float attn_fp4e4_q(float x, float s) { return attn_round_e2m1(clamp(precise::divide(x, s), -6.0f, 6.0f)) * s; }

// ---- order-preserving float key: larger float -> larger uint. 0 is reserved for "invalid" (-inf / masked). ----
inline uint attn_fkey(float f) {
  uint u = as_type<uint>(f);
  uint k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (f == -INFINITY || isnan(f)) ? 0u : k;
}
inline float attn_unkey(uint k) {
  uint u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return as_type<float>(u);
}

// rope on one interleaved pair with table entry cs = (cos, sin)
inline float2 attn_rot(float2 x, float2 cs) { return float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x); }
inline float2 attn_rot_inv(float2 x, float2 cs) { return float2(x.x * cs.x + x.y * cs.y, -x.x * cs.y + x.y * cs.x); }
#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_index.metal"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ------------------------------------------------------------------------------------------------------------------
// threadgroup helpers (1024 threads = 32 simdgroups unless noted)
// exclusive prefix sum over the threadgroup; tmp needs 33 uints. Contains barriers: call from uniform control flow.
inline uint attn_scan_excl(uint v, threadgroup uint* tmp, ushort sg, ushort ln, ushort nsg, thread uint& total) {
  uint pre = simd_prefix_exclusive_sum(v);
  if (ln == 31) tmp[sg] = pre + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint x = ln < nsg ? tmp[ln] : 0u;
    uint px = simd_prefix_exclusive_sum(x);
    uint tot = simd_sum(x);
    tmp[ln] = px;
    if (ln == 0) tmp[32] = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = tmp[32];
  uint r = pre + tmp[sg];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// Exact top-k (largest keys; key 0 = invalid) over N elements read through `src`. Result (unordered) in
// out_key/out_idx[0..*out_cnt). Ties at the threshold resolve to the lowest element index (deterministic).
// TG memory: hist[2048], tmp[40]. Must be called by all 1024 threads.
template <typename Src>
inline void attn_select(thread const Src& src, int N, int k, threadgroup uint* hist, threadgroup uint* tmp,
                        threadgroup uint* out_key, threadgroup int* out_idx, threadgroup atomic_int* out_cnt,
                        ushort tid, ushort sg, ushort ln) {
  constexpr int NT = 1024;
  uint prefix = 0, pmask = 0;
  int rem = k;
  bool take_all = false;
  for (int ps = 0; ps < 3; ps++) {
    const int sh = ps == 0 ? 21 : (ps == 1 ? 10 : 0);
    const int nb = ps == 2 ? 1024 : 2048;
    for (int b = tid; b < 2048; b += NT) hist[b] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < N; i += NT) {
      uint kk = src.key(i);
      if (kk != 0u && (kk & pmask) == prefix)
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(kk >> sh) & uint(nb - 1)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int bpt = nb / NT;  // bins per thread, thread t owns the bpt bins just below nb - t*bpt (descending)
    uint s = 0;
    for (int j = 0; j < bpt; j++) s += hist[nb - 1 - (tid * bpt + j)];
    uint total;
    uint above = attn_scan_excl(s, tmp, sg, ln, NT / 32, total);
    if (ps == 0 && total <= uint(k)) { take_all = true; break; }
    if (above < uint(rem) && uint(rem) <= above + s) {
      uint acc = above;
      for (int j = 0; j < bpt; j++) {
        int b = nb - 1 - (tid * bpt + j);
        uint c = hist[b];
        if (acc + c >= uint(rem)) { tmp[34] = uint(b); tmp[35] = uint(rem) - acc; break; }
        acc += c;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= tmp[34] << sh;
    pmask |= uint(nb - 1) << sh;
    rem = int(tmp[35]);
  }
  if (tid == 0) atomic_store_explicit(out_cnt, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (take_all) {
    for (int i = tid; i < N; i += NT) {
      uint kk = src.key(i);
      if (kk != 0u) { int sl = atomic_fetch_add_explicit(out_cnt, 1, memory_order_relaxed); out_key[sl] = kk; out_idx[sl] = src.idx(i); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
  }
  const uint T = prefix;
  const bool all_ties = int(hist[T & 1023u]) == rem;
  for (int i = tid; i < N; i += NT) {
    uint kk = src.key(i);
    if (kk > T || (all_ties && kk == T)) {
      int sl = atomic_fetch_add_explicit(out_cnt, 1, memory_order_relaxed); out_key[sl] = kk; out_idx[sl] = src.idx(i);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!all_ties) {  // rare: more elements equal to the threshold than needed -> take the lowest indices, in order
    int base = atomic_load_explicit(out_cnt, memory_order_relaxed);
    int taken = 0;
    for (int i0 = 0; i0 < N && taken < rem; i0 += NT) {
      int i = i0 + tid;
      uint f = (i < N && src.key(i) == T) ? 1u : 0u;
      uint tot;
      uint ex = attn_scan_excl(f, tmp, sg, ln, NT / 32, tot);
      if (f && taken + int(ex) < rem) { out_key[base + taken + ex] = T; out_idx[base + taken + ex] = src.idx(i); }
      taken += int(tot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) atomic_store_explicit(out_cnt, base + rem, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ascending bitonic sort of a[0..m) (pads with INT_MAX up to the next power of two <= cap)
inline void attn_sort_int(threadgroup int* a, int m, int cap, ushort tid) {
  int P = 1;
  while (P < m) P <<= 1;
  for (int i = m + tid; i < P; i += 1024) a[i] = INT_MAX;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k2 = 2; k2 <= P; k2 <<= 1)
    for (int j = k2 >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < P; i += 1024) {
        int ixj = i ^ j;
        if (ixj > i) {
          int x = a[i], y = a[ixj];
          bool up = (i & k2) == 0;
          if ((x > y) == up) { a[i] = y; a[ixj] = x; }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// indexer query prep: rope(last 64 of each 128-dim head) + fp4 e8m0/32 sim; thread = (head, 32-dim block)
inline void attn_idx_prep_thread(device const float* qraw, device const float2* tab, int pos, int actq, ushort t,
                                 thread float* v) {
  int h = t >> 2, b = t & 3;
  device const float4* src = (device const float4*)(qraw + h * ATTN_ID + b * 32);
  for (int i = 0; i < 8; i++) { float4 x = src[i]; v[4 * i] = x.x; v[4 * i + 1] = x.y; v[4 * i + 2] = x.z; v[4 * i + 3] = x.w; }
  if (b >= 2) {
    device const float2* cs = tab + (size_t)pos * 32 + (b - 2) * 16;
    for (int i = 0; i < 16; i++) { float2 r = attn_rot(float2(v[2 * i], v[2 * i + 1]), cs[i]); v[2 * i] = r.x; v[2 * i + 1] = r.y; }
  }
  if (actq) {
    float m = 0;
    for (int i = 0; i < 32; i++) m = max(m, fabs(v[i]));
    int e = attn_log2_ceil(max(m, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    float s = attn_pow2(e), is = attn_pow2(-e);
    for (int i = 0; i < 32; i++) v[i] = attn_fp4e8_q(v[i], s, is);
  }
}

struct AttnIdxDecArgs {
  int pos;           // query position
  int len;           // visible compressed rows = (pos+1)/ratio
  int nvirt;         // virtual keys: len (full scan) or ncand*8 (candidate list mode)
  int list_mode;     // 1: virtual key v -> row blist[v/8]*8 + v%8
  int actq;
  int tiles_per_tg;  // 64-key tiles per threadgroup
  int cand_src;      // 1: also write block maxima (blocks of 8, newest pinned to +inf) to bscores
};

// Decode scoring. grid (ceil(ceil(nvirt/64)/tiles_per_tg)), 128 threads (4 simdgroups x 16 keys per tile).
kernel void attn_idx_score_dec(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                               device const float2* tab [[buffer(2)]], device const half* ikey [[buffer(3)]],
                               device const int* blist [[buffer(4)]], device float* scores [[buffer(5)]],
                               device float* bscores [[buffer(6)]], constant AttnIdxDecArgs& a [[buffer(7)]],
                               uint tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup half Qs[ATTN_IH * ATTN_ID];
  threadgroup float ws[ATTN_IH];
  {
    float v[32];
    attn_idx_prep_thread(qraw, tab, a.pos, a.actq, t, v);
    threadgroup half* d = Qs + (t >> 2) * ATTN_ID + (t & 3) * 32;
    for (int i = 0; i < 32; i++) d[i] = half(v[i]);
    if (t < 32) ws[t] = wraw[t] * (1.0f / 64.0f);  // 128^-0.5 * 32^-0.5
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qa[4][16];
  for (int ht = 0; ht < 4; ht++)
    for (int kk = 0; kk < 16; kk++) simdgroup_load(qa[ht][kk], Qs + ht * 8 * ATTN_ID + kk * 8, ATTN_ID);
  const ushort fm = attn_frag_row(ln), fn = attn_frag_col(ln);
  float wl[4];
  for (int ht = 0; ht < 4; ht++) wl[ht] = ws[ht * 8 + fm];
  const int nblocks = (a.len + 7) / 8;
  for (int tt = 0; tt < a.tiles_per_tg; tt++) {
    const int v0 = (int(tg) * a.tiles_per_tg + tt) * 64 + sg * 16;
    if (v0 >= a.nvirt) break;
    int base[2];
    bool ok[2];
    for (int kt = 0; kt < 2; kt++) {
      int v = v0 + kt * 8;
      if (a.list_mode) { int blk = v < a.nvirt ? blist[v >> 3] : -1; ok[kt] = blk >= 0; base[kt] = ok[kt] ? blk * 8 : 0; }
      else { ok[kt] = true; base[kt] = v; }
    }
    simdgroup_float8x8 acc[4][2];
    for (int ht = 0; ht < 4; ht++) for (int kt = 0; kt < 2; kt++) acc[ht][kt] = simdgroup_float8x8(0);
    for (int kk = 0; kk < 16; kk++) {
      for (int kt = 0; kt < 2; kt++) {
        simdgroup_half8x8 kb;
        simdgroup_load(kb, ikey + (size_t)base[kt] * ATTN_ID + kk * 8, ATTN_ID, ulong2(0, 0), true);
        for (int ht = 0; ht < 4; ht++) simdgroup_multiply_accumulate(acc[ht][kt], qa[ht][kk], kb, acc[ht][kt]);
      }
    }
    for (int kt = 0; kt < 2; kt++) {
      float2 s = 0;
      for (int ht = 0; ht < 4; ht++) {
        auto e = acc[ht][kt].thread_elements();
        s.x += max(float(e[0]), 0.0f) * wl[ht];
        s.y += max(float(e[1]), 0.0f) * wl[ht];
      }
      s += simd_shuffle_xor(s, 2);
      s += simd_shuffle_xor(s, 4);
      s += simd_shuffle_xor(s, 16);
      const int va = v0 + kt * 8 + fn;
      const int pa = base[kt] + fn;
      float sx = (ok[kt] && va < a.nvirt && pa < a.len) ? s.x : -INFINITY;
      float sy = (ok[kt] && va + 1 < a.nvirt && pa + 1 < a.len) ? s.y : -INFINITY;
      if ((ln & 0x16) == 0) {
        if (va < a.nvirt) scores[va] = sx;
        if (va + 1 < a.nvirt) scores[va + 1] = sy;
      }
      if (a.cand_src) {
        float bm = max(sx, sy);
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        int b = (v0 + kt * 8) >> 3;
        if (b == (a.len - 1) / 8) bm = INFINITY;
        if (ln == 0 && b < nblocks) bscores[b] = bm;
      }
    }
  }
}

struct AttnSrcScores {
  device const float* s;
  uint key(int i) const { return attn_fkey(s[i]); }
  int idx(int i) const { return i; }
};

struct AttnTopkArgs { int n; int k; int list_mode; };
// Decode top-k: one threadgroup (1024 threads). out[k]: selected rows sorted ascending, -1 padded.
// list_mode maps virtual index v -> blist[v/8]*8 + v%8 (sorted list => order preserved).
kernel void attn_topk_dec(device const float* scores [[buffer(0)]], device const int* blist [[buffer(1)]],
                          device int* out [[buffer(2)]], constant AttnTopkArgs& a [[buffer(3)]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  AttnSrcScores src{scores};
  attn_select(src, a.n, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  attn_sort_int(oidx, m, 2048, tid);
  for (int i = tid; i < a.k; i += 1024) {
    int v = i < m ? oidx[i] : -1;
    if (v >= 0 && a.list_mode) v = blist[v >> 3] * 8 + (v & 7);
    out[i] = v;
  }
}

// ------------------------------------------------------------------------------------------------------------------
// Prefill
struct AttnIdxQArgs { int start; int n; int actq; };
// grid (n), 128 threads -> Qi half [n][32][128], wq f32 [n][32] (scaled)
kernel void attn_idx_qprep(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                           device const float2* tab [[buffer(2)]], device half* Qi [[buffer(3)]],
                           device float* wq [[buffer(4)]], constant AttnIdxQArgs& a [[buffer(5)]],
                           uint q [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  float v[32];
  attn_idx_prep_thread(qraw + (size_t)q * ATTN_IH * ATTN_ID, tab, a.start + int(q), a.actq, t, v);
  device half4* d = (device half4*)(Qi + (size_t)q * ATTN_IH * ATTN_ID + (t >> 2) * ATTN_ID + (t & 3) * 32);
  for (int i = 0; i < 8; i++) d[i] = half4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  if (t < 32) wq[q * ATTN_IH + t] = wraw[q * ATTN_IH + t] * (1.0f / 64.0f);
}

struct AttnIdxPfArgs {
  int start, n, ratio;
  int kc0, kc;       // key chunk [kc0, kc0+kc)
  int ld_s;          // row stride of S (and 8*row stride of BS)
  int use_mask, mask_words;
  int cand_src;
  int krows, qrows;  // tensor extents (allocated rows) of ikey and Qi
};
// grid (ceil(kc/128), ceil(n/2)), 128 threads. S[q][t-kc0] = sum_h relu(q_h.k_t) w_h (or -inf if invisible/masked).
// Tile: QT queries (QT*32 rows of Qi) x NK keys, tensor-op matmul, relu*w epilogue through threadgroup memory.
template <int QT, int NK>
inline void attn_idx_score_pf_impl(device half* Qi, device const float* wq, device half* ikey, device float* S,
                                   device const uint* mask, device float* BS, constant AttnIdxPfArgs& a, uint2 tg,
                                   ushort t, threadgroup float* T) {
  constexpr int M = QT * 32;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, a.krows));
  constexpr auto desc = matmul2d_descriptor(M, NK, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, M>(0, int(tg.y) * M);
  auto mB = tB.template slice<ATTN_ID, NK>(0, a.kc0 + int(tg.x) * NK);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mB, cT);
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      int col = id[0], row = id[1];
      int q = int(tg.y) * QT + row / 32;
      float w = q < a.n ? wq[q * ATTN_IH + (row & 31)] : 0.0f;
      T[row * NK + col] = max(cT[i], 0.0f) * w;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int o = t; o < QT * NK; o += 128) {
    const int ql = o / NK, key = o % NK;
    const int q = int(tg.y) * QT + ql;
    float s = 0;
    for (int h = 0; h < 32; h++) s += T[(ql * 32 + h) * NK + key];
    const int tl = int(tg.x) * NK + key, tabs = a.kc0 + tl;
    const int len = (a.start + q + 1) / a.ratio;
    bool vis = q < a.n && tabs < len && tl < a.kc;
    if (vis && a.use_mask) vis = (mask[(size_t)q * a.mask_words + (tabs >> 8)] >> ((tabs >> 3) & 31)) & 1u;
    float sc = vis ? s : -INFINITY;
    const bool live = q < a.n && tl < a.kc;
    if (live) S[(size_t)q * a.ld_s + tl] = sc;
    if (a.cand_src) {  // 8 consecutive keys = 8 consecutive lanes
      float bm = sc;
      bm = max(bm, simd_shuffle_xor(bm, 1)); bm = max(bm, simd_shuffle_xor(bm, 2)); bm = max(bm, simd_shuffle_xor(bm, 4));
      if (len > 0 && (tabs >> 3) == (len - 1) / 8) bm = INFINITY;
      if (live && (key & 7) == 0) BS[(size_t)q * (a.ld_s / 8) + (tl >> 3)] = bm;
    }
  }
}
#define ATTN_IDX_PF_KERNEL(NAME, QT, NK)                                                                          \
  kernel void NAME(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],                            \
                   device half* ikey [[buffer(2)]], device float* S [[buffer(3)]],                                 \
                   device const uint* mask [[buffer(4)]], device float* BS [[buffer(5)]],                          \
                   constant AttnIdxPfArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],             \
                   ushort t [[thread_index_in_threadgroup]]) {                                                     \
    threadgroup float T[QT * 32 * NK];                                                                             \
    attn_idx_score_pf_impl<QT, NK>(Qi, wq, ikey, S, mask, BS, a, tg, t, T);                                        \
  }
ATTN_IDX_PF_KERNEL(attn_idx_score_pf, 2, 128)  // (4 queries x 64 keys: -30%; 2 x 64: -3%)

// Register epilogue variant (2 queries x 128 keys, 4 simdgroups). Relies on the cooperative destination layout of
// matmul2d(64, 128, 128) on 4 simdgroups (verified at init by attn_idx_layout_check; else the TG-memory kernel is used):
//   thread (sg, l), element i: row = R(sg,l) + 8*((i>>2)&1) + 32*(i>>5), col = C(sg,l) + (i&3) + 32*((i>>3)&3)
//   R = ((l>>1)&3) + 4*((l>>4)&1) + 16*(sg>>1),  C = 4*(l&1) + 8*((l>>3)&1) + 16*(sg&1)
// Head sum: in-thread (rows r, r+8), reduce-scatter over lane bits 1, 2, 4 (16 heads), then sg 2/3 -> sg 0/1 via 1 KB of
// threadgroup memory. Each lane of sg 0/1 ends with 4 consecutive keys of one query.
inline int attn_pf2_row(ushort sg, ushort l) { return ((l >> 1) & 3) + 4 * ((l >> 4) & 1) + 16 * (sg >> 1); }
inline int attn_pf2_col(ushort sg, ushort l) { return 4 * (l & 1) + 8 * ((l >> 3) & 1) + 16 * (sg & 1); }
kernel void attn_idx_score_pf2(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],
                               device half* ikey [[buffer(2)]], device float* S [[buffer(3)]],
                               device const uint* mask [[buffer(4)]], device float* BS [[buffer(5)]],
                               constant AttnIdxPfArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  constexpr int NK = 128;
  threadgroup float4 red[64];
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, a.krows));
  constexpr auto desc = matmul2d_descriptor(64, NK, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, 64>(0, int(tg.y) * 64);
  auto mB = tB.template slice<ATTN_ID, NK>(0, a.kc0 + int(tg.x) * NK);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 64; ++i) cT[i] = 0;
  op.run(mA, mB, cT);
  const int r0 = attn_pf2_row(sg, l), q0 = int(tg.y) * 2;
  float w[2][2];  // [query][row +0 / +8]
  for (int qq = 0; qq < 2; qq++)
    for (int hb = 0; hb < 2; hb++) w[qq][hb] = q0 + qq < a.n ? wq[(q0 + qq) * ATTN_IH + r0 + 8 * hb] : 0.0f;
  float v[32];  // [q][j][c] = v[q*16 + j*4 + c]
  #pragma unroll
  for (int qq = 0; qq < 2; qq++)
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int i0 = c + 8 * j + 32 * qq;
        v[qq * 16 + j * 4 + c] = max(cT[i0], 0.0f) * w[qq][0] + max(cT[i0 + 4], 0.0f) * w[qq][1];
      }
  // lane bit 1: keep query b1
  const bool b1 = (l >> 1) & 1, b2 = (l >> 2) & 1, b4 = (l >> 4) & 1;
  float u[16];
  #pragma unroll
  for (int x = 0; x < 16; x++) {
    float send = b1 ? v[x] : v[16 + x];
    float keep = b1 ? v[16 + x] : v[x];
    u[x] = keep + simd_shuffle_xor(send, 2);
  }
  // lane bit 2: keep j in {2*b2, 2*b2+1}
  float t8[8];
  #pragma unroll
  for (int x = 0; x < 8; x++) {
    float send = b2 ? u[x] : u[8 + x];
    float keep = b2 ? u[8 + x] : u[x];
    t8[x] = keep + simd_shuffle_xor(send, 4);
  }
  // lane bit 4: keep j = 2*b2 + b4
  float4 f;
  #pragma unroll
  for (int x = 0; x < 4; x++) {
    float send = b4 ? t8[x] : t8[4 + x];
    float keep = b4 ? t8[4 + x] : t8[x];
    f[x] = keep + simd_shuffle_xor(send, 16);
  }
  if (sg >= 2) red[(sg & 1) * 32 + l] = f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= 2) return;
  f += red[(sg & 1) * 32 + l];
  const int q = q0 + int(b1);
  const int key0 = attn_pf2_col(sg, l) + 32 * (2 * int(b2) + int(b4));
  const int tl = int(tg.x) * NK + key0, tabs = a.kc0 + tl;
  const int len = (a.start + q + 1) / a.ratio;
  const bool qok = q < a.n;
  float4 sc;
  #pragma unroll
  for (int c = 0; c < 4; c++) {
    bool vis = qok && tabs + c < len && tl + c < a.kc;
    if (vis && a.use_mask) vis = (mask[(size_t)q * a.mask_words + ((tabs + c) >> 8)] >> (((tabs + c) >> 3) & 31)) & 1u;
    sc[c] = vis ? f[c] : -INFINITY;
  }
  if (qok) {
    device float* dst = S + (size_t)q * a.ld_s + tl;
    if (tl + 4 <= a.kc) *(device float4*)dst = sc;
    else for (int c = 0; c < 4; c++) if (tl + c < a.kc) dst[c] = sc[c];
  }
  if (a.cand_src) {  // block of 8 keys = this lane's 4 + lane^1's 4
    float bm = max(max(sc[0], sc[1]), max(sc[2], sc[3]));
    bm = max(bm, simd_shuffle_xor(bm, 1));
    if (len > 0 && (tabs >> 3) == (len - 1) / 8) bm = INFINITY;
    if (qok && tl < a.kc && (l & 1) == 0) BS[(size_t)q * (a.ld_s / 8) + (tl >> 3)] = bm;
  }
}

// Layout check for attn_idx_score_pf2: out[t] = number of elements whose (row, col) differ from the assumed formula
// (all zeros = layout as assumed). grid 1, 128 threads.
kernel void attn_idx_layout_check(device half* Qi [[buffer(0)]], device half* ikey [[buffer(1)]], device int* out [[buffer(2)]],
                                  ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, 64));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, 128));
  constexpr auto desc = matmul2d_descriptor(64, 128, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  int bad = cT.get_capacity() == 64 ? 0 : 1000;
  for (uint16_t i = 0; i < cT.get_capacity() && i < 64; ++i) {
    if (!cT.is_valid_element(i)) { bad++; continue; }
    auto id = cT.get_multidimensional_index(i);
    int row = attn_pf2_row(sg, l) + 8 * ((i >> 2) & 1) + 32 * (i >> 5);
    int col = attn_pf2_col(sg, l) + (i & 3) + 32 * ((i >> 3) & 3);
    if (int(id[1]) != row || int(id[0]) != col) bad++;
  }
  {
    threadgroup half Kg[64 * ATTN_ID];
    auto tB2 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kg, dextents<int32_t, 2>(ATTN_ID, 64));
    constexpr auto desc2 = matmul2d_descriptor(64, 64, ATTN_ID, false, true, false);
    matmul2d<desc2, execution_simdgroups<4>> op2;
    auto c2 = op2.template get_destination_cooperative_tensor<decltype(tA), decltype(tB2), float>();
    if (c2.get_capacity() != 32) bad += 1000;
    for (uint16_t i = 0; i < c2.get_capacity() && i < 32; ++i) {
      if (!c2.is_valid_element(i)) { bad++; continue; }
      auto id = c2.get_multidimensional_index(i);
      int row = attn_pf2_row(sg, l) + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
      int col = attn_pf2_col(sg, l) + (i & 3) + 32 * ((i >> 3) & 1);
      if (int(id[1]) != row || int(id[0]) != col) bad++;
    }
  }
  out[sg * 32 + l] = bad;
}

struct AttnSrcMerge {
  device const uint* rk;
  device const int* ri;
  int rc;
  device const float* s;
  int base;
  uint key(int i) const { return i < rc ? rk[i] : attn_fkey(s[i - rc]); }
  int idx(int i) const { return i < rc ? ri[i] : base + (i - rc); }
};
struct AttnMergeArgs {
  int n, k;
  int cnt;    // valid elements in this chunk row
  int ld_s;   // S row stride
  int base;   // element index of S[q][0]
  int first;  // 1: running list empty
};
// Running top-k per query: R[q] <- topk(R[q] U S[q][0..cnt)). grid (n), 1024 threads.
kernel void attn_topk_merge(device const float* S [[buffer(0)]], device uint* Rk [[buffer(1)]], device int* Ri [[buffer(2)]],
                            device int* Rc [[buffer(3)]], constant AttnMergeArgs& a [[buffer(4)]],
                            uint q [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  int rc = a.first ? 0 : Rc[q];
  AttnSrcMerge src{Rk + (size_t)q * a.k, Ri + (size_t)q * a.k, rc, S + (size_t)q * a.ld_s, a.base};
  attn_select(src, rc + a.cnt, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  for (int i = tid; i < m; i += 1024) { Rk[(size_t)q * a.k + i] = okey[i]; Ri[(size_t)q * a.k + i] = oidx[i]; }
  if (tid == 0) Rc[q] = m;
}

// ---- threshold-filtered running merge (k <= 512): when the running list is full (rc == k), a new element can only enter
// if its key is strictly greater than the list minimum T (ties at T lose: running entries precede the chunk in source
// order, i.e. have lower indices). The chunk is compacted in index order to the survivors (> T), then the usual exact
// select runs over running + survivors -> same result as the full merge. Survivor overflow -> full merge.
constant constexpr int ATTN_MF_CAP = 2048;
struct AttnSrcMergeTG {
  device const uint* rk;
  device const int* ri;
  int rc;
  threadgroup const uint* ck;
  threadgroup const int* ci;
  uint key(int i) const { return i < rc ? rk[i] : ck[i - rc]; }
  int idx(int i) const { return i < rc ? ri[i] : ci[i - rc]; }
};
// Returns survivors count (> ATTN_MF_CAP: overflow). key_of(j) / idx_of(j) read chunk element j.
template <typename KF, typename IF>
inline int attn_mf_compact(int cnt, uint T, thread const KF& key_of, thread const IF& idx_of, threadgroup uint* ck,
                           threadgroup int* ci, threadgroup uint* tmp, ushort tid, ushort sg, ushort ln) {
  int base = 0;
  for (int j0 = 0; j0 < cnt; j0 += 1024) {
    const int j = j0 + tid;
    uint kk = j < cnt ? key_of(j) : 0u;
    uint f = kk > T ? 1u : 0u;
    uint tot;
    uint ex = attn_scan_excl(f, tmp, sg, ln, 32, tot);
    if (f && base + int(ex) < ATTN_MF_CAP) { ck[base + ex] = kk; ci[base + ex] = idx_of(j); }
    base += int(tot);
    if (base > ATTN_MF_CAP) break;  // uniform
  }
  return base;
}
inline uint attn_mf_min(device const uint* rk, int rc, threadgroup uint* tmp, ushort tid, ushort ln) {
  if (tid == 0) tmp[36] = 0xFFFFFFFFu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mn = 0xFFFFFFFFu;
  for (int i = tid; i < rc; i += 1024) mn = min(mn, rk[i]);
  mn = simd_min(mn);
  if (ln == 0) atomic_fetch_min_explicit((threadgroup atomic_uint*)&tmp[36], mn, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint T = tmp[36];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return T;
}
// Same contract as attn_topk_merge (k <= 512).
kernel void attn_topk_merge_f(device const float* S [[buffer(0)]], device uint* Rk [[buffer(1)]], device int* Ri [[buffer(2)]],
                              device int* Rc [[buffer(3)]], constant AttnMergeArgs& a [[buffer(4)]],
                              uint q [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[512];
  threadgroup int oidx[512];
  threadgroup uint ck[ATTN_MF_CAP];
  threadgroup int ci[ATTN_MF_CAP];
  threadgroup atomic_int ocnt;
  const int rc = a.first ? 0 : Rc[q];
  device const uint* rk = Rk + (size_t)q * a.k;
  device const int* ri = Ri + (size_t)q * a.k;
  device const float* sq = S + (size_t)q * a.ld_s;
  int m;
  bool done = false;
  if (rc == a.k) {
    const uint T = attn_mf_min(rk, rc, tmp, tid, ln);
    auto kf = [&](int j) { return attn_fkey(sq[j]); };
    auto jf = [&](int j) { return a.base + j; };
    const int ns = attn_mf_compact(a.cnt, T, kf, jf, ck, ci, tmp, tid, sg, ln);
    if (ns <= ATTN_MF_CAP) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      AttnSrcMergeTG src{rk, ri, rc, ck, ci};
      attn_select(src, rc + ns, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
      done = true;
    }
  }
  if (!done) {
    AttnSrcMerge src{rk, ri, rc, sq, a.base};
    attn_select(src, rc + a.cnt, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  }
  m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);  // all reads of rk/ri done before overwriting them
  for (int i = tid; i < m; i += 1024) { Rk[(size_t)q * a.k + i] = okey[i]; Ri[(size_t)q * a.k + i] = oidx[i]; }
  if (tid == 0) Rc[q] = m;
}

struct AttnFinArgs { int k; int mode; int mask_words; };  // mode 0: sorted index list [n][k] (-1 pad); 1: block bitmap
kernel void attn_topk_finalize(device const int* Ri [[buffer(0)]], device const int* Rc [[buffer(1)]],
                               device int* out [[buffer(2)]], device uint* mask [[buffer(3)]],
                               constant AttnFinArgs& a [[buffer(4)]], uint q [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup int arr[2048];
  int m = Rc[q];
  if (a.mode == 0) {
    for (int i = tid; i < m; i += 1024) arr[i] = Ri[(size_t)q * a.k + i];
    attn_sort_int(arr, m, 2048, tid);
    for (int i = tid; i < a.k; i += 1024) out[(size_t)q * a.k + i] = i < m ? arr[i] : -1;
  } else {
    threadgroup atomic_uint* bits = (threadgroup atomic_uint*)arr;  // up to 2048 words = 65536 blocks
    for (int i = tid; i < a.mask_words; i += 1024) atomic_store_explicit(&bits[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < m; i += 1024) {
      int b = Ri[(size_t)q * a.k + i];
      atomic_fetch_or_explicit(&bits[b >> 5], 1u << (b & 31), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < a.mask_words; i += 1024)
      mask[(size_t)q * a.mask_words + i] = atomic_load_explicit(&bits[i], memory_order_relaxed);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// Decode fast path: tensor-op scoring (M = 32 heads, N = 64 keys, K = 128) + 4096-bin histogram of the top 12 key bits,
// then a grid-wide compaction and a small final select (attn_topk_compact / attn_topk_final).
struct AttnIdxDec2Args {
  int pos, len, nvirt, actq, tiles_per_tg, cand_src, krows;
};
// grid (ceil(ceil(nvirt/64)/tiles_per_tg)), 128 threads. hist/bhist: device uint[4096] (zeroed by attn_topk_final).
kernel void attn_idx_score_dec2(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                                device const float2* tab [[buffer(2)]], device half* ikey [[buffer(3)]],
                                device float* scores [[buffer(4)]], device float* bscores [[buffer(5)]],
                                device atomic_uint* hist [[buffer(6)]], device atomic_uint* bhist [[buffer(7)]],
                                constant AttnIdxDec2Args& a [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                                ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Qs[ATTN_IH * ATTN_ID];
  threadgroup float T[ATTN_IH * 64];
  threadgroup atomic_uint lh[4096];
  {
    float v[32];
    attn_idx_prep_thread(qraw, tab, a.pos, a.actq, t, v);
    threadgroup half* d = Qs + (t >> 2) * ATTN_ID + (t & 3) * 32;
    for (int i = 0; i < 32; i++) d[i] = half(v[i]);
    for (int b = t; b < 4096; b += 128) atomic_store_explicit(&lh[b], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Qs, dextents<int32_t, 2>(ATTN_ID, ATTN_IH));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(ikey, dextents<int32_t, 2>(ATTN_ID, a.krows));
  constexpr auto desc = matmul2d_descriptor(ATTN_IH, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  const int nblocks = (a.len + 7) / 8;
  for (int tt = 0; tt < a.tiles_per_tg; tt++) {
    const int k0 = (int(tg) * a.tiles_per_tg + tt) * 64;
    if (k0 >= a.nvirt) break;
    auto mB = tB.template slice<ATTN_ID, 64>(0, k0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(mB), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(tA, mB, cT);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto id = cT.get_multidimensional_index(i);
        T[id[1] * 64 + id[0]] = max(cT[i], 0.0f) * (wraw[id[1]] * (1.0f / 64.0f));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64) {
      float s = 0;
      for (int h = 0; h < ATTN_IH; h++) s += T[h * 64 + t];
      const int key = k0 + t;
      const bool valid = key < a.len;
      float sc = valid ? s : -INFINITY;
      if (key < a.nvirt) scores[key] = sc;
      if (valid) atomic_fetch_add_explicit(&lh[attn_fkey(sc) >> 20], 1u, memory_order_relaxed);
      if (a.cand_src) {
        float bm = sc;
        bm = max(bm, simd_shuffle_xor(bm, 1)); bm = max(bm, simd_shuffle_xor(bm, 2)); bm = max(bm, simd_shuffle_xor(bm, 4));
        int b = key >> 3;
        if (b == (a.len - 1) / 8) bm = INFINITY;
        if ((t & 7) == 0 && b < nblocks) {
          bscores[b] = bm;
          atomic_fetch_add_explicit(&bhist[attn_fkey(bm) >> 20], 1u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int b = t; b < 4096; b += 128) {
    uint v = atomic_load_explicit(&lh[b], memory_order_relaxed);
    if (v) atomic_fetch_add_explicit(&hist[b], v, memory_order_relaxed);
  }
}

// threshold bin from a 4096-bin histogram (descending): largest b0 with count(bins >= b0) >= k (or 0 if total <= k).
// 256 threads, 16 bins each. Returns b0 and count of elements in bins > b0 via tmp.
inline void attn_hist_threshold(device const uint* hist, int k, threadgroup uint* tmp, ushort t, ushort sg, ushort ln,
                                thread uint& b0, thread uint& above_b0, thread uint& total) {
  uint c[16], s = 0;
  for (int j = 0; j < 16; j++) { c[j] = hist[4095 - (t * 16 + j)]; s += c[j]; }
  uint above = attn_scan_excl(s, tmp, sg, ln, 8, total);
  if (total <= uint(k)) { b0 = 0; above_b0 = 0; return; }
  if (above < uint(k) && uint(k) <= above + s) {
    uint acc = above;
    for (int j = 0; j < 16; j++) {
      if (acc + c[j] >= uint(k)) { tmp[34] = 4095 - (t * 16 + j); tmp[35] = acc; break; }
      acc += c[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  b0 = tmp[34]; above_b0 = tmp[35];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ctl layout (device uint): [0] count of definite picks (bins > b0) in sel, [1] count of candidates (bin == b0) in cand,
// [2] overflow flag. sel: int[k]; cand: uint2 (key, idx) [cap].
struct AttnCompactArgs { int n; int k; int cap; int per_tg; };
// grid (ceil(n/per_tg)), 256 threads
kernel void attn_topk_compact(device const float* scores [[buffer(0)]], device const uint* hist [[buffer(1)]],
                              device atomic_uint* ctl [[buffer(2)]], device int* sel [[buffer(3)]],
                              device uint2* cand [[buffer(4)]], constant AttnCompactArgs& a [[buffer(5)]],
                              uint tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint tmp[40];
  uint b0, above, total;
  attn_hist_threshold(hist, a.k, tmp, t, sg, ln, b0, above, total);
  const int i0 = int(tg) * a.per_tg, i1 = min(a.n, i0 + a.per_tg);
  for (int i = i0 + t; i < i1; i += 256) {
    uint kk = attn_fkey(scores[i]);
    if (kk == 0u) continue;
    uint bin = kk >> 20;
    if (bin > b0) {
      uint sl = atomic_fetch_add_explicit(&ctl[0], 1u, memory_order_relaxed);
      sel[sl] = i;
    } else if (bin == b0) {
      uint sl = atomic_fetch_add_explicit(&ctl[1], 1u, memory_order_relaxed);
      if (sl < uint(a.cap)) cand[sl] = uint2(kk, uint(i)); else atomic_store_explicit(&ctl[2], 1u, memory_order_relaxed);
    }
  }
}

struct AttnSrcCand {
  device const uint2* c;
  uint key(int i) const { return c[i].x; }
  int idx(int i) const { return int(c[i].y); }
};
struct AttnFinalArgs { int n; int k; int list_mode; };
// One threadgroup (1024): picks = sel[0..ctl0) + top (k - ctl0) of cand; sorted ascending; -1 padded; maps list mode;
// resets hist/ctl for the next use. Falls back to a full exact select over scores on candidate overflow or a
// partially-taken tie (keeps results deterministic).
kernel void attn_topk_final(device const float* scores [[buffer(0)]], device uint* hist [[buffer(1)]],
                            device uint* ctl [[buffer(2)]], device const int* sel [[buffer(3)]],
                            device const uint2* cand [[buffer(4)]], device const int* blist [[buffer(5)]],
                            device int* out [[buffer(6)]], constant AttnFinalArgs& a [[buffer(7)]],
                            ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint lhist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  threadgroup int flag;
  const int nsel = int(ctl[0]), ncand = int(ctl[1]);
  const bool overflow = ctl[2] != 0u;
  int m = 0;
  if (tid == 0) flag = 0;
  if (!overflow) {
    for (int i = tid; i < nsel; i += 1024) oidx[i] = sel[i];
    const int need = min(a.k, nsel + ncand) - nsel;
    AttnSrcCand src{cand};
    // select `need` of the candidates into okey/oidx after the definite picks (reuse lhist as histogram)
    threadgroup uint* ok2 = okey;
    threadgroup int* oi2 = oidx + nsel;
    if (need > 0) {
      attn_select(src, ncand, need, lhist, tmp, ok2, oi2, &ocnt, tid, sg, ln);
      // detect a partially taken tie at the threshold: if any unselected candidate equals the minimum selected key
      // we cannot know the lowest-index rule was respected (candidate order is arbitrary) -> full fallback.
      if (need < ncand) {
        if (tid == 0) { tmp[37] = 0xFFFFFFFFu; tmp[38] = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint mn = 0xFFFFFFFFu;
        for (int i = tid; i < need; i += 1024) mn = min(mn, ok2[i]);
        mn = simd_min(mn);
        if (ln == 0) atomic_fetch_min_explicit((threadgroup atomic_uint*)&tmp[37], mn, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint T = tmp[37];
        uint unsel = 0;  // candidates equal to T minus selected equal to T (>= 0)
        for (int i = tid; i < ncand; i += 1024) unsel += cand[i].x == T ? 1u : 0u;
        for (int i = tid; i < need; i += 1024) unsel -= ok2[i] == T ? 1u : 0u;
        unsel = simd_sum(unsel);
        if (ln == 0) atomic_fetch_add_explicit((threadgroup atomic_uint*)&tmp[38], unsel, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0 && tmp[38] != 0u) flag = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    m = nsel + max(need, 0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (overflow || flag) {
    AttnSrcScores src{scores};
    attn_select(src, a.n, a.k, lhist, tmp, okey, oidx, &ocnt, tid, sg, ln);
    m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  }
  attn_sort_int(oidx, m, 2048, tid);
  for (int i = tid; i < a.k; i += 1024) {
    int v = i < m ? oidx[i] : -1;
    if (v >= 0 && a.list_mode) v = blist[v >> 3] * 8 + (v & 7);
    out[i] = v;
  }
  for (int b = tid; b < 4096; b += 1024) hist[b] = 0;
  if (tid < 4) ctl[tid] = 0;
}

// ------------------------------------------------------------------------------------------------------------------
// Prefill, candidate-restricted index layers (24..36): score only the union of the candidate blocks of each 2-query
// tile (instead of all keys + mask).
struct AttnUnionArgs { int n; int mask_words; int cap; };  // cap = max blocks per union list (4096)
// grid (ceil(n/2)), 1024 threads: ulist[tile][0..ucount) = sorted union of the candidate blocks of queries 2t, 2t+1
kernel void attn_cand_union(device const uint* mask [[buffer(0)]], device int* ulist [[buffer(1)]],
                            device int* ucount [[buffer(2)]], constant AttnUnionArgs& a [[buffer(3)]],
                            uint tile [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint tmp[40];
  const int q0 = int(tile) * 2, q1 = min(q0 + 1, a.n - 1);
  int base = 0;
  for (int w0 = 0; w0 < a.mask_words; w0 += 1024) {
    int w = w0 + tid;
    uint bits = w < a.mask_words ? (mask[(size_t)q0 * a.mask_words + w] | mask[(size_t)q1 * a.mask_words + w]) : 0u;
    uint tot;
    uint ex = attn_scan_excl(popcount(bits), tmp, sg, ln, 32, tot);
    int o = base + int(ex);
    while (bits) {
      int b = ctz(bits);
      bits &= bits - 1;
      if (o < a.cap) ulist[(size_t)tile * a.cap + o] = w * 32 + b;
      o++;
    }
    base += int(tot);
  }
  if (tid == 0) ucount[tile] = min(base, a.cap);
}

struct AttnIdxPfListArgs {
  int start, n, ratio;
  int kc0, kc;       // virtual key chunk
  int ld_s, mask_words, cap, qrows;
};
// grid (kc/64, ceil(n/2)), 128 threads. Virtual key v of tile y -> position ulist[y][v/8]*8 + v%8.
kernel void attn_idx_score_pf_list(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],
                                   device const half* ikey [[buffer(2)]], device float* S [[buffer(3)]],
                                   device const uint* mask [[buffer(4)]], device const int* ulist [[buffer(5)]],
                                   device const int* ucount [[buffer(6)]], constant AttnIdxPfListArgs& a [[buffer(7)]],
                                   uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Kg[64 * ATTN_ID];
  threadgroup float T[64 * 64];
  const int nvirt = ucount[tg.y] * 8;
  const int v0 = a.kc0 + int(tg.x) * 64;
  if (v0 >= nvirt) return;
  {  // gather 8 blocks x 8 keys x 128 dims: thread t -> key t/2, half row (64 dims)
    int kl = t >> 1, part = t & 1, v = v0 + kl;
    threadgroup uint4* dst = (threadgroup uint4*)(Kg + kl * ATTN_ID + part * 64);
    if (v < nvirt) {
      int pos = ulist[(size_t)tg.y * a.cap + (v >> 3)] * 8 + (v & 7);
      device const uint4* src = (device const uint4*)(ikey + (size_t)pos * ATTN_ID + part * 64);
      for (int j = 0; j < 8; j++) dst[j] = src[j];
    } else {
      for (int j = 0; j < 8; j++) dst[j] = uint4(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kg, dextents<int32_t, 2>(ATTN_ID, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, 64>(0, int(tg.y) * 64);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, tB, cT);
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      int col = id[0], row = id[1];
      int q = int(tg.y) * 2 + row / 32;
      float w = q < a.n ? wq[q * ATTN_IH + (row & 31)] : 0.0f;
      T[row * 64 + col] = max(cT[i], 0.0f) * w;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ql = t >> 6, key = t & 63;
  const int q = int(tg.y) * 2 + ql;
  float s = 0;
  for (int h = 0; h < 32; h++) s += T[(ql * 32 + h) * 64 + key];
  const int v = v0 + key, tl = v - a.kc0;
  if (q >= a.n || tl >= a.kc) return;
  bool vis = v < nvirt;
  if (vis) {
    int blk = ulist[(size_t)tg.y * a.cap + (v >> 3)];
    int pos = blk * 8 + (v & 7);
    int len = (a.start + q + 1) / a.ratio;
    vis = pos < len && ((mask[(size_t)q * a.mask_words + (blk >> 5)] >> (blk & 31)) & 1u);
  }
  S[(size_t)q * a.ld_s + tl] = vis ? s : -INFINITY;
}

// Register-epilogue variant of attn_idx_score_pf_list (same layout family as attn_idx_score_pf2; the 64x64 destination:
// element i -> row = R + 8*((i>>2)&1) + 32*((i>>4)&1), col = C + (i&3) + 32*((i>>3)&1); checked at init).
kernel void attn_idx_score_pf_list2(device half* Qi [[buffer(0)]], device const float* wq [[buffer(1)]],
                                    device const half* ikey [[buffer(2)]], device float* S [[buffer(3)]],
                                    device const uint* mask [[buffer(4)]], device const int* ulist [[buffer(5)]],
                                    device const int* ucount [[buffer(6)]], constant AttnIdxPfListArgs& a [[buffer(7)]],
                                    uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half Kg[64 * ATTN_ID];
  threadgroup float2 red[64];
  const int nvirt = ucount[tg.y] * 8;
  const int v0 = a.kc0 + int(tg.x) * 64;
  if (v0 >= nvirt) return;
  {  // gather 8 blocks x 8 keys x 128 dims: thread t -> key t/2, half row (64 dims)
    int kl = t >> 1, part = t & 1, v = v0 + kl;
    threadgroup uint4* dst = (threadgroup uint4*)(Kg + kl * ATTN_ID + part * 64);
    if (v < nvirt) {
      int pos = ulist[(size_t)tg.y * a.cap + (v >> 3)] * 8 + (v & 7);
      device const uint4* src = (device const uint4*)(ikey + (size_t)pos * ATTN_ID + part * 64);
      for (int j = 0; j < 8; j++) dst[j] = src[j];
    } else {
      for (int j = 0; j < 8; j++) dst[j] = uint4(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qi, dextents<int32_t, 2>(ATTN_ID, a.qrows));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kg, dextents<int32_t, 2>(ATTN_ID, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<ATTN_ID, 64>(0, int(tg.y) * 64);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(tB), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 32; ++i) cT[i] = 0;
  op.run(mA, tB, cT);
  const int r0 = attn_pf2_row(sg, l), q0 = int(tg.y) * 2;
  float w[2][2];
  for (int qq = 0; qq < 2; qq++)
    for (int hb = 0; hb < 2; hb++) w[qq][hb] = q0 + qq < a.n ? wq[(q0 + qq) * ATTN_IH + r0 + 8 * hb] : 0.0f;
  float v[16];  // [q][j][c]
  #pragma unroll
  for (int qq = 0; qq < 2; qq++)
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int i0 = c + 8 * j + 16 * qq;
        v[qq * 8 + j * 4 + c] = max(cT[i0], 0.0f) * w[qq][0] + max(cT[i0 + 4], 0.0f) * w[qq][1];
      }
  const bool b1 = (l >> 1) & 1, b2 = (l >> 2) & 1, b4 = (l >> 4) & 1;
  float u[8];
  #pragma unroll
  for (int x = 0; x < 8; x++) { float sd = b1 ? v[x] : v[8 + x]; float kp = b1 ? v[8 + x] : v[x]; u[x] = kp + simd_shuffle_xor(sd, 2); }
  float t4[4];
  #pragma unroll
  for (int x = 0; x < 4; x++) { float sd = b2 ? u[x] : u[4 + x]; float kp = b2 ? u[4 + x] : u[x]; t4[x] = kp + simd_shuffle_xor(sd, 4); }
  float2 f;
  #pragma unroll
  for (int x = 0; x < 2; x++) { float sd = b4 ? t4[x] : t4[2 + x]; float kp = b4 ? t4[2 + x] : t4[x]; f[x] = kp + simd_shuffle_xor(sd, 16); }
  if (sg >= 2) red[(sg & 1) * 32 + l] = f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= 2) return;
  f += red[(sg & 1) * 32 + l];
  const int q = q0 + int(b1);
  const int key0 = attn_pf2_col(sg, l) + 32 * int(b2) + 2 * int(b4);
  if (q >= a.n) return;
  const int len = (a.start + q + 1) / a.ratio;
  float2 sc;
  for (int c = 0; c < 2; c++) {
    const int vv = v0 + key0 + c;
    bool vis = vv < nvirt;
    if (vis) {
      int blk = ulist[(size_t)tg.y * a.cap + (vv >> 3)];
      int pos = blk * 8 + (vv & 7);
      vis = pos < len && ((mask[(size_t)q * a.mask_words + (blk >> 5)] >> (blk & 31)) & 1u);
    }
    sc[c] = vis ? f[c] : -INFINITY;
  }
  const int tl = v0 + key0 - a.kc0;
  device float* dst = S + (size_t)q * a.ld_s + tl;
  if (tl + 2 <= a.kc) *(device float2*)dst = sc;
  else if (tl < a.kc) dst[0] = sc[0];
}

struct AttnSrcMergeList {
  device const uint* rk;
  device const int* ri;
  int rc;
  device const float* s;
  int base;
  device const int* ul;
  uint key(int i) const { return i < rc ? rk[i] : attn_fkey(s[i - rc]); }
  int idx(int i) const { if (i < rc) return ri[i]; int v = base + (i - rc); return ul[v >> 3] * 8 + (v & 7); }
};
struct AttnMergeListArgs { int n, k, kc, ld_s, base, first, cap; };
// list-mode merge: chunk element count per query = clamp(ucount[q/2]*8 - base, 0, kc)
kernel void attn_topk_merge_list(device const float* S [[buffer(0)]], device uint* Rk [[buffer(1)]],
                                 device int* Ri [[buffer(2)]], device int* Rc [[buffer(3)]],
                                 device const int* ulist [[buffer(4)]], device const int* ucount [[buffer(5)]],
                                 constant AttnMergeListArgs& a [[buffer(6)]], uint q [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[2048];
  threadgroup int oidx[2048];
  threadgroup atomic_int ocnt;
  const int tile = int(q) / 2;
  const int cnt = clamp(ucount[tile] * 8 - a.base, 0, a.kc);
  const int rc = a.first ? 0 : Rc[q];
  if (!a.first && cnt == 0) return;
  AttnSrcMergeList src{Rk + (size_t)q * a.k, Ri + (size_t)q * a.k, rc, S + (size_t)q * a.ld_s, a.base,
                       ulist + (size_t)tile * a.cap};
  attn_select(src, rc + cnt, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  for (int i = tid; i < m; i += 1024) { Rk[(size_t)q * a.k + i] = okey[i]; Ri[(size_t)q * a.k + i] = oidx[i]; }
  if (tid == 0) Rc[q] = m;
}


// Threshold-filtered variant of attn_topk_merge_list (k <= 512); see attn_topk_merge_f.
kernel void attn_topk_merge_list_f(device const float* S [[buffer(0)]], device uint* Rk [[buffer(1)]],
                                   device int* Ri [[buffer(2)]], device int* Rc [[buffer(3)]],
                                   device const int* ulist [[buffer(4)]], device const int* ucount [[buffer(5)]],
                                   constant AttnMergeListArgs& a [[buffer(6)]], uint q [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup uint hist[2048];
  threadgroup uint tmp[40];
  threadgroup uint okey[512];
  threadgroup int oidx[512];
  threadgroup uint ck[ATTN_MF_CAP];
  threadgroup int ci[ATTN_MF_CAP];
  threadgroup atomic_int ocnt;
  const int tile = int(q) / 2;
  const int cnt = clamp(ucount[tile] * 8 - a.base, 0, a.kc);
  const int rc = a.first ? 0 : Rc[q];
  if (!a.first && cnt == 0) return;
  device const uint* rk = Rk + (size_t)q * a.k;
  device const int* ri = Ri + (size_t)q * a.k;
  device const float* sq = S + (size_t)q * a.ld_s;
  device const int* ul = ulist + (size_t)tile * a.cap;
  bool done = false;
  if (rc == a.k) {
    const uint T = attn_mf_min(rk, rc, tmp, tid, ln);
    auto kf = [&](int j) { return attn_fkey(sq[j]); };
    auto jf = [&](int j) { int v = a.base + j; return ul[v >> 3] * 8 + (v & 7); };
    const int ns = attn_mf_compact(cnt, T, kf, jf, ck, ci, tmp, tid, sg, ln);
    if (ns <= ATTN_MF_CAP) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      AttnSrcMergeTG src{rk, ri, rc, ck, ci};
      attn_select(src, rc + ns, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
      done = true;
    }
  }
  if (!done) {
    AttnSrcMergeList src{rk, ri, rc, sq, a.base, ul};
    attn_select(src, rc + cnt, a.k, hist, tmp, okey, oidx, &ocnt, tid, sg, ln);
  }
  const int m = atomic_load_explicit(&ocnt, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < m; i += 1024) { Rk[(size_t)q * a.k + i] = okey[i]; Ri[(size_t)q * a.k + i] = oidx[i]; }
  if (tid == 0) Rc[q] = m;
}

// Decode fast path, candidate-list mode (layers 24..36): virtual key v -> row blist[v/8]*8 + v%8. Keys of each 64-key
// tile (8 blocks) are gathered into threadgroup memory; histogram via device atomics. Same outputs as dec2.
kernel void attn_idx_score_dec2_list(device const float* qraw [[buffer(0)]], device const float* wraw [[buffer(1)]],
                                     device const float2* tab [[buffer(2)]], device const half* ikey [[buffer(3)]],
                                     device float* scores [[buffer(4)]], device const int* blist [[buffer(5)]],
                                     device atomic_uint* hist [[buffer(6)]], constant AttnIdxDec2Args& a [[buffer(8)]],
                                     uint tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Qs[ATTN_IH * ATTN_ID];
  threadgroup float T[ATTN_IH * 64];
  threadgroup half Kg[64 * ATTN_ID];
  {
    float v[32];
    attn_idx_prep_thread(qraw, tab, a.pos, a.actq, t, v);
    threadgroup half* d = Qs + (t >> 2) * ATTN_ID + (t & 3) * 32;
    for (int i = 0; i < 32; i++) d[i] = half(v[i]);
  }
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Qs, dextents<int32_t, 2>(ATTN_ID, ATTN_IH));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kg, dextents<int32_t, 2>(ATTN_ID, 64));
  constexpr auto desc = matmul2d_descriptor(ATTN_IH, 64, ATTN_ID, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  for (int tt = 0; tt < a.tiles_per_tg; tt++) {
    const int k0 = (int(tg) * a.tiles_per_tg + tt) * 64;
    if (k0 >= a.nvirt) break;
    {  // gather: thread t -> key t/2, 64 dims
      int kl = t >> 1, part = t & 1, v = k0 + kl;
      int blk = v < a.nvirt ? blist[v >> 3] : -1;
      threadgroup uint4* dst = (threadgroup uint4*)(Kg + kl * ATTN_ID + part * 64);
      if (blk >= 0) {
        device const uint4* src = (device const uint4*)(ikey + (size_t)(blk * 8 + (v & 7)) * ATTN_ID + part * 64);
        for (int j = 0; j < 8; j++) dst[j] = src[j];
      } else {
        for (int j = 0; j < 8; j++) dst[j] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(tA, tB, cT);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto id = cT.get_multidimensional_index(i);
        T[id[1] * 64 + id[0]] = max(cT[i], 0.0f) * (wraw[id[1]] * (1.0f / 64.0f));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64) {
      float s = 0;
      for (int h = 0; h < ATTN_IH; h++) s += T[h * 64 + t];
      const int v = k0 + t;
      int blk = v < a.nvirt ? blist[v >> 3] : -1;
      int pos = blk * 8 + (v & 7);
      const bool valid = blk >= 0 && pos < a.len;
      float sc = valid ? s : -INFINITY;
      if (v < a.nvirt) scores[v] = sc;
      if (valid) atomic_fetch_add_explicit(&hist[attn_fkey(sc) >> 20], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_kv.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_kv.metal
// Attention module: window KV write, compressor (pool + norm + index key + rope + fp4 sim), state carry, ring commit.
#line 3 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_kv.metal"

struct AttnKvArgs {
  int start;   // absolute position of input row 0
  int n;       // rows
  int ring;    // 1: dst row = (start+i) % 128 (decode, writes into the window ring); 0: dst row = i (prefill chunk)
  int actq;    // fp8 e4m3 / ue8m0-32 simulation
};

inline float attn_tg_sum4(float v, threadgroup float* red, ushort sg, ushort ln) {
  v = simd_sum(v);
  if (ln == 0) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return red[0] + red[1] + red[2] + red[3];
}

// kv = fp8sim(rope(rmsnorm(kv_raw) * w)) -> half. grid (n), 128 threads; thread t owns dims 4t..4t+3.
kernel void attn_kv_window(device const float* kv [[buffer(0)]], device const float* w [[buffer(1)]],
                           device const float2* tab [[buffer(2)]], device half* dst [[buffer(3)]],
                           constant AttnKvArgs& a [[buffer(4)]], uint i [[threadgroup_position_in_grid]],
                           ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup float red[4];
  const int pos = a.start + int(i);
  float4 x = ((device const float4*)(kv + (size_t)i * ATTN_D))[t];
  float ss = attn_tg_sum4(dot(x, x), red, sg, ln);
  float r = rsqrt(ss * (1.0f / ATTN_D) + 1e-20f);
  x = ((device const float4*)w)[t] * (x * r);
  if (t >= 112) {
    int pi = 2 * (t - 112);
    device const float2* cs = tab + (size_t)pos * 32 + pi;
    x.xy = attn_rot(x.xy, cs[0]);
    x.zw = attn_rot(x.zw, cs[1]);
  }
  if (a.actq) {
    float m = max(max(fabs(x.x), fabs(x.y)), max(fabs(x.z), fabs(x.w)));
    m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2)); m = max(m, simd_shuffle_xor(m, 4));
    int e = attn_log2_ceil(max(m, 1e-4f) * (1.0f / 448.0f));
    float s = attn_pow2(e), is = attn_pow2(-e);
    x = float4(attn_fp8_q(x.x, s, is), attn_fp8_q(x.y, s, is), attn_fp8_q(x.z, s, is), attn_fp8_q(x.w, s, is));
  }
  int row = a.ring ? (pos % ATTN_W) : int(i);
  ((device half4*)(dst + (size_t)row * ATTN_D))[t] = half4(x);
}

struct AttnCompArgs {
  int start;   // absolute position of input row 0
  int n;       // input rows (tokens)
  int ratio;   // 1 or 2 (any <= 4)
  int j0;      // first compressed row produced by this dispatch (= start / ratio)
  int actq;
};

// One threadgroup (256 threads) per completed group j = j0 + tg. Thread t owns channels 2t, 2t+1.
// state layout (f32): kv[ratio][512] then score[ratio][512].
kernel void attn_compress(device const float* ckv_in [[buffer(0)]], device const float* cgate_in [[buffer(1)]],
                          device const float* state [[buffer(2)]], device const float* cnorm [[buffer(3)]],
                          device const ushort* wk [[buffer(4)]], device const float* knorm [[buffer(5)]],
                          device const float2* tab [[buffer(6)]], device half* ckv [[buffer(7)]],
                          device half* ikey [[buffer(8)]], constant AttnCompArgs& a [[buffer(9)]],
                          uint g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup float lat[ATTN_D];
  threadgroup float kk[ATTN_ID];
  threadgroup float red[8];
  const int r = a.ratio;
  const int j = a.j0 + int(g);
  float2 v;
  if (r == 1) {
    v = ((device const float2*)(ckv_in + (size_t)(j - a.start) * ATTN_D))[t];
  } else {
    float2 kvs[4], scs[4];
    float2 m = float2(-INFINITY);
    for (int q = 0; q < r; q++) {
      int p = j * r + q;
      if (p < a.start) {
        kvs[q] = ((device const float2*)(state + (size_t)(p % r) * ATTN_D))[t];
        scs[q] = ((device const float2*)(state + (size_t)(r + p % r) * ATTN_D))[t];
      } else {
        kvs[q] = ((device const float2*)(ckv_in + (size_t)(p - a.start) * ATTN_D))[t];
        scs[q] = ((device const float2*)(cgate_in + (size_t)(p - a.start) * ATTN_D))[t];
      }
      m = max(m, scs[q]);
    }
    float2 e[4], sum = 0;
    for (int q = 0; q < r; q++) { e[q] = float2(precise::exp(scs[q].x - m.x), precise::exp(scs[q].y - m.y)); sum += e[q]; }
    v = 0;
    for (int q = 0; q < r; q++) v += kvs[q] * float2(precise::divide(e[q].x, sum.x), precise::divide(e[q].y, sum.y));
  }
  // rmsnorm(512)
  float ss = simd_sum(dot(v, v));
  if (ln == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss = 0;
  for (int i = 0; i < 8; i++) ss += red[i];
  float rr = rsqrt(ss * (1.0f / ATTN_D) + 1e-20f);
  v = ((device const float2*)cnorm)[t] * (v * rr);
  lat[2 * t] = v.x; lat[2 * t + 1] = v.y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int rpos = j * r;
  device const float2* cs = tab + (size_t)rpos * 32;
  // index key: k[o] = wk[o] . lat, simdgroup sg computes outputs sg*16 .. sg*16+15, lanes split 512 dims (16 each)
  {
    float l16[16];
    for (int i = 0; i < 16; i++) l16[i] = lat[ln * 16 + i];
    for (int oo = 0; oo < 16; oo++) {
      int o = sg * 16 + oo;
      device const ushort* wr = wk + (size_t)o * ATTN_D + ln * 16;
      float acc = 0;
      for (int i = 0; i < 16; i++) acc += as_type<float>(uint(wr[i]) << 16) * l16[i];
      acc = simd_sum(acc);
      if (ln == 0) kk[o] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {  // k_norm(128) + rope(last 64) + fp4 e8m0/32 -> ikey[j]; lane owns dims 4ln..4ln+3
    float4 k = float4(kk[4 * ln], kk[4 * ln + 1], kk[4 * ln + 2], kk[4 * ln + 3]);
    float s2 = simd_sum(dot(k, k));
    float kr = rsqrt(s2 * (1.0f / ATTN_ID) + 1e-20f);
    k = ((device const float4*)knorm)[ln] * (k * kr);
    if (ln >= 16) {
      int pi = 2 * (ln - 16);
      k.xy = attn_rot(k.xy, cs[pi]);
      k.zw = attn_rot(k.zw, cs[pi + 1]);
    }
    if (a.actq) {
      float m = max(max(fabs(k.x), fabs(k.y)), max(fabs(k.z), fabs(k.w)));
      m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2)); m = max(m, simd_shuffle_xor(m, 4));
      int e = attn_log2_ceil(max(m, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
      float s = attn_pow2(e), is = attn_pow2(-e);
      k = float4(attn_fp4e8_q(k.x, s, is), attn_fp4e8_q(k.y, s, is), attn_fp4e8_q(k.z, s, is), attn_fp4e8_q(k.w, s, is));
    }
    ((device half4*)(ikey + (size_t)j * ATTN_ID))[ln] = half4(k);
  }
  // latent: rope(last 64) + fp4 e4m3/16 -> ckv[j]
  if (t >= 224) v = attn_rot(v, cs[t - 224]);
  if (a.actq) {
    float m = max(fabs(v.x), fabs(v.y));
    m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2)); m = max(m, simd_shuffle_xor(m, 4));
    float s = attn_fp4e4_scale(m);
    v = float2(attn_fp4e4_q(v.x, s), attn_fp4e4_q(v.y, s));
  }
  ((device half2*)(ckv + (size_t)j * ATTN_D))[t] = half2(v);
}

// Store the trailing (incomplete-group) tokens of a chunk into the compressor state. grid (count), 128 threads.
struct AttnStateArgs { int start; int first; int ratio; };  // first = absolute position of the first trailing token
kernel void attn_compress_state(device const float* ckv_in [[buffer(0)]], device const float* cgate_in [[buffer(1)]],
                                device float* state [[buffer(2)]], constant AttnStateArgs& a [[buffer(3)]],
                                uint g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  int p = a.first + int(g), slot = p % a.ratio, i = p - a.start;
  ((device float4*)(state + (size_t)slot * ATTN_D))[t] = ((device const float4*)(ckv_in + (size_t)i * ATTN_D))[t];
  ((device float4*)(state + (size_t)(a.ratio + slot) * ATTN_D))[t] =
      ((device const float4*)(cgate_in + (size_t)i * ATTN_D))[t];
}

// Prefill: copy the last min(n,128) chunk rows into the window ring. grid (rows), 64 threads x 8 halves.
struct AttnCommitArgs { int start; int first; };  // first = first chunk row to copy
kernel void attn_window_commit(device const half* chunk [[buffer(0)]], device half* ring [[buffer(1)]],
                               constant AttnCommitArgs& a [[buffer(2)]], uint g [[threadgroup_position_in_grid]],
                               ushort t [[thread_index_in_threadgroup]]) {
  int i = a.first + int(g), slot = (a.start + i) % ATTN_W;
  ((device uint4*)(ring + (size_t)slot * ATTN_D))[t] = ((device const uint4*)(chunk + (size_t)i * ATTN_D))[t];
}

// Standalone quant-dequant simulations (in place, f32), for tests and for other modules (e.g. fp8 sim of the attention
// output before wo_a). grid (n_blocks/8 rounded up), 256 threads; mode 0 = fp8 e4m3/ue8m0 block 32,
// 1 = fp4 e2m1/ue8m0 block 32, 2 = fp4 e2m1/e4m3 block 16. Thread owns 4 consecutive values.
struct AttnQuantArgs { uint n; int mode; };
kernel void attn_quant_sim(device float* x [[buffer(0)]], constant AttnQuantArgs& a [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
  uint base = gid * 4;
  bool live = base < a.n;
  float4 v = live ? ((device float4*)x)[gid] : float4(0);
  float m = max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w)));
  m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2));
  if (a.mode != 2) m = max(m, simd_shuffle_xor(m, 4));
  if (a.mode == 0) {
    int e = attn_log2_ceil(max(m, 1e-4f) * (1.0f / 448.0f));
    float s = attn_pow2(e), is = attn_pow2(-e);
    v = float4(attn_fp8_q(v.x, s, is), attn_fp8_q(v.y, s, is), attn_fp8_q(v.z, s, is), attn_fp8_q(v.w, s, is));
  } else if (a.mode == 1) {
    int e = attn_log2_ceil(max(m, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    float s = attn_pow2(e), is = attn_pow2(-e);
    v = float4(attn_fp4e8_q(v.x, s, is), attn_fp4e8_q(v.y, s, is), attn_fp4e8_q(v.z, s, is), attn_fp4e8_q(v.w, s, is));
  } else {
    float s = attn_fp4e4_scale(m);
    v = float4(attn_fp4e4_q(v.x, s), attn_fp4e4_q(v.y, s), attn_fp4e4_q(v.z, s), attn_fp4e4_q(v.w, s));
  }
  if (live) ((device float4*)x)[gid] = v;
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_probe.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_probe.metal
// Throughput probes (bench/attn_mmprobe): simdgroup_matrix vs Metal 4 tensor-op matmul2d for the indexer shape
// (rows x 128) . (cols x 128)^T. Not used by the engine.
#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_probe.metal"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// 4 simdgroups, 64x64 output tile, K = 128 read straight from device memory.
kernel void attn_probe_sg(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                          device float* out [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  const uint r0 = tg.y * 64 + (sg >> 1) * 32, c0 = tg.x * 64 + (sg & 1) * 32;
  for (int k = 0; k < 128; k += 8) {
    simdgroup_half8x8 a[4], b[4];
    for (int i = 0; i < 4; i++) simdgroup_load(a[i], A + (r0 + i * 8) * 128 + k, 128);
    for (int j = 0; j < 4; j++) simdgroup_load(b[j], B + (c0 + j * 8) * 128 + k, 128, ulong2(0, 0), true);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  float s = 0;
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { auto e = acc[i][j].thread_elements(); s += max(e[0], 0.f) + max(e[1], 0.f); }
  s = simd_sum(s);
  if (ln == 0) out[(tg.y * 1024 + tg.x) * 4 + sg] = s;
}

template <int TM, int TN>
inline void probe_mpp(device half* A, device half* B, device float* out, uint M, uint N, uint2 tg) {
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(128, M));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(128, N));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 128, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.template slice<128, TM>(0, tg.y * TM);
  auto mB = tB.template slice<128, TN>(0, tg.x * TN);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mB, cT);
  float s = 0;
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) s += max(cT[i], 0.f);
  s = simd_sum(s);
  out[tg.y * 1024 + tg.x] = s;
}
kernel void attn_probe_mpp64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* out [[buffer(2)]],
                             constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
  probe_mpp<64, 64>(A, B, out, M, N, tg);
}
kernel void attn_probe_mpp128(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* out [[buffer(2)]],
                              constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
  probe_mpp<128, 64>(A, B, out, M, N, tg);
}

// streaming read bandwidth probe: each thread reads float4s, grid-stride
kernel void attn_probe_bw(device const float4* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                          uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  float4 s = 0;
  for (uint i = gid; i < n4; i += gsz) s += a[i];
  if (s.x == 1234.5f) out[0] = s.y + s.z + s.w;
}

// lane layout probe: load an 8x8 matrix whose element (r,c) = r*8+c, dump each lane's two thread_elements.
kernel void attn_probe_layout(device float* out [[buffer(0)]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup float m[64];
  for (int i = ln; i < 64; i += 32) m[i] = i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 a; simdgroup_load(a, m, 8);
  auto e = a.thread_elements();
  out[ln * 2] = e[0]; out[ln * 2 + 1] = e[1];
}

// cooperative destination layout probe: out[t * 256 + i] = row * 4096 + col (or -1), out[128*256 + t] = capacity
template <int TM, int TN>
inline void probe_coop_layout(device half* A, device half* B, device int* out, ushort t) {
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(128, TM));
  auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(128, TN));
  constexpr auto desc = matmul2d_descriptor(TM, TN, 128, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    int v = -1;
    if (cT.is_valid_element(i)) { auto id = cT.get_multidimensional_index(i); v = id[1] * 4096 + id[0]; }
    if (i < 256) out[t * 256 + i] = v;
  }
  out[128 * 256 + t] = cT.get_capacity();
}
kernel void attn_probe_coop_64x128(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device int* out [[buffer(2)]],
                                   ushort t [[thread_index_in_threadgroup]]) { probe_coop_layout<64, 128>(A, B, out, t); }
kernel void attn_probe_coop_64x64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device int* out [[buffer(2)]],
                                  ushort t [[thread_index_in_threadgroup]]) { probe_coop_layout<64, 64>(A, B, out, t); }
kernel void attn_probe_coop_128x64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device int* out [[buffer(2)]],
                                   ushort t [[thread_index_in_threadgroup]]) { probe_coop_layout<128, 64>(A, B, out, t); }
// 8-simdgroup NN 64x128 (K=64) destination layout: out[t * 256 + i] = row * 4096 + col, out[256*256 + t] = capacity
kernel void attn_probe_coop8_64x128(device int* out [[buffer(0)]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half X[64 * 128];
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(128, 64));
  constexpr auto desc = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    int v = -1;
    if (cT.is_valid_element(i)) { auto id = cT.get_multidimensional_index(i); v = id[1] * 4096 + id[0]; }
    if (i < 256) out[t * 256 + i] = v;
  }
  out[256 * 256 + t] = cT.get_capacity();
}

// cache-capacity probe: every thread re-reads its grid-stride share of a W-byte buffer `reps` times
kernel void attn_probe_bw_rep(device const float4* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint2& nr [[buffer(2)]],
                              uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  float4 s = 0;
  for (uint r = 0; r < nr.y; r++)
    for (uint i = gid; i < nr.x; i += gsz) s += a[(i + r * 4099u) % nr.x];
  if (s.x == 1234.5f) out[0] = s.y + s.z + s.w;
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse.metal
// Attention module: sparse attention with per-head sink over window rows (ring / prefill chunk) + top-k compressed
// rows, single shared K==V latent (512 dims), 64 heads. simdgroup-matrix flash-style kernel.
#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse.metal"

struct AttnSparseArgs {
  int start;           // absolute position of query 0
  int nq;
  int chunk_start;     // window positions >= chunk_start come from chunk rows (pos - chunk_start), else ring[pos%128]
  int ntop;            // compressed rows listed per query (0 for ratio-0 layers)
  int splits;          // > 1: write partials for attn_sparse_combine (decode split-K); 1: final output
  int rows_per_split;  // multiple of 16
  float scale;         // 512^-0.5
};

constant constexpr int ATTN_LDK = 520;  // padded row stride (halves) of the staged rows

// grid (64/(8*HT) head groups, splits, nq), 32*NSG threads. HT = head tiles of 8 per threadgroup; NSG simdgroups,
// simdgroup s owns latent dims [s*DPS, (s+1)*DPS) with DPS = 512/NSG. Per 16-row chunk: gather rows into threadgroup
// memory, S = Q K^T (partial over each simdgroup's dims, reduced through threadgroup memory), online softmax with
// running max/sum per head, O += P V. Per-thread state: Q fragments HT*DPS/8, O fragments HT*DPS/8.
template <int HT, int NSG>
inline void attn_sparse_impl(device const float* q, device const half* ring, device const half* chunk,
                             device const half* ckv, device const int* topk, device const float* sinks,
                             device const float2* tab, device float* out, device float* part, device float2* ml,
                             constant AttnSparseArgs& a, uint3 tg, ushort t, ushort sg, ushort ln,
                             threadgroup half* Ks, threadgroup float* Sp, threadgroup half* Ps, threadgroup float* As,
                             threadgroup int* vflag, threadgroup float* Ms, threadgroup float* Ls) {
  constexpr int HG = 8 * HT;       // heads per threadgroup
  constexpr int NT = 32 * NSG;     // threads
  constexpr int DPS = 512 / NSG;   // dims per simdgroup
  constexpr int KT = DPS / 8;      // 8-wide tiles per simdgroup
  constexpr int LPH = NT / HG;     // softmax lanes per head
  constexpr int RPL = 16 / LPH;    // rows per lane
  static_assert(LPH <= 16 && RPL >= 1, "softmax mapping");
  constexpr int TPR = NT / 16;     // gather threads per row
  constexpr int U4 = 512 / TPR / 8;  // uint4 per gather thread
  const int h0 = int(tg.x) * HG, split = int(tg.y), qi = int(tg.z);
  const int p = a.start + qi;
  // ---- stage Q (HG heads x 512, rope on dims 448..511) as half, keep in registers as 8x8 fragments ----
  {
    device const float4* qs = (device const float4*)(q + ((size_t)qi * ATTN_H + h0) * ATTN_D);
    for (int i = t; i < HG * 128; i += NT) {
      int hh = i >> 7, c4 = i & 127;
      float4 x = qs[i];
      if (c4 >= 112) {
        int pi = 2 * (c4 - 112);
        x.xy = attn_rot(x.xy, tab[(size_t)p * 32 + pi]);
        x.zw = attn_rot(x.zw, tab[(size_t)p * 32 + pi + 1]);
      }
      *(threadgroup half4*)(Ks + hh * ATTN_LDK + c4 * 4) = half4(x);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qa[HT][KT];
  for (int ht = 0; ht < HT; ht++)
    for (int kk = 0; kk < KT; kk++) simdgroup_load(qa[ht][kk], Ks + ht * 8 * ATTN_LDK + sg * DPS + kk * 8, ATTN_LDK);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 o[HT][KT];
  for (int ht = 0; ht < HT; ht++) for (int dt = 0; dt < KT; dt++) o[ht][dt] = simdgroup_float8x8(0);
  const ushort fm = attn_frag_row(ln), fn = attn_frag_col(ln);
  const int hh = t / LPH, c = (t % LPH) * RPL;  // softmax ownership: head hh, rows c..c+RPL-1 of each chunk
  float m_run = -INFINITY, l_run = 0;
  const int R = ATTN_W + a.ntop;
  const int r_begin = split * a.rows_per_split, r_end = min(R, r_begin + a.rows_per_split);
  for (int r0 = r_begin; r0 < r_end; r0 += 16) {
    {  // gather 16 rows: TPR threads per row
      int rr = t / TPR, pt = t % TPR, r = r0 + rr;
      device const half* src = nullptr;
      if (r < r_end) {
        if (r < ATTN_W) {
          int wp = p - (ATTN_W - 1) + r;
          if (wp >= 0) src = wp >= a.chunk_start ? chunk + (size_t)(wp - a.chunk_start) * ATTN_D : ring + (size_t)(wp % ATTN_W) * ATTN_D;
        } else {
          int id = topk[(size_t)qi * a.ntop + (r - ATTN_W)];
          if (id >= 0) src = ckv + (size_t)id * ATTN_D;
        }
      }
      if (pt == 0) vflag[rr] = src != nullptr;
      threadgroup uint4* dst = (threadgroup uint4*)(Ks + rr * ATTN_LDK + pt * U4 * 8);
      if (src) {
        device const uint4* s4 = (device const uint4*)(src + pt * U4 * 8);
        for (int j = 0; j < U4; j++) dst[j] = s4[j];
      } else {
        for (int j = 0; j < U4; j++) dst[j] = uint4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int any = 0;
    for (int i = 0; i < 16; i++) any |= vflag[i];
    if (any) {
      simdgroup_float8x8 sacc[HT][2];
      for (int ht = 0; ht < HT; ht++) for (int rt = 0; rt < 2; rt++) sacc[ht][rt] = simdgroup_float8x8(0);
      for (int kk = 0; kk < KT; kk++)
        for (int rt = 0; rt < 2; rt++) {
          simdgroup_half8x8 kb;
          simdgroup_load(kb, Ks + rt * 8 * ATTN_LDK + sg * DPS + kk * 8, ATTN_LDK, ulong2(0, 0), true);
          for (int ht = 0; ht < HT; ht++) simdgroup_multiply_accumulate(sacc[ht][rt], qa[ht][kk], kb, sacc[ht][rt]);
        }
      for (int ht = 0; ht < HT; ht++)
        for (int rt = 0; rt < 2; rt++) simdgroup_store(sacc[ht][rt], Sp + sg * (HG * 16) + ht * 8 * 16 + rt * 8, 16);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float sv[RPL];
      float mx = -INFINITY;
      for (int j = 0; j < RPL; j++) {
        float x = 0;
        for (int g = 0; g < NSG; g++) x += Sp[g * (HG * 16) + hh * 16 + c + j];
        sv[j] = vflag[c + j] ? x * a.scale : -INFINITY;
        mx = max(mx, sv[j]);
      }
      for (int sh = 1; sh < LPH; sh <<= 1) mx = max(mx, simd_shuffle_xor(mx, sh));
      float m_new = max(m_run, mx);
      float alpha = 1, ps = 0;
      float pv[RPL];
      for (int j = 0; j < RPL; j++) pv[j] = 0;
      if (m_new != -INFINITY) {
        alpha = exp(m_run - m_new);
        for (int j = 0; j < RPL; j++) { pv[j] = exp(sv[j] - m_new); ps += pv[j]; }
      }
      for (int sh = 1; sh < LPH; sh <<= 1) ps += simd_shuffle_xor(ps, sh);
      l_run = l_run * alpha + ps;
      m_run = m_new;
      for (int j = 0; j < RPL; j++) Ps[hh * 16 + c + j] = half(pv[j]);
      if ((t % LPH) == 0) As[hh] = alpha;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float al[HT];
      for (int ht = 0; ht < HT; ht++) al[ht] = As[ht * 8 + fm];
      simdgroup_half8x8 pa[HT][2];
      for (int ht = 0; ht < HT; ht++)
        for (int rt = 0; rt < 2; rt++) simdgroup_load(pa[ht][rt], Ps + ht * 8 * 16 + rt * 8, 16);
      for (int dt = 0; dt < KT; dt++) {
        for (int ht = 0; ht < HT; ht++) {
          o[ht][dt].thread_elements()[0] *= al[ht];
          o[ht][dt].thread_elements()[1] *= al[ht];
        }
        for (int rt = 0; rt < 2; rt++) {
          simdgroup_half8x8 vb;
          simdgroup_load(vb, Ks + rt * 8 * ATTN_LDK + sg * DPS + dt * 8, ATTN_LDK);
          for (int ht = 0; ht < HT; ht++) simdgroup_multiply_accumulate(o[ht][dt], pa[ht][rt], vb, o[ht][dt]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((t % LPH) == 0) { Ms[hh] = m_run; Ls[hh] = l_run; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ht = 0; ht < HT; ht++) {
    const int head = ht * 8 + fm, hgl = h0 + head;
    float mh = Ms[head], lh = Ls[head];
    if (a.splits == 1) {
      float sink = sinks[hgl];
      float mm = max(mh, sink);
      float em = exp(mh - mm);
      float f = em / (lh * em + exp(sink - mm));
      device float* dst = out + ((size_t)qi * ATTN_H + hgl) * ATTN_D;
      for (int dt = 0; dt < KT; dt++) {
        int d = sg * DPS + dt * 8 + fn;
        auto e = o[ht][dt].thread_elements();
        float2 v = float2(e[0], e[1]) * f;
        if (d >= ATTN_D - ATTN_RD) v = attn_rot_inv(v, tab[(size_t)p * 32 + (d - (ATTN_D - ATTN_RD)) / 2]);
        *(device float2*)(dst + d) = v;
      }
    } else {
      device float* dst = part + ((size_t)split * ATTN_H + hgl) * ATTN_D;
      for (int dt = 0; dt < KT; dt++) {
        int d = sg * DPS + dt * 8 + fn;
        auto e = o[ht][dt].thread_elements();
        *(device float2*)(dst + d) = float2(e[0], e[1]);
      }
      if (sg == 0 && (ln & 0x9) == 0) ml[split * ATTN_H + hgl] = float2(mh, lh);  // lanes with fn==0
    }
  }
}

#define ATTN_SPARSE_KERNEL(NAME, HT, NSG)                                                                         \
  kernel void NAME(device const float* q [[buffer(0)]], device const half* ring [[buffer(1)]],                     \
                   device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],                   \
                   device const int* topk [[buffer(4)]], device const float* sinks [[buffer(5)]],                  \
                   device const float2* tab [[buffer(6)]], device float* out [[buffer(7)]],                        \
                   device float* part [[buffer(8)]], device float2* ml [[buffer(9)]],                              \
                   constant AttnSparseArgs& a [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],           \
                   ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],         \
                   ushort ln [[thread_index_in_simdgroup]]) {                                                      \
    threadgroup half Ks[16 * ATTN_LDK];                                                                            \
    threadgroup float Sp[NSG * 8 * HT * 16];                                                                       \
    threadgroup half Ps[8 * HT * 16];                                                                              \
    threadgroup float As[8 * HT], Ms[8 * HT], Ls[8 * HT];                                                          \
    threadgroup int vflag[16];                                                                                     \
    attn_sparse_impl<HT, NSG>(q, ring, chunk, ckv, topk, sinks, tab, out, part, ml, a, tg, t, sg, ln, Ks, Sp, Ps,  \
                              As, vflag, Ms, Ls);                                                                  \
  }
ATTN_SPARSE_KERNEL(attn_sparse_h8, 1, 4)    // 8 heads, 128 threads
ATTN_SPARSE_KERNEL(attn_sparse_h16, 2, 8)   // 16 heads, 256 threads (64 dims per simdgroup)

struct AttnCombArgs { int pos; int splits; };
// grid (64 heads), 128 threads x 4 dims: merge split partials, sink, normalize, inverse rope -> out [64][512]
kernel void attn_sparse_combine(device const float* part [[buffer(0)]], device const float2* ml [[buffer(1)]],
                                device const float* sinks [[buffer(2)]], device const float2* tab [[buffer(3)]],
                                device float* out [[buffer(4)]], constant AttnCombArgs& a [[buffer(5)]],
                                uint h [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  float sink = sinks[h];
  float M = sink;
  for (int s = 0; s < a.splits; s++) M = max(M, ml[s * ATTN_H + h].x);
  float L = exp(sink - M);
  float4 acc = 0;
  for (int s = 0; s < a.splits; s++) {
    float2 v = ml[s * ATTN_H + h];
    float w = exp(v.x - M);
    L += w * v.y;
    acc += w * ((device const float4*)(part + ((size_t)s * ATTN_H + h) * ATTN_D))[t];
  }
  float4 o = acc / L;
  if (t >= 112) {
    int pi = 2 * (t - 112);
    o.xy = attn_rot_inv(o.xy, tab[(size_t)a.pos * 32 + pi]);
    o.zw = attn_rot_inv(o.zw, tab[(size_t)a.pos * 32 + pi + 1]);
  }
  ((device float4*)(out + (size_t)h * ATTN_D))[t] = o;
}

// ---- decode: sparse attention + split-K combine in ONE dispatch (MACSEEK_DEC_COMBINE) ----
// Every threadgroup runs attn_sparse_impl (partials as in the split path); the last threadgroup of each head group to
// finish (device atomic counter per head group, reset by that threadgroup) then merges the partials of its heads with
// exactly attn_sparse_combine's arithmetic (thread t of 128 owns dims 4t..4t+3 of each head) -> bit-identical output.
inline bool attn_last_group(device atomic_uint* counter, uint n_groups, ushort tid, threadgroup uint* flag) {
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

#define ATTN_SPARSE_LC_KERNEL(NAME, HT, NSG)                                                                      \
  kernel void NAME(device const float* q [[buffer(0)]], device const half* ring [[buffer(1)]],                     \
                   device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],                   \
                   device const int* topk [[buffer(4)]], device const float* sinks [[buffer(5)]],                  \
                   device const float2* tab [[buffer(6)]], device float* out [[buffer(7)]],                        \
                   device float* part [[buffer(8)]], device float2* ml [[buffer(9)]],                              \
                   constant AttnSparseArgs& a [[buffer(10)]], device atomic_uint* cnt [[buffer(11)]],              \
                   coherent(device) device const float* part_c [[buffer(12)]],                                     \
                   coherent(device) device const float2* ml_c [[buffer(13)]],                                      \
                   uint3 tg [[threadgroup_position_in_grid]],                                                      \
                   ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],         \
                   ushort ln [[thread_index_in_simdgroup]]) {                                                      \
    threadgroup half Ks[16 * ATTN_LDK];                                                                            \
    threadgroup float Sp[NSG * 8 * HT * 16];                                                                       \
    threadgroup half Ps[8 * HT * 16];                                                                              \
    threadgroup float As[8 * HT], Ms[8 * HT], Ls[8 * HT];                                                          \
    threadgroup int vflag[16];                                                                                     \
    threadgroup uint flag;                                                                                         \
    attn_sparse_impl<HT, NSG>(q, ring, chunk, ckv, topk, sinks, tab, out, part, ml, a, tg, t, sg, ln, Ks, Sp, Ps,  \
                              As, vflag, Ms, Ls);                                                                  \
    if (!attn_last_group(cnt + tg.x, uint(a.splits), t, &flag)) return;                                           \
    if (t < 128) {  /* all heads of the group interleaved (independent loads in flight); per head the same    */  \
      constexpr int HG = 8 * HT; /* operations in the same order as attn_sparse_combine                       */  \
      const int hb = int(tg.x) * HG;                                                                               \
      float sink[HG], M[HG], L[HG];                                                                                \
      float4 acc[HG];                                                                                              \
      for (int hh = 0; hh < HG; hh++) { sink[hh] = sinks[hb + hh]; M[hh] = sink[hh]; }                             \
      for (int s = 0; s < a.splits; s++)                                                                           \
        for (int hh = 0; hh < HG; hh++) M[hh] = max(M[hh], ml_c[s * ATTN_H + hb + hh].x);                          \
      for (int hh = 0; hh < HG; hh++) { L[hh] = exp(sink[hh] - M[hh]); acc[hh] = 0; }                              \
      for (int s = 0; s < a.splits; s++)                                                                           \
        for (int hh = 0; hh < HG; hh++) {                                                                          \
          float2 v = ml_c[s * ATTN_H + hb + hh];                                                                   \
          float w = exp(v.x - M[hh]);                                                                              \
          L[hh] += w * v.y;                                                                                        \
          acc[hh] += w * ((coherent(device) device const float4*)(part_c + ((size_t)s * ATTN_H + hb + hh) * ATTN_D))[t]; \
        }                                                                                                          \
      for (int hh = 0; hh < HG; hh++) {                                                                            \
        float4 o = acc[hh] / L[hh];                                                                                \
        if (t >= 112) {                                                                                            \
          int pi = 2 * (t - 112);                                                                                  \
          o.xy = attn_rot_inv(o.xy, tab[(size_t)a.start * 32 + pi]);                                               \
          o.zw = attn_rot_inv(o.zw, tab[(size_t)a.start * 32 + pi + 1]);                                           \
        }                                                                                                          \
        ((device float4*)(out + (size_t)(hb + hh) * ATTN_D))[t] = o;                                               \
      }                                                                                                            \
    }                                                                                                              \
    if (t == 0) atomic_store_explicit(cnt + tg.x, 0u, memory_order_relaxed);                                      \
  }
ATTN_SPARSE_LC_KERNEL(attn_sparse_lc_h8, 1, 4)
ATTN_SPARSE_LC_KERNEL(attn_sparse_lc_h16, 2, 8)
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse_pf.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse_pf.metal
// Attention module: prefill sparse attention on the tensor units (Metal 4 matmul2d), two-pass form per query block:
//   attn_pf_qprep   : Qh[q][h][:] = half(rope(q))
//   attn_pf_scores  : S[q][h][r] = scale * Qh[q][h] . row_r(q)   (64 heads x 64 rows tiles, K chunks of 64 dims)
//   attn_pf_softmax : P[q][h][r] = exp(S - m) / (sum exp(S - m) + exp(sink_h - m))   (half)
//   attn_pf_pv      : out[q][h][d] = inverse_rope(sum_r P[q][h][r] row_r(q)[d])        (64 heads x 64 dims tiles)
// Row r of query q: r < 128 -> window position p-127+r (chunk row if >= chunk_start, else ring), else topk[q][r-128].
#line 8 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse_pf.metal"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct AttnPfArgs {
  int start;        // absolute position of query 0 of this block
  int nq;           // queries in this block
  int q0;           // index of query 0 of this block within the chunk (topk / chunk rows are chunk-indexed)
  int chunk_start;  // absolute position of chunk row 0
  int ntop;         // compressed rows per query (0 or 512)
  int R;            // rows per query = 128 + ntop (multiple of 64)
  float scale;
};

inline device const half* attn_pf_row(constant AttnPfArgs& a, int qb, int r, device const half* ring,
                                      device const half* chunk, device const half* ckv, device const int* topk) {
  const int p = a.start + qb;
  if (r < ATTN_W) {
    int wp = p - (ATTN_W - 1) + r;
    if (wp < 0) return nullptr;
    return wp >= a.chunk_start ? chunk + (size_t)(wp - a.chunk_start) * ATTN_D : ring + (size_t)(wp % ATTN_W) * ATTN_D;
  }
  int id = topk[(size_t)(a.q0 + qb) * a.ntop + (r - ATTN_W)];
  return id >= 0 ? ckv + (size_t)id * ATTN_D : nullptr;
}

// grid (nq), 128 threads: Qh half [nq][64][512] with rope on dims 448..511
kernel void attn_pf_qprep(device const float* q [[buffer(0)]], device const float2* tab [[buffer(1)]],
                          device half* Qh [[buffer(2)]], constant AttnPfArgs& a [[buffer(3)]],
                          uint qb [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  const int p = a.start + int(qb);
  device const float4* src = (device const float4*)(q + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D);
  device half4* dst = (device half4*)(Qh + (size_t)qb * ATTN_H * ATTN_D);
  for (int i = t; i < ATTN_H * 128; i += 128) {
    int c4 = i & 127;
    float4 x = src[i];
    if (c4 >= 112) {
      int pi = 2 * (c4 - 112);
      x.xy = attn_rot(x.xy, tab[(size_t)p * 32 + pi]);
      x.zw = attn_rot(x.zw, tab[(size_t)p * 32 + pi + 1]);
    }
    dst[i] = half4(x);
  }
}

// grid (R/64, nq), 128 threads. S f32 [nq][64][R]
kernel void attn_pf_scores(device half* Qh [[buffer(0)]], device const half* ring [[buffer(1)]],
                           device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                           device const int* topk [[buffer(4)]], device float* S [[buffer(5)]],
                           constant AttnPfArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Kt[64 * 64];
  threadgroup int vf[64];
  const int qb = int(tg.y), r0 = int(tg.x) * 64;
  const int rl = t >> 1, part = t & 1;
  device const half* src = attn_pf_row(a, qb, r0 + rl, ring, chunk, ckv, topk);
  if (part == 0) vf[rl] = src != nullptr;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qh, dextents<int32_t, 2>(ATTN_D, a.nq * ATTN_H));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kt, dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (int d0 = 0; d0 < ATTN_D; d0 += 64) {
    threadgroup uint4* dst = (threadgroup uint4*)(Kt + rl * 64 + part * 32);
    if (src) {
      device const uint4* s4 = (device const uint4*)(src + d0 + part * 32);
      for (int j = 0; j < 4; j++) dst[j] = s4[j];
    } else {
      for (int j = 0; j < 4; j++) dst[j] = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mA = tA.template slice<64, 64>(d0, qb * ATTN_H);
    op.run(mA, tB, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device float* Sq = S + (size_t)qb * ATTN_H * a.R;
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      int col = id[0], h = id[1];
      Sq[(size_t)h * a.R + r0 + col] = vf[col] ? cT[i] * a.scale : -INFINITY;
    }
  }
}

// grid (nq), 1024 threads (32 simdgroups x 2 heads). P half [nq][64][R]
kernel void attn_pf_softmax(device const float* S [[buffer(0)]], device const float* sinks [[buffer(1)]],
                            device half* P [[buffer(2)]], constant AttnPfArgs& a [[buffer(3)]],
                            uint qb [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort ln [[thread_index_in_simdgroup]]) {
  for (int hh = 0; hh < 2; hh++) {
    const int h = sg * 2 + hh;
    device const float* s = S + ((size_t)qb * ATTN_H + h) * a.R;
    device half* pp = P + ((size_t)qb * ATTN_H + h) * a.R;
    float m = sinks[h];
    for (int r = ln; r < a.R; r += 32) m = max(m, s[r]);
    m = simd_max(m);
    float l = 0;
    for (int r = ln; r < a.R; r += 32) l += exp(s[r] - m);
    l = simd_sum(l) + exp(sinks[h] - m);
    const float inv = 1.0f / l;
    for (int r = ln; r < a.R; r += 32) pp[r] = half(exp(s[r] - m) * inv);
  }
}

// grid (8, nq), 128 threads: out[q][h][d0..d0+63] for d0 = 64*tg.x
kernel void attn_pf_pv(device half* P [[buffer(0)]], device const half* ring [[buffer(1)]],
                       device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                       device const int* topk [[buffer(4)]], device const float2* tab [[buffer(5)]],
                       device float* out [[buffer(6)]], constant AttnPfArgs& a [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup half Vt[64 * 64];
  threadgroup float Ot[64 * 64];
  const int qb = int(tg.y), d0 = int(tg.x) * 64;
  const int rl = t >> 1, part = t & 1;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(P, dextents<int32_t, 2>(a.R, a.nq * ATTN_H));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt, dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (int r0 = 0; r0 < a.R; r0 += 64) {
    device const half* src = attn_pf_row(a, qb, r0 + rl, ring, chunk, ckv, topk);
    threadgroup uint4* dst = (threadgroup uint4*)(Vt + rl * 64 + part * 32);
    if (src) {
      device const uint4* s4 = (device const uint4*)(src + d0 + part * 32);
      for (int j = 0; j < 4; j++) dst[j] = s4[j];
    } else {
      for (int j = 0; j < 4; j++) dst[j] = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mA = tA.template slice<64, 64>(r0, qb * ATTN_H);
    op.run(mA, tB, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto id = cT.get_multidimensional_index(i);
      Ot[id[1] * 64 + id[0]] = cT[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int p = a.start + qb;
  device float* oq = out + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D;
  for (int i = t; i < 64 * 32; i += 128) {  // (head, dim pair)
    int h = i >> 5, dp = (i & 31) * 2;
    float2 v = float2(Ot[h * 64 + dp], Ot[h * 64 + dp + 1]);
    int d = d0 + dp;
    if (d >= ATTN_D - ATTN_RD) v = attn_rot_inv(v, tab[(size_t)p * 32 + (d - (ATTN_D - ATTN_RD)) / 2]);
    *(device float2*)(oq + (size_t)h * ATTN_D + d) = v;
  }
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse_pf2.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse_pf2.metal
// Attention module: prefill sparse attention, two-pass on the tensor units with the softmax folded into both passes
// (no f32 score matrix, no separate softmax dispatch):
//   attn_pf2_scores : per (64-row tile t, query): S = scale * Qh . rows^T (64 heads x 64 rows), per-head tile max m_t and
//                     sum l_t = sum exp(S - m_t); writes P_t = exp(S - m_t) (half, <= 1) and ML[q][t][h] = (m_t, l_t)
//   attn_pf2_pv     : per (64-dim slice, query): M = max(sink, m_t), L = exp(sink - M) + sum l_t exp(m_t - M);
//                     out = inverse_rope(sum_t (exp(m_t - M) / L) * (P_t . rows_t))
// Tiles whose 64 rows are all invalid (shallow depth, -1 top-k padding) are skipped in both passes.
// Row gathers are double-buffered (global loads of chunk k+1 are issued before the matmul of chunk k).
// Row r of query q: r < 128 -> window position p-127+r (chunk row if >= chunk_start, else ring), else topk[q][r-128].
#line 11 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/attn_sparse_pf2.metal"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct AttnPf2Args {
  int start;        // absolute position of query 0 of this block
  int nq;           // queries in this block
  int q0;           // index of query 0 of this block within the chunk
  int chunk_start;  // absolute position of chunk row 0
  int ntop;         // compressed rows per query (0 or 512)
  int R;            // rows per query = 128 + ntop (multiple of 64)
  float scale;
};

inline device const half* attn_pf2_rowp(constant AttnPf2Args& a, int qb, int r, device const half* ring,
                                        device const half* chunk, device const half* ckv, device const int* topk) {
  const int p = a.start + qb;
  if (r < ATTN_W) {
    int wp = p - (ATTN_W - 1) + r;
    if (wp < 0) return nullptr;
    return wp >= a.chunk_start ? chunk + (size_t)(wp - a.chunk_start) * ATTN_D : ring + (size_t)(wp % ATTN_W) * ATTN_D;
  }
  int id = topk[(size_t)(a.q0 + qb) * a.ntop + (r - ATTN_W)];
  return id >= 0 ? ckv + (size_t)id * ATTN_D : nullptr;
}

// cooperative destination layout of matmul2d(64, 64, K) on 4 simdgroups (same family as attn_idx_score_pf2; checked at
// init): element i -> row = R + 8*((i>>2)&1) + 32*((i>>4)&1), col = C + (i&3) + 32*((i>>3)&1)
inline int attn_pf2_r(ushort sg, ushort l) { return ((l >> 1) & 3) + 4 * ((l >> 4) & 1) + 16 * (sg >> 1); }
inline int attn_pf2_c(ushort sg, ushort l) { return 4 * (l & 1) + 8 * ((l >> 3) & 1) + 16 * (sg & 1); }
inline int attn_pf2_ri(int i) { return 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1); }
inline int attn_pf2_ci(int i) { return (i & 3) + 32 * ((i >> 3) & 1); }

// grid (R/64, nq), 128 threads. P half [nq][64][R], ML float2 [nq][R/64][64].
// A = Qh (half, roped, from attn_pf_qprep). (An f32-A variant reading the raw q was tried: slower, and the f32-A
// destination layout differs from the one the epilogue assumes.)
inline void attn_pf2_scores_impl(device half* Qh, device const half* ring, device const half* chunk,
                                 device const half* ckv, device const int* topk, device half* P, device float2* ML,
                                 constant AttnPf2Args& a, uint2 tg, ushort t, ushort sg, ushort l,
                                 threadgroup half (&Kt)[2][64 * 64], threadgroup int* vf, threadgroup float (&red)[2][64],
                                 threadgroup int& anyv) {
  const int qb = int(tg.y), r0 = int(tg.x) * 64, nt = a.R / 64;
  const int rl = t >> 1, part = t & 1;
  device const half* src = attn_pf2_rowp(a, qb, r0 + rl, ring, chunk, ckv, topk);
  if (t == 0) anyv = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (part == 0) { vf[rl] = src != nullptr; if (src) anyv = 1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float2* mlq = ML + ((size_t)qb * nt + tg.x) * ATTN_H;
  if (!anyv) {  // whole tile invalid: P never read for it (pass 2 skips tiles with m = -inf)
    if (t < 64) mlq[t] = float2(-INFINITY, 0.0f);
    return;
  }
  const int arow0 = qb * ATTN_H;
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Qh, dextents<int32_t, 2>(ATTN_D, a.nq * ATTN_H));
  auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kt[0], dextents<int32_t, 2>(64, 64));
  auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Kt[1], dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, arow0);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 32; ++i) cT[i] = 0;
  uint4 reg[4];
  auto load = [&](int d0) {
    if (src) { device const uint4* s4 = (device const uint4*)(src + d0 + part * 32); for (int j = 0; j < 4; j++) reg[j] = s4[j]; }
    else for (int j = 0; j < 4; j++) reg[j] = uint4(0);
  };
  load(0);
  for (int k = 0; k < ATTN_D / 64; k++) {
    threadgroup uint4* dst = (threadgroup uint4*)(Kt[k & 1] + rl * 64 + part * 32);
    for (int j = 0; j < 4; j++) dst[j] = reg[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 1 < ATTN_D / 64) load((k + 1) * 64);
    auto mA = tA.template slice<64, 64>(k * 64, arow0);
    if (k & 1) op.run(mA, tB1, cT); else op.run(mA, tB0, cT);
  }
  // per-head max over the tile's 64 rows (cols of the tile)
  const int R0 = attn_pf2_r(sg, l), C0 = attn_pf2_c(sg, l);
  float m4[4];
  #pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    float m = -INFINITY;
    #pragma unroll
    for (int x = 0; x < 8; x++) {
      const int i = (x & 3) | ((hh & 1) << 2) | ((x >> 2) << 3) | ((hh >> 1) << 4);
      const int col = C0 + attn_pf2_ci(i);
      float s = vf[col] ? cT[i] * a.scale : -INFINITY;
      cT[i] = s;
      m = max(m, s);
    }
    m = max(m, simd_shuffle_xor(m, 1));
    m = max(m, simd_shuffle_xor(m, 8));
    m4[hh] = m;
  }
  const bool wr = (l & 9) == 0;
  if (wr) for (int hh = 0; hh < 4; hh++) red[sg & 1][R0 + 8 * (hh & 1) + 32 * (hh >> 1)] = m4[hh];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int hh = 0; hh < 4; hh++) { const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1); m4[hh] = max(red[0][h], red[1][h]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device half* Pq = P + (size_t)qb * ATTN_H * a.R + r0;
  float s4[4];
  #pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1);
    float sum = 0;
    #pragma unroll
    for (int jx = 0; jx < 2; jx++) {
      half4 pv;
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int i = c | ((hh & 1) << 2) | (jx << 3) | ((hh >> 1) << 4);
        float e = exp(cT[i] - m4[hh]);  // -inf -> 0 (m finite: the tile has a valid row)
        sum += e;
        pv[c] = half(e);
      }
      *(device half4*)(Pq + (size_t)h * a.R + C0 + 32 * jx) = pv;
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 8);
    s4[hh] = sum;
  }
  if (wr) for (int hh = 0; hh < 4; hh++) red[sg & 1][R0 + 8 * (hh & 1) + 32 * (hh >> 1)] = s4[hh];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (wr && (sg & 1) == 0)
    for (int hh = 0; hh < 4; hh++) {
      const int h = R0 + 8 * (hh & 1) + 32 * (hh >> 1);
      mlq[h] = float2(m4[hh], red[0][h] + red[1][h]);
    }
}

kernel void attn_pf2_scores(device half* Qh [[buffer(0)]], device const half* ring [[buffer(1)]],
                            device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                            device const int* topk [[buffer(4)]], device half* P [[buffer(5)]],
                            device float2* ML [[buffer(6)]], constant AttnPf2Args& a [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half Kt[2][64 * 64];
  threadgroup int vf[64];
  threadgroup float red[2][64];
  threadgroup int anyv;
  attn_pf2_scores_impl(Qh, ring, chunk, ckv, topk, P, ML, a, tg, t, sg, l, Kt, vf, red, anyv);
}
// grid (8, nq), 128 threads: out[q][h][d0..d0+63] for d0 = 64*tg.x
kernel void attn_pf2_pv(device half* P [[buffer(0)]], device const half* ring [[buffer(1)]],
                        device const half* chunk [[buffer(2)]], device const half* ckv [[buffer(3)]],
                        device const int* topk [[buffer(4)]], device const float2* tab [[buffer(5)]],
                        device float* out [[buffer(6)]], device const float2* ML [[buffer(7)]],
                        device const float* sinks [[buffer(8)]], constant AttnPf2Args& a [[buffer(9)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half Vt[2][64 * 64];
  threadgroup float al[10][64];  // alpha[tile][head] (R <= 640)
  threadgroup int tv[10];         // tile has a valid row
  const int qb = int(tg.y), d0 = int(tg.x) * 64, nt = a.R / 64;
  const int rl = t >> 1, part = t & 1;
  device const float2* mlq = ML + (size_t)qb * nt * ATTN_H;
  if (t < 64) {
    float M = sinks[t];
    for (int k = 0; k < nt; k++) M = max(M, mlq[k * ATTN_H + t].x);
    float L = exp(sinks[t] - M);
    for (int k = 0; k < nt; k++) { float2 v = mlq[k * ATTN_H + t]; L += v.y * exp(v.x - M); }
    const float inv = 1.0f / L;
    for (int k = 0; k < nt; k++) { float mk = mlq[k * ATTN_H + t].x; al[k][t] = mk == -INFINITY ? 0.0f : exp(mk - M) * inv; }
    if (t < nt) tv[t] = mlq[t * ATTN_H].x != -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(P, dextents<int32_t, 2>(a.R, a.nq * ATTN_H));
  auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt[0], dextents<int32_t, 2>(64, 64));
  auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Vt[1], dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = tA.template slice<64, 64>(0, qb * ATTN_H);
  auto acc = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  auto tmp = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(tB0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < 32; ++i) acc[i] = 0;
  const int R0 = attn_pf2_r(sg, l);
  // valid tiles (uniform per tile: all heads share row validity)
  uint4 reg[4];
  auto load = [&](int k) {
    device const half* src = attn_pf2_rowp(a, qb, k * 64 + rl, ring, chunk, ckv, topk);
    if (src) { device const uint4* s4 = (device const uint4*)(src + d0 + part * 32); for (int j = 0; j < 4; j++) reg[j] = s4[j]; }
    else for (int j = 0; j < 4; j++) reg[j] = uint4(0);
  };
  auto next_valid = [&](int k) { while (k < nt && !tv[k]) k++; return k; };
  int k = next_valid(0);
  if (k < nt) load(k);
  int buf = 0;
  while (k < nt) {
    const int kn = next_valid(k + 1);
    threadgroup uint4* dst = (threadgroup uint4*)(Vt[buf] + rl * 64 + part * 32);
    for (int j = 0; j < 4; j++) dst[j] = reg[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kn < nt) load(kn);
    auto mA = tA.template slice<64, 64>(k * 64, qb * ATTN_H);
    if (buf) op.run(mA, tB1, tmp); else op.run(mA, tB0, tmp);
    #pragma unroll
    for (uint16_t i = 0; i < 32; ++i) acc[i] += al[k][R0 + attn_pf2_ri(i)] * tmp[i];
    buf ^= 1;
    k = kn;
  }
  const int C0 = attn_pf2_c(sg, l);
  const int p = a.start + qb;
  device float* oq = out + (size_t)(a.q0 + qb) * ATTN_H * ATTN_D;
  #pragma unroll
  for (int g = 0; g < 8; g++) {  // groups of 4 consecutive dims: i = 4g .. 4g+3
    const int i = 4 * g, h = R0 + attn_pf2_ri(i), d = d0 + C0 + attn_pf2_ci(i);
    float4 v = float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
    if (d >= ATTN_D - ATTN_RD) {
      const int pi = (d - (ATTN_D - ATTN_RD)) / 2;
      v.xy = attn_rot_inv(v.xy, tab[(size_t)p * 32 + pi]);
      v.zw = attn_rot_inv(v.zw, tab[(size_t)p * 32 + pi + 1]);
    }
    *(device float4*)(oq + (size_t)h * ATTN_D + d) = v;
  }
}

// Layout check for the pf2 kernels: out[t] = elements of the (64,64,64) NT multiply-accumulate and NN multiply
// destinations whose (row, col) differ from attn_pf2_r/c/ri/ci. grid 1, 128 threads.
template <bool TR, bool ACC>
inline int attn_pf2_check_one(threadgroup half* X, ushort sg, ushort l) {
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
  auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(64, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, TR, false,
                                            ACC ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  int bad = cT.get_capacity() == 32 ? 0 : 1000;
  for (uint16_t i = 0; i < cT.get_capacity() && i < 32; ++i) {
    if (!cT.is_valid_element(i)) { bad++; continue; }
    auto id = cT.get_multidimensional_index(i);
    if (int(id[1]) != attn_pf2_r(sg, l) + attn_pf2_ri(i) || int(id[0]) != attn_pf2_c(sg, l) + attn_pf2_ci(i)) bad++;
  }
  return bad;
}
kernel void attn_pf2_layout_check(device int* out [[buffer(0)]], ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort l [[thread_index_in_simdgroup]]) {
  threadgroup half X[64 * 64];
  out[sg * 32 + l] = attn_pf2_check_one<true, true>(X, sg, l) + attn_pf2_check_one<false, false>(X, sg, l);
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/debug.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/debug.metal
#include <metal_stdlib>
using namespace metal;
// Diagnostics only: plain float copy (dst[i] = src[i]).
kernel void dbg_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/model.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/model.metal
// Small glue kernels used by the model orchestration (src/model). Prefix m_.
#line 3 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/model.metal"

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
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_bench.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_bench.metal
// Bandwidth probe kernels for bench/membw (q_ prefix: quant module).
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_common.h
// Shared Metal helpers for the quant module (all kernels prefixed q_).

#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_common.h"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_grids.h
// GENERATED by tools/gen_quant_grids.py -- do not edit. Grid VALUES from the public ggml format tables.
// Shared by Metal (src/kernels) and the CPU reference (tests/ref). Each macro is a comma list.


// IQ1_S: 2048 entries, 8 x int8 in {-1,0,1} per entry (byte j = element j)
#define Q_IQ1S_GRID_VALUES \
  0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff, 0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff, \
  0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000, 0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000, \
  0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101, 0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff, \
  0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00, 0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101, \
  0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100, 0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00, \
  0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000, 0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000, \
  0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101, 0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff, \
  0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100, 0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01, \
  0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000, 0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff, \
  0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001, 0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000, \
  0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00, 0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff, \
  0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff, 0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000, \
  0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff, 0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff, \
  0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101, 0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff, \
  0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000, 0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100, \
  0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01, 0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00, \
  0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00, 0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01, \
  0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff, 0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000, \
  0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff, 0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000, \
  0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101, 0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100, \
  0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff, 0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff, \
  0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001, 0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01, \
  0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff, 0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000, \
  0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000, 0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff, \
  0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01, 0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00, \
  0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000, 0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000, \
  0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000, 0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001, \
  0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff, 0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01, \
  0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff, 0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff, \
  0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000, 0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000, \
  0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00, 0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001, \
  0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100, 0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101, \
  0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00, 0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00, \
  0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000, 0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001, \
  0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff, 0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000, \
  0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000, 0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff, \
  0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100, 0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000, \
  0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101, 0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001, \
  0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000, 0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff, \
  0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01, 0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100, \
  0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000, 0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01, \
  0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101, 0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000, \
  0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff, 0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff, \
  0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001, 0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001, \
  0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000, 0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000, \
  0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00, 0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100, \
  0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01, 0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00, \
  0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff, 0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000, \
  0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00, 0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000, \
  0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff, 0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00, \
  0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000, 0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000, \
  0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff, 0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00, \
  0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00, 0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001, \
  0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001, 0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000, \
  0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100, 0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101, \
  0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101, 0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000, \
  0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00, 0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff, \
  0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000, 0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff, \
  0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff, 0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff, \
  0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101, 0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001, \
  0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100, 0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101, \
  0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01, 0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00, \
  0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00, 0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000, \
  0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101, 0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100, \
  0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001, 0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff, \
  0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00, 0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000, \
  0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100, 0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00, \
  0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01, 0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff, \
  0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101, 0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100, \
  0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100, 0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000, \
  0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff, 0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff, \
  0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000, 0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101, \
  0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000, 0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101, \
  0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101, 0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100, \
  0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01, 0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001, \
  0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001, 0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff, \
  0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff, 0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000, \
  0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000, 0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101, \
  0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff, 0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001, \
  0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000, 0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff, \
  0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00, 0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff, \
  0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000, 0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00, \
  0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff, 0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000, \
  0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000, 0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff, \
  0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000, 0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000, \
  0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000, 0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101, \
  0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff, 0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100, \
  0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101, 0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001, \
  0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff, 0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100, \
  0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff, 0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01, \
  0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff, 0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff, \
  0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff, 0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff, \
  0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001, 0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff, \
  0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01, 0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00, \
  0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100, 0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101, \
  0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff, 0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00, \
  0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01, 0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00, \
  0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100, 0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00, \
  0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000, 0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000, \
  0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000, 0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000, \
  0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001, 0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff, \
  0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00, 0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff, \
  0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00, 0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000, \
  0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff, 0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff, \
  0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101, 0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000, \
  0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff, 0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff, \
  0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff, 0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff, \
  0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000, 0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000, \
  0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100, 0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00, \
  0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100, 0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100, \
  0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00, 0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff, \
  0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff, 0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100, \
  0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff, 0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000, \
  0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00, 0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001, \
  0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00, 0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100, \
  0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff, 0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff, \
  0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001, 0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00, \
  0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101, 0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001, \
  0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000, 0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100, \
  0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000, 0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001, \
  0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000, 0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff, \
  0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101, 0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000, \
  0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101, 0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000, \
  0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff, 0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000, \
  0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff, 0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01, \
  0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100, 0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff, \
  0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101, 0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001, \
  0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01, 0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff, \
  0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01, 0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff, \
  0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00, 0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff, \
  0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff, 0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff, \
  0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001, 0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff, \
  0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff, 0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001, \
  0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff, 0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff, \
  0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000, 0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00, \
  0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100, 0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01, \
  0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff, 0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff, \
  0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000, 0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101, \
  0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01, 0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100, \
  0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100, 0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00, \
  0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101, 0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001, \
  0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01, 0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100, \
  0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff, 0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01, \
  0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff, 0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000, \
  0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001, 0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01, \
  0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff, 0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff, \
  0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00, 0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff, \
  0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100, 0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000, \
  0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001, 0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100, \
  0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff, 0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff, \
  0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01, 0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101, \
  0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001, 0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff, \
  0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101, 0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100, \
  0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00, 0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01, \
  0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001, 0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00, \
  0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000, 0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101, \
  0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001, 0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100, \
  0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000, 0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000, \
  0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00, 0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101, \
  0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff, 0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101, \
  0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001, 0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01, \
  0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100, 0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000, \
  0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00, 0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff, \
  0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001, 0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001, \
  0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff, 0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00, \
  0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100, 0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00, \
  0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101, 0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff, \
  0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff, 0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff, \
  0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00, 0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff, \
  0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff, 0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101, \
  0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100, 0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101, \
  0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00, 0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00, \
  0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff, 0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff, \
  0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001, 0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000, \
  0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff, 0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff, \
  0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff, 0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff, \
  0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001, 0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100, \
  0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101, 0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001, \
  0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001, 0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101, \
  0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101, 0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff, \
  0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff, 0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000, \
  0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101, 0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001, \
  0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff, 0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000, \
  0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff, 0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100, \
  0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000, 0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101, \
  0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff, 0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100, \
  0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff, 0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01, \
  0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100, 0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000, \
  0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100, 0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100, \
  0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001, 0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000, \
  0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000, 0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000, \
  0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00, 0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff, \
  0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001, 0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000, \
  0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101, 0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101, \
  0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000, 0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff, \
  0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000, 0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101, \
  0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff, 0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff, \
  0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000, 0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101, \
  0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff, 0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff, \
  0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01, 0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff, \
  0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000, 0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101, \
  0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff, 0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff, \
  0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff, 0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01, \
  0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00, 0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000, \
  0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000, 0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101, \
  0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001, 0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff, \
  0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000, 0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff, \
  0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000, 0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000, \
  0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000, 0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000, \
  0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001, 0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff, \
  0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001, 0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000, \
  0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000, 0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00, \
  0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000, 0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101, \
  0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001, 0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00, \
  0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001, 0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff, \
  0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000, 0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00, \
  0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100, 0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101, \
  0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001, 0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01, \
  0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff, 0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff, \
  0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00, 0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01, \
  0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100, 0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000, \
  0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff, 0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff, \
  0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff, 0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000, \
  0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff, 0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101, \
  0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff, 0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001, \
  0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001, 0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001, \
  0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000, 0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000, \
  0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff, 0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101, \
  0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001, 0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff, \
  0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000, 0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000, \
  0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff, 0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101, \
  0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000, 0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100, \
  0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00, 0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000, \
  0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff, 0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01, \
  0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00, 0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff, \
  0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000, 0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101, \
  0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff, 0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001, \
  0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff, 0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000, \
  0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100, 0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff, \
  0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101, 0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100, \
  0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff, 0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01, \
  0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000, 0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff, \
  0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00, 0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100, \
  0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff, 0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001, \
  0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00, 0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100, \
  0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff, 0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01, \
  0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00, 0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000, \
  0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01, 0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff, \
  0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000, 0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff, \
  0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff, 0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff, \
  0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001, 0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff, \
  0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01, 0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff, \
  0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00, 0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00, \
  0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001, 0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101, \
  0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101, 0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff, \
  0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000, 0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101

// IQ2_XXS: 256 entries, 8 x uint8 in {8,25,43} per entry
#define Q_IQ2XXS_GRID_VALUES \
  0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808, \
  0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819, 0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819, \
  0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b, 0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808, \
  0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b, \
  0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08, \
  0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08, \
  0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808, 0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808, \
  0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819, \
  0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08, 0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908, \
  0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819, 0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808, \
  0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808, 0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908, \
  0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808, 0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08, \
  0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819, \
  0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819, 0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908, \
  0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19, 0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819, \
  0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b, 0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808, \
  0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908, 0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08, \
  0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08, 0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908, \
  0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819, 0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808, \
  0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808, 0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19, \
  0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819, 0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, \
  0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b, 0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08, \
  0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808, 0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908, \
  0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b, 0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819, \
  0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08, \
  0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808, 0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b, \
  0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908, \
  0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808, \
  0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908, 0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b, \
  0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808, 0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b, \
  0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b, 0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808, \
  0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19, 0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908

// ksigns_iq2xs: 7-bit index -> 8 sign bits (bit 7 = parity)
#define Q_KSIGNS_IQ2XS_VALUES \
  0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, \
  0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f, \
  0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, \
  0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f, \
  0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, \
  0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf, \
  0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, \
  0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f, \
  0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, \
  0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf, \
  0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, \
  0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f, \
  0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, \
  0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f, \
  0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, \
  0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff

// derived: 7-bit sign index -> 8 byte masks (0xFF = negate element j)
#define Q_IQ2_SIGNMASK_VALUES \
  0x0000000000000000, 0xff000000000000ff, 0xff0000000000ff00, 0x000000000000ffff, 0xff00000000ff0000, 0x0000000000ff00ff, 0x0000000000ffff00, 0xff00000000ffffff, \
  0xff000000ff000000, 0x00000000ff0000ff, 0x00000000ff00ff00, 0xff000000ff00ffff, 0x00000000ffff0000, 0xff000000ffff00ff, 0xff000000ffffff00, 0x00000000ffffffff, \
  0xff0000ff00000000, 0x000000ff000000ff, 0x000000ff0000ff00, 0xff0000ff0000ffff, 0x000000ff00ff0000, 0xff0000ff00ff00ff, 0xff0000ff00ffff00, 0x000000ff00ffffff, \
  0x000000ffff000000, 0xff0000ffff0000ff, 0xff0000ffff00ff00, 0x000000ffff00ffff, 0xff0000ffffff0000, 0x000000ffffff00ff, 0x000000ffffffff00, 0xff0000ffffffffff, \
  0xff00ff0000000000, 0x0000ff00000000ff, 0x0000ff000000ff00, 0xff00ff000000ffff, 0x0000ff0000ff0000, 0xff00ff0000ff00ff, 0xff00ff0000ffff00, 0x0000ff0000ffffff, \
  0x0000ff00ff000000, 0xff00ff00ff0000ff, 0xff00ff00ff00ff00, 0x0000ff00ff00ffff, 0xff00ff00ffff0000, 0x0000ff00ffff00ff, 0x0000ff00ffffff00, 0xff00ff00ffffffff, \
  0x0000ffff00000000, 0xff00ffff000000ff, 0xff00ffff0000ff00, 0x0000ffff0000ffff, 0xff00ffff00ff0000, 0x0000ffff00ff00ff, 0x0000ffff00ffff00, 0xff00ffff00ffffff, \
  0xff00ffffff000000, 0x0000ffffff0000ff, 0x0000ffffff00ff00, 0xff00ffffff00ffff, 0x0000ffffffff0000, 0xff00ffffffff00ff, 0xff00ffffffffff00, 0x0000ffffffffffff, \
  0xffff000000000000, 0x00ff0000000000ff, 0x00ff00000000ff00, 0xffff00000000ffff, 0x00ff000000ff0000, 0xffff000000ff00ff, 0xffff000000ffff00, 0x00ff000000ffffff, \
  0x00ff0000ff000000, 0xffff0000ff0000ff, 0xffff0000ff00ff00, 0x00ff0000ff00ffff, 0xffff0000ffff0000, 0x00ff0000ffff00ff, 0x00ff0000ffffff00, 0xffff0000ffffffff, \
  0x00ff00ff00000000, 0xffff00ff000000ff, 0xffff00ff0000ff00, 0x00ff00ff0000ffff, 0xffff00ff00ff0000, 0x00ff00ff00ff00ff, 0x00ff00ff00ffff00, 0xffff00ff00ffffff, \
  0xffff00ffff000000, 0x00ff00ffff0000ff, 0x00ff00ffff00ff00, 0xffff00ffff00ffff, 0x00ff00ffffff0000, 0xffff00ffffff00ff, 0xffff00ffffffff00, 0x00ff00ffffffffff, \
  0x00ffff0000000000, 0xffffff00000000ff, 0xffffff000000ff00, 0x00ffff000000ffff, 0xffffff0000ff0000, 0x00ffff0000ff00ff, 0x00ffff0000ffff00, 0xffffff0000ffffff, \
  0xffffff00ff000000, 0x00ffff00ff0000ff, 0x00ffff00ff00ff00, 0xffffff00ff00ffff, 0x00ffff00ffff0000, 0xffffff00ffff00ff, 0xffffff00ffffff00, 0x00ffff00ffffffff, \
  0xffffffff00000000, 0x00ffffff000000ff, 0x00ffffff0000ff00, 0xffffffff0000ffff, 0x00ffffff00ff0000, 0xffffffff00ff00ff, 0xffffffff00ffff00, 0x00ffffff00ffffff, \
  0x00ffffffff000000, 0xffffffffff0000ff, 0xffffffffff00ff00, 0x00ffffffff00ffff, 0xffffffffffff0000, 0x00ffffffffff00ff, 0x00ffffffffffff00, 0xffffffffffffffff

// derived: IQ1_S grid as half values, 4 x uint32 (= 8 halfs) per entry
#define Q_IQ1S_GRID_HALF_VALUES \
  0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, \
  0x00000000, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, \
  0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc000000, 0xbc000000, 0xbc00bc00, 0xbc00bc00, \
  0x00000000, 0xbc000000, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, \
  0xbc003c00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, \
  0x3c003c00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x00000000, 0x0000bc00, 0xbc00bc00, 0xbc00bc00, \
  0xbc000000, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x0000bc00, 0x00000000, 0xbc00bc00, 0xbc00bc00, \
  0x00003c00, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x00000000, 0x00003c00, 0xbc00bc00, 0xbc00bc00, \
  0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, \
  0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, \
  0x00000000, 0x3c000000, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, \
  0xbc003c00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, \
  0x3c003c00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0xbc00bc00, \
  0x00000000, 0xbc00bc00, 0xbc000000, 0xbc00bc00, 0xbc000000, 0xbc000000, 0xbc000000, 0xbc00bc00, \
  0x0000bc00, 0xbc000000, 0xbc000000, 0xbc00bc00, 0x00003c00, 0xbc000000, 0xbc000000, 0xbc00bc00, \
  0x3c000000, 0xbc000000, 0xbc000000, 0xbc00bc00, 0x3c003c00, 0xbc000000, 0xbc000000, 0xbc00bc00, \
  0x00000000, 0xbc003c00, 0xbc000000, 0xbc00bc00, 0xbc000000, 0x0000bc00, 0xbc000000, 0xbc00bc00, \
  0x00003c00, 0x0000bc00, 0xbc000000, 0xbc00bc00, 0x3c000000, 0x0000bc00, 0xbc000000, 0xbc00bc00, \
  0xbc003c00, 0x00000000, 0xbc000000, 0xbc00bc00, 0x00000000, 0x00000000, 0xbc000000, 0xbc00bc00, \
  0x3c003c00, 0x00000000, 0xbc000000, 0xbc00bc00, 0xbc000000, 0x00003c00, 0xbc000000, 0xbc00bc00, \
  0x0000bc00, 0x00003c00, 0xbc000000, 0xbc00bc00, 0x00003c00, 0x00003c00, 0xbc000000, 0xbc00bc00, \
  0x3c00bc00, 0x00003c00, 0xbc000000, 0xbc00bc00, 0x00000000, 0x3c00bc00, 0xbc000000, 0xbc00bc00, \
  0xbc000000, 0x3c000000, 0xbc000000, 0xbc00bc00, 0x0000bc00, 0x3c000000, 0xbc000000, 0xbc00bc00, \
  0x00003c00, 0x3c000000, 0xbc000000, 0xbc00bc00, 0x00000000, 0x3c003c00, 0xbc000000, 0xbc00bc00, \
  0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, \
  0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, \
  0x00000000, 0xbc000000, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, \
  0xbc003c00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, \
  0x3c003c00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, 0x00000000, 0x0000bc00, 0xbc003c00, 0xbc00bc00, \
  0xbc000000, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0xbc00bc00, \
  0x0000bc00, 0x00003c00, 0xbc003c00, 0xbc00bc00, 0x3c000000, 0x00003c00, 0xbc003c00, 0xbc00bc00, \
  0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, \
  0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, \
  0xbc000000, 0x3c000000, 0xbc003c00, 0xbc00bc00, 0x00000000, 0x3c000000, 0xbc003c00, 0xbc00bc00, \
  0x3c000000, 0x3c000000, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, \
  0xbc003c00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, \
  0x3c003c00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0xbc000000, 0xbc000000, 0x0000bc00, 0xbc00bc00, \
  0x0000bc00, 0xbc000000, 0x0000bc00, 0xbc00bc00, 0x00003c00, 0xbc000000, 0x0000bc00, 0xbc00bc00, \
  0x00000000, 0xbc003c00, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0x0000bc00, 0x0000bc00, 0xbc00bc00, \
  0x3c000000, 0x0000bc00, 0x0000bc00, 0xbc00bc00, 0x00000000, 0x00000000, 0x0000bc00, 0xbc00bc00, \
  0x3c003c00, 0x00000000, 0x0000bc00, 0xbc00bc00, 0x0000bc00, 0x00003c00, 0x0000bc00, 0xbc00bc00, \
  0x00000000, 0x00003c00, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x0000bc00, 0xbc00bc00, \
  0x3c000000, 0x3c000000, 0x0000bc00, 0xbc00bc00, 0x00000000, 0x3c003c00, 0x0000bc00, 0xbc00bc00, \
  0xbc000000, 0xbc00bc00, 0x00000000, 0xbc00bc00, 0x0000bc00, 0xbc00bc00, 0x00000000, 0xbc00bc00, \
  0x00000000, 0xbc00bc00, 0x00000000, 0xbc00bc00, 0x00003c00, 0xbc00bc00, 0x00000000, 0xbc00bc00, \
  0x00000000, 0xbc000000, 0x00000000, 0xbc00bc00, 0x3c00bc00, 0xbc000000, 0x00000000, 0xbc00bc00, \
  0x3c003c00, 0xbc000000, 0x00000000, 0xbc00bc00, 0x3c000000, 0xbc003c00, 0x00000000, 0xbc00bc00, \
  0xbc00bc00, 0x0000bc00, 0x00000000, 0xbc00bc00, 0x00000000, 0x0000bc00, 0x00000000, 0xbc00bc00, \
  0x3c003c00, 0x0000bc00, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x00000000, 0x00000000, 0xbc00bc00, \
  0xbc000000, 0x00000000, 0x00000000, 0xbc00bc00, 0x0000bc00, 0x00000000, 0x00000000, 0xbc00bc00, \
  0x00000000, 0x00000000, 0x00000000, 0xbc00bc00, 0x00003c00, 0x00000000, 0x00000000, 0xbc00bc00, \
  0x3c000000, 0x00000000, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x00003c00, 0x00000000, 0xbc00bc00, \
  0xbc003c00, 0x00003c00, 0x00000000, 0xbc00bc00, 0x00000000, 0x00003c00, 0x00000000, 0xbc00bc00, \
  0x3c00bc00, 0x00003c00, 0x00000000, 0xbc00bc00, 0x3c003c00, 0x00003c00, 0x00000000, 0xbc00bc00, \
  0xbc000000, 0x3c00bc00, 0x00000000, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x00000000, 0xbc00bc00, \
  0x00000000, 0x3c000000, 0x00000000, 0xbc00bc00, 0x3c00bc00, 0x3c000000, 0x00000000, 0xbc00bc00, \
  0x3c003c00, 0x3c000000, 0x00000000, 0xbc00bc00, 0xbc000000, 0x3c003c00, 0x00000000, 0xbc00bc00, \
  0x0000bc00, 0x3c003c00, 0x00000000, 0xbc00bc00, 0x00000000, 0x3c003c00, 0x00000000, 0xbc00bc00, \
  0x00003c00, 0x3c003c00, 0x00000000, 0xbc00bc00, 0x3c000000, 0x3c003c00, 0x00000000, 0xbc00bc00, \
  0x0000bc00, 0xbc000000, 0x00003c00, 0xbc00bc00, 0x3c000000, 0xbc000000, 0x00003c00, 0xbc00bc00, \
  0xbc000000, 0x0000bc00, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x0000bc00, 0x00003c00, 0xbc00bc00, \
  0xbc00bc00, 0x00000000, 0x00003c00, 0xbc00bc00, 0xbc003c00, 0x00000000, 0x00003c00, 0xbc00bc00, \
  0x00000000, 0x00000000, 0x00003c00, 0xbc00bc00, 0x3c00bc00, 0x00000000, 0x00003c00, 0xbc00bc00, \
  0xbc00bc00, 0x00003c00, 0x00003c00, 0xbc00bc00, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc00bc00, \
  0x00003c00, 0x00003c00, 0x00003c00, 0xbc00bc00, 0x3c000000, 0x00003c00, 0x00003c00, 0xbc00bc00, \
  0x00000000, 0x3c00bc00, 0x00003c00, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x00003c00, 0xbc00bc00, \
  0x0000bc00, 0x3c000000, 0x00003c00, 0xbc00bc00, 0x3c000000, 0x3c000000, 0x00003c00, 0xbc00bc00, \
  0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, \
  0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, \
  0x00000000, 0xbc000000, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, \
  0xbc003c00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, \
  0x3c003c00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x00000000, 0x0000bc00, 0x3c00bc00, 0xbc00bc00, \
  0xbc000000, 0x00000000, 0x3c00bc00, 0xbc00bc00, 0x00003c00, 0x00000000, 0x3c00bc00, 0xbc00bc00, \
  0x00000000, 0x00003c00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, \
  0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, \
  0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0x00000000, 0x3c000000, 0x3c00bc00, 0xbc00bc00, \
  0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, \
  0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, \
  0x00000000, 0xbc00bc00, 0x3c000000, 0xbc00bc00, 0xbc000000, 0xbc000000, 0x3c000000, 0xbc00bc00, \
  0x0000bc00, 0xbc000000, 0x3c000000, 0xbc00bc00, 0x3c000000, 0xbc000000, 0x3c000000, 0xbc00bc00, \
  0x0000bc00, 0xbc003c00, 0x3c000000, 0xbc00bc00, 0x00000000, 0xbc003c00, 0x3c000000, 0xbc00bc00, \
  0xbc000000, 0x0000bc00, 0x3c000000, 0xbc00bc00, 0xbc00bc00, 0x00000000, 0x3c000000, 0xbc00bc00, \
  0xbc000000, 0x00000000, 0x3c000000, 0xbc00bc00, 0x00000000, 0x00000000, 0x3c000000, 0xbc00bc00, \
  0xbc000000, 0x00003c00, 0x3c000000, 0xbc00bc00, 0x0000bc00, 0x00003c00, 0x3c000000, 0xbc00bc00, \
  0x3c000000, 0x00003c00, 0x3c000000, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x3c000000, 0xbc00bc00, \
  0x0000bc00, 0x3c000000, 0x3c000000, 0xbc00bc00, 0x00003c00, 0x3c000000, 0x3c000000, 0xbc00bc00, \
  0x3c000000, 0x3c000000, 0x3c000000, 0xbc00bc00, 0x00000000, 0x3c003c00, 0x3c000000, 0xbc00bc00, \
  0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, \
  0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, \
  0x00000000, 0xbc000000, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, \
  0xbc003c00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, \
  0x3c003c00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0x00000000, 0x0000bc00, 0x3c003c00, 0xbc00bc00, \
  0xbc000000, 0x00000000, 0x3c003c00, 0xbc00bc00, 0x3c000000, 0x00000000, 0x3c003c00, 0xbc00bc00, \
  0xbc000000, 0x00003c00, 0x3c003c00, 0xbc00bc00, 0x00000000, 0x00003c00, 0x3c003c00, 0xbc00bc00, \
  0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, \
  0x00000000, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, \
  0x3c003c00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0x00000000, 0x3c000000, 0x3c003c00, 0xbc00bc00, \
  0xbc00bc00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, \
  0x3c00bc00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, \
  0xbc00bc00, 0xbc000000, 0xbc00bc00, 0xbc000000, 0xbc000000, 0xbc000000, 0xbc00bc00, 0xbc000000, \
  0x0000bc00, 0xbc000000, 0xbc00bc00, 0xbc000000, 0x3c000000, 0xbc000000, 0xbc00bc00, 0xbc000000, \
  0x0000bc00, 0xbc003c00, 0xbc00bc00, 0xbc000000, 0x00000000, 0xbc003c00, 0xbc00bc00, 0xbc000000, \
  0xbc000000, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0x0000bc00, 0x0000bc00, 0xbc00bc00, 0xbc000000, \
  0xbc00bc00, 0x00000000, 0xbc00bc00, 0xbc000000, 0x00000000, 0x00000000, 0xbc00bc00, 0xbc000000, \
  0x3c00bc00, 0x00000000, 0xbc00bc00, 0xbc000000, 0xbc000000, 0x00003c00, 0xbc00bc00, 0xbc000000, \
  0x0000bc00, 0x00003c00, 0xbc00bc00, 0xbc000000, 0x00000000, 0x00003c00, 0xbc00bc00, 0xbc000000, \
  0x3c000000, 0x00003c00, 0xbc00bc00, 0xbc000000, 0xbc000000, 0x3c000000, 0xbc00bc00, 0xbc000000, \
  0x0000bc00, 0x3c000000, 0xbc00bc00, 0xbc000000, 0x00003c00, 0x3c000000, 0xbc00bc00, 0xbc000000, \
  0xbc000000, 0x3c003c00, 0xbc00bc00, 0xbc000000, 0x00000000, 0x3c003c00, 0xbc00bc00, 0xbc000000, \
  0xbc000000, 0xbc00bc00, 0xbc000000, 0xbc000000, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0xbc000000, \
  0x00003c00, 0xbc00bc00, 0xbc000000, 0xbc000000, 0x3c000000, 0xbc00bc00, 0xbc000000, 0xbc000000, \
  0xbc00bc00, 0xbc000000, 0xbc000000, 0xbc000000, 0xbc003c00, 0xbc000000, 0xbc000000, 0xbc000000, \
  0x00000000, 0xbc000000, 0xbc000000, 0xbc000000, 0x3c00bc00, 0xbc000000, 0xbc000000, 0xbc000000, \
  0xbc000000, 0xbc003c00, 0xbc000000, 0xbc000000, 0x0000bc00, 0xbc003c00, 0xbc000000, 0xbc000000, \
  0x3c000000, 0xbc003c00, 0xbc000000, 0xbc000000, 0x00000000, 0x0000bc00, 0xbc000000, 0xbc000000, \
  0x3c003c00, 0x0000bc00, 0xbc000000, 0xbc000000, 0xbc00bc00, 0x00000000, 0xbc000000, 0xbc000000, \
  0xbc000000, 0x00000000, 0xbc000000, 0xbc000000, 0xbc003c00, 0x00000000, 0xbc000000, 0xbc000000, \
  0x0000bc00, 0x00000000, 0xbc000000, 0xbc000000, 0x00000000, 0x00000000, 0xbc000000, 0xbc000000, \
  0x00003c00, 0x00000000, 0xbc000000, 0xbc000000, 0x3c000000, 0x00000000, 0xbc000000, 0xbc000000, \
  0xbc00bc00, 0x00003c00, 0xbc000000, 0xbc000000, 0x00000000, 0x00003c00, 0xbc000000, 0xbc000000, \
  0x0000bc00, 0x3c00bc00, 0xbc000000, 0xbc000000, 0xbc003c00, 0x3c000000, 0xbc000000, 0xbc000000, \
  0x00000000, 0x3c000000, 0xbc000000, 0xbc000000, 0xbc000000, 0x3c003c00, 0xbc000000, 0xbc000000, \
  0x0000bc00, 0x3c003c00, 0xbc000000, 0xbc000000, 0xbc000000, 0xbc000000, 0xbc003c00, 0xbc000000, \
  0x0000bc00, 0xbc000000, 0xbc003c00, 0xbc000000, 0x00003c00, 0xbc000000, 0xbc003c00, 0xbc000000, \
  0x00000000, 0xbc003c00, 0xbc003c00, 0xbc000000, 0xbc00bc00, 0x0000bc00, 0xbc003c00, 0xbc000000, \
  0x00003c00, 0x0000bc00, 0xbc003c00, 0xbc000000, 0x3c000000, 0x0000bc00, 0xbc003c00, 0xbc000000, \
  0xbc003c00, 0x00000000, 0xbc003c00, 0xbc000000, 0x00000000, 0x00000000, 0xbc003c00, 0xbc000000, \
  0x3c00bc00, 0x00000000, 0xbc003c00, 0xbc000000, 0x3c003c00, 0x00000000, 0xbc003c00, 0xbc000000, \
  0x0000bc00, 0x00003c00, 0xbc003c00, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc003c00, 0xbc000000, \
  0x00000000, 0x3c00bc00, 0xbc003c00, 0xbc000000, 0xbc000000, 0x3c000000, 0xbc003c00, 0xbc000000, \
  0x0000bc00, 0x3c000000, 0xbc003c00, 0xbc000000, 0x00003c00, 0x3c000000, 0xbc003c00, 0xbc000000, \
  0x00000000, 0x3c003c00, 0xbc003c00, 0xbc000000, 0xbc000000, 0xbc00bc00, 0x0000bc00, 0xbc000000, \
  0x00003c00, 0xbc00bc00, 0x0000bc00, 0xbc000000, 0x3c000000, 0xbc00bc00, 0x0000bc00, 0xbc000000, \
  0x0000bc00, 0xbc000000, 0x0000bc00, 0xbc000000, 0x00000000, 0xbc000000, 0x0000bc00, 0xbc000000, \
  0x3c00bc00, 0xbc000000, 0x0000bc00, 0xbc000000, 0x3c000000, 0xbc000000, 0x0000bc00, 0xbc000000, \
  0xbc000000, 0xbc003c00, 0x0000bc00, 0xbc000000, 0x00003c00, 0xbc003c00, 0x0000bc00, 0xbc000000, \
  0xbc000000, 0x0000bc00, 0x0000bc00, 0xbc000000, 0x00000000, 0x0000bc00, 0x0000bc00, 0xbc000000, \
  0x00003c00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0xbc000000, \
  0x3c003c00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0xbc000000, 0x00000000, 0x0000bc00, 0xbc000000, \
  0x0000bc00, 0x00000000, 0x0000bc00, 0xbc000000, 0x00000000, 0x00000000, 0x0000bc00, 0xbc000000, \
  0x00003c00, 0x00000000, 0x0000bc00, 0xbc000000, 0x3c000000, 0x00000000, 0x0000bc00, 0xbc000000, \
  0xbc003c00, 0x00003c00, 0x0000bc00, 0xbc000000, 0x00000000, 0x00003c00, 0x0000bc00, 0xbc000000, \
  0x3c00bc00, 0x00003c00, 0x0000bc00, 0xbc000000, 0x0000bc00, 0x3c00bc00, 0x0000bc00, 0xbc000000, \
  0x3c000000, 0x3c00bc00, 0x0000bc00, 0xbc000000, 0xbc00bc00, 0x3c000000, 0x0000bc00, 0xbc000000, \
  0x0000bc00, 0x3c000000, 0x0000bc00, 0xbc000000, 0x00000000, 0x3c000000, 0x0000bc00, 0xbc000000, \
  0x3c00bc00, 0x3c000000, 0x0000bc00, 0xbc000000, 0x3c000000, 0x3c000000, 0x0000bc00, 0xbc000000, \
  0x3c003c00, 0x3c000000, 0x0000bc00, 0xbc000000, 0xbc000000, 0x3c003c00, 0x0000bc00, 0xbc000000, \
  0x0000bc00, 0x3c003c00, 0x0000bc00, 0xbc000000, 0x00000000, 0x3c003c00, 0x0000bc00, 0xbc000000, \
  0x3c000000, 0x3c003c00, 0x0000bc00, 0xbc000000, 0xbc003c00, 0xbc00bc00, 0x00000000, 0xbc000000, \
  0x00000000, 0xbc00bc00, 0x00000000, 0xbc000000, 0x3c003c00, 0xbc00bc00, 0x00000000, 0xbc000000, \
  0xbc000000, 0xbc000000, 0x00000000, 0xbc000000, 0x0000bc00, 0xbc000000, 0x00000000, 0xbc000000, \
  0x00000000, 0xbc000000, 0x00000000, 0xbc000000, 0x00003c00, 0xbc000000, 0x00000000, 0xbc000000, \
  0x3c000000, 0xbc000000, 0x00000000, 0xbc000000, 0xbc00bc00, 0xbc003c00, 0x00000000, 0xbc000000, \
  0xbc003c00, 0xbc003c00, 0x00000000, 0xbc000000, 0x00000000, 0xbc003c00, 0x00000000, 0xbc000000, \
  0x3c00bc00, 0xbc003c00, 0x00000000, 0xbc000000, 0x3c003c00, 0xbc003c00, 0x00000000, 0xbc000000, \
  0xbc000000, 0x0000bc00, 0x00000000, 0xbc000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0xbc000000, \
  0x00000000, 0x0000bc00, 0x00000000, 0xbc000000, 0x00003c00, 0x0000bc00, 0x00000000, 0xbc000000, \
  0xbc000000, 0x00000000, 0x00000000, 0xbc000000, 0xbc003c00, 0x00000000, 0x00000000, 0xbc000000, \
  0x0000bc00, 0x00000000, 0x00000000, 0xbc000000, 0x00000000, 0x00000000, 0x00000000, 0xbc000000, \
  0x00003c00, 0x00000000, 0x00000000, 0xbc000000, 0x3c000000, 0x00000000, 0x00000000, 0xbc000000, \
  0x3c003c00, 0x00000000, 0x00000000, 0xbc000000, 0xbc000000, 0x00003c00, 0x00000000, 0xbc000000, \
  0x0000bc00, 0x00003c00, 0x00000000, 0xbc000000, 0x00000000, 0x00003c00, 0x00000000, 0xbc000000, \
  0x00003c00, 0x00003c00, 0x00000000, 0xbc000000, 0x3c000000, 0x00003c00, 0x00000000, 0xbc000000, \
  0xbc00bc00, 0x3c00bc00, 0x00000000, 0xbc000000, 0xbc003c00, 0x3c00bc00, 0x00000000, 0xbc000000, \
  0x0000bc00, 0x3c00bc00, 0x00000000, 0xbc000000, 0x00000000, 0x3c00bc00, 0x00000000, 0xbc000000, \
  0x3c00bc00, 0x3c00bc00, 0x00000000, 0xbc000000, 0x3c003c00, 0x3c00bc00, 0x00000000, 0xbc000000, \
  0xbc00bc00, 0x3c000000, 0x00000000, 0xbc000000, 0xbc000000, 0x3c000000, 0x00000000, 0xbc000000, \
  0x0000bc00, 0x3c000000, 0x00000000, 0xbc000000, 0x00000000, 0x3c000000, 0x00000000, 0xbc000000, \
  0x00003c00, 0x3c000000, 0x00000000, 0xbc000000, 0x3c000000, 0x3c000000, 0x00000000, 0xbc000000, \
  0x3c003c00, 0x3c000000, 0x00000000, 0xbc000000, 0xbc00bc00, 0x3c003c00, 0x00000000, 0xbc000000, \
  0xbc003c00, 0x3c003c00, 0x00000000, 0xbc000000, 0x00000000, 0x3c003c00, 0x00000000, 0xbc000000, \
  0xbc000000, 0xbc00bc00, 0x00003c00, 0xbc000000, 0x0000bc00, 0xbc00bc00, 0x00003c00, 0xbc000000, \
  0x00000000, 0xbc00bc00, 0x00003c00, 0xbc000000, 0x00003c00, 0xbc00bc00, 0x00003c00, 0xbc000000, \
  0x00000000, 0xbc000000, 0x00003c00, 0xbc000000, 0x00003c00, 0xbc000000, 0x00003c00, 0xbc000000, \
  0x3c00bc00, 0xbc000000, 0x00003c00, 0xbc000000, 0x3c003c00, 0xbc000000, 0x00003c00, 0xbc000000, \
  0xbc000000, 0xbc003c00, 0x00003c00, 0xbc000000, 0x00003c00, 0xbc003c00, 0x00003c00, 0xbc000000, \
  0xbc00bc00, 0x0000bc00, 0x00003c00, 0xbc000000, 0xbc003c00, 0x0000bc00, 0x00003c00, 0xbc000000, \
  0x0000bc00, 0x0000bc00, 0x00003c00, 0xbc000000, 0x00000000, 0x0000bc00, 0x00003c00, 0xbc000000, \
  0x3c00bc00, 0x0000bc00, 0x00003c00, 0xbc000000, 0x3c003c00, 0x0000bc00, 0x00003c00, 0xbc000000, \
  0xbc000000, 0x00000000, 0x00003c00, 0xbc000000, 0x00000000, 0x00000000, 0x00003c00, 0xbc000000, \
  0x00003c00, 0x00000000, 0x00003c00, 0xbc000000, 0x3c00bc00, 0x00000000, 0x00003c00, 0xbc000000, \
  0x3c000000, 0x00000000, 0x00003c00, 0xbc000000, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc000000, \
  0x0000bc00, 0x00003c00, 0x00003c00, 0xbc000000, 0x00000000, 0x00003c00, 0x00003c00, 0xbc000000, \
  0x3c00bc00, 0x00003c00, 0x00003c00, 0xbc000000, 0x3c000000, 0x00003c00, 0x00003c00, 0xbc000000, \
  0x3c003c00, 0x00003c00, 0x00003c00, 0xbc000000, 0x00003c00, 0x3c00bc00, 0x00003c00, 0xbc000000, \
  0x3c003c00, 0x3c00bc00, 0x00003c00, 0xbc000000, 0xbc003c00, 0x3c000000, 0x00003c00, 0xbc000000, \
  0x00000000, 0x3c000000, 0x00003c00, 0xbc000000, 0x0000bc00, 0x3c003c00, 0x00003c00, 0xbc000000, \
  0x3c000000, 0x3c003c00, 0x00003c00, 0xbc000000, 0xbc000000, 0xbc000000, 0x3c00bc00, 0xbc000000, \
  0x00003c00, 0xbc000000, 0x3c00bc00, 0xbc000000, 0x00000000, 0xbc003c00, 0x3c00bc00, 0xbc000000, \
  0xbc000000, 0x0000bc00, 0x3c00bc00, 0xbc000000, 0x0000bc00, 0x0000bc00, 0x3c00bc00, 0xbc000000, \
  0x00003c00, 0x0000bc00, 0x3c00bc00, 0xbc000000, 0x3c000000, 0x0000bc00, 0x3c00bc00, 0xbc000000, \
  0xbc00bc00, 0x00000000, 0x3c00bc00, 0xbc000000, 0x00000000, 0x00000000, 0x3c00bc00, 0xbc000000, \
  0x3c00bc00, 0x00000000, 0x3c00bc00, 0xbc000000, 0x3c003c00, 0x00000000, 0x3c00bc00, 0xbc000000, \
  0xbc00bc00, 0x00003c00, 0x3c00bc00, 0xbc000000, 0xbc000000, 0x00003c00, 0x3c00bc00, 0xbc000000, \
  0x0000bc00, 0x00003c00, 0x3c00bc00, 0xbc000000, 0x00003c00, 0x00003c00, 0x3c00bc00, 0xbc000000, \
  0x3c000000, 0x00003c00, 0x3c00bc00, 0xbc000000, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0xbc000000, \
  0xbc000000, 0x3c000000, 0x3c00bc00, 0xbc000000, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0xbc000000, \
  0x00000000, 0x3c003c00, 0x3c00bc00, 0xbc000000, 0xbc00bc00, 0xbc000000, 0x3c000000, 0xbc000000, \
  0xbc003c00, 0xbc000000, 0x3c000000, 0xbc000000, 0x00000000, 0xbc000000, 0x3c000000, 0xbc000000, \
  0x3c003c00, 0xbc000000, 0x3c000000, 0xbc000000, 0xbc000000, 0xbc003c00, 0x3c000000, 0xbc000000, \
  0x00000000, 0xbc003c00, 0x3c000000, 0xbc000000, 0xbc003c00, 0x0000bc00, 0x3c000000, 0xbc000000, \
  0x0000bc00, 0x0000bc00, 0x3c000000, 0xbc000000, 0x00000000, 0x0000bc00, 0x3c000000, 0xbc000000, \
  0x3c00bc00, 0x0000bc00, 0x3c000000, 0xbc000000, 0xbc000000, 0x00000000, 0x3c000000, 0xbc000000, \
  0x0000bc00, 0x00000000, 0x3c000000, 0xbc000000, 0x00000000, 0x00000000, 0x3c000000, 0xbc000000, \
  0x00003c00, 0x00000000, 0x3c000000, 0xbc000000, 0x3c000000, 0x00000000, 0x3c000000, 0xbc000000, \
  0x3c003c00, 0x00000000, 0x3c000000, 0xbc000000, 0xbc00bc00, 0x00003c00, 0x3c000000, 0xbc000000, \
  0x00000000, 0x00003c00, 0x3c000000, 0xbc000000, 0x3c003c00, 0x00003c00, 0x3c000000, 0xbc000000, \
  0x3c000000, 0x3c00bc00, 0x3c000000, 0xbc000000, 0xbc000000, 0x3c000000, 0x3c000000, 0xbc000000, \
  0xbc003c00, 0x3c000000, 0x3c000000, 0xbc000000, 0x00000000, 0x3c000000, 0x3c000000, 0xbc000000, \
  0x3c00bc00, 0x3c000000, 0x3c000000, 0xbc000000, 0xbc000000, 0x3c003c00, 0x3c000000, 0xbc000000, \
  0x00003c00, 0x3c003c00, 0x3c000000, 0xbc000000, 0x3c000000, 0x3c003c00, 0x3c000000, 0xbc000000, \
  0x3c000000, 0xbc00bc00, 0x3c003c00, 0xbc000000, 0x00003c00, 0xbc000000, 0x3c003c00, 0xbc000000, \
  0x0000bc00, 0xbc003c00, 0x3c003c00, 0xbc000000, 0x00003c00, 0xbc003c00, 0x3c003c00, 0xbc000000, \
  0x0000bc00, 0x0000bc00, 0x3c003c00, 0xbc000000, 0x00003c00, 0x0000bc00, 0x3c003c00, 0xbc000000, \
  0x3c000000, 0x0000bc00, 0x3c003c00, 0xbc000000, 0xbc00bc00, 0x00000000, 0x3c003c00, 0xbc000000, \
  0xbc003c00, 0x00000000, 0x3c003c00, 0xbc000000, 0x00000000, 0x00000000, 0x3c003c00, 0xbc000000, \
  0x3c00bc00, 0x00000000, 0x3c003c00, 0xbc000000, 0xbc000000, 0x00003c00, 0x3c003c00, 0xbc000000, \
  0x00003c00, 0x00003c00, 0x3c003c00, 0xbc000000, 0x3c000000, 0x00003c00, 0x3c003c00, 0xbc000000, \
  0x00000000, 0x3c00bc00, 0x3c003c00, 0xbc000000, 0xbc000000, 0x3c000000, 0x3c003c00, 0xbc000000, \
  0x00003c00, 0x3c000000, 0x3c003c00, 0xbc000000, 0x3c003c00, 0x3c000000, 0x3c003c00, 0xbc000000, \
  0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, \
  0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0xbc003c00, \
  0x00000000, 0xbc000000, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, \
  0xbc003c00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0x00000000, 0xbc003c00, 0xbc00bc00, 0xbc003c00, \
  0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, \
  0x00000000, 0x0000bc00, 0xbc00bc00, 0xbc003c00, 0xbc000000, 0x00000000, 0xbc00bc00, 0xbc003c00, \
  0x3c000000, 0x00000000, 0xbc00bc00, 0xbc003c00, 0xbc000000, 0x00003c00, 0xbc00bc00, 0xbc003c00, \
  0x00000000, 0x00003c00, 0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, \
  0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, \
  0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0x00000000, 0x3c000000, 0xbc00bc00, 0xbc003c00, \
  0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, \
  0x00000000, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, \
  0x3c003c00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x00000000, 0xbc00bc00, 0xbc000000, 0xbc003c00, \
  0xbc000000, 0xbc000000, 0xbc000000, 0xbc003c00, 0x0000bc00, 0xbc000000, 0xbc000000, 0xbc003c00, \
  0x3c000000, 0xbc000000, 0xbc000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0xbc000000, 0xbc003c00, \
  0xbc003c00, 0x0000bc00, 0xbc000000, 0xbc003c00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0xbc003c00, \
  0x3c000000, 0x0000bc00, 0xbc000000, 0xbc003c00, 0x00000000, 0x00000000, 0xbc000000, 0xbc003c00, \
  0x3c00bc00, 0x00000000, 0xbc000000, 0xbc003c00, 0x3c003c00, 0x00000000, 0xbc000000, 0xbc003c00, \
  0xbc000000, 0x00003c00, 0xbc000000, 0xbc003c00, 0x0000bc00, 0x00003c00, 0xbc000000, 0xbc003c00, \
  0x00000000, 0x00003c00, 0xbc000000, 0xbc003c00, 0x00003c00, 0x00003c00, 0xbc000000, 0xbc003c00, \
  0x00000000, 0x3c00bc00, 0xbc000000, 0xbc003c00, 0xbc00bc00, 0x3c000000, 0xbc000000, 0xbc003c00, \
  0x00003c00, 0x3c000000, 0xbc000000, 0xbc003c00, 0x3c000000, 0x3c000000, 0xbc000000, 0xbc003c00, \
  0x00000000, 0x3c003c00, 0xbc000000, 0xbc003c00, 0xbc000000, 0xbc00bc00, 0xbc003c00, 0xbc003c00, \
  0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0xbc003c00, \
  0xbc000000, 0xbc000000, 0xbc003c00, 0xbc003c00, 0x00000000, 0xbc000000, 0xbc003c00, 0xbc003c00, \
  0xbc00bc00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, \
  0x3c00bc00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, \
  0x00000000, 0x0000bc00, 0xbc003c00, 0xbc003c00, 0xbc000000, 0x00000000, 0xbc003c00, 0xbc003c00, \
  0x00003c00, 0x00000000, 0xbc003c00, 0xbc003c00, 0x3c000000, 0x00000000, 0xbc003c00, 0xbc003c00, \
  0x00000000, 0x00003c00, 0xbc003c00, 0xbc003c00, 0xbc000000, 0x3c00bc00, 0xbc003c00, 0xbc003c00, \
  0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, \
  0xbc000000, 0x3c000000, 0xbc003c00, 0xbc003c00, 0x00000000, 0x3c000000, 0xbc003c00, 0xbc003c00, \
  0xbc00bc00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0xbc003c00, \
  0x3c00bc00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0xbc003c00, \
  0x00000000, 0xbc00bc00, 0x0000bc00, 0xbc003c00, 0x0000bc00, 0xbc000000, 0x0000bc00, 0xbc003c00, \
  0x00003c00, 0xbc000000, 0x0000bc00, 0xbc003c00, 0x3c000000, 0xbc000000, 0x0000bc00, 0xbc003c00, \
  0x00000000, 0xbc003c00, 0x0000bc00, 0xbc003c00, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0xbc003c00, \
  0x00000000, 0x0000bc00, 0x0000bc00, 0xbc003c00, 0x00003c00, 0x0000bc00, 0x0000bc00, 0xbc003c00, \
  0x3c000000, 0x0000bc00, 0x0000bc00, 0xbc003c00, 0xbc003c00, 0x00000000, 0x0000bc00, 0xbc003c00, \
  0x00000000, 0x00000000, 0x0000bc00, 0xbc003c00, 0x3c00bc00, 0x00000000, 0x0000bc00, 0xbc003c00, \
  0x3c003c00, 0x00000000, 0x0000bc00, 0xbc003c00, 0x00003c00, 0x00003c00, 0x0000bc00, 0xbc003c00, \
  0x00000000, 0x3c00bc00, 0x0000bc00, 0xbc003c00, 0xbc000000, 0x3c000000, 0x0000bc00, 0xbc003c00, \
  0x0000bc00, 0x3c000000, 0x0000bc00, 0xbc003c00, 0x3c000000, 0x3c000000, 0x0000bc00, 0xbc003c00, \
  0xbc000000, 0x3c003c00, 0x0000bc00, 0xbc003c00, 0x00000000, 0x3c003c00, 0x0000bc00, 0xbc003c00, \
  0x3c000000, 0xbc00bc00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc000000, 0x00000000, 0xbc003c00, \
  0xbc000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x3c000000, 0xbc003c00, 0x00000000, 0xbc003c00, \
  0xbc00bc00, 0x0000bc00, 0x00000000, 0xbc003c00, 0x00000000, 0x0000bc00, 0x00000000, 0xbc003c00, \
  0x3c00bc00, 0x0000bc00, 0x00000000, 0xbc003c00, 0xbc000000, 0x00000000, 0x00000000, 0xbc003c00, \
  0x0000bc00, 0x00000000, 0x00000000, 0xbc003c00, 0x00000000, 0x00000000, 0x00000000, 0xbc003c00, \
  0x3c000000, 0x00000000, 0x00000000, 0xbc003c00, 0xbc003c00, 0x00003c00, 0x00000000, 0xbc003c00, \
  0x00000000, 0x00003c00, 0x00000000, 0xbc003c00, 0x3c00bc00, 0x00003c00, 0x00000000, 0xbc003c00, \
  0x3c000000, 0x3c00bc00, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x3c000000, 0x00000000, 0xbc003c00, \
  0x0000bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0x00000000, 0x3c000000, 0x00000000, 0xbc003c00, \
  0x3c00bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0x3c003c00, 0x3c000000, 0x00000000, 0xbc003c00, \
  0xbc000000, 0x3c003c00, 0x00000000, 0xbc003c00, 0x0000bc00, 0x3c003c00, 0x00000000, 0xbc003c00, \
  0x00003c00, 0x3c003c00, 0x00000000, 0xbc003c00, 0x3c000000, 0x3c003c00, 0x00000000, 0xbc003c00, \
  0x00000000, 0xbc00bc00, 0x00003c00, 0xbc003c00, 0xbc00bc00, 0xbc000000, 0x00003c00, 0xbc003c00, \
  0xbc003c00, 0xbc000000, 0x00003c00, 0xbc003c00, 0x3c000000, 0xbc000000, 0x00003c00, 0xbc003c00, \
  0x00000000, 0xbc003c00, 0x00003c00, 0xbc003c00, 0xbc000000, 0x0000bc00, 0x00003c00, 0xbc003c00, \
  0x3c000000, 0x0000bc00, 0x00003c00, 0xbc003c00, 0x00000000, 0x00000000, 0x00003c00, 0xbc003c00, \
  0xbc00bc00, 0x00003c00, 0x00003c00, 0xbc003c00, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc003c00, \
  0x3c000000, 0x00003c00, 0x00003c00, 0xbc003c00, 0x0000bc00, 0x3c00bc00, 0x00003c00, 0xbc003c00, \
  0x00003c00, 0x3c00bc00, 0x00003c00, 0xbc003c00, 0xbc00bc00, 0x3c000000, 0x00003c00, 0xbc003c00, \
  0x3c003c00, 0x3c000000, 0x00003c00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, \
  0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, \
  0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x00000000, 0xbc000000, 0x3c00bc00, 0xbc003c00, \
  0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, \
  0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, \
  0x00000000, 0x0000bc00, 0x3c00bc00, 0xbc003c00, 0xbc000000, 0x00000000, 0x3c00bc00, 0xbc003c00, \
  0x0000bc00, 0x00000000, 0x3c00bc00, 0xbc003c00, 0x00000000, 0x00003c00, 0x3c00bc00, 0xbc003c00, \
  0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, \
  0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, \
  0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, \
  0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, \
  0x3c000000, 0xbc00bc00, 0x3c000000, 0xbc003c00, 0xbc000000, 0xbc000000, 0x3c000000, 0xbc003c00, \
  0x0000bc00, 0xbc000000, 0x3c000000, 0xbc003c00, 0x3c000000, 0xbc000000, 0x3c000000, 0xbc003c00, \
  0x00000000, 0xbc003c00, 0x3c000000, 0xbc003c00, 0x00003c00, 0x0000bc00, 0x3c000000, 0xbc003c00, \
  0x3c000000, 0x0000bc00, 0x3c000000, 0xbc003c00, 0xbc003c00, 0x00000000, 0x3c000000, 0xbc003c00, \
  0x00000000, 0x00000000, 0x3c000000, 0xbc003c00, 0xbc000000, 0x00003c00, 0x3c000000, 0xbc003c00, \
  0x0000bc00, 0x00003c00, 0x3c000000, 0xbc003c00, 0x00003c00, 0x00003c00, 0x3c000000, 0xbc003c00, \
  0x3c000000, 0x00003c00, 0x3c000000, 0xbc003c00, 0x00000000, 0x3c00bc00, 0x3c000000, 0xbc003c00, \
  0xbc00bc00, 0x3c000000, 0x3c000000, 0xbc003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0xbc003c00, \
  0x3c000000, 0x3c000000, 0x3c000000, 0xbc003c00, 0x0000bc00, 0x3c003c00, 0x3c000000, 0xbc003c00, \
  0x00000000, 0x3c003c00, 0x3c000000, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, \
  0xbc003c00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, \
  0x3c003c00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0xbc003c00, \
  0xbc003c00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0xbc003c00, \
  0x3c003c00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0x00000000, 0x0000bc00, 0x3c003c00, 0xbc003c00, \
  0xbc000000, 0x00000000, 0x3c003c00, 0xbc003c00, 0x00003c00, 0x00000000, 0x3c003c00, 0xbc003c00, \
  0x3c000000, 0x00000000, 0x3c003c00, 0xbc003c00, 0x00000000, 0x00003c00, 0x3c003c00, 0xbc003c00, \
  0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, \
  0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, \
  0x00000000, 0x3c000000, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0xbc003c00, \
  0xbc003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0xbc003c00, \
  0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x0000bc00, \
  0xbc000000, 0xbc000000, 0xbc00bc00, 0x0000bc00, 0x00003c00, 0xbc000000, 0xbc00bc00, 0x0000bc00, \
  0x00000000, 0xbc003c00, 0xbc00bc00, 0x0000bc00, 0x3c000000, 0x0000bc00, 0xbc00bc00, 0x0000bc00, \
  0xbc003c00, 0x00000000, 0xbc00bc00, 0x0000bc00, 0x00000000, 0x00000000, 0xbc00bc00, 0x0000bc00, \
  0x3c00bc00, 0x00000000, 0xbc00bc00, 0x0000bc00, 0x3c003c00, 0x00000000, 0xbc00bc00, 0x0000bc00, \
  0xbc000000, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x0000bc00, 0x00003c00, 0xbc00bc00, 0x0000bc00, \
  0x00003c00, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0xbc00bc00, 0x0000bc00, \
  0x3c000000, 0x3c000000, 0xbc00bc00, 0x0000bc00, 0xbc000000, 0x3c003c00, 0xbc00bc00, 0x0000bc00, \
  0x00003c00, 0x3c003c00, 0xbc00bc00, 0x0000bc00, 0xbc00bc00, 0xbc00bc00, 0xbc000000, 0x0000bc00, \
  0xbc000000, 0xbc00bc00, 0xbc000000, 0x0000bc00, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0x0000bc00, \
  0x00003c00, 0xbc00bc00, 0xbc000000, 0x0000bc00, 0x3c000000, 0xbc00bc00, 0xbc000000, 0x0000bc00, \
  0xbc003c00, 0xbc000000, 0xbc000000, 0x0000bc00, 0x00000000, 0xbc000000, 0xbc000000, 0x0000bc00, \
  0x00003c00, 0xbc000000, 0xbc000000, 0x0000bc00, 0x3c00bc00, 0xbc000000, 0xbc000000, 0x0000bc00, \
  0x3c003c00, 0xbc000000, 0xbc000000, 0x0000bc00, 0xbc000000, 0xbc003c00, 0xbc000000, 0x0000bc00, \
  0x00003c00, 0xbc003c00, 0xbc000000, 0x0000bc00, 0x3c000000, 0xbc003c00, 0xbc000000, 0x0000bc00, \
  0x00000000, 0x0000bc00, 0xbc000000, 0x0000bc00, 0x3c00bc00, 0x0000bc00, 0xbc000000, 0x0000bc00, \
  0x3c003c00, 0x0000bc00, 0xbc000000, 0x0000bc00, 0xbc000000, 0x00000000, 0xbc000000, 0x0000bc00, \
  0x0000bc00, 0x00000000, 0xbc000000, 0x0000bc00, 0x00000000, 0x00000000, 0xbc000000, 0x0000bc00, \
  0x00003c00, 0x00000000, 0xbc000000, 0x0000bc00, 0x3c000000, 0x00000000, 0xbc000000, 0x0000bc00, \
  0x3c003c00, 0x00000000, 0xbc000000, 0x0000bc00, 0x00000000, 0x00003c00, 0xbc000000, 0x0000bc00, \
  0x3c00bc00, 0x00003c00, 0xbc000000, 0x0000bc00, 0x3c003c00, 0x00003c00, 0xbc000000, 0x0000bc00, \
  0xbc000000, 0x3c00bc00, 0xbc000000, 0x0000bc00, 0x0000bc00, 0x3c00bc00, 0xbc000000, 0x0000bc00, \
  0x00003c00, 0x3c00bc00, 0xbc000000, 0x0000bc00, 0xbc00bc00, 0x3c000000, 0xbc000000, 0x0000bc00, \
  0xbc003c00, 0x3c000000, 0xbc000000, 0x0000bc00, 0x00000000, 0x3c000000, 0xbc000000, 0x0000bc00, \
  0xbc00bc00, 0x3c003c00, 0xbc000000, 0x0000bc00, 0xbc000000, 0x3c003c00, 0xbc000000, 0x0000bc00, \
  0xbc003c00, 0x3c003c00, 0xbc000000, 0x0000bc00, 0x00000000, 0xbc00bc00, 0xbc003c00, 0x0000bc00, \
  0xbc000000, 0xbc000000, 0xbc003c00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0xbc003c00, 0x0000bc00, \
  0x00003c00, 0xbc000000, 0xbc003c00, 0x0000bc00, 0x00000000, 0xbc003c00, 0xbc003c00, 0x0000bc00, \
  0xbc000000, 0x0000bc00, 0xbc003c00, 0x0000bc00, 0xbc003c00, 0x00000000, 0xbc003c00, 0x0000bc00, \
  0x00000000, 0x00000000, 0xbc003c00, 0x0000bc00, 0x3c003c00, 0x00000000, 0xbc003c00, 0x0000bc00, \
  0x0000bc00, 0x00003c00, 0xbc003c00, 0x0000bc00, 0x3c000000, 0x00003c00, 0xbc003c00, 0x0000bc00, \
  0x3c000000, 0x3c00bc00, 0xbc003c00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0xbc003c00, 0x0000bc00, \
  0x00000000, 0x3c003c00, 0xbc003c00, 0x0000bc00, 0xbc000000, 0xbc00bc00, 0x0000bc00, 0x0000bc00, \
  0x00000000, 0xbc000000, 0x0000bc00, 0x0000bc00, 0x3c000000, 0xbc000000, 0x0000bc00, 0x0000bc00, \
  0x3c000000, 0xbc003c00, 0x0000bc00, 0x0000bc00, 0x00000000, 0x0000bc00, 0x0000bc00, 0x0000bc00, \
  0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x0000bc00, \
  0xbc000000, 0x00000000, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x00000000, 0x0000bc00, 0x0000bc00, \
  0x00000000, 0x00000000, 0x0000bc00, 0x0000bc00, 0x00003c00, 0x00000000, 0x0000bc00, 0x0000bc00, \
  0xbc000000, 0x00003c00, 0x0000bc00, 0x0000bc00, 0xbc003c00, 0x00003c00, 0x0000bc00, 0x0000bc00, \
  0x00000000, 0x00003c00, 0x0000bc00, 0x0000bc00, 0x3c00bc00, 0x00003c00, 0x0000bc00, 0x0000bc00, \
  0x3c003c00, 0x00003c00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0x3c00bc00, 0x0000bc00, 0x0000bc00, \
  0x00003c00, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x0000bc00, \
  0xbc00bc00, 0x3c000000, 0x0000bc00, 0x0000bc00, 0xbc003c00, 0x3c000000, 0x0000bc00, 0x0000bc00, \
  0x00000000, 0x3c000000, 0x0000bc00, 0x0000bc00, 0xbc00bc00, 0x3c003c00, 0x0000bc00, 0x0000bc00, \
  0xbc000000, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x3c003c00, 0x0000bc00, 0x0000bc00, \
  0xbc000000, 0xbc00bc00, 0x00000000, 0x0000bc00, 0xbc003c00, 0xbc00bc00, 0x00000000, 0x0000bc00, \
  0x00000000, 0xbc00bc00, 0x00000000, 0x0000bc00, 0x3c003c00, 0xbc00bc00, 0x00000000, 0x0000bc00, \
  0xbc000000, 0xbc000000, 0x00000000, 0x0000bc00, 0x0000bc00, 0xbc000000, 0x00000000, 0x0000bc00, \
  0x00000000, 0xbc000000, 0x00000000, 0x0000bc00, 0x00003c00, 0xbc000000, 0x00000000, 0x0000bc00, \
  0x3c000000, 0xbc000000, 0x00000000, 0x0000bc00, 0xbc00bc00, 0xbc003c00, 0x00000000, 0x0000bc00, \
  0x00000000, 0xbc003c00, 0x00000000, 0x0000bc00, 0x3c003c00, 0xbc003c00, 0x00000000, 0x0000bc00, \
  0xbc000000, 0x0000bc00, 0x00000000, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x00000000, 0x0000bc00, \
  0x00000000, 0x0000bc00, 0x00000000, 0x0000bc00, 0x00003c00, 0x0000bc00, 0x00000000, 0x0000bc00, \
  0x3c000000, 0x0000bc00, 0x00000000, 0x0000bc00, 0xbc00bc00, 0x00000000, 0x00000000, 0x0000bc00, \
  0xbc000000, 0x00000000, 0x00000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0x00000000, 0x0000bc00, \
  0x00000000, 0x00000000, 0x00000000, 0x0000bc00, 0x00003c00, 0x00000000, 0x00000000, 0x0000bc00, \
  0x3c00bc00, 0x00000000, 0x00000000, 0x0000bc00, 0x3c000000, 0x00000000, 0x00000000, 0x0000bc00, \
  0xbc000000, 0x00003c00, 0x00000000, 0x0000bc00, 0x0000bc00, 0x00003c00, 0x00000000, 0x0000bc00, \
  0x00000000, 0x00003c00, 0x00000000, 0x0000bc00, 0x00003c00, 0x00003c00, 0x00000000, 0x0000bc00, \
  0x3c000000, 0x00003c00, 0x00000000, 0x0000bc00, 0xbc003c00, 0x3c00bc00, 0x00000000, 0x0000bc00, \
  0x0000bc00, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x00000000, 0x0000bc00, \
  0x3c00bc00, 0x3c00bc00, 0x00000000, 0x0000bc00, 0xbc000000, 0x3c000000, 0x00000000, 0x0000bc00, \
  0x0000bc00, 0x3c000000, 0x00000000, 0x0000bc00, 0x00000000, 0x3c000000, 0x00000000, 0x0000bc00, \
  0x00003c00, 0x3c000000, 0x00000000, 0x0000bc00, 0x3c000000, 0x3c000000, 0x00000000, 0x0000bc00, \
  0x3c003c00, 0x3c000000, 0x00000000, 0x0000bc00, 0x00000000, 0x3c003c00, 0x00000000, 0x0000bc00, \
  0x3c00bc00, 0x3c003c00, 0x00000000, 0x0000bc00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x0000bc00, \
  0xbc000000, 0xbc00bc00, 0x00003c00, 0x0000bc00, 0x00000000, 0xbc00bc00, 0x00003c00, 0x0000bc00, \
  0x3c000000, 0xbc00bc00, 0x00003c00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0x00003c00, 0x0000bc00, \
  0x00000000, 0xbc000000, 0x00003c00, 0x0000bc00, 0x3c00bc00, 0xbc000000, 0x00003c00, 0x0000bc00, \
  0x3c003c00, 0xbc000000, 0x00003c00, 0x0000bc00, 0xbc000000, 0xbc003c00, 0x00003c00, 0x0000bc00, \
  0x0000bc00, 0xbc003c00, 0x00003c00, 0x0000bc00, 0x3c000000, 0xbc003c00, 0x00003c00, 0x0000bc00, \
  0xbc00bc00, 0x0000bc00, 0x00003c00, 0x0000bc00, 0xbc003c00, 0x0000bc00, 0x00003c00, 0x0000bc00, \
  0x00000000, 0x0000bc00, 0x00003c00, 0x0000bc00, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0x0000bc00, \
  0xbc00bc00, 0x00000000, 0x00003c00, 0x0000bc00, 0xbc000000, 0x00000000, 0x00003c00, 0x0000bc00, \
  0xbc003c00, 0x00000000, 0x00003c00, 0x0000bc00, 0x00000000, 0x00000000, 0x00003c00, 0x0000bc00, \
  0x00003c00, 0x00000000, 0x00003c00, 0x0000bc00, 0x3c000000, 0x00000000, 0x00003c00, 0x0000bc00, \
  0xbc003c00, 0x00003c00, 0x00003c00, 0x0000bc00, 0x00000000, 0x00003c00, 0x00003c00, 0x0000bc00, \
  0x3c00bc00, 0x00003c00, 0x00003c00, 0x0000bc00, 0xbc000000, 0x3c00bc00, 0x00003c00, 0x0000bc00, \
  0x00000000, 0x3c00bc00, 0x00003c00, 0x0000bc00, 0x3c003c00, 0x3c00bc00, 0x00003c00, 0x0000bc00, \
  0x0000bc00, 0x3c000000, 0x00003c00, 0x0000bc00, 0x00000000, 0x3c000000, 0x00003c00, 0x0000bc00, \
  0xbc000000, 0x3c003c00, 0x00003c00, 0x0000bc00, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x0000bc00, \
  0x00003c00, 0x3c003c00, 0x00003c00, 0x0000bc00, 0x00000000, 0xbc00bc00, 0x3c00bc00, 0x0000bc00, \
  0xbc000000, 0xbc000000, 0x3c00bc00, 0x0000bc00, 0x00000000, 0xbc000000, 0x3c00bc00, 0x0000bc00, \
  0x3c003c00, 0xbc000000, 0x3c00bc00, 0x0000bc00, 0x00000000, 0xbc003c00, 0x3c00bc00, 0x0000bc00, \
  0xbc003c00, 0x0000bc00, 0x3c00bc00, 0x0000bc00, 0x3c000000, 0x0000bc00, 0x3c00bc00, 0x0000bc00, \
  0xbc00bc00, 0x00000000, 0x3c00bc00, 0x0000bc00, 0x00000000, 0x00000000, 0x3c00bc00, 0x0000bc00, \
  0x3c00bc00, 0x00000000, 0x3c00bc00, 0x0000bc00, 0xbc000000, 0x00003c00, 0x3c00bc00, 0x0000bc00, \
  0x0000bc00, 0x00003c00, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x0000bc00, \
  0x3c000000, 0x00003c00, 0x3c00bc00, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x0000bc00, \
  0xbc000000, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0x0000bc00, \
  0x00003c00, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x3c000000, 0x3c000000, 0x3c00bc00, 0x0000bc00, \
  0x00000000, 0x3c003c00, 0x3c00bc00, 0x0000bc00, 0xbc000000, 0xbc00bc00, 0x3c000000, 0x0000bc00, \
  0x00000000, 0xbc00bc00, 0x3c000000, 0x0000bc00, 0x00003c00, 0xbc00bc00, 0x3c000000, 0x0000bc00, \
  0x3c003c00, 0xbc00bc00, 0x3c000000, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x0000bc00, \
  0x0000bc00, 0xbc000000, 0x3c000000, 0x0000bc00, 0x00000000, 0xbc000000, 0x3c000000, 0x0000bc00, \
  0x3c00bc00, 0xbc000000, 0x3c000000, 0x0000bc00, 0xbc000000, 0xbc003c00, 0x3c000000, 0x0000bc00, \
  0x0000bc00, 0xbc003c00, 0x3c000000, 0x0000bc00, 0x00003c00, 0xbc003c00, 0x3c000000, 0x0000bc00, \
  0xbc00bc00, 0x0000bc00, 0x3c000000, 0x0000bc00, 0x00000000, 0x0000bc00, 0x3c000000, 0x0000bc00, \
  0x3c003c00, 0x0000bc00, 0x3c000000, 0x0000bc00, 0xbc000000, 0x00000000, 0x3c000000, 0x0000bc00, \
  0xbc003c00, 0x00000000, 0x3c000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0x3c000000, 0x0000bc00, \
  0x00000000, 0x00000000, 0x3c000000, 0x0000bc00, 0x00003c00, 0x00000000, 0x3c000000, 0x0000bc00, \
  0x3c000000, 0x00000000, 0x3c000000, 0x0000bc00, 0xbc00bc00, 0x00003c00, 0x3c000000, 0x0000bc00, \
  0xbc003c00, 0x00003c00, 0x3c000000, 0x0000bc00, 0x00000000, 0x00003c00, 0x3c000000, 0x0000bc00, \
  0x00003c00, 0x00003c00, 0x3c000000, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x3c000000, 0x0000bc00, \
  0x00003c00, 0x3c00bc00, 0x3c000000, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0x3c000000, 0x0000bc00, \
  0xbc003c00, 0x3c000000, 0x3c000000, 0x0000bc00, 0x00000000, 0x3c000000, 0x3c000000, 0x0000bc00, \
  0x00003c00, 0x3c000000, 0x3c000000, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0x3c000000, 0x0000bc00, \
  0xbc000000, 0x3c003c00, 0x3c000000, 0x0000bc00, 0x0000bc00, 0x3c003c00, 0x3c000000, 0x0000bc00, \
  0x00003c00, 0x3c003c00, 0x3c000000, 0x0000bc00, 0x3c000000, 0x3c003c00, 0x3c000000, 0x0000bc00, \
  0x00003c00, 0xbc000000, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x3c003c00, 0x0000bc00, \
  0x00003c00, 0x0000bc00, 0x3c003c00, 0x0000bc00, 0x3c000000, 0x0000bc00, 0x3c003c00, 0x0000bc00, \
  0x00000000, 0x00000000, 0x3c003c00, 0x0000bc00, 0x3c00bc00, 0x00000000, 0x3c003c00, 0x0000bc00, \
  0x3c003c00, 0x00000000, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x00003c00, 0x3c003c00, 0x0000bc00, \
  0x3c000000, 0x00003c00, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x3c003c00, 0x0000bc00, \
  0x00000000, 0x3c003c00, 0x3c003c00, 0x0000bc00, 0xbc000000, 0xbc00bc00, 0xbc00bc00, 0x00000000, \
  0x0000bc00, 0xbc00bc00, 0xbc00bc00, 0x00000000, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x00000000, \
  0x00003c00, 0xbc00bc00, 0xbc00bc00, 0x00000000, 0x3c000000, 0xbc00bc00, 0xbc00bc00, 0x00000000, \
  0xbc003c00, 0xbc000000, 0xbc00bc00, 0x00000000, 0x00000000, 0xbc000000, 0xbc00bc00, 0x00000000, \
  0x3c003c00, 0xbc000000, 0xbc00bc00, 0x00000000, 0xbc000000, 0xbc003c00, 0xbc00bc00, 0x00000000, \
  0x0000bc00, 0xbc003c00, 0xbc00bc00, 0x00000000, 0x3c000000, 0xbc003c00, 0xbc00bc00, 0x00000000, \
  0xbc00bc00, 0x0000bc00, 0xbc00bc00, 0x00000000, 0x00000000, 0x0000bc00, 0xbc00bc00, 0x00000000, \
  0x3c00bc00, 0x0000bc00, 0xbc00bc00, 0x00000000, 0xbc000000, 0x00000000, 0xbc00bc00, 0x00000000, \
  0x0000bc00, 0x00000000, 0xbc00bc00, 0x00000000, 0x00000000, 0x00000000, 0xbc00bc00, 0x00000000, \
  0x00003c00, 0x00000000, 0xbc00bc00, 0x00000000, 0x3c000000, 0x00000000, 0xbc00bc00, 0x00000000, \
  0x00000000, 0x00003c00, 0xbc00bc00, 0x00000000, 0x3c00bc00, 0x00003c00, 0xbc00bc00, 0x00000000, \
  0x00003c00, 0x3c00bc00, 0xbc00bc00, 0x00000000, 0x3c000000, 0x3c00bc00, 0xbc00bc00, 0x00000000, \
  0x00000000, 0x3c000000, 0xbc00bc00, 0x00000000, 0x3c00bc00, 0x3c000000, 0xbc00bc00, 0x00000000, \
  0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x00000000, 0xbc000000, 0x3c003c00, 0xbc00bc00, 0x00000000, \
  0x00003c00, 0x3c003c00, 0xbc00bc00, 0x00000000, 0x3c000000, 0x3c003c00, 0xbc00bc00, 0x00000000, \
  0x00000000, 0xbc00bc00, 0xbc000000, 0x00000000, 0x3c00bc00, 0xbc00bc00, 0xbc000000, 0x00000000, \
  0x3c000000, 0xbc00bc00, 0xbc000000, 0x00000000, 0x3c003c00, 0xbc00bc00, 0xbc000000, 0x00000000, \
  0xbc000000, 0xbc000000, 0xbc000000, 0x00000000, 0x0000bc00, 0xbc000000, 0xbc000000, 0x00000000, \
  0x00000000, 0xbc000000, 0xbc000000, 0x00000000, 0x00003c00, 0xbc000000, 0xbc000000, 0x00000000, \
  0x3c00bc00, 0xbc000000, 0xbc000000, 0x00000000, 0x3c000000, 0xbc000000, 0xbc000000, 0x00000000, \
  0xbc00bc00, 0xbc003c00, 0xbc000000, 0x00000000, 0x00000000, 0xbc003c00, 0xbc000000, 0x00000000, \
  0x00003c00, 0xbc003c00, 0xbc000000, 0x00000000, 0x3c00bc00, 0xbc003c00, 0xbc000000, 0x00000000, \
  0x3c003c00, 0xbc003c00, 0xbc000000, 0x00000000, 0xbc000000, 0x0000bc00, 0xbc000000, 0x00000000, \
  0x0000bc00, 0x0000bc00, 0xbc000000, 0x00000000, 0x00000000, 0x0000bc00, 0xbc000000, 0x00000000, \
  0x00003c00, 0x0000bc00, 0xbc000000, 0x00000000, 0x3c000000, 0x0000bc00, 0xbc000000, 0x00000000, \
  0xbc00bc00, 0x00000000, 0xbc000000, 0x00000000, 0xbc000000, 0x00000000, 0xbc000000, 0x00000000, \
  0xbc003c00, 0x00000000, 0xbc000000, 0x00000000, 0x0000bc00, 0x00000000, 0xbc000000, 0x00000000, \
  0x00000000, 0x00000000, 0xbc000000, 0x00000000, 0x00003c00, 0x00000000, 0xbc000000, 0x00000000, \
  0x3c00bc00, 0x00000000, 0xbc000000, 0x00000000, 0x3c000000, 0x00000000, 0xbc000000, 0x00000000, \
  0x3c003c00, 0x00000000, 0xbc000000, 0x00000000, 0xbc000000, 0x00003c00, 0xbc000000, 0x00000000, \
  0x0000bc00, 0x00003c00, 0xbc000000, 0x00000000, 0x00000000, 0x00003c00, 0xbc000000, 0x00000000, \
  0x00003c00, 0x00003c00, 0xbc000000, 0x00000000, 0x3c000000, 0x00003c00, 0xbc000000, 0x00000000, \
  0xbc003c00, 0x3c00bc00, 0xbc000000, 0x00000000, 0x00000000, 0x3c00bc00, 0xbc000000, 0x00000000, \
  0xbc000000, 0x3c000000, 0xbc000000, 0x00000000, 0x0000bc00, 0x3c000000, 0xbc000000, 0x00000000, \
  0x00000000, 0x3c000000, 0xbc000000, 0x00000000, 0x00003c00, 0x3c000000, 0xbc000000, 0x00000000, \
  0x3c000000, 0x3c000000, 0xbc000000, 0x00000000, 0xbc00bc00, 0x3c003c00, 0xbc000000, 0x00000000, \
  0x00000000, 0x3c003c00, 0xbc000000, 0x00000000, 0x3c003c00, 0x3c003c00, 0xbc000000, 0x00000000, \
  0xbc000000, 0xbc00bc00, 0xbc003c00, 0x00000000, 0x00003c00, 0xbc00bc00, 0xbc003c00, 0x00000000, \
  0xbc003c00, 0xbc000000, 0xbc003c00, 0x00000000, 0x00000000, 0xbc000000, 0xbc003c00, 0x00000000, \
  0x3c003c00, 0xbc000000, 0xbc003c00, 0x00000000, 0xbc000000, 0xbc003c00, 0xbc003c00, 0x00000000, \
  0x0000bc00, 0xbc003c00, 0xbc003c00, 0x00000000, 0xbc003c00, 0x0000bc00, 0xbc003c00, 0x00000000, \
  0x00000000, 0x0000bc00, 0xbc003c00, 0x00000000, 0x3c003c00, 0x0000bc00, 0xbc003c00, 0x00000000, \
  0xbc000000, 0x00000000, 0xbc003c00, 0x00000000, 0x0000bc00, 0x00000000, 0xbc003c00, 0x00000000, \
  0x00000000, 0x00000000, 0xbc003c00, 0x00000000, 0x00003c00, 0x00000000, 0xbc003c00, 0x00000000, \
  0x3c000000, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00003c00, 0xbc003c00, 0x00000000, \
  0x00000000, 0x00003c00, 0xbc003c00, 0x00000000, 0x00000000, 0x3c00bc00, 0xbc003c00, 0x00000000, \
  0xbc00bc00, 0x3c000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x3c000000, 0xbc003c00, 0x00000000, \
  0x00000000, 0x3c000000, 0xbc003c00, 0x00000000, 0x3c000000, 0x3c000000, 0xbc003c00, 0x00000000, \
  0x3c003c00, 0x3c000000, 0xbc003c00, 0x00000000, 0x0000bc00, 0x3c003c00, 0xbc003c00, 0x00000000, \
  0x0000bc00, 0xbc00bc00, 0x0000bc00, 0x00000000, 0x00000000, 0xbc00bc00, 0x0000bc00, 0x00000000, \
  0xbc000000, 0xbc000000, 0x0000bc00, 0x00000000, 0x0000bc00, 0xbc000000, 0x0000bc00, 0x00000000, \
  0x00000000, 0xbc000000, 0x0000bc00, 0x00000000, 0x00003c00, 0xbc000000, 0x0000bc00, 0x00000000, \
  0x3c00bc00, 0xbc000000, 0x0000bc00, 0x00000000, 0x3c000000, 0xbc000000, 0x0000bc00, 0x00000000, \
  0xbc000000, 0xbc003c00, 0x0000bc00, 0x00000000, 0x00000000, 0xbc003c00, 0x0000bc00, 0x00000000, \
  0x3c00bc00, 0xbc003c00, 0x0000bc00, 0x00000000, 0x3c003c00, 0xbc003c00, 0x0000bc00, 0x00000000, \
  0xbc000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x00000000, \
  0x00000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0x00003c00, 0x0000bc00, 0x0000bc00, 0x00000000, \
  0x3c000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x00000000, \
  0xbc00bc00, 0x00000000, 0x0000bc00, 0x00000000, 0xbc000000, 0x00000000, 0x0000bc00, 0x00000000, \
  0x0000bc00, 0x00000000, 0x0000bc00, 0x00000000, 0x00000000, 0x00000000, 0x0000bc00, 0x00000000, \
  0x00003c00, 0x00000000, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x00000000, \
  0x3c000000, 0x00000000, 0x0000bc00, 0x00000000, 0x3c003c00, 0x00000000, 0x0000bc00, 0x00000000, \
  0xbc000000, 0x00003c00, 0x0000bc00, 0x00000000, 0xbc003c00, 0x00003c00, 0x0000bc00, 0x00000000, \
  0x0000bc00, 0x00003c00, 0x0000bc00, 0x00000000, 0x00000000, 0x00003c00, 0x0000bc00, 0x00000000, \
  0x00003c00, 0x00003c00, 0x0000bc00, 0x00000000, 0x3c000000, 0x00003c00, 0x0000bc00, 0x00000000, \
  0xbc00bc00, 0x3c00bc00, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x0000bc00, 0x00000000, \
  0x3c003c00, 0x3c00bc00, 0x0000bc00, 0x00000000, 0xbc000000, 0x3c000000, 0x0000bc00, 0x00000000, \
  0x0000bc00, 0x3c000000, 0x0000bc00, 0x00000000, 0x00000000, 0x3c000000, 0x0000bc00, 0x00000000, \
  0x00003c00, 0x3c000000, 0x0000bc00, 0x00000000, 0x3c000000, 0x3c000000, 0x0000bc00, 0x00000000, \
  0xbc000000, 0x3c003c00, 0x0000bc00, 0x00000000, 0x0000bc00, 0x3c003c00, 0x0000bc00, 0x00000000, \
  0x00000000, 0x3c003c00, 0x0000bc00, 0x00000000, 0x3c003c00, 0x3c003c00, 0x0000bc00, 0x00000000, \
  0xbc000000, 0xbc00bc00, 0x00000000, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x00000000, 0x00000000, \
  0x0000bc00, 0xbc00bc00, 0x00000000, 0x00000000, 0x00000000, 0xbc00bc00, 0x00000000, 0x00000000, \
  0x00003c00, 0xbc00bc00, 0x00000000, 0x00000000, 0x3c000000, 0xbc00bc00, 0x00000000, 0x00000000, \
  0xbc00bc00, 0xbc000000, 0x00000000, 0x00000000, 0xbc000000, 0xbc000000, 0x00000000, 0x00000000, \
  0xbc003c00, 0xbc000000, 0x00000000, 0x00000000, 0x0000bc00, 0xbc000000, 0x00000000, 0x00000000, \
  0x00000000, 0xbc000000, 0x00000000, 0x00000000, 0x00003c00, 0xbc000000, 0x00000000, 0x00000000, \
  0x3c000000, 0xbc000000, 0x00000000, 0x00000000, 0x3c003c00, 0xbc000000, 0x00000000, 0x00000000, \
  0xbc000000, 0xbc003c00, 0x00000000, 0x00000000, 0x0000bc00, 0xbc003c00, 0x00000000, 0x00000000, \
  0x00000000, 0xbc003c00, 0x00000000, 0x00000000, 0x00003c00, 0xbc003c00, 0x00000000, 0x00000000, \
  0x3c000000, 0xbc003c00, 0x00000000, 0x00000000, 0xbc00bc00, 0x0000bc00, 0x00000000, 0x00000000, \
  0xbc000000, 0x0000bc00, 0x00000000, 0x00000000, 0xbc003c00, 0x0000bc00, 0x00000000, 0x00000000, \
  0x0000bc00, 0x0000bc00, 0x00000000, 0x00000000, 0x00000000, 0x0000bc00, 0x00000000, 0x00000000, \
  0x00003c00, 0x0000bc00, 0x00000000, 0x00000000, 0x3c00bc00, 0x0000bc00, 0x00000000, 0x00000000, \
  0x3c000000, 0x0000bc00, 0x00000000, 0x00000000, 0xbc00bc00, 0x00000000, 0x00000000, 0x00000000, \
  0xbc000000, 0x00000000, 0x00000000, 0x00000000, 0xbc003c00, 0x00000000, 0x00000000, 0x00000000, \
  0x0000bc00, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, \
  0x00003c00, 0x00000000, 0x00000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x00000000, 0x00000000, \
  0x3c000000, 0x00000000, 0x00000000, 0x00000000, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, \
  0xbc00bc00, 0x00003c00, 0x00000000, 0x00000000, 0xbc000000, 0x00003c00, 0x00000000, 0x00000000, \
  0x0000bc00, 0x00003c00, 0x00000000, 0x00000000, 0x00000000, 0x00003c00, 0x00000000, 0x00000000, \
  0x00003c00, 0x00003c00, 0x00000000, 0x00000000, 0x3c00bc00, 0x00003c00, 0x00000000, 0x00000000, \
  0x3c000000, 0x00003c00, 0x00000000, 0x00000000, 0x3c003c00, 0x00003c00, 0x00000000, 0x00000000, \
  0xbc000000, 0x3c00bc00, 0x00000000, 0x00000000, 0x0000bc00, 0x3c00bc00, 0x00000000, 0x00000000, \
  0x00000000, 0x3c00bc00, 0x00000000, 0x00000000, 0x3c000000, 0x3c00bc00, 0x00000000, 0x00000000, \
  0x3c003c00, 0x3c00bc00, 0x00000000, 0x00000000, 0xbc00bc00, 0x3c000000, 0x00000000, 0x00000000, \
  0xbc000000, 0x3c000000, 0x00000000, 0x00000000, 0x0000bc00, 0x3c000000, 0x00000000, 0x00000000, \
  0x00000000, 0x3c000000, 0x00000000, 0x00000000, 0x00003c00, 0x3c000000, 0x00000000, 0x00000000, \
  0x3c00bc00, 0x3c000000, 0x00000000, 0x00000000, 0x3c000000, 0x3c000000, 0x00000000, 0x00000000, \
  0xbc000000, 0x3c003c00, 0x00000000, 0x00000000, 0x0000bc00, 0x3c003c00, 0x00000000, 0x00000000, \
  0x00000000, 0x3c003c00, 0x00000000, 0x00000000, 0x00003c00, 0x3c003c00, 0x00000000, 0x00000000, \
  0x3c000000, 0x3c003c00, 0x00000000, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x00003c00, 0x00000000, \
  0xbc000000, 0xbc00bc00, 0x00003c00, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x00003c00, 0x00000000, \
  0x0000bc00, 0xbc00bc00, 0x00003c00, 0x00000000, 0x00003c00, 0xbc00bc00, 0x00003c00, 0x00000000, \
  0x3c00bc00, 0xbc00bc00, 0x00003c00, 0x00000000, 0x3c000000, 0xbc00bc00, 0x00003c00, 0x00000000, \
  0xbc000000, 0xbc000000, 0x00003c00, 0x00000000, 0x0000bc00, 0xbc000000, 0x00003c00, 0x00000000, \
  0x00000000, 0xbc000000, 0x00003c00, 0x00000000, 0x3c00bc00, 0xbc000000, 0x00003c00, 0x00000000, \
  0x3c000000, 0xbc000000, 0x00003c00, 0x00000000, 0xbc00bc00, 0xbc003c00, 0x00003c00, 0x00000000, \
  0xbc000000, 0xbc003c00, 0x00003c00, 0x00000000, 0xbc003c00, 0xbc003c00, 0x00003c00, 0x00000000, \
  0x0000bc00, 0xbc003c00, 0x00003c00, 0x00000000, 0x00000000, 0xbc003c00, 0x00003c00, 0x00000000, \
  0x00003c00, 0xbc003c00, 0x00003c00, 0x00000000, 0x3c00bc00, 0xbc003c00, 0x00003c00, 0x00000000, \
  0x3c000000, 0xbc003c00, 0x00003c00, 0x00000000, 0xbc000000, 0x0000bc00, 0x00003c00, 0x00000000, \
  0x00000000, 0x0000bc00, 0x00003c00, 0x00000000, 0x00003c00, 0x0000bc00, 0x00003c00, 0x00000000, \
  0x3c00bc00, 0x0000bc00, 0x00003c00, 0x00000000, 0x3c000000, 0x0000bc00, 0x00003c00, 0x00000000, \
  0x3c003c00, 0x0000bc00, 0x00003c00, 0x00000000, 0xbc00bc00, 0x00000000, 0x00003c00, 0x00000000, \
  0xbc000000, 0x00000000, 0x00003c00, 0x00000000, 0xbc003c00, 0x00000000, 0x00003c00, 0x00000000, \
  0x0000bc00, 0x00000000, 0x00003c00, 0x00000000, 0x00000000, 0x00000000, 0x00003c00, 0x00000000, \
  0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x3c00bc00, 0x00000000, 0x00003c00, 0x00000000, \
  0x3c000000, 0x00000000, 0x00003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x00003c00, 0x00000000, \
  0xbc000000, 0x00003c00, 0x00003c00, 0x00000000, 0x0000bc00, 0x00003c00, 0x00003c00, 0x00000000, \
  0x00000000, 0x00003c00, 0x00003c00, 0x00000000, 0x3c000000, 0x00003c00, 0x00003c00, 0x00000000, \
  0xbc003c00, 0x3c00bc00, 0x00003c00, 0x00000000, 0x00000000, 0x3c00bc00, 0x00003c00, 0x00000000, \
  0x00003c00, 0x3c00bc00, 0x00003c00, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x00003c00, 0x00000000, \
  0x3c000000, 0x3c00bc00, 0x00003c00, 0x00000000, 0x3c003c00, 0x3c00bc00, 0x00003c00, 0x00000000, \
  0xbc000000, 0x3c000000, 0x00003c00, 0x00000000, 0x00000000, 0x3c000000, 0x00003c00, 0x00000000, \
  0x3c003c00, 0x3c000000, 0x00003c00, 0x00000000, 0xbc003c00, 0x3c003c00, 0x00003c00, 0x00000000, \
  0x00000000, 0x3c003c00, 0x00003c00, 0x00000000, 0x00003c00, 0x3c003c00, 0x00003c00, 0x00000000, \
  0x3c00bc00, 0x3c003c00, 0x00003c00, 0x00000000, 0x3c000000, 0x3c003c00, 0x00003c00, 0x00000000, \
  0x0000bc00, 0xbc00bc00, 0x3c00bc00, 0x00000000, 0x00000000, 0xbc00bc00, 0x3c00bc00, 0x00000000, \
  0x00003c00, 0xbc00bc00, 0x3c00bc00, 0x00000000, 0x3c000000, 0xbc00bc00, 0x3c00bc00, 0x00000000, \
  0xbc00bc00, 0xbc000000, 0x3c00bc00, 0x00000000, 0x00000000, 0xbc000000, 0x3c00bc00, 0x00000000, \
  0x3c00bc00, 0xbc000000, 0x3c00bc00, 0x00000000, 0xbc000000, 0xbc003c00, 0x3c00bc00, 0x00000000, \
  0x3c003c00, 0xbc003c00, 0x3c00bc00, 0x00000000, 0x00000000, 0x0000bc00, 0x3c00bc00, 0x00000000, \
  0x3c00bc00, 0x0000bc00, 0x3c00bc00, 0x00000000, 0x3c003c00, 0x0000bc00, 0x3c00bc00, 0x00000000, \
  0xbc000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x00000000, \
  0x00000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x00003c00, 0x00000000, 0x3c00bc00, 0x00000000, \
  0x3c00bc00, 0x00000000, 0x3c00bc00, 0x00000000, 0x3c000000, 0x00000000, 0x3c00bc00, 0x00000000, \
  0xbc00bc00, 0x00003c00, 0x3c00bc00, 0x00000000, 0xbc003c00, 0x00003c00, 0x3c00bc00, 0x00000000, \
  0x0000bc00, 0x00003c00, 0x3c00bc00, 0x00000000, 0x00000000, 0x00003c00, 0x3c00bc00, 0x00000000, \
  0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0x3c000000, 0x3c00bc00, 0x3c00bc00, 0x00000000, \
  0xbc00bc00, 0x3c000000, 0x3c00bc00, 0x00000000, 0xbc003c00, 0x3c000000, 0x3c00bc00, 0x00000000, \
  0x00000000, 0x3c000000, 0x3c00bc00, 0x00000000, 0x3c00bc00, 0x3c000000, 0x3c00bc00, 0x00000000, \
  0xbc000000, 0x3c003c00, 0x3c00bc00, 0x00000000, 0x3c000000, 0x3c003c00, 0x3c00bc00, 0x00000000, \
  0xbc000000, 0xbc00bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x3c000000, 0x00000000, \
  0x00000000, 0xbc00bc00, 0x3c000000, 0x00000000, 0x3c003c00, 0xbc00bc00, 0x3c000000, 0x00000000, \
  0xbc000000, 0xbc000000, 0x3c000000, 0x00000000, 0x0000bc00, 0xbc000000, 0x3c000000, 0x00000000, \
  0x00000000, 0xbc000000, 0x3c000000, 0x00000000, 0x00003c00, 0xbc000000, 0x3c000000, 0x00000000, \
  0x3c000000, 0xbc000000, 0x3c000000, 0x00000000, 0x00000000, 0xbc003c00, 0x3c000000, 0x00000000, \
  0xbc000000, 0x0000bc00, 0x3c000000, 0x00000000, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x00000000, \
  0x00000000, 0x0000bc00, 0x3c000000, 0x00000000, 0x00003c00, 0x0000bc00, 0x3c000000, 0x00000000, \
  0x3c000000, 0x0000bc00, 0x3c000000, 0x00000000, 0xbc00bc00, 0x00000000, 0x3c000000, 0x00000000, \
  0xbc000000, 0x00000000, 0x3c000000, 0x00000000, 0xbc003c00, 0x00000000, 0x3c000000, 0x00000000, \
  0x0000bc00, 0x00000000, 0x3c000000, 0x00000000, 0x00000000, 0x00000000, 0x3c000000, 0x00000000, \
  0x00003c00, 0x00000000, 0x3c000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x3c000000, 0x00000000, \
  0x3c000000, 0x00000000, 0x3c000000, 0x00000000, 0x3c003c00, 0x00000000, 0x3c000000, 0x00000000, \
  0xbc000000, 0x00003c00, 0x3c000000, 0x00000000, 0x0000bc00, 0x00003c00, 0x3c000000, 0x00000000, \
  0x00000000, 0x00003c00, 0x3c000000, 0x00000000, 0x00003c00, 0x00003c00, 0x3c000000, 0x00000000, \
  0x3c000000, 0x00003c00, 0x3c000000, 0x00000000, 0xbc000000, 0x3c00bc00, 0x3c000000, 0x00000000, \
  0x00000000, 0x3c00bc00, 0x3c000000, 0x00000000, 0x3c000000, 0x3c00bc00, 0x3c000000, 0x00000000, \
  0xbc000000, 0x3c000000, 0x3c000000, 0x00000000, 0x0000bc00, 0x3c000000, 0x3c000000, 0x00000000, \
  0x00000000, 0x3c000000, 0x3c000000, 0x00000000, 0x00003c00, 0x3c000000, 0x3c000000, 0x00000000, \
  0x3c00bc00, 0x3c000000, 0x3c000000, 0x00000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x00000000, \
  0x00000000, 0x3c003c00, 0x3c000000, 0x00000000, 0x0000bc00, 0xbc00bc00, 0x3c003c00, 0x00000000, \
  0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0x00000000, 0x00000000, 0xbc000000, 0x3c003c00, 0x00000000, \
  0x3c003c00, 0xbc000000, 0x3c003c00, 0x00000000, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0x00000000, \
  0x00000000, 0xbc003c00, 0x3c003c00, 0x00000000, 0x00003c00, 0xbc003c00, 0x3c003c00, 0x00000000, \
  0x3c000000, 0xbc003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x0000bc00, 0x3c003c00, 0x00000000, \
  0x3c00bc00, 0x0000bc00, 0x3c003c00, 0x00000000, 0x3c000000, 0x0000bc00, 0x3c003c00, 0x00000000, \
  0xbc000000, 0x00000000, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x3c003c00, 0x00000000, \
  0x00003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c00bc00, 0x00000000, 0x3c003c00, 0x00000000, \
  0x3c000000, 0x00000000, 0x3c003c00, 0x00000000, 0xbc003c00, 0x00003c00, 0x3c003c00, 0x00000000, \
  0x00000000, 0x00003c00, 0x3c003c00, 0x00000000, 0x3c00bc00, 0x00003c00, 0x3c003c00, 0x00000000, \
  0x3c003c00, 0x00003c00, 0x3c003c00, 0x00000000, 0xbc000000, 0x3c00bc00, 0x3c003c00, 0x00000000, \
  0x3c003c00, 0x3c00bc00, 0x3c003c00, 0x00000000, 0xbc003c00, 0x3c000000, 0x3c003c00, 0x00000000, \
  0x00000000, 0x3c000000, 0x3c003c00, 0x00000000, 0x00003c00, 0x3c000000, 0x3c003c00, 0x00000000, \
  0x3c00bc00, 0x3c000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x3c000000, 0x3c003c00, 0x00000000, \
  0xbc000000, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0xbc00bc00, 0xbc00bc00, 0x00003c00, \
  0x0000bc00, 0xbc000000, 0xbc00bc00, 0x00003c00, 0x00003c00, 0xbc000000, 0xbc00bc00, 0x00003c00, \
  0x3c000000, 0xbc000000, 0xbc00bc00, 0x00003c00, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x00003c00, \
  0x0000bc00, 0x0000bc00, 0xbc00bc00, 0x00003c00, 0xbc00bc00, 0x00000000, 0xbc00bc00, 0x00003c00, \
  0x00000000, 0x00000000, 0xbc00bc00, 0x00003c00, 0x00003c00, 0x00000000, 0xbc00bc00, 0x00003c00, \
  0x3c00bc00, 0x00000000, 0xbc00bc00, 0x00003c00, 0x3c003c00, 0x00000000, 0xbc00bc00, 0x00003c00, \
  0xbc000000, 0x00003c00, 0xbc00bc00, 0x00003c00, 0x0000bc00, 0x00003c00, 0xbc00bc00, 0x00003c00, \
  0x00003c00, 0x00003c00, 0xbc00bc00, 0x00003c00, 0x3c000000, 0x00003c00, 0xbc00bc00, 0x00003c00, \
  0xbc000000, 0x3c00bc00, 0xbc00bc00, 0x00003c00, 0x00003c00, 0x3c000000, 0xbc00bc00, 0x00003c00, \
  0x00000000, 0x3c003c00, 0xbc00bc00, 0x00003c00, 0xbc000000, 0xbc00bc00, 0xbc000000, 0x00003c00, \
  0x0000bc00, 0xbc00bc00, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc00bc00, 0xbc000000, 0x00003c00, \
  0x3c000000, 0xbc00bc00, 0xbc000000, 0x00003c00, 0xbc003c00, 0xbc000000, 0xbc000000, 0x00003c00, \
  0x00000000, 0xbc000000, 0xbc000000, 0x00003c00, 0xbc000000, 0xbc003c00, 0xbc000000, 0x00003c00, \
  0xbc003c00, 0xbc003c00, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc003c00, 0xbc000000, 0x00003c00, \
  0x3c000000, 0xbc003c00, 0xbc000000, 0x00003c00, 0x00000000, 0x0000bc00, 0xbc000000, 0x00003c00, \
  0x3c000000, 0x0000bc00, 0xbc000000, 0x00003c00, 0xbc000000, 0x00000000, 0xbc000000, 0x00003c00, \
  0x00000000, 0x00000000, 0xbc000000, 0x00003c00, 0x00003c00, 0x00000000, 0xbc000000, 0x00003c00, \
  0x3c000000, 0x00000000, 0xbc000000, 0x00003c00, 0x00000000, 0x00003c00, 0xbc000000, 0x00003c00, \
  0x00003c00, 0x00003c00, 0xbc000000, 0x00003c00, 0x3c003c00, 0x00003c00, 0xbc000000, 0x00003c00, \
  0x0000bc00, 0x3c00bc00, 0xbc000000, 0x00003c00, 0x3c003c00, 0x3c00bc00, 0xbc000000, 0x00003c00, \
  0xbc003c00, 0x3c000000, 0xbc000000, 0x00003c00, 0x00000000, 0x3c000000, 0xbc000000, 0x00003c00, \
  0xbc000000, 0x3c003c00, 0xbc000000, 0x00003c00, 0x00003c00, 0x3c003c00, 0xbc000000, 0x00003c00, \
  0x3c000000, 0x3c003c00, 0xbc000000, 0x00003c00, 0xbc000000, 0xbc000000, 0xbc003c00, 0x00003c00, \
  0x00003c00, 0xbc000000, 0xbc003c00, 0x00003c00, 0x3c000000, 0xbc000000, 0xbc003c00, 0x00003c00, \
  0xbc00bc00, 0x0000bc00, 0xbc003c00, 0x00003c00, 0xbc000000, 0x0000bc00, 0xbc003c00, 0x00003c00, \
  0x00003c00, 0x0000bc00, 0xbc003c00, 0x00003c00, 0x00000000, 0x00000000, 0xbc003c00, 0x00003c00, \
  0x00003c00, 0x00000000, 0xbc003c00, 0x00003c00, 0x3c00bc00, 0x00000000, 0xbc003c00, 0x00003c00, \
  0xbc00bc00, 0x00003c00, 0xbc003c00, 0x00003c00, 0x00000000, 0x3c00bc00, 0xbc003c00, 0x00003c00, \
  0xbc000000, 0x3c000000, 0xbc003c00, 0x00003c00, 0x00003c00, 0x3c000000, 0xbc003c00, 0x00003c00, \
  0x00000000, 0x3c003c00, 0xbc003c00, 0x00003c00, 0xbc000000, 0xbc000000, 0x0000bc00, 0x00003c00, \
  0xbc003c00, 0xbc000000, 0x0000bc00, 0x00003c00, 0x00000000, 0xbc000000, 0x0000bc00, 0x00003c00, \
  0x00003c00, 0xbc000000, 0x0000bc00, 0x00003c00, 0x3c003c00, 0xbc000000, 0x0000bc00, 0x00003c00, \
  0xbc000000, 0xbc003c00, 0x0000bc00, 0x00003c00, 0x00003c00, 0xbc003c00, 0x0000bc00, 0x00003c00, \
  0x3c000000, 0xbc003c00, 0x0000bc00, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x0000bc00, 0x00003c00, \
  0xbc003c00, 0x0000bc00, 0x0000bc00, 0x00003c00, 0x00000000, 0x0000bc00, 0x0000bc00, 0x00003c00, \
  0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x00003c00, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x00003c00, \
  0xbc000000, 0x00000000, 0x0000bc00, 0x00003c00, 0x0000bc00, 0x00000000, 0x0000bc00, 0x00003c00, \
  0x00000000, 0x00000000, 0x0000bc00, 0x00003c00, 0x00003c00, 0x00000000, 0x0000bc00, 0x00003c00, \
  0x3c000000, 0x00000000, 0x0000bc00, 0x00003c00, 0x3c003c00, 0x00000000, 0x0000bc00, 0x00003c00, \
  0xbc00bc00, 0x00003c00, 0x0000bc00, 0x00003c00, 0xbc003c00, 0x00003c00, 0x0000bc00, 0x00003c00, \
  0x00000000, 0x00003c00, 0x0000bc00, 0x00003c00, 0x0000bc00, 0x3c00bc00, 0x0000bc00, 0x00003c00, \
  0x00000000, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x00003c00, \
  0xbc00bc00, 0x3c000000, 0x0000bc00, 0x00003c00, 0xbc003c00, 0x3c000000, 0x0000bc00, 0x00003c00, \
  0x0000bc00, 0x3c000000, 0x0000bc00, 0x00003c00, 0x00000000, 0x3c000000, 0x0000bc00, 0x00003c00, \
  0x00003c00, 0x3c000000, 0x0000bc00, 0x00003c00, 0x3c00bc00, 0x3c000000, 0x0000bc00, 0x00003c00, \
  0x3c003c00, 0x3c000000, 0x0000bc00, 0x00003c00, 0xbc000000, 0x3c003c00, 0x0000bc00, 0x00003c00, \
  0x0000bc00, 0x3c003c00, 0x0000bc00, 0x00003c00, 0x3c000000, 0x3c003c00, 0x0000bc00, 0x00003c00, \
  0x00000000, 0xbc00bc00, 0x00000000, 0x00003c00, 0x3c00bc00, 0xbc00bc00, 0x00000000, 0x00003c00, \
  0x3c003c00, 0xbc00bc00, 0x00000000, 0x00003c00, 0xbc000000, 0xbc000000, 0x00000000, 0x00003c00, \
  0x00000000, 0xbc000000, 0x00000000, 0x00003c00, 0x00003c00, 0xbc000000, 0x00000000, 0x00003c00, \
  0x3c000000, 0xbc000000, 0x00000000, 0x00003c00, 0x0000bc00, 0x0000bc00, 0x00000000, 0x00003c00, \
  0x00000000, 0x0000bc00, 0x00000000, 0x00003c00, 0x00003c00, 0x0000bc00, 0x00000000, 0x00003c00, \
  0x3c000000, 0x0000bc00, 0x00000000, 0x00003c00, 0xbc00bc00, 0x00000000, 0x00000000, 0x00003c00, \
  0xbc000000, 0x00000000, 0x00000000, 0x00003c00, 0x0000bc00, 0x00000000, 0x00000000, 0x00003c00, \
  0x00000000, 0x00000000, 0x00000000, 0x00003c00, 0x00003c00, 0x00000000, 0x00000000, 0x00003c00, \
  0x3c000000, 0x00000000, 0x00000000, 0x00003c00, 0xbc000000, 0x00003c00, 0x00000000, 0x00003c00, \
  0x0000bc00, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, \
  0x00003c00, 0x00003c00, 0x00000000, 0x00003c00, 0x3c000000, 0x00003c00, 0x00000000, 0x00003c00, \
  0x00003c00, 0x3c00bc00, 0x00000000, 0x00003c00, 0x3c000000, 0x3c00bc00, 0x00000000, 0x00003c00, \
  0x3c003c00, 0x3c00bc00, 0x00000000, 0x00003c00, 0xbc000000, 0x3c000000, 0x00000000, 0x00003c00, \
  0x00000000, 0x3c000000, 0x00000000, 0x00003c00, 0x00003c00, 0x3c000000, 0x00000000, 0x00003c00, \
  0x3c000000, 0x3c000000, 0x00000000, 0x00003c00, 0x3c003c00, 0x3c000000, 0x00000000, 0x00003c00, \
  0xbc003c00, 0x3c003c00, 0x00000000, 0x00003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x00003c00, \
  0x00003c00, 0x3c003c00, 0x00000000, 0x00003c00, 0x3c00bc00, 0x3c003c00, 0x00000000, 0x00003c00, \
  0xbc003c00, 0xbc00bc00, 0x00003c00, 0x00003c00, 0x3c000000, 0xbc00bc00, 0x00003c00, 0x00003c00, \
  0x00000000, 0xbc000000, 0x00003c00, 0x00003c00, 0xbc00bc00, 0xbc003c00, 0x00003c00, 0x00003c00, \
  0x00003c00, 0xbc003c00, 0x00003c00, 0x00003c00, 0x3c00bc00, 0xbc003c00, 0x00003c00, 0x00003c00, \
  0x3c000000, 0xbc003c00, 0x00003c00, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x00003c00, 0x00003c00, \
  0x00000000, 0x0000bc00, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0x00003c00, \
  0x3c003c00, 0x0000bc00, 0x00003c00, 0x00003c00, 0xbc000000, 0x00000000, 0x00003c00, 0x00003c00, \
  0x0000bc00, 0x00000000, 0x00003c00, 0x00003c00, 0x00000000, 0x00000000, 0x00003c00, 0x00003c00, \
  0x00003c00, 0x00000000, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x00000000, 0x00003c00, 0x00003c00, \
  0x3c003c00, 0x00000000, 0x00003c00, 0x00003c00, 0xbc00bc00, 0x00003c00, 0x00003c00, 0x00003c00, \
  0x00000000, 0x00003c00, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x00003c00, 0x00003c00, 0x00003c00, \
  0xbc00bc00, 0x3c00bc00, 0x00003c00, 0x00003c00, 0xbc003c00, 0x3c00bc00, 0x00003c00, 0x00003c00, \
  0x00000000, 0x3c00bc00, 0x00003c00, 0x00003c00, 0x3c003c00, 0x3c00bc00, 0x00003c00, 0x00003c00, \
  0x0000bc00, 0x3c000000, 0x00003c00, 0x00003c00, 0x00003c00, 0x3c000000, 0x00003c00, 0x00003c00, \
  0x3c00bc00, 0x3c000000, 0x00003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0x00003c00, 0x00003c00, \
  0xbc00bc00, 0x3c003c00, 0x00003c00, 0x00003c00, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x00003c00, \
  0x00003c00, 0x3c003c00, 0x00003c00, 0x00003c00, 0x3c003c00, 0x3c003c00, 0x00003c00, 0x00003c00, \
  0x00003c00, 0xbc000000, 0x3c00bc00, 0x00003c00, 0x3c000000, 0xbc000000, 0x3c00bc00, 0x00003c00, \
  0x00000000, 0xbc003c00, 0x3c00bc00, 0x00003c00, 0xbc000000, 0x0000bc00, 0x3c00bc00, 0x00003c00, \
  0xbc003c00, 0x00000000, 0x3c00bc00, 0x00003c00, 0x00000000, 0x00000000, 0x3c00bc00, 0x00003c00, \
  0x3c003c00, 0x00000000, 0x3c00bc00, 0x00003c00, 0xbc000000, 0x00003c00, 0x3c00bc00, 0x00003c00, \
  0x3c000000, 0x00003c00, 0x3c00bc00, 0x00003c00, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x00003c00, \
  0xbc000000, 0x3c000000, 0x3c00bc00, 0x00003c00, 0x3c00bc00, 0x3c000000, 0x3c00bc00, 0x00003c00, \
  0x00003c00, 0x3c003c00, 0x3c00bc00, 0x00003c00, 0xbc000000, 0xbc00bc00, 0x3c000000, 0x00003c00, \
  0x0000bc00, 0xbc00bc00, 0x3c000000, 0x00003c00, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x00003c00, \
  0x00000000, 0xbc000000, 0x3c000000, 0x00003c00, 0xbc000000, 0xbc003c00, 0x3c000000, 0x00003c00, \
  0x0000bc00, 0xbc003c00, 0x3c000000, 0x00003c00, 0x00003c00, 0xbc003c00, 0x3c000000, 0x00003c00, \
  0x3c000000, 0xbc003c00, 0x3c000000, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x3c000000, 0x00003c00, \
  0xbc000000, 0x0000bc00, 0x3c000000, 0x00003c00, 0x00000000, 0x0000bc00, 0x3c000000, 0x00003c00, \
  0x00003c00, 0x0000bc00, 0x3c000000, 0x00003c00, 0x3c00bc00, 0x0000bc00, 0x3c000000, 0x00003c00, \
  0xbc000000, 0x00000000, 0x3c000000, 0x00003c00, 0x0000bc00, 0x00000000, 0x3c000000, 0x00003c00, \
  0x00000000, 0x00000000, 0x3c000000, 0x00003c00, 0x00003c00, 0x00000000, 0x3c000000, 0x00003c00, \
  0x3c000000, 0x00000000, 0x3c000000, 0x00003c00, 0xbc00bc00, 0x00003c00, 0x3c000000, 0x00003c00, \
  0x00000000, 0x00003c00, 0x3c000000, 0x00003c00, 0x3c003c00, 0x00003c00, 0x3c000000, 0x00003c00, \
  0xbc003c00, 0x3c00bc00, 0x3c000000, 0x00003c00, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0x00003c00, \
  0x3c003c00, 0x3c00bc00, 0x3c000000, 0x00003c00, 0x00000000, 0x3c000000, 0x3c000000, 0x00003c00, \
  0xbc000000, 0x3c003c00, 0x3c000000, 0x00003c00, 0x0000bc00, 0x3c003c00, 0x3c000000, 0x00003c00, \
  0x00000000, 0x3c003c00, 0x3c000000, 0x00003c00, 0x3c000000, 0x3c003c00, 0x3c000000, 0x00003c00, \
  0xbc000000, 0xbc000000, 0x3c003c00, 0x00003c00, 0x00003c00, 0xbc000000, 0x3c003c00, 0x00003c00, \
  0x3c00bc00, 0xbc000000, 0x3c003c00, 0x00003c00, 0xbc000000, 0x0000bc00, 0x3c003c00, 0x00003c00, \
  0x0000bc00, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x3c000000, 0x0000bc00, 0x3c003c00, 0x00003c00, \
  0xbc00bc00, 0x00000000, 0x3c003c00, 0x00003c00, 0x00000000, 0x00000000, 0x3c003c00, 0x00003c00, \
  0x3c00bc00, 0x00000000, 0x3c003c00, 0x00003c00, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00003c00, \
  0x0000bc00, 0x00003c00, 0x3c003c00, 0x00003c00, 0x00000000, 0x00003c00, 0x3c003c00, 0x00003c00, \
  0x3c000000, 0x00003c00, 0x3c003c00, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x3c003c00, 0x00003c00, \
  0x0000bc00, 0x3c000000, 0x3c003c00, 0x00003c00, 0x3c00bc00, 0x3c000000, 0x3c003c00, 0x00003c00, \
  0x3c003c00, 0x3c000000, 0x3c003c00, 0x00003c00, 0x00003c00, 0x3c003c00, 0x3c003c00, 0x00003c00, \
  0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, \
  0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, \
  0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, \
  0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0x3c00bc00, \
  0x00000000, 0x0000bc00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0x00000000, 0xbc00bc00, 0x3c00bc00, \
  0xbc000000, 0x00000000, 0xbc00bc00, 0x3c00bc00, 0x0000bc00, 0x00000000, 0xbc00bc00, 0x3c00bc00, \
  0x00003c00, 0x00000000, 0xbc00bc00, 0x3c00bc00, 0x3c000000, 0x00000000, 0xbc00bc00, 0x3c00bc00, \
  0x00000000, 0x00003c00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, \
  0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, \
  0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x00000000, 0x3c000000, 0xbc00bc00, 0x3c00bc00, \
  0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, \
  0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, \
  0x00000000, 0xbc00bc00, 0xbc000000, 0x3c00bc00, 0xbc000000, 0xbc000000, 0xbc000000, 0x3c00bc00, \
  0x0000bc00, 0xbc000000, 0xbc000000, 0x3c00bc00, 0x00003c00, 0xbc000000, 0xbc000000, 0x3c00bc00, \
  0x3c000000, 0xbc000000, 0xbc000000, 0x3c00bc00, 0x00000000, 0xbc003c00, 0xbc000000, 0x3c00bc00, \
  0xbc000000, 0x0000bc00, 0xbc000000, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0xbc000000, 0x3c00bc00, \
  0x3c000000, 0x0000bc00, 0xbc000000, 0x3c00bc00, 0xbc00bc00, 0x00000000, 0xbc000000, 0x3c00bc00, \
  0xbc003c00, 0x00000000, 0xbc000000, 0x3c00bc00, 0x00000000, 0x00000000, 0xbc000000, 0x3c00bc00, \
  0x00003c00, 0x00000000, 0xbc000000, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0xbc000000, 0x3c00bc00, \
  0x3c000000, 0x00000000, 0xbc000000, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0xbc000000, 0x3c00bc00, \
  0x00003c00, 0x00003c00, 0xbc000000, 0x3c00bc00, 0x3c000000, 0x00003c00, 0xbc000000, 0x3c00bc00, \
  0x00000000, 0x3c00bc00, 0xbc000000, 0x3c00bc00, 0x3c000000, 0x3c00bc00, 0xbc000000, 0x3c00bc00, \
  0x0000bc00, 0x3c000000, 0xbc000000, 0x3c00bc00, 0x00003c00, 0x3c000000, 0xbc000000, 0x3c00bc00, \
  0x3c000000, 0x3c000000, 0xbc000000, 0x3c00bc00, 0x00000000, 0x3c003c00, 0xbc000000, 0x3c00bc00, \
  0xbc00bc00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, \
  0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, \
  0x00000000, 0xbc000000, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, \
  0xbc003c00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, \
  0x3c003c00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, 0xbc000000, 0x00000000, 0xbc003c00, 0x3c00bc00, \
  0x0000bc00, 0x00000000, 0xbc003c00, 0x3c00bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0x3c00bc00, \
  0x00000000, 0x00003c00, 0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, \
  0xbc003c00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, \
  0x3c003c00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0x00000000, 0x3c000000, 0xbc003c00, 0x3c00bc00, \
  0xbc00bc00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, \
  0x3c00bc00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, \
  0x0000bc00, 0xbc000000, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0xbc000000, 0x0000bc00, 0x3c00bc00, \
  0xbc000000, 0x0000bc00, 0x0000bc00, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x3c00bc00, \
  0xbc000000, 0x00000000, 0x0000bc00, 0x3c00bc00, 0x00000000, 0x00000000, 0x0000bc00, 0x3c00bc00, \
  0x3c003c00, 0x00000000, 0x0000bc00, 0x3c00bc00, 0xbc000000, 0x00003c00, 0x0000bc00, 0x3c00bc00, \
  0x0000bc00, 0x00003c00, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0x00003c00, 0x0000bc00, 0x3c00bc00, \
  0x0000bc00, 0x3c000000, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0x3c000000, 0x0000bc00, 0x3c00bc00, \
  0xbc000000, 0xbc00bc00, 0x00000000, 0x3c00bc00, 0x3c000000, 0xbc00bc00, 0x00000000, 0x3c00bc00, \
  0xbc003c00, 0xbc000000, 0x00000000, 0x3c00bc00, 0x00000000, 0xbc000000, 0x00000000, 0x3c00bc00, \
  0x3c003c00, 0xbc000000, 0x00000000, 0x3c00bc00, 0x00003c00, 0xbc003c00, 0x00000000, 0x3c00bc00, \
  0x3c000000, 0xbc003c00, 0x00000000, 0x3c00bc00, 0xbc00bc00, 0x0000bc00, 0x00000000, 0x3c00bc00, \
  0xbc000000, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x00000000, 0x3c00bc00, \
  0x3c00bc00, 0x0000bc00, 0x00000000, 0x3c00bc00, 0xbc000000, 0x00000000, 0x00000000, 0x3c00bc00, \
  0x0000bc00, 0x00000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x00000000, 0x00000000, 0x3c00bc00, \
  0x00003c00, 0x00000000, 0x00000000, 0x3c00bc00, 0x3c000000, 0x00000000, 0x00000000, 0x3c00bc00, \
  0x3c003c00, 0x00000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x00003c00, 0x00000000, 0x3c00bc00, \
  0x00003c00, 0x00003c00, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x00003c00, 0x00000000, 0x3c00bc00, \
  0x3c003c00, 0x00003c00, 0x00000000, 0x3c00bc00, 0xbc000000, 0x3c00bc00, 0x00000000, 0x3c00bc00, \
  0x0000bc00, 0x3c00bc00, 0x00000000, 0x3c00bc00, 0x00003c00, 0x3c00bc00, 0x00000000, 0x3c00bc00, \
  0x3c000000, 0x3c00bc00, 0x00000000, 0x3c00bc00, 0xbc00bc00, 0x3c000000, 0x00000000, 0x3c00bc00, \
  0xbc003c00, 0x3c000000, 0x00000000, 0x3c00bc00, 0x00000000, 0x3c000000, 0x00000000, 0x3c00bc00, \
  0x3c00bc00, 0x3c000000, 0x00000000, 0x3c00bc00, 0x00003c00, 0x3c003c00, 0x00000000, 0x3c00bc00, \
  0xbc000000, 0xbc000000, 0x00003c00, 0x3c00bc00, 0x00003c00, 0xbc000000, 0x00003c00, 0x3c00bc00, \
  0x3c000000, 0xbc000000, 0x00003c00, 0x3c00bc00, 0x00000000, 0xbc003c00, 0x00003c00, 0x3c00bc00, \
  0xbc000000, 0x0000bc00, 0x00003c00, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x00003c00, 0x3c00bc00, \
  0x3c000000, 0x0000bc00, 0x00003c00, 0x3c00bc00, 0x3c003c00, 0x0000bc00, 0x00003c00, 0x3c00bc00, \
  0xbc00bc00, 0x00000000, 0x00003c00, 0x3c00bc00, 0x00000000, 0x00000000, 0x00003c00, 0x3c00bc00, \
  0x3c000000, 0x00000000, 0x00003c00, 0x3c00bc00, 0x3c003c00, 0x00000000, 0x00003c00, 0x3c00bc00, \
  0xbc000000, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x00003c00, 0x00003c00, 0x00003c00, 0x3c00bc00, \
  0x3c003c00, 0x00003c00, 0x00003c00, 0x3c00bc00, 0x00000000, 0x3c00bc00, 0x00003c00, 0x3c00bc00, \
  0xbc000000, 0x3c000000, 0x00003c00, 0x3c00bc00, 0x3c003c00, 0x3c000000, 0x00003c00, 0x3c00bc00, \
  0x0000bc00, 0x3c003c00, 0x00003c00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, \
  0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, \
  0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0xbc000000, 0x3c00bc00, 0x3c00bc00, \
  0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, \
  0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, 0x3c00bc00, 0x3c00bc00, \
  0xbc000000, 0x0000bc00, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x3c00bc00, 0x3c00bc00, \
  0xbc000000, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x0000bc00, 0x00000000, 0x3c00bc00, 0x3c00bc00, \
  0x3c000000, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0x00003c00, 0x3c00bc00, 0x3c00bc00, \
  0x3c000000, 0x00003c00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, \
  0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, \
  0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0x3c000000, 0x3c00bc00, 0x3c00bc00, \
  0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, \
  0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, \
  0x00000000, 0xbc00bc00, 0x3c000000, 0x3c00bc00, 0x00003c00, 0xbc00bc00, 0x3c000000, 0x3c00bc00, \
  0xbc000000, 0xbc000000, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0xbc000000, 0x3c000000, 0x3c00bc00, \
  0x00003c00, 0xbc000000, 0x3c000000, 0x3c00bc00, 0x00000000, 0xbc003c00, 0x3c000000, 0x3c00bc00, \
  0xbc000000, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x3c00bc00, \
  0x00003c00, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0x3c000000, 0x0000bc00, 0x3c000000, 0x3c00bc00, \
  0xbc00bc00, 0x00000000, 0x3c000000, 0x3c00bc00, 0xbc003c00, 0x00000000, 0x3c000000, 0x3c00bc00, \
  0x00000000, 0x00000000, 0x3c000000, 0x3c00bc00, 0x3c003c00, 0x00000000, 0x3c000000, 0x3c00bc00, \
  0xbc000000, 0x00003c00, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0x3c000000, 0x3c00bc00, \
  0x00000000, 0x3c00bc00, 0x3c000000, 0x3c00bc00, 0x00003c00, 0x3c000000, 0x3c000000, 0x3c00bc00, \
  0x3c000000, 0x3c000000, 0x3c000000, 0x3c00bc00, 0x00000000, 0x3c003c00, 0x3c000000, 0x3c00bc00, \
  0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, \
  0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, \
  0x00000000, 0xbc000000, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, \
  0xbc003c00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, \
  0x3c003c00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x3c003c00, 0x3c00bc00, \
  0xbc000000, 0x00000000, 0x3c003c00, 0x3c00bc00, 0x0000bc00, 0x00000000, 0x3c003c00, 0x3c00bc00, \
  0x00003c00, 0x00000000, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, \
  0xbc003c00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, \
  0x3c003c00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0x00000000, 0x3c000000, 0x3c003c00, 0x3c00bc00, \
  0xbc00bc00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, \
  0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, \
  0x00000000, 0xbc00bc00, 0xbc00bc00, 0x3c000000, 0xbc000000, 0xbc000000, 0xbc00bc00, 0x3c000000, \
  0x00003c00, 0xbc000000, 0xbc00bc00, 0x3c000000, 0x3c00bc00, 0xbc000000, 0xbc00bc00, 0x3c000000, \
  0x3c000000, 0xbc000000, 0xbc00bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0xbc00bc00, 0x3c000000, \
  0xbc000000, 0x0000bc00, 0xbc00bc00, 0x3c000000, 0x00003c00, 0x0000bc00, 0xbc00bc00, 0x3c000000, \
  0x3c000000, 0x0000bc00, 0xbc00bc00, 0x3c000000, 0x00000000, 0x00000000, 0xbc00bc00, 0x3c000000, \
  0x3c00bc00, 0x00000000, 0xbc00bc00, 0x3c000000, 0x3c003c00, 0x00000000, 0xbc00bc00, 0x3c000000, \
  0x3c000000, 0x00003c00, 0xbc00bc00, 0x3c000000, 0x3c003c00, 0x00003c00, 0xbc00bc00, 0x3c000000, \
  0x00000000, 0x3c00bc00, 0xbc00bc00, 0x3c000000, 0xbc000000, 0x3c000000, 0xbc00bc00, 0x3c000000, \
  0x0000bc00, 0x3c000000, 0xbc00bc00, 0x3c000000, 0x00003c00, 0x3c000000, 0xbc00bc00, 0x3c000000, \
  0x3c000000, 0x3c000000, 0xbc00bc00, 0x3c000000, 0x00000000, 0x3c003c00, 0xbc00bc00, 0x3c000000, \
  0xbc000000, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x0000bc00, 0xbc00bc00, 0xbc000000, 0x3c000000, \
  0x00003c00, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x3c000000, 0xbc00bc00, 0xbc000000, 0x3c000000, \
  0xbc00bc00, 0xbc000000, 0xbc000000, 0x3c000000, 0x00000000, 0xbc000000, 0xbc000000, 0x3c000000, \
  0x3c00bc00, 0xbc000000, 0xbc000000, 0x3c000000, 0x3c003c00, 0xbc000000, 0xbc000000, 0x3c000000, \
  0xbc000000, 0xbc003c00, 0xbc000000, 0x3c000000, 0x0000bc00, 0xbc003c00, 0xbc000000, 0x3c000000, \
  0x00003c00, 0xbc003c00, 0xbc000000, 0x3c000000, 0x3c000000, 0xbc003c00, 0xbc000000, 0x3c000000, \
  0xbc00bc00, 0x0000bc00, 0xbc000000, 0x3c000000, 0x00000000, 0x0000bc00, 0xbc000000, 0x3c000000, \
  0xbc00bc00, 0x00000000, 0xbc000000, 0x3c000000, 0xbc000000, 0x00000000, 0xbc000000, 0x3c000000, \
  0x0000bc00, 0x00000000, 0xbc000000, 0x3c000000, 0x00000000, 0x00000000, 0xbc000000, 0x3c000000, \
  0x00003c00, 0x00000000, 0xbc000000, 0x3c000000, 0x3c000000, 0x00000000, 0xbc000000, 0x3c000000, \
  0xbc003c00, 0x00003c00, 0xbc000000, 0x3c000000, 0x00000000, 0x00003c00, 0xbc000000, 0x3c000000, \
  0x0000bc00, 0x3c00bc00, 0xbc000000, 0x3c000000, 0x00003c00, 0x3c00bc00, 0xbc000000, 0x3c000000, \
  0xbc003c00, 0x3c000000, 0xbc000000, 0x3c000000, 0x00000000, 0x3c000000, 0xbc000000, 0x3c000000, \
  0x3c00bc00, 0x3c000000, 0xbc000000, 0x3c000000, 0xbc000000, 0x3c003c00, 0xbc000000, 0x3c000000, \
  0x0000bc00, 0x3c003c00, 0xbc000000, 0x3c000000, 0x00003c00, 0x3c003c00, 0xbc000000, 0x3c000000, \
  0x3c000000, 0x3c003c00, 0xbc000000, 0x3c000000, 0x00000000, 0xbc00bc00, 0xbc003c00, 0x3c000000, \
  0xbc000000, 0xbc000000, 0xbc003c00, 0x3c000000, 0x0000bc00, 0xbc000000, 0xbc003c00, 0x3c000000, \
  0x3c000000, 0xbc000000, 0xbc003c00, 0x3c000000, 0x00000000, 0xbc003c00, 0xbc003c00, 0x3c000000, \
  0x0000bc00, 0x0000bc00, 0xbc003c00, 0x3c000000, 0x00003c00, 0x0000bc00, 0xbc003c00, 0x3c000000, \
  0x3c000000, 0x0000bc00, 0xbc003c00, 0x3c000000, 0xbc00bc00, 0x00000000, 0xbc003c00, 0x3c000000, \
  0xbc003c00, 0x00000000, 0xbc003c00, 0x3c000000, 0x00000000, 0x00000000, 0xbc003c00, 0x3c000000, \
  0x3c00bc00, 0x00000000, 0xbc003c00, 0x3c000000, 0x00003c00, 0x00003c00, 0xbc003c00, 0x3c000000, \
  0x3c000000, 0x00003c00, 0xbc003c00, 0x3c000000, 0x00000000, 0x3c00bc00, 0xbc003c00, 0x3c000000, \
  0x0000bc00, 0x3c000000, 0xbc003c00, 0x3c000000, 0x00003c00, 0x3c000000, 0xbc003c00, 0x3c000000, \
  0x3c000000, 0x3c003c00, 0xbc003c00, 0x3c000000, 0xbc000000, 0xbc00bc00, 0x0000bc00, 0x3c000000, \
  0x0000bc00, 0xbc00bc00, 0x0000bc00, 0x3c000000, 0x00003c00, 0xbc00bc00, 0x0000bc00, 0x3c000000, \
  0xbc00bc00, 0xbc000000, 0x0000bc00, 0x3c000000, 0x00000000, 0xbc000000, 0x0000bc00, 0x3c000000, \
  0x3c00bc00, 0xbc000000, 0x0000bc00, 0x3c000000, 0x00003c00, 0xbc003c00, 0x0000bc00, 0x3c000000, \
  0xbc00bc00, 0x0000bc00, 0x0000bc00, 0x3c000000, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x3c000000, \
  0xbc000000, 0x00000000, 0x0000bc00, 0x3c000000, 0x0000bc00, 0x00000000, 0x0000bc00, 0x3c000000, \
  0x00000000, 0x00000000, 0x0000bc00, 0x3c000000, 0x00003c00, 0x00000000, 0x0000bc00, 0x3c000000, \
  0x3c00bc00, 0x00000000, 0x0000bc00, 0x3c000000, 0x3c000000, 0x00000000, 0x0000bc00, 0x3c000000, \
  0xbc00bc00, 0x00003c00, 0x0000bc00, 0x3c000000, 0xbc000000, 0x00003c00, 0x0000bc00, 0x3c000000, \
  0xbc003c00, 0x00003c00, 0x0000bc00, 0x3c000000, 0x00000000, 0x00003c00, 0x0000bc00, 0x3c000000, \
  0x0000bc00, 0x3c00bc00, 0x0000bc00, 0x3c000000, 0x00003c00, 0x3c00bc00, 0x0000bc00, 0x3c000000, \
  0xbc003c00, 0x3c000000, 0x0000bc00, 0x3c000000, 0x0000bc00, 0x3c000000, 0x0000bc00, 0x3c000000, \
  0x00000000, 0x3c000000, 0x0000bc00, 0x3c000000, 0x3c00bc00, 0x3c000000, 0x0000bc00, 0x3c000000, \
  0xbc000000, 0x3c003c00, 0x0000bc00, 0x3c000000, 0x3c000000, 0x3c003c00, 0x0000bc00, 0x3c000000, \
  0xbc00bc00, 0xbc00bc00, 0x00000000, 0x3c000000, 0x00000000, 0xbc00bc00, 0x00000000, 0x3c000000, \
  0x3c00bc00, 0xbc00bc00, 0x00000000, 0x3c000000, 0x3c003c00, 0xbc00bc00, 0x00000000, 0x3c000000, \
  0xbc00bc00, 0xbc000000, 0x00000000, 0x3c000000, 0xbc000000, 0xbc000000, 0x00000000, 0x3c000000, \
  0x0000bc00, 0xbc000000, 0x00000000, 0x3c000000, 0x00000000, 0xbc000000, 0x00000000, 0x3c000000, \
  0x00003c00, 0xbc000000, 0x00000000, 0x3c000000, 0x3c000000, 0xbc000000, 0x00000000, 0x3c000000, \
  0xbc000000, 0xbc003c00, 0x00000000, 0x3c000000, 0x00000000, 0xbc003c00, 0x00000000, 0x3c000000, \
  0x3c000000, 0xbc003c00, 0x00000000, 0x3c000000, 0x3c003c00, 0xbc003c00, 0x00000000, 0x3c000000, \
  0xbc000000, 0x0000bc00, 0x00000000, 0x3c000000, 0x0000bc00, 0x0000bc00, 0x00000000, 0x3c000000, \
  0x00000000, 0x0000bc00, 0x00000000, 0x3c000000, 0x00003c00, 0x0000bc00, 0x00000000, 0x3c000000, \
  0x3c000000, 0x0000bc00, 0x00000000, 0x3c000000, 0xbc00bc00, 0x00000000, 0x00000000, 0x3c000000, \
  0xbc000000, 0x00000000, 0x00000000, 0x3c000000, 0xbc003c00, 0x00000000, 0x00000000, 0x3c000000, \
  0x0000bc00, 0x00000000, 0x00000000, 0x3c000000, 0x00000000, 0x00000000, 0x00000000, 0x3c000000, \
  0x00003c00, 0x00000000, 0x00000000, 0x3c000000, 0x3c00bc00, 0x00000000, 0x00000000, 0x3c000000, \
  0x3c000000, 0x00000000, 0x00000000, 0x3c000000, 0x3c003c00, 0x00000000, 0x00000000, 0x3c000000, \
  0xbc000000, 0x00003c00, 0x00000000, 0x3c000000, 0x0000bc00, 0x00003c00, 0x00000000, 0x3c000000, \
  0x00000000, 0x00003c00, 0x00000000, 0x3c000000, 0x00003c00, 0x00003c00, 0x00000000, 0x3c000000, \
  0x3c000000, 0x00003c00, 0x00000000, 0x3c000000, 0xbc000000, 0x3c00bc00, 0x00000000, 0x3c000000, \
  0x00000000, 0x3c00bc00, 0x00000000, 0x3c000000, 0x3c00bc00, 0x3c00bc00, 0x00000000, 0x3c000000, \
  0xbc000000, 0x3c000000, 0x00000000, 0x3c000000, 0xbc003c00, 0x3c000000, 0x00000000, 0x3c000000, \
  0x0000bc00, 0x3c000000, 0x00000000, 0x3c000000, 0x00000000, 0x3c000000, 0x00000000, 0x3c000000, \
  0x00003c00, 0x3c000000, 0x00000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x00000000, 0x3c000000, \
  0x3c003c00, 0x3c000000, 0x00000000, 0x3c000000, 0xbc00bc00, 0x3c003c00, 0x00000000, 0x3c000000, \
  0xbc003c00, 0x3c003c00, 0x00000000, 0x3c000000, 0x00000000, 0x3c003c00, 0x00000000, 0x3c000000, \
  0x3c00bc00, 0x3c003c00, 0x00000000, 0x3c000000, 0x3c003c00, 0x3c003c00, 0x00000000, 0x3c000000, \
  0xbc000000, 0xbc00bc00, 0x00003c00, 0x3c000000, 0x0000bc00, 0xbc00bc00, 0x00003c00, 0x3c000000, \
  0xbc00bc00, 0xbc000000, 0x00003c00, 0x3c000000, 0x00000000, 0xbc000000, 0x00003c00, 0x3c000000, \
  0x3c000000, 0xbc000000, 0x00003c00, 0x3c000000, 0xbc00bc00, 0xbc003c00, 0x00003c00, 0x3c000000, \
  0x00003c00, 0xbc003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0xbc003c00, 0x00003c00, 0x3c000000, \
  0x00000000, 0x0000bc00, 0x00003c00, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x00003c00, 0x3c000000, \
  0x3c000000, 0x0000bc00, 0x00003c00, 0x3c000000, 0xbc000000, 0x00000000, 0x00003c00, 0x3c000000, \
  0xbc003c00, 0x00000000, 0x00003c00, 0x3c000000, 0x00000000, 0x00000000, 0x00003c00, 0x3c000000, \
  0x00003c00, 0x00000000, 0x00003c00, 0x3c000000, 0x3c000000, 0x00000000, 0x00003c00, 0x3c000000, \
  0x00000000, 0x00003c00, 0x00003c00, 0x3c000000, 0x3c00bc00, 0x00003c00, 0x00003c00, 0x3c000000, \
  0xbc003c00, 0x3c00bc00, 0x00003c00, 0x3c000000, 0x0000bc00, 0x3c00bc00, 0x00003c00, 0x3c000000, \
  0x3c000000, 0x3c00bc00, 0x00003c00, 0x3c000000, 0x3c003c00, 0x3c00bc00, 0x00003c00, 0x3c000000, \
  0xbc003c00, 0x3c000000, 0x00003c00, 0x3c000000, 0x0000bc00, 0x3c000000, 0x00003c00, 0x3c000000, \
  0x00000000, 0x3c000000, 0x00003c00, 0x3c000000, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x3c000000, \
  0x00003c00, 0x3c003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0x3c003c00, 0x00003c00, 0x3c000000, \
  0x00000000, 0xbc00bc00, 0x3c00bc00, 0x3c000000, 0x00003c00, 0xbc000000, 0x3c00bc00, 0x3c000000, \
  0x3c000000, 0xbc000000, 0x3c00bc00, 0x3c000000, 0x00000000, 0xbc003c00, 0x3c00bc00, 0x3c000000, \
  0xbc000000, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0x00003c00, 0x0000bc00, 0x3c00bc00, 0x3c000000, \
  0xbc00bc00, 0x00000000, 0x3c00bc00, 0x3c000000, 0xbc003c00, 0x00000000, 0x3c00bc00, 0x3c000000, \
  0x00000000, 0x00000000, 0x3c00bc00, 0x3c000000, 0x00003c00, 0x00000000, 0x3c00bc00, 0x3c000000, \
  0x3c003c00, 0x00000000, 0x3c00bc00, 0x3c000000, 0x0000bc00, 0x00003c00, 0x3c00bc00, 0x3c000000, \
  0x00000000, 0x00003c00, 0x3c00bc00, 0x3c000000, 0x00000000, 0x3c00bc00, 0x3c00bc00, 0x3c000000, \
  0xbc000000, 0x3c000000, 0x3c00bc00, 0x3c000000, 0x00003c00, 0x3c000000, 0x3c00bc00, 0x3c000000, \
  0x3c000000, 0x3c000000, 0x3c00bc00, 0x3c000000, 0x00000000, 0x3c003c00, 0x3c00bc00, 0x3c000000, \
  0x0000bc00, 0xbc00bc00, 0x3c000000, 0x3c000000, 0x00003c00, 0xbc00bc00, 0x3c000000, 0x3c000000, \
  0x3c000000, 0xbc00bc00, 0x3c000000, 0x3c000000, 0xbc00bc00, 0xbc000000, 0x3c000000, 0x3c000000, \
  0xbc003c00, 0xbc000000, 0x3c000000, 0x3c000000, 0x00000000, 0xbc000000, 0x3c000000, 0x3c000000, \
  0x3c00bc00, 0xbc000000, 0x3c000000, 0x3c000000, 0x3c003c00, 0xbc000000, 0x3c000000, 0x3c000000, \
  0xbc00bc00, 0xbc003c00, 0x3c000000, 0x3c000000, 0xbc000000, 0xbc003c00, 0x3c000000, 0x3c000000, \
  0x0000bc00, 0xbc003c00, 0x3c000000, 0x3c000000, 0x00003c00, 0xbc003c00, 0x3c000000, 0x3c000000, \
  0xbc00bc00, 0x0000bc00, 0x3c000000, 0x3c000000, 0xbc003c00, 0x0000bc00, 0x3c000000, 0x3c000000, \
  0x00000000, 0x0000bc00, 0x3c000000, 0x3c000000, 0x3c00bc00, 0x0000bc00, 0x3c000000, 0x3c000000, \
  0x3c003c00, 0x0000bc00, 0x3c000000, 0x3c000000, 0xbc000000, 0x00000000, 0x3c000000, 0x3c000000, \
  0x0000bc00, 0x00000000, 0x3c000000, 0x3c000000, 0x00000000, 0x00000000, 0x3c000000, 0x3c000000, \
  0x00003c00, 0x00000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x00000000, 0x3c000000, 0x3c000000, \
  0xbc003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0x00000000, 0x00003c00, 0x3c000000, 0x3c000000, \
  0x00003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0x3c003c00, 0x00003c00, 0x3c000000, 0x3c000000, \
  0xbc000000, 0x3c00bc00, 0x3c000000, 0x3c000000, 0x0000bc00, 0x3c00bc00, 0x3c000000, 0x3c000000, \
  0xbc00bc00, 0x3c000000, 0x3c000000, 0x3c000000, 0xbc003c00, 0x3c000000, 0x3c000000, 0x3c000000, \
  0x00000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c003c00, 0x3c000000, 0x3c000000, 0x3c000000, \
  0xbc000000, 0x3c003c00, 0x3c000000, 0x3c000000, 0x00003c00, 0x3c003c00, 0x3c000000, 0x3c000000, \
  0x00000000, 0xbc00bc00, 0x3c003c00, 0x3c000000, 0x00000000, 0xbc000000, 0x3c003c00, 0x3c000000, \
  0x00000000, 0xbc003c00, 0x3c003c00, 0x3c000000, 0x0000bc00, 0x0000bc00, 0x3c003c00, 0x3c000000, \
  0x00003c00, 0x0000bc00, 0x3c003c00, 0x3c000000, 0x3c000000, 0x0000bc00, 0x3c003c00, 0x3c000000, \
  0xbc00bc00, 0x00000000, 0x3c003c00, 0x3c000000, 0x00000000, 0x00000000, 0x3c003c00, 0x3c000000, \
  0x3c00bc00, 0x00000000, 0x3c003c00, 0x3c000000, 0xbc000000, 0x00003c00, 0x3c003c00, 0x3c000000, \
  0x00000000, 0x3c00bc00, 0x3c003c00, 0x3c000000, 0xbc000000, 0x3c000000, 0x3c003c00, 0x3c000000, \
  0x0000bc00, 0x3c000000, 0x3c003c00, 0x3c000000, 0x00000000, 0x3c000000, 0x3c003c00, 0x3c000000, \
  0x00003c00, 0x3c000000, 0x3c003c00, 0x3c000000, 0xbc00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, \
  0xbc003c00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, \
  0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0x00000000, 0xbc000000, 0xbc00bc00, 0x3c003c00, \
  0xbc00bc00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, \
  0x3c00bc00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, \
  0x00000000, 0x0000bc00, 0xbc00bc00, 0x3c003c00, 0xbc000000, 0x00000000, 0xbc00bc00, 0x3c003c00, \
  0x0000bc00, 0x00000000, 0xbc00bc00, 0x3c003c00, 0x00003c00, 0x00000000, 0xbc00bc00, 0x3c003c00, \
  0x3c000000, 0x00000000, 0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, \
  0xbc003c00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, \
  0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0x00000000, 0x3c000000, 0xbc00bc00, 0x3c003c00, \
  0xbc00bc00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, \
  0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, \
  0x00000000, 0xbc00bc00, 0xbc000000, 0x3c003c00, 0x3c000000, 0xbc00bc00, 0xbc000000, 0x3c003c00, \
  0xbc000000, 0xbc000000, 0xbc000000, 0x3c003c00, 0x0000bc00, 0xbc000000, 0xbc000000, 0x3c003c00, \
  0x00003c00, 0xbc000000, 0xbc000000, 0x3c003c00, 0x3c000000, 0xbc000000, 0xbc000000, 0x3c003c00, \
  0x3c003c00, 0xbc000000, 0xbc000000, 0x3c003c00, 0x00003c00, 0x0000bc00, 0xbc000000, 0x3c003c00, \
  0x3c000000, 0x0000bc00, 0xbc000000, 0x3c003c00, 0xbc000000, 0x00000000, 0xbc000000, 0x3c003c00, \
  0x00000000, 0x00000000, 0xbc000000, 0x3c003c00, 0x3c00bc00, 0x00000000, 0xbc000000, 0x3c003c00, \
  0x3c003c00, 0x00000000, 0xbc000000, 0x3c003c00, 0xbc000000, 0x00003c00, 0xbc000000, 0x3c003c00, \
  0x0000bc00, 0x00003c00, 0xbc000000, 0x3c003c00, 0x00000000, 0x3c00bc00, 0xbc000000, 0x3c003c00, \
  0xbc00bc00, 0x3c000000, 0xbc000000, 0x3c003c00, 0xbc003c00, 0x3c000000, 0xbc000000, 0x3c003c00, \
  0x00003c00, 0x3c000000, 0xbc000000, 0x3c003c00, 0x3c000000, 0x3c000000, 0xbc000000, 0x3c003c00, \
  0xbc003c00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, \
  0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0xbc000000, 0xbc003c00, 0x3c003c00, \
  0x3c000000, 0xbc000000, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0xbc003c00, 0x3c003c00, \
  0x3c00bc00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0x3c003c00, \
  0x00000000, 0x0000bc00, 0xbc003c00, 0x3c003c00, 0xbc000000, 0x00000000, 0xbc003c00, 0x3c003c00, \
  0x00003c00, 0x00000000, 0xbc003c00, 0x3c003c00, 0x3c000000, 0x00000000, 0xbc003c00, 0x3c003c00, \
  0x00000000, 0x00003c00, 0xbc003c00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, \
  0xbc003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, \
  0x3c003c00, 0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x00000000, 0x3c000000, 0xbc003c00, 0x3c003c00, \
  0xbc00bc00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0xbc003c00, 0x3c003c00, \
  0x3c00bc00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0x3c003c00, \
  0x3c000000, 0xbc000000, 0x0000bc00, 0x3c003c00, 0x00000000, 0xbc003c00, 0x0000bc00, 0x3c003c00, \
  0xbc000000, 0x0000bc00, 0x0000bc00, 0x3c003c00, 0x0000bc00, 0x0000bc00, 0x0000bc00, 0x3c003c00, \
  0xbc00bc00, 0x00000000, 0x0000bc00, 0x3c003c00, 0x0000bc00, 0x00000000, 0x0000bc00, 0x3c003c00, \
  0x00000000, 0x00000000, 0x0000bc00, 0x3c003c00, 0x3c00bc00, 0x00000000, 0x0000bc00, 0x3c003c00, \
  0x3c003c00, 0x00000000, 0x0000bc00, 0x3c003c00, 0xbc000000, 0x00003c00, 0x0000bc00, 0x3c003c00, \
  0x00000000, 0x00003c00, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x00003c00, 0x0000bc00, 0x3c003c00, \
  0x3c00bc00, 0x00003c00, 0x0000bc00, 0x3c003c00, 0x3c000000, 0x00003c00, 0x0000bc00, 0x3c003c00, \
  0x00000000, 0x3c00bc00, 0x0000bc00, 0x3c003c00, 0x00003c00, 0xbc00bc00, 0x00000000, 0x3c003c00, \
  0x3c000000, 0xbc00bc00, 0x00000000, 0x3c003c00, 0xbc00bc00, 0xbc000000, 0x00000000, 0x3c003c00, \
  0xbc003c00, 0xbc000000, 0x00000000, 0x3c003c00, 0x00000000, 0xbc000000, 0x00000000, 0x3c003c00, \
  0x3c00bc00, 0xbc000000, 0x00000000, 0x3c003c00, 0x00003c00, 0xbc003c00, 0x00000000, 0x3c003c00, \
  0x3c000000, 0xbc003c00, 0x00000000, 0x3c003c00, 0xbc003c00, 0x0000bc00, 0x00000000, 0x3c003c00, \
  0x00000000, 0x0000bc00, 0x00000000, 0x3c003c00, 0xbc000000, 0x00000000, 0x00000000, 0x3c003c00, \
  0x0000bc00, 0x00000000, 0x00000000, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x3c003c00, \
  0x00003c00, 0x00000000, 0x00000000, 0x3c003c00, 0x3c000000, 0x00000000, 0x00000000, 0x3c003c00, \
  0x00000000, 0x00003c00, 0x00000000, 0x3c003c00, 0x3c003c00, 0x00003c00, 0x00000000, 0x3c003c00, \
  0xbc000000, 0x3c00bc00, 0x00000000, 0x3c003c00, 0x0000bc00, 0x3c00bc00, 0x00000000, 0x3c003c00, \
  0x00000000, 0x3c00bc00, 0x00000000, 0x3c003c00, 0x00003c00, 0x3c00bc00, 0x00000000, 0x3c003c00, \
  0x3c000000, 0x3c00bc00, 0x00000000, 0x3c003c00, 0xbc003c00, 0x3c000000, 0x00000000, 0x3c003c00, \
  0x00000000, 0x3c000000, 0x00000000, 0x3c003c00, 0x3c00bc00, 0x3c000000, 0x00000000, 0x3c003c00, \
  0x00000000, 0xbc00bc00, 0x00003c00, 0x3c003c00, 0xbc000000, 0xbc000000, 0x00003c00, 0x3c003c00, \
  0x00003c00, 0xbc000000, 0x00003c00, 0x3c003c00, 0x3c003c00, 0xbc000000, 0x00003c00, 0x3c003c00, \
  0xbc000000, 0xbc003c00, 0x00003c00, 0x3c003c00, 0x00000000, 0xbc003c00, 0x00003c00, 0x3c003c00, \
  0x0000bc00, 0x0000bc00, 0x00003c00, 0x3c003c00, 0x00003c00, 0x0000bc00, 0x00003c00, 0x3c003c00, \
  0x3c003c00, 0x0000bc00, 0x00003c00, 0x3c003c00, 0xbc003c00, 0x00000000, 0x00003c00, 0x3c003c00, \
  0x00000000, 0x00000000, 0x00003c00, 0x3c003c00, 0x00003c00, 0x00000000, 0x00003c00, 0x3c003c00, \
  0x3c00bc00, 0x00000000, 0x00003c00, 0x3c003c00, 0xbc00bc00, 0x00003c00, 0x00003c00, 0x3c003c00, \
  0xbc003c00, 0x00003c00, 0x00003c00, 0x3c003c00, 0x00003c00, 0x3c00bc00, 0x00003c00, 0x3c003c00, \
  0xbc00bc00, 0x3c000000, 0x00003c00, 0x3c003c00, 0x00000000, 0x3c000000, 0x00003c00, 0x3c003c00, \
  0x00003c00, 0x3c000000, 0x00003c00, 0x3c003c00, 0x3c000000, 0x3c000000, 0x00003c00, 0x3c003c00, \
  0xbc000000, 0x3c003c00, 0x00003c00, 0x3c003c00, 0x0000bc00, 0x3c003c00, 0x00003c00, 0x3c003c00, \
  0x00003c00, 0x3c003c00, 0x00003c00, 0x3c003c00, 0xbc00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, \
  0xbc003c00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, \
  0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, \
  0xbc003c00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, \
  0x3c003c00, 0xbc003c00, 0x3c00bc00, 0x3c003c00, 0xbc000000, 0x00000000, 0x3c00bc00, 0x3c003c00, \
  0x0000bc00, 0x00000000, 0x3c00bc00, 0x3c003c00, 0x00003c00, 0x00000000, 0x3c00bc00, 0x3c003c00, \
  0x3c000000, 0x00000000, 0x3c00bc00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, \
  0xbc003c00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, \
  0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0x00000000, 0x3c000000, 0x3c00bc00, 0x3c003c00, \
  0xbc00bc00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, \
  0x3c00bc00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0x3c003c00, \
  0x00000000, 0xbc00bc00, 0x3c000000, 0x3c003c00, 0x0000bc00, 0xbc000000, 0x3c000000, 0x3c003c00, \
  0x3c000000, 0xbc000000, 0x3c000000, 0x3c003c00, 0xbc000000, 0xbc003c00, 0x3c000000, 0x3c003c00, \
  0x00000000, 0xbc003c00, 0x3c000000, 0x3c003c00, 0xbc000000, 0x0000bc00, 0x3c000000, 0x3c003c00, \
  0xbc00bc00, 0x00000000, 0x3c000000, 0x3c003c00, 0x00000000, 0x00000000, 0x3c000000, 0x3c003c00, \
  0x3c003c00, 0x00000000, 0x3c000000, 0x3c003c00, 0xbc000000, 0x00003c00, 0x3c000000, 0x3c003c00, \
  0x00003c00, 0x00003c00, 0x3c000000, 0x3c003c00, 0x3c000000, 0x00003c00, 0x3c000000, 0x3c003c00, \
  0xbc00bc00, 0x3c000000, 0x3c000000, 0x3c003c00, 0x00003c00, 0x3c000000, 0x3c000000, 0x3c003c00, \
  0xbc00bc00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, \
  0x3c00bc00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc00bc00, 0x3c003c00, 0x3c003c00, \
  0xbc00bc00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0xbc003c00, 0x3c003c00, 0x3c003c00, \
  0x3c00bc00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0x3c003c00, \
  0xbc000000, 0x00000000, 0x3c003c00, 0x3c003c00, 0x0000bc00, 0x00000000, 0x3c003c00, 0x3c003c00, \
  0x00003c00, 0x00000000, 0x3c003c00, 0x3c003c00, 0xbc00bc00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, \
  0xbc003c00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x3c00bc00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, \
  0x3c003c00, 0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x3c000000, 0x3c003c00, 0x3c003c00, \
  0xbc00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0xbc003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, \
  0x3c00bc00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00

// derived: IQ2_XXS grid as half values, 4 x uint32 per entry
#define Q_IQ2XXS_GRID_HALF_VALUES \
  0x48004800, 0x48004800, 0x48004800, 0x48004800, 0x48005160, 0x48004800, 0x48004800, 0x48004800, \
  0x4e404e40, 0x48004800, 0x48004800, 0x48004800, 0x51604800, 0x48004800, 0x48004800, 0x48004800, \
  0x51605160, 0x48004800, 0x48004800, 0x48004800, 0x48004e40, 0x48004e40, 0x48004800, 0x48004800, \
  0x4e404800, 0x48004e40, 0x48004800, 0x48004800, 0x48004800, 0x48005160, 0x48004800, 0x48004800, \
  0x48005160, 0x48005160, 0x48004800, 0x48004800, 0x51604800, 0x48005160, 0x48004800, 0x48004800, \
  0x51605160, 0x48005160, 0x48004800, 0x48004800, 0x48004e40, 0x4e404800, 0x48004800, 0x48004800, \
  0x4e404800, 0x4e404800, 0x48004800, 0x48004800, 0x48004800, 0x4e404e40, 0x48004800, 0x48004800, \
  0x51604800, 0x4e404e40, 0x48004800, 0x48004800, 0x48004e40, 0x4e405160, 0x48004800, 0x48004800, \
  0x4e404800, 0x4e405160, 0x48004800, 0x48004800, 0x48004800, 0x51604800, 0x48004800, 0x48004800, \
  0x48005160, 0x51604800, 0x48004800, 0x48004800, 0x51605160, 0x51604800, 0x48004800, 0x48004800, \
  0x48005160, 0x51605160, 0x48004800, 0x48004800, 0x48004e40, 0x48004800, 0x48004e40, 0x48004800, \
  0x4e404800, 0x48004800, 0x48004e40, 0x48004800, 0x48004800, 0x48004e40, 0x48004e40, 0x48004800, \
  0x4e404e40, 0x48004e40, 0x48004e40, 0x48004800, 0x48004800, 0x4e404800, 0x48004e40, 0x48004800, \
  0x4e404800, 0x51604800, 0x48004e40, 0x48004800, 0x51604800, 0x51604e40, 0x48004e40, 0x48004800, \
  0x48004800, 0x48004800, 0x48005160, 0x48004800, 0x48005160, 0x48004800, 0x48005160, 0x48004800, \
  0x48005160, 0x48005160, 0x48005160, 0x48004800, 0x48005160, 0x51604800, 0x48005160, 0x48004800, \
  0x48004e40, 0x48004800, 0x4e404800, 0x48004800, 0x4e404800, 0x48004800, 0x4e404800, 0x48004800, \
  0x48004800, 0x48004e40, 0x4e404800, 0x48004800, 0x48004e40, 0x48005160, 0x4e404800, 0x48004800, \
  0x4e404800, 0x48005160, 0x4e404800, 0x48004800, 0x48004800, 0x4e404800, 0x4e404800, 0x48004800, \
  0x48005160, 0x4e404800, 0x4e404800, 0x48004800, 0x51604800, 0x4e404800, 0x4e404800, 0x48004800, \
  0x48004800, 0x4e405160, 0x4e404800, 0x48004800, 0x48004e40, 0x51604800, 0x4e404800, 0x48004800, \
  0x4e404800, 0x51604800, 0x4e404800, 0x48004800, 0x48004800, 0x51604e40, 0x4e404800, 0x48004800, \
  0x4e404800, 0x51605160, 0x4e404800, 0x48004800, 0x48004800, 0x48004800, 0x4e404e40, 0x48004800, \
  0x48005160, 0x48004800, 0x4e404e40, 0x48004800, 0x51604800, 0x48004800, 0x4e404e40, 0x48004800, \
  0x48004800, 0x48005160, 0x4e404e40, 0x48004800, 0x4e405160, 0x4e404800, 0x4e404e40, 0x48004800, \
  0x51604e40, 0x4e405160, 0x4e404e40, 0x48004800, 0x48004800, 0x51604800, 0x4e404e40, 0x48004800, \
  0x48004e40, 0x51604e40, 0x4e404e40, 0x48004800, 0x51604e40, 0x48004800, 0x4e405160, 0x48004800, \
  0x48004800, 0x48004e40, 0x4e405160, 0x48004800, 0x48004800, 0x4e404800, 0x4e405160, 0x48004800, \
  0x4e404800, 0x51604800, 0x4e405160, 0x48004800, 0x4e404800, 0x51605160, 0x4e405160, 0x48004800, \
  0x48004800, 0x48004800, 0x51604800, 0x48004800, 0x4e404e40, 0x48004800, 0x51604800, 0x48004800, \
  0x51604800, 0x48004800, 0x51604800, 0x48004800, 0x4e404800, 0x48004e40, 0x51604800, 0x48004800, \
  0x51604800, 0x48005160, 0x51604800, 0x48004800, 0x48004e40, 0x4e404800, 0x51604800, 0x48004800, \
  0x4e404800, 0x4e404800, 0x51604800, 0x48004800, 0x48004800, 0x4e404e40, 0x51604800, 0x48004800, \
  0x48005160, 0x4e404e40, 0x51604800, 0x48004800, 0x51604800, 0x51604800, 0x51604800, 0x48004800, \
  0x4e404800, 0x48004800, 0x51604e40, 0x48004800, 0x48004800, 0x4e404800, 0x51604e40, 0x48004800, \
  0x48005160, 0x48004800, 0x51605160, 0x48004800, 0x4e404800, 0x48004e40, 0x51605160, 0x48004800, \
  0x48004e40, 0x48004800, 0x48004800, 0x48004e40, 0x4e404800, 0x48004800, 0x48004800, 0x48004e40, \
  0x48004800, 0x48004e40, 0x48004800, 0x48004e40, 0x48004e40, 0x48005160, 0x48004800, 0x48004e40, \
  0x48004800, 0x4e404800, 0x48004800, 0x48004e40, 0x48004800, 0x4e405160, 0x48004800, 0x48004e40, \
  0x4e404800, 0x51604800, 0x48004800, 0x48004e40, 0x48004800, 0x51604e40, 0x48004800, 0x48004e40, \
  0x4e404e40, 0x51604e40, 0x48004800, 0x48004e40, 0x48004800, 0x48004800, 0x48004e40, 0x48004e40, \
  0x51604800, 0x48004800, 0x48004e40, 0x48004e40, 0x48004800, 0x48005160, 0x48004e40, 0x48004e40, \
  0x48004800, 0x4e404e40, 0x48004e40, 0x48004e40, 0x51605160, 0x4e404e40, 0x48004e40, 0x48004e40, \
  0x48004800, 0x51604800, 0x48004e40, 0x48004e40, 0x4e404800, 0x48005160, 0x48005160, 0x48004e40, \
  0x4e404e40, 0x4e404800, 0x48005160, 0x48004e40, 0x48004800, 0x48004800, 0x4e404800, 0x48004e40, \
  0x51604800, 0x48004800, 0x4e404800, 0x48004e40, 0x48004800, 0x48005160, 0x4e404800, 0x48004e40, \
  0x4e404e40, 0x48005160, 0x4e404800, 0x48004e40, 0x51604e40, 0x4e404800, 0x4e404800, 0x48004e40, \
  0x48004800, 0x51604800, 0x4e404800, 0x48004e40, 0x51604800, 0x48004e40, 0x4e404e40, 0x48004e40, \
  0x48005160, 0x4e405160, 0x4e404e40, 0x48004e40, 0x48004800, 0x48004800, 0x4e405160, 0x48004e40, \
  0x4e405160, 0x48004e40, 0x4e405160, 0x48004e40, 0x48004e40, 0x48004800, 0x51604800, 0x48004e40, \
  0x4e404800, 0x48004800, 0x51604800, 0x48004e40, 0x48004800, 0x48004e40, 0x51604800, 0x48004e40, \
  0x48004800, 0x4e404800, 0x51604800, 0x48004e40, 0x48004e40, 0x51604800, 0x51604800, 0x48004e40, \
  0x48004800, 0x48004800, 0x51604e40, 0x48004e40, 0x4e404e40, 0x48004800, 0x51604e40, 0x48004e40, \
  0x48004800, 0x51605160, 0x51604e40, 0x48004e40, 0x48004e40, 0x4e404e40, 0x51605160, 0x48004e40, \
  0x48004800, 0x48004800, 0x48004800, 0x48005160, 0x48005160, 0x48004800, 0x48004800, 0x48005160, \
  0x51605160, 0x48004800, 0x48004800, 0x48005160, 0x4e404800, 0x4e404800, 0x48004800, 0x48005160, \
  0x48004e40, 0x4e405160, 0x48004800, 0x48005160, 0x48004800, 0x51604800, 0x48004800, 0x48005160, \
  0x48005160, 0x51604800, 0x48004800, 0x48005160, 0x51604e40, 0x48005160, 0x48004e40, 0x48005160, \
  0x51604800, 0x4e404800, 0x48004e40, 0x48005160, 0x48004800, 0x48004800, 0x48005160, 0x48005160, \
  0x48005160, 0x48004800, 0x48005160, 0x48005160, 0x48004e40, 0x48004800, 0x4e404800, 0x48005160, \
  0x4e404800, 0x48004800, 0x4e404800, 0x48005160, 0x48004800, 0x48004e40, 0x4e404800, 0x48005160, \
  0x48004800, 0x4e404800, 0x4e404800, 0x48005160, 0x4e405160, 0x4e404e40, 0x4e404800, 0x48005160, \
  0x48004800, 0x48004800, 0x4e404e40, 0x48005160, 0x48004e40, 0x4e404800, 0x4e404e40, 0x48005160, \
  0x4e404800, 0x4e405160, 0x4e404e40, 0x48005160, 0x48004800, 0x51604e40, 0x4e405160, 0x48005160, \
  0x51604800, 0x48004800, 0x51604800, 0x48005160, 0x48004800, 0x48005160, 0x51604800, 0x48005160, \
  0x4e404800, 0x51604e40, 0x51604800, 0x48005160, 0x4e404800, 0x4e404800, 0x51605160, 0x48005160, \
  0x48004e40, 0x48004800, 0x48004800, 0x4e404800, 0x4e404800, 0x48004800, 0x48004800, 0x4e404800, \
  0x48004800, 0x48004e40, 0x48004800, 0x4e404800, 0x51604800, 0x48004e40, 0x48004800, 0x4e404800, \
  0x48004e40, 0x48005160, 0x48004800, 0x4e404800, 0x4e404800, 0x48005160, 0x48004800, 0x4e404800, \
  0x48004800, 0x4e404800, 0x48004800, 0x4e404800, 0x51604800, 0x4e404800, 0x48004800, 0x4e404800, \
  0x4e405160, 0x4e404e40, 0x48004800, 0x4e404800, 0x48004800, 0x4e405160, 0x48004800, 0x4e404800, \
  0x48004e40, 0x51604800, 0x48004800, 0x4e404800, 0x4e404800, 0x51604800, 0x48004800, 0x4e404800, \
  0x48004800, 0x51604e40, 0x48004800, 0x4e404800, 0x48004800, 0x48004800, 0x48004e40, 0x4e404800, \
  0x48004800, 0x48005160, 0x48004e40, 0x4e404800, 0x48004e40, 0x4e405160, 0x48004e40, 0x4e404800, \
  0x48004800, 0x51604800, 0x48004e40, 0x4e404800, 0x4e404e40, 0x51604800, 0x48004e40, 0x4e404800, \
  0x48004e40, 0x48004800, 0x48005160, 0x4e404800, 0x48004800, 0x48004e40, 0x48005160, 0x4e404800, \
  0x51604800, 0x4e404800, 0x48005160, 0x4e404800, 0x4e405160, 0x4e404e40, 0x48005160, 0x4e404800, \
  0x51604800, 0x4e405160, 0x48005160, 0x4e404800, 0x48004800, 0x48004800, 0x4e404800, 0x4e404800, \
  0x51604800, 0x48004800, 0x4e404800, 0x4e404800, 0x48004800, 0x48005160, 0x4e404800, 0x4e404800, \
  0x48004800, 0x51604800, 0x4e404800, 0x4e404800, 0x51604e40, 0x51604e40, 0x4e404800, 0x4e404800, \
  0x48005160, 0x48004e40, 0x4e404e40, 0x4e404800, 0x4e404800, 0x48005160, 0x4e404e40, 0x4e404800, \
  0x48004800, 0x48004800, 0x4e405160, 0x4e404800, 0x48004e40, 0x48004800, 0x51604800, 0x4e404800, \
  0x4e404800, 0x48004800, 0x51604800, 0x4e404800, 0x48004800, 0x48004e40, 0x51604800, 0x4e404800, \
  0x48004800, 0x4e404800, 0x51604800, 0x4e404800, 0x4e404e40, 0x4e404800, 0x51604800, 0x4e404800, \
  0x48004800, 0x48004800, 0x51604e40, 0x4e404800, 0x51604800, 0x4e404e40, 0x51604e40, 0x4e404800, \
  0x48004e40, 0x4e405160, 0x51604e40, 0x4e404800, 0x48005160, 0x51604800, 0x51604e40, 0x4e404800, \
  0x4e404e40, 0x4e404800, 0x51605160, 0x4e404800, 0x48004800, 0x51604e40, 0x51605160, 0x4e404800, \
  0x48004800, 0x48004800, 0x48004800, 0x4e404e40, 0x51604800, 0x48004800, 0x48004800, 0x4e404e40, \
  0x48004e40, 0x48004e40, 0x48004800, 0x4e404e40, 0x51604e40, 0x48004e40, 0x48004800, 0x4e404e40, \
  0x48004800, 0x48005160, 0x48004800, 0x4e404e40, 0x48004800, 0x51604800, 0x48004800, 0x4e404e40, \
  0x51604800, 0x51604800, 0x48004800, 0x4e404e40, 0x4e404800, 0x48004800, 0x48004e40, 0x4e404e40, \
  0x48005160, 0x4e404800, 0x48004e40, 0x4e404e40, 0x4e404800, 0x51605160, 0x48004e40, 0x4e404e40, \
  0x48004e40, 0x51604e40, 0x48005160, 0x4e404e40, 0x48004800, 0x51604e40, 0x4e404800, 0x4e404e40, \
  0x48005160, 0x51604e40, 0x4e404800, 0x4e404e40, 0x51605160, 0x48004800, 0x4e404e40, 0x4e404e40, \
  0x48004e40, 0x48004800, 0x4e405160, 0x4e404e40, 0x4e404800, 0x4e404e40, 0x4e405160, 0x4e404e40, \
  0x48004800, 0x48004800, 0x51604800, 0x4e404e40, 0x48004e40, 0x48004e40, 0x51604800, 0x4e404e40, \
  0x51604e40, 0x48004e40, 0x51604800, 0x4e404e40, 0x4e404800, 0x4e405160, 0x51604800, 0x4e404e40, \
  0x48004800, 0x4e404800, 0x51604e40, 0x4e404e40, 0x51604800, 0x48004800, 0x51605160, 0x4e404e40, \
  0x4e404800, 0x48004800, 0x48004800, 0x4e405160, 0x48004800, 0x48004e40, 0x48004800, 0x4e405160, \
  0x48004800, 0x4e404800, 0x48004800, 0x4e405160, 0x51604800, 0x4e405160, 0x48004800, 0x4e405160, \
  0x48004800, 0x48004800, 0x48004e40, 0x4e405160, 0x4e404e40, 0x4e404e40, 0x48004e40, 0x4e405160, \
  0x51604800, 0x48004e40, 0x48005160, 0x4e405160, 0x48004800, 0x4e405160, 0x48005160, 0x4e405160, \
  0x48004800, 0x48004800, 0x4e404800, 0x4e405160, 0x4e404e40, 0x48004800, 0x4e404800, 0x4e405160, \
  0x48004800, 0x48004e40, 0x4e404e40, 0x4e405160, 0x48005160, 0x48004e40, 0x4e404e40, 0x4e405160, \
  0x4e404800, 0x51604800, 0x4e404e40, 0x4e405160, 0x48005160, 0x4e404800, 0x51604800, 0x4e405160, \
  0x48004800, 0x48004800, 0x48004800, 0x51604800, 0x48005160, 0x48004800, 0x48004800, 0x51604800, \
  0x51605160, 0x48004800, 0x48004800, 0x51604800, 0x48004e40, 0x4e404800, 0x48004800, 0x51604800, \
  0x48005160, 0x51604800, 0x48004800, 0x51604800, 0x4e404800, 0x48004800, 0x48004e40, 0x51604800, \
  0x51604800, 0x48004e40, 0x48004e40, 0x51604800, 0x48004800, 0x4e404800, 0x48004e40, 0x51604800, \
  0x48004e40, 0x48004e40, 0x48005160, 0x51604800, 0x48004e40, 0x48004800, 0x4e404800, 0x51604800, \
  0x4e404800, 0x48004800, 0x4e404800, 0x51604800, 0x48004800, 0x48004e40, 0x4e404800, 0x51604800, \
  0x4e404e40, 0x48004e40, 0x4e404800, 0x51604800, 0x48004800, 0x4e404800, 0x4e404800, 0x51604800, \
  0x48004800, 0x4e405160, 0x4e404800, 0x51604800, 0x48004800, 0x48004800, 0x4e404e40, 0x51604800, \
  0x4e405160, 0x4e404800, 0x4e404e40, 0x51604800, 0x4e404800, 0x51604e40, 0x4e404e40, 0x51604800, \
  0x51604e40, 0x48004800, 0x4e405160, 0x51604800, 0x48004800, 0x4e404800, 0x4e405160, 0x51604800, \
  0x48004800, 0x4e405160, 0x4e405160, 0x51604800, 0x48005160, 0x48004800, 0x51604800, 0x51604800, \
  0x4e404800, 0x48004800, 0x51604e40, 0x51604800, 0x48004e40, 0x48004e40, 0x51605160, 0x51604800, \
  0x4e404800, 0x48004800, 0x48004800, 0x51604e40, 0x48004800, 0x48004e40, 0x48004800, 0x51604e40, \
  0x4e404800, 0x48005160, 0x48004800, 0x51604e40, 0x48004800, 0x4e404800, 0x48004800, 0x51604e40, \
  0x48004e40, 0x51605160, 0x48004800, 0x51604e40, 0x4e405160, 0x48004e40, 0x48004e40, 0x51604e40, \
  0x48004800, 0x51604800, 0x48004e40, 0x51604e40, 0x4e404e40, 0x4e404800, 0x48005160, 0x51604e40, \
  0x48004800, 0x48004800, 0x4e404800, 0x51604e40, 0x48005160, 0x48005160, 0x4e404800, 0x51604e40, \
  0x4e404800, 0x4e404800, 0x4e404800, 0x51604e40, 0x48004e40, 0x4e404e40, 0x4e404e40, 0x51604e40, \
  0x48004e40, 0x51604800, 0x51604800, 0x51604e40, 0x48004800, 0x48005160, 0x51604e40, 0x51604e40, \
  0x48005160, 0x48004800, 0x48004800, 0x51605160, 0x48004800, 0x4e404e40, 0x48004800, 0x51605160, \
  0x4e404e40, 0x51604800, 0x48004800, 0x51605160, 0x51604e40, 0x48004800, 0x48004e40, 0x51605160, \
  0x48004800, 0x48004800, 0x48005160, 0x51605160, 0x51604800, 0x48004e40, 0x4e404800, 0x51605160, \
  0x48004800, 0x4e404e40, 0x51604800, 0x51605160, 0x4e404800, 0x48004800, 0x51604e40, 0x51605160

// derived: 7-bit sign index -> half sign-bit masks (4 x uint32 = 8 halfs)
#define Q_IQ2_SIGNH_VALUES \
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00008000, 0x00000000, 0x00000000, 0x80000000, \
  0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x80008000, 0x00000000, 0x00000000, 0x00000000, \
  0x00000000, 0x00008000, 0x00000000, 0x80000000, 0x00008000, 0x00008000, 0x00000000, 0x00000000, \
  0x80000000, 0x00008000, 0x00000000, 0x00000000, 0x80008000, 0x00008000, 0x00000000, 0x80000000, \
  0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00008000, 0x80000000, 0x00000000, 0x00000000, \
  0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x80008000, 0x80000000, 0x00000000, 0x80000000, \
  0x00000000, 0x80008000, 0x00000000, 0x00000000, 0x00008000, 0x80008000, 0x00000000, 0x80000000, \
  0x80000000, 0x80008000, 0x00000000, 0x80000000, 0x80008000, 0x80008000, 0x00000000, 0x00000000, \
  0x00000000, 0x00000000, 0x00008000, 0x80000000, 0x00008000, 0x00000000, 0x00008000, 0x00000000, \
  0x80000000, 0x00000000, 0x00008000, 0x00000000, 0x80008000, 0x00000000, 0x00008000, 0x80000000, \
  0x00000000, 0x00008000, 0x00008000, 0x00000000, 0x00008000, 0x00008000, 0x00008000, 0x80000000, \
  0x80000000, 0x00008000, 0x00008000, 0x80000000, 0x80008000, 0x00008000, 0x00008000, 0x00000000, \
  0x00000000, 0x80000000, 0x00008000, 0x00000000, 0x00008000, 0x80000000, 0x00008000, 0x80000000, \
  0x80000000, 0x80000000, 0x00008000, 0x80000000, 0x80008000, 0x80000000, 0x00008000, 0x00000000, \
  0x00000000, 0x80008000, 0x00008000, 0x80000000, 0x00008000, 0x80008000, 0x00008000, 0x00000000, \
  0x80000000, 0x80008000, 0x00008000, 0x00000000, 0x80008000, 0x80008000, 0x00008000, 0x80000000, \
  0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00008000, 0x00000000, 0x80000000, 0x00000000, \
  0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80008000, 0x00000000, 0x80000000, 0x80000000, \
  0x00000000, 0x00008000, 0x80000000, 0x00000000, 0x00008000, 0x00008000, 0x80000000, 0x80000000, \
  0x80000000, 0x00008000, 0x80000000, 0x80000000, 0x80008000, 0x00008000, 0x80000000, 0x00000000, \
  0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x00008000, 0x80000000, 0x80000000, 0x80000000, \
  0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80008000, 0x80000000, 0x80000000, 0x00000000, \
  0x00000000, 0x80008000, 0x80000000, 0x80000000, 0x00008000, 0x80008000, 0x80000000, 0x00000000, \
  0x80000000, 0x80008000, 0x80000000, 0x00000000, 0x80008000, 0x80008000, 0x80000000, 0x80000000, \
  0x00000000, 0x00000000, 0x80008000, 0x00000000, 0x00008000, 0x00000000, 0x80008000, 0x80000000, \
  0x80000000, 0x00000000, 0x80008000, 0x80000000, 0x80008000, 0x00000000, 0x80008000, 0x00000000, \
  0x00000000, 0x00008000, 0x80008000, 0x80000000, 0x00008000, 0x00008000, 0x80008000, 0x00000000, \
  0x80000000, 0x00008000, 0x80008000, 0x00000000, 0x80008000, 0x00008000, 0x80008000, 0x80000000, \
  0x00000000, 0x80000000, 0x80008000, 0x80000000, 0x00008000, 0x80000000, 0x80008000, 0x00000000, \
  0x80000000, 0x80000000, 0x80008000, 0x00000000, 0x80008000, 0x80000000, 0x80008000, 0x80000000, \
  0x00000000, 0x80008000, 0x80008000, 0x00000000, 0x00008000, 0x80008000, 0x80008000, 0x80000000, \
  0x80000000, 0x80008000, 0x80008000, 0x80000000, 0x80008000, 0x80008000, 0x80008000, 0x00000000, \
  0x00000000, 0x00000000, 0x00000000, 0x80008000, 0x00008000, 0x00000000, 0x00000000, 0x00008000, \
  0x80000000, 0x00000000, 0x00000000, 0x00008000, 0x80008000, 0x00000000, 0x00000000, 0x80008000, \
  0x00000000, 0x00008000, 0x00000000, 0x00008000, 0x00008000, 0x00008000, 0x00000000, 0x80008000, \
  0x80000000, 0x00008000, 0x00000000, 0x80008000, 0x80008000, 0x00008000, 0x00000000, 0x00008000, \
  0x00000000, 0x80000000, 0x00000000, 0x00008000, 0x00008000, 0x80000000, 0x00000000, 0x80008000, \
  0x80000000, 0x80000000, 0x00000000, 0x80008000, 0x80008000, 0x80000000, 0x00000000, 0x00008000, \
  0x00000000, 0x80008000, 0x00000000, 0x80008000, 0x00008000, 0x80008000, 0x00000000, 0x00008000, \
  0x80000000, 0x80008000, 0x00000000, 0x00008000, 0x80008000, 0x80008000, 0x00000000, 0x80008000, \
  0x00000000, 0x00000000, 0x00008000, 0x00008000, 0x00008000, 0x00000000, 0x00008000, 0x80008000, \
  0x80000000, 0x00000000, 0x00008000, 0x80008000, 0x80008000, 0x00000000, 0x00008000, 0x00008000, \
  0x00000000, 0x00008000, 0x00008000, 0x80008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, \
  0x80000000, 0x00008000, 0x00008000, 0x00008000, 0x80008000, 0x00008000, 0x00008000, 0x80008000, \
  0x00000000, 0x80000000, 0x00008000, 0x80008000, 0x00008000, 0x80000000, 0x00008000, 0x00008000, \
  0x80000000, 0x80000000, 0x00008000, 0x00008000, 0x80008000, 0x80000000, 0x00008000, 0x80008000, \
  0x00000000, 0x80008000, 0x00008000, 0x00008000, 0x00008000, 0x80008000, 0x00008000, 0x80008000, \
  0x80000000, 0x80008000, 0x00008000, 0x80008000, 0x80008000, 0x80008000, 0x00008000, 0x00008000, \
  0x00000000, 0x00000000, 0x80000000, 0x00008000, 0x00008000, 0x00000000, 0x80000000, 0x80008000, \
  0x80000000, 0x00000000, 0x80000000, 0x80008000, 0x80008000, 0x00000000, 0x80000000, 0x00008000, \
  0x00000000, 0x00008000, 0x80000000, 0x80008000, 0x00008000, 0x00008000, 0x80000000, 0x00008000, \
  0x80000000, 0x00008000, 0x80000000, 0x00008000, 0x80008000, 0x00008000, 0x80000000, 0x80008000, \
  0x00000000, 0x80000000, 0x80000000, 0x80008000, 0x00008000, 0x80000000, 0x80000000, 0x00008000, \
  0x80000000, 0x80000000, 0x80000000, 0x00008000, 0x80008000, 0x80000000, 0x80000000, 0x80008000, \
  0x00000000, 0x80008000, 0x80000000, 0x00008000, 0x00008000, 0x80008000, 0x80000000, 0x80008000, \
  0x80000000, 0x80008000, 0x80000000, 0x80008000, 0x80008000, 0x80008000, 0x80000000, 0x00008000, \
  0x00000000, 0x00000000, 0x80008000, 0x80008000, 0x00008000, 0x00000000, 0x80008000, 0x00008000, \
  0x80000000, 0x00000000, 0x80008000, 0x00008000, 0x80008000, 0x00000000, 0x80008000, 0x80008000, \
  0x00000000, 0x00008000, 0x80008000, 0x00008000, 0x00008000, 0x00008000, 0x80008000, 0x80008000, \
  0x80000000, 0x00008000, 0x80008000, 0x80008000, 0x80008000, 0x00008000, 0x80008000, 0x00008000, \
  0x00000000, 0x80000000, 0x80008000, 0x00008000, 0x00008000, 0x80000000, 0x80008000, 0x80008000, \
  0x80000000, 0x80000000, 0x80008000, 0x80008000, 0x80008000, 0x80000000, 0x80008000, 0x00008000, \
  0x00000000, 0x80008000, 0x80008000, 0x80008000, 0x00008000, 0x80008000, 0x80008000, 0x00008000, \
  0x80000000, 0x80008000, 0x80008000, 0x00008000, 0x80008000, 0x80008000, 0x80008000, 0x80008000
#line 5 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_common.h"

// half-valued tables (16-byte aligned so one entry = one uint4 load)
constant uint q_iq1s_gridh[8192] __attribute__((aligned(16))) = {Q_IQ1S_GRID_HALF_VALUES};
constant uint q_iq2xxs_gridh[1024] __attribute__((aligned(16))) = {Q_IQ2XXS_GRID_HALF_VALUES};
constant uint q_iq2_signh[512] __attribute__((aligned(16))) = {Q_IQ2_SIGNH_VALUES};
static inline uint4 q_tbl4(constant uint* t, uint i) { return ((constant uint4*)t)[i]; }

// ---- unaligned-safe loads: GGUF blocks are only 2-byte aligned (34/50/66/176/210-byte blocks) ----
static inline uint q_ld32(device const uchar* p) {
  device const ushort* s = (device const ushort*)p;
  return uint(s[0]) | (uint(s[1]) << 16);
}
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
#line 3 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_bench.metal"

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
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_mm.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_mm.metal
// Prefill matmul probes / plain F16 GEMM (peak reference) using the Metal 4 tensor API (MetalPerformancePrimitives).
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_tile.h
// Tensor-API (MPP matmul2d) helpers shared by the prefill matmul and MoE prefill kernels.

// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_blocks.h
// Per-type block decoders for the quant module. Every type exposes the same static interface so the matvec /
// matvec_n / matmul / MoE kernels are written once as templates:
//   QK   elements per (super)block      BB  bytes per block
//   LPB  lanes cooperating on one block EPL elements handled per lane  X4 float4 registers of x per lane
//   load_x(xb, sub, xr)  : load the x slice of lane `sub` (xb = x + block*QK) into xr[X4] (+ precomputed sums)
//   dot(b, sub, xr)      : partial dot of block b (lane sub) with xr
//   deq(b, sub, w)       : dequantized weights of lane sub, w[EPL/4] in the same element order as load_x
//   pos(sub, i)          : element index (0..QK-1) of lane-element i  (element order used by load_x / deq)

#line 11 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_blocks.h"

// ---------------- dense: F32 / F16 / BF16 (virtual block of 256, lanes read 2 coalesced float4/half4) ----------------
struct QB_F32 {
  static constexpr constant int QK = 256, BB = 1024, LPB = 32, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return (i >> 2) * 128 + sub * 4 + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    xr[0] = *(device const float4*)(xb + 4 * sub); xr[1] = *(device const float4*)(xb + 128 + 4 * sub);
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    device const float4* p = (device const float4*)b;
    w[0] = p[sub]; w[1] = p[32 + sub];
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    device const float4* p = (device const float4*)b;
    return metal::dot(p[sub], xr[0]) + metal::dot(p[32 + sub], xr[1]);
  }
};
struct QB_F16 {
  static constexpr constant int QK = 256, BB = 512, LPB = 32, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return (i >> 2) * 128 + sub * 4 + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) { QB_F32::load_x(xb, sub, xr); }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    device const half4* p = (device const half4*)b;
    w[0] = float4(p[sub]); w[1] = float4(p[32 + sub]);
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    device const half4* p = (device const half4*)b;
    return metal::dot(float4(p[sub]), xr[0]) + metal::dot(float4(p[32 + sub]), xr[1]);
  }
};
struct QB_BF16 {
  static constexpr constant int QK = 256, BB = 512, LPB = 32, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return (i >> 2) * 128 + sub * 4 + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) { QB_F32::load_x(xb, sub, xr); }
  static inline float4 cvt(ushort4 u) { return as_type<float4>(uint4(u) << 16); }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    device const ushort4* p = (device const ushort4*)b;
    w[0] = cvt(p[sub]); w[1] = cvt(p[32 + sub]);
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    device const ushort4* p = (device const ushort4*)b;
    return metal::dot(cvt(p[sub]), xr[0]) + metal::dot(cvt(p[32 + sub]), xr[1]);
  }
};

// ---------------- Q8_0: {half d; int8 qs[32]} 34 B. 4 lanes per block, 8 elements each ----------------
struct QB_Q8_0 {
  static constexpr constant int QK = 32, BB = 34, LPB = 4, EPL = 8, X4 = 2;
  static inline ushort pos(ushort sub, ushort i) { return sub * 8 + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    xr[0] = *(device const float4*)(xb + 8 * sub); xr[1] = *(device const float4*)(xb + 8 * sub + 4);
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    float d = q_ldh(b);
    w[0] = d * q_i8x4(q_ld32(b + 2 + 8 * sub)); w[1] = d * q_i8x4(q_ld32(b + 6 + 8 * sub));
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    return q_ldh(b) * (metal::dot(q_i8x4(q_ld32(b + 2 + 8 * sub)), xr[0]) + metal::dot(q_i8x4(q_ld32(b + 6 + 8 * sub)), xr[1]));
  }  // 4 consecutive weights starting at element j (j % 4 == 0) of block b (MoE prefill cooperative-tensor dequant)
  static inline float4 deq4(device const uchar* b, uint j) { return q_ldh(b) * q_i8x4(q_ld32(b + 2 + j)); }
};

// ---------------- Q6_K: {ql[128]; qh[64]; int8 sc[16]; half d} 210 B. 16 lanes; lane = (half h, l0 = 4*(sub&7)),
// elements 128h + 32k + l0 + j (k,j in 0..3): the 4 elements that share one ql/qh byte column ----------------
struct QB_Q6K {
  static constexpr constant int QK = 256, BB = 210, LPB = 16, EPL = 16, X4 = 4;
  static inline ushort pos(ushort sub, ushort i) { return 128 * (sub >> 3) + 32 * (i >> 2) + 4 * (sub & 7) + (i & 3); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float* p = xb + 128 * (sub >> 3) + 4 * (sub & 7);
    for (int k = 0; k < 4; k++) xr[k] = *(device const float4*)(p + 32 * k);
  }
  static inline void quants(device const uchar* b, ushort sub, thread float4* q, thread float* s) {
    const int h = sub >> 3, l0 = 4 * (sub & 7);
    const uint lo = q_ld32(b + 64 * h + l0), hi = q_ld32(b + 64 * h + 32 + l0), qh = q_ld32(b + 128 + 32 * h + l0);
    device const char* sc = (device const char*)(b + 192 + 8 * h + (l0 >> 4));
    q[0] = q_u8x4((lo & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4)) - 32.f;
    q[1] = q_u8x4((hi & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2)) - 32.f;
    q[2] = q_u8x4(((lo >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u)) - 32.f;
    q[3] = q_u8x4(((hi >> 4) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2)) - 32.f;
    s[0] = sc[0]; s[1] = sc[2]; s[2] = sc[4]; s[3] = sc[6];
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    float4 q[4]; float s[4]; quants(b, sub, q, s);
    const float d = q_ldh(b + 208);
    for (int k = 0; k < 4; k++) w[k] = (d * s[k]) * q[k];
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    float4 q[4]; float s[4]; quants(b, sub, q, s);
    return q_ldh(b + 208) * (s[0] * metal::dot(q[0], xr[0]) + s[1] * metal::dot(q[1], xr[1]) +
                             s[2] * metal::dot(q[2], xr[2]) + s[3] * metal::dot(q[3], xr[3]));
  }  static inline float4 deq4(device const uchar* b, uint j) {
    const uint h = j >> 7, kk = (j >> 5) & 3, l = j & 31;
    const uint ql = q_ld32(b + 64 * h + 32 * (kk & 1) + l) >> (4 * (kk >> 1));
    const uint qh = q_ld32(b + 128 + 32 * h + l) >> (2 * kk);
    const float sc = float(((device const char*)b)[192 + 8 * h + (l >> 4) + 2 * kk]);
    return (q_ldh(b + 208) * sc) * (q_u8x4((ql & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4)) - 32.f);
  }
};

// ---------------- Q5_K: {half d, dmin; sc[12]; qh[32]; qs[128]} 176 B. 16 lanes; lane = (chunk j = sub>>2,
// l0 = 8*(sub&3)): elements 64j + l0 + i (low nibble) and 64j + 32 + l0 + i (high nibble), i in 0..7 ----------------
struct QB_Q5K {
  static constexpr constant int QK = 256, BB = 176, LPB = 16, EPL = 16, X4 = 4;
  static inline ushort pos(ushort sub, ushort i) { return 64 * (sub >> 2) + 8 * (sub & 3) + (i >> 3) * 32 + (i & 7); }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float* p = xb + 64 * (sub >> 2) + 8 * (sub & 3);
    xr[0] = *(device const float4*)(p); xr[1] = *(device const float4*)(p + 4);
    xr[2] = *(device const float4*)(p + 32); xr[3] = *(device const float4*)(p + 36);
  }
  static inline void quants(device const uchar* b, ushort sub, thread float4* q, thread float* sm) {
    const int j = sub >> 2, l0 = 8 * (sub & 3);
    const uint ql0 = q_ld32(b + 48 + 32 * j + l0), ql1 = q_ld32(b + 52 + 32 * j + l0);
    const uint qh0 = q_ld32(b + 16 + l0) >> (2 * j), qh1 = q_ld32(b + 20 + l0) >> (2 * j);
    q[0] = q_u8x4((ql0 & 0x0F0F0F0Fu) | ((qh0 & 0x01010101u) << 4));
    q[1] = q_u8x4((ql1 & 0x0F0F0F0Fu) | ((qh1 & 0x01010101u) << 4));
    q[2] = q_u8x4(((ql0 >> 4) & 0x0F0F0F0Fu) | ((qh0 & 0x02020202u) << 3));
    q[3] = q_u8x4(((ql1 >> 4) & 0x0F0F0F0Fu) | ((qh1 & 0x02020202u) << 3));
    const float d = q_ldh(b), dm = q_ldh(b + 2);
    float s1, m1, s2, m2;
    q_scale_min_k4(2 * j, b + 4, s1, m1); q_scale_min_k4(2 * j + 1, b + 4, s2, m2);
    sm[0] = d * s1; sm[1] = dm * m1; sm[2] = d * s2; sm[3] = dm * m2;
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    float4 q[4]; float sm[4]; quants(b, sub, q, sm);
    w[0] = sm[0] * q[0] - sm[1]; w[1] = sm[0] * q[1] - sm[1]; w[2] = sm[2] * q[2] - sm[3]; w[3] = sm[2] * q[3] - sm[3];
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    float4 q[4]; float sm[4]; quants(b, sub, q, sm);
    const float4 one = 1.f;
    return sm[0] * (metal::dot(q[0], xr[0]) + metal::dot(q[1], xr[1])) - sm[1] * metal::dot(xr[0] + xr[1], one) +
           sm[2] * (metal::dot(q[2], xr[2]) + metal::dot(q[3], xr[3])) - sm[3] * metal::dot(xr[2] + xr[3], one);
  }
};

// ---------------- IQ1_S {half d; qs[32]; u16 qh[8]} 50 B and IQ2_XXS {half d; u16 qs[32]} 66 B.
// L lanes per block (16 -> 2 grid groups per lane, 8 -> 4 groups = one sub-block),
// half-valued grid tables (one uint4 constant load per 8 weights, half->float conversion is free) ----
template <int L>
struct QB_IQ1S_T {
  static constexpr constant int QK = 256, BB = 50, LPB = L, EPL = 256 / L, X4 = 256 / L / 4 + 1, G = 256 / L / 8;
  static inline ushort pos(ushort sub, ushort i) { return EPL * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + EPL * sub);
    float4 s = 0;
    for (int k = 0; k < EPL / 4; k++) { xr[k] = p[k]; s += xr[k]; }
    xr[EPL / 4] = float4(metal::dot(s, float4(1.f)), 0, 0, 0);
  }
  static inline uint qsbytes(device const uchar* b, int ib, int l0) {
    return G == 4 ? q_ld32(b + 2 + 4 * ib) : uint(q_ld16(b + 2 + 4 * ib + l0));
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = qsbytes(b, ib, l0);
    const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1), delta = (qh & 0x8000) ? -0.125f : 0.125f;
    for (int g = 0; g < G; g++) {
      const uint4 e = q_tbl4(q_iq1s_gridh, ((qs >> (8 * g)) & 0xFF) | (((qh >> (3 * (l0 + g))) & 7) << 8));
      w[2 * g] = dl * (float4(as_type<half4>(e.xy)) + delta);
      w[2 * g + 1] = dl * (float4(as_type<half4>(e.zw)) + delta);
    }
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = qsbytes(b, ib, l0);
    float s = 0;
    for (int g = 0; g < G; g++) {
      const uint4 e = q_tbl4(q_iq1s_gridh, ((qs >> (8 * g)) & 0xFF) | (((qh >> (3 * (l0 + g))) & 7) << 8));
      s += metal::dot(float4(as_type<half4>(e.xy)), xr[2 * g]) + metal::dot(float4(as_type<half4>(e.zw)), xr[2 * g + 1]);
    }
    const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1);
    return dl * (s + ((qh & 0x8000) ? -0.125f : 0.125f) * xr[EPL / 4].x);
  }  static inline float4 deq4(device const uchar* b, uint j) {
    const uint ib = j >> 5, gq = (j >> 3) & 3;
    const uint qh = q_ld16(b + 34 + 2 * ib), qs = b[2 + 4 * ib + gq];
    const float dl = q_ldh(b) * float(2 * ((qh >> 12) & 7) + 1), delta = (qh & 0x8000) ? -0.125f : 0.125f;
    const uint2 e = ((constant uint2*)q_iq1s_gridh)[2 * (qs | (((qh >> (3 * gq)) & 7) << 8)) + ((j >> 2) & 1)];
    return dl * (float4(as_type<half4>(e)) + delta);
  }
};

template <int L>
struct QB_IQ2XXS_T {
  static constexpr constant int QK = 256, BB = 66, LPB = L, EPL = 256 / L, X4 = 256 / L / 4, G = 256 / L / 8;
  static inline ushort pos(ushort sub, ushort i) { return EPL * sub + i; }
  static inline void load_x(device const float* xb, ushort sub, thread float4* xr) {
    device const float4* p = (device const float4*)(xb + EPL * sub);
    for (int k = 0; k < EPL / 4; k++) xr[k] = p[k];
  }
  static inline uint idxbytes(device const uchar* b, int ib, int l0) {
    return G == 4 ? q_ld32(b + 2 + 8 * ib) : uint(q_ld16(b + 2 + 8 * ib + l0));
  }
  static inline uint4 sgrid(uint idx, uint a1, int l) {
    return q_tbl4(q_iq2xxs_gridh, idx) ^ q_tbl4(q_iq2_signh, (a1 >> (7 * l)) & 127);
  }
  static inline void deq(device const uchar* b, ushort sub, thread float4* w) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = idxbytes(b, ib, l0);
    const float db = q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
    for (int g = 0; g < G; g++) {
      const uint4 e = sgrid((idx >> (8 * g)) & 0xFF, a1, l0 + g);
      w[2 * g] = db * float4(as_type<half4>(e.xy)); w[2 * g + 1] = db * float4(as_type<half4>(e.zw));
    }
  }
  static inline float dot(device const uchar* b, ushort sub, thread const float4* xr) {
    const int ib = (EPL * sub) >> 5, l0 = ((EPL * sub) & 31) >> 3;
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = idxbytes(b, ib, l0);
    float s = 0;
    for (int g = 0; g < G; g++) {
      const uint4 e = sgrid((idx >> (8 * g)) & 0xFF, a1, l0 + g);
      s += metal::dot(float4(as_type<half4>(e.xy)), xr[2 * g]) + metal::dot(float4(as_type<half4>(e.zw)), xr[2 * g + 1]);
    }
    return q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f * s;
  }  static inline float4 deq4(device const uchar* b, uint j) {
    const uint ib = j >> 5, gq = (j >> 3) & 3, hf = (j >> 2) & 1;
    const uint a1 = q_ld32(b + 6 + 8 * ib), idx = b[2 + 8 * ib + gq];
    const uint2 e = ((constant uint2*)q_iq2xxs_gridh)[2 * idx + hf] ^ ((constant uint2*)q_iq2_signh)[2 * ((a1 >> (7 * gq)) & 127) + hf];
    return (q_ldh(b) * (0.5f + float(a1 >> 28)) * 0.25f) * float4(as_type<half4>(e));
  }
};

// production choice (bench_qmatvec: 16 lanes beat 8 on the expert shapes; half tables beat int8 + convert by ~30%)
typedef QB_IQ1S_T<16> QB_IQ1S;
typedef QB_IQ2XXS_T<16> QB_IQ2XXS;
#line 4 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_tile.h"
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

#line 3 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_mm.metal"

// C[n x m] = A[n x k] . B[m x k]^T  (A = activations half, B = weights half, both k-contiguous; C float m-contiguous)
// Threadgroup tile TM (tokens) x TN (weight rows), NSG simdgroups, K loop in chunks of TK.
template <int TM, int TN, int TK, int NSG>
kernel void q_gemm_f16(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], device float* C [[buffer(2)]],
                       constant uint4& dims [[buffer(3)]],  // n, m, k
                       uint2 tg [[threadgroup_position_in_grid]]) {
  const int n = dims.x, m = dims.y, k = dims.z;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A, dextents<int32_t, 2>(k, n));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)B, dextents<int32_t, 2>(k, m));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(m, n));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  const int r0 = tg.x * TN, t0 = tg.y * TM;
  for (int k0 = 0; k0 < k; k0 += TK) {
    auto a = tA.slice(k0, t0);
    auto b = tB.slice(k0, r0);
    op.run(a, b, cT);
  }
  auto c = tC.slice(r0, t0);
  cT.store(c);
}

template [[host_name("q_gemm_f16_64x64")]] kernel void q_gemm_f16<64, 64, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x128")]] kernel void q_gemm_f16<64, 128, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_128x64")]] kernel void q_gemm_f16<128, 64, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_128x128")]] kernel void q_gemm_f16<128, 128, 32, 8>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_32x64")]] kernel void q_gemm_f16<32, 64, 32, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);

// same with F32 activations (left operand float x right half)
template <int TM, int TN, int TK, int NSG>
kernel void q_gemm_f32a(device const float* A [[buffer(0)]], device const half* B [[buffer(1)]], device float* C [[buffer(2)]],
                        constant uint4& dims [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int n = dims.x, m = dims.y, k = dims.z;
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tA((device float*)A, dextents<int32_t, 2>(k, n));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)B, dextents<int32_t, 2>(k, m));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(m, n));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  const int r0 = tg.x * TN, t0 = tg.y * TM;
  for (int k0 = 0; k0 < k; k0 += TK) {
    auto a = tA.slice(k0, t0);
    auto b = tB.slice(k0, r0);
    op.run(a, b, cT);
  }
  auto c = tC.slice(r0, t0);
  cT.store(c);
}
template [[host_name("q_gemm_f32a_64x128")]] kernel void q_gemm_f32a<64, 128, 32, 4>(device const float*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x128_k64")]] kernel void q_gemm_f16<64, 128, 64, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x128_k128")]] kernel void q_gemm_f16<64, 128, 128, 4>(device const half*, device const half*,
    device float*, constant uint4&, uint2);
template [[host_name("q_gemm_f16_64x256")]] kernel void q_gemm_f16<64, 256, 32, 8>(device const half*, device const half*,
    device float*, constant uint4&, uint2);

// ======================================================================================================== prefill
struct QMmArgs {
  uint n;          // tokens (rows of X16 / Y)
  uint rows;       // weight rows (output features)
  uint cols;       // K
  uint row_bytes;  // bytes per weight row
  uint x_stride;   // halves between token rows of X16
  uint y_stride;   // floats between token rows of Y
  uint accumulate;
};

// X F32 -> half (activations for the tensor-API matmuls); n4 = count / 4
kernel void q_f32_to_f16(device const float4* x [[buffer(0)]], device half4* y [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < n4) y[i] = half4(x[i]);
}

// Y[n x rows] (+)= X16[n x cols] . dequant(W)^T. TG tile: TM tokens x TN weight rows, K step 128, NBUF tile buffers.
template <typename Q, int TM, int TN, int NSG, int NBUF>
kernel void q_mm(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]],
                 constant QMmArgs& a [[buffer(3)]], threadgroup half* tiles [[threadgroup(0)]],
                 uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int TK = 128, TKP = 128;
  const uint r0 = tg.x * TN, t0 = tg.y * TM;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)X, dextents<int32_t, 2>(a.cols, a.n),
                                                              array<int32_t, 2>{1, (int32_t)a.x_stride});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.rows, a.n),
                                                               array<int32_t, 2>{1, (int32_t)a.y_stride});
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW0(tiles, dextents<int32_t, 2>(TK, TN));
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW0), float>();
  auto ty = tY.slice(r0, t0);
  if (a.accumulate) cT.load(ty);
  else for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  uint it = 0;
  for (uint k0 = 0; k0 < a.cols; k0 += TK, it++) {
    threadgroup half* tile = tiles + (NBUF > 1 ? (it & 1) : 0) * (TN * TKP);
    q_tile_deq<Q, TN, TKP, 32 * NSG>(W, a.row_bytes, a.rows, r0, k0, tile, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(tile, dextents<int32_t, 2>(TK, TN));
    auto xs = tX.slice(k0, t0);
    op.run(xs, tW, cT);
    if (NBUF == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  cT.store(ty);
}

// dense weights (F16 / BF16 / F32): right operand straight from device memory
template <typename WT, int TM, int TN, int NSG>
kernel void q_mm_dense(device const WT* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]],
                       constant QMmArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const uint r0 = tg.x * TN, t0 = tg.y * TM;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)X, dextents<int32_t, 2>(a.cols, a.n),
                                                              array<int32_t, 2>{1, (int32_t)a.x_stride});
  tensor<device WT, dextents<int32_t, 2>, tensor_inline> tW((device WT*)W, dextents<int32_t, 2>(a.cols, a.rows));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.rows, a.n),
                                                               array<int32_t, 2>{1, (int32_t)a.y_stride});
  constexpr auto desc = matmul2d_descriptor(TM, TN, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  auto ty = tY.slice(r0, t0);
  if (a.accumulate) cT.load(ty);
  else for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  for (uint k0 = 0; k0 < a.cols; k0 += 64) {
    auto xs = tX.slice(k0, t0);
    auto ws = tW.slice(k0, r0);
    op.run(xs, ws, cT);
  }
  cT.store(ty);
}

// Whole-matrix dequantization to half (prefill with large n: dequant once, then the dense tensor-API GEMM).
// One thread per lane-slice; grid = rows x (cols / EPL).
template <typename Q>
kernel void q_deq_f16(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint4& d [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {  // d = (rows, cols, row_bytes)
  const uint row = gid.y, g = gid.x;
  if (row >= d.x || g >= d.y / Q::EPL) return;
  const uint blk = g / Q::LPB;
  const ushort sub = g - blk * Q::LPB;
  float4 w[Q::EPL / 4];
  Q::deq(W + (ulong)row * d.z + (ulong)blk * Q::BB, sub, w);
  device half* o = out + (ulong)row * d.y + blk * Q::QK;
  for (int q = 0; q < Q::EPL / 4; q++) *(device half4*)(o + Q::pos(sub, 4 * q)) = half4(w[q]);
}
#define Q_DEQ_INST(Q, name) \
  template [[host_name("q_deq_f16_" name)]] kernel void q_deq_f16<Q>(device const uchar*, device half*, constant uint4&, uint2);
Q_DEQ_INST(QB_Q8_0, "q8_0")
Q_DEQ_INST(QB_Q5K, "q5_k")
Q_DEQ_INST(QB_Q6K, "q6_k")
Q_DEQ_INST(QB_IQ1S, "iq1_s")
Q_DEQ_INST(QB_IQ2XXS, "iq2_xxs")

#define Q_MM_INST1(Q, name, TM, TN, NSG, NBUF)                                                                   \
  template [[host_name("q_mm_" name "_" #TM "x" #TN "_b" #NBUF)]] kernel void q_mm<Q, TM, TN, NSG, NBUF>(      \
      device const uchar*, device const half*, device float*, constant QMmArgs&, threadgroup half*, uint2, ushort);
#define Q_MM_INST1S(Q, name, TM, TN, NSG, NBUF)                                                                  \
  template [[host_name("q_mm_" name "_" #TM "x" #TN "_b" #NBUF "_s" #NSG)]] kernel void q_mm<Q, TM, TN, NSG, NBUF>( \
      device const uchar*, device const half*, device float*, constant QMmArgs&, threadgroup half*, uint2, ushort);
#define Q_MM_INST(Q, name) \
  Q_MM_INST1(Q, name, 64, 64, 4, 2) Q_MM_INST1(Q, name, 128, 64, 4, 2) Q_MM_INST1(Q, name, 64, 128, 4, 1) \
  Q_MM_INST1(Q, name, 128, 128, 8, 1) Q_MM_INST1(Q, name, 32, 64, 4, 2) Q_MM_INST1(Q, name, 16, 64, 4, 2) \
  Q_MM_INST1(Q, name, 16, 32, 4, 1) Q_MM_INST1(Q, name, 32, 32, 4, 1) Q_MM_INST1S(Q, name, 16, 64, 2, 1) Q_MM_INST1(Q, name, 32, 64, 4, 1)

Q_MM_INST(QB_Q8_0, "q8_0")
Q_MM_INST(QB_Q5K, "q5_k")
Q_MM_INST(QB_Q6K, "q6_k")
Q_MM_INST(QB_IQ1S, "iq1_s")
Q_MM_INST(QB_IQ2XXS, "iq2_xxs")

#define Q_MMD_INST(WT, name)                                                                                     \
  template [[host_name("q_mm_" name "_64x128")]] kernel void q_mm_dense<WT, 64, 128, 4>(device const WT*,       \
      device const half*, device float*, constant QMmArgs&, uint2);
Q_MMD_INST(half, "f16")
Q_MMD_INST(bfloat, "bf16")
Q_MMD_INST(float, "f32")
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_moe.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_moe.metal
// Fused MoE decode (one token): routed experts (IQ1_S / IQ2_XXS) + shared expert (Q6_K / Q8_0).
//   q_moe_gu  : h[e][r] = silu(min(gate_e[r].x, 10)) * clamp(up_e[r].x, -10, 10) * w_e    (all experts, one dispatch)
//   q_moe_down: y[r] += sum_e down_e[r] . h[e]                                            (all experts, one dispatch)
// Expert e < n_routed uses the routed type QR and buffers gate[e]/up[e]/down[e]; e == n_routed is the shared expert
// (type QS, weight 1). Every expert's matrix is a separate binding (buffer + byte offset) so slots can live anywhere.
#line 7 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_moe.metal"

struct QMoeArgs {
  uint n_routed;     // routed experts (0..6)
  uint has_shared;   // 1 if the shared expert is present
  uint dim;          // 5120 (gate/up cols, down rows)
  uint inter;        // 2304 (gate/up rows, down cols)
  uint rb_gu_r, rb_d_r, rb_gu_s, rb_d_s;  // row bytes: routed gate/up, routed down, shared gate/up, shared down
  uint ksplit;       // simdgroups splitting K per threadgroup
  uint tiles;        // threadgroups per expert (q_moe_gu)
};

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
    q_gu_rows<QR, NR>(q_pick6(e, g0, g1, g2, g3, g4, g5), q_pick6(e, u0, u1, u2, u3, u4, u5), a.rb_gu_r, a.inter, row0,
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

Q_MOE_INST(QB_IQ1S, QB_Q6K, "iq1_s_q6_k")
Q_MOE_INST(QB_IQ1S, QB_Q8_0, "iq1_s_q8_0")
Q_MOE_INST(QB_IQ2XXS, QB_Q6K, "iq2_xxs_q6_k")
Q_MOE_INST(QB_IQ2XXS, QB_Q8_0, "iq2_xxs_q8_0")


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
  if (!shared) q_gu_rows<QR, NR>(tbl[e].g, tbl[e].u, a.rb_gu_r, a.inter, row0, x, a.dim, kk, ks, lane, ag, au);
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
Q_MOE_T_ALL(QB_IQ1S, QB_Q6K, "iq1_s_q6_k")
Q_MOE_T_ALL(QB_IQ1S, QB_Q8_0, "iq1_s_q8_0")
Q_MOE_T_ALL(QB_IQ2XXS, QB_Q6K, "iq2_xxs_q6_k")
Q_MOE_T_ALL(QB_IQ2XXS, QB_Q8_0, "iq2_xxs_q8_0")
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_moe_prefill.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_moe_prefill.metal
// MoE prefill: grouped GEMM over experts with the tensor API.
//   q_moe_gather : Xg[i] = half(X[tok[i]]) for the expert-sorted entries, then Xg[total + t] = half(X[t]) (shared)
//   q_moe_mm_gu  : per (expert, token tile, row tile): G = Xg_e . Wg^T, U = Xg_e . Wu^T  (two dequant tiles, two
//                  accumulators) -> H[i][r] = half(silu(min(G,10)) * clamp(U,-10,10) * wt[i])
//   q_moe_mm_down: Z[i] = H_e[i] . Wd^T  (F32)
//   q_moe_reduce : Y[t] += Z[total + t] + sum_k Z[slot_of[t][k]]   (deterministic order)
// Expert weights are reached through a device pointer table (gpuAddress + offset, written by the host into the
// scratch header); tiles map to experts through a host-written tile table.
#line 10 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_moe_prefill.metal"

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
// tables: tbl = device copy (IQ1_S half grid 4096 uint2 | IQ2_XXS half grid 512 uint2 | IQ2 sign masks); IQ2_XXS
// kernels copy their 4 KB grid into threadgroup memory first (lower-latency divergent lookups: -8% MoE time).
template <typename Q, int NSG>
static inline QMoeTbl q_moe_tbl(device const uint2* tbl, threadgroup uint2* tgt, ushort sg, ushort lane) {
  if (is_same<Q, QB_IQ2XXS>::value) {
    for (uint i = sg * 32 + lane; i < 512; i += 32 * NSG) tgt[i] = tbl[4096 + i];
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
#define Q_MOER_GU(Q, name, TM, TN, TK, NSG) Q_MOER_GU1(Q, name, TM, TN, TK, NSG, "")
#define Q_MOER_DN(Q, name, TM, TN, TK, NSG) Q_MOER_DN1(Q, name, TM, TN, TK, NSG, "")
#define Q_MOER_ALL(Q, name)                                                                                         \
  Q_MOER_GU(Q, name, 64, 16, 64, 4) Q_MOER_GU(Q, name, 48, 16, 64, 4) Q_MOER_GU(Q, name, 32, 32, 64, 4)              \
  Q_MOER_GU(Q, name, 16, 32, 64, 4) Q_MOER_GU(Q, name, 64, 16, 32, 4) Q_MOER_GU(Q, name, 48, 16, 32, 4)              \
  Q_MOER_GU(Q, name, 32, 32, 32, 4) Q_MOER_GU(Q, name, 16, 32, 32, 4)                                               \
  Q_MOER_DN(Q, name, 64, 32, 32, 4) Q_MOER_DN(Q, name, 48, 32, 32, 4) Q_MOER_DN(Q, name, 32, 64, 32, 4)              \
  Q_MOER_DN(Q, name, 16, 64, 32, 4)
Q_MOER_ALL(QB_IQ1S, "iq1_s")
Q_MOER_ALL(QB_IQ2XXS, "iq2_xxs")
// shared expert through the grouped kernels (MACSEEK_MOEP_SHDENSE=0)
Q_MOER_GU(QB_Q6K, "q6_k", 32, 32, 32, 4) Q_MOER_DN(QB_Q6K, "q6_k", 32, 64, 32, 4)
Q_MOER_GU(QB_Q8_0, "q8_0", 32, 32, 32, 4) Q_MOER_DN(QB_Q8_0, "q8_0", 32, 64, 32, 4)


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
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_mv.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_mv.metal
// Decode matvec (n = 1): y[r] (+)= W[r] . x  for raw GGUF block rows.
// Threadgroup = RSG row-simdgroups x KS k-split simdgroups (args.ksplit); each simdgroup owns NR consecutive rows and
// keeps its x slice in registers across them. Lanes are split into 32/LPB block slots; a slot's LPB lanes cooperate
// on one block, so each simdgroup iteration reads 32/LPB consecutive blocks of each of its rows.
#line 6 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/quant_mv.metal"

template <typename Q, int NR>
kernel void q_mv(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                 constant QMvArgs& a [[buffer(3)]], threadgroup float* red [[threadgroup(0)]],
                 uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  const ushort ks = a.ksplit, kk = sg % ks, rs = sg / ks, rsg = nsg / ks;
  const uint row0 = (tg * rsg + rs) * NR;
  constexpr ushort BPI = 32 / Q::LPB;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK;
  device const float* xg = x + (row0 / a.rows_per_group) * a.x_group_stride;
  device const uchar* wr[NR];
  for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
  float acc[NR];
  for (int r = 0; r < NR; r++) acc[r] = 0.f;
  for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
    float4 xr[Q::X4];
    Q::load_x(xg + ib * Q::QK, sub, xr);
    for (int r = 0; r < NR; r++) acc[r] += Q::dot(wr[r] + ib * Q::BB, sub, xr);
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
      if (row < a.rows) y[row] = a.accumulate ? y[row] + acc[r] : acc[r];
    }
}

#define Q_MV_INST(Q, name)                                                                                     \
  template [[host_name("q_mv_" name "_r1")]] kernel void q_mv<Q, 1>(device const uchar*, device const float*,  \
      device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);                     \
  template [[host_name("q_mv_" name "_r2")]] kernel void q_mv<Q, 2>(device const uchar*, device const float*,  \
      device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);                     \
  template [[host_name("q_mv_" name "_r4")]] kernel void q_mv<Q, 4>(device const uchar*, device const float*,  \
      device float*, constant QMvArgs&, threadgroup float*, uint, ushort, ushort, ushort);

Q_MV_INST(QB_F32, "f32")
Q_MV_INST(QB_F16, "f16")
Q_MV_INST(QB_BF16, "bf16")
Q_MV_INST(QB_Q8_0, "q8_0")
Q_MV_INST(QB_Q5K, "q5_k")
Q_MV_INST(QB_Q6K, "q6_k")
Q_MV_INST(QB_IQ1S, "iq1_s")
Q_MV_INST(QB_IQ2XXS, "iq2_xxs")

// ---------------- decode matvec with the input RMSNorm fused into the x loads ----------------
// y = W . (x * rsqrt(mean(x^2) + eps) * nw)   (decode q_a_norm -> attn_q_b / indexer q_b).
// Every threadgroup computes the scale itself with exactly the reduction of arch_rmsnorm (threadgroup variant, one row
// of na.dim elements, na.nt threads): virtual thread v sums x[v], x[v + nt], ... ; simd_sum per 32 virtual threads;
// simd_sum of the per-simdgroup sums; rsqrt(sum / dim + eps). The normalized element is (x * r) * nw as in
// arch_rmsnorm, so the result is bit-identical to rmsnorm -> barrier -> q_mv. No block-diagonal groups.
struct QNormArgs { uint dim; float eps; uint nt; uint tiles_per_tg; };

// A threadgroup computes the scale once and then walks tiles_per_tg consecutive row tiles (rsg*NR rows each), so the
// norm prologue is amortized; each row's arithmetic is exactly q_mv's.
template <typename Q, int NR>
kernel void q_mv_rn(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                    constant QMvArgs& a [[buffer(3)]], device const float* nw [[buffer(4)]],
                    constant QNormArgs& na [[buffer(5)]], threadgroup float* red [[threadgroup(0)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float nred[32];
  {
    const ushort nvs = na.nt / 32;
    for (ushort v = sg; v < nvs; v += nsg) {
      const uint vt = uint(v) * 32 + lane;
      float ss = 0.0f;
      for (uint i = vt; i < na.dim; i += na.nt) ss += x[i] * x[i];
      ss = simd_sum(ss);
      if (lane == 0) nred[v] = ss;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rsum = lane < ushort(na.nt / 32) ? nred[lane] : 0.0f;
  rsum = simd_sum(rsum);
  const float rs = rsqrt(rsum / float(na.dim) + na.eps);

  const ushort ks = a.ksplit, kk = sg % ks, rs_ = sg / ks, rsg = nsg / ks;
  constexpr ushort BPI = 32 / Q::LPB;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK;
  const uint ntiles = (a.rows + rsg * NR - 1) / (rsg * NR);
  const uint t0 = tg * na.tiles_per_tg, t1 = min(ntiles, t0 + na.tiles_per_tg);
  for (uint tile = t0; tile < t1; tile++) {
    const uint row0 = (tile * rsg + rs_) * NR;
    device const uchar* wr[NR];
    for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.f;
    for (uint ib = kk * BPI + slot; ib < nb; ib += ks * BPI) {
      float4 xr[Q::X4], wn[Q::X4];
      Q::load_x(x + ib * Q::QK, sub, xr);
      Q::load_x(nw + ib * Q::QK, sub, wn);
      for (int k = 0; k < Q::X4; k++) xr[k] = xr[k] * rs * wn[k];
      for (int r = 0; r < NR; r++) acc[r] += Q::dot(wr[r] + ib * Q::BB, sub, xr);
    }
    for (int r = 0; r < NR; r++) acc[r] = simd_sum(acc[r]);
    if (ks > 1) {
      if (lane == 0) for (int r = 0; r < NR; r++) red[(rs_ * NR + r) * ks + kk] = acc[r];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (kk == 0) {
        for (int r = 0; r < NR; r++) { float s = 0; for (ushort k = 0; k < ks; k++) s += red[(rs_ * NR + r) * ks + k]; acc[r] = s; }
        if (lane == 0)
          for (int r = 0; r < NR; r++) {
            const uint row = row0 + r;
            if (row < a.rows) y[row] = a.accumulate ? y[row] + acc[r] : acc[r];
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (lane == 0) {
      for (int r = 0; r < NR; r++) {
        const uint row = row0 + r;
        if (row < a.rows) y[row] = a.accumulate ? y[row] + acc[r] : acc[r];
      }
    }
  }
}

#define Q_MVRN_INST1(Q, name, NR)                                                                                  \
  template [[host_name("q_mv_rn_" name "_r" #NR)]] kernel void q_mv_rn<Q, NR>(device const uchar*, device const float*, \
      device float*, constant QMvArgs&, device const float*, constant QNormArgs&, threadgroup float*, uint, ushort,      \
      ushort, ushort);
#define Q_MVRN_INST(Q, name) Q_MVRN_INST1(Q, name, 1) Q_MVRN_INST1(Q, name, 2) Q_MVRN_INST1(Q, name, 4)
Q_MVRN_INST(QB_F32, "f32")
Q_MVRN_INST(QB_F16, "f16")
Q_MVRN_INST(QB_BF16, "bf16")
Q_MVRN_INST(QB_Q8_0, "q8_0")
Q_MVRN_INST(QB_Q5K, "q5_k")
Q_MVRN_INST(QB_Q6K, "q6_k")

// ---------------- matvec_n: n in 2..NT activation columns share one weight read ----------------
// The threadgroup stages x[t][c0 .. c0+CK) for all n columns in threadgroup memory as half (CK = one simdgroup
// iteration = 32/LPB blocks), so every x element is fetched from L2 once per threadgroup instead of once per
// simdgroup-row; each lane dequantizes its EPL weights of NR rows once per block and dots them with all n columns.
// No K split (args.ksplit ignored); every row of a threadgroup must lie in the same block-diagonal group.
template <typename Q, int NR, int NT>
kernel void q_mvn(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                  constant QMvArgs& a [[buffer(3)]], threadgroup half* xs [[threadgroup(0)]],
                  uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort nsg [[simdgroups_per_threadgroup]],
                  ushort tid [[thread_index_in_threadgroup]]) {
  constexpr ushort BPI = 32 / Q::LPB;
  constexpr int E4 = Q::EPL / 4, CK = BPI * Q::QK, CK4 = CK / 4;
  const uint row0 = (tg * nsg + sg) * NR;
  const ushort sub = lane % Q::LPB, slot = lane / Q::LPB;
  const uint nb = a.cols / Q::QK, n = a.n, nthr = 32 * nsg;
  device const float* xg = x + ((tg * nsg * NR) / a.rows_per_group) * a.x_group_stride;
  device const uchar* wr[NR];
  for (int r = 0; r < NR; r++) wr[r] = W + (ulong)min(row0 + r, a.rows - 1) * a.row_bytes;
  float acc[NR][NT];
  for (int r = 0; r < NR; r++) _Pragma("unroll") for (int t = 0; t < NT; t++) acc[r][t] = 0.f;
  for (uint c0 = 0; c0 < a.cols; c0 += CK) {
    const uint lim4 = min((uint)CK4, (a.cols - c0) / 4);
    for (uint i = tid; i < n * CK4; i += nthr) {
      const uint t = i / CK4, j = i - t * CK4;
      if (j < lim4) ((threadgroup half4*)xs)[i] = half4(*(device const float4*)(xg + t * a.x_stride + c0 + 4 * j));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ib = c0 / Q::QK + slot;
    if (ib < nb) {
      float4 w[NR][E4];
      for (int r = 0; r < NR; r++) Q::deq(wr[r] + ib * Q::BB, sub, w[r]);
      threadgroup const half* xb = xs + slot * Q::QK;
      _Pragma("unroll") for (int t = 0; t < NT; t++) {
        if (t < (int)n) {
          float4 xr[E4];
          for (int k = 0; k < E4; k++) xr[k] = float4(*(threadgroup const half4*)(xb + t * CK + Q::pos(sub, 4 * k)));
          for (int r = 0; r < NR; r++) {
            float s = 0;
            for (int k = 0; k < E4; k++) s += metal::dot(w[r][k], xr[k]);
            acc[r][t] += s;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r = 0; r < NR; r++) _Pragma("unroll") for (int t = 0; t < NT; t++) acc[r][t] = simd_sum(acc[r][t]);
  if (lane < NT && lane < n) {
    for (int r = 0; r < NR; r++) {
      const uint row = row0 + r;
      float v = 0;
      _Pragma("unroll") for (int t = 0; t < NT; t++) if (t == lane) v = acc[r][t];
      if (row < a.rows) {
        device float* yp = y + lane * a.y_stride + row;
        *yp = a.accumulate ? *yp + v : v;
      }
    }
  }
}

#define Q_MVN_INST1(Q, name, NR, NT)                                                                          \
  template [[host_name("q_mvn_" name "_r" #NR "_n" #NT)]] kernel void q_mvn<Q, NR, NT>(device const uchar*, \
      device const float*, device float*, constant QMvArgs&, threadgroup half*, uint, ushort, ushort, ushort, ushort);
#define Q_MVN_INST(Q, name) \
  Q_MVN_INST1(Q, name, 1, 2) Q_MVN_INST1(Q, name, 1, 4) Q_MVN_INST1(Q, name, 1, 8) Q_MVN_INST1(Q, name, 1, 16) \
  Q_MVN_INST1(Q, name, 2, 2) Q_MVN_INST1(Q, name, 2, 4) Q_MVN_INST1(Q, name, 2, 8) Q_MVN_INST1(Q, name, 2, 16)

Q_MVN_INST(QB_F32, "f32")
Q_MVN_INST(QB_F16, "f16")
Q_MVN_INST(QB_BF16, "bf16")
Q_MVN_INST(QB_Q8_0, "q8_0")
Q_MVN_INST(QB_Q5K, "q5_k")
Q_MVN_INST(QB_Q6K, "q6_k")
Q_MVN_INST(QB_IQ1S, "iq1_s")
Q_MVN_INST(QB_IQ2XXS, "iq2_xxs")
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/smoke.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/smoke.metal
#line 2 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/smoke.metal"
kernel void smoke_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]],
                      constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) c[i] = a[i] + b[i];
}
#line 1 "/Users/arnav/Desktop/Projects/MacSeek/src/kernels/sync.metal"
// ---- /Users/arnav/Desktop/Projects/MacSeek/src/kernels/sync.metal
// GPU<->CPU handshake probe/test kernels (bench/sync_probe.mm, tests/test_dec_sync.mm; docs/modules/dec_sync.md).
// Not used by the engine: the decode path keeps MTLSharedEvent handshakes and hides them behind independent GPU work
// (MACSEEK_DEC_SYNC=sgu|shared). Measured here: CPU stores reach a running kernel only via volatile coherent(device)
// loads, eviction-driven (tens of us to ms) and unordered across cache lines, in both directions => no GPU polling.
#include <metal_stdlib>
using namespace metal;

// Publish a monotone value to the CPU: flag = v (after all previous writes of this thread are visible).
kernel void sync_publish(device atomic_uint* flag [[buffer(0)]], constant uint& v [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_store_explicit(flag, v, memory_order_relaxed);
}

struct SyncWaitArgs {
  uint want;        // wait until *go >= want
  uint max_spin;    // bound (iterations of the poll loop); on timeout stat[0]++ and the table is NOT copied
  uint n_words;     // uint32 words to copy src -> dst after the wait (0 = none)
  uint stat_idx;    // stat[2 + stat_idx] = spins used (diagnostics)
};

// One simdgroup: lane 0 spins on go, then all lanes copy the CPU-written table words into dst.
// stat[0] = number of timeouts (atomic add), stat[2 + i] = spins for wait i.
// MEASURED (bench/sync_probe, M5 Pro): CPU stores reach a running kernel only through `volatile coherent(device)`
// loads (atomic loads / RMW / fences keep hitting a stale GPU cache line, sometimes for seconds), visibility is
// eviction-driven (round trip p50 40-300 us, p99 1-4 ms) and NOT ordered across cache lines (payload written before
// the flag is often stale). So this is a probe of a rejected design, not a production protocol.
kernel void sync_wait_copy(device coherent(device) uint* go [[buffer(0)]], constant SyncWaitArgs& a [[buffer(1)]],
                           device coherent(device) uint* src [[buffer(2)]], device uint* dst [[buffer(3)]],
                           device atomic_uint* stat [[buffer(4)]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  uint n = 0;
  bool ok = false;
  if (lane == 0) {
    for (; n < a.max_spin; n++)
      if (*(volatile device coherent(device) uint*)go >= a.want) { ok = true; break; }
    if (!ok) atomic_fetch_add_explicit(stat, 1u, memory_order_relaxed);
    atomic_store_explicit(stat + 2 + a.stat_idx, n, memory_order_relaxed);
  }
  ok = simd_broadcast((ushort)ok, (ushort)0) != 0;
  if (!ok) return;
  for (uint i = lane; i < a.n_words; i += 32) dst[i] = ((volatile device coherent(device) uint*)src)[i];
}

// probe-only: GPU busy work of ~iters rounds per thread in `groups` threadgroups (gap filler)
kernel void sync_busy(device uint* junk [[buffer(0)]], constant uint& iters [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
  uint v = tid;
  for (uint i = 0; i < iters; i++) v = v * 1664525u + 1013904223u + (v >> 13);
  if (v == 0x12345678u) junk[tid & 1023] = v;
}

// ---- probe-only kernel: one dependent "layer". out[l] = mix(out[l-1] ^ tbl[l*ts]) after `iters` rounds of work;
// checks that tbl[l*ts] equals the value the CPU must have written for out[l-1] (stale-read detector: bad[0]++).
// mode bit 0: read tbl with an atomic load (else a plain load). bit 1: publish flag = l+1 at the end (atomic store).
struct SyncLayerArgs { uint l, iters, ts, mode, check; };
static inline uint sync_mix(uint x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
kernel void sync_layer(device uint* out [[buffer(0)]], device atomic_uint* tbl [[buffer(1)]],
                       constant SyncLayerArgs& a [[buffer(2)]], device atomic_uint* bad [[buffer(3)]],
                       device atomic_uint* flag [[buffer(4)]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint prev = a.l ? out[a.l - 1] : 12345u;
  uint t;
  if (a.mode & 1) t = atomic_load_explicit(tbl + a.l * a.ts, memory_order_relaxed);
  else t = ((device uint*)tbl)[a.l * a.ts];
  if (a.check && lane == 0 && t != sync_mix(prev + 0x9e3779b9U * (a.l + 1))) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
  uint v = prev ^ t ^ lane;
  for (uint i = 0; i < a.iters; i++) v = sync_mix(v + i);
  v = simd_xor(v);
  if (lane == 0) {
    out[a.l] = v;
    if (a.mode & 2) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_store_explicit(flag, a.l + 1, memory_order_relaxed);
    }
  }
}

// ---- probe-only: CPU->running-kernel visibility variants. Spins (bounded) until *p >= want; out[0] = spins,
// out[1] = 1 if seen. mode: 0 relaxed atomic load, 1 atomic fetch_add(0), 2 volatile load, 3 coherent(device) load,
// 4 atomic exchange-free compare_exchange(0->0), 5 relaxed load + seq_cst fence per poll
kernel void sync_vis(device atomic_uint* p [[buffer(0)]], constant uint4& a [[buffer(1)]], device uint* out [[buffer(2)]],
                     device coherent(device) uint* pc [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint want = a.x, max_spin = a.y, mode = a.z;
  uint n = 0; bool ok = false;
  for (; n < max_spin; n++) {
    uint v;
    if (mode == 0) v = atomic_load_explicit(p, memory_order_relaxed);
    else if (mode == 1) v = atomic_fetch_add_explicit(p, 0u, memory_order_relaxed);
    else if (mode == 2) v = *(volatile device uint*)p;
    else if (mode == 3) v = *pc;
    else if (mode == 4) { uint e = 0; atomic_compare_exchange_weak_explicit(p, &e, 0u, memory_order_relaxed, memory_order_relaxed); v = e; }
    else { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); v = atomic_load_explicit(p, memory_order_relaxed); }
    if (v >= want) { ok = true; break; }
  }
  out[0] = n; out[1] = ok ? 1u : 0u;
}

// ---- probe-only: ping-pong between the CPU and one GPU thread inside ONE kernel.
// for k = 1..n: wait ping >= k (coherent loads, bounded); check CPU payload cp[k % 16] == k*2654435761 (coherent loads);
// write GPU payload gp[k % 16] = k*40503 (store kind a.w: 0 plain, 1 coherent, 2 atomic) ; fence; pong = k (atomic store)
// res[0] = rounds done, res[1] = CPU payload mismatches, res[2] = timeouts
kernel void sync_pingpong(device coherent(device) uint* ping [[buffer(0)]], device atomic_uint* pong [[buffer(1)]],
                          device coherent(device) uint* cp [[buffer(2)]], device uint* gp [[buffer(3)]],
                          constant uint4& a [[buffer(4)]], device uint* res [[buffer(5)]],
                          device coherent(device) uint* gpc [[buffer(6)]], device atomic_uint* gpa [[buffer(7)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint n = a.x, max_spin = a.y, kind = a.w;
  uint bad = 0, to = 0, k = 1;
  for (; k <= n; k++) {
    uint s = 0;
    while (*(volatile device coherent(device) uint*)ping < k && s < max_spin) s++;
    if (s >= max_spin) { to++; break; }
    if (((volatile device coherent(device) uint*)cp)[k % 16] != k * 2654435761u) bad++;
    const uint v = k * 40503u;
    if (kind == 0) gp[k % 16] = v;
    else if (kind == 1) gpc[k % 16] = v;
    else atomic_store_explicit(gpa + k % 16, v, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_store_explicit(pong, k, memory_order_relaxed);
  }
  res[0] = k - 1; res[1] = bad; res[2] = to;
}

// ---- tests/test_dec_sync.mm: emulated decode layer (router -> handshake -> gap work -> table MoE) ----
// All kernels: `groups` threadgroups x 256 threads burn `iters` rounds (stand-in for real work); thread 0 of
// threadgroup 0 does the protocol logic. Integer math only so the CPU reference matches exactly.
struct SyncTArgs { uint l, n_exp, iters, tok; };
static inline uint sync_t_burn(uint tid, uint iters) {
  uint v = tid;
  for (uint i = 0; i < iters; i++) v = v * 1664525u + 1013904223u + (v >> 13);
  return v;
}
static inline uint sync_t_sig(uint l, uint e, uint j, uint n_exp) { return 0x5EED0000u ^ ((l * n_exp + e) * 3u + j); }

// router: 6 distinct skewed ids from the state
kernel void sync_t_router(device uint* st [[buffer(0)]], device int* rids [[buffer(1)]], constant SyncTArgs& a [[buffer(2)]],
                          device uint* junk [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const uint v = sync_t_burn(tid, a.iters);
  if (v == 0x12345678u) junk[0] = v;
  if (tid != 0) return;
  const uint s = st[0];
  int ids[6];
  for (int k = 0; k < 6; k++) {
    const uint h = sync_mix(s + 0x9e3779b9u * (uint)(k + 1) + a.l * 7919u);
    const uint u = h & 0xffffu;
    const uint u3 = (((u * u) >> 16) * u) >> 16;   // skewed: hot experts (cache hits) dominate
    uint id = (uint)(((ulong)u3 * a.n_exp) >> 16);
    for (bool dup = true; dup;) { dup = false; for (int q = 0; q < k; q++) if ((uint)ids[q] == id) { dup = true; id = (id + 1) % a.n_exp; } }
    ids[k] = (int)id;
  }
  for (int k = 0; k < 6; k++) rids[k] = ids[k];
}

// gap work (shared-expert stand-in): sh = f(state)
kernel void sync_t_gap(device uint* st [[buffer(0)]], device uint* sh [[buffer(1)]], constant SyncTArgs& a [[buffer(2)]],
                       device uint* junk [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const uint v = sync_t_burn(tid, a.iters);
  if (v == 0x12345678u) junk[0] = v;
  if (tid != 0) return;
  sh[0] = sync_mix(st[0] ^ (a.l * 0x1234567u) ^ a.tok);
}

// routed MoE stand-in: expert matrices from the CPU-written table; checks every matrix signature against the router
// ids (bad[0]) and that the gap result belongs to this layer's state (bad[1]: ordering), folds expert data into state.
struct SyncTPtr { device const uint* m[3]; };
kernel void sync_t_moe(device uint* st [[buffer(0)]], device const uint* sh [[buffer(1)]], constant SyncTArgs& a [[buffer(2)]],
                       device uint* junk [[buffer(3)]], device const int* rids [[buffer(4)]],
                       device const SyncTPtr* tbl [[buffer(5)]], device atomic_uint* bad [[buffer(6)]],
                       uint tid [[thread_position_in_grid]]) {
  const uint v = sync_t_burn(tid, a.iters);
  if (v == 0x12345678u) junk[0] = v;
  if (tid != 0) return;
  const uint s = st[0];
  if (sh[0] != sync_mix(s ^ (a.l * 0x1234567u) ^ a.tok)) atomic_fetch_add_explicit(bad + 1, 1u, memory_order_relaxed);
  uint acc = 0;
  for (int k = 0; k < 6; k++)
    for (int j = 0; j < 3; j++) {
      device const uint* p = tbl[k].m[j];
      if (p[0] != sync_t_sig(a.l, (uint)rids[k], (uint)j, a.n_exp)) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
      acc ^= p[1 + (s + (uint)(k * 3 + j)) % 15u] + (uint)(k * 3 + j);
    }
  st[0] = sync_mix(s ^ acc ^ sh[0]);
}
