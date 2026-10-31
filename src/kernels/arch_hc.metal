// Hyper-connection kernels: hc_post (expand), hc_mixes (rms + F32 20480->24 projection + Sinkhorn), hc_pre (collapse)
// fused with the following RMSNorm. Layout: h [n][4][5120] F32, mix [n][24] = pre[4], post[4], comb[4][4] (comb[j][k]
// at 8 + 4j + k), fn [24][20480] F32 (GGUF hc_*_fn as stored), x [n][5120].
//
// One "step" at a sublayer boundary (see SPEC 3):
//   h_new = has_post ? post[i]*x_sub + sum_j comb[j][i]*h[j] : h          (mix_prev supplies post/comb)
//   mix_out = has_mixes ? hc_split_sinkhorn(fn . h_new * rsqrt(mean(h_new^2) + norm_eps)) : -
//   x_out = rmsnorm(sum_j pre_lag[j] * h_new[j]) * norm_w                    (pre_lag = mix_prev[0:4], lagged mixes)
#include "arch_common.h"

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
