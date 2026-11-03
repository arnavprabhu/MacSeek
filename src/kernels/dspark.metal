// dspark block drafter kernels (prefix dsp_): see src/model/dspark.hpp. The drafter only proposes tokens (the target's
// exact verify decides), so these favour simplicity over the last bit of speed.
#include "common.h"

constant constexpr int DSP_D = 512, DSP_RD = 64, DSP_W = 128, DSP_H = 64;

inline float2 dsp_rot(float2 x, float2 cs) { return float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x); }
inline float2 dsp_rot_inv(float2 x, float2 cs) { return float2(x.x * cs.x + x.y * cs.y, -x.x * cs.y + x.y * cs.x); }

// ---- kv rows: y = rope(rmsnorm(x) * w) as half. Row i at position pos0 + i; dst row = ring ? pos % 128 : i.
// grid (n), 128 threads x 4 dims
struct DspKvArgs { int pos0, n, ring; float eps; };
kernel void dsp_kv(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                   device const float2* tab [[buffer(2)]], device half* dst [[buffer(3)]],
                   constant DspKvArgs& a [[buffer(4)]], uint i [[threadgroup_position_in_grid]],
                   ushort t [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup float red[4];
  const int pos = a.pos0 + int(i);
  float4 v = ((device const float4*)(x + (size_t)i * DSP_D))[t];
  float ss = simd_sum(dot(v, v));
  if (ln == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss = red[0] + red[1] + red[2] + red[3];
  v = ((device const float4*)w)[t] * (v * rsqrt(ss / DSP_D + a.eps));
  if (t >= 112) {
    const int pi = 2 * (t - 112);
    v.xy = dsp_rot(v.xy, tab[(size_t)pos * 32 + pi]);
    v.zw = dsp_rot(v.zw, tab[(size_t)pos * 32 + pi + 1]);
  }
  const int row = a.ring ? pos % DSP_W : int(i);
  ((device half4*)(dst + (size_t)row * DSP_D))[t] = half4(v);
}

// ---- block attention: queries at positions P..P+B-1 see the ring rows of positions [P-128, P-1] (slot_pos) and all
// B block rows (non-causal), per-head sink. q pre-rope [B][64][512] -> out (inverse-roped) [B][64][512].
// grid (64 heads, B), 128 threads
struct DspAttnArgs { int P, B; float scale; };
kernel void dsp_attn(device const float* q [[buffer(0)]], device const half* ring [[buffer(1)]],
                     device const int* slot_pos [[buffer(2)]], device const half* blk [[buffer(3)]],
                     device const float* sinks [[buffer(4)]], device const float2* tab [[buffer(5)]],
                     device float* out [[buffer(6)]], constant DspAttnArgs& a [[buffer(7)]],
                     uint2 g [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  threadgroup float qs[DSP_D];
  threadgroup float sc[DSP_W + 16];
  threadgroup float red[8];
  const int h = int(g.x), i = int(g.y), pos = a.P + i;
  device const float* qr = q + ((size_t)i * DSP_H + h) * DSP_D;
  float4 qv = ((device const float4*)qr)[t];
  if (t >= 112) {
    const int pi = 2 * (t - 112);
    qv.xy = dsp_rot(qv.xy, tab[(size_t)pos * 32 + pi]);
    qv.zw = dsp_rot(qv.zw, tab[(size_t)pos * 32 + pi + 1]);
  }
  ((threadgroup float4*)qs)[t] = qv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int R = DSP_W + a.B;
  for (int j = sg; j < R; j += 4) {
    device const half* k;
    bool ok = true;
    if (j < DSP_W) { const int p = slot_pos[j]; ok = p >= 0 && p >= a.P - DSP_W && p < a.P; k = ring + (size_t)j * DSP_D; }
    else k = blk + (size_t)(j - DSP_W) * DSP_D;
    float s = 0;
    if (ok) for (int d = ln * 16; d < ln * 16 + 16; d++) s += qs[d] * float(k[d]);
    s = simd_sum(s);
    if (ln == 0) sc[j] = ok ? s * a.scale : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float sink = sinks[h];
  float m = sink;
  for (int j = 0; j < R; j++) m = max(m, sc[j]);
  float l = exp(sink - m);
  for (int j = 0; j < R; j++) l += exp(sc[j] - m);
  float4 o = 0;
  for (int j = 0; j < R; j++) {
    const float p = exp(sc[j] - m);
    if (p == 0.f) continue;
    device const half* k = j < DSP_W ? ring + (size_t)j * DSP_D : blk + (size_t)(j - DSP_W) * DSP_D;
    o += p * float4(((device const half4*)k)[t]);
  }
  o /= l;
  if (t >= 112) {
    const int pi = 2 * (t - 112);
    o.xy = dsp_rot_inv(o.xy, tab[(size_t)pos * 32 + pi]);
    o.zw = dsp_rot_inv(o.zw, tab[(size_t)pos * 32 + pi + 1]);
  }
  ((device float4*)(out + ((size_t)i * DSP_H + h) * DSP_D))[t] = o;
  (void)red;
}

// ---- MoE router: score = sqrt(softplus(logit)); top-k of score + bias; w = score / (sum + 1e-20) * scale.
// grid (B), 128 threads (n_exp <= 128)
struct DspRouteArgs { int n_exp, k; float scale; };
kernel void dsp_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device int* ids [[buffer(2)]], device float* wts [[buffer(3)]], constant DspRouteArgs& a [[buffer(4)]],
                      uint b [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup float sel[128], scr[128];
  if (t < a.n_exp) {
    const float v = logits[(size_t)b * a.n_exp + t];
    const float sp = v > 20.f ? v : log(1.f + exp(v));
    scr[t] = sqrt(sp);
    sel[t] = scr[t] + bias[t];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    int top[8]; float sum = 0;
    for (int k = 0; k < a.k; k++) {
      int best = -1;
      for (int x = 0; x < a.n_exp; x++) {
        bool used = false;
        for (int j = 0; j < k; j++) used |= top[j] == x;
        if (!used && (best < 0 || sel[x] > sel[best])) best = x;
      }
      top[k] = best; sum += scr[best];
    }
    for (int k = 0; k < a.k; k++) { ids[b * a.k + k] = top[k]; wts[b * a.k + k] = scr[top[k]] / (sum + 1e-20f) * a.scale; }
  }
}

// ---- MXFP4 rows (block: e8m0 scale byte + 16 bytes of e2m1 pairs; element j / j+16 = low / high nibble) ----
constant float dsp_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
inline float dsp_e8m0_half(uint x) { return x < 2 ? as_type<float>(0x00200000u << x) : as_type<float>((x - 1) << 23); }
// dot of one MXFP4 row (nb blocks) with x, lanes over blocks; simd_sum'd
inline float dsp_mx_dot(device const uchar* row, device const float* x, uint nb, ushort ln) {
  float s = 0;
  for (uint b = ln; b < nb; b += 32) {
    device const uchar* p = row + b * 17;
    const float d = dsp_e8m0_half(p[0]);
    device const float* xb = x + b * 32;
    float acc = 0;
    for (int j = 0; j < 16; j++) {
      const uint q = p[1 + j];
      acc += dsp_fp4[q & 15] * xb[j] + dsp_fp4[q >> 4] * xb[j + 16];
    }
    s += d * acc;
  }
  return simd_sum(s);
}

// routed gate/up for entries e = token * k + slot: h[e][row] = silu(min(g, lim)) * clamp(u, -lim, lim) * w[e]
// grid (rows / 8, n_entries), 256 threads (one row per simdgroup)
struct DspMxArgs { uint k, rows, cols, row_bytes; ulong expert_bytes; float lim; };
kernel void dsp_mx_gu(device const float* x [[buffer(0)]], device const uchar* G [[buffer(1)]], device const uchar* U [[buffer(2)]],
                      device const int* ids [[buffer(3)]], device const float* wts [[buffer(4)]], device float* h [[buffer(5)]],
                      constant DspMxArgs& a [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort ln [[thread_index_in_simdgroup]]) {
  const uint row = g.x * 8 + sg, e = g.y;
  if (row >= a.rows) return;
  const uint tkn = e / a.k;
  const ulong off = (ulong)ids[e] * a.expert_bytes + (ulong)row * a.row_bytes;
  device const float* xt = x + (size_t)tkn * a.cols;
  const float gv = dsp_mx_dot(G + off, xt, a.cols / 32, ln), uv = dsp_mx_dot(U + off, xt, a.cols / 32, ln);
  if (ln == 0) {
    const float gg = min(gv, a.lim), uu = clamp(uv, -a.lim, a.lim);
    h[(size_t)e * a.rows + row] = gg / (1.f + exp(-gg)) * uu * wts[e];
  }
}
// routed down: y[t][row] += sum_slot down_{ids[t][slot]}[row] . h[t*k + slot]. grid (rows / 8, n_tokens), 256 threads
kernel void dsp_mx_down(device const float* h [[buffer(0)]], device const uchar* Dn [[buffer(1)]], device const int* ids [[buffer(2)]],
                        device float* y [[buffer(3)]], constant DspMxArgs& a [[buffer(4)]],
                        uint2 g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort ln [[thread_index_in_simdgroup]]) {
  const uint row = g.x * 8 + sg, tkn = g.y;
  if (row >= a.rows) return;
  float s = 0;
  for (uint k = 0; k < a.k; k++) {
    const uint e = tkn * a.k + k;
    s += dsp_mx_dot(Dn + (ulong)ids[e] * a.expert_bytes + (ulong)row * a.row_bytes, h + (size_t)e * a.cols, a.cols / 32, ln);
  }
  if (ln == 0) y[(size_t)tkn * a.rows + row] += s;
}

// shared expert activation: out = silu(min(g, lim)) * clamp(u, -lim, lim)
kernel void dsp_swiglu(device const float* gt [[buffer(0)]], device const float* up [[buffer(1)]], device float* out [[buffer(2)]],
                       constant float& lim [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  const float g = min(gt[i], lim), u = clamp(up[i], -lim, lim);
  out[i] = g / (1.f + exp(-g)) * u;
}

// Markov head input: out[j] = float(w1[id][j]) (BF16 rows of `dim`), id = ids[idx]
kernel void dsp_gather_bf16(device const ushort* w1 [[buffer(0)]], device const int* ids [[buffer(1)]], device float* out [[buffer(2)]],
                            constant uint2& a [[buffer(3)]], uint j [[thread_position_in_grid]]) {
  const int id = ids[a.x];
  out[j] = as_type<float>(uint(w1[(size_t)id * a.y + j]) << 16);
}
