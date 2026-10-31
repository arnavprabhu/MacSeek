// Attention module: window KV write, compressor (pool + norm + index key + rope + fp4 sim), state carry, ring commit.
#include "attn_common.h"

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
