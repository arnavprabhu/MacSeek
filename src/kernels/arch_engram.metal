// Engram block (GPU part) and row dequant kernels: IQ4_NL engram table rows, Q6_K token_embd rows.
#include "arch_common.h"

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
