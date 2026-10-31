// MoE router: scores = sqrt(softplus(logit)); top-6 of (scores + bias); weights = score / (sum + 1e-20) * route_scale.
// Plus prefill routing lists (per-expert token lists for grouped GEMM), deterministic (no atomics).
#include "arch_common.h"

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
