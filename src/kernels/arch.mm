// Host side of the arch kernels (see arch.hpp).
#include "kernels/arch.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "core/env.hpp"

namespace ms::arch {
namespace {

struct HcParams { uint32_t n_groups, has_post, has_mixes, n_tokens; float norm_eps, hc_eps; int32_t iters; uint32_t fused, has_pre; };
struct RouterParams { uint32_t n_tokens, n_groups; float route_scale; uint32_t pad; };
struct EngramParams { uint32_t n_tokens; float eps, clamp; uint32_t pad; };
struct EmbedParams { uint32_t n_tokens, row_stride, copies, dim; };
struct SampleParams { uint32_t V, n_groups, chunk; float inv_temp; uint32_t seed_lo, seed_hi, gumbel, k; };
struct NormParams { uint32_t n_rows, dim; float eps; uint32_t has_weight, in_stride, out_stride; };

constexpr int kCounterHc = 0, kCounterRouter = 1, kCounterRows = 64;
constexpr int kHcMaxGroups = 64;
constexpr int kArgmaxGroups = 64, kTopkGroups = 32, kSampleT = 1024;

void* pso(const char* name) { return Device::get().pipeline(name); }
#define ARCH_PSO(var, name) static void* var = pso(name)

enum class HcMode { Fused = 0, Split = 1, Rows = 2 };
HcMode hc_mode() { return (HcMode)options().hc_mode; }
uint32_t hc_threads() {
  int v = options().hc_threads;
  return (v == 128 || v == 256 || v == 512 || v == 1024) ? (uint32_t)v : 512u;
}
bool hc_prefill_rows() { return options().hc_prefill_rows; }
uint32_t router_groups() {
  int v = options().router_groups;
  return (v > 0 && v <= 384 && 384 % v == 0) ? (uint32_t)v : 96u;
}

Buffer or_dummy(const Buffer& b, const Scratch& s) { return b ? b : s.dummy; }
Buffer counter(const Scratch& s, int slot) { return s.counters.sub((size_t)slot * 4, 4); }

}  // namespace

Options& options() {
  static Options o = [] {
    Options x;
    std::string m = env_str("MACSEEK_ARCH_HC", "fused");
    x.hc_mode = m == "split" ? 1 : m == "rows" ? 2 : 0;
    x.hc_threads = (int)env_int("MACSEEK_ARCH_HC_T", 512);
    std::string pf = env_str("MACSEEK_ARCH_HC_PREFILL", "mma");
    x.hc_prefill_rows = pf == "rows";
    x.hc_prefill_mma8 = pf == "mma8";
    x.router_groups = (int)env_int("MACSEEK_ARCH_ROUTER_G", 96);
    return x;
  }();
  return o;
}

Scratch Scratch::create(int max_tokens, int max_rows) {
  auto& d = Device::get();
  Scratch s;
  s.max_tokens = max_tokens;
  s.max_rows = max_rows;
  s.hc_part = d.alloc(kHcMaxGroups * 32 * 4, "arch.hc_part");
  s.counters = d.alloc((size_t)(kCounterRows + 2 * max_rows) * 4, "arch.counters");
  s.router_logits = d.alloc(kExperts * 4, "arch.router_logits");
  s.rstd = d.alloc((size_t)max_tokens * 4, "arch.rstd");
  s.gate = d.alloc((size_t)max_tokens * 16, "arch.gate");
  size_t argmax_b = (size_t)max_rows * kArgmaxGroups * 8;
  size_t topk_b = (size_t)max_rows * kTopkGroups * (64 * 8 + 8);
  s.sample_part = d.alloc(std::max(argmax_b, topk_b), "arch.sample_part");
  s.dummy = d.alloc(256, "arch.dummy");
  return s;
}

void fill_identity_mix(float* m) {
  std::memset(m, 0, kMix * sizeof(float));
  m[0] = 1.0f;
}

void encode_hc_step(Enc& e, const HcStep& a, const Scratch& s) {
  ARCH_PSO(p_step, "arch_hc_step");
  ARCH_PSO(p_fin, "arch_hc_step_finalize");
  ARCH_PSO(p_pp, "arch_hc_post_pre_n");
  ARCH_PSO(p_mma, "arch_hc_mixes_mma");
  ARCH_PSO(p_rows, "arch_hc_mixes_rows");
  ARCH_PSO(p_mma32, "arch_hc_mixes_mma32");
  const bool has_post = (bool)a.x_sub, has_mixes = (bool)a.fn, has_pre = (bool)a.x_out;
  HcParams P{0, has_post ? 1u : 0u, has_mixes ? 1u : 0u, (uint32_t)a.n, kNormEps, kHcEps, kSinkhornIters, 1,
             has_pre ? 1u : 0u};
  const Buffer h_new = has_post ? a.h_out : a.h_in;
  HcMode mode = hc_mode();
  if (a.n == 1 && mode != HcMode::Rows) {
    const uint32_t T = hc_threads();
    P.n_groups = kDim / T;
    P.fused = mode == HcMode::Fused ? 1u : 0u;
    e.set_pipeline(p_step);
    e.set_buffer(0, a.h_in);
    e.set_buffer(1, has_post ? a.h_out : s.dummy);
    e.set_buffer(2, or_dummy(a.x_sub, s));
    e.set_buffer(3, a.mix_prev);
    e.set_buffer(4, or_dummy(a.fn, s));
    e.set_buffer(5, or_dummy(a.scale, s));
    e.set_buffer(6, or_dummy(a.base, s));
    e.set_buffer(7, or_dummy(a.norm_w, s));
    e.set_buffer(8, or_dummy(a.x_out, s));
    e.set_buffer(9, or_dummy(a.mix_out, s));
    e.set_buffer(10, s.hc_part);
    e.set_buffer(11, counter(s, kCounterHc));
    e.set_value(12, P);
    e.dispatch_groups(P.n_groups, 1, 1, T);
    if (mode == HcMode::Split) {
      e.barrier();
      e.set_pipeline(p_fin);
      e.set_buffer(0, s.hc_part);
      e.set_buffer(1, or_dummy(a.x_out, s));
      e.set_buffer(2, or_dummy(a.norm_w, s));
      e.set_buffer(3, or_dummy(a.mix_out, s));
      e.set_buffer(4, or_dummy(a.scale, s));
      e.set_buffer(5, or_dummy(a.base, s));
      e.set_value(6, P);
      e.dispatch_groups(1, 1, 1, 256);
    }
    return;
  }
  if (a.n > s.max_tokens) { fprintf(stderr, "arch: hc_step n=%d > scratch max_tokens %d\n", a.n, s.max_tokens); abort(); }
  e.set_pipeline(p_pp);
  e.set_buffer(0, a.h_in);
  e.set_buffer(1, has_post ? a.h_out : s.dummy);
  e.set_buffer(2, or_dummy(a.x_sub, s));
  e.set_buffer(3, a.mix_prev);
  e.set_buffer(4, or_dummy(a.norm_w, s));
  e.set_buffer(5, or_dummy(a.x_out, s));
  e.set_buffer(6, s.rstd);
  e.set_value(7, P);
  e.dispatch_groups(a.n, 1, 1, 512);
  if (!has_mixes) return;
  e.barrier();
  // per-token rows kernel (n < 8 or forced), 8-token MMA (8 <= n < 32 or forced), 32-token MMA
  const int kind = hc_prefill_rows() || a.n < 8 ? 0 : (a.n < 32 || options().hc_prefill_mma8) ? 1 : 2;
  e.set_pipeline(kind == 0 ? p_rows : kind == 1 ? p_mma : p_mma32);
  e.set_buffer(0, h_new);
  e.set_buffer(1, a.fn);
  e.set_buffer(2, a.scale);
  e.set_buffer(3, a.base);
  e.set_buffer(4, s.rstd);
  e.set_buffer(5, a.mix_out);
  e.set_value(6, P);
  if (kind == 2) e.dispatch_groups((a.n + 31) / 32, 1, 1, 256);
  else if (kind == 1) e.dispatch_groups((a.n + 7) / 8, 1, 1, 128);
  else e.dispatch_groups(a.n, 1, 1, 256);
}

void encode_hc_mean(Enc& e, Buffer h, Buffer out, int n) {
  ARCH_PSO(p, "arch_hc_mean");
  uint32_t nn = n;
  e.set_pipeline(p);
  e.set_buffer(0, h);
  e.set_buffer(1, out);
  e.set_value(2, nn);
  e.dispatch_threads(n * kDim, 1, 1, 256);
}

void encode_router_fused(Enc& e, Buffer x, Buffer W, Buffer bias, Buffer ids, Buffer w, const Scratch& s) {
  ARCH_PSO(p, "arch_router_fused");
  const uint32_t G = router_groups();
  RouterParams P{1, G, kRouteScale, 0};
  e.set_pipeline(p);
  e.set_buffer(0, x);
  e.set_buffer(1, W);
  e.set_buffer(2, bias);
  e.set_buffer(3, s.router_logits);
  e.set_buffer(4, ids);
  e.set_buffer(5, w);
  e.set_buffer(6, counter(s, kCounterRouter));
  e.set_value(7, P);
  uint32_t rows = kExperts / G;
  uint32_t T = std::min<uint32_t>(256, std::max<uint32_t>(32, rows * 32));
  e.dispatch_groups(G, 1, 1, T);
}

void encode_router_topk(Enc& e, Buffer logits, Buffer bias, Buffer ids, Buffer w, int n) {
  ARCH_PSO(p, "arch_router_topk");
  RouterParams P{(uint32_t)n, 0, kRouteScale, 0};
  e.set_pipeline(p);
  e.set_buffer(0, logits);
  e.set_buffer(1, bias);
  e.set_buffer(2, ids);
  e.set_buffer(3, w);
  e.set_value(4, P);
  e.dispatch_groups((n + 3) / 4, 1, 1, 128);
}

RouteLists RouteLists::create(int max_tokens) {
  auto& d = Device::get();
  RouteLists L;
  L.max_tokens = max_tokens;
  size_t ne = (size_t)max_tokens * kTopK;
  L.counts = d.alloc(kExperts * 4, "arch.route.counts");
  L.offsets = d.alloc((kExperts + 1) * 4, "arch.route.offsets");
  L.entry = d.alloc(ne * 4, "arch.route.entry");
  L.tok = d.alloc(ne * 4, "arch.route.tok");
  L.w = d.alloc(ne * 4, "arch.route.w");
  L.slot_of = d.alloc(ne * 4, "arch.route.slot_of");
  return L;
}

void encode_route_lists(Enc& e, Buffer ids, Buffer w, int n, const RouteLists& L) {
  ARCH_PSO(p_count, "arch_route_count");
  ARCH_PSO(p_scatter, "arch_route_scatter");
  if (n > L.max_tokens) { fprintf(stderr, "arch: route_lists n=%d > max %d\n", n, L.max_tokens); abort(); }
  RouterParams P{(uint32_t)n, 0, kRouteScale, 0};
  e.set_pipeline(p_count);
  e.set_buffer(0, ids);
  e.set_buffer(1, L.counts);
  e.set_value(2, P);
  e.dispatch_groups(kExperts / 8, 1, 1, 256);
  e.barrier();
  e.set_pipeline(p_scatter);
  e.set_buffer(0, ids);
  e.set_buffer(1, w);
  e.set_buffer(2, L.counts);
  e.set_buffer(3, L.offsets);
  e.set_buffer(4, L.entry);
  e.set_buffer(5, L.tok);
  e.set_buffer(6, L.w);
  e.set_buffer(7, L.slot_of);
  e.set_value(8, P);
  e.dispatch_groups(kExperts / 8, 1, 1, 256);
}

void encode_engram_gate(Enc& e, Buffer h, Buffer kv, Buffer w, int n, const Scratch& s, Buffer gate_out) {
  ARCH_PSO(p, "arch_engram_gate");
  if (!gate_out && n > s.max_tokens) { fprintf(stderr, "arch: engram n=%d > scratch\n", n); abort(); }
  EngramParams P{(uint32_t)n, kNormEps, kEngramClamp, 0};
  e.set_pipeline(p);
  e.set_buffer(0, h);
  e.set_buffer(1, kv);
  e.set_buffer(2, w);
  e.set_buffer(3, gate_out ? gate_out : s.gate);
  e.set_value(4, P);
  e.dispatch_groups(n, 1, 1, 1024);
}

void encode_iq4nl_rows(Enc& e, Buffer src, uint32_t row_stride, Buffer dst, int n_rows) {
  ARCH_PSO(p, "arch_iq4nl_rows");
  uint32_t v[2] = {(uint32_t)n_rows, row_stride};
  e.set_pipeline(p);
  e.set_buffer(0, src);
  e.set_buffer(1, dst);
  e.set_bytes(2, v, sizeof(v));
  e.dispatch_threads(n_rows * 128, 1, 1, 128);
}

void encode_embed_q6k(Enc& e, Buffer rows, uint32_t row_stride, Buffer h, int n, int copies) {
  ARCH_PSO(p, "arch_embed_q6k");
  EmbedParams P{(uint32_t)n, row_stride, (uint32_t)copies, (uint32_t)kDim};
  e.set_pipeline(p);
  e.set_buffer(0, rows);
  e.set_buffer(1, h);
  e.set_value(2, P);
  e.dispatch_threads(n * (kDim / 256) * 64, 1, 1, 256);
}

void encode_argmax(Enc& e, Buffer logits, int rows, int V, Buffer out_idx, Buffer out_val, const Scratch& s,
                   float temperature, uint64_t seed) {
  ARCH_PSO(p, "arch_argmax");
  if (rows > s.max_rows) { fprintf(stderr, "arch: argmax rows=%d > scratch max_rows %d\n", rows, s.max_rows); abort(); }
  uint32_t G = kArgmaxGroups, chunk = (V + G - 1) / G;
  SampleParams P{(uint32_t)V, G, chunk, temperature > 0 ? 1.0f / temperature : 1.0f, (uint32_t)seed,
                 (uint32_t)(seed >> 32), temperature > 0 ? 1u : 0u, 0};
  e.set_pipeline(p);
  e.set_buffer(0, logits);
  e.set_buffer(1, s.sample_part);
  e.set_buffer(2, out_idx);
  e.set_buffer(3, out_val);
  e.set_buffer(4, s.counters.sub(kCounterRows * 4, (size_t)s.max_rows * 4));
  e.set_value(5, P);
  e.dispatch_groups(G, rows, 1, kSampleT);
}

void encode_topk(Enc& e, Buffer logits, int rows, int V, int k, float temperature, Buffer out_v, Buffer out_i,
                 Buffer out_lse, const Scratch& s) {
  ARCH_PSO(p, "arch_topk");
  if (rows > s.max_rows || k < 1 || k > kSampleK) { fprintf(stderr, "arch: topk rows=%d k=%d invalid\n", rows, k); abort(); }
  uint32_t G = kTopkGroups, chunk = (V + G - 1) / G;
  if (chunk > 4 * kSampleT) { fprintf(stderr, "arch: topk V=%d too large\n", V); abort(); }
  SampleParams P{(uint32_t)V, G, chunk, temperature > 0 ? 1.0f / temperature : 1.0f, 0, 0, 0, (uint32_t)k};
  size_t nv = (size_t)s.max_rows * G * 64 * 4;
  e.set_pipeline(p);
  e.set_buffer(0, logits);
  e.set_buffer(1, s.sample_part.sub(0, nv));
  e.set_buffer(2, s.sample_part.sub(nv, nv));
  e.set_buffer(3, s.sample_part.sub(2 * nv, (size_t)s.max_rows * G * 8));
  e.set_buffer(4, out_v);
  e.set_buffer(5, out_i);
  e.set_buffer(6, out_lse);
  e.set_buffer(7, s.counters.sub((kCounterRows + s.max_rows) * 4, (size_t)s.max_rows * 4));
  e.set_value(8, P);
  e.dispatch_groups(G, rows, 1, kSampleT);
}

void encode_log_softmax(Enc& e, Buffer in, Buffer out, int rows, int V) {
  ARCH_PSO(p, "arch_log_softmax");
  uint32_t v = V;
  e.set_pipeline(p);
  e.set_buffer(0, in);
  e.set_buffer(1, out);
  e.set_value(2, v);
  e.dispatch_groups(rows, 1, 1, 1024);
}

uint32_t sample_top_p(const float* vals, const uint32_t* ids, int k, float lse, float temperature, float top_p,
                      double u) {
  if (temperature <= 0.0f || k <= 1) return ids[0];
  const double inv_t = 1.0 / temperature;
  double probs[kSampleK];
  double cum = 0.0;
  int keep = k;
  for (int i = 0; i < k; i++) {
    probs[i] = std::exp((double)vals[i] * inv_t - (double)lse);
    cum += probs[i];
    if (cum >= top_p) { keep = i + 1; break; }
  }
  double tot = 0.0;
  for (int i = 0; i < keep; i++) tot += probs[i];
  double target = u * tot, acc = 0.0;
  for (int i = 0; i < keep; i++) {
    acc += probs[i];
    if (target < acc) return ids[i];
  }
  return ids[keep - 1];
}

void encode_rmsnorm(Enc& e, Buffer x, Buffer w, Buffer y, int rows, int dim, float eps, int in_stride,
                    int out_stride) {
  ARCH_PSO(p_tg, "arch_rmsnorm");
  ARCH_PSO(p_sg, "arch_rmsnorm_sg");
  NormParams P{(uint32_t)rows, (uint32_t)dim, eps, w ? 1u : 0u, (uint32_t)(in_stride ? in_stride : dim),
               (uint32_t)(out_stride ? out_stride : dim)};
  // one simdgroup per row only pays off with many rows (or tiny rows); a single row uses the whole threadgroup
  bool small = dim <= 128 || (dim <= 1024 && rows >= 16);
  e.set_pipeline(small ? p_sg : p_tg);
  e.set_buffer(0, x);
  e.set_buffer(1, w ? w : x);
  e.set_buffer(2, y);
  e.set_value(3, P);
  if (small) e.dispatch_groups((rows + 7) / 8, 1, 1, 256);
  else e.dispatch_groups(rows, 1, 1, std::min(1024, std::max(32, (dim / 4 + 31) / 32 * 32)));
}

}  // namespace ms::arch
