// DeepSeek-V4.1-Flash forward pass. See ds41.hpp, SPEC.md §3, docs/integration.md.
#import <Metal/Metal.h>
#include "model/ds41.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unistd.h>

#include "core/env.hpp"
#include "core/timer.hpp"
#include "gguf/gguf.hpp"
#include "kernels/arch.hpp"
#include "kernels/attn.hpp"
#include "kernels/quant.hpp"
#include "model/engram_hash.hpp"
#include "store/engram.hpp"
#include "store/experts.hpp"
#include "store/memguard.hpp"
#include "store/weights.hpp"

namespace ms::ds41 {

using q::QType;

ModelOptions ModelOptions::from_env() {
  ModelOptions o;
  o.model_path = env_str("MACSEEK_MODEL", "");
  o.arena_gb = env_f("MACSEEK_ARENA_GB", 0);
  o.max_chunk = (int)env_int("MACSEEK_CHUNK", 4096);
  o.actq_out = env_int("MACSEEK_ACTQ_OUT", env_int("MACSEEK_ACTQ", 0)) != 0;
  o.moe_group = (int)env_int("MACSEEK_MOE_GROUP", 64);
  o.verbose = env_int("MACSEEK_VERBOSE", 1) != 0;
  return o;
}

namespace {
struct W {  // one weight: buffer + kernel type + shape
  Buffer b; QType t = QType::F32; int rows = 0, cols = 0;
  explicit operator bool() const { return (bool)b; }
};
struct LayerW {
  W attn_norm, ffn_norm, q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b, sinks;
  W c_kv, c_gate, c_norm, idx_qb, idx_proj, idx_k, idx_knorm;
  W gate_inp, bias;
  W sh_gate, sh_up, sh_down;
  W hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
  W engram_kv, engram_gate;
};
}  // namespace

struct Model::Impl {
  std::vector<LayerW> L;
  W out_norm, head;
  std::vector<W> pred_w;             // Q8_0 copies of each layer's router (prediction only: never affects outputs)
  attn::State ast;
  arch::Scratch asc;
  int maxn = 0;
  // activations
  Buffer h, x, sub, mixA, mixF, qa, qrn, q, kv, ckv, cgate, iq, iw, aout, aout16, otmp, oa, rlog, rids, rw;
  Buffer embrows, erow, edeq, ekv, logits, amax_i, amax_v, moe_dec_scratch;
  Buffer moe_scr[2], route_dev[2];
  Buffer pids, pw;                   // (unused)
  Buffer plog;                       // decode: router logits of layers l+1..l+D applied to layer l's FFN input
  Buffer plog_n;
  Buffer tbl;                        // decode table mode: [40][6][3] device addresses of the routed experts                     // prefill: router logits of layer l+1 on layer l's FFN input, [n][384]
  arch::Scratch asc2;                // second scratch for the predictor router (same kernel family)   // prefill MoE scratch + routing tables (double-buffered)
};

static W wt(const Weights& w, const std::string& name) {
  const WTensor* t = w.find(name);
  if (!t) return W{};
  W r; r.b = t->buf; r.t = (QType)(int)t->type;
  r.cols = (int)t->ne[0]; r.rows = t->n_dims > 1 ? (int)t->ne[1] : 1;
  return r;
}
static std::string ln(int l, const char* s) { char b[128]; snprintf(b, sizeof b, "blk.%d.%s", l, s); return b; }

static void fill(Enc& e, const Buffer& y, size_t nfloats, float v = 0.f) {
  uint32_t n4 = (uint32_t)(nfloats / 4);
  e.set_pipeline("m_fill"); e.set_buffer(0, y); e.set_value(1, v); e.set_value(2, n4);
  e.dispatch_threads(n4, 1, 1, 256);
}

// y[n][rows] = x[n][cols] . W^T  (picks decode / small-n / prefill kernel)
static void linear(Enc& e, const W& w, const Buffer& x, const Buffer& y, int n, bool acc = false) {
  if (n == 1) q::encode_matvec(e, w.t, w.b, w.rows, w.cols, x, y, acc);
  else if (n <= 16) q::encode_matvec_n(e, w.t, w.b, w.rows, w.cols, x, y, n, acc);
  else q::encode_matmul(e, w.t, w.b, w.rows, w.cols, x, y, n, acc);
}

Model::~Model() = default;

Model::Model(const ModelOptions& o) : p_(new Impl), o_(o) {
  Timer t;
  auto& d = Device::get();
  g_ = std::make_unique<Gguf>(o.model_path.empty() ? Gguf::default_model_path() : o.model_path);
  w_ = std::make_unique<Weights>(*g_);
  te_ = std::make_unique<TokenEmbd>(*g_, w_.get());
  if (o.verbose) fprintf(stderr, "macseek: weights %.2f GiB in %.0f ms (%.1f GB/s)\n", w_->bytes() / 1073741824.0, w_->load_ms(), w_->gbps());

  auto& P = *p_;
  P.L.resize(cfg_.n_layers);
  for (int l = 0; l < cfg_.n_layers; l++) {
    LayerW& L = P.L[l];
    auto g = [&](const char* s) { return wt(*w_, ln(l, s)); };
    L.attn_norm = g("attn_norm.weight"); L.ffn_norm = g("ffn_norm.weight");
    L.q_a = g("attn_q_a.weight"); L.q_a_norm = g("attn_q_a_norm.weight"); L.q_b = g("attn_q_b.weight");
    L.kv = g("attn_kv.weight"); L.kv_norm = g("attn_kv_a_norm.weight");
    L.o_a = g("attn_output_a.weight"); L.o_b = g("attn_output_b.weight"); L.sinks = g("attn_sinks.weight");
    L.c_kv = g("attn_compressor_kv.weight"); L.c_gate = g("attn_compressor_gate.weight"); L.c_norm = g("attn_compressor_norm.weight");
    L.idx_qb = g("indexer.attn_q_b.weight"); L.idx_proj = g("indexer.proj.weight");
    L.idx_k = g("indexer.attn_k.weight"); L.idx_knorm = g("indexer.k_norm.weight");
    L.gate_inp = g("ffn_gate_inp.weight"); L.bias = g("exp_probs_b.bias");
    L.sh_gate = g("ffn_gate_shexp.weight"); L.sh_up = g("ffn_up_shexp.weight"); L.sh_down = g("ffn_down_shexp.weight");
    L.hc_attn_fn = g("hc_attn_fn.weight"); L.hc_attn_scale = g("hc_attn_scale.weight"); L.hc_attn_base = g("hc_attn_base.weight");
    L.hc_ffn_fn = g("hc_ffn_fn.weight"); L.hc_ffn_scale = g("hc_ffn_scale.weight"); L.hc_ffn_base = g("hc_ffn_base.weight");
    L.engram_kv = g("engram_kv.weight"); L.engram_gate = g("engram_gate.weight");
  }
  P.out_norm = wt(*w_, "output_norm.weight");
  P.head = wt(*w_, "output.weight");
  {  // Q8_0 router copies for next-layer prediction: 2.1 MB instead of 7.9 MB per layer read each decode token
    const int R = 384, C = 5120, nb = C / 32;
    const size_t rb = (size_t)nb * 34;
    Buffer all = d.alloc(rb * R * cfg_.n_layers, "pred_w");
    P.pred_w.resize(cfg_.n_layers);
    for (int l = 0; l < cfg_.n_layers; l++) {
      const float* src = (const float*)P.L[l].gate_inp.b.cpu();
      uint8_t* dst = (uint8_t*)all.cpu() + (size_t)l * rb * R;
      for (int r = 0; r < R; r++)
        for (int b = 0; b < nb; b++) {
          const float* x = src + (size_t)r * C + b * 32;
          float amax = 0; for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[i]));
          float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
          uint8_t* blk = dst + (size_t)r * rb + (size_t)b * 34;
          __fp16 h = (__fp16)dd; memcpy(blk, &h, 2);
          for (int i = 0; i < 32; i++) ((int8_t*)blk)[2 + i] = (int8_t)std::lround(x[i] * id);
        }
      W w; w.b = all.sub((size_t)l * rb * R, rb * R); w.t = QType::Q8_0; w.rows = R; w.cols = C;
      P.pred_w[l] = w;
    }
  }

  // engram hashing
  ehp_ = std::make_unique<EngramHashParams>();
  if (!ehp_->init(g_->get_arr_i64("deepseek41.engram.token_map"), g_->get_i64("deepseek41.engram.pad_token_id", 2),
                  g_->get_arr_i64("deepseek41.ple.layer_multipliers"), g_->get_arr_i64("deepseek41.ple.head_vocab_sizes"),
                  g_->get_arr_i64("deepseek41.ple.head_offsets"))) {
    fprintf(stderr, "macseek: engram metadata invalid\n"); abort();
  }
  hasher_ = std::make_unique<EngramHasher>(ehp_.get());
  ef_ = std::make_unique<EngramFetcher>(*g_);

  // attention state
  attn::Options ao = attn::Options::from_env();
  ao.max_chunk = o.max_chunk;
  o_.max_chunk = ao.max_chunk;
  attn::init(P.ast, ao);
  P.maxn = o_.max_chunk;
  P.asc = arch::Scratch::create(std::max(P.maxn, 64), 16);

  const size_t n = P.maxn;
  auto A = [&](size_t bytes, const char* lbl) { return d.alloc(bytes, lbl); };
  P.h = A(n * 4 * 5120 * 4, "h"); P.x = A(n * 5120 * 4, "x"); P.sub = A(n * 5120 * 4, "sub");
  P.mixA = A(n * 24 * 4, "mixA"); P.mixF = A(n * 24 * 4, "mixF");
  P.qa = A(n * 1280 * 4, "qa"); P.qrn = A(n * 1280 * 4, "qrn"); P.q = A(n * 32768 * 4, "q");
  P.kv = A(n * 512 * 4, "kv"); P.ckv = A(n * 512 * 4, "ckv"); P.cgate = A(n * 512 * 4, "cgate");
  P.iq = A(n * 4096 * 4, "iq"); P.iw = A(n * 32 * 4, "iw");
  P.aout = A(n * 32768 * 4, "aout"); P.aout16 = A(n * 32768 * 2, "aout16"); P.otmp = A(n * 8192 * 4, "otmp");
  P.oa = A(n * 8192 * 4, "oa"); P.rlog = A(n * 384 * 4, "rlog"); P.rids = A(n * 6 * 4, "rids"); P.rw = A(n * 6 * 4, "rw");
  P.embrows = A(n * 4200 + 16384, "embrows"); P.erow = A(2 * n * 24 * 144 + 16384, "erow");
  P.edeq = A(n * 6144 * 4, "edeq"); P.ekv = A(2 * n * 25600 * 4, "ekv");
  P.logits = A((size_t)std::max(o.max_logits_rows, 1) * cfg_.vocab * 4, "logits");
  P.amax_i = A(64, "amax_i"); P.amax_v = A(64, "amax_v");
  P.pids = A(64, "pids"); P.pw = A(64, "pw"); P.plog = A(4 * 384 * 4, "plog"); P.plog_n = A(n * 384 * 4, "plog_n"); P.tbl = A(40 * 6 * 3 * 8, "tbl");
  P.asc2 = arch::Scratch::create(64, 1);
  P.moe_dec_scratch = A(q::moe_decode_scratch_bytes(), "moe_dec");
  const int total = (int)n * 6;
  for (int i = 0; i < 2; i++) {
    P.moe_scr[i] = A(q::moe_prefill_scratch_bytes(std::min(total, 8192), (int)n, o.moe_group), "moe_pf");  // grows on demand
    P.route_dev[i] = A((size_t)total * 4 * 3 + 4096, "route");
  }

  // expert cache last: it takes what is left
  ExpertLayout layout = ExpertLayout::from_gguf(*g_);
  ExpertCacheOptions co = ExpertCacheOptions::from_env();
  // memory freed by a process that just exited takes a few seconds to show up as available: wait until it settles
  MemStatus ms0 = mem_status();
  for (int i = 0; i < 40; i++) {
    usleep(250000);
    MemStatus m1 = mem_status();
    bool settled = m1.available < ms0.available + (256ull << 20);
    ms0 = m1;
    if (settled && i >= 3) break;
  }
  uint64_t planned = 2ull << 30;  // prefill transients + slack
  uint64_t arena = o.arena_gb > 0 ? (uint64_t)(o.arena_gb * 1073741824.0) : recommend_arena_bytes(planned, (uint64_t)(env_f("MACSEEK_HEADROOM_GB", 10) * 1073741824.0), 40ull << 30);
  co.arena_bytes = arena;
  cache_ = std::make_unique<ExpertCache>(layout, co);
  if (o.verbose) {
    fprintf(stderr, "macseek: %s\n", ms0.str().c_str());
    fprintf(stderr, "macseek: %s\n", cache_->describe().c_str());
    fprintf(stderr, "macseek: arena %.1f GiB; after alloc: %s\n", arena / 1073741824.0, mem_status().str().c_str());
    fprintf(stderr, "macseek: model ready in %.1f s\n", t.ms() / 1000);
  }
}

void Model::reset() {
  attn::reset(p_->ast);
  hasher_->reset();
  pos_ = 0;
}

int32_t Model::prefill(const int32_t* toks, int n, float* logits_all, float* logits_last) {
  int32_t next = -1;
  for (int i = 0; i < n; i += o_.max_chunk) {
    int m = std::min(o_.max_chunk, n - i);
    bool last = i + m == n;
    next = forward(toks + i, m, logits_all ? logits_all + (size_t)i * cfg_.vocab : nullptr, last ? logits_last : nullptr, 0.f, 0);
  }
  return next;
}

int32_t Model::decode(int32_t tok, float* logits, float temperature, uint64_t seed) {
  return forward(&tok, 1, nullptr, logits, temperature, seed);
}

// ------------------------------------------------------------------------------------------------------------------
int32_t Model::forward(const int32_t* toks, int n, float* logits_all, float* logits_last, float temperature, uint64_t seed) {
  auto& P = *p_;
  Timer tt;
  const int start = pos_;
  const int V = cfg_.vocab;
  if (logits_all && n > o_.max_logits_rows) { fprintf(stderr, "macseek: logits_all needs n <= %d\n", o_.max_logits_rows); abort(); }

  // ---- CPU: engram hashes + rows (async), token embedding rows ----
  Timer te;
  std::vector<uint32_t> hashes((size_t)n * 48);
  hasher_->hash(toks, n, hashes.data());
  std::vector<uint64_t> rows0((size_t)n * 24), rows1((size_t)n * 24);
  for (int t = 0; t < n; t++)
    for (int c = 0; c < 24; c++) { rows0[t * 24 + c] = hashes[t * 48 + c]; rows1[t * 24 + c] = hashes[t * 48 + 24 + c]; }
  char* erow = (char*)P.erow.cpu();
  auto tk0 = ef_->fetch(rows0.data(), rows0.size(), erow);
  auto tk1 = ef_->fetch(rows1.data(), rows1.size(), erow + (size_t)n * 24 * 144);
  te_->fetch(toks, n, P.embrows.cpu());
  tk0.wait(); tk1.wait();
  double engram_ms = te.ms();

  const bool dec = n == 1;
  Buffer h = P.h.sub(0, (size_t)n * 20480 * 4);
  double wait_ms = 0, exp_ms = 0, gpu_ms = 0, moe_gpu_ms = 0, route_ms = 0;
  long experts_used = 0, groups_run = 0;
  const LayerStats st0 = cache_->total_stats();

  auto e = std::make_unique<Enc>();
  arch::encode_embed_q6k(*e, P.embrows, 4200, h, n, 4);
  // layer-0 attention mix_prev = identity (pre = [1,0,0,0])
  {
    float* mf = (float*)P.mixF.cpu();
    for (int t = 0; t < n; t++) arch::fill_identity_mix(mf + t * 24);
  }
  // engram kv for both engram layers (depends only on token ids): ekv[s] = engram_kv(dequant(rows of layer s))
  Buffer ekv_l[2];
  for (int s = 0; s < 2; s++) {
    const int l = cfg_.engram_layers[s];
    ekv_l[s] = P.ekv.sub((size_t)s * P.maxn * 25600 * 4, (size_t)n * 25600 * 4);
    arch::encode_iq4nl_rows(*e, P.erow.sub((size_t)s * n * 24 * 144, (size_t)n * 24 * 144), 144, P.edeq, n * 24);
    e->barrier();
    linear(*e, P.L[l].engram_kv, P.edeq, ekv_l[s], n);
    e->barrier();
  }

  static const int pf_prefetch_min = (int)env_int("MACSEEK_PF_PREFETCH", 1) ? (int)env_int("MACSEEK_PF_PREFETCH_MIN", 1024) : 1 << 30;
  // Prefill waits for expert reads on the CPU: a GPU command buffer blocked on an I/O event for long trips the GPU
  // watchdog ("GPU Timeout Error"). MACSEEK_PF_GPU_WAIT=1 restores GPU-side waits. MACSEEK_CPU_WAIT_IO=1 also makes
  // decode wait on the CPU (profiling: GPU times are only valid for command buffers without event waits).
  static const bool cpu_wait_io = env_int("MACSEEK_PF_GPU_WAIT", 0) == 0;
  static const bool dec_cpu_wait = env_int("MACSEEK_CPU_WAIT_IO", 0) != 0;
  // prefill: prefetch only the union of each token's predicted top-k (layer l+1's router on layer l's FFN input)
  // instead of all 384 experts of layer l+1. MACSEEK_PF_PRED=0: prefetch everything (previous behaviour).
  static const bool pf_pred = env_int("MACSEEK_PF_PRED", 0) != 0;
  static const int pf_pred_k = (int)env_int("MACSEEK_PF_PRED_K", 8);
  static const bool pred_on = env_int("MACSEEK_PRED", 1) != 0;   // decode next-layer expert prefetch (win: +9%)
  static const int pred_k = (int)env_int("MACSEEK_PRED_K", 6);      // experts prefetched per predicted layer
  static const int pred_depth = std::min(4, (int)env_int("MACSEEK_PRED_DEPTH", 1));
  auto prefetch_predicted = [&](int l) {
    for (int d = 1; d <= pred_depth && l + d < cfg_.n_layers; d++) {
      const float* lg = (const float*)P.plog.cpu() + (d - 1) * 384;
      const float* bias = (const float*)P.L[l + d].bias.b.cpu();
      std::pair<float, int> sc[384];
      for (int x = 0; x < 384; x++) {
        float v = lg[x];
        float sp = v > 20.f ? v : std::log1p(std::exp(v));
        sc[x] = {std::sqrt(sp) + bias[x], x};
      }
      const int k = std::min(pred_k, 384);
      std::partial_sort(sc, sc + k, sc + 384, [](auto& a, auto& b) { return a.first > b.first; });
      int ids[384];
      for (int i = 0; i < k; i++) ids[i] = sc[i].second;
      cache_->prefetch(l + d, ids, k);
    }
  };
  static std::vector<int> all_ids = [] { std::vector<int> v(384); for (int i = 0; i < 384; i++) v[i] = i; return v; }();
  // Attention side of layer l for this call: hc boundary (+engram), attention, hc boundary into the FFN, router,
  // and zeroing of `sub` (the MoE accumulates into it). Encodes into E; no commit.
  auto encode_attn_side = [&](Enc& E, int l) {
    LayerW& L = P.L[l];
    // ---- hyper-connection boundary into attention (+ engram) ----
    if (cfg_.is_engram(l)) {
      arch::HcStep a; a.h_in = h; a.h_out = h; a.x_sub = P.sub; a.mix_prev = P.mixF; a.n = n;
      arch::encode_hc_step(E, a, P.asc); E.barrier();
      Buffer kvb = l == cfg_.engram_layers[0] ? ekv_l[0] : ekv_l[1];
      arch::encode_engram_gate(E, h, kvb, L.engram_gate.b, n, P.asc); E.barrier();
      arch::HcStep c; c.h_in = h; c.h_out = h; c.mix_prev = P.mixF; c.fn = L.hc_attn_fn.b; c.scale = L.hc_attn_scale.b;
      c.base = L.hc_attn_base.b; c.mix_out = P.mixA; c.norm_w = L.attn_norm.b; c.x_out = P.x; c.n = n;
      arch::encode_hc_step(E, c, P.asc); E.barrier();
    } else {
      arch::HcStep a; a.h_in = h; a.h_out = h; if (l > 0) a.x_sub = P.sub; a.mix_prev = P.mixF;
      a.fn = L.hc_attn_fn.b; a.scale = L.hc_attn_scale.b; a.base = L.hc_attn_base.b; a.mix_out = P.mixA;
      a.norm_w = L.attn_norm.b; a.x_out = P.x; a.n = n;
      arch::encode_hc_step(E, a, P.asc); E.barrier();
    }
    // ---- attention projections ----
    linear(E, L.q_a, P.x, P.qa, n);
    linear(E, L.kv, P.x, P.kv, n);
    const bool kvsrc = cfg_.is_kv_source(l), idxsrc = cfg_.is_index_source(l);
    if (kvsrc) { linear(E, L.c_kv, P.x, P.ckv, n); if (L.c_gate) linear(E, L.c_gate, P.x, P.cgate, n); }
    if (idxsrc) linear(E, L.idx_proj, P.x, P.iw, n);
    E.barrier();
    arch::encode_rmsnorm(E, P.qa, L.q_a_norm.b, P.qrn, n, 1280, cfg_.norm_eps); E.barrier();
    linear(E, L.q_b, P.qrn, P.q, n);
    if (idxsrc) linear(E, L.idx_qb, P.qrn, P.iq, n);
    E.barrier();
    attn::LayerIO io; io.layer = l; io.q = P.q; io.kv = P.kv; io.kv_norm = L.kv_norm.b; io.sinks = L.sinks.b; io.out = P.aout;
    if (kvsrc) { io.c_kv = P.ckv; io.c_gate = L.c_gate ? P.cgate : Buffer{}; io.c_norm = L.c_norm.b; io.idx_wk = L.idx_k.b; io.idx_knorm = L.idx_knorm.b; }
    if (idxsrc) { io.idx_q = P.iq; io.idx_w = P.iw; }
    attn::encode_layer(E, P.ast, io, start, n);
    if (o_.actq_out) { attn::encode_quant_sim(E, P.aout, (uint32_t)(n * 32768), 0); E.barrier(); }
    // wo_a (block-diagonal, 8 groups) then wo_b
    if (n <= 16) {
      if (n == 1) q::encode_matvec(E, L.o_a.t, L.o_a.b, 8192, 4096, P.aout, P.oa, false, 8);
      else q::encode_matvec_n(E, L.o_a.t, L.o_a.b, 8192, 4096, P.aout, P.oa, n, false, 8);
      E.barrier();
    } else {
      q::encode_f32_to_f16(E, P.aout, P.aout16, (size_t)n * 32768); E.barrier();
      const size_t rb = (size_t)L.o_a.b.size / 8192;  // bytes per weight row
      for (int g = 0; g < 8; g++)
        q::encode_matmul_h(E, L.o_a.t, L.o_a.b.sub((size_t)g * 1024 * rb, 1024 * rb), 1024, 4096,
                           P.aout16.sub((size_t)g * 4096 * 2, (size_t)n * 32768 * 2 - (size_t)g * 4096 * 2),
                           P.otmp.sub((size_t)g * n * 1024 * 4, (size_t)n * 1024 * 4), n, false, 32768);
      E.barrier();
      uint32_t a4[4] = {(uint32_t)n, 8, 1024 / 4, 0};
      e->set_pipeline("m_ungroup"); e->set_buffer(0, P.otmp); e->set_buffer(1, P.oa); e->set_bytes(2, a4, sizeof a4);
      e->dispatch_threads(1024 / 4, n, 8, 64, 1, 1);
      E.barrier();
    }
    linear(E, L.o_b, P.oa, P.sub, n); E.barrier();
    // ---- hyper-connection boundary into the FFN ----
    {
      arch::HcStep a; a.h_in = h; a.h_out = h; a.x_sub = P.sub; a.mix_prev = P.mixA;
      a.fn = L.hc_ffn_fn.b; a.scale = L.hc_ffn_scale.b; a.base = L.hc_ffn_base.b; a.mix_out = P.mixF;
      a.norm_w = L.ffn_norm.b; a.x_out = P.x; a.n = n;
      arch::encode_hc_step(E, a, P.asc); E.barrier();
    }
    // ---- router ----
    if (dec) {
      arch::encode_router_fused(E, P.x, L.gate_inp.b, L.bias.b, P.rids, P.rw, P.asc);
      // next-layer expert prediction: layer l+1's router applied to this layer's (normed) FFN input
      // (logits only; the CPU picks the top pred_k by score+bias after the router handoff)
      if (pred_on)
        for (int d = 1; d <= pred_depth && l + d < cfg_.n_layers; d++) {
          static const bool pred_q8 = env_int("MACSEEK_PRED_Q8", 1) != 0;
          const W& g = pred_q8 ? P.pred_w[l + d] : P.L[l + d].gate_inp;
          q::encode_matvec(E, g.t, g.b, g.rows, g.cols, P.x, P.plog.sub((size_t)(d - 1) * 384 * 4, 384 * 4));
        }
    }
    else {
      linear(E, L.gate_inp, P.x, P.rlog, n);
      if (pf_pred && n >= pf_prefetch_min && l + 1 < cfg_.n_layers) linear(E, P.L[l + 1].gate_inp, P.x, P.plog_n, n);
      E.barrier();
      arch::encode_router_topk(E, P.rlog, L.bias.b, P.rids, P.rw, n);
    }
    E.barrier();
    fill(E, P.sub, (size_t)n * 5120);  // MoE accumulates into sub (sub was consumed by the hc step above)
  };

  const bool pipe = dec && env_int("MACSEEK_DEC_PIPE", 0) != 0;
  auto encode_head = [&](Enc& E) {
    arch::HcStep a; a.h_in = h; a.h_out = h; a.x_sub = P.sub; a.mix_prev = P.mixF; a.norm_w = P.out_norm.b; a.x_out = P.x; a.n = n;
    arch::encode_hc_step(E, a, P.asc); E.barrier();
    if (logits_all) {
      linear(E, P.head, P.x, P.logits, n); E.barrier();
      arch::encode_argmax(E, P.logits.sub((size_t)(n - 1) * V * 4, (size_t)V * 4), 1, V, P.amax_i, P.amax_v, P.asc, temperature, seed);
    } else {
      q::encode_matvec(E, P.head.t, P.head.b, P.head.rows, P.head.cols, P.x.sub((size_t)(n - 1) * 5120 * 4, 5120 * 4), P.logits, false);
      E.barrier();
      arch::encode_argmax(E, P.logits, 1, V, P.amax_i, P.amax_v, P.asc, temperature, seed);
    }
  };  bool head_done = false;
  static const bool tbl_mode = env_int("MACSEEK_DEC_TABLE", 1) != 0;
  if (dec && tbl_mode && env_int("MACSEEK_PROF_FORCE", 0) != 2) {
    static const bool tforce = env_int("MACSEEK_PROF_FORCE", 0) == 1;   // PROFILING ONLY
    static const int tforced[6] = {0, 1, 2, 3, 4, 5};
    // Table decode: the whole token is ONE command buffer encoded up front. Per layer the GPU signals `rdy` after the
    // router and waits on `go`; the CPU reads the ids, ensures the experts, writes their device addresses into
    // P.tbl[l] and signals `go`. No command-buffer round trip per layer.
    id<MTLDevice> dev = (__bridge id<MTLDevice>)Device::get().device();
    static id<MTLSharedEvent> rdy = [dev newSharedEvent];
    static id<MTLSharedEvent> go = [dev newSharedEvent];
    static uint64_t tb = 0;
    const uint64_t b0 = tb;
    tb += cfg_.n_layers;
    const std::vector<Buffer> arena = cache_->buffers();
    // MACSEEK_DEC_SYNC (docs/modules/dec_sync.md): how the GPU hides the per-layer router handshake.
    //   event  (0, default): signal rdy; wait go; whole MoE. The GPU idles for the full round trip (~X 15 us GPU->CPU
    //          + CPU + Y 30-60 us CPU->GPU wake-up; bench/sync_probe).
    //   sgu    (1): the shared expert's gate/up runs between the rdy signal and the go wait (it does not depend on the
    //          routing), so the wait is usually already satisfied when the GPU reaches it. Bit-identical to event.
    //   shared (2): the whole shared expert (gate/up + down) runs in the gap: a bigger cover, but y = shared + routed is
    //          summed in a different order (not bit-identical; same math).
    // In sgu/shared the CPU also answers all-hit layers before the next-layer prefetch (off the critical path).
    // Gap work is ordered after the attention side and before the routed MoE by MTLFences: consecutive compute
    // encoders over untracked buffers are not ordered by themselves, and the go wait only orders work after it.
    static const int dsync = [] {
      const std::string v = env_str("MACSEEK_DEC_SYNC", "shared");
      if (v == "1" || v == "sgu") return 1;
      if (v == "2" || v == "shared") return 2;
      return 0;
    }();
    static id<MTLFence> f_att = [dev newFence];
    static id<MTLFence> f_gap = [dev newFence];
    auto ce_of = [](Enc& E) { return (__bridge id<MTLComputeCommandEncoder>)E.encoder(); };
    Timer tenc;
    for (int l = 0; l < cfg_.n_layers; l++) {
      LayerW& L = P.L[l];
      encode_attn_side(*e, l);
      const auto& lay = cache_->layout().layers[l];
      q::MoeDecodeArgs m;
      m.t_gate_up = (QType)(int)lay.mat[0].type; m.t_down = (QType)(int)lay.mat[2].type;
      m.n_routed = 6; m.weights = P.rw; m.has_shared = true; m.t_shared = L.sh_gate.t;
      m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
      m.x = P.x; m.y = P.sub; m.scratch = P.moe_dec_scratch;
      const Buffer tl = P.tbl.sub((size_t)l * 6 * 3 * 8, 6 * 3 * 8);
      if (dsync == 0) {
        e->between([&](void* cb) {
          id<MTLCommandBuffer> c = (__bridge id<MTLCommandBuffer>)cb;
          [c encodeSignalEvent:rdy value:b0 + 1 + l];
          [c encodeWaitForEvent:go value:b0 + 1 + l];
        });
        e->use_resources(arena);
        q::encode_moe_decode_table(*e, m, tl);
      } else {
        [ce_of(*e) updateFence:f_att];
        e->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:rdy value:b0 + 1 + l]; });
        [ce_of(*e) waitForFence:f_att];
        q::encode_moe_decode_table_part(*e, m, tl, dsync == 1 ? q::MoePart::SharedGu : q::MoePart::Shared);
        [ce_of(*e) updateFence:f_gap];
        e->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:go value:b0 + 1 + l]; });
        [ce_of(*e) waitForFence:f_gap];
        e->use_resources(arena);
        q::encode_moe_decode_table_part(*e, m, tl, dsync == 1 ? q::MoePart::AfterSharedGu : q::MoePart::Routed);
      }
      e->barrier();
    }
    encode_head(*e);
    head_done = true;
    prof_.encode_ms += tenc.ms();
    e->commit();
    std::vector<ExpertLease> leases;
    leases.reserve(cfg_.n_layers);
    for (int l = 0; l < cfg_.n_layers; l++) {
      Timer tw;
      while (rdy.signaledValue < b0 + 1 + l) { }
      wait_ms += tw.ms();
      Timer tx;
      ExpertLease lease = cache_->ensure(l, tforce ? tforced : (const int*)P.rids.cpu(), 6);
      const bool pf = pred_on && !tforce;
      uint64_t* t = (uint64_t*)P.tbl.cpu() + (size_t)l * 18;
      for (int k = 0; k < 6; k++) for (int j = 0; j < 3; j++) t[k * 3 + j] = lease.views[k].gpu[j];
      if (dsync != 0 && !lease.ticket) {
        // all hits: release the GPU first, then predict + prefetch the next layer
        go.signaledValue = b0 + 1 + l;
        if (pf) prefetch_predicted(l);
      } else {
        if (pf) prefetch_predicted(l);   // misses: issue the prefetch before blocking on the demand reads
        if (lease.ticket) lease.wait();
        go.signaledValue = b0 + 1 + l;
      }
      leases.push_back(std::move(lease));
      exp_ms += tx.ms();
    }
  } else
  if (pipe) {
    // Pipelined decode (docs/integration.md): GPU-ordered chain att(0) moe(0) att(1) ... ; att(l+1) is encoded and
    // committed while att(l) runs; moe(l) is reserved in the queue before att(l+1) and encoded once the router ids are
    // known. Ordering via an MTLEvent (embed signals c0+1, att(l) c0+2+2l, moe(l) c0+3+2l); router readiness via an
    // MTLSharedEvent the CPU spins on.
    id<MTLDevice> dev = (__bridge id<MTLDevice>)Device::get().device();
    static id<MTLEvent> chain = [dev newEvent];
    static id<MTLSharedEvent> rdy = [dev newSharedEvent];
    static uint64_t base = 0, rbase = 0;
    const uint64_t c0 = base, r0 = rbase;
    e->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:chain value:c0 + 1]; });
    e->commit();
    std::vector<std::unique_ptr<Enc>> keep;
    keep.push_back(std::move(e));
    auto make_att = [&](int l) {
      auto E = std::make_unique<Enc>(false);
      E->enqueue();
      const uint64_t wv = c0 + 1 + 2 * (uint64_t)l, sv = wv + 1;
      E->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:chain value:wv]; });
      Timer tenc;
      encode_attn_side(*E, l);
      prof_.encode_ms += tenc.ms();
      E->between([&](void* cb) {
        [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:chain value:sv];
        [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:rdy value:r0 + 1 + (uint64_t)l];
      });
      E->commit();
      return E;
    };
    std::unique_ptr<Enc> att = make_att(0);
    for (int l = 0; l < cfg_.n_layers; l++) {
      LayerW& L = P.L[l];
      auto moe = std::make_unique<Enc>(false);
      moe->enqueue();
      std::unique_ptr<Enc> next_att;
      if (l + 1 < cfg_.n_layers) next_att = make_att(l + 1);
      else { next_att = std::make_unique<Enc>(false); next_att->enqueue(); }   // reserved head CB
      Timer tw;
      while (rdy.signaledValue < r0 + 1 + (uint64_t)l) { }
      wait_ms += tw.ms();
      Timer tx;
      const int* ids = (const int*)P.rids.cpu();
      ExpertLease lease = cache_->ensure(l, ids, 6);
      if (pred_on) prefetch_predicted(l);
      const uint64_t wv = c0 + 2 + 2 * (uint64_t)l, sv = wv + 1;
      moe->between([&](void* cb) {
        [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:chain value:wv];
        lease.encode_wait(cb);
      });
      q::MoeDecodeArgs m;
      m.t_gate_up = (QType)(int)lease.views[0].type[0]; m.t_down = (QType)(int)lease.views[0].type[2];
      m.n_routed = 6;
      for (int k = 0; k < 6; k++) m.slot[k] = q::ExpertSlot{lease.views[k].mat[0], lease.views[k].mat[1], lease.views[k].mat[2]};
      m.weights = P.rw; m.has_shared = true; m.t_shared = L.sh_gate.t;
      m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
      m.x = P.x; m.y = P.sub; m.scratch = P.moe_dec_scratch;
      q::encode_moe_decode(*moe, m);
      moe->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:chain value:sv]; });
      lease.release_after(moe->cmdbuf());
      moe->commit();
      exp_ms += tx.ms();
      keep.push_back(std::move(att));
      keep.push_back(std::move(moe));
      att = std::move(next_att);
    }
    e = std::move(att);
    const uint64_t wv = c0 + 1 + 2 * (uint64_t)cfg_.n_layers;
    e->between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeWaitForEvent:chain value:wv]; });
    base = wv; rbase = r0 + cfg_.n_layers;
    for (auto& k : keep) { k->wait(); gpu_ms += k->gpu_ms(); }
  } else
  for (int l = 0; l < cfg_.n_layers; l++) {
    LayerW& L = P.L[l];
    Timer tenc;
    // prefill: a big chunk routes to nearly every expert of a layer, so start streaming this layer's experts now (they
    // load while attention runs) and the next layer's while this layer's MoE runs. MACSEEK_PF_PREFETCH=0 disables.
    if (!dec && n >= pf_prefetch_min) {
      if (l == 0) cache_->prefetch(0, all_ids.data(), 384);
      if (!pf_pred && l + 1 < cfg_.n_layers) cache_->prefetch(l + 1, all_ids.data(), 384);
    }
    encode_attn_side(*e, l);
    prof_.encode_ms += tenc.ms();
    static const int force_mode = (int)env_int("MACSEEK_PROF_FORCE", 0);
    if (dec && force_mode == 2) {
      // PROFILING ONLY: forced experts are known up front, so the whole token goes in one command buffer (no handoff)
      static const int forced[6] = {0, 1, 2, 3, 4, 5};
      ExpertLease lease = cache_->ensure(l, forced, 6);
      lease.wait();
      q::MoeDecodeArgs m;
      m.t_gate_up = (QType)(int)lease.views[0].type[0]; m.t_down = (QType)(int)lease.views[0].type[2];
      m.n_routed = 6;
      for (int k = 0; k < 6; k++) m.slot[k] = q::ExpertSlot{lease.views[k].mat[0], lease.views[k].mat[1], lease.views[k].mat[2]};
      m.weights = P.rw; m.has_shared = true; m.t_shared = L.sh_gate.t;
      m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
      m.x = P.x; m.y = P.sub; m.scratch = P.moe_dec_scratch;
      q::encode_moe_decode(*e, m);
      e->barrier();
      lease.release_after(e->cmdbuf());
      continue;
    }
    Timer tw;
    e->wait();
    wait_ms += tw.ms();
    gpu_ms += e->gpu_ms();

    // ---- MoE ----
    Timer tx;
    const int* ids = (const int*)P.rids.cpu();
    if (dec) {
      static const bool force_ids = env_int("MACSEEK_PROF_FORCE", 0) != 0;   // PROFILING ONLY: wrong output
      static const int forced[6] = {0, 1, 2, 3, 4, 5};
      ExpertLease lease = cache_->ensure(l, force_ids ? forced : ids, 6);
      if (pred_on && !force_ids) prefetch_predicted(l);
      e = std::make_unique<Enc>();
      if (dec_cpu_wait) lease.wait();
      else e->between([&](void* cb) { lease.encode_wait(cb); });
      q::MoeDecodeArgs m;
      m.t_gate_up = (QType)(int)lease.views[0].type[0]; m.t_down = (QType)(int)lease.views[0].type[2];
      m.n_routed = 6;
      for (int k = 0; k < 6; k++) m.slot[k] = q::ExpertSlot{lease.views[k].mat[0], lease.views[k].mat[1], lease.views[k].mat[2]};
      m.weights = P.rw; m.has_shared = true; m.t_shared = L.sh_gate.t;
      m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
      m.x = P.x; m.y = P.sub; m.scratch = P.moe_dec_scratch;
      q::encode_moe_decode(*e, m);
      e->barrier();
      lease.release_after(e->cmdbuf());
    } else {
      if (pf_pred && n >= pf_prefetch_min && l + 1 < cfg_.n_layers) {
        const float* lg = (const float*)P.plog_n.cpu();
        const float* bias = (const float*)P.L[l + 1].bias.b.cpu();
        std::vector<char> want(384, 0);
        std::pair<float, int> sc[384];
        const int k = std::min(pf_pred_k, 384);
        for (int t = 0; t < n; t++) {
          const float* r = lg + (size_t)t * 384;
          for (int x = 0; x < 384; x++) { float v = r[x]; float sp = v > 20.f ? v : std::log1p(std::exp(v)); sc[x] = {std::sqrt(sp) + bias[x], x}; }
          std::partial_sort(sc, sc + k, sc + 384, [](auto& a, auto& b) { return a.first > b.first; });
          for (int i = 0; i < k; i++) want[sc[i].second] = 1;
        }
        std::vector<int> ids;
        for (int x = 0; x < 384; x++) if (want[x]) ids.push_back(x);
        cache_->prefetch(l + 1, ids.data(), (int)ids.size());
      }
      // CPU routing tables, experts processed in groups that fit the cache
      Timer troute;
      const float* rw = (const float*)P.rw.cpu();
      std::vector<std::vector<int>> toks_of(384);
      for (int t = 0; t < n; t++) for (int k = 0; k < 6; k++) toks_of[ids[t * 6 + k]].push_back(t * 6 + k);
      std::vector<int> used;
      for (int x = 0; x < 384; x++) if (!toks_of[x].empty()) used.push_back(x);
      // cached experts first, so the GPU starts while misses load
      std::stable_partition(used.begin(), used.end(), [&](int x) { return cache_->contains(l, x); });
      const int G = std::max(1, o_.moe_group);
      const int ngroups = ((int)used.size() + G - 1) / G;
      route_ms += troute.ms();
      experts_used += (long)used.size(); groups_run += ngroups;
      std::vector<ExpertLease> leases(ngroups);
      static id<MTLEvent> pchain = [(__bridge id<MTLDevice>)Device::get().device() newEvent];
      static uint64_t pval = 0;
      std::vector<std::unique_ptr<Enc>> encs(ngroups);
      auto group_ids = [&](int gi) { return std::vector<int>(used.begin() + gi * G, used.begin() + std::min((int)used.size(), (gi + 1) * G)); };
      if (ngroups > 0) leases[0] = cache_->ensure(l, group_ids(0));
      for (int gi = 0; gi < ngroups; gi++) {
        if (gi + 1 < ngroups) leases[gi + 1] = cache_->ensure(l, group_ids(gi + 1));  // overlap I/O with this group
        if (gi >= 2) { encs[gi - 2]->wait(); moe_gpu_ms += encs[gi - 2]->gpu_ms(); encs[gi - 2].reset(); }  // scratch reuse
        std::vector<int> gids = group_ids(gi);
        const int sb = gi & 1;
        // routing arrays for this group in route_dev[sb]: tok[total], wt[total], slot_of[n*6]
        int total = 0;
        for (int x : gids) total += (int)toks_of[x].size();
        int* tok = (int*)P.route_dev[sb].cpu();
        float* wtp = (float*)(tok + n * 6);
        int* slot_of = (int*)(wtp + n * 6);
        for (int i = 0; i < n * 6; i++) slot_of[i] = -1;
        std::vector<int> offs(gids.size() + 1, 0);
        std::vector<q::ExpertSlot> slots(gids.size());
        int c = 0;
        for (size_t j = 0; j < gids.size(); j++) {
          offs[j] = c;
          for (int ent : toks_of[gids[j]]) { tok[c] = ent / 6; wtp[c] = rw[ent]; slot_of[ent] = c; c++; }
          auto& v = leases[gi].views[j];
          slots[j] = q::ExpertSlot{v.mat[0], v.mat[1], v.mat[2]};
        }
        offs[gids.size()] = c;
        encs[gi] = std::make_unique<Enc>();
        Enc& eg = *encs[gi];
        if (cpu_wait_io) leases[gi].wait();
        // groups accumulate into the same Y (sub): order their command buffers on the GPU (they may otherwise
        // overlap: untracked resources), while the CPU keeps encoding / loading ahead
        eg.between([&](void* cb) {
          id<MTLCommandBuffer> c = (__bridge id<MTLCommandBuffer>)cb;
          if (gi > 0) [c encodeWaitForEvent:pchain value:pval];
          if (!cpu_wait_io) leases[gi].encode_wait(cb);
        });
        {  // grow this scratch if the group needs more (only after the CB that used it finished: waited above)
          size_t need = q::moe_prefill_scratch_bytes(total, n, (int)gids.size());
          if (need > P.moe_scr[sb].size) {
            if (gi >= 1 && encs[gi - 1] && sb == ((gi - 1) & 1)) encs[gi - 1]->wait();
            Device::get().release(P.moe_scr[sb]);
            P.moe_scr[sb] = Device::get().alloc(need + (need >> 3), "moe_pf");
          }
        }
        q::MoePrefillArgs m;
        m.t_gate_up = (QType)(int)leases[gi].views[0].type[0]; m.t_down = (QType)(int)leases[gi].views[0].type[2];
        m.n_tokens = n; m.n_exp = (int)gids.size(); m.slots = slots.data(); m.offs = offs.data();
        m.tok = P.route_dev[sb].sub(0, (size_t)n * 6 * 4); m.wt = P.route_dev[sb].sub((size_t)n * 6 * 4, (size_t)n * 6 * 4);
        m.slot_of = P.route_dev[sb].sub((size_t)n * 6 * 8, (size_t)n * 6 * 4); m.k_per_token = 6;
        m.has_shared = gi == 0; m.t_shared = L.sh_gate.t; m.shared = q::ExpertSlot{L.sh_gate.b, L.sh_up.b, L.sh_down.b};
        m.X = P.x; m.Y = P.sub; m.scratch = P.moe_scr[sb];
        q::encode_moe_prefill(eg, m);
        eg.between([&](void* cb) { [(__bridge id<MTLCommandBuffer>)cb encodeSignalEvent:pchain value:++pval]; });
        leases[gi].release_after(eg.cmdbuf());
        eg.commit();
        (void)total;
      }
      for (auto& x : encs) if (x) { x->wait(); moe_gpu_ms += x->gpu_ms(); }
      if (ngroups == 0) { /* impossible: every token routes to 6 experts */ }
      e = std::make_unique<Enc>();
    }
    exp_ms += tx.ms();
  }

  // ---- head ----

  int32_t next = -1;
  if (!head_done) encode_head(*e);

  Timer tw;
  e->wait();
  wait_ms += tw.ms();
  gpu_ms += e->gpu_ms();
  next = *(const int32_t*)P.amax_i.cpu();
  if (logits_all) memcpy(logits_all, P.logits.cpu(), (size_t)n * V * 4);
  if (logits_last) memcpy(logits_last, (const float*)P.logits.cpu() + (logits_all ? (size_t)(n - 1) * V : 0), (size_t)V * 4);
  pos_ += n;
  const LayerStats st1 = cache_->total_stats();
  if (dec) {
    prof_.tokens++; prof_.total_ms += tt.ms(); prof_.gpu_wait_ms += wait_ms; prof_.expert_ms += exp_ms; prof_.engram_ms += engram_ms;
    prof_.gpu_ms += gpu_ms; prof_.io_stall_ms += st1.stall_ms - st0.stall_ms; prof_.io_bytes += st1.bytes_read - st0.bytes_read;
  } else {
    pprof_.chunks++; pprof_.tokens += n; pprof_.total_ms += tt.ms(); pprof_.engram_ms += engram_ms;
    pprof_.main_gpu_ms += gpu_ms; pprof_.main_wait_ms += wait_ms; pprof_.moe_ms += exp_ms; pprof_.moe_gpu_ms += moe_gpu_ms;
    pprof_.route_ms += route_ms; pprof_.io_bytes += st1.bytes_read - st0.bytes_read;
    pprof_.experts_used += experts_used; pprof_.groups += groups_run;
    if (env_int("MACSEEK_PROF", 0))
      fprintf(stderr, "prefill-prof: pos %d n %d total %.0f ms | engram %.0f | main gpu %.0f wait %.0f | moe %.0f (gpu %.0f) route %.1f | "
              "io %.2f GB | experts %ld groups %ld\n", start, n, tt.ms(), engram_ms, gpu_ms, wait_ms, exp_ms, moe_gpu_ms, route_ms,
              (st1.bytes_read - st0.bytes_read) / 1e9, experts_used, groups_run);
  }
  return next;
}

// ---------------------------------------------------------------- snapshot ----
static constexpr uint64_t kMagic = 0x3134534b5345534dull;  // "MSESKS41"

bool Model::save_state(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  auto spans = attn::persistent_state(p_->ast, pos_);
  uint64_t hdr[3] = {kMagic, (uint64_t)pos_, spans.size()};
  f.write((const char*)hdr, sizeof hdr);
  EngramHasher::State hs = hasher_->state();
  f.write((const char*)&hs, sizeof hs);
  for (auto& s : spans) { uint64_t b = s.bytes; f.write((const char*)&b, 8); f.write((const char*)s.buf.cpu(), s.bytes); }
  return (bool)f;
}

bool Model::load_state(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint64_t hdr[3];
  f.read((char*)hdr, sizeof hdr);
  if (hdr[0] != kMagic) return false;
  EngramHasher::State hs;
  f.read((char*)&hs, sizeof hs);
  auto spans = attn::persistent_state(p_->ast, (int)hdr[1]);
  if (spans.size() != hdr[2]) return false;
  attn::reset(p_->ast);
  for (auto& s : spans) {
    uint64_t b; f.read((char*)&b, 8);
    if (b != s.bytes) return false;
    f.read((char*)s.buf.cpu(), b);
  }
  hasher_->set_state(hs);
  pos_ = (int)hdr[1];
  return (bool)f;
}

std::string Model::stats_report() const { return cache_->stats_report(); }

std::string PrefillProfile::str() const {
  char b[512];
  snprintf(b, sizeof b, "prefill: %ld tok in %d chunks, %.1f tok/s | per chunk ms: total %.0f engram %.0f main_gpu %.0f main_wait %.0f "
           "moe %.0f moe_gpu %.0f route %.1f | io %.2f GB/chunk | experts/chunk %.0f groups/chunk %.1f",
           tokens, chunks, tokens / (total_ms / 1e3), total_ms / chunks, engram_ms / chunks, main_gpu_ms / chunks,
           main_wait_ms / chunks, moe_ms / chunks, moe_gpu_ms / chunks, route_ms / chunks, io_bytes / 1e9 / chunks,
           (double)experts_used / chunks, (double)groups / chunks);
  return b;
}

}  // namespace ms::ds41
