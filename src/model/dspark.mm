// dspark block drafter. See dspark.hpp.
#import <Metal/Metal.h>
#include "model/dspark.hpp"

#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include "core/env.hpp"
#include "core/model_path.hpp"
#include "core/timer.hpp"
#include "gguf/gguf.hpp"
#include "kernels/arch.hpp"
#include "kernels/quant.hpp"
#include "store/weights.hpp"

namespace ms::ds41 {

using q::QType;

namespace {
W wt(const Weights& w, const std::string& name) {
  const WTensor* t = w.find(name);
  if (!t) return W{};
  W r; r.b = t->buf; r.t = (QType)(int)t->type;
  r.cols = (int)t->ne[0]; r.rows = t->n_dims > 1 ? (int)t->ne[1] : 1;
  return r;
}
std::string ln(int l, const char* s) { char b[128]; snprintf(b, sizeof b, "blk.%d.%s", l, s); return b; }
// y[n][rows] = x[n][cols] . W^T (draft: accuracy of the half-staged small-n kernels is plenty)
void lin(Enc& e, const W& w, const Buffer& x, const Buffer& y, int n, int groups = 1) {
  if (n == 1) q::encode_matvec(e, w.t, w.b, w.rows, w.cols, x, y, false, groups);
  else if (n <= 16 && (groups > 1 || n < 8)) q::encode_matvec_n(e, w.t, w.b, w.rows, w.cols, x, y, n, false, groups);
  else if (groups == 1) q::encode_matmul(e, w.t, w.b, w.rows, w.cols, x, y, n, false);
  else { fprintf(stderr, "dspark: grouped linear n=%d\n", n); abort(); }
}
struct DspKvArgs { int pos0, n, ring; float eps; };
struct DspAttnArgs { int P, B; float scale; };
struct DspRouteArgs { int n_exp, k; float scale; };
struct DspMxArgs { uint32_t k, rows, cols, row_bytes; uint64_t expert_bytes; float lim; uint32_t pad; };
}  // namespace

// $MACSEEK_DRAFT, else <source tree>/models/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf, else the same file name in the
// models/ directory that holds the target model folder (<models>/<repo>/IQ2_XXS/split1).
std::string DSpark::default_path() {
  std::string e = env_str("MACSEEK_DRAFT", "");
  if (!e.empty()) return e;
  static const char* kName = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf";
#ifdef MACSEEK_SOURCE_DIR
  {
    const std::string p = std::string(MACSEEK_SOURCE_DIR) + "/models/" + kName;
    if (access(p.c_str(), R_OK) == 0) return p;
  }
#endif
  std::string m = default_model_dir();
  for (int up = 0; up < 2 && m.rfind('/') != std::string::npos; up++) m = m.substr(0, m.rfind('/'));
  const std::string p = m + "/" + kName;
  if (access(p.c_str(), R_OK) == 0) return p;
#ifdef MACSEEK_SOURCE_DIR
  return std::string(MACSEEK_SOURCE_DIR) + "/models/" + kName;
#else
  return "";
#endif
}

DSpark::DSpark(const std::string& path, const DSparkTarget& tgt) : t_(tgt) {
  Timer tm;
  auto& d = Device::get();
  g_ = std::make_unique<Gguf>(std::vector<std::string>{path});
  if (g_->get_str("general.architecture") != "dflash") { fprintf(stderr, "dspark: %s is not a dflash GGUF\n", path.c_str()); abort(); }
  block_ = (int)g_->get_i64("dflash.block_size", 5);
  n_exp_ = (int)g_->get_i64("dflash.expert_count", 128);
  k_exp_ = (int)g_->get_i64("dflash.expert_used_count", 3);
  scale_ = (float)g_->get_f64("dflash.expert_weights_scale", 1.5);
  eps_ = (float)g_->get_f64("dflash.attention.layer_norm_rms_epsilon", 1e-20);
  noise_ = (int)g_->get_i64("tokenizer.ggml.mask_token_id", 128799);
  const int nl = (int)g_->get_i64("dflash.block_count", 3);
  if (nl != 3 || n_exp_ > 128 || k_exp_ > 8 || block_ > 8) { fprintf(stderr, "dspark: unsupported shape\n"); abort(); }
  w_ = std::make_unique<Weights>(*g_);
  bytes_ = w_->bytes();
  L_.resize(nl);
  // experts: preaded into wired anonymous memory (Weights skips *_exps)
  struct XT { const TensorInfo* t; Buffer* dst; };
  std::vector<XT> xts;
  for (int l = 0; l < nl; l++) {
    L3& L = L_[l];
    auto g = [&](const char* s) { return wt(*w_, ln(l, s)); };
    L.attn_norm = g("attn_norm.weight"); L.ffn_norm = g("ffn_norm.weight");
    L.q_a = g("attn_q_a.weight"); L.q_a_norm = g("attn_q_a_norm.weight"); L.q_b = g("attn_q_b.weight");
    L.kv = g("attn_kv.weight"); L.kv_norm = g("attn_kv_a_norm.weight");
    L.o_a = g("attn_output_a.weight"); L.o_b = g("attn_output_b.weight"); L.sinks = g("attn_sinks.weight");
    L.gate_inp = g("ffn_gate_inp.weight"); L.bias = g("exp_probs_b.bias");
    L.sh_gate = g("ffn_gate_shexp.weight"); L.sh_up = g("ffn_up_shexp.weight"); L.sh_down = g("ffn_down_shexp.weight");
    L.hc_attn_fn = g("hc_attn_fn.weight"); L.hc_attn_scale = g("hc_attn_scale.weight"); L.hc_attn_base = g("hc_attn_base.weight");
    L.hc_ffn_fn = g("hc_ffn_fn.weight"); L.hc_ffn_scale = g("hc_ffn_scale.weight"); L.hc_ffn_base = g("hc_ffn_base.weight");
    if (!L.q_b || !L.gate_inp || L.gate_inp.t != QType::F32 || !L.hc_attn_fn) { fprintf(stderr, "dspark: missing tensors\n"); abort(); }
    for (auto [nm, dst] : {std::pair<const char*, Buffer*>{"ffn_gate_exps.weight", &L.gate_exps},
                           {"ffn_up_exps.weight", &L.up_exps}, {"ffn_down_exps.weight", &L.down_exps}}) {
      const TensorInfo* t = g_->find(ln(l, nm));
      if (!t || t->type != GType::MXFP4) { fprintf(stderr, "dspark: %s missing or not MXFP4\n", ln(l, nm).c_str()); abort(); }
      xts.push_back({t, dst});
    }
  }
  fc_ = wt(*w_, "fc.weight"); enc_norm_ = wt(*w_, "enc.output_norm.weight"); out_norm_ = wt(*w_, "output_norm.weight");
  mk_w1_ = wt(*w_, "markov_w1.weight"); mk_w2_ = wt(*w_, "markov_w2.weight");
  if (!fc_ || !enc_norm_ || !out_norm_ || !mk_w1_ || !mk_w2_ || mk_w2_.t != QType::BF16) { fprintf(stderr, "dspark: missing head tensors\n"); abort(); }
  {
    const size_t pg = 16384;
    size_t total = 0;
    for (auto& x : xts) total += (x.t->nbytes + pg - 1) / pg * pg;
    void* mem = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (mem == MAP_FAILED) { fprintf(stderr, "dspark: mmap %zu failed\n", total); abort(); }
    if (env_int("MACSEEK_WEIGHTS_MLOCK", 1)) mlock(mem, total);
    xmem_ = mem; xmem_bytes_ = total;
    struct Job { int fd; char* dst; uint64_t off, len; };
    std::vector<Job> jobs;
    size_t o = 0;
    for (auto& x : xts) {
      char* dst = (char*)mem + o;
      const uint64_t chunk = 8ull << 20;
      for (uint64_t a = 0; a < x.t->nbytes; a += chunk)
        jobs.push_back({g_->fd_nocache(x.t->split), dst + a, x.t->offset + a, std::min(chunk, x.t->nbytes - a)});
      *x.dst = d.wrap(dst, (x.t->nbytes + pg - 1) / pg * pg, "dspark.exps");
      x.dst->size = x.t->nbytes;
      o += (x.t->nbytes + pg - 1) / pg * pg;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> th;
    for (int i = 0; i < 8; i++)
      th.emplace_back([&] {
        for (size_t j; (j = next++) < jobs.size();)
          if (pread_full(jobs[j].fd, jobs[j].dst, jobs[j].len, jobs[j].off) != (long long)jobs[j].len) bad = true;
      });
    for (auto& t : th) t.join();
    if (bad) { fprintf(stderr, "dspark: expert read failed\n"); abort(); }
    bytes_ += total;
  }
  const int K = 8, D = 5120, V = t_.head.rows;
  auto A = [&](size_t b, const char* l) { return d.alloc(b, l); };
  for (int l = 0; l < 3; l++) ring_[l] = A(128 * 512 * 2, "dspark.ring");
  slot_pos_ = A(128 * 4, "dspark.slots");
  slot_host_.assign(128, -1);
  for (int i = 0; i < 128; i++) ((int*)slot_pos_.cpu())[i] = -1;
  feat_ = A((size_t)128 * 3 * D * 4, "dspark.feat"); mainx_ = A((size_t)128 * D * 4, "dspark.mainx");
  kvtmp_ = A((size_t)128 * 512 * 4, "dspark.kvtmp");
  embrows_ = A((size_t)K * 8192 + 16384, "dspark.emb");
  h_ = A((size_t)K * 20480 * 4, "dspark.h"); x_ = A((size_t)K * D * 4, "dspark.x"); sub_ = A((size_t)K * D * 4, "dspark.sub");
  mixA_ = A((size_t)K * 96, "dspark.mixA"); mixF_ = A((size_t)K * 96, "dspark.mixF");
  qa_ = A((size_t)K * 1280 * 4, "dspark.qa"); qrn_ = A((size_t)K * 1280 * 4, "dspark.qrn"); q_ = A((size_t)K * 32768 * 4, "dspark.q");
  kvb_ = A((size_t)K * 512 * 4, "dspark.kv"); blk_ = A((size_t)K * 512 * 2, "dspark.blk");
  aout_ = A((size_t)K * 32768 * 4, "dspark.aout"); oa_ = A((size_t)K * 8192 * 4, "dspark.oa");
  rlog_ = A((size_t)K * 128 * 4, "dspark.rlog"); rids_ = A((size_t)K * 8 * 4, "dspark.rids"); rw_ = A((size_t)K * 8 * 4, "dspark.rw");
  hx_ = A((size_t)K * 8 * 2304 * 4, "dspark.hx");
  shg_ = A((size_t)K * 2304 * 4, "dspark.shg"); shu_ = A((size_t)K * 2304 * 4, "dspark.shu"); sha_ = A((size_t)K * 2304 * 4, "dspark.sha");
  logits_ = A((size_t)K * V * 4, "dspark.logits"); amax_i_ = A(256, "dspark.amax_i"); amax_v_ = A(256, "dspark.amax_v");
  mk_in_ = A(256 * 4, "dspark.mk_in");
  fprintf(stderr, "dspark: %s: %.2f GiB (dense %.2f, experts %.2f), block %d, %d experts top-%d, loaded in %.1f s\n", path.c_str(),
          bytes_ / 1073741824.0, w_->bytes() / 1073741824.0, xmem_bytes_ / 1073741824.0, block_, n_exp_, k_exp_, tm.ms() / 1e3);
}

DSpark::~DSpark() {
  if (xmem_) { munlock(xmem_, xmem_bytes_); munmap(xmem_, xmem_bytes_); }
}

void DSpark::reset() {
  slot_host_.assign(128, -1);
  for (int i = 0; i < 128; i++) ((int*)slot_pos_.cpu())[i] = -1;
  pend_.clear();
}

void DSpark::add_feature(int pos, const float* f) {
  pend_.emplace_back(pos, std::vector<float>(f, f + 3 * 5120));
}

double DSpark::draft(int32_t anchor, int P, int B, int32_t* out) {
  Timer tm;
  const int D = 5120, V = t_.head.rows;
  if (B < 1 || B > block_) { fprintf(stderr, "dspark: B=%d\n", B); abort(); }
  static arch::Scratch asc = arch::Scratch::create(64, 16);
  static const int profm = (int)env_int("MACSEEK_DSP_PROF", 0);   // PROFILING: 1 = GPU ms per segment (splits the CB), 2 = cpu/gpu split
  static const bool prof = profm == 1;
  static double seg[8] = {0}; static long calls = 0;
  auto ep = std::make_unique<Enc>();
  auto cut = [&](int k) { if (!prof) return; ep->wait(); seg[k] += ep->gpu_ms(); ep = std::make_unique<Enc>(); };
#define e (*ep)
  // ---- 1. inject pending features of positions [P-128, P-1] (newest wins per slot) ----
  {
    std::vector<std::pair<int, const float*>> todo;
    for (auto& [p, f] : pend_) if (p >= P - 128 && p < P) todo.push_back({p, f.data()});
    std::stable_sort(todo.begin(), todo.end(), [](auto& a, auto& b) { return a.first < b.first; });
    // dedupe positions (keep the last added)
    std::vector<std::pair<int, const float*>> rows;
    for (size_t i = 0; i < todo.size(); i++) {
      if (!rows.empty() && rows.back().first == todo[i].first) rows.back() = todo[i]; else rows.push_back(todo[i]);
    }
    const int n = (int)rows.size();
    if (n > 0) {
      float* fb = (float*)feat_.cpu();
      for (int i = 0; i < n; i++) memcpy(fb + (size_t)i * 3 * D, rows[i].second, (size_t)3 * D * 4);   // [f37|f38|f39]
      lin(e, fc_, feat_, mainx_, n);
      e.barrier();
      arch::encode_rmsnorm(e, mainx_, enc_norm_.b, mainx_, n, D, eps_);
      e.barrier();
      for (int l = 0; l < 3; l++) {
        lin(e, L_[l].kv, mainx_, kvtmp_.sub((size_t)0, (size_t)n * 512 * 4), n);
        e.barrier();
        // runs of consecutive positions
        for (int i = 0; i < n;) {
          int j = i + 1;
          while (j < n && rows[j].first == rows[j - 1].first + 1) j++;
          DspKvArgs a{rows[i].first, j - i, 1, eps_};
          e.set_pipeline("dsp_kv");
          e.set_buffer(0, kvtmp_.sub((size_t)i * 512 * 4, (size_t)(j - i) * 512 * 4)); e.set_buffer(1, L_[l].kv_norm.b);
          e.set_buffer(2, t_.rope_plain); e.set_buffer(3, ring_[l]); e.set_value(4, a);
          e.dispatch_groups(j - i, 1, 1, 128);
          i = j;
        }
        e.barrier();
      }
      for (auto& [p, f] : rows) { slot_host_[p % 128] = p; ((int*)slot_pos_.cpu())[p % 128] = p; (void)f; }
    }
    pend_.erase(std::remove_if(pend_.begin(), pend_.end(), [&](auto& x) { return x.first < P; }), pend_.end());
  }
  cut(0);
  // ---- 2. block [anchor, noise...] at positions P..P+B-1 ----
  std::vector<int32_t> toks(B, noise_);
  toks[0] = anchor;
  t_.te->fetch(toks.data(), B, embrows_.cpu());
  arch::encode_embed(e, (int)t_.te->type(), embrows_, (uint32_t)t_.te->row_bytes(), h_, B, 4);
  e.barrier();
  { float* mf = (float*)mixF_.cpu(); for (int t = 0; t < B; t++) arch::fill_identity_mix(mf + t * 24); }
  for (int l = 0; l < 3; l++) {
    const L3& L = L_[l];
    {
      arch::HcStep a; a.h_in = h_; a.h_out = h_; if (l > 0) a.x_sub = sub_; a.mix_prev = mixF_;
      a.fn = L.hc_attn_fn.b; a.scale = L.hc_attn_scale.b; a.base = L.hc_attn_base.b; a.mix_out = mixA_;
      a.norm_w = L.attn_norm.b; a.x_out = x_; a.n = B;
      arch::encode_hc_step(e, a, asc); e.barrier();
    }
    cut(1);
    lin(e, L.q_a, x_, qa_, B);
    lin(e, L.kv, x_, kvb_, B);
    e.barrier();
    arch::encode_rmsnorm(e, qa_, L.q_a_norm.b, qrn_, B, 1280, eps_);
    {
      DspKvArgs a{P, B, 0, eps_};
      e.set_pipeline("dsp_kv");
      e.set_buffer(0, kvb_); e.set_buffer(1, L.kv_norm.b); e.set_buffer(2, t_.rope_plain); e.set_buffer(3, blk_); e.set_value(4, a);
      e.dispatch_groups(B, 1, 1, 128);
    }
    e.barrier();
    lin(e, L.q_b, qrn_, q_, B);
    e.barrier();
    {
      DspAttnArgs a{P, B, 1.0f / std::sqrt(512.f)};
      e.set_pipeline("dsp_attn");
      e.set_buffer(0, q_); e.set_buffer(1, ring_[l]); e.set_buffer(2, slot_pos_); e.set_buffer(3, blk_);
      e.set_buffer(4, L.sinks.b); e.set_buffer(5, t_.rope_plain); e.set_buffer(6, aout_); e.set_value(7, a);
      e.dispatch_groups(64, B, 1, 128);
    }
    e.barrier();
    cut(2);
    lin(e, L.o_a, aout_, oa_, B, 8);
    e.barrier();
    lin(e, L.o_b, oa_, sub_, B);
    e.barrier();
    {
      arch::HcStep a; a.h_in = h_; a.h_out = h_; a.x_sub = sub_; a.mix_prev = mixA_;
      a.fn = L.hc_ffn_fn.b; a.scale = L.hc_ffn_scale.b; a.base = L.hc_ffn_base.b; a.mix_out = mixF_;
      a.norm_w = L.ffn_norm.b; a.x_out = x_; a.n = B;
      arch::encode_hc_step(e, a, asc); e.barrier();
    }
    cut(3);
    // router (GPU top-k) + shared expert
    lin(e, L.gate_inp, x_, rlog_, B);
    lin(e, L.sh_gate, x_, shg_, B);
    lin(e, L.sh_up, x_, shu_, B);
    e.barrier();
    {
      DspRouteArgs a{n_exp_, k_exp_, scale_};
      e.set_pipeline("dsp_route");
      e.set_buffer(0, rlog_); e.set_buffer(1, L.bias.b); e.set_buffer(2, rids_); e.set_buffer(3, rw_); e.set_value(4, a);
      e.dispatch_groups(B, 1, 1, 128);
      const float lim = 10.f;
      e.set_pipeline("dsp_swiglu");
      e.set_buffer(0, shg_); e.set_buffer(1, shu_); e.set_buffer(2, sha_); e.set_value(3, lim);
      e.dispatch_threads(B * 2304, 1, 1, 256);
    }
    e.barrier();
    lin(e, L.sh_down, sha_, sub_, B);   // sub = shared expert output (the routed experts accumulate into it)
    cut(4);
    {
      const uint32_t rb_gu = (uint32_t)(5120 / 32 * 17), rb_d = (uint32_t)(2304 / 32 * 17);
      DspMxArgs a{(uint32_t)k_exp_, 2304, 5120, rb_gu, (uint64_t)2304 * rb_gu, 10.f, 0};
      e.set_pipeline("dsp_mx_gu");
      e.set_buffer(0, x_); e.set_buffer(1, L.gate_exps); e.set_buffer(2, L.up_exps); e.set_buffer(3, rids_);
      e.set_buffer(4, rw_); e.set_buffer(5, hx_); e.set_value(6, a);
      e.dispatch_groups(2304 / 8, B * k_exp_, 1, 256);
      e.barrier();
      DspMxArgs b{(uint32_t)k_exp_, 5120, 2304, rb_d, (uint64_t)5120 * rb_d, 10.f, 0};
      e.set_pipeline("dsp_mx_down");
      e.set_buffer(0, hx_); e.set_buffer(1, L.down_exps); e.set_buffer(2, rids_); e.set_buffer(3, sub_); e.set_value(4, b);
      e.dispatch_groups(5120 / 8, B, 1, 256);
    }
    e.barrier();
    cut(5);
  }
  {
    arch::HcStep a; a.h_in = h_; a.h_out = h_; a.x_sub = sub_; a.mix_prev = mixF_; a.norm_w = out_norm_.b; a.x_out = x_; a.n = B;
    arch::encode_hc_step(e, a, asc); e.barrier();
  }
  lin(e, t_.head, x_, logits_, B);
  e.barrier();
  cut(6);
  // ---- 3. Markov chain: logits_i += w2 . w1[prev]; d_i = argmax; prev = d_i ----
  ((int32_t*)amax_i_.cpu())[0] = anchor;
  for (int i = 0; i < B; i++) {
    uint32_t ga[2] = {(uint32_t)i, (uint32_t)mk_w1_.cols};
    e.set_pipeline("dsp_gather_bf16");
    e.set_buffer(0, mk_w1_.b); e.set_buffer(1, amax_i_); e.set_buffer(2, mk_in_); e.set_bytes(3, ga, sizeof ga);
    e.dispatch_threads(mk_w1_.cols, 1, 1, 256);
    e.barrier();
    q::encode_matvec(e, mk_w2_.t, mk_w2_.b, mk_w2_.rows, mk_w2_.cols, mk_in_, logits_.sub((size_t)i * V * 4, (size_t)V * 4), true);
    e.barrier();
    arch::encode_argmax(e, logits_.sub((size_t)i * V * 4, (size_t)V * 4), 1, V, amax_i_.sub((size_t)(i + 1) * 4, 4),
                        amax_v_.sub((size_t)(i + 1) * 4, 4), asc, 0.f, 0);
    e.barrier();
  }
  const double enc_ms = tm.ms();
  e.wait();
  if (profm == 2) {
    static double se = 0, sw = 0, sg = 0; static long nc = 0;
    se += enc_ms; sw += tm.ms() - enc_ms; sg += e.gpu_ms();
    if (++nc % 32 == 0) fprintf(stderr, "dspark-time (B=%d, avg of %ld): encode %.2f wait %.2f gpu %.2f ms\n", B, nc, se / nc, sw / nc, sg / nc);
  }
  if (prof) {
    seg[7] += e.gpu_ms();
    if (++calls % 16 == 0)
      fprintf(stderr, "dspark-prof (B=%d, avg of %ld): inject %.2f | per 3 stages: hc_a %.2f attn %.2f o_proj+hc_f %.2f router+shared %.2f routed %.2f | head %.2f markov %.2f ms\n",
              B, calls, seg[0] / calls, seg[1] / calls, seg[2] / calls, seg[3] / calls, seg[4] / calls, seg[5] / calls, seg[6] / calls, seg[7] / calls);
  }
#undef e
  for (int i = 0; i < B; i++) out[i] = ((const int32_t*)amax_i_.cpu())[i + 1];
  (void)D;
  return tm.ms();
}

}  // namespace ms::ds41
