// Decode fusion switches (model/dec_layer, docs/modules/dec_gpu.md) must not change outputs:
//  (1) q::encode_matvec_rmsnorm == arch::encode_rmsnorm -> barrier -> q::encode_matvec, bit for bit (several types/dims)
//  (2) per layer kind (engram 1, kv+index source 2, plain ratio-2 5, candidate source 20, candidate user 24, Q5_K /
//      IQ2_XXS 31): attention side + table MoE with every fused variant vs the unfused sequence -> identical h, x, sub,
//      q, attention output, KV/compressed state rows, router ids/weights (bit-exact)
//  (3) a 3-layer chain (0, 1, 2) with the engram_kv background chunks (MACSEEK_DEC_ENGRAM_BG 1 and 2)
//  (4) run-to-run determinism of the baseline (arch_hc_step in-place race fix)
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "dec_synth.hpp"
#include "kernels/arch.hpp"
#include "kernels/attn.hpp"
#include "kernels/quant.hpp"
#include "model/dec_layer.hpp"

using namespace ms;
using namespace ms::ds41;
using q::QType;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void test_matvec_rmsnorm() {
  auto& d = Device::get();
  struct Case { QType t; int rows, cols; };
  const Case cases[] = {{QType::Q6_K, 32768, 1280}, {QType::Q5_K, 32768, 1280}, {QType::BF16, 4096, 1280},
                        {QType::Q8_0, 1000, 2048}, {QType::F32, 77, 1536}, {QType::Q6_K, 300, 5120}};
  for (const Case& c : cases) {
    Buffer W = d.alloc(q::row_bytes(c.t, c.cols) * c.rows, "w");
    dsyn::fill_rows(c.t, (uint8_t*)W.cpu(), c.rows, c.cols, 11 + c.cols, 0.03f);
    Buffer x = d.alloc(c.cols * 4, "x"), nw = d.alloc(c.cols * 4, "nw"), xn = d.alloc(c.cols * 4, "xn");
    Buffer y0 = d.alloc(c.rows * 4, "y0"), y1 = d.alloc(c.rows * 4, "y1");
    uint64_t s = 5;
    for (int i = 0; i < c.cols; i++) { ((float*)x.cpu())[i] = dsyn::unif(s) * 3.f; ((float*)nw.cpu())[i] = 1.f + 0.2f * dsyn::unif(s); }
    {
      Enc e;
      arch::encode_rmsnorm(e, x, nw, xn, 1, c.cols, 1e-20f);
      e.barrier();
      q::encode_matvec(e, c.t, W, c.rows, c.cols, xn, y0);
      q::encode_matvec_rmsnorm(e, c.t, W, c.rows, c.cols, x, nw, 1e-20f, y1);
      e.wait();
    }
    const int diff = memcmp(y0.cpu(), y1.cpu(), (size_t)c.rows * 4);
    size_t nd = 0;
    for (int i = 0; i < c.rows; i++) nd += ((uint32_t*)y0.cpu())[i] != ((uint32_t*)y1.cpu())[i];
    printf("matvec_rmsnorm %-7s %5d x %4d: %zu rows differ\n", q::type_name(c.t), c.rows, c.cols, nd);
    CHECK(diff == 0, "matvec_rmsnorm %s %dx%d not bit-identical", q::type_name(c.t), c.rows, c.cols);
  }
}

struct Env {
  Config cfg;
  dsyn::Pool P;
  std::vector<LayerW> L;
  std::vector<W> pred;
  attn::State S;
  arch::Scratch asc;
  DecBufs B;
  Buffer erow, tbl;
  std::vector<Buffer> arena;
  id<MTLSharedEvent> rdy, go;
  uint64_t ev = 0;
  int pos = 3001;
  std::vector<int> layers;
  std::vector<std::vector<uint8_t>> init;  // initial contents of every compared / mutable buffer
  std::vector<Buffer> mut;

  void setup() {
    auto& d = Device::get();
    L.resize(cfg.n_layers); pred.resize(cfg.n_layers);
    // every layer (tensors shared per class: bit-exactness does not need distinct weights)
    for (int l = 0; l < cfg.n_layers; l++) { L[l] = dsyn::make_layer(P, cfg, l, 1); pred[l] = dsyn::make_pred(P, l, 1); }
    attn::Options ao = attn::Options::from_env();
    ao.max_chunk = 1; ao.max_pos = 4096;
    attn::init(S, ao);
    attn::reset(S);
    dsyn::fill_attn_state(S, pos, 9);
    asc = arch::Scratch::create(64, 16);
    auto A = [&](size_t n) { return d.alloc(n, "t"); };
    B.h = A(20480 * 4); B.h2 = A(20480 * 4); B.x = A(5120 * 4); B.sub = A(5120 * 4); B.mixA = A(96); B.mixF = A(96);
    B.qa = A(1280 * 4); B.qrn = A(1280 * 4); B.q = A(32768 * 4); B.kv = A(512 * 4); B.ckv = A(512 * 4);
    B.cgate = A(512 * 4); B.iq = A(4096 * 4); B.iw = A(32 * 4); B.aout = A(32768 * 4); B.oa = A(8192 * 4);
    B.rids = A(64); B.rw = A(64); B.plog = A(4 * 384 * 4); B.ekv[0] = A(25600 * 4); B.ekv[1] = A(25600 * 4);
    B.moe_scratch = A(q::moe_decode_scratch_bytes()); B.edeq = A(2 * 6144 * 4);
    uint64_t s = 3;
    for (int i = 0; i < 20480; i++) ((float*)B.h.cpu())[i] = dsyn::unif(s) * 2.f;
    for (int i = 0; i < 5120; i++) ((float*)B.sub.cpu())[i] = dsyn::unif(s);
    float* mf = (float*)B.mixF.cpu();   // a valid lagged mix: pre, post, comb (rows sum to 1)
    for (int i = 0; i < 4; i++) { mf[i] = 0.25f + 0.1f * i; mf[4 + i] = 0.5f + 0.2f * i; }
    for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) mf[8 + 4 * j + k] = j == k ? 0.7f : 0.1f;
    erow = A(2 * 24 * 144);
    dsyn::fill_rows(QType::IQ4_NL, (uint8_t*)erow.cpu(), 48, 256, 4, 1.f);
    // experts: 6 per (gate/up, down) type combo, shared by the layers of that combo
    tbl = A(40 * 18 * 8);
    std::vector<std::pair<int, Buffer>> combos;
    for (int l = 0; l < cfg.n_layers; l++) {
      const QType gu = dsyn::exp_gu_type(l), dn = dsyn::exp_down_type(l);
      const size_t gb = q::row_bytes(gu, 5120) * 2304, db = q::row_bytes(dn, 2304) * 5120;
      const int key = (int)gu * 100 + (int)dn;
      Buffer b;
      for (auto& cb : combos) if (cb.first == key) b = cb.second;
      const bool fresh = !b;
      if (fresh) { b = d.alloc((2 * gb + db) * 6, "exp"); arena.push_back(b); combos.push_back({key, b}); }
      const uint64_t base = [(__bridge id<MTLBuffer>)b.buf gpuAddress] + b.off;
      for (int k = 0; k < 6; k++) {
        uint8_t* p = (uint8_t*)b.cpu() + (2 * gb + db) * k;
        if (fresh) {
          dsyn::fill_rows(gu, p, 2304, 5120, 100 + key + k, 0.014f);
          dsyn::fill_rows(gu, p + gb, 2304, 5120, 300 + key + k, 0.014f);
          dsyn::fill_rows(dn, p + 2 * gb, 5120, 2304, 500 + key + k, 0.021f);
        }
        uint64_t* t = (uint64_t*)tbl.cpu() + (size_t)l * 18 + k * 3;
        t[0] = base + (2 * gb + db) * k; t[1] = t[0] + gb; t[2] = t[0] + 2 * gb;
      }
    }
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d.device();
    rdy = [dev newSharedEvent]; go = [dev newSharedEvent]; go.signaledValue = 1ull << 60;
    // everything a run reads or writes, restored before each run
    mut = {B.h, B.h2, B.x, B.sub, B.mixA, B.mixF, B.qa, B.qrn, B.q, B.kv, B.ckv, B.cgate, B.iq, B.iw, B.aout, B.oa, B.rids, B.rw,
           B.plog, B.ekv[0], B.ekv[1], B.moe_scratch, B.edeq, S.topk, S.cand_list, S.part, S.ml};
    for (int l = 0; l < attn::kLayers; l++) mut.push_back(S.win[l]);
    for (int i = 0; i < 4; i++) { mut.push_back(S.ckv[i]); mut.push_back(S.ikey[i]); mut.push_back(S.cst[i]); }
    for (auto& b : mut) init.emplace_back((uint8_t*)b.cpu(), (uint8_t*)b.cpu() + b.size);
  }
  void restore() { for (size_t i = 0; i < mut.size(); i++) memcpy(mut[i].cpu(), init[i].data(), mut[i].size); }
  std::vector<std::vector<uint8_t>> run(const DecOptions& o, int sync_mode) {
    restore();
    DecCtx c; c.cfg = &cfg; c.L = &L; c.pred_w = &pred; c.B = B; c.ast = &S; c.asc = &asc; c.o = o;
    {
      Enc e;
      encode_dec_prologue(e, c, erow);
      DecSync s; s.rdy = (__bridge void*)rdy; s.go = (__bridge void*)go; s.value_base = ev; s.mode = sync_mode;
      ev += 64;
      for (int l : layers) {
        encode_dec_attn_side(e, c, l, pos);
        encode_dec_moe_table(e, c, l, dsyn::exp_gu_type(l), dsyn::exp_down_type(l), tbl.sub((size_t)l * 144, 144), arena, s);
      }
      dec_bg_flush(e, c, -1);   // engram_bg: remaining chunks (layers not run here)
      e.wait();
    }
    std::vector<std::vector<uint8_t>> out;
    const Buffer hc = dec_cur_h(c);   // the current h (ping-pong variants end in B.h2 or B.h)
    out.emplace_back((uint8_t*)hc.cpu(), (uint8_t*)hc.cpu() + hc.size);
    for (size_t i = 0; i < mut.size(); i++) {
      if (mut[i].buf == B.h.buf || mut[i].buf == B.h2.buf ||
          mut[i].buf == B.plog.buf || mut[i].buf == S.part.buf || mut[i].buf == S.ml.buf || mut[i].buf == B.qrn.buf ||
          mut[i].buf == B.edeq.buf) { out.emplace_back(); continue; }   // scratch (fused variants may skip them)
      out.emplace_back((uint8_t*)mut[i].cpu(), (uint8_t*)mut[i].cpu() + mut[i].size);
    }
    return out;
  }
};

static DecOptions base_opts() { DecOptions o; o.pred_on = true; o.pred_depth = 1; o.pred_q8 = true; return o; }

static bool same(const std::vector<std::vector<uint8_t>>& a, const std::vector<std::vector<uint8_t>>& b, double* maxrel, int* nbuf) {
  bool eq = true; *maxrel = 0; *nbuf = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] == b[i]) continue;
    eq = false; (*nbuf)++;
    const float* x = (const float*)a[i].data(); const float* y = (const float*)b[i].data();
    for (size_t k = 0; k < a[i].size() / 4; k++)
      if (std::isfinite(x[k]) && std::fabs(x[k]) > 1e-20) *maxrel = std::max(*maxrel, std::fabs((double)x[k] - y[k]) / std::fabs(x[k]));
  }
  return eq;
}

static void test_layers(Env& E, const std::vector<int>& ls, const char* name) {
  E.layers = ls;
  struct V { const char* n; DecOptions o; int sync; };
  std::vector<V> vs;
  DecOptions b = base_opts();
  auto with = [&](auto f) { DecOptions o = b; f(o); return o; };
  vs.push_back({"base (repeat)", b, 2});
  vs.push_back({"reorder", with([](DecOptions& o) { o.reorder = true; }), 2});
  vs.push_back({"normqb", with([](DecOptions& o) { o.norm_qb = true; }), 2});
  vs.push_back({"normqb+reorder", with([](DecOptions& o) { o.norm_qb = o.reorder = true; }), 2});
  vs.push_back({"combine", with([](DecOptions& o) { o.attn_comb = true; }), 2});
  vs.push_back({"sgu_router=1", with([](DecOptions& o) { o.sgu_router = 1; }), 2});
  vs.push_back({"sgu_router=2", with([](DecOptions& o) { o.sgu_router = 2; }), 2});
  vs.push_back({"engram_bg=1", with([](DecOptions& o) { o.engram_bg = 1; }), 2});
  vs.push_back({"engram_bg=2", with([](DecOptions& o) { o.engram_bg = 2; }), 2});
  vs.push_back({"fuse(all)", with([](DecOptions& o) { o.fuse = o.reorder = o.norm_qb = o.attn_comb = true; o.engram_bg = 2; o.sgu_router = 1; }), 2});
  vs.push_back({"fuse(all) sgur=2", with([](DecOptions& o) { o.fuse = o.reorder = o.norm_qb = o.attn_comb = true; o.engram_bg = 2; o.sgu_router = 2; }), 2});
  vs.push_back({"fuse(all) sync=sgu", with([](DecOptions& o) { o.fuse = o.reorder = o.norm_qb = o.attn_comb = true; o.engram_bg = 2; o.sgu_router = 1; }), 1});
  vs.push_back({"hc_splits=2", with([](DecOptions& o) { o.hc_splits = 2; }), 2});
  vs.push_back({"hc_splits=3", with([](DecOptions& o) { o.hc_splits = 3; }), 2});
  vs.push_back({"hc_splits=4", with([](DecOptions& o) { o.hc_splits = 4; }), 2});
  vs.push_back({"fuse(all) hcs=2", with([](DecOptions& o) { o.fuse = o.reorder = o.norm_qb = o.attn_comb = true; o.engram_bg = 2; o.sgu_router = 2; o.hc_splits = 2; }), 2});
  vs.push_back({"moe shapes", with([](DecOptions& o) { o.moe_gu_nr = 1; o.moe_gu_rsg = 8; o.moe_d_nr = 2; o.moe_d_rsg = 4; }), 2});
  const auto ref = E.run(b, 2);
  const auto ref_sgu = E.run(b, 1);
  for (auto& v : vs) {
    const auto r = E.run(v.o, v.sync);
    double mr; int nb;
    const bool eq = same(v.sync == 1 ? ref_sgu : ref, r, &mr, &nb);
    printf("%-12s %-22s %s", name, v.n, eq ? "bit-identical\n" : "DIFFERENT");
    if (!eq) printf(" (%d buffers, max rel %.3g)\n", nb, mr);
    CHECK(eq, "%s %s not bit-identical", name, v.n);
  }
}

int main() {
  test_matvec_rmsnorm();
  Env E;
  E.setup();
  printf("synthetic weights %.2f GB\n", E.P.bytes / 1e9);
  test_layers(E, {1}, "layer 1");
  test_layers(E, {2}, "layer 2");
  test_layers(E, {5}, "layer 5");
  test_layers(E, {20}, "layer 20");
  test_layers(E, {24}, "layer 24");
  test_layers(E, {31}, "layer 31");
  test_layers(E, {0, 1, 2}, "layers 0-2");
  test_layers(E, {12, 13, 14}, "layers 12-14");
  printf(g_fail ? "FAILED (%d)\n" : "OK\n", g_fail);
  return g_fail ? 1 : 0;
}
