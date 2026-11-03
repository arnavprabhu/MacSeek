// bench_qmatvec: decode matvec (n=1) and matvec_n timing on the real shapes, as % of measured read bandwidth.
// Weights are rotated over >= 512 MiB of copies so every call streams from DRAM (as in real decode).
// Usage: bench_qmatvec [--sweep] [--n N] [filter-substring]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/metal.hpp"
#include "kernels/quant.hpp"

using namespace ms;
using ms::q::QType;

struct Shape { QType t; int rows, cols, groups; const char* what; };

static double measure_bw() {
  auto& d = Device::get();
  size_t bytes = 1024ull << 20;
  Buffer src = d.alloc(bytes), out = d.alloc(1 << 20);
  uint32_t per_tg = 64 * 1024 / 16, ntg = (uint32_t)(bytes / (64 * 1024));
  std::vector<double> t;
  for (int i = 0; i < 20; i++) {
    Enc e; e.set_pipeline("q_bw_read_chunk"); e.set_buffer(0, src); e.set_buffer(1, out); e.set_value(2, per_tg);
    e.dispatch_groups(ntg, 1, 1, 256); e.wait(); if (i >= 2) t.push_back(e.gpu_ms());
  }
  std::sort(t.begin(), t.end());
  d.release(src); d.release(out);
  return bytes / t[0] / 1e6;
}

// MoE decode: 6 routed experts + shared, all type combos; arenas rotated over >= 512 MiB.
static int bench_moe(double bw, int reps, const char* filt) {
  auto& d = Device::get();
  struct Combo { QType gu, dn, sh; };
  std::vector<Combo> combos = {{QType::IQ1_S, QType::IQ1_S, QType::Q6_K}, {QType::IQ1_S, QType::IQ2_XXS, QType::Q8_0},
                          {QType::IQ1_S, QType::IQ2_XXS, QType::Q6_K}, {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q6_K},
                          // GSQ-RCO model (Q5_K shared expert)
                          {QType::IQ1_M, QType::IQ1_M, QType::Q5_K}, {QType::IQ2_XXS, QType::IQ2_XXS, QType::Q5_K},
                          {QType::IQ2_XS, QType::IQ2_XS, QType::Q5_K}, {QType::IQ1_M, QType::IQ2_XXS, QType::Q5_K},
                          {QType::IQ2_XS, QType::IQ1_M, QType::Q5_K}, {QType::IQ2_XXS, QType::IQ2_XS, QType::Q5_K}};
  if (filt && strchr(filt, ',')) {  // explicit combo "gu,down,shared" (lower-case type names)
    auto parse = [](std::string n) {
      for (QType t : {QType::IQ1_S, QType::IQ2_XXS, QType::IQ1_M, QType::IQ2_XS, QType::Q6_K, QType::Q8_0, QType::Q5_K}) {
        std::string tn = q::type_name(t);
        for (auto& ch : tn) ch = (char)tolower(ch);
        if (tn == n) return t;
      }
      fprintf(stderr, "unknown type %s\n", n.c_str()); exit(2);
    };
    std::string f = filt;
    const size_t c1 = f.find(','), c2 = f.find(',', c1 + 1);
    combos = {{parse(f.substr(0, c1)), parse(f.substr(c1 + 1, c2 - c1 - 1)), parse(f.substr(c2 + 1))}};
    filt = nullptr;
  }
  const int D = q::kExpertDim, I = q::kExpertInter;
  printf("MoE decode (6 routed + shared, 2 dispatches + barrier)\n");
  printf("%-8s %-8s %-6s %9s %8s %7s  %s\n", "gate/up", "down", "shared", "MB", "us", "GB/s", "%BW");
  for (const Combo& c : combos) {
    const std::string lab = std::string(q::type_name(c.gu)) + "/" + q::type_name(c.dn) + "+" + q::type_name(c.sh);
    if (filt && lab.find(filt) == std::string::npos) continue;
    const size_t gu = q::row_bytes(c.gu, D) * I, dn = q::row_bytes(c.dn, I) * D;
    const size_t sgu = q::row_bytes(c.sh, D) * I, sdn = q::row_bytes(c.sh, I) * D;
    const size_t per = 6 * (2 * gu + dn) + 2 * sgu + sdn;
    const int copies = (int)std::max<size_t>(2, (512ull << 20) / per);
    std::vector<Buffer> arenas(copies);
    std::vector<q::MoeDecodeArgs> args(copies);
    Buffer x = d.alloc(D * 4), y = d.alloc(D * 4), w = d.alloc(64), scratch = d.alloc(q::moe_decode_scratch_bytes());
    for (int k = 0; k < copies; k++) {
      arenas[k] = d.alloc(per);
      uint32_t* p = (uint32_t*)arenas[k].cpu(); uint32_t st = 777 + k;
      for (size_t i = 0; i < per / 4; i++) { st = st * 1664525u + 1013904223u; p[i] = st & 0x3bff3bffu; }
      q::MoeDecodeArgs& a = args[k];
      a.t_gate_up = c.gu; a.t_down = c.dn; a.t_shared = c.sh; a.n_routed = 6; a.has_shared = true;
      size_t off = 0;
      for (int e = 0; e < 6; e++) {
        a.slot[e].gate = arenas[k].sub(off, gu); off += gu;
        a.slot[e].up = arenas[k].sub(off, gu); off += gu;
        a.slot[e].down = arenas[k].sub(off, dn); off += dn;
      }
      a.shared.gate = arenas[k].sub(off, sgu); off += sgu;
      a.shared.up = arenas[k].sub(off, sgu); off += sgu;
      a.shared.down = arenas[k].sub(off, sdn);
      a.weights = w; a.x = x; a.y = y; a.scratch = scratch;
    }
    std::vector<double> t;
    for (int rep = 0; rep < reps; rep++) {
      Enc e;
      for (int k = 0; k < copies; k++) { q::encode_moe_decode(e, args[k]); e.barrier(); }
      e.wait();
      if (rep) t.push_back(e.gpu_ms() * 1e3 / copies);
    }
    std::sort(t.begin(), t.end());
    printf("%-8s %-8s %-6s %9.2f %8.1f %7.1f  %5.1f%%\n", q::type_name(c.gu), q::type_name(c.dn), q::type_name(c.sh),
           per / 1e6, t[0], per / t[0] / 1e3, 100.0 * per / t[0] / 1e3 / bw);
    for (auto& b : arenas) d.release(b);
    d.release(x); d.release(y); d.release(w); d.release(scratch);
  }
  return 0;
}

int main(int argc, char** argv) {
  bool sweep = false, moe = false; int n = 1; int reps = 7; const char* filt = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--sweep")) sweep = true;
    else if (!strcmp(argv[i], "--moe")) moe = true;
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
    else filt = argv[i];
  }
  auto& d = Device::get();
  const double bw = measure_bw();
  printf("measured read BW (chunk kernel, best): %.1f GB/s   n=%d\n", bw, n);
  const Shape shapes[] = {
      {QType::Q6_K, 1280, 5120, 1, "attn_q_a"},      {QType::Q5_K, 1280, 5120, 1, "attn_q_a"},
      {QType::Q6_K, 32768, 1280, 1, "attn_q_b"},     {QType::Q5_K, 32768, 1280, 1, "attn_q_b"},
      {QType::Q6_K, 512, 5120, 1, "attn_kv"},        {QType::Q6_K, 8192, 4096, 8, "attn_output_a"},
      {QType::Q6_K, 5120, 8192, 1, "attn_output_b"}, {QType::Q5_K, 5120, 8192, 1, "attn_output_b"},
      {QType::IQ1_S, 2304, 5120, 1, "exp gate/up"},  {QType::IQ1_S, 5120, 2304, 1, "exp down"},
      {QType::IQ2_XXS, 2304, 5120, 1, "exp gate/up"}, {QType::IQ2_XXS, 5120, 2304, 1, "exp down"},
      {QType::Q6_K, 2304, 5120, 1, "shexp gate/up"}, {QType::Q6_K, 5120, 2304, 1, "shexp down"},
      {QType::Q8_0, 2304, 5120, 1, "shexp gate/up"}, {QType::Q8_0, 5120, 2304, 1, "shexp down"},
      {QType::Q6_K, 129280, 5120, 1, "output head"}, {QType::Q6_K, 25600, 6144, 1, "engram_kv"},
      {QType::F32, 384, 5120, 1, "router"},          {QType::F32, 24, 20480, 1, "hc_fn"},
      {QType::BF16, 512, 5120, 1, "compressor_kv"},  {QType::BF16, 4096, 1280, 1, "indexer q_b"},
      {QType::F16, 5120, 5120, 1, "f16 synthetic"},
      {QType::IQ1_S, 32768, 5120, 1, "big synthetic"}, {QType::IQ2_XXS, 32768, 5120, 1, "big synthetic"},
      {QType::IQ1_M, 2304, 5120, 1, "exp gate/up"},  {QType::IQ1_M, 5120, 2304, 1, "exp down"},
      {QType::IQ2_XS, 2304, 5120, 1, "exp gate/up"}, {QType::IQ2_XS, 5120, 2304, 1, "exp down"},
      {QType::IQ1_M, 32768, 5120, 1, "big synthetic"}, {QType::IQ2_XS, 32768, 5120, 1, "big synthetic"},
      {QType::Q5_K, 2304, 5120, 1, "shexp gate/up"}, {QType::Q5_K, 5120, 2304, 1, "shexp down"},
  };
  if (moe) return bench_moe(bw, reps, filt);
  printf("%-8s %-15s %-16s %9s %8s %7s  %s\n", "type", "tensor", "shape", "MB", "us", "GB/s", "%BW");
  for (const Shape& s : shapes) {
    std::string lab = std::string(q::type_name(s.t)) + " " + s.what;
    if (filt && lab.find(filt) == std::string::npos) continue;
    const size_t wbytes = q::row_bytes(s.t, s.cols) * s.rows;
    const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes);
    std::vector<Buffer> W(copies);
    for (auto& b : W) {
      b = d.alloc(wbytes);
      uint32_t* p = (uint32_t*)b.cpu(); uint32_t st = 12345;
      for (size_t i = 0; i < wbytes / 4; i++) { st = st * 1664525u + 1013904223u; p[i] = st; }
      // keep fp16 scales finite: zero-cost for timing, garbage values are fine otherwise
    }
    Buffer x = d.alloc((size_t)s.cols * s.groups * 4 * n), y = d.alloc((size_t)s.rows * 4 * n);
    auto run = [&](q::MvCfg cfg) {
      std::vector<double> t;
      for (int rep = 0; rep < reps; rep++) {
        Enc e;
        for (int i = 0; i < copies; i++) {
          if (n == 1) q::encode_matvec(e, s.t, W[i], s.rows, s.cols, x, y, false, s.groups, cfg);
          else q::encode_matvec_n(e, s.t, W[i], s.rows, s.cols, x, y, n, false, s.groups);
          e.barrier();
        }
        e.wait();
        if (rep) t.push_back(e.gpu_ms() * 1e3 / copies);
      }
      std::sort(t.begin(), t.end());
      return t[0];  // min: other GPU users only ever slow a run down
    };
    double us = run({});
    char shape[32]; snprintf(shape, sizeof shape, "[%d->%d]%s", s.cols, s.rows, s.groups > 1 ? "g8" : "");
    printf("%-8s %-15s %-16s %9.2f %8.1f %7.1f  %5.1f%%\n", q::type_name(s.t), s.what, shape, wbytes / 1e6, us,
           wbytes / us / 1e3, 100.0 * wbytes / us / 1e3 / bw);
    if (sweep && n == 1) {
      double best = 1e30; q::MvCfg bc;
      for (int nr : {1, 2, 4}) for (int rsg : {1, 2, 4, 8}) for (int ks : {1, 2, 4, 8, 16, 32}) {
        if (rsg * ks > 32 || rsg * ks < 1) continue;
        double t = run({nr, rsg, ks});
        if (t < best) { best = t; bc = {nr, rsg, ks}; }
      }
      printf("    sweep best: nr=%d rsg=%d ks=%d  %.1f us  %.1f%%\n", bc.nr, bc.rsg, bc.ks, best,
             100.0 * wbytes / best / 1e3 / bw);
    }
    for (auto& b : W) d.release(b);
    d.release(x); d.release(y);
  }
  return 0;
}
