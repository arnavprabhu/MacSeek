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

int main(int argc, char** argv) {
  bool sweep = false; int n = 1; int reps = 7; const char* filt = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--sweep")) sweep = true;
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
  };
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
