// Host encode functions of the quant module.
#include "kernels/quant.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "core/env.hpp"

namespace ms::q {

namespace {

struct QMvArgs {  // must match quant_common.h
  uint32_t rows, cols, row_bytes, rows_per_group, x_group_stride, accumulate, ksplit, n, x_stride, y_stride;
};

const char* kname(QType t) {
  switch (t) {
    case QType::F32: return "f32"; case QType::F16: return "f16"; case QType::BF16: return "bf16";
    case QType::Q8_0: return "q8_0"; case QType::Q5_K: return "q5_k"; case QType::Q6_K: return "q6_k";
    case QType::IQ1_S: return "iq1_s"; case QType::IQ2_XXS: return "iq2_xxs";
    default: return nullptr;
  }
}

// GPU block geometry (dense types use a virtual 256-element block)
int gpu_qk(QType t) { return block_elems(t) == 1 ? 256 : block_elems(t); }

[[noreturn]] void die(const char* what, QType t, int rows, int cols) {
  fprintf(stderr, "macseek quant: %s unsupported (type %s rows %d cols %d)\n", what, type_name(t), rows, cols);
  abort();
}

// Tuned launch table (bench_qmatvec sweep on M5 Pro). Small row counts get K split for occupancy.
MvCfg auto_cfg(QType t, int rows, int cols) {
  MvCfg c;
  c.nr = 2; c.rsg = 4; c.ks = 1;
  const long sgs = (long)rows / c.nr;  // simdgroups without K split
  if (sgs < 2048) c.ks = (int)std::min<long>(8, std::max<long>(1, 2048 / std::max<long>(sgs, 1)));
  while (c.ks > 1 && cols / gpu_qk(t) < (32 / 16) * c.ks) c.ks /= 2;
  if (c.ks > 1) c.rsg = std::max(1, 8 / c.ks);
  (void)t;
  return c;
}

}  // namespace

bool matvec_supported(QType t) { return kname(t) != nullptr; }

void encode_matvec(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& x, const Buffer& y,
                   bool accumulate, int groups, MvCfg cfg) {
  const char* kn = kname(t);
  if (!kn || cols % gpu_qk(t) != 0 || groups < 1 || rows % groups != 0) die("matvec", t, rows, cols);
  MvCfg a = auto_cfg(t, rows, cols);
  if (cfg.nr) a.nr = cfg.nr;
  if (cfg.rsg) a.rsg = cfg.rsg;
  if (cfg.ks) a.ks = cfg.ks;
  static const long env_nr = env_int("MACSEEK_MV_NR", 0), env_rsg = env_int("MACSEEK_MV_RSG", 0),
                    env_ks = env_int("MACSEEK_MV_KS", 0);
  if (env_nr) a.nr = (int)env_nr;
  if (env_rsg) a.rsg = (int)env_rsg;
  if (env_ks) a.ks = (int)env_ks;
  if (groups > 1) {  // every simdgroup's NR rows must stay inside one group
    while ((rows / groups) % a.nr) a.nr /= 2;
  }
  QMvArgs args{(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes(t, cols), (uint32_t)(rows / groups),
               (uint32_t)cols, accumulate ? 1u : 0u, (uint32_t)a.ks, 1, (uint32_t)cols, (uint32_t)rows};
  std::string fn = std::string("q_mv_") + kn + "_r" + std::to_string(a.nr);
  e.set_pipeline(fn.c_str());
  e.set_buffer(0, W); e.set_buffer(1, x); e.set_buffer(2, y);
  e.set_value(3, args);
  const int rows_per_tg = a.nr * a.rsg;
  e.set_threadgroup_memory(0, sizeof(float) * rows_per_tg * a.ks);
  e.dispatch_groups((rows + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * a.rsg * a.ks);
}

}  // namespace ms::q
