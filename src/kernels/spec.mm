// Host side of the speculative-verify kernels (spec.metal). See spec.hpp.
#include "kernels/spec.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "core/env.hpp"

namespace ms::sv {

using q::QType;

namespace {
struct QMvArgs {  // must match quant_common.h
  uint32_t rows, cols, row_bytes, rows_per_group, x_group_stride, accumulate, ksplit, n, x_stride, y_stride;
};
struct SvMoeArgs { uint32_t n, dim, inter, rb_gu, rb_d, ksplit; };

const char* kname(QType t) {
  switch (t) {
    case QType::F32: return "f32"; case QType::F16: return "f16"; case QType::BF16: return "bf16";
    case QType::Q8_0: return "q8_0"; case QType::Q5_K: return "q5_k"; case QType::Q6_K: return "q6_k";
    default: return nullptr;
  }
}
int nt_of(int k) { return k <= 2 ? 2 : k <= 4 ? 4 : 8; }
[[noreturn]] void die(const char* what, QType t, int a, int b) {
  fprintf(stderr, "macseek spec: %s unsupported (type %s, %d, %d)\n", what, q::type_name(t), a, b);
  abort();
}
}  // namespace

bool matvec_k_supported(QType t) { return kname(t) != nullptr; }

void encode_matvec_k(Enc& e, QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int k,
                     bool accumulate, int groups) {
  const char* kn = kname(t);
  if (!kn || k < 1 || k > kMaxK || groups < 1 || rows % groups != 0) die("matvec_k", t, rows, k);
  // the launch shape encode_matvec uses (auto rule + MACSEEK_MV_* overrides, groups rule): same per-lane arithmetic
  q::MvCfg a = q::matvec_cfg(t, rows, cols);
  // rows per simdgroup (nr) and row-simdgroups (rsg) do not change any row's arithmetic: the k-token kernel picks its
  // own (fewer rows = fewer registers per lane for k accumulator sets). Only the K split (ks) must match.
  static const long env_nr = env_int("MACSEEK_SV_NR", 0), env_rsg = env_int("MACSEEK_SV_RSG", 0);
  if (env_nr) a.nr = (int)env_nr;
  if (env_rsg) a.rsg = (int)env_rsg;
  if (groups > 1) while ((rows / groups) % a.nr) a.nr /= 2;
  const int nt = nt_of(k);
  QMvArgs args{(uint32_t)rows, (uint32_t)cols, (uint32_t)q::row_bytes(t, cols), (uint32_t)(rows / groups), (uint32_t)cols,
               accumulate ? 1u : 0u, (uint32_t)a.ks, (uint32_t)k, (uint32_t)(cols * groups), (uint32_t)rows};
  std::string fn = std::string("sv_mvk_") + kn + "_r" + std::to_string(a.nr) + "_n" + std::to_string(nt);
  e.set_pipeline(fn.c_str());
  e.set_buffer(0, W); e.set_buffer(1, X); e.set_buffer(2, Y);
  e.set_value(3, args);
  const int rows_per_tg = a.nr * a.rsg;
  e.set_threadgroup_memory(0, std::max<size_t>(16, sizeof(float) * rows_per_tg * a.ks * nt));
  e.dispatch_groups((rows + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * a.rsg * a.ks);
}

void encode_moe_shared_k(Enc& e, QType ts, const q::ExpertSlot& sh, const Buffer& X, const Buffer& H, const Buffer& Y,
                         int k) {
  const char* kn = (ts == QType::Q6_K || ts == QType::Q8_0 || ts == QType::Q5_K) ? kname(ts) : nullptr;
  if (!kn || k < 1 || k > kMaxK) die("moe_shared_k", ts, k, 0);
  // decode launch parameters (quant.mm encode_moe_decode_table_part, shared-only parts): only the K split changes
  // the summation grouping; nr / rsg do not
  static const long gu_ks = env_int("MACSEEK_MOE_GU_KS", 1), d_ks = env_int("MACSEEK_MOE_D_KS", 1);
  const int nt = nt_of(k);
  const int D = q::kExpertDim, I = q::kExpertInter;
  SvMoeArgs args{(uint32_t)k, (uint32_t)D, (uint32_t)I, (uint32_t)q::row_bytes(ts, D), (uint32_t)q::row_bytes(ts, I), 1};
  {
    const int nr = 2, rsg = 4, ks = (int)gu_ks, rows_per_tg = nr * rsg;
    args.ksplit = ks;
    std::string fn = std::string("sv_moe_gu_s_") + kn + "_r" + std::to_string(nr) + "_n" + std::to_string(nt);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, X); e.set_buffer(1, H); e.set_value(2, args); e.set_buffer(3, sh.gate); e.set_buffer(4, sh.up);
    e.set_threadgroup_memory(0, std::max<size_t>(16, sizeof(float) * rows_per_tg * 2 * ks * nt));
    e.dispatch_groups((I + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * rsg * ks);
  }
  e.barrier();
  {
    const int nr = 4, rsg = 2, ks = (int)d_ks, rows_per_tg = nr * rsg;
    args.ksplit = ks;
    std::string fn = std::string("sv_moe_down_s_") + kn + "_r" + std::to_string(nr) + "_n" + std::to_string(nt);
    e.set_pipeline(fn.c_str());
    e.set_buffer(0, H); e.set_buffer(1, Y); e.set_value(2, args); e.set_buffer(3, sh.down);
    e.set_threadgroup_memory(0, std::max<size_t>(16, sizeof(float) * rows_per_tg * ks * nt));
    e.dispatch_groups((D + rows_per_tg - 1) / rows_per_tg, 1, 1, 32 * rsg * ks);
  }
}

}  // namespace ms::sv
