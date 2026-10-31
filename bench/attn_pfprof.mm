// bench_attn_pfprof: prefill indexer split (score vs merge) at depth 200K, n=2048
#include <algorithm>
#include <tuple>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/env.hpp"
#include "core/metal.hpp"
#include "kernels/attn.hpp"
using namespace ms;
namespace A = ms::attn;
struct IdxPfArgs { int start, n, ratio, kc0, kc, ld_s, use_mask, mask_words, cand_src, krows, qrows; };
struct MergeArgs { int n, k, cnt, ld_s, base, first; };
int main() {
  A::Options o = A::Options::from_env(); o.max_chunk = 2048;
  A::State st; A::init(st, o);
  auto& d = Device::get();
  for (int i = 0; i < 4; i++) { uint16_t* k = (uint16_t*)st.ikey[i].cpu(); for (size_t j = 0; j < st.ikey[i].size / 2; j++) k[j] = (uint16_t)(0x3000 + ((j * 2654435761u) >> 21) % 0x400) ^ ((j * 40503u) & 0x8000); }
  uint16_t* qi = (uint16_t*)st.Qi.cpu(); for (size_t j = 0; j < st.Qi.size / 2; j++) qi[j] = (uint16_t)(0x3800 ^ ((j * 2654435761u) & 0x8000));
  float* w = (float*)st.wq.cpu(); for (size_t j = 0; j < st.wq.size / 4; j++) w[j] = ((j * 7) % 13 - 6) * 0.1f;
  const int n = 2048, start = 200000, KC = o.key_chunk;
  auto tmin = [&](auto&& f) { double b = 1e9; for (int i = 0; i < 3; i++) { Enc e; f(e); e.wait(); b = std::min(b, e.gpu_ms()); } return b; };
  for (int cs : {0, 1}) {
    int len_max = start + n;
    double ts = tmin([&](Enc& e) {
      for (int kc0 = 0; kc0 < len_max; kc0 += KC) {
        int c = std::min(KC, len_max - kc0);
        IdxPfArgs a{start, n, 1, kc0, c, KC, 0, st.mask_words, cs, st.cap[3] + 64, (n + 1) / 2 * 64};
        e.set_pipeline("attn_idx_score_pf"); e.set_buffer(0, st.Qi); e.set_buffer(1, st.wq); e.set_buffer(2, st.ikey[3]);
        e.set_buffer(3, st.S); e.set_buffer(4, st.cand_mask); e.set_buffer(5, st.BS); e.set_value(6, a);
        e.dispatch_groups((c + 127) / 128, (n + 1) / 2, 1, 128); e.barrier();
      } });
    double tm = tmin([&](Enc& e) {
      for (int kc0 = 0; kc0 < len_max; kc0 += KC) {
        int c = std::min(KC, len_max - kc0);
        MergeArgs m{n, 512, c, KC, kc0, kc0 == 0};
        e.set_pipeline("attn_topk_merge"); e.set_buffer(0, st.S); e.set_buffer(1, st.Rk); e.set_buffer(2, st.Ri); e.set_buffer(3, st.Rc); e.set_value(4, m);
        e.dispatch_groups(n, 1, 1, 1024); e.barrier();
      } });
    double fl = 2.0 * n * 32 * 128 * len_max;
    printf("layer-20 shape cand_src=%d: score %.1f ms (%.1f TFLOP/s), merge(k=512) %.1f ms\n", cs, ts, fl / ts / 1e9, tm);
  }
  return 0;
}
