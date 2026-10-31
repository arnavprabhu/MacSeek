// bench_attn_prof: per-kernel timings for tuning (decode indexer score / top-k / sparse attention).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/env.hpp"
#include "core/metal.hpp"
#include "kernels/attn.hpp"
using namespace ms;
namespace A = ms::attn;
template <class F> static double tmed(int iters, F&& f) {
  std::vector<double> t;
  const int reps = 20;
  for (int i = 0; i < iters; i++) { Enc e; for (int r = 0; r < reps; r++) f(e); e.wait(); t.push_back(e.gpu_ms() / reps); }
  std::sort(t.begin(), t.end()); return t[0];
}
int main() {
  auto& d = Device::get();
  A::Options o = A::Options::from_env(); o.max_chunk = 16;
  A::State st; A::init(st, o);
  for (int i = 0; i < 4; i++) { uint16_t* k = (uint16_t*)st.ikey[i].cpu(); for (size_t j = 0; j < st.ikey[i].size / 2; j++) k[j] = 0x3000 + (j * 2654435761u >> 20) % 0x800; }
  for (int i = 0; i < 4; i++) { uint16_t* k = (uint16_t*)st.ckv[i].cpu(); for (size_t j = 0; j < st.ckv[i].size / 2; j++) k[j] = 0x3000 + (j * 2654435761u >> 20) % 0x800; }
  Buffer q = d.alloc(64 * 512 * 4), iq = d.alloc(4096 * 4), iw = d.alloc(128), sinks = d.alloc(256), out = d.alloc(64 * 512 * 4);
  for (int i = 0; i < 4096; i++) ((float*)iq.cpu())[i] = (i % 7) - 3;
  for (int i = 0; i < 32; i++) ((float*)iw.cpu())[i] = (i % 5) - 2;
  for (int i = 0; i < 64 * 512; i++) ((float*)q.cpu())[i] = ((i * 37) % 11 - 5) * 0.02f;
  const int pos = 200000;
  int* tk = (int*)st.topk.cpu(); for (int i = 0; i < 512; i++) tk[i] = i * (int)env_int("TOPK_STRIDE", 97);
  const int reps = (int)env_int("REPS", 20);
  // sparse attention, one layer, N layers in one cb
  for (int nl : {1, 10}) {
    double t = tmed(9, [&](Enc& e) { for (int i = 0; i < nl; i++) A::encode_sparse(e, st, 3, q, sinks, out, pos, 1); });
    printf("sparse decode x%d (ratio>0, 640 rows): %.3f ms  (%.1f us/layer)\n", nl, t, t * 1e3 / nl);
    t = tmed(9, [&](Enc& e) { for (int i = 0; i < nl; i++) A::encode_sparse(e, st, 0, q, sinks, out, pos, 1); });
    printf("sparse decode x%d (ratio 0, 128 rows): %.3f ms  (%.1f us/layer)\n", nl, t, t * 1e3 / nl);
  }
  // empty command buffer / single barrier overhead
  printf("empty cb: %.3f ms\n", tmed(9, [&](Enc& e) { e.barrier(); }));
  // indexer decode parts
  for (int L : {2, 20, 24}) {
    double t = tmed(reps, [&](Enc& e) { A::encode_index_decode(e, st, L, iq, iw, pos); });
    printf("index decode layer %d total: %.3f ms\n", L, t);
  }
  // score kernel alone / topk alone, layer 20 shapes
  struct IdxDecArgs { int pos, len, nvirt, list_mode, actq, tiles_per_tg, cand_src; };
  struct TopkArgs { int n, k, list_mode; };
  for (int len : {100000, 200001}) for (int tpt : {1, 2, 4, 8, 16}) {
    IdxDecArgs a{pos, len, len, 0, 1, tpt, 0};
    int tiles = (len + 63) / 64;
    double t = tmed(9, [&](Enc& e) {
      e.set_pipeline("attn_idx_score_dec");
      e.set_buffer(0, iq); e.set_buffer(1, iw); e.set_buffer(2, st.rope_yarn); e.set_buffer(3, st.ikey[3]);
      e.set_buffer(4, st.cand_list); e.set_buffer(5, st.dec_scores); e.set_buffer(6, st.dec_bscores); e.set_value(7, a);
      e.dispatch_groups((tiles + tpt - 1) / tpt, 1, 1, 128); e.barrier(); });
    printf("score len=%d tiles/tg=%d: %.3f ms  (%.0f GB/s keys)\n", len, tpt, t, len * 256.0 / t / 1e6);
  }
  struct IdxDec2Args { int pos, len, nvirt, actq, tiles_per_tg, cand_src, krows; };
  for (int len : {200001}) for (int tpt : {4, 8, 16, 32, 64}) for (int cs : {0, 1}) {
    IdxDec2Args a{pos, len, len, 1, tpt, cs, st.cap[3] + 64};
    int tiles = (len + 63) / 64;
    double t = tmed(9, [&](Enc& e) {
      e.set_pipeline("attn_idx_score_dec2");
      e.set_buffer(0, iq); e.set_buffer(1, iw); e.set_buffer(2, st.rope_yarn); e.set_buffer(3, st.ikey[3]);
      e.set_buffer(4, st.dec_scores); e.set_buffer(5, st.dec_bscores); e.set_buffer(6, st.dec_hist); e.set_buffer(7, st.dec_hist.sub(16384, 16384)); e.set_value(8, a);
      e.dispatch_groups((tiles + tpt - 1) / tpt, 1, 1, 128); e.barrier(); });
    printf("score2 (MPP) len=%d tiles/tg=%d cand=%d: %.3f ms  (%.0f GB/s keys)\n", len, tpt, cs, t, len * 256.0 / t / 1e6);
  }
  for (int n : {100000, 200001, 25001}) {
    TopkArgs a{n, n == 25001 ? 2048 : 512, 0};
    double t = tmed(9, [&](Enc& e) {
      e.set_pipeline("attn_topk_dec"); e.set_buffer(0, st.dec_scores); e.set_buffer(1, st.cand_list); e.set_buffer(2, st.topk);
      e.set_value(3, a); e.dispatch_groups(1, 1, 1, 1024); e.barrier(); });
    printf("topk n=%d k=%d: %.3f ms\n", n, a.k, t);
  }
  return 0;
}
