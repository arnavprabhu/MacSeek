#pragma once
// Private: Model::Impl (activation buffers, layer weights, attention state) shared by ds41.mm and ds41_verify.mm.
#include <memory>
#include <vector>
#include "kernels/arch.hpp"
#include "kernels/attn.hpp"
#include "model/dec_layer.hpp"
#include "model/ds41.hpp"
#include "store/engram.hpp"

namespace ms::ds41 {

struct VerifyState;   // ds41_verify.mm


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
  Buffer zall, route_all;            // prefill ZALL mode: Z rows of the whole layer [6n routed | n shared], global routing
  std::vector<std::unique_ptr<Enc>> moe_inflight;   // prefill MoE group command buffers not yet waited for
  Buffer pids, pw;                   // (unused)
  Buffer plog;                       // decode: router logits of layers l+1..l+D applied to layer l's FFN input
  Buffer plog_n;
  Buffer dbg, dbg2;                        // MACSEEK_DBG_DUMP: per-layer FFN input x (after attention) / layer output x
  Buffer erow_b[2]; int erow_cur = 0;   // engram row buffers: current chunk / next chunk prefetch
  struct EPend { bool active = false; const int32_t* toks = nullptr; int n = 0, pos = 0, b = 0; EngramFetcher::Ticket t0, t1; } epend;
  Buffer tbl;                        // decode table mode: [40][6][3] device addresses of the routed experts                     // prefill: router logits of layer l+1 on layer l's FFN input, [n][384]
  std::unique_ptr<VerifyState> vs;   // speculative verify buffers (lazily created)
  Impl();
  ~Impl();
  arch::Scratch asc2;                // second scratch for the predictor router (same kernel family)   // prefill MoE scratch + routing tables (double-buffered)
};


}  // namespace ms::ds41
