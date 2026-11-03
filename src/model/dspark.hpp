#pragma once
// dspark block drafter for DeepSeek-V4.1 (GGUF arch "dflash", 3 DSV4 stages, block 5, Markov head), mirroring
// llama.cpp src/models/dflash.cpp graph_dsv41 + common/speculative.cpp (draft-dspark, sample_from_anchor):
//   * injection: target features f(p) = [mean_hc h_37(p) | mean_hc h_38(p) | mean_hc h_39(p)] (the attention inputs of
//     target layers 37..39) -> main_x = rmsnorm(fc f) -> per stage kv = rope(rmsnorm(wkv main_x)) into a 128-row ring
//   * draft(anchor at position P, B tokens): block [anchor, mask x (B-1)] at positions P..P+B-1 through the 3 stages
//     (hyper-connections, MLA attention over ring rows of positions [P-128, P-1] + all block rows, non-causal, sinks;
//     MoE 128 experts top-3 MXFP4 + shared Q8_0), collapse, output_norm, the TARGET's output head, then the Markov
//     chain: d_i = argmax(logits_i + w2 . w1[d_{i-1}]), d_{-1} = anchor.  -> B drafted tokens for positions P+1..P+B.
// Weights are read in place from the draft GGUF (dense via Weights, experts preaded into wired memory).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "core/metal.hpp"
#include "model/dec_layer.hpp"

namespace ms {
class Gguf; class Weights; class TokenEmbd;
}

namespace ms::ds41 {

struct DSparkTarget {        // what the drafter borrows from the target model
  W head;                    // output.weight
  TokenEmbd* te = nullptr;   // token_embd rows
  Buffer rope_plain;         // float2 [max_pos][32], theta 10000, no YaRN (the draft's rope)
};

class DSpark {
 public:
  DSpark(const std::string& path, const DSparkTarget& tgt);
  ~DSpark();
  static std::string default_path();   // $MACSEEK_DRAFT, else dspark-*.gguf next to the model / known location
  int block_size() const { return block_; }
  uint64_t bytes() const { return bytes_; }   // resident bytes (dense + experts)

  // Target features of position pos (host [3][5120] in target-layer order); written into the ring on the next draft.
  void add_feature(int pos, const float* f3x5120);
  // Draft B (1..block_size) tokens after `anchor` at position P. Returns the GPU+CPU time in ms.
  double draft(int32_t anchor, int P, int B, int32_t* out);
  void reset();   // forget the ring (new sequence)

 private:
  struct L3 { W attn_norm, ffn_norm, q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b, sinks, gate_inp, bias, sh_gate, sh_up, sh_down,
              hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
              Buffer gate_exps, up_exps, down_exps; };
  std::unique_ptr<Gguf> g_;
  std::unique_ptr<Weights> w_;
  DSparkTarget t_;
  std::vector<L3> L_;
  W fc_, enc_norm_, out_norm_, mk_w1_, mk_w2_;
  int block_ = 5, n_exp_ = 128, k_exp_ = 3, noise_ = 128799;
  float scale_ = 1.5f, eps_ = 1e-20f;
  uint64_t bytes_ = 0;
  void* xmem_ = nullptr; size_t xmem_bytes_ = 0;
  // ring: per stage half [128][512]; slot -> position (-1 empty); pending host features to inject
  Buffer ring_[3], slot_pos_;
  std::vector<int> slot_host_;
  std::vector<std::pair<int, std::vector<float>>> pend_;
  // scratch
  Buffer feat_, mainx_, kvtmp_, embrows_, h_, x_, sub_, mixA_, mixF_, qa_, qrn_, q_, kvb_, blk_, aout_, oa_, rlog_, rids_, rw_,
      hx_, shg_, shu_, sha_, logits_, amax_i_, amax_v_, mk_in_;
  std::vector<int32_t> noise_row_;
};

}  // namespace ms::ds41
