#pragma once
// Attention module host API (MacSeek). See docs/modules/attn.md for layouts, semantics and numbers.
// All encode_* functions only encode into the given Enc (no commit / wait). They insert the barriers they need
// internally and end with a barrier, so outputs are safe to consume by the next dispatch.
#include <cstdint>
#include <functional>
#include <vector>
#include "core/metal.hpp"

namespace ms::attn {

constexpr int kLayers = 40, kD = 512, kRD = 64, kH = 64, kWin = 128;
constexpr int kIH = 32, kID = 128, kTopk = 512, kCandBlocks = 2048, kCandBS = 8;
constexpr int kRowPad = 128; // extra rows allocated past each cache capacity (128-key tile over-read)

// ---- layer roles (DeepSeek-V4.1-Flash) ----
inline int ratio_of(int L) { return L < 2 ? 0 : (L < 20 ? 2 : 1); }
inline bool is_kv_source(int L) { return L == 2 || L == 8 || L == 14 || L == 20; }
inline int kv_source_of(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
inline int src_slot(int s) { return s == 2 ? 0 : s == 8 ? 1 : s == 14 ? 2 : 3; }
inline bool is_index_source(int L) { return is_kv_source(L) || L == 24 || L == 28 || L == 32 || L == 36; }
inline int index_source_of(int L) { if (L < 20) return kv_source_of(L); return L < 24 ? 20 : 24 + 4 * ((L - 24) / 4); }
inline bool is_cand_source(int L) { return L == 20; }
inline bool uses_cand(int L) { return L > 20 && is_index_source(L); }

struct Options {
  bool actq = true;        // fp8/fp4 activation quant-dequant simulation (reference behaviour). env MACSEEK_ACTQ
  int max_pos = 262144;    // positions supported: rope tables + compressed-cache capacity. env MACSEEK_MAX_POS
  int max_chunk = 2048;    // max prefill chunk (n)
  int key_chunk = 16384;   // prefill indexer key chunk (multiple of 64). env MACSEEK_ATTN_KCHUNK
  int dec_rows_split = 64; // decode split-K rows per threadgroup (multiple of 16). env MACSEEK_ATTN_DEC_SPLIT
  int dec_heads_tg = 8;    // heads per sparse-attention threadgroup, decode (8|16). env MACSEEK_ATTN_DEC_HEADS
  bool dec_prefetch = false;  // decode sparse attention (8 heads): next 16-row chunk loaded into registers during the
                              // current chunk's math (bit-identical; measured no gain at 262K). env MACSEEK_ATTN_DEC_PF
  bool dec_fast_index = true;  // tensor-op decode scorer + histogram/compaction top-k (0: simdgroup scorer +
                               // single-threadgroup radix select). env MACSEEK_ATTN_DEC_FAST
  int dec_cand_cap = 32768;    // candidate capacity of the decode compaction (overflow -> exact fallback)
  bool pf_cand_union = true;  // prefill layers 24..36 score only the per-tile union of candidate blocks (0: dense
                              // scoring of all keys + mask). env MACSEEK_ATTN_PF_UNION
  bool pf_mpp = true;      // prefill sparse attention on the tensor units, two-pass (scores, softmax, P.V)
                           // (0: flash-style simdgroup kernel). env MACSEEK_ATTN_PF_MPP
  bool dec_mpp = false;    // decode through the two-pass tensor path too (measured slower: 3 dependent dispatches)
  int pf_qblock = 512;     // queries per two-pass block (bounds scratch: ~0.3 MB per query)
  bool pf_score_reg = true;  // prefill indexer scorer with the head sum in registers (cooperative-tensor layout checked
                             // at init; 0 or layout mismatch: threadgroup-memory epilogue). env MACSEEK_ATTN_PF_SCORE_REG
  bool pf_fold_softmax = true;  // two-pass prefill attention with the softmax folded into both passes (tile max/sum,
                                // P = exp(S - m_tile) half; no f32 score matrix / softmax dispatch). env MACSEEK_ATTN_PF_FOLD
  bool pf_merge_filter = true;  // prefill running top-k merge (k=512): only chunk elements above the running minimum
                                // are compacted and merged (exact). env MACSEEK_ATTN_PF_MFILTER
  int pf2_qblock = 64;   // folded path: queries per pipelined block (2 scratch sets of ~150 KB/query). env MACSEEK_ATTN_PF2_QBLOCK
  int pf_idx_qblock = 256;  // prefill candidate-union layers: score/merge per block of this many queries, 2 blocks in
                            // flight (score rows stay cache-resident; 0: whole chunk). env MACSEEK_ATTN_PF_IDX_QBLOCK
  int pf_heads_tg = 8;     // heads per sparse-attention threadgroup, prefill (8: 128 threads | 16: 256 threads). env MACSEEK_ATTN_PF_HEADS
  static Options from_env();
};

// All attention-side state + scratch. Layouts (all row-major, little endian):
//   rope_plain / rope_yarn : float2 [max_pos][32] (cos, sin) — theta 1e4 (layers 0,1) / theta 1.6e5 + YaRN (layers 2..39)
//   win[L]                 : half [128][512]   window KV ring, slot = pos % 128 (fp8-sim values, exact in f16)
//   ckv[s]                 : half [cap_s + 128][512] compressed KV of source s (slot 0..3 = layer 2,8,14,20), row j = group j
//   ikey[s]                : half [cap_s + 128][128] index keys (fp4-sim values)
//   cst[s]                 : f32 [2][ratio][512] compressor partial group (kv then gate score), slot = pos % ratio
// cap_s = max_pos / ratio_s. Only rows < (pos+1)/ratio are meaningful.
struct State {
  Options opt;
  Buffer rope_plain, rope_yarn;
  Buffer win[kLayers];
  Buffer ckv[4], ikey[4], cst[4];
  int cap[4] = {0, 0, 0, 0};
  int mask_words = 0;          // candidate bitmap words per query = ceil(max_pos/8/32)
  // ---- scratch ----
  Buffer chunk_kv;             // half [max_chunk][512]
  Buffer part, ml;             // f32 [splits][64][512], float2 [splits][64]
  Buffer dec_scores, dec_bscores;
  Buffer dec_hist, dec_ctl, dec_sel, dec_cand;  // [2] problems (positions, blocks): uint[4096], uint[4], int[2048], uint2[cap]
  Buffer topk;                 // int [max_chunk][512]: current shared top-k (row q = query q); decode uses row 0
  Buffer cand_list;            // int [2048]: decode candidate blocks from layer 20 (sorted, -1 pad)
  Buffer cand_mask;            // uint [max_chunk][mask_words]: prefill candidate block bitmap from layer 20
  Buffer Qi, wq;               // half [round2(max_chunk)][32][128], f32 [max_chunk][32]
  Buffer S, BS;                // f32 [max_chunk][key_chunk], [max_chunk][key_chunk/8]
  Buffer Rk, Ri, Rc, Bk, Bi, Bc;  // running top-k (positions k=512, blocks k=2048) per query
  Buffer pfQ, pfS, pfP;   // prefill two-pass scratch: half [qblock][64][512], f32 / half [qblock][64][640]
  bool pf_score_reg_ok = false;
  bool pf_fold_ok = false;        // folded-softmax attention usable (option on + layout verified at init)
  Buffer pfML;                    // float2 [qblock][R/64][64] per-tile (max, sum) for the folded-softmax path   // register-epilogue scorer usable (option on + layout verified at init)
  Buffer ulist, ucount;           // prefill: union of candidate blocks per 2-query tile, int [n/2][4096], int [n/2]
  Buffer dec_cnt;                 // decode fused combine: uint [16] last-threadgroup counters (self-resetting)
};
void init(State& s, const Options& o);
// Buffers that make up the persistent attention state after `pos` tokens (for save/load snapshots), with the number
// of meaningful bytes from the start of each buffer: 40 window rings, 4 x (compressed KV rows, index-key rows,
// compressor partial state). Everything else in State is scratch or derived (rope tables).
struct StateSpan { Buffer buf; size_t bytes; };
std::vector<StateSpan> persistent_state(const State& s, int pos);
void reset(State& s);  // zero caches + compressor state (new sequence)

// Per-layer inputs/outputs. All f32. n = tokens in the call (1 = decode).
struct LayerIO {
  int layer = 0;
  Buffer q;          // [n][64][512]  wq_b(q_norm(wq_a x)), pre-rope
  Buffer kv;         // [n][512]      wkv x (pre-norm)
  Buffer kv_norm;    // [512]         attn_kv_a_norm weight
  Buffer sinks;      // [64]
  Buffer out;        // [n][64][512]  attention output after inverse rope (before wo_a / its fp8 sim)
  // kv source layers (2, 8, 14, 20):
  Buffer c_kv, c_gate;  // [n][512] attn_compressor_kv x, attn_compressor_gate x (gate: ratio-2 only)
  Buffer c_norm;        // [512] attn_compressor_norm
  Buffer idx_wk;        // bf16 [128][512] indexer.attn_k (raw GGUF bytes)
  Buffer idx_knorm;     // [128] indexer.k_norm
  // index source layers (2, 8, 14, 20, 24, 28, 32, 36):
  Buffer idx_q;         // [n][32][128] indexer.attn_q_b(qr), pre-rope
  Buffer idx_w;         // [n][32]      indexer.proj(x), unscaled
};
// Full attention side of one layer for tokens at positions start..start+n-1 (appended after the existing state).
void encode_layer(Enc& e, State& s, const LayerIO& io, int start, int n);

// ---- building blocks (used by encode_layer; exposed for tests / benchmarks) ----
void encode_kv_window(Enc& e, State& s, int layer, const Buffer& kv, const Buffer& norm_w, int start, int n);
void encode_compress(Enc& e, State& s, int layer, const LayerIO& io, int start, int n, bool barrier_after = true);
// hook (optional): called inside the latency-bound stages (after the compaction and after the final select dispatches,
// before their barriers) so a caller can put independent background work next to them (stage 0, 1).
using StageHook = std::function<void(int stage)>;
void encode_index_decode(Enc& e, State& s, int layer, const Buffer& iq, const Buffer& iw, int pos,
                         const StageHook& hook = {});
void encode_index_prefill(Enc& e, State& s, int layer, const Buffer& iq, const Buffer& iw, int start, int n);
// fused_combine (decode only): the split-K combine runs in the last threadgroup of each head group (one dispatch,
// bit-identical output; MACSEEK_DEC_COMBINE via ds41 DecOptions).
// hook (decode): called after the sparse dispatch (stage 2) and after the combine dispatch (stage 3, unfused only),
// before their barriers.
void encode_sparse(Enc& e, State& s, int layer, const Buffer& q, const Buffer& sinks, const Buffer& out, int start, int n,
                   bool fused_combine = false, const StageHook& hook = {});
void encode_window_commit(Enc& e, State& s, int layer, int start, int n);
// in-place quant-dequant simulation on f32 x[n] (n multiple of the block): mode 0 fp8 e4m3/ue8m0-32,
// 1 fp4 e2m1/ue8m0-32, 2 fp4 e2m1/e4m3-16
void encode_quant_sim(Enc& e, const Buffer& x, uint32_t n, int mode);

// rope tables on the CPU (float2 [max_pos][32])
void make_rope_table(float* dst, int max_pos, bool yarn);

}  // namespace ms::attn
