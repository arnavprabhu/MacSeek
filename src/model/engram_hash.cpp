#include "model/engram_hash.hpp"
#include <cstdio>

namespace ms {

bool EngramHashParams::init(const std::vector<int64_t>& tm, int64_t pad_token_id, const std::vector<int64_t>& lm,
                            const std::vector<int64_t>& hv, const std::vector<int64_t>& ho) {
  if (lm.size() != (size_t)kLayers * kMaxNgram || hv.size() != (size_t)kLayers * kCols ||
      ho.size() != (size_t)kLayers * kCols || pad_token_id < 0 || (size_t)pad_token_id >= tm.size()) {
    fprintf(stderr, "engram_hash: metadata size mismatch (mult %zu, primes %zu, offsets %zu)\n", lm.size(), hv.size(),
            ho.size());
    return false;
  }
  token_map.assign(tm.begin(), tm.end());
  pad_id = (int32_t)tm[pad_token_id];
  for (int l = 0; l < kLayers; l++)
    for (int s = 0; s < kMaxNgram; s++) mult[l][s] = (uint64_t)lm[l * kMaxNgram + s];
  for (int i = 0; i < kLayers * kCols; i++) { prime[i] = (uint64_t)hv[i]; offset[i] = (uint64_t)ho[i]; }
  return true;
}

void EngramHasher::set_history(const int32_t* last, int n_last, int64_t total_len) {
  st_ = State{};
  st_.pos = total_len;
  int m = n_last < 3 ? n_last : 3;
  for (int s = 0; s < m; s++) {
    int32_t t = last[n_last - 1 - s];
    st_.hist[s] = t < 0 ? kDead : p_->token_map[t];
  }
}

EngramHasher::State EngramHasher::hash_from(const State& from, const int32_t* tokens, int n, uint32_t* out,
                                            const bool* mask) const {
  using P = EngramHashParams;
  State st = from;
  for (int i = 0; i < n; i++) {
    int32_t cur = (mask && !mask[i]) ? kDead : p_->token_map[tokens[i]];
    // c[s] for s = 0..3 with the cumulative "blocked" rule of engram.py
    uint64_t c[P::kMaxNgram];
    bool blocked = false;
    for (int s = 0; s < P::kMaxNgram; s++) {
      int32_t src = s == 0 ? cur : st.hist[s - 1];
      blocked = blocked || src == kNone || src == kDead;
      c[s] = (uint64_t)(blocked ? p_->pad_id : src);
    }
    uint32_t* o = out + (size_t)i * kRowsPerToken;
    for (int l = 0; l < P::kLayers; l++) {
      uint64_t rolling = c[0] * p_->mult[l][0];
      for (int s = 1; s < P::kMaxNgram; s++) {
        rolling ^= c[s] * p_->mult[l][s];
        for (int k = 0; k < P::kHeads; k++) {
          int col = l * P::kCols + (s - 1) * P::kHeads + k;
          o[col] = (uint32_t)(rolling % p_->prime[col] + p_->offset[col]);
        }
      }
    }
    st.hist[2] = st.hist[1];
    st.hist[1] = st.hist[0];
    st.hist[0] = cur;
    st.pos++;
  }
  return st;
}

void EngramHasher::hash(const int32_t* tokens, int n, uint32_t* out, const bool* mask) {
  st_ = hash_from(st_, tokens, n, out, mask);
}

}  // namespace ms
