// Eviction policies for ExpertCache: LRU (O(1) intrusive list per class), LFU with exponential aging (scan of the
// class's resident slots, ~1-3 µs for ~2000 slots), Static = LRU for the dynamic part (the frequency-pinned set is
// handled by the cache's pin flags).
#include "store/experts.hpp"

#include <cmath>

namespace ms {

const char* policy_name(PolicyKind k) {
  switch (k) { case PolicyKind::LRU: return "lru"; case PolicyKind::LFU: return "lfu"; case PolicyKind::Static: return "static"; }
  return "?";
}

namespace {

class LruPolicy : public EvictionPolicy {
 public:
  void init(const std::vector<int>& cls, int n_classes, int) override {
    cls_ = cls; prev_.assign(cls.size(), -1); next_.assign(cls.size(), -1); in_.assign(cls.size(), 0);
    head_.assign(n_classes, -1); tail_.assign(n_classes, -1);
  }
  void on_insert(int s, uint32_t) override { touch(s); }
  void on_access(int s, uint32_t) override { touch(s); }
  void on_evict(int s, uint32_t) override { unlink(s); }
  int victim(int c, bool (*ok)(const void*, int), const void* ctx) override {
    for (int s = tail_[c]; s >= 0; s = prev_[s]) if (ok(ctx, s)) return s;
    return -1;
  }
 private:
  void unlink(int s) {
    if (!in_[s]) return;
    int c = cls_[s];
    if (prev_[s] >= 0) next_[prev_[s]] = next_[s]; else head_[c] = next_[s];
    if (next_[s] >= 0) prev_[next_[s]] = prev_[s]; else tail_[c] = prev_[s];
    prev_[s] = next_[s] = -1; in_[s] = 0;
  }
  void touch(int s) {   // move to MRU (head)
    unlink(s);
    int c = cls_[s];
    next_[s] = head_[c]; prev_[s] = -1;
    if (head_[c] >= 0) prev_[head_[c]] = s; else tail_[c] = s;
    head_[c] = s; in_[s] = 1;
  }
  std::vector<int> cls_, prev_, next_, head_, tail_;
  std::vector<uint8_t> in_;
};

class LfuPolicy : public EvictionPolicy {
 public:
  LfuPolicy(double decay, uint64_t period, int sample) : decay_(decay), period_(period ? period : 1), sample_(sample) {}
  void init(const std::vector<int>& cls, int n_classes, int n_keys) override {
    count_.assign(n_keys, 0.0); slot_key_.assign(cls.size(), ~0u); last_.assign(cls.size(), 0);
    members_.assign(n_classes, {});
    for (size_t s = 0; s < cls.size(); s++) members_[cls[s]].push_back((int)s);
  }
  void on_insert(int s, uint32_t k) override { slot_key_[s] = k; last_[s] = ++tick_; }
  void on_access(int s, uint32_t k) override {
    slot_key_[s] = k; last_[s] = ++tick_;
    count_[k] += 1.0;
    if (++accesses_ % period_ == 0) for (double& c : count_) c *= decay_;   // aging: old popularity fades
  }
  void on_evict(int s, uint32_t) override { slot_key_[s] = ~0u; }
  int victim(int c, bool (*ok)(const void*, int), const void* ctx) override {
    const auto& mem = members_[c];
    int best = -1; double bc = 0; uint64_t bl = 0;
    auto consider = [&](int s) {
      if (slot_key_[s] == ~0u || !ok(ctx, s)) return false;
      double v = count_[slot_key_[s]];
      if (best < 0 || v < bc || (v == bc && last_[s] < bl)) { best = s; bc = v; bl = last_[s]; }
      return true;
    };
    if (sample_ > 0 && (int)mem.size() > 2 * sample_) {
      // sampled LFU (Redis-style): least frequent of `sample_` random evictable candidates; O(sample) per miss
      int found = 0;
      for (int probe = 0; probe < 8 * sample_ && found < sample_; probe++) {
        rng_ ^= rng_ << 13; rng_ ^= rng_ >> 7; rng_ ^= rng_ << 17;
        found += consider(mem[rng_ % mem.size()]);
      }
      if (best >= 0) return best;
    }
    for (int s : mem) consider(s);
    return best;
  }
 private:
  double decay_; uint64_t period_, accesses_ = 0, tick_ = 0, rng_ = 0x9E3779B97F4A7C15ULL; int sample_;
  std::vector<double> count_;
  std::vector<uint32_t> slot_key_;
  std::vector<uint64_t> last_;
  std::vector<std::vector<int>> members_;
};

}  // namespace

std::unique_ptr<EvictionPolicy> make_policy(PolicyKind k, double decay, uint64_t period, int sample) {
  if (k == PolicyKind::LFU) return std::make_unique<LfuPolicy>(decay, period, sample);
  return std::make_unique<LruPolicy>();
}

}  // namespace ms
