#pragma once
// Resident dense weights: every tensor except the streamed ones (*_exps -> ExpertCache, per_layer_token_embd ->
// EngramFetcher; token_embd optional, default NOT resident -> TokenEmbd) is read with big parallel F_NOCACHE preads
// into one page-aligned anonymous arena, wrapped as no-copy shared MTLBuffers (<= maxBufferLength each).
// The arena mirrors the file layout of each contiguous run (dst = file offset mod 16 KiB), so every read is
// 16 KiB-aligned on both sides and a tensor keeps its 32-byte GGUF alignment.
#include "core/metal.hpp"
#include "gguf/gguf.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ms {

struct WTensor {
  std::string name;
  GType type = GType::F32;
  int n_dims = 0;
  int64_t ne[4] = {1, 1, 1, 1};
  uint64_t nbytes = 0;
  Buffer buf;                       // bind buf.buf at buf.off; buf.size == nbytes
  void* host = nullptr;             // CPU pointer to the same bytes
  const TensorInfo* info = nullptr; // source tensor in the GGUF
  bool repacked = false;
};

struct WeightsOptions {
  bool token_embd_resident = false;   // env MACSEEK_EMBD_RESIDENT=1
  int threads = 8;                    // pread threads (env MACSEEK_WEIGHTS_THREADS)
  size_t read_chunk = 8 << 20;        // bytes per pread
  bool mlock = true;                  // env MACSEEK_WEIGHTS_MLOCK (default 1): wire the arena before reading. Measured:
                                      // without it macOS compressed ~5 of 6.2 GB of fresh weights within a second
  bool nocache = true;                // use the F_NOCACHE fds (no page-cache double copy)
  bool prefault = false;              // env MACSEEK_WEIGHTS_PREFAULT=1: touch the arena before reading (if !mlock).
                                      // Measured worse (2.8 vs 6.7 GB/s): idle zero pages get compressed, then refaulted
  size_t max_buffer_bytes = 0;        // 0 = min(device maxBufferLength, 16 GiB)
  std::function<bool(const TensorInfo&)> filter;   // optional extra filter (tests load a subset)
  static WeightsOptions from_env();
};

class Weights {
 public:
  explicit Weights(const Gguf& g, const WeightsOptions& o = WeightsOptions::from_env());
  ~Weights();
  Weights(const Weights&) = delete; Weights& operator=(const Weights&) = delete;

  static bool is_streamed(const std::string& name);   // *_exps or per_layer_token_embd

  const WTensor* find(const std::string& name) const;
  const WTensor& at(const std::string& name) const;   // aborts if missing
  const std::vector<WTensor>& all() const { return tensors_; }
  bool has_token_embd() const { return find("token_embd.weight") != nullptr; }

  // Repack hook: called once per tensor (in load order). Return true after filling `out` (buf/host/type/ne/nbytes;
  // the callee owns `out.buf`, e.g. Device::alloc) to replace the tensor; the original arena pages are then returned
  // to the OS (madvise FREE_REUSABLE on whole interior pages) when release_src is true.
  using RepackFn = std::function<bool(const WTensor& src, WTensor& out)>;
  size_t repack(const RepackFn& fn, bool release_src = true);

  std::vector<Buffer> buffers() const { return bufs_; }   // arena MTLBuffers (residency sets / useResource)
  uint64_t bytes() const { return bytes_; }              // tensor bytes loaded
  uint64_t bytes_read() const { return bytes_read_; }    // bytes actually read (incl. 16K rounding)
  uint64_t arena_bytes() const { return arena_size_; }
  double load_ms() const { return load_ms_; }            // total: mlock (if on) + read + wrap
  double lock_ms() const { return lock_ms_; }
  double gbps() const { return load_ms_ > 0 ? bytes_read_ / (load_ms_ * 1e6) : 0; }
  bool locked() const { return locked_; }

 private:
  std::vector<WTensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
  std::vector<Buffer> bufs_;
  char* arena_ = nullptr;
  uint64_t arena_size_ = 0, bytes_ = 0, bytes_read_ = 0;
  double load_ms_ = 0, lock_ms_ = 0;
  bool locked_ = false;
};

// token_embd row reader. Resident mode: memcpy from Weights. Default: pread on demand through the page-cached fd
// (a decode step reads 1 row of 4200 B, ~0.05-0.1 ms off the critical path if issued right after sampling; hot
// tokens stay in the OS page cache, which is reclaimable, unlike 518 MiB pinned in our arena).
class IoPool;
class TokenEmbd {
 public:
  TokenEmbd(const Gguf& g, const Weights* w = nullptr, int threads = 8);
  ~TokenEmbd();
  uint64_t row_bytes() const { return row_bytes_; }
  GType type() const { return type_; }
  int64_t n_rows() const { return n_rows_; }
  void fetch(const int32_t* toks, size_t n, void* dst);   // dst[i*row_bytes] = row toks[i]; blocks until done
 private:
  const TensorInfo* t_ = nullptr;
  const WTensor* resident_ = nullptr;
  int fd_ = -1;
  uint64_t row_bytes_ = 0; int64_t n_rows_ = 0; GType type_ = GType::F32;
  std::unique_ptr<IoPool> pool_;
};

}  // namespace ms
