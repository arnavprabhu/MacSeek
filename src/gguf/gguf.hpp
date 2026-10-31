#pragma once
// GGUF parser (no mmap): metadata with typed values (scalars + arrays, incl. nested), tensor table with ABSOLUTE
// file offsets across all splits, ggml type sizes. Keeps two read-only fds per split: a normal (page-cached) one and
// an F_NOCACHE one (direct-ish SSD reads that do not pollute the unified-memory page cache).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ms {

enum class GType : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
  Q5_K = 13, Q6_K = 14, Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21,
  IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30, TQ1_0 = 34,
  TQ2_0 = 35, MXFP4 = 39,
};

struct TypeInfo { const char* name; uint32_t blck; uint32_t bytes; };  // elements per block, bytes per block
const TypeInfo* type_info(GType t);                // nullptr if unknown
const char* type_name(GType t);                    // "?" if unknown
uint64_t type_bytes(GType t, uint64_t nelem);      // bytes for nelem elements (nelem multiple of block size)

enum class KvType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 };

struct KvValue {
  KvType type = KvType::U8;
  KvType arr_type = KvType::U8;       // element type when type == ARR
  // scalars: integers/bools fill i and u (same bits), floats fill f (ints also mirrored into f), strings fill s
  int64_t i = 0; uint64_t u = 0; double f = 0; std::string s;
  // arrays (by element type): integer/bool -> ai, float -> af, string -> as, nested array -> aa
  std::vector<int64_t> ai; std::vector<double> af; std::vector<std::string> as; std::vector<KvValue> aa;
  size_t size() const;                // array length (0 for scalars)
  bool is_int() const;
  bool is_float() const;
};

struct TensorInfo {
  std::string name;
  GType type = GType::F32;
  int n_dims = 0;
  int64_t ne[4] = {1, 1, 1, 1};       // ggml order: ne[0] contiguous
  uint64_t nbytes = 0;
  int split = 0;                      // index into Gguf::path()/fd()
  uint64_t offset = 0;                // ABSOLUTE byte offset of the data in that split file
  int64_t nelements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
  uint64_t row_bytes() const;         // bytes of one ne[0]-row
};

class Gguf {
 public:
  // One path: if it matches "*-0000N-of-0000M.gguf" all M splits are opened, else just that file.
  explicit Gguf(const std::string& path);
  explicit Gguf(const std::vector<std::string>& paths);
  ~Gguf();
  Gguf(const Gguf&) = delete; Gguf& operator=(const Gguf&) = delete;

  static std::vector<std::string> expand_splits(const std::string& path);
  static std::string default_model_path();   // $MACSEEK_MODEL or the Unsloth UD-IQ1_S split 1 path
  static bool exists(const std::string& path);

  // metadata (merged across splits, first occurrence wins; insertion order preserved in keys())
  const KvValue* kv(const std::string& key) const;
  const std::vector<std::string>& keys() const { return kv_order_; }
  int64_t get_i64(const std::string& key, int64_t dflt = 0) const;
  double get_f64(const std::string& key, double dflt = 0) const;
  std::string get_str(const std::string& key, const std::string& dflt = "") const;
  std::vector<int64_t> get_arr_i64(const std::string& key) const;
  std::vector<float> get_arr_f32(const std::string& key) const;
  const std::vector<std::string>& get_arr_str(const std::string& key) const;   // empty if missing

  // tensors (in file order, split by split)
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(const std::string& name) const;
  const TensorInfo& at(const std::string& name) const;   // aborts if missing

  int n_splits() const { return (int)files_.size(); }
  const std::string& path(int split) const { return files_[split].path; }
  std::string basename(int split) const;
  int fd(int split) const { return files_[split].fd; }                  // page-cached
  int fd_nocache(int split) const { return files_[split].fd_nc; }       // F_NOCACHE + no readahead
  uint64_t file_size(int split) const { return files_[split].size; }
  uint64_t data_offset(int split) const { return files_[split].data_off; }
  uint32_t alignment() const { return align_; }
  uint32_t version() const { return version_; }

 private:
  struct File { std::string path; int fd = -1, fd_nc = -1; uint64_t size = 0, data_off = 0; };
  void parse_file(int split);
  std::vector<File> files_;
  std::unordered_map<std::string, KvValue> kv_;
  std::vector<std::string> kv_order_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, size_t> tindex_;
  uint32_t align_ = 32, version_ = 0;
};

// pread loop: retries EINTR / short reads until len bytes or EOF. Returns bytes read (or -1 on error).
long long pread_full(int fd, void* dst, size_t len, uint64_t off);

}  // namespace ms
