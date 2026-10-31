#pragma once
// Test-only helpers for the arch tests: synthetic data, comparisons, and optional real-model reads (small tensors and
// GGUF metadata only, via pread; never the whole model). Real-data checks are skipped if the model files are absent.
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace arch_test {

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t seed) : g(seed) {}
  float normal(float sd = 1.0f) { return std::normal_distribution<float>(0.0f, sd)(g); }
  float uniform(float a, float b) { return std::uniform_real_distribution<float>(a, b)(g); }
  uint32_t u32() { return (uint32_t)g(); }
  void fill_normal(float* p, size_t n, float sd = 1.0f) { for (size_t i = 0; i < n; i++) p[i] = normal(sd); }
};

// max |a-b| / (atol + rtol*|b|) style check; returns true if ok, prints the worst element.
bool close(const char* what, const float* got, const float* ref, size_t n, float atol, float rtol);

struct TensorInfo { std::string name, type, file; std::vector<uint64_t> dims; uint64_t nbytes = 0, offset = 0; };
// Model snapshot dir (MACSEEK_MODEL_DIR or the SPEC path); empty if not present.
std::string model_dir();
// Parses docs/tensors.txt (relative to this source tree).
bool find_tensor(const std::string& name, TensorInfo& out);
// pread nbytes at tensor offset + off. Refuses reads > 100 MB.
bool read_tensor_bytes(const TensorInfo& t, uint64_t off, uint64_t nbytes, void* dst);
bool read_tensor_f32(const std::string& name, std::vector<float>& out);

// Integer metadata arrays from split 1 (e.g. deepseek41.engram.token_map). Returns false if not found.
bool read_meta_int_arrays(const std::vector<std::string>& keys, std::map<std::string, std::vector<int64_t>>& out);

}  // namespace arch_test
