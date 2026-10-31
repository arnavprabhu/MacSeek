// One-off oracle check (test-only, not built by CMake): compares the arch CPU dequant references (tests/ref/arch_ref.cpp,
// which the GPU kernels match bit-exactly in test_arch_engram) against libggml's to_float on real rows of
// token_embd (Q6_K) and per_layer_token_embd (IQ4_NL). Reads ~0.2 MB via pread; never loads the model.
// Build: clang++ -std=c++20 -O2 -Itests -I/opt/homebrew/include tools/arch_dequant_check.cpp tests/ref/arch_ref.cpp \
//          $PWD/tests/ref/arch_testutil.cpp -L/opt/homebrew/lib -lggml-base -Wl,-rpath,/opt/homebrew/lib -o build/arch_dequant_check
//   (arch_testutil.cpp must be given as an absolute path: it locates docs/tensors.txt via __FILE__)
#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml.h"
#include "ref/arch_ref.hpp"
#include "ref/arch_testutil.hpp"

static int check(const char* tensor, ggml_type type, int row_vals, int row_bytes, const std::vector<uint64_t>& rows,
                 void (*mine)(const uint8_t*, float*, int)) {
  arch_test::TensorInfo t;
  if (!arch_test::find_tensor(tensor, t)) { printf("%s not found\n", tensor); return 1; }
  const ggml_type_traits* tr = ggml_get_type_traits(type);
  std::vector<uint8_t> raw(row_bytes);
  std::vector<float> a(row_vals), b(row_vals);
  size_t bad = 0;
  for (uint64_t r : rows) {
    if (!arch_test::read_tensor_bytes(t, r * row_bytes, row_bytes, raw.data())) { printf("read failed\n"); return 1; }
    tr->to_float(raw.data(), a.data(), row_vals);
    mine(raw.data(), b.data(), row_vals);
    for (int i = 0; i < row_vals; i++) bad += memcmp(&a[i], &b[i], 4) != 0;
  }
  printf("%-28s %s: %zu rows, %zu values differ from libggml\n", tensor, bad ? "FAIL" : "bit-exact", rows.size(), bad);
  return bad != 0;
}

int main() {
  if (arch_test::model_dir().empty()) { printf("model not present\n"); return 0; }
  std::vector<uint64_t> erows, prows;
  for (uint64_t i = 0; i < 129280; i += 2693) erows.push_back(i);
  erows.push_back(129279);
  for (uint64_t i = 0; i < 768022850ull; i += 7680223ull) prows.push_back(i);
  prows.push_back(768022849ull);
  int f = check("token_embd.weight", GGML_TYPE_Q6_K, 5120, 4200, erows, ref_arch::q6k_row);
  f |= check("output.weight", GGML_TYPE_Q6_K, 5120, 4200, erows, ref_arch::q6k_row);
  f |= check("per_layer_token_embd.weight", GGML_TYPE_IQ4_NL, 256, 144, prows, ref_arch::iq4nl_row);
  return f;
}
