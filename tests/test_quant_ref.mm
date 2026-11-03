// Bit-exact check of the CPU reference dequant (tests/ref/dequant.cpp) against Homebrew libggml (dlopen, test-only):
// random blocks for every type + real rows pread from the model files (a few MB). Skips libggml parts if absent.
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "core/model_path.hpp"
#include "ref/dequant.hpp"
#if __has_include("/opt/homebrew/include/ggml.h")
#include "/opt/homebrew/include/ggml.h"
#define HAVE_GGML_H 1
#endif

using ms::q::QType;

typedef void (*to_float_fn)(const void*, float*, int64_t);
static to_float_fn ggml_to_float(QType t) {
#ifdef HAVE_GGML_H
  static void* h = dlopen("/opt/homebrew/lib/libggml-base.dylib", RTLD_NOW | RTLD_LOCAL);
  if (!h) return nullptr;
  using gt = const ggml_type_traits* (*)(enum ggml_type);
  static gt get = (gt)dlsym(h, "ggml_get_type_traits");
  if (!get) return nullptr;
  const ggml_type_traits* tr = get((enum ggml_type)(int)t);
  if (!tr || tr->blck_size != ms::q::block_elems(t) || tr->type_size != ms::q::block_bytes(t)) {
    printf("  libggml traits mismatch for %s\n", ms::q::type_name(t)); return nullptr;
  }
  return (to_float_fn)tr->to_float;
#else
  (void)t; return nullptr;
#endif
}

static int compare(QType t, const std::vector<uint8_t>& blocks, int64_t n, const char* what) {
  to_float_fn f = t == QType::F32 ? (to_float_fn)1 : ggml_to_float(t);
  if (!f) { printf("  %-8s %-28s SKIP (no libggml)\n", ms::q::type_name(t), what); return 0; }
  std::vector<float> a(n), b(n);
  ref::dequant_row(t, blocks.data(), a.data(), n);
  if (t == QType::F32) memcpy(b.data(), blocks.data(), n * 4); else f(blocks.data(), b.data(), n);  // ggml has no F32 to_float
  int64_t bad = 0, nanc = 0;
  for (int64_t i = 0; i < n; i++) {
    if (std::isnan(a[i]) && std::isnan(b[i])) { nanc++; continue; }
    if (memcmp(&a[i], &b[i], 4) != 0) { if (bad < 3) printf("    [%lld] ref=%.9g ggml=%.9g\n", (long long)i, a[i], b[i]); bad++; }
  }
  printf("  %-8s %-28s n=%-9lld %s\n", ms::q::type_name(t), what, (long long)n, bad ? "MISMATCH" : "bit-exact");
  return bad ? 1 : 0;
}

struct TInfo { std::string name, type, file; std::vector<int64_t> dims; size_t nbytes = 0, off = 0; };
static bool find_tensor(const std::string& name, TInfo& ti, const char* table) {
  std::string src = __FILE__; src = src.substr(0, src.rfind("/tests/"));
  std::ifstream f(src + "/docs/" + table);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string n, ty, dims, nb, file, off;
    std::getline(ss, n, '\t'); std::getline(ss, ty, '\t'); std::getline(ss, dims, '\t');
    std::getline(ss, nb, '\t'); std::getline(ss, file, '\t'); std::getline(ss, off, '\t');
    if (n != name) continue;
    ti.name = n; ti.type = ty; ti.file = file; ti.nbytes = std::stoull(nb); ti.off = std::stoull(off);
    for (char& c : dims) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream ds(dims); int64_t d; while (ds >> d) ti.dims.push_back(d);
    return true;
  }
  return false;
}

static int real_rows(const char* tname, QType t, int64_t elems, const char* table = "tensors.txt", const char* sub = "") {
  TInfo ti;
  if (!find_tensor(tname, ti, table)) { printf("  tensor %s not in docs/tensors.txt\n", tname); return 1; }
  std::string path = ms::default_model_dir() + "/" + sub + ti.file;
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) { printf("  %-8s %-28s SKIP (model file not readable)\n", ms::q::type_name(t), tname); return 0; }
  size_t bytes = ms::q::row_bytes(t, elems);
  // read from the middle of the tensor (more representative than the first rows)
  size_t start = (ti.nbytes / 2) / ms::q::block_bytes(t) * ms::q::block_bytes(t);
  if (start + bytes > ti.nbytes) start = 0;
  std::vector<uint8_t> buf(bytes);
  ssize_t r = pread(fd, buf.data(), bytes, (off_t)(ti.off + start));
  close(fd);
  if (r != (ssize_t)bytes) { printf("  short read on %s\n", tname); return 1; }
  return compare(t, buf, elems, tname);
}

int main() {
  int fails = 0;
  printf("random blocks:\n");
  const QType types[] = {QType::F32, QType::F16, QType::BF16, QType::Q8_0, QType::Q5_K, QType::Q6_K,
                         QType::IQ1_S, QType::IQ2_XXS, QType::IQ1_M, QType::IQ2_XS, QType::IQ4_NL,
                         QType::MXFP4};
  for (QType t : types) {
    const int64_t n = 256 * 4096;
    auto blocks = ref::random_blocks(t, n, 1234 + (int)t);
    fails += compare(t, blocks, n, "random");
  }
  printf("real rows (pread, middle of tensor; IQ1_M / IQ2_XS: random blocks only, their model is not used here):\n");
  const int64_t N = 5120 * 64;  // 64 rows of 5120
  fails += real_rows("blk.3.ffn_gate_exps.weight", QType::IQ1_S, N);
  fails += real_rows("blk.3.ffn_down_exps.weight", QType::IQ1_S, N);
  fails += real_rows("blk.30.ffn_up_exps.weight", QType::IQ2_XXS, N);
  fails += real_rows("blk.25.ffn_down_exps.weight", QType::IQ2_XXS, N);
  fails += real_rows("blk.5.attn_q_b.weight", QType::Q6_K, N);
  fails += real_rows("blk.30.attn_q_b.weight", QType::Q5_K, N);
  fails += real_rows("blk.15.ffn_up_shexp.weight", QType::Q8_0, N);
  fails += real_rows("blk.2.attn_compressor_kv.weight", QType::BF16, N);
  fails += real_rows("blk.3.ffn_gate_inp.weight", QType::F32, N);
  fails += real_rows("per_layer_token_embd.weight", QType::IQ4_NL, 256 * 4096);
  fails += real_rows("blk.0.ffn_down_exps.weight", QType::MXFP4, N, "dspark_tensors.txt", "../");
  printf(fails ? "FAILED (%d)\n" : "all OK\n", fails);
  return fails ? 1 : 0;
}
