#include "ref/arch_testutil.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace arch_test {

bool close(const char* what, const float* got, const float* ref, size_t n, float atol, float rtol) {
  double worst = 0;
  size_t wi = 0;
  for (size_t i = 0; i < n; i++) {
    double err = std::fabs((double)got[i] - ref[i]);
    double lim = atol + rtol * std::fabs((double)ref[i]);
    double r = err / lim;
    if (!(r <= worst)) { worst = r; wi = i; }
    if (std::isnan(got[i]) != std::isnan(ref[i])) { worst = INFINITY; wi = i; break; }
  }
  bool ok = worst <= 1.0;
  printf("  %-44s %s  worst %.3g of tol at [%zu] got %.8g ref %.8g\n", what, ok ? "ok  " : "FAIL", worst, wi,
         n ? got[wi] : 0.f, n ? ref[wi] : 0.f);
  return ok;
}

static std::string src_root() {
  std::string f = __FILE__;  // .../tests/ref/arch_testutil.cpp
  for (int i = 0; i < 3; i++) f = f.substr(0, f.find_last_of('/'));
  return f;
}

std::string model_dir() {
  const char* e = getenv("MACSEEK_MODEL_DIR");
  std::string d = e ? e : std::string(getenv("HOME") ? getenv("HOME") : "") +
                              "/Unsloth Models/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/"
                              "8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-IQ1_S";
  struct stat st;
  std::string probe = d + "/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf";
  if (stat(probe.c_str(), &st) != 0) return "";
  return d;
}

bool find_tensor(const std::string& name, TensorInfo& out) {
  std::ifstream f(src_root() + "/docs/tensors.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, name.size() + 1, name + "\t") != 0) continue;
    std::stringstream ss(line);
    std::string dims;
    std::getline(ss, out.name, '\t');
    std::getline(ss, out.type, '\t');
    std::getline(ss, dims, '\t');
    std::string nb, off;
    std::getline(ss, nb, '\t');
    std::getline(ss, out.file, '\t');
    std::getline(ss, off, '\t');
    out.nbytes = std::stoull(nb);
    out.offset = std::stoull(off);
    out.dims.clear();
    for (char& c : dims) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::stringstream ds(dims);
    uint64_t v;
    while (ds >> v) out.dims.push_back(v);
    return true;
  }
  return false;
}

bool read_tensor_bytes(const TensorInfo& t, uint64_t off, uint64_t nbytes, void* dst) {
  if (nbytes > 100ull << 20) { fprintf(stderr, "refusing read of %llu bytes\n", (unsigned long long)nbytes); return false; }
  std::string d = model_dir();
  if (d.empty()) return false;
  int fd = open((d + "/" + t.file).c_str(), O_RDONLY);
  if (fd < 0) return false;
  ssize_t r = pread(fd, dst, nbytes, (off_t)(t.offset + off));
  ::close(fd);
  return r == (ssize_t)nbytes;
}

bool read_tensor_f32(const std::string& name, std::vector<float>& out) {
  TensorInfo t;
  if (!find_tensor(name, t) || t.type != "F32") return false;
  out.resize(t.nbytes / 4);
  return read_tensor_bytes(t, 0, t.nbytes, out.data());
}

namespace {
struct Reader {
  FILE* f;
  template <class T> T get() { T v{}; if (fread(&v, sizeof(T), 1, f) != 1) throw 1; return v; }
  std::string str() { uint64_t n = get<uint64_t>(); std::string s(n, '\0'); if (n && fread(s.data(), 1, n, f) != n) throw 1; return s; }
  static int scalar_size(uint32_t t) {
    switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4;
      case 10: case 11: case 12: return 8; default: return -1; }
  }
  int64_t get_int(uint32_t t) {
    switch (t) { case 0: return get<uint8_t>(); case 1: return get<int8_t>(); case 2: return get<uint16_t>();
      case 3: return get<int16_t>(); case 4: return get<uint32_t>(); case 5: return get<int32_t>();
      case 10: return (int64_t)get<uint64_t>(); case 11: return get<int64_t>(); default: throw 2; }
  }
  void skip(uint32_t t) {
    if (t == 8) { str(); return; }
    if (t == 9) { uint32_t at = get<uint32_t>(); uint64_t n = get<uint64_t>(); 
      int s = scalar_size(at);
      if (s > 0) fseeko(f, (off_t)(s * n), SEEK_CUR); else for (uint64_t i = 0; i < n; i++) skip(at);
      return; }
    int s = scalar_size(t);
    if (s < 0) throw 3;
    fseeko(f, s, SEEK_CUR);
  }
};
}  // namespace

bool read_meta_int_arrays(const std::vector<std::string>& keys, std::map<std::string, std::vector<int64_t>>& out) {
  std::string d = model_dir();
  if (d.empty()) return false;
  FILE* f = fopen((d + "/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf").c_str(), "rb");
  if (!f) return false;
  Reader r{f};
  try {
    if (r.get<uint32_t>() != 0x46554747u) throw 4;  // "GGUF"
    r.get<uint32_t>();
    r.get<uint64_t>();
    uint64_t nkv = r.get<uint64_t>();
    for (uint64_t i = 0; i < nkv; i++) {
      std::string k = r.str();
      uint32_t t = r.get<uint32_t>();
      bool want = false;
      for (auto& w : keys) want |= (w == k);
      if (!want) { r.skip(t); continue; }
      std::vector<int64_t> v;
      if (t == 9) {
        uint32_t at = r.get<uint32_t>();
        uint64_t n = r.get<uint64_t>();
        v.resize(n);
        for (uint64_t j = 0; j < n; j++) v[j] = r.get_int(at);
      } else {
        v.push_back(r.get_int(t));
      }
      out[k] = std::move(v);
    }
  } catch (...) {
    fclose(f);
    return false;
  }
  fclose(f);
  for (auto& k : keys) if (!out.count(k)) return false;
  return true;
}

}  // namespace arch_test
