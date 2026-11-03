#include "gguf/gguf.hpp"
#include "core/model_path.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <stdexcept>

namespace ms {

static const struct { GType t; TypeInfo i; } kTypes[] = {
    {GType::F32, {"F32", 1, 4}},         {GType::F16, {"F16", 1, 2}},         {GType::Q4_0, {"Q4_0", 32, 18}},
    {GType::Q4_1, {"Q4_1", 32, 20}},     {GType::Q5_0, {"Q5_0", 32, 22}},     {GType::Q5_1, {"Q5_1", 32, 24}},
    {GType::Q8_0, {"Q8_0", 32, 34}},     {GType::Q8_1, {"Q8_1", 32, 36}},     {GType::Q2_K, {"Q2_K", 256, 84}},
    {GType::Q3_K, {"Q3_K", 256, 110}},   {GType::Q4_K, {"Q4_K", 256, 144}},   {GType::Q5_K, {"Q5_K", 256, 176}},
    {GType::Q6_K, {"Q6_K", 256, 210}},   {GType::Q8_K, {"Q8_K", 256, 292}},   {GType::IQ2_XXS, {"IQ2_XXS", 256, 66}},
    {GType::IQ2_XS, {"IQ2_XS", 256, 74}}, {GType::IQ3_XXS, {"IQ3_XXS", 256, 98}}, {GType::IQ1_S, {"IQ1_S", 256, 50}},
    {GType::IQ4_NL, {"IQ4_NL", 32, 18}}, {GType::IQ3_S, {"IQ3_S", 256, 110}}, {GType::IQ2_S, {"IQ2_S", 256, 82}},
    {GType::IQ4_XS, {"IQ4_XS", 256, 136}}, {GType::I8, {"I8", 1, 1}},        {GType::I16, {"I16", 1, 2}},
    {GType::I32, {"I32", 1, 4}},         {GType::I64, {"I64", 1, 8}},         {GType::F64, {"F64", 1, 8}},
    {GType::IQ1_M, {"IQ1_M", 256, 56}},  {GType::BF16, {"BF16", 1, 2}},       {GType::TQ1_0, {"TQ1_0", 256, 54}},
    {GType::TQ2_0, {"TQ2_0", 256, 66}},  {GType::MXFP4, {"MXFP4", 32, 17}},
};

const TypeInfo* type_info(GType t) {
  for (auto& e : kTypes) if (e.t == t) return &e.i;
  return nullptr;
}
const char* type_name(GType t) { auto* i = type_info(t); return i ? i->name : "?"; }
uint64_t type_bytes(GType t, uint64_t n) {
  auto* i = type_info(t);
  if (!i) return 0;
  return n / i->blck * i->bytes;
}
uint64_t TensorInfo::row_bytes() const { return type_bytes(type, (uint64_t)ne[0]); }

size_t KvValue::size() const {
  if (type != KvType::ARR) return 0;
  return ai.size() + af.size() + as.size() + aa.size();
}
bool KvValue::is_int() const {
  KvType t = type == KvType::ARR ? arr_type : type;
  return t != KvType::F32 && t != KvType::F64 && t != KvType::STR && t != KvType::ARR;
}
bool KvValue::is_float() const { KvType t = type == KvType::ARR ? arr_type : type; return t == KvType::F32 || t == KvType::F64; }

long long pread_full(int fd, void* dst, size_t len, uint64_t off) {
  size_t got = 0;
  while (got < len) {
    ssize_t r = pread(fd, (char*)dst + got, len - got, (off_t)(off + got));
    if (r < 0) { if (errno == EINTR) continue; return -1; }
    if (r == 0) break;
    got += (size_t)r;
  }
  return (long long)got;
}

// ---- buffered sequential reader over pread (headers are a few MB; no mmap) ----
namespace {
struct Reader {
  int fd; uint64_t pos; std::vector<char> buf; uint64_t buf_pos = 0; size_t buf_len = 0; std::string path;
  Reader(int fd, const std::string& p) : fd(fd), pos(0), buf(4 << 20), path(p) {}
  [[noreturn]] void fail(const char* what) {
    fprintf(stderr, "gguf: %s: parse error (%s) at offset %llu\n", path.c_str(), what, (unsigned long long)pos);
    abort();
  }
  void read(void* dst, size_t n) {
    char* d = (char*)dst;
    while (n) {
      if (pos < buf_pos || pos >= buf_pos + buf_len) {
        long long r = pread_full(fd, buf.data(), buf.size(), pos);
        if (r <= 0) fail("unexpected EOF");
        buf_pos = pos; buf_len = (size_t)r;
      }
      size_t avail = (size_t)(buf_pos + buf_len - pos), take = avail < n ? avail : n;
      memcpy(d, buf.data() + (pos - buf_pos), take);
      d += take; n -= take; pos += take;
    }
  }
  template <class T> T get() { T v; read(&v, sizeof v); return v; }
  std::string str() {
    uint64_t n = get<uint64_t>();
    if (n > (1ull << 30)) fail("string too long");
    std::string s(n, '\0');
    if (n) read(s.data(), n);
    return s;
  }
};

void read_scalar(Reader& r, KvType t, KvValue& v) {
  v.type = t;
  switch (t) {
    case KvType::U8: v.u = r.get<uint8_t>(); v.i = (int64_t)v.u; break;
    case KvType::I8: v.i = r.get<int8_t>(); v.u = (uint64_t)v.i; break;
    case KvType::U16: v.u = r.get<uint16_t>(); v.i = (int64_t)v.u; break;
    case KvType::I16: v.i = r.get<int16_t>(); v.u = (uint64_t)v.i; break;
    case KvType::U32: v.u = r.get<uint32_t>(); v.i = (int64_t)v.u; break;
    case KvType::I32: v.i = r.get<int32_t>(); v.u = (uint64_t)v.i; break;
    case KvType::U64: v.u = r.get<uint64_t>(); v.i = (int64_t)v.u; break;
    case KvType::I64: v.i = r.get<int64_t>(); v.u = (uint64_t)v.i; break;
    case KvType::BOOL: v.u = r.get<uint8_t>() ? 1 : 0; v.i = (int64_t)v.u; break;
    case KvType::F32: v.f = r.get<float>(); return;
    case KvType::F64: v.f = r.get<double>(); return;
    case KvType::STR: v.s = r.str(); return;
    default: r.fail("bad scalar type");
  }
  v.f = (t == KvType::U64) ? (double)v.u : (double)v.i;
}

void read_value(Reader& r, KvType t, KvValue& v) {
  if (t != KvType::ARR) { read_scalar(r, t, v); return; }
  v.type = KvType::ARR;
  v.arr_type = (KvType)r.get<uint32_t>();
  uint64_t n = r.get<uint64_t>();
  if (n > (1ull << 32)) r.fail("array too long");
  switch (v.arr_type) {
    case KvType::STR: v.as.resize(n); for (uint64_t k = 0; k < n; k++) v.as[k] = r.str(); break;
    case KvType::F32: v.af.resize(n); for (uint64_t k = 0; k < n; k++) v.af[k] = r.get<float>(); break;
    case KvType::F64: v.af.resize(n); for (uint64_t k = 0; k < n; k++) v.af[k] = r.get<double>(); break;
    case KvType::ARR: v.aa.resize(n); for (uint64_t k = 0; k < n; k++) read_value(r, KvType::ARR, v.aa[k]); break;
    default: {
      v.ai.resize(n);
      for (uint64_t k = 0; k < n; k++) { KvValue e; read_scalar(r, v.arr_type, e); v.ai[k] = e.i; }
    }
  }
}
}  // namespace

bool Gguf::exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

std::string Gguf::default_model_path() {
  if (const char* e = getenv("MACSEEK_MODEL")) if (*e) return e;
  return ms::default_model_split1();
}

std::vector<std::string> Gguf::expand_splits(const std::string& path) {
  static const std::regex re("^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$");
  std::smatch m;
  if (!std::regex_match(path, m, re)) return {path};
  int n = std::stoi(m[3].str());
  std::vector<std::string> out;
  char tail[64];
  for (int i = 1; i <= n; i++) {
    snprintf(tail, sizeof tail, "-%05d-of-%05d.gguf", i, n);
    out.push_back(m[1].str() + tail);
  }
  // GSQ-RCO builds ship the Engram tables as a separate file (DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf) that is the
  // last split of every build; llama.cpp wants a link named like the last split. If that link is missing, use the
  // Engram file from the split directory or its parent.
  if (!exists(out.back())) {
    const std::string dir = out.back().substr(0, out.back().rfind('/'));
    for (const std::string cand : {dir + "/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", dir + "/../DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"})
      if (exists(cand)) { out.back() = cand; break; }
  }
  return out;
}

Gguf::Gguf(const std::string& path) : Gguf(expand_splits(path)) {}

Gguf::Gguf(const std::vector<std::string>& paths) {
  files_.resize(paths.size());
  for (size_t i = 0; i < paths.size(); i++) {
    File& f = files_[i];
    f.path = paths[i];
    f.fd = open(f.path.c_str(), O_RDONLY);
    f.fd_nc = open(f.path.c_str(), O_RDONLY);
    if (f.fd < 0 || f.fd_nc < 0) { fprintf(stderr, "gguf: cannot open %s: %s\n", f.path.c_str(), strerror(errno)); abort(); }
    fcntl(f.fd_nc, F_NOCACHE, 1);
    fcntl(f.fd_nc, F_RDAHEAD, 0);
    struct stat st; fstat(f.fd, &st); f.size = (uint64_t)st.st_size;
    parse_file((int)i);
  }
  for (size_t i = 0; i < tensors_.size(); i++) tindex_[tensors_[i].name] = i;
}

Gguf::~Gguf() {
  for (auto& f : files_) { if (f.fd >= 0) close(f.fd); if (f.fd_nc >= 0) close(f.fd_nc); }
}

void Gguf::parse_file(int split) {
  File& f = files_[split];
  Reader r(f.fd, f.path);
  char magic[4]; r.read(magic, 4);
  if (memcmp(magic, "GGUF", 4) != 0) r.fail("bad magic");
  uint32_t ver = r.get<uint32_t>();
  if (ver < 2) r.fail("unsupported version");
  if (split == 0) version_ = ver;
  uint64_t nt = r.get<uint64_t>(), nkv = r.get<uint64_t>();
  uint32_t align = 32;
  for (uint64_t k = 0; k < nkv; k++) {
    std::string key = r.str();
    KvType t = (KvType)r.get<uint32_t>();
    KvValue v; read_value(r, t, v);
    if (key == "general.alignment") align = (uint32_t)v.u;
    if (!kv_.count(key)) { kv_order_.push_back(key); kv_.emplace(key, std::move(v)); }
  }
  if (split == 0) align_ = align;
  size_t first = tensors_.size();
  for (uint64_t k = 0; k < nt; k++) {
    TensorInfo ti;
    ti.name = r.str();
    ti.n_dims = (int)r.get<uint32_t>();
    if (ti.n_dims < 1 || ti.n_dims > 4) r.fail("bad n_dims");
    for (int d = 0; d < ti.n_dims; d++) ti.ne[d] = (int64_t)r.get<uint64_t>();
    ti.type = (GType)r.get<uint32_t>();
    if (!type_info(ti.type)) r.fail("unknown tensor type");
    ti.offset = r.get<uint64_t>();   // relative for now
    ti.split = split;
    ti.nbytes = type_bytes(ti.type, (uint64_t)ti.nelements());
    tensors_.push_back(std::move(ti));
  }
  f.data_off = (r.pos + align - 1) / align * align;
  for (size_t k = first; k < tensors_.size(); k++) {
    tensors_[k].offset += f.data_off;
    if (tensors_[k].offset + tensors_[k].nbytes > f.size) r.fail("tensor data past EOF");
  }
}

std::string Gguf::basename(int split) const {
  const std::string& p = files_[split].path;
  size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

const KvValue* Gguf::kv(const std::string& key) const { auto it = kv_.find(key); return it == kv_.end() ? nullptr : &it->second; }
int64_t Gguf::get_i64(const std::string& key, int64_t d) const { auto* v = kv(key); return v ? (v->is_float() ? (int64_t)v->f : v->i) : d; }
double Gguf::get_f64(const std::string& key, double d) const { auto* v = kv(key); return v ? v->f : d; }
std::string Gguf::get_str(const std::string& key, const std::string& d) const { auto* v = kv(key); return v && v->type == KvType::STR ? v->s : d; }
std::vector<int64_t> Gguf::get_arr_i64(const std::string& key) const {
  auto* v = kv(key);
  if (!v || v->type != KvType::ARR) return {};
  if (v->is_float()) { std::vector<int64_t> o; for (double x : v->af) o.push_back((int64_t)x); return o; }
  return v->ai;
}
std::vector<float> Gguf::get_arr_f32(const std::string& key) const {
  auto* v = kv(key);
  std::vector<float> o;
  if (!v || v->type != KvType::ARR) return o;
  if (v->is_float()) for (double x : v->af) o.push_back((float)x);
  else for (int64_t x : v->ai) o.push_back((float)x);
  return o;
}
const std::vector<std::string>& Gguf::get_arr_str(const std::string& key) const {
  static const std::vector<std::string> empty;
  auto* v = kv(key);
  return v && v->type == KvType::ARR ? v->as : empty;
}
const TensorInfo* Gguf::find(const std::string& name) const { auto it = tindex_.find(name); return it == tindex_.end() ? nullptr : &tensors_[it->second]; }
const TensorInfo& Gguf::at(const std::string& name) const {
  auto* t = find(name);
  if (!t) { fprintf(stderr, "gguf: tensor '%s' not found\n", name.c_str()); abort(); }
  return *t;
}

}  // namespace ms
