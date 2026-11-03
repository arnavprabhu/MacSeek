// Weights test: load a SUBSET of resident tensors (layers 0-1 + globals except token_embd, ~0.25 GB) and compare
// every byte with plain preads; check lookup, buffer views, repack hook, TokenEmbd rows, memguard.
#import <Metal/Metal.h>

#include "core/metal.hpp"
#include "store/memguard.hpp"
#include "store/weights.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

int main() {
  ms::MemStatus m0 = ms::mem_status();
  printf("mem: %s\n", m0.str().c_str());
  CHECK(m0.total == 64ull << 30 || m0.total > 0, "memsize");
  CHECK(m0.available > 0 && m0.available <= m0.total, "available");
  printf("recommend_arena_bytes(other=10 GiB) = %.1f GiB (now)\n", ms::recommend_arena_bytes(10ull << 30) / 1073741824.0);

  std::string mp = ms::Gguf::default_model_path();
  if (!ms::Gguf::exists(mp)) { printf("test_store_weights: no model, SKIP\n"); return fails ? 1 : 0; }
  ms::Gguf g(mp);
  CHECK(ms::Weights::is_streamed("blk.3.ffn_up_exps.weight") && ms::Weights::is_streamed("per_layer_token_embd.weight") &&
        !ms::Weights::is_streamed("blk.3.ffn_up_shexp.weight") && !ms::Weights::is_streamed("token_embd.weight"), "is_streamed");

  ms::WeightsOptions o;
  o.filter = [](const ms::TensorInfo& t) {
    return t.name.rfind("blk.0.", 0) == 0 || t.name.rfind("blk.1.", 0) == 0 || t.name.rfind("blk.", 0) != 0;
  };
  o.max_buffer_bytes = 600 << 20;  // force several MTLBuffers
  ms::Weights w(g, o);
  printf("loaded %zu tensors, %.3f GB in %.1f ms (%.2f GB/s), %zu buffers\n", w.all().size(), w.bytes() / 1e9, w.load_ms(), w.gbps(),
         w.buffers().size());
  CHECK(w.buffers().size() > 1, "buffer split");
  CHECK(!w.has_token_embd() && w.find("output.weight") && w.find("blk.1.engram_kv.weight") && !w.find("blk.1.ffn_up_exps.weight"), "lookup");
  CHECK(w.find("blk.2.attn_kv.weight") == nullptr, "filter");
  size_t bad = 0, n = 0;
  for (auto& t : w.all()) {
    n++;
    std::vector<char> ref(t.nbytes);
    ms::pread_full(g.fd(t.info->split), ref.data(), t.nbytes, t.info->offset);
    bool ok = memcmp(ref.data(), t.host, t.nbytes) == 0 && t.buf.cpu() == t.host && t.buf.size == t.nbytes &&
              ((uintptr_t)t.host % 32) == 0 && t.buf.off + t.nbytes <= [(__bridge id<MTLBuffer>)t.buf.buf length];
    if (!ok) { bad++; fprintf(stderr, "  mismatch %s\n", t.name.c_str()); }
  }
  CHECK(bad == 0, "%zu/%zu tensors differ", bad, n);
  const ms::WTensor& q = w.at("blk.0.attn_q_b.weight");
  const bool coder = g.n_splits() == 6;   // GSQ-RCO Coder: Q5_K attention / token_embd; Unsloth UD-IQ1_S: Q6_K
  CHECK(q.type == (coder ? ms::GType::Q5_K : ms::GType::Q6_K) && q.ne[0] == 1280 && q.ne[1] == 32768, "attn_q_b meta");

  // repack hook: replace F32 norms with a copy (as a kernel-layout repack would)
  size_t nrep = w.repack([](const ms::WTensor& src, ms::WTensor& out) {
    if (src.type != ms::GType::F32 || src.name.find("attn_norm") == std::string::npos) return false;
    out.buf = ms::Device::get().alloc(src.nbytes, "repacked");
    out.host = out.buf.cpu();
    memcpy(out.host, src.host, src.nbytes);
    return true;
  });
  CHECK(nrep == 2 && w.at("blk.0.attn_norm.weight").repacked, "repack %zu", nrep);
  std::vector<char> ref(20480);
  ms::pread_full(g.fd(w.at("blk.1.attn_norm.weight").info->split), ref.data(), 20480, w.at("blk.1.attn_norm.weight").info->offset);
  CHECK(memcmp(ref.data(), w.at("blk.1.attn_norm.weight").host, 20480) == 0, "repacked data");

  // token_embd rows (pread on demand)
  ms::TokenEmbd te(g, &w);
  CHECK(te.row_bytes() == (coder ? 3520u : 4200u) && te.n_rows() == 129280, "token_embd geometry");
  std::vector<int32_t> toks = {0, 1, 129279, 4242, 4242, 77, 100000};
  std::vector<char> rows(toks.size() * te.row_bytes()), r1(te.row_bytes());
  te.fetch(toks.data(), toks.size(), rows.data());
  const ms::TensorInfo& et = g.at("token_embd.weight");
  bool ok = true;
  for (size_t i = 0; i < toks.size(); i++) {
    ms::pread_full(g.fd(et.split), r1.data(), te.row_bytes(), et.offset + (uint64_t)toks[i] * te.row_bytes());
    ok &= memcmp(r1.data(), rows.data() + i * te.row_bytes(), te.row_bytes()) == 0;
  }
  CHECK(ok, "token_embd rows");
  ms::MemStatus m1 = ms::mem_status();
  printf("after: self rss %.2f GiB footprint %.2f GiB; swap grew: %s\n", m1.self_rss / 1073741824.0, m1.self_footprint / 1073741824.0,
         ms::swap_grew(m0, m1) ? "yes" : "no");
  printf("test_store_weights: %s\n", fails ? "FAILED" : "OK");
  return fails ? 1 : 0;
}
