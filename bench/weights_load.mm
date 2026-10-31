// Full resident-weight load (everything except experts, engram table and token_embd: ~6.15 GiB read, ~6.2 GiB RAM).
//   bench_weights_load [--threads N] [--chunk-mb N] [--embd] [--mlock]
// Refuses to run if available memory < 6.2 GiB + 8 GiB headroom.
#include "store/memguard.hpp"
#include "store/weights.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ms;

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  WeightsOptions o = WeightsOptions::from_env();
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) o.threads = atoi(argv[++i]);
    else if (a == "--chunk-mb" && i + 1 < argc) o.read_chunk = (size_t)atoi(argv[++i]) << 20;
    else if (a == "--embd") o.token_embd_resident = true;
    else if (a == "--mlock") o.mlock = true;
    else if (a == "--no-mlock") o.mlock = false;
    else if (a == "--prefault") o.prefault = true;
  }
  MemStatus m0 = mem_status();
  printf("mem before: %s\n", m0.str().c_str());
  if (m0.available < (6ull << 30) + (8ull << 30)) { printf("not enough free memory for a full load; abort\n"); return 2; }
  Gguf g(Gguf::default_model_path());
  Weights w(g, o);
  MemStatus m1 = mem_status();
  printf("resident weights: %zu tensors, %.3f GiB (%.3f GB read incl. 16K rounding) in %.0f ms = %.2f GB/s; %zu MTLBuffers; "
         "threads %d chunk %zu MiB; prefault %d; token_embd %s; mlock %s (%.0f ms)\n",
         w.all().size(), w.bytes() / 1073741824.0, w.bytes_read() / 1e9, w.load_ms(), w.gbps(), w.buffers().size(), o.threads,
         o.read_chunk >> 20, (int)o.prefault, o.token_embd_resident ? "resident" : "pread-on-demand", o.mlock ? (w.locked() ? "ok" : "FAILED") : "off", w.lock_ms());
  printf("mem after: %s\nswap grew: %s\n", m1.str().c_str(), swap_grew(m0, m1) ? "YES" : "no");
  return 0;
}
