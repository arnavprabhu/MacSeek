#include "core/metal.hpp"
#include <cstdio>
int main() {
  auto& d = ms::Device::get(); auto o = d.alloc(256);
  { ms::Enc e; e.set_pipeline("attn_probe_layout"); e.set_buffer(0, o); e.dispatch_groups(1,1,1,32); e.wait(); }
  float* p = (float*)o.cpu(); int bad = 0;
  for (int l = 0; l < 32; l++) {
    int qid = l / 4, fm = (qid & 4) + ((l / 2) % 4), fn = (qid & 2) * 2 + (l % 2) * 2;
    if (p[2*l] != fm*8+fn || p[2*l+1] != fm*8+fn+1) { bad++; printf("lane %d: %g %g expected %d %d\n", l, p[2*l], p[2*l+1], fm*8+fn, fm*8+fn+1); }
  }
  printf("layout %s\n", bad ? "MISMATCH" : "matches formula");
}
