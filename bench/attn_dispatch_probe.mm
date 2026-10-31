// dispatch + barrier overhead probe
#include <algorithm>
#include <cstdio>
#include <vector>
#include "core/metal.hpp"
using namespace ms;
int main() {
  auto& d = Device::get();
  auto a = d.alloc(1 << 20), b = d.alloc(1 << 20), c = d.alloc(1 << 20);
  uint32_t n = 1024;
  for (int groups : {1, 40, 400}) for (bool bar : {false, true}) {
    std::vector<double> t;
    for (int it = 0; it < 9; it++) {
      Enc e;
      for (int i = 0; i < 100; i++) { e.set_pipeline("smoke_add"); e.set_buffer(0, a); e.set_buffer(1, b); e.set_buffer(2, c); e.set_value(3, n); e.dispatch_groups(groups, 1, 1, 256); if (bar) e.barrier(); }
      e.wait(); t.push_back(e.gpu_ms());
    }
    std::sort(t.begin(), t.end());
    printf("100 dispatches x %d TGs, barrier=%d: %.3f ms -> %.2f us/dispatch\n", groups, bar, t[4], t[4] * 10);
  }
}
