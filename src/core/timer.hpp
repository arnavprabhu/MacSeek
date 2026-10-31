#pragma once
#include <chrono>
namespace ms {
struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
  void reset() { t0 = std::chrono::steady_clock::now(); }
};
}
