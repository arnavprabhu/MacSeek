#include "store/memguard.hpp"

#include <mach/mach.h>
#include <sys/sysctl.h>

#include <cstdio>

namespace ms {

MemStatus mem_status() {
  MemStatus m;
  size_t len = sizeof(m.total);
  sysctlbyname("hw.memsize", &m.total, &len, nullptr, 0);
  xsw_usage sw{}; len = sizeof sw;
  if (sysctlbyname("vm.swapusage", &sw, &len, nullptr, 0) == 0) { m.swap_total = sw.xsu_total; m.swap_used = sw.xsu_used; }
  vm_size_t page = 0; host_page_size(mach_host_self(), &page);
  vm_statistics64_data_t vs{}; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &cnt) == KERN_SUCCESS) {
    m.free = (uint64_t)vs.free_count * page;
    m.inactive = (uint64_t)vs.inactive_count * page;
    m.speculative = (uint64_t)vs.speculative_count * page;
    m.purgeable = (uint64_t)vs.purgeable_count * page;
    m.file_backed = (uint64_t)vs.external_page_count * page;
    m.active = (uint64_t)vs.active_count * page;
    m.wired = (uint64_t)vs.wire_count * page;
    m.compressed = (uint64_t)vs.compressor_page_count * page;
  }
  m.available = m.free + m.inactive + m.speculative + m.purgeable;
  task_vm_info_data_t ti{}; cnt = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &cnt) == KERN_SUCCESS) {
    m.self_rss = ti.resident_size;
    m.self_footprint = ti.phys_footprint;
  }
  return m;
}

std::string MemStatus::str() const {
  char b[320];
  const double G = 1024.0 * 1024 * 1024;
  snprintf(b, sizeof b,
           "total %.1f GiB | avail %.1f (free %.1f inactive %.1f spec %.1f purg %.1f) | active %.1f wired %.1f "
           "compressed %.1f | swap %.2f/%.2f | self rss %.2f footprint %.2f",
           total / G, available / G, free / G, inactive / G, speculative / G, purgeable / G, active / G, wired / G,
           compressed / G, swap_used / G, swap_total / G, self_rss / G, self_footprint / G);
  return b;
}

uint64_t recommend_arena_bytes(uint64_t planned_other, uint64_t headroom, uint64_t cap) {
  MemStatus m = mem_status();
  // Pages this process already owns are not in `available`; arena memory not yet touched is not counted either, so
  // the caller should ask BEFORE allocating the arena.
  int64_t r = (int64_t)m.available - (int64_t)headroom - (int64_t)planned_other;
  if (r < 0) r = 0;
  return (uint64_t)r < cap ? (uint64_t)r : cap;
}

bool swap_grew(const MemStatus& a, const MemStatus& b, uint64_t slack) { return b.swap_used > a.swap_used + slack; }

}  // namespace ms
