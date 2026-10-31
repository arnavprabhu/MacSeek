#pragma once
// Memory guard: free RAM / swap / process footprint, and the recommended expert-arena size with headroom.
#include <cstddef>
#include <cstdint>
#include <string>

namespace ms {

struct MemStatus {
  uint64_t total = 0;          // hw.memsize
  uint64_t free = 0;           // free pages
  uint64_t inactive = 0;       // inactive (mostly clean file cache; reclaimable)
  uint64_t speculative = 0;    // speculative read-ahead pages (reclaimable)
  uint64_t purgeable = 0;      // purgeable (reclaimable)
  uint64_t file_backed = 0;    // external (file-backed) pages: page cache incl. active
  uint64_t active = 0, wired = 0, compressed = 0;   // compressed = bytes occupied by the compressor
  uint64_t swap_total = 0, swap_used = 0;
  uint64_t available = 0;      // free + inactive + speculative + purgeable (what we can take without swapping)
  uint64_t self_rss = 0;       // resident size of this process
  uint64_t self_footprint = 0; // phys_footprint (what Activity Monitor shows; includes wired/IOKit memory)
  std::string str() const;     // one-line human summary (GiB)
};

MemStatus mem_status();

// Recommended expert slot-arena bytes for THIS process right now:
//   available + already_reserved_by_us - headroom - planned_other
// where planned_other = everything else this process still has to allocate (weights not yet loaded, scratch, state,
// engram cache). Clamped to [0, cap]. headroom default 8 GiB (SPEC §5: never swap).
uint64_t recommend_arena_bytes(uint64_t planned_other, uint64_t headroom = 8ull << 30, uint64_t cap = ~0ull);

// true if swap use grew by more than `slack` since `before` (call before/after runs).
bool swap_grew(const MemStatus& before, const MemStatus& after, uint64_t slack = 64ull << 20);

}  // namespace ms
