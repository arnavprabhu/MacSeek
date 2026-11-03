#pragma once
// Minimal Metal runtime shared by all modules. Objective-C types are hidden behind void* so plain .cpp files can
// include this header; .mm files can use the ObjC accessors (MACSEEK_OBJC section).
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ms {

struct Buffer {
  void*  buf  = nullptr;   // id<MTLBuffer> (retained by the owner, see Device::alloc / Device::wrap)
  size_t off  = 0;         // byte offset bound with the buffer
  size_t size = 0;         // bytes usable from off
  void*  cpu() const;      // contents() + off (shared storage only)
  Buffer sub(size_t o, size_t s) const { return Buffer{buf, off + o, s}; }
  explicit operator bool() const { return buf != nullptr; }
};

class Device {
 public:
  static Device& get();                       // process-wide singleton; loads $MACSEEK_METALLIB or the build-time default (build/macseek.metal bundled source, compiled at startup; a .metallib path is also accepted)
  Buffer alloc(size_t bytes, const char* label = nullptr);          // shared storage, zero-filled, owned by Device
  Buffer wrap(void* page_aligned_ptr, size_t bytes, const char* label = nullptr);  // no-copy shared buffer
  void   release(Buffer& b);
  void*  pipeline(const std::string& fn);     // cached id<MTLComputePipelineState>; aborts with a message if missing
  void*  pipeline(const std::string& fn, const std::vector<std::pair<int, int>>& int_constants);  // function constants
  void*  device();                            // id<MTLDevice>
  void*  queue();                             // id<MTLCommandQueue>
  size_t max_buffer_length() const;
  size_t recommended_working_set() const;
 private:
  Device();
  struct Impl; Impl* p;
};

// Process-wide encode counters (profiling: dispatches, barrier calls, barriers that follow at least one dispatch,
// compute encoders opened). Not thread-safe; reset by assigning {}.
struct EncStats { uint64_t dispatches = 0, barriers = 0, stages = 0, encoders = 0; };
EncStats& enc_stats();

// One command buffer + one compute encoder. encode_* functions take an Enc&. Nothing commits until commit()/wait().
class Enc {
 public:
  Enc();                  // new command buffer on Device::get().queue()
  explicit Enc(bool open_encoder);  // false: no compute encoder yet (call begin_encoder()/between before dispatching)
  void enqueue();         // reserve this command buffer's place in the queue now (commit later)
  ~Enc();                 // commits + waits if still open
  void set_pipeline(const char* fn);
  void set_pipeline(void* pso);
  void set_buffer(int idx, const Buffer& b);
  void set_bytes(int idx, const void* p, size_t n);
  template <class T> void set_value(int idx, const T& v) { set_bytes(idx, &v, sizeof(T)); }
  void set_threadgroup_memory(int idx, size_t n);
  void dispatch_groups(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty = 1, uint32_t tz = 1);
  void dispatch_threads(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t tx, uint32_t ty = 1, uint32_t tz = 1);
  void use_resources(const std::vector<Buffer>& bufs);  // make buffers resident for indirect (pointer) access
  void barrier();         // memory barrier on buffers between dependent dispatches (concurrent encoder)
  // End the current compute encoder, run f(cmdbuf) (e.g. encodeWaitForEvent / encodeSignalEvent, which need no
  // open encoder), then open a new compute encoder in the same command buffer. Pipelines/bindings must be re-set.
  template <class F> void between(F&& f) { end_encoder(); f(cmdbuf()); begin_encoder(); }
  void end_encoder();
  void begin_encoder();
  void commit();          // end encoding + commit (async)
  void wait();            // commit if needed + waitUntilCompleted; aborts on GPU error
  double gpu_ms() const;  // GPUEndTime - GPUStartTime after wait()
  void gpu_span(double& start_s, double& end_s) const;   // absolute GPUStartTime / GPUEndTime (s) after wait()
  void* encoder();        // id<MTLComputeCommandEncoder>
  void* cmdbuf();         // id<MTLCommandBuffer>
 private:
  struct Impl; Impl* p;
};

}  // namespace ms
