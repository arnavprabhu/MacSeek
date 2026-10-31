#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "core/metal.hpp"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <map>

namespace ms {

void* Buffer::cpu() const { return (char*)[(__bridge id<MTLBuffer>)buf contents] + off; }

struct Device::Impl {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLLibrary> lib;
  std::mutex mu;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos;
  std::vector<id<MTLBuffer>> owned;
};

Device& Device::get() { static Device d; return d; }

Device::Device() : p(new Impl) {
  p->dev = MTLCreateSystemDefaultDevice();
  if (!p->dev) { fprintf(stderr, "macseek: no Metal device\n"); abort(); }
  p->q = [p->dev newCommandQueueWithMaxCommandBufferCount:64];
  const char* path = getenv("MACSEEK_METALLIB");
#ifdef MACSEEK_METALLIB
  if (!path) path = MACSEEK_METALLIB;
#endif
  NSError* err = nil;
  if (path) {
    NSString* ps = [NSString stringWithUTF8String:path];
    if ([ps hasSuffix:@".metallib"]) {
      p->lib = [p->dev newLibraryWithURL:[NSURL fileURLWithPath:ps] error:&err];
    } else {  // bundled source from metalc: compile once at startup (Metal 4, fast math)
      NSString* src = [NSString stringWithContentsOfFile:ps encoding:NSUTF8StringEncoding error:&err];
      MTLCompileOptions* o = [MTLCompileOptions new];
      o.languageVersion = MTLLanguageVersion4_0;
      o.mathMode = MTLMathModeFast;
      if (src) p->lib = [p->dev newLibraryWithSource:src options:o error:&err];
    }
  }
  if (!p->lib) fprintf(stderr, "macseek: warning: metallib not loaded (%s): %s\n", path ? path : "(none)",
                       err ? err.localizedDescription.UTF8String : "");
}

Buffer Device::alloc(size_t bytes, const char* label) {
  id<MTLBuffer> b = [p->dev newBufferWithLength:(bytes ? bytes : 16) options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
  if (!b) { fprintf(stderr, "macseek: alloc of %zu bytes failed\n", bytes); abort(); }
  if (label) b.label = [NSString stringWithUTF8String:label];
  memset(b.contents, 0, bytes);
  std::lock_guard<std::mutex> g(p->mu);
  p->owned.push_back(b);
  return Buffer{(__bridge void*)b, 0, bytes};
}

Buffer Device::wrap(void* ptr, size_t bytes, const char* label) {
  id<MTLBuffer> b = [p->dev newBufferWithBytesNoCopy:ptr length:bytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
  if (!b) { fprintf(stderr, "macseek: wrap of %zu bytes failed (page aligned? <= maxBufferLength?)\n", bytes); abort(); }
  if (label) b.label = [NSString stringWithUTF8String:label];
  std::lock_guard<std::mutex> g(p->mu);
  p->owned.push_back(b);
  return Buffer{(__bridge void*)b, 0, bytes};
}

void Device::release(Buffer& b) {
  std::lock_guard<std::mutex> g(p->mu);
  for (size_t i = 0; i < p->owned.size(); i++)
    if ((__bridge void*)p->owned[i] == b.buf) { p->owned.erase(p->owned.begin() + i); break; }
  b = Buffer{};
}

void* Device::pipeline(const std::string& fn) { return pipeline(fn, {}); }

void* Device::pipeline(const std::string& fn, const std::vector<std::pair<int, int>>& consts) {
  std::string key = fn;
  for (auto& c : consts) key += "|" + std::to_string(c.first) + "=" + std::to_string(c.second);
  std::lock_guard<std::mutex> g(p->mu);
  auto it = p->psos.find(key);
  if (it != p->psos.end()) return (__bridge void*)it->second;
  NSError* err = nil;
  id<MTLFunction> f;
  if (consts.empty()) {
    f = [p->lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
  } else {
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    for (auto& c : consts) { int v = c.second; [cv setConstantValue:&v type:MTLDataTypeInt atIndex:c.first]; }
    f = [p->lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()] constantValues:cv error:&err];
  }
  if (!f) { fprintf(stderr, "macseek: kernel '%s' not found in metallib %s\n", fn.c_str(), err ? err.localizedDescription.UTF8String : ""); abort(); }
  id<MTLComputePipelineState> pso = [p->dev newComputePipelineStateWithFunction:f error:&err];
  if (!pso) { fprintf(stderr, "macseek: pipeline '%s' failed: %s\n", fn.c_str(), err.localizedDescription.UTF8String); abort(); }
  p->psos[key] = pso;
  return (__bridge void*)pso;
}

void* Device::device() { return (__bridge void*)p->dev; }
void* Device::queue() { return (__bridge void*)p->q; }
size_t Device::max_buffer_length() const { return p->dev.maxBufferLength; }
size_t Device::recommended_working_set() const { return p->dev.recommendedMaxWorkingSetSize; }

struct Enc::Impl {
  id<MTLCommandBuffer> cb;
  id<MTLComputeCommandEncoder> ce;
  bool committed = false, waited = false;
};

Enc::Enc() : p(new Impl) {
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)Device::get().queue();
  p->cb = [q commandBufferWithUnretainedReferences];
  p->ce = [p->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
}
Enc::Enc(bool open_encoder) : p(new Impl) {
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)Device::get().queue();
  p->cb = [q commandBufferWithUnretainedReferences];
  p->ce = open_encoder ? [p->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : nil;
}
void Enc::enqueue() { [p->cb enqueue]; }
Enc::~Enc() { if (!p->waited) wait(); delete p; }
void Enc::set_pipeline(const char* fn) { set_pipeline(Device::get().pipeline(fn)); }
void Enc::set_pipeline(void* pso) { [p->ce setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso]; }
void Enc::set_buffer(int i, const Buffer& b) { [p->ce setBuffer:(__bridge id<MTLBuffer>)b.buf offset:b.off atIndex:i]; }
void Enc::set_bytes(int i, const void* d, size_t n) { [p->ce setBytes:d length:n atIndex:i]; }
void Enc::set_threadgroup_memory(int i, size_t n) { [p->ce setThreadgroupMemoryLength:(n + 15) / 16 * 16 atIndex:i]; }
void Enc::dispatch_groups(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
  [p->ce dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}
void Enc::dispatch_threads(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t tx, uint32_t ty, uint32_t tz) {
  [p->ce dispatchThreads:MTLSizeMake(nx, ny, nz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}
void Enc::barrier() { [p->ce memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
void Enc::end_encoder() { if (p->ce) { [p->ce endEncoding]; p->ce = nil; } }
void Enc::begin_encoder() { if (!p->ce) p->ce = [p->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; }
void Enc::commit() { if (!p->committed) { end_encoder(); [p->cb commit]; p->committed = true; } }
void Enc::wait() {
  commit();
  if (!p->waited) {
    [p->cb waitUntilCompleted];
    p->waited = true;
    if (p->cb.status == MTLCommandBufferStatusError) {
      fprintf(stderr, "macseek: GPU error: %s\n", p->cb.error.localizedDescription.UTF8String); abort();
    }
  }
}
double Enc::gpu_ms() const { return (p->cb.GPUEndTime - p->cb.GPUStartTime) * 1e3; }
void* Enc::encoder() { return (__bridge void*)p->ce; }
void* Enc::cmdbuf() { return (__bridge void*)p->cb; }

}  // namespace ms
