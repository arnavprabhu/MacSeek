#pragma once
// CPU references for the arch kernels (double accumulation). Mirrors reference/inference/model.py + kernel.py.
#include <cstdint>
#include <vector>

namespace ref_arch {

float half_to_float(uint16_t h);

// hc_mixes for one token: h [4*5120] -> mix [24] (pre, post, comb) incl. Sinkhorn.
void hc_mixes(const float* h, const float* fn, const float* scale, const float* base, float* mix, int iters = 20,
              double norm_eps = 1e-20, double hc_eps = 1e-6);
// Sinkhorn split of raw (already rms-scaled) mixes.
void hc_split(const double* m, const float* scale, const float* base, float* mix, int iters, double eps);
// One sublayer boundary for one token (see arch.hpp HcStep). x_sub / fn / x_out may be null.
void hc_step(const float* h_in, const float* x_sub, const float* mix_prev, const float* fn, const float* scale,
             const float* base, const float* norm_w, float* h_out, float* mix_out, float* x_out);

void rmsnorm(const float* x, const float* w, float* y, int dim, double eps);

// routed experts used by router / route_lists (384 or 192)
extern int g_n_experts;
// router for one token: logits [E] -> ids [6], w [6] (descending by score+bias)
void router(const float* logits, const float* bias, int* ids, float* w, float route_scale = 1.5f);
// grouped lists, ordering identical to the GPU kernels
void route_lists(const int* ids, const float* w, int n, std::vector<uint32_t>& counts, std::vector<uint32_t>& offsets,
                 std::vector<uint32_t>& entry, std::vector<uint32_t>& tok, std::vector<float>& lw,
                 std::vector<uint32_t>& slot_of);

// engram gate for one token: h [4][5120] updated, kv [25600], w [4][5120]; gate [4] out
void engram_gate(float* h, const float* kv, const float* w, float* gate, double eps = 1e-20, double clamp = 1e-6);

// row dequant
void iq4nl_row(const uint8_t* src, float* dst, int n_values);  // n_values multiple of 32
void q6k_row(const uint8_t* src, float* dst, int n_values);    // n_values multiple of 256

int argmax(const float* x, int n);  // ties -> lowest index
void topk(const float* x, int n, int k, float* v, uint32_t* idx);  // descending, ties -> lower index
double logsumexp(const float* x, int n, double scale = 1.0);
void log_softmax(const float* x, float* y, int n);

}  // namespace ref_arch
