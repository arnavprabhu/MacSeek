#pragma once
// CPU reference dequantization (block bytes -> float), bit-exact with libggml's to_float for every supported type.
#include <cstddef>
#include <cstdint>
#include <vector>
#include "kernels/quant_types.hpp"

namespace ref {
using ms::q::QType;

float fp16_to_f32(uint16_t h);
uint16_t f32_to_fp16(float f);
float bf16_to_f32(uint16_t h);

// Dequantize n elements (n multiple of the block size) starting at src.
void dequant_row(QType t, const void* src, float* dst, int64_t n);

// y[r] = sum_c dequant(W[r])[c] * x[c]  in double precision (reference matvec)
void matvec_ref(QType t, const void* W, int rows, int cols, const float* x, float* y);

// Random but valid blocks of a type (random bytes with sane scales; for IQ types the grid indices are random).
std::vector<uint8_t> random_blocks(QType t, int64_t n_elems, uint32_t seed, float scale = 0.01f);
}  // namespace ref
