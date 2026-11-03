#pragma once
// Speculative-decoding verify kernels (spec.metal): K-token versions of decode kernels whose per-token outputs are
// bit-identical to the decode kernels (same launch shape and per-lane arithmetic), with one weight fetch for all K.
#include "core/metal.hpp"
#include "kernels/quant.hpp"

namespace ms::sv {

constexpr int kMaxK = 8;   // tokens per verify pass

// Y[t] (+)= W . X[t] for t < k, bit-identical to k calls of q::encode_matvec(W, X[t], Y[t], accumulate, groups) with
// the default launch shape. X row stride = cols * groups, Y row stride = rows. Types F32 F16 BF16 Q8_0 Q5_K Q6_K.
void encode_matvec_k(Enc& e, q::QType t, const Buffer& W, int rows, int cols, const Buffer& X, const Buffer& Y, int k,
                     bool accumulate = false, int groups = 1);
bool matvec_k_supported(q::QType t);

// Shared expert of the decode MoE (MoePart::Shared: gate/up + swiglu, barrier, down accumulated into Y), k tokens:
// X [k][5120], H scratch [k][2304], Y [k][5120] (accumulated). Bit-identical per token to
// q::encode_moe_decode_table_part(..., MoePart::Shared). Ends without a trailing barrier.
void encode_moe_shared_k(Enc& e, q::QType t_shared, const q::ExpertSlot& shared, const Buffer& X, const Buffer& H,
                         const Buffer& Y, int k);

}  // namespace ms::sv
