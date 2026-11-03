#pragma once
// Quant type ids and block geometry (plain C++, no Metal). Values match ggml_type ids so GGUF type fields map 1:1.
#include <cstddef>
#include <cstdint>

namespace ms::q {

enum class QType : int {
  F32 = 0, F16 = 1, Q8_0 = 8, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17, IQ1_S = 19, IQ4_NL = 20, IQ1_M = 29,
  BF16 = 30, MXFP4 = 39,
};

// elements per block (dense types report 1) and bytes per block
inline int block_elems(QType t) {
  switch (t) {
    case QType::F32: case QType::F16: case QType::BF16: return 1;
    case QType::Q8_0: case QType::IQ4_NL: case QType::MXFP4: return 32;
    default: return 256;
  }
}
inline size_t block_bytes(QType t) {
  switch (t) {
    case QType::F32: return 4;
    case QType::F16: case QType::BF16: return 2;
    case QType::Q8_0: return 34;
    case QType::Q5_K: return 176;
    case QType::Q6_K: return 210;
    case QType::IQ2_XXS: return 66;
    case QType::IQ2_XS: return 74;
    case QType::IQ1_S: return 50;
    case QType::IQ1_M: return 56;
    case QType::IQ4_NL: return 18;
    case QType::MXFP4: return 17;
  }
  return 0;
}
inline size_t row_bytes(QType t, int64_t cols) { return (size_t)(cols / block_elems(t)) * block_bytes(t); }
inline const char* type_name(QType t) {
  switch (t) {
    case QType::F32: return "F32"; case QType::F16: return "F16"; case QType::BF16: return "BF16";
    case QType::Q8_0: return "Q8_0"; case QType::Q5_K: return "Q5_K"; case QType::Q6_K: return "Q6_K";
    case QType::IQ2_XXS: return "IQ2_XXS"; case QType::IQ1_S: return "IQ1_S"; case QType::IQ4_NL: return "IQ4_NL";
    case QType::IQ2_XS: return "IQ2_XS"; case QType::IQ1_M: return "IQ1_M";
    case QType::MXFP4: return "MXFP4";
  }
  return "?";
}

}  // namespace ms::q
