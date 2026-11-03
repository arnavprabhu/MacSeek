// Layout probes for the MoE prefill register kernels (bench_moep_probe): dump the cooperative destination layout of
// matmul2d<(TM, TN, TK, transpose_right), execution_simdgroup> (out[lane * 256 + i] = row * 4096 + token, -1 invalid).
#include "quant_tile.h"

template <int TM, int TN, int TK>
inline void q_moep_probe_dst(device half* X, device int* out, ushort lane) {
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tX(X, dextents<int32_t, 2>(TK, TM));
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto w = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(tX), decltype(w), float>();
  for (int i = 0; i < 256; i++) out[lane * 256 + i] = -2;
  for (uint16_t i = 0; i < c.get_capacity(); ++i) {
    int v = -1;
    if (c.is_valid_element(i)) { auto id = c.get_multidimensional_index(i); v = id[0] * 4096 + id[1]; }
    out[lane * 256 + i] = v;
  }
}
kernel void q_moep_probe_dst_64x32(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                   ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<64, 32, 32>(X, out, lane); }
kernel void q_moep_probe_dst_64x16(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                   ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<64, 16, 32>(X, out, lane); }
kernel void q_moep_probe_dst_32x64(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                   ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<32, 64, 32>(X, out, lane); }
kernel void q_moep_probe_dst_128x16(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                    ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<128, 16, 32>(X, out, lane); }
kernel void q_moep_probe_dst_48x32(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                   ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<48, 32, 32>(X, out, lane); }
kernel void q_moep_probe_dst_16x64(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                   ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<16, 64, 32>(X, out, lane); }
kernel void q_moep_probe_dst_64x64(device half* X [[buffer(0)]], device int* out [[buffer(1)]],
                                   ushort lane [[thread_index_in_simdgroup]]) { q_moep_probe_dst<64, 64, 32>(X, out, lane); }
