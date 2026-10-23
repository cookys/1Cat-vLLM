#pragma once
#include <stdint.h>
#if defined(__CUDACC__)
#define PVH_LOADER_HD __host__ __device__
#else
#define PVH_LOADER_HD
#endif
namespace pvh_loader {
constexpr int kThreads = 256;
constexpr int kSlots = 2;
constexpr int kRows = 64;
constexpr int kUnitsPerRow = 8;
constexpr int kDimsPerUnit = 32;
constexpr int kSmemStride = 272;
constexpr int kPageTokens = 4096;
constexpr int kDataBytesPerToken = 128;
constexpr int kScaleBytesPerToken = 16;
PVH_LOADER_HD constexpr int unit_index(int tid, int slot) { return tid + slot * kThreads; }
PVH_LOADER_HD constexpr int row(int unit) { return unit / kUnitsPerRow; }
PVH_LOADER_HD constexpr int unit_in_row(int unit) { return unit % kUnitsPerRow; }
PVH_LOADER_HD constexpr bool active(int unit, int valid_rows) { return row(unit) < valid_rows; }
PVH_LOADER_HD constexpr int64_t data_in_page_offset(int token_in_page, int unit_in_token) {
  return int64_t(token_in_page) * 128 + unit_in_token * 16;
}
PVH_LOADER_HD constexpr int64_t scale_in_page_offset(int token_in_page, int unit_in_token) {
  return int64_t(4096) * 128 + int64_t(token_in_page) * 16 + unit_in_token * 2;
}
// Offsets relative to the current K-side or V-side base pointer. For an
// interleaved [page, side, bytes] allocation, page_stride = 2*4096*144.
PVH_LOADER_HD constexpr int64_t data_offset(int physical_page, int64_t page_stride,
                                           int token_in_page, int unit_in_token) {
  return int64_t(physical_page) * page_stride + data_in_page_offset(token_in_page, unit_in_token);
}
PVH_LOADER_HD constexpr int64_t scale_offset(int physical_page, int64_t page_stride,
                                            int token_in_page, int unit_in_token) {
  return int64_t(physical_page) * page_stride + scale_in_page_offset(token_in_page, unit_in_token);
}
PVH_LOADER_HD constexpr int smem_half_offset(int tile_row, int unit_in_token, int element) {
  return tile_row * kSmemStride + unit_in_token * kDimsPerUnit + element;
}
}  // namespace pvh_loader
#undef PVH_LOADER_HD
