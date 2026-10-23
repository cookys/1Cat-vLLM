#pragma once
// PTX ISA m8n8k4 fragment layout, FP16 inputs / FP32 C and D.
// These helpers are also compiled by the pure-host mapping test.
#ifdef __CUDACC__
#define PVH_HD __host__ __device__
#else
#define PVH_HD
#endif
namespace pvh {
constexpr int kM = 8, kN = 64, kD = 256, kThreads = 256;
constexpr int kQStride = 272, kKVStride = 272, kPStride = 72;
PVH_HD constexpr int group(int lane) { return (lane >> 2) & 3; }
PVH_HD constexpr int a_row(int lane) { return (lane & 3) + (lane / 16) * 4; }
PVH_HD constexpr int b_row(int lane) { return lane & 3; }
PVH_HD constexpr int b_col4(int lane) { return (lane / 16) * 4; }
PVH_HD constexpr int c_row(int lane, int i) {
  return (lane & 1) + (i & 2) + (lane / 16) * 4;
}
PVH_HD constexpr int c_col(int lane, int i) {
  return (i & 4) + (lane & 2) + (i & 1);
}
PVH_HD constexpr int qk_token(int warp, int lane, int col) {
  return warp * 16 + (group(lane) & 1) * 8 + col;
}
PVH_HD constexpr int qk_dim(int lane, int k) { return (group(lane) >> 1) * 128 + k; }
PVH_HD constexpr int pv_dim(int warp, int fragment, int lane, int col) {
  return (warp + fragment * 8) * 32 + group(lane) * 8 + col;
}
}  // namespace pvh
#undef PVH_HD
