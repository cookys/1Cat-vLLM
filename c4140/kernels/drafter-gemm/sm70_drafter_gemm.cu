// sm70_drafter_gemm: SWAP weight-streaming wmma GEMM (mainloop = lmhead_gemm_swap<64,1> of sm70_lmhead_custom_v4.cu sha8 024eb504, byte-for-byte) with split-K over blockIdx.z,
// for the 7 DFlash2 drafter shapes (DESIGN-drafter-gemm-v1-fable-20261028.md sha8 4d9978d7, section 1).
// D(M,N) fp16 row-major = X(M,K) fp16 row-major . W(N,K)^T  (W K-contiguous: the live F.linear layout, UnquantizedLinearMethod, no bias).
// CTA tile 128 (M, padded, rows >= M zero-filled / never stored) x 128 (N) x BK64; 8 warps = 2 (M) x 4 (N); the HMMA roles are swapped (C^T = W . X^T): weight tile -> matrix_a row_major, activation tile -> matrix_b col_major.
// Split-K: CTA z of S handles K range [z*Ks, (z+1)*Ks), Ks = K / S, Ks % 64 == 0. The pointer shift X += z*Ks, W += z*Ks happens BEFORE the mainloop so the mainloop text is untouched (row stride stays K); only KT = Ks / BK differs from v4.
// Epilogue EPI == 0 (S == 1): fp32 accumulators -> fp16, 16-byte stores to out[M,N]; rows >= M never stored.
// Epilogue EPI == 1 (S  > 1): fp32 partial tile -> workspace [S, M, N] (plain 16-byte stores, no atomics); drafter_splitk_reduce then sums z = 0..S-1 IN ORDER in fp32 and stores fp16 (deterministic run to run; not bit-equal to S == 1, never claimed to be).
// No allocation, no sync, no host state on the launch path (CUDA-graph capturable). Host contract (checked, non-zero rc otherwise): 1 <= M <= 128, N % 128 == 0, K % 64 == 0, 1 <= S <= 32, K % S == 0, (K / S) % 64 == 0, X/W/out 16-byte aligned, workspace 32-byte aligned (S > 1).
#ifndef DG_CPU_EMU   // emu_dg_test.cpp defines DG_CPU_EMU plus shims for the CUDA/wmma names and includes this very file to run the kernel bodies on the CPU
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#endif
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int BM = 128, BN = 128, THREADS = 256;

template <int BK>
struct Cfg {
  static constexpr int LD = BK + 8;                      // half row pitch: +16 B pad, keeps 32-byte wmma pointer alignment and spreads banks
  static constexpr int A_ELEMS = BM * LD, B_ELEMS = BN * LD, STAGE = A_ELEMS + B_ELEMS;
  static constexpr int NV = BK / 16;                     // 16-byte vectors per thread per operand per K step (128 rows * BK/8 vectors / 256 threads)
  static constexpr int VPR = BK / 8;                     // vectors per row
  static constexpr int SMEM_BYTES = 2 * STAGE * 2;
};

// EPI 0: OUT is __half[M,N]; EPI 1: OUT is float[S,M,N] (partials). Xg / Wg are the unshifted base pointers.
template <int BK, int MINB, int EPI>
__global__ void __launch_bounds__(THREADS, MINB) drafter_gemm_swap(const __half* __restrict__ Xg, const __half* __restrict__ Wg, void* __restrict__ OUT, int M, int N, int K, int Ks) {
  using G = Cfg<BK>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* smem = reinterpret_cast<__half*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int KT = Ks / BK;
  const __half* __restrict__ X = Xg + (size_t)blockIdx.z * Ks;          // split-K: this CTA's K slice (row stride of both operands stays K)
  const __half* __restrict__ W = Wg + (size_t)blockIdx.z * Ks;

  // MAINLOOP-BEGIN (byte-identical to lmhead_gemm_swap<64,1> of the vendored sm70_lmhead_custom_v4.cu; checked by selftest_dg.py)
  uint4 ra[G::NV], rb[G::NV];
  auto load_g = [&](int kt) {
    const int k0 = kt * BK;
#pragma unroll
    for (int i = 0; i < G::NV; ++i) {
      const int v = tid + i * THREADS, row = v / G::VPR, c8 = (v % G::VPR) * 8;
      const int gm = m0 + row;
      ra[i] = gm < M ? __ldg(reinterpret_cast<const uint4*>(X + (size_t)gm * K + k0 + c8)) : make_uint4(0u, 0u, 0u, 0u);
      rb[i] = __ldcs(reinterpret_cast<const uint4*>(W + (size_t)(n0 + row) * K + k0 + c8));
    }
  };
  auto store_s = [&](int stage) {
    __half* As = smem + stage * G::STAGE;
    __half* Bs = As + G::A_ELEMS;
#pragma unroll
    for (int i = 0; i < G::NV; ++i) {
      const int v = tid + i * THREADS, row = v / G::VPR, c8 = (v % G::VPR) * 8;
      *reinterpret_cast<uint4*>(As + row * G::LD + c8) = ra[i];
      *reinterpret_cast<uint4*>(Bs + row * G::LD + c8) = rb[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];     // acc[i][j]: i over the 2 n-fragments, j over the 4 m-fragments (C^T tile: rows n, cols m)
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  load_g(0);
  store_s(0);
  __syncthreads();
  for (int kt = 0; kt < KT; ++kt) {
    if (kt + 1 < KT) load_g(kt + 1);
    const __half* As = smem + (kt & 1) * G::STAGE;
    const __half* Bs = As + G::A_ELEMS;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> wfrag[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(wfrag[i], Bs + (wn * 32 + i * 16) * G::LD + kk * 16, G::LD);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> afrag;
        wmma::load_matrix_sync(afrag, As + (wm * 64 + j * 16) * G::LD + kk * 16, G::LD);
#pragma unroll
        for (int i = 0; i < 2; ++i) wmma::mma_sync(acc[i][j], wfrag[i], afrag, acc[i][j]);
      }
    }
    if (kt + 1 < KT) store_s((kt + 1) & 1);
    __syncthreads();
  }
  // MAINLOOP-END

  // epilogue: each 16x16 fragment (C^T: rows n, cols m) is staged in this warp's 1 KB smem slice (smem is free: barrier above) with mem_col_major -> scratch[m_local * 16 + n_local],
  // then lane l copies row er = l >> 1, 8 consecutive n starting at ec = (l & 1) * 8 with 16-byte stores (fp16 for EPI 0, two fp32 float4 for EPI 1). Row-masked: rows >= M are never stored.
  float* scratch = reinterpret_cast<float*>(smem_raw) + warp * 256;
  const int lane = tid & 31;
  const int er = lane >> 1, ec = (lane & 1) * 8;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int row0 = m0 + wm * 64 + j * 16;
    if (row0 >= M) continue;                                            // warp-uniform
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int col0 = n0 + wn * 32 + i * 16;
      wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_col_major);   // scratch[m_local * 16 + n_local]
      __syncwarp();
      if (row0 + er < M) {
        const float4 lo = *reinterpret_cast<const float4*>(scratch + er * 16 + ec);
        const float4 hi = *reinterpret_cast<const float4*>(scratch + er * 16 + ec + 4);
        if (EPI == 0) {
          uint4 pk;
          pk.x = (unsigned)__half_as_ushort(__float2half_rn(lo.x)) | ((unsigned)__half_as_ushort(__float2half_rn(lo.y)) << 16);
          pk.y = (unsigned)__half_as_ushort(__float2half_rn(lo.z)) | ((unsigned)__half_as_ushort(__float2half_rn(lo.w)) << 16);
          pk.z = (unsigned)__half_as_ushort(__float2half_rn(hi.x)) | ((unsigned)__half_as_ushort(__float2half_rn(hi.y)) << 16);
          pk.w = (unsigned)__half_as_ushort(__float2half_rn(hi.z)) | ((unsigned)__half_as_ushort(__float2half_rn(hi.w)) << 16);
          *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(OUT) + (size_t)(row0 + er) * N + col0 + ec) = pk;
        } else {
          float* wp = reinterpret_cast<float*>(OUT) + (size_t)blockIdx.z * M * N + (size_t)(row0 + er) * N + col0 + ec;
          *reinterpret_cast<float4*>(wp) = lo;
          *reinterpret_cast<float4*>(wp + 4) = hi;
        }
      }
      __syncwarp();
    }
  }
}

// Fixed-order reduce: one thread per (m, 8 consecutive n); acc = 0, then acc += partial[z] for z = 0 .. S-1 in order (fp32), fp16 store (16 bytes).
__global__ void __launch_bounds__(256) drafter_splitk_reduce(const float* __restrict__ ws, __half* __restrict__ out, int M, int N, int S) {
  const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t total = (size_t)M * (size_t)(N / 8);
  if (t >= total) return;
  const size_t e = t * 8;                                               // element index in the row-major [M, N] output (N % 8 == 0: the 8 elements share a row)
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
  for (int z = 0; z < S; ++z) {
    const float4* p = reinterpret_cast<const float4*>(ws + (size_t)z * M * N + e);
    const float4 u = p[0], v = p[1];
    a0 += u.x; a1 += u.y; a2 += u.z; a3 += u.w; a4 += v.x; a5 += v.y; a6 += v.z; a7 += v.w;
  }
  uint4 pk;
  pk.x = (unsigned)__half_as_ushort(__float2half_rn(a0)) | ((unsigned)__half_as_ushort(__float2half_rn(a1)) << 16);
  pk.y = (unsigned)__half_as_ushort(__float2half_rn(a2)) | ((unsigned)__half_as_ushort(__float2half_rn(a3)) << 16);
  pk.z = (unsigned)__half_as_ushort(__float2half_rn(a4)) | ((unsigned)__half_as_ushort(__float2half_rn(a5)) << 16);
  pk.w = (unsigned)__half_as_ushort(__float2half_rn(a6)) | ((unsigned)__half_as_ushort(__float2half_rn(a7)) << 16);
  *reinterpret_cast<uint4*>(out + e) = pk;
}

#ifndef DG_CPU_EMU
struct Variant { const char* name; const void* fn; int threads, smem; };
const Variant kVariants[] = {
    {"S1-F16", (const void*)drafter_gemm_swap<64, 1, 0>, THREADS, Cfg<64>::SMEM_BYTES},
    {"SPLIT-PARTIAL", (const void*)drafter_gemm_swap<64, 1, 1>, THREADS, Cfg<64>::SMEM_BYTES},
    {"SPLIT-REDUCE", (const void*)drafter_splitk_reduce, 256, 0},
};
constexpr int NVAR = sizeof(kVariants) / sizeof(kVariants[0]);
bool g_inited = false;
#endif  // DG_CPU_EMU

}  // namespace

#ifndef DG_CPU_EMU
extern "C" {
int dg_version() { return 1; }
int dg_nvariants() { return NVAR; }
const char* dg_name(int v) { return (v >= 0 && v < NVAR) ? kVariants[v].name : ""; }
int dg_init() {
  for (int v = 0; v < 2; ++v) {
    cudaError_t e = cudaFuncSetAttribute(kVariants[v].fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kVariants[v].smem);
    if (e != cudaSuccess) return (int)e;
  }
  g_inited = true;
  return 0;
}
int dg_info(int v, int* out) {
  if (v < 0 || v >= NVAR) return -1;
  cudaFuncAttributes fa;
  cudaError_t e = cudaFuncGetAttributes(&fa, kVariants[v].fn);
  if (e != cudaSuccess) return (int)e;
  int nb = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kVariants[v].fn, kVariants[v].threads, kVariants[v].smem);
  if (e != cudaSuccess) return (int)e;
  out[0] = fa.numRegs; out[1] = (int)fa.localSizeBytes; out[2] = (int)fa.sharedSizeBytes; out[3] = kVariants[v].smem; out[4] = nb; out[5] = fa.maxThreadsPerBlock;
  return 0;
}
// workspace bytes the host must provide for (m, n, s): s * 128 * n * 4 (m is padded to the tile), 0 for s == 1. Allocate once per shape at bind time, never on the launch path.
unsigned long long dg_ws_bytes(int m, int n, int s) { (void)m; return s > 1 ? (unsigned long long)s * 128ull * (unsigned long long)n * 4ull : 0ull; }
// (X, W, out, workspace, M, N, K, S): rc 0 ok; -1 bad args; -2 not init; -3 alignment; -4 N/K contract; -5 M > 128; -6 split contract; -7 workspace missing (S > 1)
int dg_run(int m, int n, int k, int s, const void* x, const void* w, void* out, void* ws, void* stream) {
  if (m <= 0 || n <= 0 || k <= 0 || s <= 0 || !x || !w || !out) return -1;
  if (!g_inited) return -2;
  if (((uintptr_t)x | (uintptr_t)w | (uintptr_t)out) & 15) return -3;
  if (n % BN || k % 64) return -4;
  if (m > BM) return -5;
  if (s > 32 || k % s || (k / s) % 64) return -6;
  if (s > 1) { if (!ws) return -7; if ((uintptr_t)ws & 31) return -3; }
  const int ks = k / s;
  dim3 grid(n / BN, 1, s), block(THREADS);
  const __half* xx = (const __half*)x; const __half* ww = (const __half*)w; cudaStream_t st = (cudaStream_t)stream;
  if (s == 1) {
    drafter_gemm_swap<64, 1, 0><<<grid, block, Cfg<64>::SMEM_BYTES, st>>>(xx, ww, out, m, n, k, ks);
  } else {
    drafter_gemm_swap<64, 1, 1><<<grid, block, Cfg<64>::SMEM_BYTES, st>>>(xx, ww, ws, m, n, k, ks);
    const unsigned long long tot = (unsigned long long)m * (unsigned long long)(n / 8);
    drafter_splitk_reduce<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>((const float*)ws, (__half*)out, m, n, s);
  }
  return (int)cudaPeekAtLastError();
}
}  // extern "C"
#endif  // DG_CPU_EMU
