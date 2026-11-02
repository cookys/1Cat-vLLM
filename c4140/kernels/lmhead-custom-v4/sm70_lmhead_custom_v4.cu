// sm70_lmhead_custom: weight-streaming wmma GEMM for the lm_head (pool candidate "custom lm_head GEMM", DESIGN-lmhead-custom-fable-20261028).
// D(M,N) fp32 row-major = X(M,K) fp16 row-major . W(N,K)^T  (W K-contiguous, exactly the live torch.mm(x, W.t(), out_dtype=float32) layout; vocab_parallel_embedding.py:443).
// CTA tile 128 (M, padded, rows >= M zero-filled / never stored) x 128 (N) x BK; 8 warps = 2 (M) x 4 (N); each warp owns 64 x 32 of C = 4 x 2 wmma 16x16x16 fp32 accumulators.
// A and B K-slices are staged through registers (LDG.128 -> STS.128, no cp.async on sm_70) into a 2-stage shared double buffer, one __syncthreads per K step.
// W is read with the streaming (evict-first) hint so the 1.3 MB A operand stays L2 resident. General M by M-tiles on blockIdx.y. No allocation, no sync, no host state on the launch path (CUDA-graph capturable).
// Host contract (checked, non-zero rc otherwise): N % 128 == 0, K % BK == 0, K % 8 == 0, X/W 16-byte aligned, out 32-byte aligned.
#ifndef LMCU_CPU_EMU   // emu_kernel_test.cpp defines LMCU_CPU_EMU plus shims for the CUDA/wmma names and includes this very file to run the kernel body on the CPU
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

template <int BK, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) lmhead_gemm(const __half* __restrict__ X, const __half* __restrict__ W, float* __restrict__ C, int M, int N, int K) {
  using G = Cfg<BK>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* smem = reinterpret_cast<__half*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int KT = K / BK;

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

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  load_g(0);
  store_s(0);
  __syncthreads();
  for (int kt = 0; kt < KT; ++kt) {
    if (kt + 1 < KT) load_g(kt + 1);                                   // next K step's global loads in flight while this one computes
    const __half* As = smem + (kt & 1) * G::STAGE;
    const __half* Bs = As + G::A_ELEMS;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bs + (wn * 32 + j * 16) * G::LD + kk * 16, G::LD);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::load_matrix_sync(a, As + (wm * 64 + i * 16) * G::LD + kk * 16, G::LD);
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a, b[j], acc[i][j]);
      }
    }
    if (kt + 1 < KT) store_s((kt + 1) & 1);                            // other buffer: last read one barrier ago
    __syncthreads();
  }

  // epilogue: fp32 accumulators straight to C. 16-row blocks fully below M go through store_matrix_sync; the block that straddles M is staged in this warp's 1 KB smem slice (smem is free: barrier above) and copied row-masked.
  float* scratch = reinterpret_cast<float*>(smem_raw) + warp * 256;
  const int lane = tid & 31;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row0 = m0 + wm * 64 + i * 16;
    if (row0 >= M) continue;                                            // warp-uniform
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col0 = n0 + wn * 32 + j * 16;
      if (row0 + 16 <= M) {
        wmma::store_matrix_sync(C + (size_t)row0 * N + col0, acc[i][j], N, wmma::mem_row_major);
      } else {
        wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
          const int r = e >> 4, c = e & 15;
          if (row0 + r < M) C[(size_t)(row0 + r) * N + col0 + c] = scratch[e];
        }
        __syncwarp();
      }
    }
  }
}

// ---- v4: SWAP. lmhead_gemm above is CUSTOM-B, unchanged (lines 1-114 are byte-identical to lmhead-custom-v3 / v2).
// RULING-lmhead-frag-probe-v1 (sections 2-3): on sm70 the matrix_a row_major fragment load of an activation tile at pitch 72 is 2-way bank conflicted (16 wavefronts, 8 excess per fragment);
// matrix_b col_major from the same memory shape is clean (8 wavefronts, 0 excess). SWAP computes C^T = W . X^T, so the activation tile As[m][k] feeds matrix_b (col_major, ldm 72) and the weight tile Bs[n][k] feeds matrix_a (row_major, ldm 72).
//   weight fragment    wfrag[i] = load_matrix_sync(matrix_a, row_major, ldm 72) from &Bs[n0 + 16*i][k0]  == lmhead-frag-probe-v1 kind a at pitch 72 (16 wavefronts / 8 excess per fragment, measured)
//   activation fragment afrag[j] = load_matrix_sync(matrix_b, col_major, ldm 72) from &As[m0 + 16*j][k0]  == lmhead-frag-probe-v1 kind d at pitch 72 ( 8 wavefronts / 0 excess per fragment, measured)
//   mma_sync(acc[i][j], wfrag[i], afrag[j], acc[i][j]) gives the 16x16 tile of C^T (rows = n, cols = m); epilogue store_matrix_sync(..., mem_col_major) puts element (r = n_local, c = m_local) at C[m0+16j+c][n0+16i+r] (row-major C, ldc = N).
// Per warp-k-step: 2 weight fragments x 16 + 4 activation fragments x 8 = 64 shared-load wavefronts (CUSTOM-B: 80), excess 16 (CUSTOM-B: 32). Tiling, smem layout / pitch, global->shared staging, k loop and launch geometry are CUSTOM-B's.
// The HMMA role swap may change the fp32 accumulation order: bit-equality to CUSTOM-B is NOT guaranteed on the GPU (the CPU emulator is sequential and checks the index mapping).
template <int BK, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) lmhead_gemm_swap(const __half* __restrict__ X, const __half* __restrict__ W, float* __restrict__ C, int M, int N, int K) {
  using G = Cfg<BK>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* smem = reinterpret_cast<__half*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int KT = K / BK;

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

  // epilogue: same shape as CUSTOM-B (16-row blocks of C fully below M go straight to global, the block straddling M is staged in the warp's 1 KB smem slice and copied row-masked); the accumulator holds C^T so the stores are mem_col_major
  float* scratch = reinterpret_cast<float*>(smem_raw) + warp * 256;
  const int lane = tid & 31;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int row0 = m0 + wm * 64 + j * 16;
    if (row0 >= M) continue;                                            // warp-uniform
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int col0 = n0 + wn * 32 + i * 16;
      if (row0 + 16 <= M) {
        wmma::store_matrix_sync(C + (size_t)row0 * N + col0, acc[i][j], N, wmma::mem_col_major);
      } else {
        wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_col_major);   // scratch[m_local * 16 + n_local]
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
          const int r = e >> 4, c = e & 15;
          if (row0 + r < M) C[(size_t)(row0 + r) * N + col0 + c] = scratch[e];
        }
        __syncwarp();
      }
    }
  }
}

#ifndef LMCU_CPU_EMU
struct Variant { const char* name; const void* fn; int bk, bn, threads, smem; };
const Variant kVariants[] = {
    {"CUSTOM-B", (const void*)lmhead_gemm<64, 1>, 64, 128, 256, Cfg<64>::SMEM_BYTES},
    {"SWAP", (const void*)lmhead_gemm_swap<64, 1>, 64, 128, 256, Cfg<64>::SMEM_BYTES},
};
constexpr int NVAR = sizeof(kVariants) / sizeof(kVariants[0]);
bool g_inited = false;
#endif  // LMCU_CPU_EMU

}  // namespace

#ifndef LMCU_CPU_EMU
extern "C" {
int lmcu_version() { return 4; }
int lmcu_nvariants() { return NVAR; }
const char* lmcu_name(int v) { return (v >= 0 && v < NVAR) ? kVariants[v].name : ""; }
int lmcu_init() {
  for (int v = 0; v < NVAR; ++v) {
    cudaError_t e = cudaFuncSetAttribute(kVariants[v].fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kVariants[v].smem);
    if (e != cudaSuccess) return (int)e;
  }
  g_inited = true;
  return 0;
}
int lmcu_info(int v, int* out) {
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
// host contract: N % 128 == 0, K % 64 == 0, K % 8 == 0, X/W 16-byte aligned, out 32-byte aligned
int lmcu_run(int v, int m, int n, int k, const void* x, const void* w, void* out, void* stream) {
  if (v < 0 || v >= NVAR || m <= 0 || n <= 0 || k <= 0 || !x || !w || !out) return -1;
  if (!g_inited) return -2;
  const Variant& V = kVariants[v];
  if (((uintptr_t)x | (uintptr_t)w) & 15 || ((uintptr_t)out) & 31) return -3;
  if (n % V.bn || k % V.bk || k % 8) return -4;
  dim3 grid(n / V.bn, (m + BM - 1) / BM), block(V.threads);
  const __half* xx = (const __half*)x; const __half* ww = (const __half*)w; float* oo = (float*)out; cudaStream_t s = (cudaStream_t)stream;
  switch (v) {
    case 0: lmhead_gemm<64, 1><<<grid, block, V.smem, s>>>(xx, ww, oo, m, n, k); break;
    default: lmhead_gemm_swap<64, 1><<<grid, block, V.smem, s>>>(xx, ww, oo, m, n, k); break;
  }
  return (int)cudaPeekAtLastError();
}
}  // extern "C"
#endif  // LMCU_CPU_EMU
