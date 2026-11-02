// sm70_lmhead_lt: tiny cublasLt wrapper for the lm_head fp16 x fp16 -> fp32 GEMM (bundle-3 lever 2). Plain C ABI (ctypes), no torch dependency, no CUDA kernels (host code only).
// Computes D(M,N) fp32 row-major = X(M,K) fp16 row-major . W(N,K)^T, exactly the layout of P0-LM p0_cublaslt.py (col-major view: D^T(N,M) = op_T(W as (K,N) ld K) . X as (K,M) ld K, compute 32F).
// Capture safety: lt_init / lt_add create the handle, descriptors and layouts; lt_run only calls cublasLtMatmul with caller-owned workspace on the caller's stream: no allocation, no sync, no host-pointer state beyond static alpha/beta.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <map>

namespace {
struct Entry {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t ws = 0;
};
cublasLtHandle_t g_handle = nullptr;
int g_n = 0, g_k = 0;
std::map<int, Entry> g_entries;
const float g_alpha = 1.0f, g_beta = 0.0f;
}  // namespace

extern "C" {

int lt_version() { return (int)cublasLtGetVersion(); }

// 0 ok; non-zero = cublasStatus_t (or -1 bad args). Idempotent for the same (n, k).
int lt_init(int n, int k) {
  if (n <= 0 || k <= 0) return -1;
  if (g_handle) return (g_n == n && g_k == k) ? 0 : -2;
  cublasStatus_t st = cublasLtCreate(&g_handle);
  if (st != CUBLAS_STATUS_SUCCESS) { g_handle = nullptr; return (int)st; }
  g_n = n; g_k = k;
  return 0;
}

// Register shape M with a pinned algo (the 8 x uint64 cublasLtMatmulAlgo_t). Validates it with cublasLtMatmulAlgoCheck.
// 0 ok; else cublasStatus_t, or 1000+state when the check says the algo cannot run this problem.
int lt_add(int m, const uint64_t* algo8) {
  if (!g_handle || m <= 0) return -1;
  if (g_entries.count(m)) return 0;
  Entry e;
  std::memcpy(e.algo.data, algo8, sizeof(e.algo.data));
  cublasStatus_t st;
  if ((st = cublasLtMatmulDescCreate(&e.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)) != CUBLAS_STATUS_SUCCESS) return (int)st;
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  if ((st = cublasLtMatmulDescSetAttribute(e.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT))) != CUBLAS_STATUS_SUCCESS) return (int)st;
  if ((st = cublasLtMatmulDescSetAttribute(e.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))) != CUBLAS_STATUS_SUCCESS) return (int)st;
  if ((st = cublasLtMatrixLayoutCreate(&e.a, CUDA_R_16F, g_k, g_n, g_k)) != CUBLAS_STATUS_SUCCESS) return (int)st;
  if ((st = cublasLtMatrixLayoutCreate(&e.b, CUDA_R_16F, g_k, m, g_k)) != CUBLAS_STATUS_SUCCESS) return (int)st;
  if ((st = cublasLtMatrixLayoutCreate(&e.c, CUDA_R_32F, g_n, m, g_n)) != CUBLAS_STATUS_SUCCESS) return (int)st;
  cublasLtMatmulHeuristicResult_t hr;
  std::memset(&hr, 0, sizeof(hr));
  st = cublasLtMatmulAlgoCheck(g_handle, e.desc, e.a, e.b, e.c, e.c, &e.algo, &hr);
  if (st != CUBLAS_STATUS_SUCCESS) return (int)st;
  if (hr.state != CUBLAS_STATUS_SUCCESS) return 1000 + (int)hr.state;
  e.ws = hr.workspaceSize;
  g_entries[m] = e;
  return 0;
}

// Workspace bytes the validated algo needs for this M (0 = none), or (size_t)-1 when M is not registered.
size_t lt_ws_bytes(int m) {
  auto it = g_entries.find(m);
  return it == g_entries.end() ? (size_t)-1 : it->second.ws;
}

int lt_has(int m) { return g_entries.count(m) ? 1 : 0; }

// One matmul on `stream`. Returns 0 ok, -1 not registered, -3 workspace too small, else cublasStatus_t.
int lt_run(int m, const void* w, const void* x, void* out, void* ws, size_t ws_cap, void* stream) {
  auto it = g_entries.find(m);
  if (it == g_entries.end() || !g_handle) return -1;
  Entry& e = it->second;
  if (e.ws > ws_cap) return -3;
  return (int)cublasLtMatmul(g_handle, e.desc, &g_alpha, w, e.a, x, e.b, &g_beta, out, e.c, out, e.c, &e.algo, ws, e.ws, (cudaStream_t)stream);
}

}  // extern "C"
