// Exploratory nospec NVFP4 G6 kernel: full-D shared tile, PV tensor cores.
// Included inside flash_decode_paged.cu's anonymous namespace.
#include "pv_hmma_layout.h"
#include "pvh_loader.cuh"

struct alignas(256) PVHSmem {
  alignas(16) __half q[8 * 272];
  alignas(16) __half kv[64 * 272];  // K and V lifetimes do not overlap.
  alignas(16) float score[8 * 64];
  alignas(16) __half prob[8 * 72];
  alignas(16) float row_max[8], row_sum[8], alpha[8];
  alignas(16) int page_ids[32];
  volatile int part_tokens, num_tiles;  // CTA-uniform, in alignment slack.
};
static_assert(sizeof(PVHSmem) == 42752, "PVH shared budget changed");

#define PVH_MMA_F32(NAME, BLAY) \
__device__ __forceinline__ void NAME(float (&c)[8], uint2 a, uint2 b) { \
  asm volatile( \
      "mma.sync.aligned.m8n8k4.row." BLAY ".f32.f16.f16.f32 " \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, " \
      "{%0,%1,%2,%3,%4,%5,%6,%7};" \
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), \
        "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) \
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y)); \
}
PVH_MMA_F32(pvh_mma_rr, "row")
PVH_MMA_F32(pvh_mma_rc, "col")
#undef PVH_MMA_F32

__device__ __forceinline__ uint2 pvh_load4(const __half* p) {
  return *reinterpret_cast<const uint2*>(p);
}

template <int FRAGMENT>
__device__ __forceinline__ void pvh_pv(float (&carry)[8], PVHSmem& s,
                                     int warp, int lane) {
  if (warp + FRAGMENT * 8 < 8) {  // Warp-uniform, including every mma lane.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      carry[i] *= s.alpha[pvh::c_row(lane, i)];
    }
#pragma unroll
    for (int k = 0; k < 64; k += 4) {
      const uint2 a = pvh_load4(s.prob + pvh::a_row(lane) * 72 + k);
      const uint2 b = pvh_load4(s.kv + (k + pvh::b_row(lane)) * 272 +
          pvh::pv_dim(warp, FRAGMENT, lane, pvh::b_col4(lane)));
      pvh_mma_rr(carry, a, b);
    }
  }
}

template <int FRAGMENT>
__device__ __forceinline__ void pvh_store(
    const float (&carry)[8], const PVHSmem& s, __half* out,
    int warp, int lane, int batch, int part,
    int64_t stride_b, int64_t stride_h, int64_t stride_p, float v_scale) {
  if (warp + FRAGMENT * 8 < 8) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = pvh::c_row(lane, i);
      if (row < 6) {
        const float inv = s.row_sum[row] > 0.f ? 1.f / s.row_sum[row] : 0.f;
        const int d = pvh::pv_dim(warp, FRAGMENT, lane, pvh::c_col(lane, i));
        out[static_cast<int64_t>(batch) * stride_b + row * stride_h +
            part * stride_p + d] = __float2half(carry[i] * (inv * v_scale));
      }
    }
  }
}

template <int PARTITION_SIZE>
__global__ void __launch_bounds__(256, 2)
flash_attention_decode_pvh_partition_kernel_256_wide(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, __half* __restrict__ tmp_out,
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, const int batch_size,
    const int max_num_blocks, const int max_num_partitions,
    const int num_heads_q, const int num_heads_kv, const int block_size,
    const int64_t q_stride0, const int64_t q_stride1,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const int64_t v_block_stride, const int64_t v_token_stride,
    const int64_t v_head_stride, const float softmax_scale,
    const float k_scale, const float v_scale, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  static_assert(PARTITION_SIZE == 256 || PARTITION_SIZE == 1024,
                "PVH prototype only P256/P1024");
  const int batch = blockIdx.x, part = blockIdx.z;
  if (batch >= batch_size || blockIdx.y != 0 || part >= max_num_partitions) return;
  const int seq_len = seq_lens[batch];
  const int start = part * PARTITION_SIZE;
  if (seq_len <= 0 || start >= seq_len) return;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  extern __shared__ char pvh_smem[];
  auto& s = *reinterpret_cast<PVHSmem*>(pvh_smem);
  // Host guard pins block4096, Hkv1, token-stride144. Each P fits one page.
  if (tid == 0) {
    s.part_tokens = min(PARTITION_SIZE, seq_len - start);
    s.num_tiles = (s.part_tokens + 63) / 64;
    s.page_ids[0] = __ldg(block_table + batch * max_num_blocks + start / 4096);
  }
  if (tid < 8) {
    s.row_max[tid] = kXQANegInf;
    s.row_sum[tid] = 0.f;
    s.alpha[tid] = 1.f;
  }
  // Six heads * 32 uint4; Q loaded once for the whole partition.
  const int qrow = tid / 32, qvec = tid % 32;
  if (tid < 192) {  // Six real Q heads; the other two warps are PV/load workers.
    reinterpret_cast<uint4*>(s.q)[qrow * 34 + qvec] =
        __ldg(reinterpret_cast<const uint4*>(q + batch * q_stride0 + qrow * q_stride1) + qvec);
  }
  if (tid < 64) {
    reinterpret_cast<uint4*>(s.q)[(6 + tid / 32) * 34 + tid % 32] = make_uint4(0,0,0,0);
  }
  for (int i = tid; i < 2 * 72; i += 256) s.prob[6 * 72 + i] = __float2half(0.f);
  float carry0[8] = {};
  __syncthreads();  // Initial Q, metadata, padded P publication.

  // P256/P1024 each lies inside one page4096; physical page and side stride
  // are invariant for the entire partition, including its final partial tile.
  const auto* k_page = reinterpret_cast<const uint8_t*>(k_cache) +
      static_cast<int64_t>(s.page_ids[0]) * k_block_stride;
  const auto* v_page = reinterpret_cast<const uint8_t*>(v_cache) +
      static_cast<int64_t>(s.page_ids[0]) * v_block_stride;
  for (int tile = 0; tile < s.num_tiles; ++tile) {
    const int valid = min(64, s.part_tokens - tile * 64);
    const int tile_offset = (start & 4095) + tile * 64;
    {
      const PVHRawPair raw_k = valid == 64
          ? pvh_load_pair<true>(k_page, tile_offset, valid, tid)
          : pvh_load_pair<false>(k_page, tile_offset, valid, tid);
      pvh_store_pair(s.kv, raw_k, tid);
    }
    __syncthreads();  // 1. K ready.
    // Explicit two raw slots, issued before QK; no shared K overwrite here.
    const PVHRawPair raw_v = valid == 64
        ? pvh_load_pair<true>(v_page, tile_offset, valid, tid)
        : pvh_load_pair<false>(v_page, tile_offset, valid, tid);
    if (warp < 4) {
      // Four independent m8n8 products: N8 x two columns x two D128 sums.
      // D is split only in register computation; shared/global D256 stays whole.
      float acc[8] = {};
#pragma unroll
      for (int dk = 0; dk < 128; dk += 4) {
        const uint2 a = pvh_load4(s.q + pvh::a_row(lane) * 272 + pvh::qk_dim(lane, dk));
        const int token = pvh::qk_token(warp, lane, pvh::a_row(lane));
        const uint2 b = pvh_load4(s.kv + token * 272 + pvh::qk_dim(lane, dk));
        pvh_mma_rc(acc, a, b);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float score = acc[i] + __shfl_xor_sync(0xffffffffu, acc[i], 8);
        if (pvh::group(lane) < 2) {
          s.score[pvh::c_row(lane, i) * 64 +
                  pvh::qk_token(warp, lane, pvh::c_col(lane, i))] =
              score * (softmax_scale * k_scale);
        }
      }
    }
    __syncthreads();  // 2. QK done; K storage can be reused.
    if (warp < 6) {
      const int row = warp;  // Only the six real heads update softmax state.
      const float x0 = lane < valid ? s.score[row * 64 + lane] : kXQANegInf;
      const float x1 = lane + 32 < valid ? s.score[row * 64 + lane + 32] : kXQANegInf;
      float mx = fmaxf(x0, x1);
#pragma unroll
      for (int delta = 16; delta; delta >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, delta));
      mx = __shfl_sync(0xffffffffu, mx, 0);
      const float old_max = s.row_max[row], old_sum = s.row_sum[row];
      const float new_max = fmaxf(old_max, mx);
      const float alpha = __expf(old_max - new_max);
      const float e0 = lane < valid ? __expf(fmaxf(x0 - new_max, -80.f)) : 0.f;
      const float e1 = lane + 32 < valid ? __expf(fmaxf(x1 - new_max, -80.f)) : 0.f;
      s.prob[row * 72 + lane] = __float2half_rn(e0);
      s.prob[row * 72 + lane + 32] = __float2half_rn(e1);
      float sum = e0 + e1;
#pragma unroll
      for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
      __syncwarp();
      if (lane == 0) {
        s.row_max[row] = new_max;
        s.row_sum[row] = alpha * old_sum + sum;
        s.alpha[row] = alpha;
      }
    }
    // P/alpha and V live in disjoint storage. QK's barrier already released
    // K; publish the softmax state and dequantized V together, exactly once.
    pvh_store_pair(s.kv, raw_v, tid);
    __syncthreads();  // 3. P, alpha and V ready.
    pvh_pv<0>(carry0, s, warp, lane);
    __syncthreads();  // 4. PV done; next tile can overwrite V and P.
  }
  pvh_store<0>(carry0, s, tmp_out, warp, lane, batch, part,
              tmp_out_stride0, tmp_out_stride1, tmp_out_stride2, v_scale);
  if (tid < 6) {
    const int64_t index = static_cast<int64_t>(batch) * stats_stride0 + tid * stats_stride1 + part;
    max_logits[index] = s.row_max[tid];
    exp_sums[index] = s.row_sum[tid];
  }
}

// PVH_HOST_LAUNCH
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM,
          int NUM_THREADS = kXQATC256WideThreads, int MIN_BLOCKS_PER_SM = 1,
          int BLOCK_SIZE = 0, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool ALIGNED_PADDED_SMEM = false,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          int KV_DTYPE_OVERRIDE = -1, bool E4M3_SHARED_LUT = false,
          typename PARTIAL_T = std::conditional_t<
              KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3, float,
              __half>,
          bool NVFP4_HALF_P = false>
void launch_flash_attention_decode_pvh_256_wide(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const float k_scale, const float v_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true, int64_t* route_info = nullptr) {
  static_assert(!E4M3_SHARED_LUT ||
                    KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
                "The shared conversion LUT requires an E4M3 specialization");
  static_assert(!NVFP4_HALF_P ||
                    KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_NVFP4,
                "NVFP4_HALF_P requires an NVFP4 specialization");
  static_assert(KV_DTYPE_OVERRIDE != flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                    std::is_same_v<PARTIAL_T, float>,
                "E4M3 XQA must preserve partition outputs in FP32");
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  // ATTN-18: host evidence from the exact immutable launch grid. This CPU
  // out-arg is never passed to a kernel, copied from CUDA, or captured as work.
  if (route_info != nullptr) {
    cudaStreamCaptureStatus capture_status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    route_info[0] = partition_grid.x;
    route_info[1] = PARTITION_SIZE;
    route_info[2] = partition_grid.z;
    route_info[3] = capture_status != cudaStreamCaptureStatusNone;
  }

  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  const size_t shared_mem =
      sizeof(PVHSmem) + (E4M3_SHARED_LUT ? 256 * sizeof(uint16_t) : 0);
#define LAUNCH_XQA_PARTITION(KV_DTYPE)                                         \
  do {                                                                         \
    auto partition_kernel =                                                    \
        (void*)flash_attention_decode_pvh_partition_kernel_256_wide<PARTITION_SIZE>;                                                     \
    cudaFuncSetAttribute(partition_kernel,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         shared_mem);                                          \
    if constexpr (MIN_BLOCKS_PER_SM > 1) {                                     \
      cudaFuncSetAttribute(partition_kernel,                                   \
                           cudaFuncAttributePreferredSharedMemoryCarveout,     \
                           100);                                               \
    }                                                                          \
    flash_attention_decode_pvh_partition_kernel_256_wide<PARTITION_SIZE>                                                          \
        <<<partition_grid, 256, shared_mem, stream>>>(                 \
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),           \
            k_cache.data_ptr(), v_cache.data_ptr(),                            \
            reinterpret_cast<PARTIAL_T*>(tmp_out.data_ptr()),                  \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),          \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),             \
            active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks, \
            launch_num_partitions, num_heads_q, num_heads_kv, k_cache.size(1), \
            q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),    \
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),     \
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),           \
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),           \
            softmax_scale, k_scale, v_scale, route_seq_len_begin,              \
            route_seq_len_end, route_seq_len_final);                           \
  } while (0)

  if constexpr (KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_NVFP4) {
    TORCH_CHECK(k_cache.scalar_type() == at::kByte,
                "NVFP4 XQA requires uint8 KV cache");
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_NVFP4);
  } else if constexpr (KV_DTYPE_OVERRIDE ==
                       flash_v100::KV_CACHE_DTYPE_FP8_E4M3) {
    static_assert(!(FP8_PAIR_LOAD && E4M3_SHARED_LUT),
                  "Paired E4M3 conversion and the shared LUT are exclusive");
    TORCH_CHECK(k_cache.scalar_type() == at::kByte,
                "E4M3 XQA requires uint8 KV cache");
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E4M3);
  } else if constexpr (FP8_PAIR_LOAD) {
    static_assert(!E4M3_SHARED_LUT, "The E4M3 conversion LUT requires E4M3 KV");
    TORCH_CHECK(k_cache.scalar_type() == at::kByte,
                "Paired XQA loads require uint8 FP8 KV cache");
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E5M2);
  } else if (k_cache.scalar_type() == at::kByte) {
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E5M2);
  } else {
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP16);
  }
#undef LAUNCH_XQA_PARTITION
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (!launch_reduce) {
    return;
  }

  if constexpr (PARTITION_SIZE == -1) {
    TORCH_CHECK(!use_split_reduce,
                "Runtime wave partitions do not support split reduction");
  }
  if (use_split_reduce) {
    if constexpr (PARTITION_SIZE != -1) {
      launch_flash_attention_decode_xqa_split_reduce<PARTITION_SIZE,
                                                     SEQ_LEN_ROUTE, PARTIAL_T>(
          out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
          split_reduce_dim_tile, stream);
    }
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, PARTITION_SIZE, SEQ_LEN_ROUTE,
                                         PARTIAL_T>
        <<<reduce_grid, block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const PARTIAL_T*>(tmp_out.data_ptr()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            launch_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1),
            route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin,
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
