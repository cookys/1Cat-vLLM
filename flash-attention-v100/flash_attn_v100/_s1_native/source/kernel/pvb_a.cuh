namespace pvb_a {
#include "pvb_loader.cuh"
constexpr int kGroupedVerifyQ8MaxQ = 8;
constexpr int kGroupedVerifyQ16MaxQ = 16;
constexpr int kGroupedVerifyMaxSupportedQ = kGroupedVerifyQ16MaxQ;
constexpr int kGroupedVerifyHeads = 6;
constexpr int kGroupedVerifyHeadDim = 256;
constexpr int kGroupedVerifyRows = 48;
constexpr int kGroupedVerifyBlockN = 32;
constexpr int kGroupedVerifyQStride = 264;
constexpr int kGroupedVerifyKVStride = 264;
// 32 floats = 128 B puts every score row in bank 0 for store_matrix_sync; 36
// keeps the 16-byte ldm alignment wmma requires (multiple of 4 floats) and
// spreads rows over banks.
constexpr int kGroupedVerifyScoreStride = 36;
constexpr int kGroupedVerifyProbStride = 40;
constexpr int kGroupedVerifyPageIdsCapacity = 16;
// One packed head group times eighty context splits maps one 512-thread CTA to
// each of V100's eighty SMs. Short sequences still reduce active_splits using
// kGroupedVerifyMinTokensPerSplit inside the captured graph.
constexpr int kGroupedVerifyQ8Splits = 80;
constexpr int kGroupedVerifyWorkspaceRows =
    kGroupedVerifyQ8MaxQ * kGroupedVerifyQ8Splits;
// A packed CTA replaces two 256-thread head-group CTAs and can occupy only one
// SM. Use 64-token chunks to expose enough short-context parallelism without
// paying one split reduction per N32 tile; longer contexts saturate at the
// unchanged eighty splits. The single-query utility variant keeps the old
// 128-token split contract for its stricter FP32-reference error envelope.
constexpr int kGroupedVerifyMinTokensPerSplit = 64;
constexpr int kGroupedVerifySingleQueryMinTokensPerSplit = 128;
constexpr int kGroupedVerifyShortContextMaxTokens = 128;
constexpr int kGroupedVerifyThreads = 512;
constexpr int kGroupedVerifyWarps = kGroupedVerifyThreads / kWarpSize;
constexpr int kGroupedVerifyQKWarps =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);
constexpr int kGroupedVerifyOutputTiles =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp =
    kGroupedVerifyOutputTiles / kGroupedVerifyWarps;

template <int MAX_QUERY_TOKENS>
struct GroupedVerifyTraits {
  static_assert(MAX_QUERY_TOKENS == kGroupedVerifyQ8MaxQ ||
                    MAX_QUERY_TOKENS == kGroupedVerifyQ16MaxQ,
                "grouped verifier supports q8 and q16 workspaces only");
  static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
  static constexpr int kHeadGroups = kGroupedVerifyHeads / kHeadsPerCta;
  static constexpr int kSplits = kGroupedVerifyWorkspaceRows / MAX_QUERY_TOKENS;
  static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows,
                "grouped verifier CTA must retain 48 rows");
  static_assert(kGroupedVerifyHeads % kHeadsPerCta == 0,
                "query heads must divide evenly across grouped CTAs");
  static_assert(kSplits * MAX_QUERY_TOKENS == kGroupedVerifyWorkspaceRows,
                "q8 and q16 workspaces must have equal element counts");
};

struct alignas(256) GroupedVerifySmem {
  union {
    struct {
      alignas(16) __half q[kGroupedVerifyRows * kGroupedVerifyQStride];
      alignas(16) __half kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) float scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
      alignas(16) __half probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
    } compute;
    alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
  } storage;
  alignas(16) float row_max[kGroupedVerifyRows];
  alignas(16) float row_sum[kGroupedVerifyRows];
  alignas(16) float row_scale[kGroupedVerifyRows];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[kGroupedVerifyBlockN / 4];
};

static_assert(sizeof(GroupedVerifySmem) <= 64 * 1024,
              "packed grouped verifier must fit Volta's 64 KiB opt-in budget");
// Proto B (warp-specialised loads): NVFP4 fused single-pass groups overlap the
// V(t) and K(t+1) panel loads with QK(t). Warps [0, kGroupedVerifyQKWarps) run
// QK; the remaining warps load. That needs a second K buffer and a separate V
// buffer. All other specializations keep GroupedVerifySmem unchanged.
template <int KV_DTYPE, bool TWO_PASS, bool SPARSE_PAGE4, bool ROW_SEQLENS,
          bool COMPENSATE_P>
inline constexpr bool kGroupedVerifyOverlapLoads =
    KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4 && !TWO_PASS &&
    !SPARSE_PAGE4 && !ROW_SEQLENS && !COMPENSATE_P;

struct alignas(256) GroupedVerifySmemOverlap {
  union {
    struct {
      alignas(16) __half q[kGroupedVerifyRows * kGroupedVerifyQStride];
      alignas(16) __half kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) float scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
      alignas(16) __half probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
      alignas(16) __half kv1[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) __half v[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
    } compute;
    alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
  } storage;
  alignas(16) float row_max[kGroupedVerifyRows];
  alignas(16) float row_sum[kGroupedVerifyRows];
  alignas(16) float row_scale[kGroupedVerifyRows];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[kGroupedVerifyBlockN / 4];
};

// Volta's opt-in dynamic shared memory maximum is 96 KiB.
static_assert(sizeof(GroupedVerifySmemOverlap) <= 96 * 1024,
              "overlap grouped verifier must fit Volta's 96 KiB opt-in budget");

template <int KV_DTYPE, bool TWO_PASS, bool SPARSE_PAGE4 = false,
          bool ROW_SEQLENS = false, bool COMPENSATE_P = false>
constexpr size_t grouped_verify_smem_bytes() {
  return kGroupedVerifyOverlapLoads<KV_DTYPE, TWO_PASS, SPARSE_PAGE4,
                                    ROW_SEQLENS, COMPENSATE_P>
             ? sizeof(GroupedVerifySmemOverlap)
             : sizeof(GroupedVerifySmem);
}

static_assert(kGroupedVerifyOutputTiles % kGroupedVerifyWarps == 0,
              "output tiles must divide evenly across warps");

// One NVFP4 K or V tile into `dst` (rows >= valid_rows zero-filled) using
// NUM_THREADS cooperating threads numbered copy_tid = 0..NUM_THREADS-1.
template <int NUM_THREADS>
__device__ __forceinline__ void grouped_verify_load_nvfp4_tile(
    __half* __restrict__ dst, const void* __restrict__ cache,
    const int* __restrict__ page_ids, const int valid_rows,
    const int tile_page_offset, const int block_size, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride, const int copy_tid) {
  constexpr int kPanelStrideVec = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;
  load_xqa_tc_kv_panel<0, false, NUM_THREADS,
                       flash_v100::KV_CACHE_DTYPE_NVFP4, false>(
      dst, cache, page_ids, valid_rows, kPanelStrideVec, kSharedStrideVec,
      tile_page_offset, 0, block_size, 0, block_stride, token_stride,
      head_stride, 0, copy_tid);
  for (int idx = copy_tid + valid_rows * kSharedStrideVec;
       idx < kGroupedVerifyBlockN * kSharedStrideVec; idx += NUM_THREADS) {
    reinterpret_cast<uint4*>(dst)[idx] = make_uint4(0, 0, 0, 0);
  }
}

template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY>
__device__ __forceinline__ int grouped_verify_active_splits(
    const int total_kv) {
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  constexpr int kMinTokensPerSplit =
      SINGLE_QUERY ? kGroupedVerifySingleQueryMinTokensPerSplit
                   : kGroupedVerifyMinTokensPerSplit;
  const int active_splits =
      min(Traits::kSplits,
          max(1, (total_kv + kMinTokensPerSplit - 1) / kMinTokensPerSplit));
  if constexpr (SINGLE_QUERY) {
    return active_splits;
  }
  return total_kv <= kGroupedVerifyShortContextMaxTokens ? 1 : active_splits;
}

template <bool COMPENSATE = false>
__device__ __forceinline__ void grouped_verify_qk(
    const __half* __restrict__ shared_q, const __half* __restrict__ shared_k,
    float* __restrict__ shared_scores, const float qk_scale,
    const int active_m_tiles) {
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kGroupedVerifyQKWarps) {
    return;
  }

  const int m_tile = warp_id / (kGroupedVerifyBlockN / 16);
  if ((active_m_tiles & (1 << m_tile)) == 0) {
    return;
  }
  const int n_tile = warp_id % (kGroupedVerifyBlockN / 16);
  volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
      q_fragment;
  volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major>
      k_fragment;
  volta::fragment<volta::accumulator, 16, 16, 16, float> score_fragment;
  volta::fill_fragment(score_fragment, 0.0f);
  float correction[8] = {};

#pragma unroll
  for (int k_offset = 0; k_offset < kGroupedVerifyHeadDim; k_offset += 16) {
    volta::load_matrix_sync(
        q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset,
        kGroupedVerifyQStride);
    // K is row-major [N, D].  The same bytes represent K^T as a col-major
    // [D, N] matrix, which is the B operand needed by Q @ K^T.
    volta::load_matrix_sync(
        k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset,
        kGroupedVerifyKVStride);
    if constexpr (COMPENSATE) {
      // E4M3 x FP16 products fit comfortably in FP32, but a D256 Tensor
      // Core accumulation can still lose low bits. Sum short K16 products
      // with compensated FP32 additions; explicit RN operations preserve
      // the correction under the standard fast-math build.
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile_fragment;
      volta::fill_fragment(tile_fragment, 0.0f);
      volta::mma_sync(tile_fragment, q_fragment, k_fragment, tile_fragment);
#pragma unroll
      for (int i = 0; i < score_fragment.num_elements; ++i) {
        const float y = __fsub_rn(tile_fragment.x[i], correction[i]);
        const float sum = __fadd_rn(score_fragment.x[i], y);
        correction[i] = __fsub_rn(__fsub_rn(sum, score_fragment.x[i]), y);
        score_fragment.x[i] = sum;
      }
    } else {
      volta::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
    }
  }
#pragma unroll
  for (int i = 0; i < score_fragment.num_elements; ++i) {
    score_fragment.x[i] *= qk_scale;
  }
  volta::store_matrix_sync(
      shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
      score_fragment, kGroupedVerifyScoreStride, volta::mem_row_major);
}

__device__ __forceinline__ void grouped_verify_scale_output_fragment(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& fragment,
    const float* __restrict__ row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
  if (__all_sync(0xffffffffu, first_scale == 1.0f && second_scale == 1.0f))
    return;
  fragment.x[0] *= first_scale;
  fragment.x[1] *= first_scale;
  fragment.x[2] *= second_scale;
  fragment.x[3] *= second_scale;
  fragment.x[4] *= first_scale;
  fragment.x[5] *= first_scale;
  fragment.x[6] *= second_scale;
  fragment.x[7] *= second_scale;
}

__device__ __forceinline__ void grouped_verify_add_output_tile(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& output,
    const volta::fragment<volta::accumulator, 16, 16, 16, float>& tile,
    const float* row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float scale = (i & 2) ? second_scale : first_scale;
    output.x[i] = fmaf(output.x[i], scale, tile.x[i]);
  }
}

template <bool SPARSE_PAGE4, bool ROW_SEQLENS = false>
__device__ __forceinline__ bool grouped_verify_key_visible(
    const uint32_t* __restrict__ sparse_tile_masks, const int token_idx,
    const int query_len, const int head_idx, const int kv_idx,
    const int valid_k_rows, const int lane_or_col, const int prefix_kv_len,
    const int* row_lengths = nullptr) {
  const bool row_valid = token_idx < query_len &&
                         head_idx < kGroupedVerifyHeads &&
                         lane_or_col < valid_k_rows;
  if (!row_valid) {
    return false;
  }
  if constexpr (SPARSE_PAGE4) {
    const uint32_t token_mask =
        sparse_tile_masks[(kv_idx & (kGroupedVerifyBlockN - 1)) >> 2];
    const int mask_bit = token_idx * 4 + (kv_idx & 3);
    return (token_mask & (1u << mask_bit)) != 0;
  }
  if constexpr (ROW_SEQLENS) {
    return kv_idx < row_lengths[token_idx];
  }
  return kv_idx <= prefix_kv_len + token_idx;
}

template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE = 0,
          bool SINGLE_QUERY = false, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool STAGE_PARTITION_PAGE_IDS = false,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
          bool SPARSE_PAGE4 = false, typename PARTIAL_T = __half,
          bool ROW_SEQLENS = false, bool COMPENSATE_P = false>
__global__
__launch_bounds__(kGroupedVerifyThreads, 1) void flash_attention_pvb_a_verify_partial_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, PARTIAL_T* __restrict__ partial_out,
    float* __restrict__ partial_lse, const int query_len,
    const int max_num_blocks, const int page_block_size,
    const int64_t k_block_stride, const int64_t k_token_stride,
    const int64_t k_head_stride, const int64_t v_block_stride,
    const int64_t v_token_stride, const int64_t v_head_stride,
    const float qk_scale, const float v_scale,
    const uint32_t* __restrict__ sparse_token_masks = nullptr,
    const int num_groups = 1, const int* row_lengths = nullptr) {
  static_assert(!ROW_SEQLENS || !SPARSE_PAGE4,
                "explicit row lengths apply to dense causal groups only");
  static_assert(
      !COMPENSATE_P || (!TWO_PASS && ROW_SEQLENS &&
                        KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                        std::is_same_v<PARTIAL_T, float>),
      "probability compensation is isolated to E4M3 FP32 groups");
  // NVFP4 reads through the 32-dim unit branch of load_xqa_tc_kv_panel: runtime
  // page size, plain layout, half partials, and none of the E4M3-only or
  // sparse-page features. D=256 gives 8 units per row (32 uint4), which fits the
  // 264-half (33 uint4) shared row stride unchanged.
  static_assert(
      KV_DTYPE != flash_v100::KV_CACHE_DTYPE_NVFP4 ||
          (PAGE_BLOCK_SIZE == 0 && !CONTIGUOUS_HKV1_LAYOUT &&
           !STAGE_PARTITION_PAGE_IDS && !SPARSE_PAGE4 && !COMPENSATE_P &&
           !ROW_SEQLENS && std::is_same_v<PARTIAL_T, __half> &&
           kGroupedVerifyHeadDim % 32 == 0),
      "NVFP4 grouped verify: runtime page, plain layout, half partials only");
  static_assert(MAX_QUERY_TOKENS == 8 && !TWO_PASS && !SINGLE_QUERY &&
                PAGE_BLOCK_SIZE == 0 && !CONTIGUOUS_HKV1_LAYOUT &&
                !STAGE_PARTITION_PAGE_IDS && KV_DTYPE == 3 && !SPARSE_PAGE4 &&
                !ROW_SEQLENS && !COMPENSATE_P, "dense M8 NVFP4 only");
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  const int head_group = blockIdx.x;
  const int split_id = blockIdx.y;
  const int group_idx = blockIdx.z;
  if (head_group >= Traits::kHeadGroups || split_id >= Traits::kSplits ||
      group_idx >= num_groups || query_len <= 0 ||
      query_len > MAX_QUERY_TOKENS) {
    return;
  }
  if constexpr (ROW_SEQLENS) {
    row_lengths += static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS;
  }

  if constexpr (!SPARSE_PAGE4) {
    partial_out += static_cast<int64_t>(group_idx) * Traits::kSplits *
                   MAX_QUERY_TOKENS * kGroupedVerifyHeads *
                   kGroupedVerifyHeadDim;
    partial_lse += static_cast<int64_t>(group_idx) * Traits::kSplits *
                   MAX_QUERY_TOKENS * kGroupedVerifyHeads *
                   (ROW_SEQLENS ? 2 : 1);
  }

  int total_kv = seq_lens[group_idx];
  if constexpr (ROW_SEQLENS) {
    total_kv = 0;
    for (int i = 0; i < query_len; ++i)
      total_kv = max(total_kv, row_lengths[i]);
  }
  if (total_kv <= 0) {
    if constexpr (SPARSE_PAGE4) {
      constexpr int kGroupOutputElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads * kGroupedVerifyHeadDim;
      const int64_t group_output_offset =
          static_cast<int64_t>(group_idx) * kGroupOutputElements;
      for (int idx = threadIdx.x; idx < kGroupOutputElements;
           idx += kGroupedVerifyThreads) {
        partial_out[group_output_offset + idx] = __float2half_rn(0.0f);
      }
      constexpr int kGroupLseElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads;
      if (threadIdx.x < kGroupLseElements) {
        partial_lse[static_cast<int64_t>(group_idx) * kGroupLseElements +
                    threadIdx.x] = kXQANegInf;
      }
    }
    return;
  }
  const int active_splits =
      SPARSE_PAGE4
          ? 1
          : grouped_verify_active_splits<MAX_QUERY_TOKENS, SINGLE_QUERY>(
                total_kv);
  if (split_id >= active_splits) {
    return;
  }
  const int total_tiles =
      (total_kv + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
  const int base_tiles = total_tiles / active_splits;
  const int extra_tiles = total_tiles - base_tiles * active_splits;
  const int split_tile_start =
      split_id * base_tiles + min(split_id, extra_tiles);
  const int split_tiles = base_tiles + (split_id < extra_tiles ? 1 : 0);
  const int split_start = split_tile_start * kGroupedVerifyBlockN;
  const int split_end =
      min(total_kv, split_start + split_tiles * kGroupedVerifyBlockN);
  const int prefix_kv_len = max(0, total_kv - query_len);
  const int head_start = head_group * Traits::kHeadsPerCta;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  extern __shared__ char grouped_verify_smem_raw[];
  constexpr bool kOverlapLoads =
      kGroupedVerifyOverlapLoads<KV_DTYPE, TWO_PASS, SPARSE_PAGE4, ROW_SEQLENS,
                                 COMPENSATE_P>;
  using SmemT = std::conditional_t<kOverlapLoads, GroupedVerifySmemOverlap,
                                   GroupedVerifySmem>;
  SmemT& smem = *reinterpret_cast<SmemT*>(grouped_verify_smem_raw);
  __half* shared_q = smem.storage.compute.q;
  __half* shared_kv = smem.storage.compute.kv;
  float* shared_scores = smem.storage.compute.scores;
  __half* shared_probs = smem.storage.compute.probs;
  // The compensated entry reserves one padded probability panel after the
  // legacy layout. Match P's shared-memory stride without aliasing live QK
  // rows. Legacy entries retain their existing shared-memory footprint.
  __half* shared_prob_residual = reinterpret_cast<__half*>(
      grouped_verify_smem_raw + sizeof(GroupedVerifySmem));
  constexpr int kResidualStride = kGroupedVerifyProbStride;
  const int* page_ids =
      block_table + static_cast<int64_t>(group_idx) * max_num_blocks;
  int split_page_offset = 0;
  bool use_staged_page_ids = false;
  if constexpr (STAGE_PARTITION_PAGE_IDS) {
    static_assert(PAGE_BLOCK_SIZE > 0,
                  "partition page staging requires a specialized page size");
    const int split_start_page = split_start / PAGE_BLOCK_SIZE;
    split_page_offset = split_start - split_start_page * PAGE_BLOCK_SIZE;
    const int split_page_count =
        (split_page_offset + split_end - split_start + PAGE_BLOCK_SIZE - 1) /
        PAGE_BLOCK_SIZE;
    use_staged_page_ids = split_page_count <= kGroupedVerifyPageIdsCapacity;
    if (use_staged_page_ids) {
      for (int idx = tid; idx < split_page_count;
           idx += kGroupedVerifyThreads) {
        smem.page_ids[idx] = __ldg(&page_ids[split_start_page + idx]);
      }
    }
    if (use_staged_page_ids) {
      page_ids = smem.page_ids;
    }
  }

  constexpr int kVecsPerRow = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedQVecsPerRow = kGroupedVerifyQStride / 8;
  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* shared_q_vec = reinterpret_cast<uint4*>(shared_q);
  for (int idx = tid; idx < kGroupedVerifyRows * kVecsPerRow;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kVecsPerRow;
    const int vec_col = idx % kVecsPerRow;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const int64_t query_row =
          static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = __ldg(
          q_vec + (query_row * kGroupedVerifyHeads + head_idx) * kVecsPerRow +
          vec_col);
    } else {
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = make_uint4(0, 0, 0, 0);
    }
  }
  if (tid < kGroupedVerifyRows) {
    smem.row_max[tid] = kXQANegInf;
    smem.row_sum[tid] = 0.0f;
    smem.row_scale[tid] = 1.0f;
  }
  // This barrier publishes both Q and the optional split-local page IDs.
  __syncthreads();

  constexpr int kPanelStrideVec = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;

  volta::fragment<volta::accumulator, 16, 16, 16, float>
      output_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    volta::fill_fragment(output_fragments[fragment_idx], 0.0f);
  }

  // The conservative baseline computes exact per-split FP32 max/sum in a
  // separate pass. The candidate below fuses this state update with P x V.
  if constexpr (TWO_PASS) {
    for (int tile_start = split_start; tile_start < split_end;
         tile_start += kGroupedVerifyBlockN) {
      const int valid_k_rows =
          min(kGroupedVerifyBlockN, split_end - tile_start);
      if constexpr (SPARSE_PAGE4) {
        const int valid_sparse_pages = (valid_k_rows + 3) / 4;
        if (tid < kGroupedVerifyBlockN / 4) {
          smem.sparse_token_masks[tid] =
              tid < valid_sparse_pages
                  ? __ldg(sparse_token_masks +
                          static_cast<int64_t>(group_idx) * max_num_blocks +
                          (tile_start >> 2) + tid)
                  : 0;
        }
      }
      const int tile_page_offset =
          use_staged_page_ids ? split_page_offset + tile_start - split_start
                              : tile_start;
      load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                           kGroupedVerifyThreads, KV_DTYPE,
                           KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                               (KV_DTYPE ==
                                    flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                                !SPARSE_PAGE4 && !ROW_SEQLENS)>(
          shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
          kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
          k_block_stride, k_token_stride, k_head_stride, 0);
      for (int idx = tid + valid_k_rows * kSharedStrideVec;
           idx < kGroupedVerifyBlockN * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();
      // W12: defensively zero K rows no query attends (see the V panel below).
      // K NaN is already masked out of the scores by
      // grouped_verify_key_visible, but zeroing keeps the loaded panel free of
      // the null block's E4M3 NaN. Masks are visible after the __syncthreads()
      // above; the one added inside publishes the zeros before
      // grouped_verify_qk consumes shared_kv.
      if constexpr (SPARSE_PAGE4) {
        for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
             idx += kGroupedVerifyThreads) {
          const int row = idx / kSharedStrideVec;
          const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
          if ((token_mask & (0x11111111u << (row & 3))) == 0) {
            reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
          }
        }
        __syncthreads();
      }

      int active_m_tiles = 0x7;
      if constexpr (SPARSE_PAGE4) {
        uint32_t active_query_nibbles = 0;
#pragma unroll
        for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
          active_query_nibbles |= smem.sparse_token_masks[page];
        }
        active_m_tiles = 0;
#pragma unroll
        for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
          if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
            const int first_row = token * kGroupedVerifyHeads;
            const int last_row = first_row + kGroupedVerifyHeads - 1;
            active_m_tiles |= 1 << (first_row / 16);
            active_m_tiles |= 1 << (last_row / 16);
          }
        }
      }
      grouped_verify_qk(shared_q, shared_kv, shared_scores, qk_scale,
                        active_m_tiles);
      __syncthreads();

#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float probability =
            visible ? __expf(fmaxf(score - tile_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        if (lane_id == 0 && tile_sum > 0.0f) {
          const float old_max = smem.row_max[row];
          const float old_sum = smem.row_sum[row];
          const float new_max = fmaxf(old_max, tile_max);
          smem.row_sum[row] =
              old_sum * __expf(fmaxf(old_max - new_max, -80.0f)) +
              tile_sum * __expf(fmaxf(tile_max - new_max, -80.0f));
          smem.row_max[row] = new_max;
        }
      }
      __syncthreads();
    }
  }

  // Recompute QK for the conservative path, or consume it once while updating
  // the online state for the fused path. Both form half P and use identical
  // Volta WMMA P @ V arithmetic.
  if constexpr (kOverlapLoads) {
    // Warp-specialised pipeline, three barriers per tile instead of five:
    //   A: warps [0,QKWarps) QK(t) from k[b]; the other warps load V(t) -> v and
    //      K(t+1) -> k[b^1]
    //   B: softmax (all warps)
    //   C: rescale + P@V from v (all warps)
    // Race check: v is rewritten only in phase A of tile t+1, after the
    // barrier that ends P@V(t). k[b^1] is written in phase A of tile t; its
    // last reader was QK(t-1), which ended at barrier A(t-1). k[b] (read by
    // QK(t)) is next written in phase A of tile t+1, after QK(t). scores are
    // written by QK(t+1) after the P@V(t) barrier and read by softmax(t)
    // before it. probs/row_scale are written by softmax(t) and read by
    // P@V(t) after barrier B. Arithmetic order is unchanged.
    constexpr int kLoaderFirstThread = kGroupedVerifyQKWarps * kWarpSize;
    constexpr int kLoaderThreads = kGroupedVerifyThreads - kLoaderFirstThread;
    __half* const k_buf0 = smem.storage.compute.kv;
    __half* const k_buf1 = smem.storage.compute.kv1;
    __half* const shared_v = smem.storage.compute.v;
    // Invariant: active_splits <= total_tiles, so split_start < split_end for
    // every launched split; guard anyway. The barrier stays outside the branch.
    if (split_start < split_end) {
      const int valid_first = min(kGroupedVerifyBlockN, split_end - split_start);
      grouped_verify_load_nvfp4_tile<kGroupedVerifyThreads>(
          k_buf0, k_cache, page_ids, valid_first,
          use_staged_page_ids ? split_page_offset : split_start,
          page_block_size, k_block_stride, k_token_stride, k_head_stride, tid);
    }
    __syncthreads();
    int buf = 0;
    for (int tile_start = split_start; tile_start < split_end;
         tile_start += kGroupedVerifyBlockN) {
      const int valid_k_rows =
          min(kGroupedVerifyBlockN, split_end - tile_start);
      const int tile_page_offset =
          use_staged_page_ids ? split_page_offset + tile_start - split_start
                              : tile_start;
      __half* const k_cur = buf ? k_buf1 : k_buf0;
      __half* const k_nxt = buf ? k_buf0 : k_buf1;
      if (warp_id < kGroupedVerifyQKWarps) {
        grouped_verify_qk<false>(shared_q, k_cur, shared_scores, qk_scale, 0x7);
      } else {
        const int copy_tid = tid - kLoaderFirstThread;
        const int next_start = tile_start + 32;
        const int valid_next = max(0, min(32, split_end - next_start));
        // Both tiles are N32/page4096 aligned. Never dereference next-page
        // metadata when next_start is beyond the split (including final page).
        const int vp = __ldg(page_ids + (tile_start >> 12));
        const int kp = valid_next ? __ldg(page_ids + (next_start >> 12)) : 0;
        const auto* v_page = reinterpret_cast<const uint8_t*>(v_cache) +
            static_cast<int64_t>(vp) * v_block_stride;
        const auto* k_page = reinterpret_cast<const uint8_t*>(k_cache) +
            static_cast<int64_t>(kp) * k_block_stride;
        pvb_load_vk_pair(shared_v, k_nxt, v_page, k_page,
                         tile_start & 4095, next_start & 4095,
                         valid_k_rows, valid_next, copy_tid);

      }
      __syncthreads();  // A

#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float old_max = smem.row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        const float probability =
            visible ? __expf(fmaxf(score - new_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        const float exp_diff =
            tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
        shared_probs[row * kGroupedVerifyProbStride + lane_id] =
            __float2half_rn(probability);
        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {
            smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
            smem.row_max[row] = new_max;
          }
          smem.row_scale[row] = exp_diff;
        }
      }
      __syncthreads();  // B
#pragma unroll
      for (int fragment_idx = 0;
           fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
        const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
        grouped_verify_scale_output_fragment(output_fragments[fragment_idx],
                                             smem.row_scale, m_tile * 16);
      }
#pragma unroll
      for (int fragment_idx = 0;
           fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
        const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
        const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
        volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
            probability_fragment;
        volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major>
            value_fragment;
        auto& pv_fragment = output_fragments[fragment_idx];
#pragma unroll
        for (int k_offset = 0; k_offset < kGroupedVerifyBlockN;
             k_offset += 16) {
          volta::load_matrix_sync(
              probability_fragment,
              shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
              kGroupedVerifyProbStride);
          volta::load_matrix_sync(
              value_fragment,
              shared_v + k_offset * kGroupedVerifyKVStride + d_tile * 16,
              kGroupedVerifyKVStride);
          volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                          pv_fragment);
        }
      }
      __syncthreads();  // C
      buf ^= 1;
    }
  } else {
  for (int tile_start = split_start; tile_start < split_end;
       tile_start += kGroupedVerifyBlockN) {
    const int valid_k_rows = min(kGroupedVerifyBlockN, split_end - tile_start);
    if constexpr (SPARSE_PAGE4) {
      const int valid_sparse_pages = (valid_k_rows + 3) / 4;
      if (tid < kGroupedVerifyBlockN / 4) {
        smem.sparse_token_masks[tid] =
            tid < valid_sparse_pages
                ? __ldg(sparse_token_masks +
                        static_cast<int64_t>(group_idx) * max_num_blocks +
                        (tile_start >> 2) + tid)
                : 0;
      }
    }
    const int tile_page_offset =
        use_staged_page_ids ? split_page_offset + tile_start - split_start
                            : tile_start;
    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE,
                         KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                             (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                              !SPARSE_PAGE4 && !ROW_SEQLENS)>(
        shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        k_block_stride, k_token_stride, k_head_stride, 0);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    // W12: defensively zero K rows no query attends (see the V panel below).
    // K NaN is already masked out of the scores by grouped_verify_key_visible,
    // but zeroing keeps the loaded panel free of the null block's E4M3 NaN.
    // Masks are visible after the __syncthreads() above; the one added inside
    // publishes the zeros before grouped_verify_qk consumes shared_kv.
    if constexpr (SPARSE_PAGE4) {
      for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kSharedStrideVec;
        const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
        if ((token_mask & (0x11111111u << (row & 3))) == 0) {
          reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
        }
      }
      __syncthreads();
    }

    int active_m_tiles = 0x7;
    if constexpr (COMPENSATE_P) {
      // Native MTP4 supplies five rows: only 30 of the 48 packed rows are
      // live. Avoid QK/PV for wholly padded M tiles to pay for the residual
      // product without changing arithmetic in any live row.
      active_m_tiles =
          (1 << ((query_len * Traits::kHeadsPerCta + 15) / 16)) - 1;
    }
    if constexpr (SPARSE_PAGE4) {
      uint32_t active_query_nibbles = 0;
#pragma unroll
      for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
        active_query_nibbles |= smem.sparse_token_masks[page];
      }
      active_m_tiles = 0;
#pragma unroll
      for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
        if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
          const int first_row = token * kGroupedVerifyHeads;
          const int last_row = first_row + kGroupedVerifyHeads - 1;
          active_m_tiles |= 1 << (first_row / 16);
          active_m_tiles |= 1 << (last_row / 16);
        }
      }
    }
    grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,
                                    qk_scale, active_m_tiles);
    __syncthreads();

    if constexpr (TWO_PASS) {
      for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyBlockN;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kGroupedVerifyBlockN;
        const int col = idx % kGroupedVerifyBlockN;
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + col;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, col, prefix_kv_len, row_lengths) &&
            smem.row_sum[row] > 0.0f;
        const float probability =
            visible ? __expf(fmaxf(
                          shared_scores[row * kGroupedVerifyScoreStride + col] -
                              smem.row_max[row],
                          -80.0f))
                    : 0.0f;
        shared_probs[row * kGroupedVerifyProbStride + col] =
            __float2half_rn(probability);
      }
      __syncthreads();
    } else {
#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        if constexpr (COMPENSATE_P) {
          if ((active_m_tiles & (1 << (row / 16))) == 0) {
            continue;
          }
        }
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float old_max = smem.row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        const float probability =
            visible ? __expf(fmaxf(score - new_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        const float exp_diff =
            tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
        shared_probs[row * kGroupedVerifyProbStride + lane_id] =
            __float2half_rn(probability);
        if constexpr (COMPENSATE_P) {
          const float rounded = __half2float(__float2half_rn(probability));
          shared_prob_residual[row * kResidualStride + lane_id] =
              __float2half_rn((probability - rounded) * 2048.0f);
        }
        // Finish every lane's shared-state reads before lane 0 overwrites the
        // online maximum. Shuffle synchronization does not order memory.
        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {
            smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
            smem.row_max[row] = new_max;
          }
          smem.row_scale[row] = exp_diff;
        }
      }
      __syncthreads();
      if constexpr (!COMPENSATE_P) {
#pragma unroll
        for (int fragment_idx = 0;
             fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
          const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
          const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
          grouped_verify_scale_output_fragment(output_fragments[fragment_idx],
                                               smem.row_scale, m_tile * 16);
        }
      }
    }

    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE,
                         KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                             (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                              !SPARSE_PAGE4 && !ROW_SEQLENS)>(
        shared_kv, v_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        v_block_stride, v_token_stride, v_head_stride, 0);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    // W12: the grouped planner pads each category to a multiple of 8 with
    // (physical microblock 0 = null block, mask 0) and counts the padding in
    // seq_len; the forward also carries unselected tokens inside partial
    // microblocks. Such rows get P=0, but 0 * NaN survives the P@V MMA when
    // page 0 holds fp16 GDN state that decodes to E4M3 NaN (W8). Zero every
    // row no query attends (mask bit = query*4 + token_in_microblock, so the
    // row is unattended iff (mask & (0x11111111u << (row & 3))) == 0) so the
    // reduction sees 0 * 0 = 0. The leading __syncthreads() makes the V panel
    // load finish before we overwrite its page-0 rows (else the load races past
    // the zero and resurrects the NaN); the trailing one publishes the zeros
    // before P@V. Masks were synced earlier in this iteration.
    if constexpr (SPARSE_PAGE4) {
      __syncthreads();
      for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kSharedStrideVec;
        const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
        if ((token_mask & (0x11111111u << (row & 3))) == 0) {
          reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
         ++fragment_idx) {
      const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
      const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
      const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
      if constexpr (SPARSE_PAGE4 || COMPENSATE_P) {
        if ((active_m_tiles & (1 << m_tile)) == 0) {
          continue;
        }
      }
      volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
          probability_fragment;
      volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major>
          value_fragment;
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile_fragment;
      if constexpr (COMPENSATE_P) {
        // Keep Tensor Core accumulation local to N32. Carrying a large C
        // through every KV tile loses small PV corrections at long contexts.
        // The online state remains FP32 and is updated once per tile below.
        volta::fill_fragment(tile_fragment, 0.0f);
      }
      auto& pv_fragment =
          COMPENSATE_P ? tile_fragment : output_fragments[fragment_idx];
#pragma unroll
      for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
        volta::load_matrix_sync(
            probability_fragment,
            shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
            kGroupedVerifyProbStride);
        volta::load_matrix_sync(
            value_fragment,
            shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,
            kGroupedVerifyKVStride);
        volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                        pv_fragment);
        if constexpr (COMPENSATE_P) {
          // Scale the residual up before its FP16 conversion so small
          // corrections do not underflow. Every finite E4M3 value divided
          // by 2048 is still exactly representable in FP16, so applying
          // the inverse power of two to V introduces no operand rounding.
          volta::load_matrix_sync(
              probability_fragment,
              shared_prob_residual + m_tile * 16 * kResidualStride + k_offset,
              kResidualStride);
#pragma unroll
          for (int i = 0; i < value_fragment.num_elements / 2; ++i) {
            union {
              uint32_t bits;
              __half2 pair;
            } packed_value;
            packed_value.bits = value_fragment.x[i];
            packed_value.pair =
                __hmul2(packed_value.pair, __float2half2_rn(1.0f / 2048.0f));
            value_fragment.x[i] = packed_value.bits;
          }
          volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                          pv_fragment);
        }
      }
      if constexpr (COMPENSATE_P) {
        grouped_verify_add_output_tile(output_fragments[fragment_idx],
                                       tile_fragment, smem.row_scale,
                                       m_tile * 16);
      }
    }
    __syncthreads();
  }
  }

  // The compute buffers are dead. Reuse their storage for the dense FP32
  // partial output, then normalize and write only real query/head rows.
  __syncthreads();
  float* shared_output = smem.storage.output;
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
    const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
    const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
    volta::store_matrix_sync(
        shared_output + m_tile * 16 * kGroupedVerifyHeadDim + d_tile * 16,
        output_fragments[fragment_idx], kGroupedVerifyHeadDim,
        volta::mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kGroupedVerifyHeadDim;
    const int d = idx % kGroupedVerifyHeadDim;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[row];
      // Explicit-row FP32 groups retain the numerator until the final merge.
      // Normalizing each partition and encoding its weight as m+log(sum)
      // introduces avoidable rounding, including at FP16 output midpoints.
      const float scale =
          sum > 0.0f ? (ROW_SEQLENS ? v_scale : v_scale / sum) : 0.0f;
      int64_t output_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        output_idx = (global_token_idx * kGroupedVerifyHeads + head_idx) *
                         kGroupedVerifyHeadDim +
                     d;
      } else {
        output_idx =
            (((static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                  kGroupedVerifyHeads +
              head_idx) *
                 kGroupedVerifyHeadDim +
             d);
      }
      if constexpr (std::is_same_v<PARTIAL_T, float>) {
        partial_out[output_idx] = shared_output[idx] * scale;
      } else {
        partial_out[output_idx] = __float2half_rn(shared_output[idx] * scale);
      }
    }
  }
  if (tid < kGroupedVerifyRows) {
    const int token_idx = tid / Traits::kHeadsPerCta;
    const int local_head = tid % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[tid];
      int64_t lse_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        lse_idx = global_token_idx * kGroupedVerifyHeads + head_idx;
      } else {
        lse_idx =
            (static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                kGroupedVerifyHeads +
            head_idx;
      }
      if constexpr (ROW_SEQLENS) {
        partial_lse[2 * lse_idx] = sum > 0.0f ? smem.row_max[tid] : kXQANegInf;
        partial_lse[2 * lse_idx + 1] = sum;
      } else {
        partial_lse[lse_idx] =
            sum > 0.0f ? smem.row_max[tid] + logf(sum) : kXQANegInf;
      }
    }
  }
}


}
