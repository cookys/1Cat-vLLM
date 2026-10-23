// BASE N128/D128 scalar PV; static accumulator indexing only.
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM, int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          bool ALIGNED_PADDED_SMEM, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          bool E4M3_SHARED_LUT = false, typename PARTIAL_T = __half,
          bool NVFP4_HALF_P = false>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_nospill_partition_kernel_256_wide(
        const __half* __restrict__ q, const void* __restrict__ k_cache,
        const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
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
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPVPanelDim = kXQATCStride;
  constexpr int kNumPVPanels = D / kPVPanelDim;
  constexpr int kQKPanelDim =
      QK_SW_PIPELINE ? kXQATCQKPipelinePanelDim : kXQATCStride;
  constexpr int kNumQKPanels = D / kQKPanelDim;
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  // Keep softmax P in fp32 through PV for FP16 and E4M3 KV, avoiding a half
  // round-trip before the existing fp32 FMA accumulation. Preserve E5M2's
  // established half-P numerical contract. This branch is compile-time only.
  // NVFP4_HALF_P (Q1.23 P1.2 contrast build) makes NVFP4 follow E5M2 as well.
  constexpr bool kKeepPfp32 =
      KV_DTYPE != flash_v100::KV_CACHE_DTYPE_FP8_E5M2 &&
      !(NVFP4_HALF_P && KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4);
  static_assert(!NVFP4_HALF_P || KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4,
                "NVFP4_HALF_P only applies to NVFP4 KV");
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int pv_panel_d_stride_uint4 = kPVPanelDim / 8;
  constexpr int qk_smem_stride_uint4 =
      QK_SW_PIPELINE ? SmemLayout::kQKStride / 8 : kv_smem_stride_uint4;
  constexpr int qk_panel_d_stride_uint4 = kQKPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  constexpr bool kPVHalf2 = CONTIGUOUS_HKV1_LAYOUT &&
                            (BLOCK_SIZE == 800 || BLOCK_SIZE == 1568) &&
                            KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one full softmax/PV warp");
  static_assert(
      !QK_SW_PIPELINE || (GROUP_SIZE == 6 && (NUM_THREADS == 6 * kWarpSize ||
                                              NUM_THREADS == 8 * kWarpSize)),
      "The QK software pipeline requires a six- or eight-warp G6 "
      "kernel");
  static_assert(
      !E4M3_SHARED_LUT || KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
      "The shared conversion LUT is only valid for E4M3 KV");
  static_assert(KV_DTYPE != flash_v100::KV_CACHE_DTYPE_NVFP4 ||
                    (!CONTIGUOUS_HKV1_LAYOUT && !FP8_PAIR_LOAD &&
                     !E4M3_SHARED_LUT && !QK_SW_PIPELINE && BLOCK_SIZE == 0),
                "NVFP4 KV uses the 32-dim plain loader (no pair load, LUT, "
                "fixed-stride layout, or QK software pipeline)");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    const int partition_size_begin =
        route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin;
    partition_size = seq_len < partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int start_token_idx = partition_idx * partition_size;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + partition_size - 1) / partition_size;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(partition_size, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  uint16_t* e4m3_lut = nullptr;
  if constexpr (E4M3_SHARED_LUT) {
    e4m3_lut = reinterpret_cast<uint16_t*>(smem_raw + sizeof(SmemLayout));
    for (int raw = tid; raw < 256; raw += NUM_THREADS) {
      e4m3_lut[raw] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw));
    }
  }
  __half* sQ = smem.q;
  __half* sK = smem.k_buffer(0);
  __half* sV = smem.v_buffer();
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  int partition_start_page = 0;
  int partition_page_offset = 0;
  if constexpr (PARTITION_PAGE_IDS) {
    int partition_page_count;
    if constexpr (BLOCK_SIZE == 800) {
      partition_start_page = start_token_idx / 800;
      partition_page_offset = start_token_idx - partition_start_page * 800;
      partition_page_count = (partition_page_offset + part_tokens + 799) / 800;
    } else {
      partition_start_page = start_token_idx / block_size;
      partition_page_offset =
          start_token_idx - partition_start_page * block_size;
      partition_page_count =
          (partition_page_offset + part_tokens + block_size - 1) / block_size;
    }
    for (int idx = tid; idx < partition_page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[partition_start_page + idx]);
    }
    __syncthreads();
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    int start_page = partition_start_page;
    int tile_page_offset;
    int page_count = 0;
    if constexpr (PARTITION_PAGE_IDS) {
      tile_page_offset = partition_page_offset + block_n * kXQATCBlockN;
    } else if constexpr (BLOCK_SIZE == 16) {
      start_page = tile_token_start >> 4;
      tile_page_offset = tile_token_start & 15;
      page_count = (tile_page_offset + valid_k_rows + 15) >> 4;
    } else if constexpr (BLOCK_SIZE == 784) {
      start_page = tile_token_start / 784;
      tile_page_offset = tile_token_start - start_page * 784;
      page_count = (tile_page_offset + valid_k_rows + 783) / 784;
    } else if constexpr (BLOCK_SIZE == 800) {
      start_page = tile_token_start / 800;
      tile_page_offset = tile_token_start - start_page * 800;
      page_count = 1 + (tile_page_offset + valid_k_rows > 800);
    } else if constexpr (BLOCK_SIZE == 1568) {
      start_page = tile_token_start / 1568;
      tile_page_offset = tile_token_start - start_page * 1568;
      page_count = (tile_page_offset + valid_k_rows + 1567) / 1568;
    } else {
      start_page = tile_token_start / block_size;
      tile_page_offset = tile_token_start - start_page * block_size;
      page_count =
          (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    }

    if constexpr (!PARTITION_PAGE_IDS) {
      for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
        smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
      }
      __syncthreads();
    }

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kXQATCBlockN) {
      const int valid_kv_tile_rows =
          min(kXQATCBlockN, valid_k_rows - kv_tile_start);
      // The QK fragments die before softmax/PV. This permits the compiler to
      // reuse their registers for the memory-latency-sensitive PV phase.
      volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::row_major>
          qk_a_frag;
      volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::col_major>
          qk_b_frag;
      volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
          qk_acc_frag;
      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        volta::fill_fragment(qk_acc_frag, 0.0f);
      }

      if constexpr (QK_SW_PIPELINE) {
        constexpr int kConsumerWarps = kXQATCBlockN / WMMA_N;
        constexpr int kProducerThreads =
            NUM_THREADS - kConsumerWarps * kWarpSize;
        const int producer_tid = tid - kConsumerWarps * kWarpSize;
        if (producer_tid >= 0) {
          load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                        kProducerThreads, KV_DTYPE,
                                        E4M3_SHARED_LUT>(
              smem.k_buffer(0), k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, 0, producer_tid, e4m3_lut);
        }
        __syncthreads();

#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          if (producer_tid >= 0 && panel_idx + 1 < kNumQKPanels) {
            load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                          kProducerThreads, KV_DTYPE,
                                          E4M3_SHARED_LUT>(
                smem.k_buffer((panel_idx + 1) & 1), k_cache, smem.page_ids,
                valid_kv_tile_rows, qk_panel_d_stride_uint4,
                qk_smem_stride_uint4, tile_page_offset, kv_tile_start,
                block_size, kv_head_idx, k_block_stride, k_token_stride,
                k_head_stride, panel_offset + kQKPanelDim, producer_tid,
                e4m3_lut);
          }
          if (warp_id < kConsumerWarps) {
            const int tile_n = warp_id * WMMA_N;
            const __half* sK_panel = smem.k_buffer(panel_idx & 1);
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag,
                  sK_panel + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      } else {
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                               KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
              sK, k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, panel_offset, threadIdx.x,
              e4m3_lut);
          for (int idx = tid + valid_kv_tile_rows * qk_panel_d_stride_uint4;
               idx < kXQATCBlockN * qk_panel_d_stride_uint4;
               idx += NUM_THREADS) {
            const int row = idx / qk_panel_d_stride_uint4;
            const int vec_col = idx % qk_panel_d_stride_uint4;
            reinterpret_cast<uint4*>(sK)[row * qk_smem_stride_uint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();

          if (warp_id < (kXQATCBlockN / WMMA_N)) {
            const int tile_n = warp_id * WMMA_N;
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag, sK + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      }

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
        for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
          qk_acc_frag.x[i] *= KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                  ? softmax_scale
                                  : softmax_scale * k_scale;
        }
        volta::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                 qk_acc_frag, kXQATCBlockN,
                                 volta::mem_row_major);
      }
      __syncthreads();
    }

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      float* sP_row_f = sS + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        if constexpr (kKeepPfp32) {
          reinterpret_cast<float4*>(sP_row_f)[vec_col] =
              make_float4(e0, e1, e2, e3);
        } else {
          packed_exp0 = __float22half2_rn(make_float2(e0, e1));
          packed_exp1 = __float22half2_rn(make_float2(e2, e3));
        }
      }

#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        if constexpr (kKeepPfp32) {
          sP_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
        } else {
          sP_row_h[c] =
              (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
        }
      }

#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if constexpr (!kKeepPfp32) {
        __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
        if (vec_col < vec_cols) {
          const int base_offset = vec_col * 2;
          sP_half2[base_offset] = packed_exp0;
          sP_half2[base_offset + 1] = packed_exp1;
        }
      }

      if (block_n > 0) {
#pragma unroll
        for (int i = 0; i < kAccumsPerThread; ++i) {
          out_acc[i] *= exp_diff;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int panel_idx = 0; panel_idx < kNumPVPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPVPanelDim;
      for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
           kv_tile_start += kXQATCBlockN) {
        const int valid_kv_tile_rows =
            min(kXQATCBlockN, valid_k_rows - kv_tile_start);
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
            sV, v_cache, smem.page_ids, valid_kv_tile_rows,
            pv_panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, v_block_stride,
            v_token_stride, v_head_stride, panel_offset, threadIdx.x, e4m3_lut);
        for (int idx = tid + valid_kv_tile_rows * pv_panel_d_stride_uint4;
             idx < kXQATCBlockN * pv_panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / pv_panel_d_stride_uint4;
          const int vec_col = idx % pv_panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const __half* sP_row = sP + row * kXQATCBlockN + kv_tile_start;
          const float* sP_row_f = sS + row * kXQATCBlockN + kv_tile_start;
#pragma unroll 8
          for (int token = 0; token < kXQATCBlockN; ++token) {
            if (token >= valid_kv_tile_rows) {
              break;
            }
            float prob;
            if constexpr (kKeepPfp32) {
              prob = sP_row_f[token];
            } else {
              prob = __half2float(sP_row[token]);
            }
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
            if constexpr (kPVHalf2) {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / (2 * kWarpSize));
                   ++d_iter) {
                const int local_d2 = lane_id + d_iter * kWarpSize;
                const float2 value = __half22float2(
                    reinterpret_cast<const __half2*>(sV_row)[local_d2]);
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter * 2;
                out_acc[acc_idx] = fmaf(prob, value.x, out_acc[acc_idx]);
                out_acc[acc_idx + 1] =
                    fmaf(prob, value.y, out_acc[acc_idx + 1]);
              }
            } else {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / kWarpSize);
                   ++d_iter) {
                const int local_d = lane_id + d_iter * kWarpSize;
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter;
                out_acc[acc_idx] =
                    fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    PARTIAL_T* tmp_out_ptr =
        tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
        static_cast<int64_t>(head_idx) * tmp_out_stride1 +
        static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    const float output_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                   ? inv_row_sum
                                   : inv_row_sum * v_scale;
    if constexpr (kPVHalf2) {
#pragma unroll
      for (int acc_idx = 0; acc_idx < kAccumsPerThread; ++acc_idx) {
        constexpr int kAccumsPerPanel = kPVPanelDim / kWarpSize;
        const int panel_idx = acc_idx / kAccumsPerPanel;
        const int panel_acc_idx = acc_idx % kAccumsPerPanel;
        const int d_iter = panel_acc_idx / 2;
        const int component = panel_acc_idx & 1;
        const int d = panel_idx * kPVPanelDim + d_iter * (2 * kWarpSize) +
                      thread_in_row * 2 + component;
        const float value = out_acc[acc_idx] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    } else {
#pragma unroll
      for (int acc_idx = 0; acc_idx < kAccumsPerThread; ++acc_idx) {
        const int d = thread_in_row + acc_idx * kWarpSize;
        const float value = out_acc[acc_idx] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}

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
void launch_flash_attention_decode_nospill_256_wide(
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
      sizeof(SmemLayout) + (E4M3_SHARED_LUT ? 256 * sizeof(uint16_t) : 0);
#define LAUNCH_XQA_PARTITION(KV_DTYPE)                                         \
  do {                                                                         \
    auto partition_kernel =                                                    \
        (void*)flash_attention_decode_nospill_partition_kernel_256_wide<        \
            PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,              \
            MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,             \
            ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,      \
            PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT, PARTIAL_T,         \
            NVFP4_HALF_P>;                                                     \
    cudaFuncSetAttribute(partition_kernel,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         shared_mem);                                          \
    if constexpr (MIN_BLOCKS_PER_SM > 1) {                                     \
      cudaFuncSetAttribute(partition_kernel,                                   \
                           cudaFuncAttributePreferredSharedMemoryCarveout,     \
                           100);                                               \
    }                                                                          \
    flash_attention_decode_nospill_partition_kernel_256_wide<                   \
        PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,                  \
        MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,                 \
        ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,          \
        PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT, PARTIAL_T,        \
        NVFP4_HALF_P>                                                          \
        <<<partition_grid, NUM_THREADS, shared_mem, stream>>>(                 \
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
