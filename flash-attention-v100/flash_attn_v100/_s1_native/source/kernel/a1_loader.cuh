template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool FP8_PAIR_LOAD = false, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx = threadIdx.x,
    const uint16_t* __restrict__ e4m3_lut = nullptr) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  if constexpr (FP8_PAIR_LOAD) {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
                  "Paired XQA loads require FP8 KV");
    static_assert(!E4M3_SHARED_LUT,
                  "Paired E4M3 conversion does not use the shared LUT");
    const int pair_stride = panel_d_stride_uint4 / 2;
    const int pair_count = valid_kv_tile_rows * pair_stride;
    for (int pair_idx = copy_thread_idx; pair_idx < pair_count;
         pair_idx += NUM_THREADS) {
      const int row = pair_idx / pair_stride;
      const int vec_pair = pair_idx % pair_stride;
      const int token_offset = tile_page_offset + kv_tile_start + row;
      int logical_block;
      int block_offset;
      if constexpr (BLOCK_SIZE == 4) {
        logical_block = token_offset >> 2;
        block_offset = token_offset & 3;
      } else if constexpr (BLOCK_SIZE == 16) {
        logical_block = token_offset >> 4;
        block_offset = token_offset & 15;
      } else if constexpr (BLOCK_SIZE == 784) {
        logical_block = token_offset / 784;
        block_offset = token_offset - logical_block * 784;
      } else if constexpr (BLOCK_SIZE == 800) {
        logical_block = (token_offset >= 800) + (token_offset >= 1600) +
                        (token_offset >= 2400);
        block_offset = token_offset - logical_block * 800;
      } else if constexpr (BLOCK_SIZE == 1568) {
        logical_block = token_offset / 1568;
        block_offset = token_offset - logical_block * 1568;
      } else if constexpr (BLOCK_SIZE == 1648) {
        logical_block = token_offset / 1648;
        block_offset = token_offset - logical_block * 1648;
      } else if constexpr (BLOCK_SIZE == 3296) {
        logical_block = token_offset / 3296;
        block_offset = token_offset - logical_block * 3296;
      } else {
        logical_block = token_offset / block_size;
        block_offset = token_offset % block_size;
      }
      const int physical_block = page_ids[logical_block];
      int64_t physical_offset;
      if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
        constexpr int64_t kHeadDim = 256;
        constexpr int64_t kPhysicalBlockStride =
            BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
        physical_offset =
            static_cast<int64_t>(physical_block) * kPhysicalBlockStride +
            static_cast<int64_t>(block_offset) * kHeadDim + panel_offset;
      } else {
        physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                          static_cast<int64_t>(block_offset) * token_stride +
                          static_cast<int64_t>(kv_head_idx) * head_stride +
                          panel_offset;
      }
      const uint4 raw = __ldg(reinterpret_cast<const uint4*>(kv_cache) +
                              physical_offset / 16 + vec_pair);
      const int shared_offset = row * kv_smem_stride_uint4 + vec_pair * 2;
      const uint64_t raw_lo =
          static_cast<uint64_t>(raw.x) | (static_cast<uint64_t>(raw.y) << 32);
      shared_vec[shared_offset] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_lo)
              : fp8_e5m2_vector_to_half8(raw_lo);
      const uint64_t raw_hi =
          static_cast<uint64_t>(raw.z) | (static_cast<uint64_t>(raw.w) << 32);
      shared_vec[shared_offset + 1] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_hi)
              : fp8_e5m2_vector_to_half8(raw_hi);
    }
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4) {
    // 32-dim unit: one 16-byte LDG of e2m1 data (32 codes) + one 2-byte LDG of
    // the two E4M3FN scale bytes, expanded to 4 uint4 (32 halves) in smem.
    // Needs 16-byte aligned data rows (checked on the host: page stride and
    // base pointer % 16 == 0; a row is 128 bytes and dims % 32 == 0).
    static_assert(!CONTIGUOUS_HKV1_LAYOUT && !E4M3_SHARED_LUT &&
                      BLOCK_SIZE == 0,
                  "NVFP4 KV uses the runtime-page-size plain layout");
    constexpr int kUnitUint4 = 4;  // 32 dims = 4 x (8 halves)
    const int units_per_row = panel_d_stride_uint4 / kUnitUint4;
    const int unit_count = valid_kv_tile_rows * units_per_row;
    const int64_t num_kv_heads = token_stride / 144;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(kv_cache);
    const int64_t scale_region =
        static_cast<int64_t>(block_size) * num_kv_heads * 128;
    for (int unit_idx = copy_thread_idx; unit_idx < unit_count;
         unit_idx += NUM_THREADS) {
      int row = unit_idx / units_per_row;
      int unit = unit_idx - row * units_per_row;
      // kv_smem_stride_uint4 is 33 (odd): with row-major lane order 8
      // consecutive lanes share one row and hit 2 bank classes (4-way STS
      // conflict). Within a full 4-row block of 4*units_per_row indices the
      // map (row&3, unit) -> (within&3, within>>2) is a bijection, so the
      // same global addresses are read and the same smem slots written; 8
      // consecutive lanes now cover 4 rows x 2 units, bank classes
      // row + 4*unit mod 8 all distinct. Tail block (<4 rows) keeps the
      // original mapping.
      {
        const int group = row >> 2;
        const int within = ((row & 3) * units_per_row) + unit;
        if (group * 4 + 4 <= valid_kv_tile_rows) {
          row = (group << 2) | (within & 3);
          unit = within >> 2;
        }
      }
      const int token_offset = tile_page_offset + kv_tile_start + row;
      const int logical_block = token_offset / block_size;
      const int block_offset = token_offset - logical_block * block_size;
      const int physical_block = page_ids[logical_block];
      const int64_t page_base =
          static_cast<int64_t>(physical_block) * block_stride;
      const int64_t row_index =
          static_cast<int64_t>(block_offset) * num_kv_heads + kv_head_idx;
      const int64_t dims = static_cast<int64_t>(panel_offset) + unit * 32;
      const uint4 data = __ldg(reinterpret_cast<const uint4*>(
          bytes + page_base + row_index * 128 + (dims >> 1)));
      const uint32_t scales2 = __ldg(reinterpret_cast<const uint16_t*>(
          bytes + page_base + scale_region + row_index * 16 + (dims >> 4)));
      uint4 half32[kUnitUint4];
      flash_v100::nvfp4_data16_to_half32(data, scales2, half32);
#pragma unroll
      for (int i = 0; i < kUnitUint4; ++i) {
        shared_vec[row * kv_smem_stride_uint4 + ((unit * kUnitUint4 + i) ^ a1_mask(row))] =
            half32[i];
      }
    }
  } else {
    const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
    for (int copy_idx = copy_thread_idx; copy_idx < copy_count;
         copy_idx += NUM_THREADS) {
      const int row = copy_idx / panel_d_stride_uint4;
      const int vec_col = copy_idx % panel_d_stride_uint4;
      shared_vec[row * kv_smem_stride_uint4 + vec_col] =
          load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                E4M3_SHARED_LUT>(
              kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
              tile_page_offset, kv_tile_start, block_size, kv_head_idx,
              block_stride, token_stride, head_stride, panel_offset, e4m3_lut);
    }
  }
}

