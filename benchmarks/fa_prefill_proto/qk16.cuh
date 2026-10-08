    // P2: split Q rows, not the K reduction. All sixteen warps participate.
    // Each warp retains PV bottom in registers and spills only PV top into
    // its own 16x16 score quadrant (16 warps * 256 floats = existing 16 KiB).
    const int qk_tile = warp_id & 7;
    const int q_panel = warp_id >> 3;
    if (qk_tile * WMMA_N < valid_k_rows) {
      const int tile_n = qk_tile * WMMA_N;
      float* scratch = shared.score +
          q_panel * D256_BM32_PHASE_PANEL_M * D256_BM32_PHASE_BLOCK_N + tile_n;
      const __half* k_tile =
          reinterpret_cast<const __half*>(shared.k_tile_ptr[qk_tile]);
      store_matrix_sync(scratch, accumulator_top,
                        D256_BM32_PHASE_BLOCK_N, mem_row_major);
      asm volatile("" ::: "memory");
      D256BM32PhaseAccumulatorFragment qk;
      {
        D256BM32PhaseMatrixAFragment a_fragment;
        D256BM32PhaseQKMatrixBFragment b_fragment;
        fill_fragment(qk, 0.0f);
#pragma unroll
        for (int k_offset = 0; k_offset < D256_BM32_PHASE_D;
             k_offset += WMMA_K) {
          load_matrix_sync(b_fragment, k_tile + k_offset, k_token_stride);
          d256_bm32_phase_load_matrix_a(a_fragment,
              shared.query + q_panel * D256_BM32_PHASE_PANEL_M * D256_BM32_PHASE_D,
              k_offset);
          mma_sync(qk, a_fragment, b_fragment, qk);
        }
      }
      asm volatile("" ::: "memory");
      load_matrix_sync(accumulator_top, scratch,
                       D256_BM32_PHASE_BLOCK_N, mem_row_major);
      __syncwarp();
      store_matrix_sync(scratch, qk, D256_BM32_PHASE_BLOCK_N, mem_row_major);
    }
