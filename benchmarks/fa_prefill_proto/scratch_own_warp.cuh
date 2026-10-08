// P1: one warp owns all scratch in its 32-row x 16-column score strip.
// For each float2 instruction, lanes 0..15 cover 32 distinct banks:
// rows alternate by 144 words (16 mod 32); lane&7 supplies the column pair.
// Lanes 16..31 use the next two rows (the second ideal 128-byte wavefront).
__device__ __forceinline__ void d256_bm32_phase_spill_pair_scratch(
    float* __restrict__ score,
    D256BM32PhaseAccumulatorFragment& top,
    D256BM32PhaseAccumulatorFragment& bottom) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const int offset = (pair * 4 + (lane >> 3)) * D256_BM32_PHASE_SCORE_LD
                       + warp * 16 + (lane & 7) * 2;
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(score + offset));
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::
        "r"(address), "r"(__float_as_uint(top.x[2 * pair])),
        "r"(__float_as_uint(top.x[2 * pair + 1])) : "memory");
    asm volatile("st.shared.v2.u32 [%0+9216], {%1, %2};" ::
        "r"(address), "r"(__float_as_uint(bottom.x[2 * pair])),
        "r"(__float_as_uint(bottom.x[2 * pair + 1])) : "memory");
  }
}

__device__ __forceinline__ void d256_bm32_phase_reload_pair_scratch(
    const float* __restrict__ score,
    D256BM32PhaseAccumulatorFragment& top,
    D256BM32PhaseAccumulatorFragment& bottom) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const int offset = (pair * 4 + (lane >> 3)) * D256_BM32_PHASE_SCORE_LD
                       + warp * 16 + (lane & 7) * 2;
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(score + offset));
    uint32_t x, y;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) :
                 "r"(address) : "memory");
    top.x[2 * pair] = __uint_as_float(x);
    top.x[2 * pair + 1] = __uint_as_float(y);
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2+9216];" : "=r"(x), "=r"(y) :
                 "r"(address) : "memory");
    bottom.x[2 * pair] = __uint_as_float(x);
    bottom.x[2 * pair + 1] = __uint_as_float(y);
  }
}
