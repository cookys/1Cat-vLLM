#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_v100 {

constexpr int KV_CACHE_DTYPE_FP16 = 0;
constexpr int KV_CACHE_DTYPE_FP8_E4M3 = 1;
constexpr int KV_CACHE_DTYPE_FP8_E5M2 = 2;
// NVFP4 KV: e2m1 data (2 values per byte, low nibble = even element) with one
// E4M3FN scale byte per 16 values; page layout [data | scales] per side.
constexpr int KV_CACHE_DTYPE_NVFP4 = 3;

__device__ __forceinline__ float quiet_nan_f() {
  return __int_as_float(0x7fffffff);
}

__device__ __forceinline__ float inf_f() { return __int_as_float(0x7f800000); }

__device__ __forceinline__ float exp2_int(const int exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
  const int sign = raw >> 7;
  const int exp = (raw >> 3) & 0x0f;
  const int mant = raw & 0x07;

  if ((raw & 0x7f) == 0) {
    return sign ? -0.0f : 0.0f;
  }

  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.001953125f;  // 2^-9
  } else {
    if (exp == 0x0f && mant == 0x07) {
      return quiet_nan_f();
    }
    value = (1.0f + static_cast<float>(mant) * 0.125f) * exp2_int(exp - 7);
  }
  return sign ? -value : value;
}

__device__ __forceinline__ __half fp8_e5m2_to_half(uint8_t raw) {
  // E5M2 and IEEE fp16 use the same sign bit, exponent width, and exponent
  // bias. Expanding the two mantissa bits into the high fp16 mantissa bits is
  // exact for normals, subnormals, zero, inf, and NaN.
  return __ushort_as_half(static_cast<unsigned short>(raw) << 8);
}

// Finite E4M3 values map exactly to FP16 bits followed by FP32 scaling.
// Preserve the original NaN payload and signed zeros, including subnormals.
__device__ __forceinline__ float fp8_e4m3fn_to_float_bits(uint8_t raw) {
  const uint16_t half_bits = ((static_cast<uint16_t>(raw) << 7) & 0x3f80u) |
                             ((static_cast<uint16_t>(raw) << 8) & 0x8000u);
  const float value = __half2float(__ushort_as_half(half_bits)) * 256.0f;
  return (raw & 0x7fu) == 0x7fu ? quiet_nan_f() : value;
}

__device__ __forceinline__ __half2 fp8_e5m2_pair_to_half2(uint16_t raw_pair) {
  const uint32_t half2_bits = (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
                              (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = half2_bits;
  return converter.h2;
}

__device__ __forceinline__ __half2 load_fp8_e5m2_half2_unscaled(
    const void* __restrict__ cache, const int64_t byte_index) {
  const uint16_t* cache_u16 = reinterpret_cast<const uint16_t*>(cache);
  return fp8_e5m2_pair_to_half2(cache_u16[byte_index >> 1]);
}

__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t raw) {
  return __half2float(fp8_e5m2_to_half(raw));
}

// e2m1 code: sign = bit 3, exp = bits 2..1, mant = bit 0.
// value = exp == 0 ? mant * 0.5 : (1 + mant * 0.5) * 2^(exp - 1)
// i.e. the grid {0, .5, 1, 1.5, 2, 3, 4, 6}. Built as exact fp16 bits.
__device__ __forceinline__ float nvfp4_e2m1_to_float(const uint32_t nib) {
  const uint32_t mag = nib & 7u;
  const uint32_t bits =
      mag < 2u ? (mag ? 0x3800u : 0u)
               : ((((mag >> 1) + 14u) << 10) | ((mag & 1u) << 9));
  const float value = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
  return (nib & 8u) ? -value : value;
}

// Decode 8 consecutive e2m1 values (4 data bytes, element 2j in the low nibble
// of byte j) that share one E4M3FN block scale into 8 fp16 values. Each product
// (<= 2 mantissa bits x <= 4 mantissa bits, magnitude in [2^-10, 2688]) is
// exactly representable in fp16, so __float2half_rn is lossless.
__device__ __forceinline__ uint4 nvfp4_data4_to_half8(const uint32_t data,
                                                      const float scale) {
  uint32_t out[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t byte = (data >> (8 * j)) & 0xffu;
    const __half lo = __float2half_rn(nvfp4_e2m1_to_float(byte & 0xfu) * scale);
    const __half hi = __float2half_rn(nvfp4_e2m1_to_float(byte >> 4) * scale);
    out[j] = static_cast<uint32_t>(__half_as_ushort(lo)) |
             (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
  }
  return make_uint4(out[0], out[1], out[2], out[3]);
}

// ---------------------------------------------------------------------------
// 32-dim NVFP4 loader unit (Q1.23 P1.1).
//
// Half-precision bit patterns of the 16 e2m1 codes (sign = bit 3). This table
// is the single source of truth: the 8 magnitudes all have a zero low byte, so
// the device lookup below packs their high bytes into two 32-bit words and
// resolves four nibbles per PRMT instead of indexing a constant-memory table
// (constant-cache reads with a divergent per-thread index would serialize).
// tests/kernels/attention/test_sm70_flash_v100_nvfp4_e2m1_table.py parses this
// table and checks it against nvfp4_kv.E2M1_GRID.
constexpr uint16_t kNvfp4E2m1HalfBits[16] = {
    0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000, 0xc200, 0xc400, 0xc600};

constexpr uint32_t nvfp4_mag_hi_bytes(const int first) {
  return static_cast<uint32_t>(kNvfp4E2m1HalfBits[first] >> 8) |
         (static_cast<uint32_t>(kNvfp4E2m1HalfBits[first + 1] >> 8) << 8) |
         (static_cast<uint32_t>(kNvfp4E2m1HalfBits[first + 2] >> 8) << 16) |
         (static_cast<uint32_t>(kNvfp4E2m1HalfBits[first + 3] >> 8) << 24);
}
constexpr uint32_t kNvfp4MagHiBytesLo = nvfp4_mag_hi_bytes(0);  // mags 0..3
constexpr uint32_t kNvfp4MagHiBytesHi = nvfp4_mag_hi_bytes(4);  // mags 4..7
constexpr bool nvfp4_table_prmt_compatible() {
  for (int m = 0; m < 8; ++m) {
    // magnitudes: zero low byte, sign clear; negatives: same | 0x8000.
    if ((kNvfp4E2m1HalfBits[m] & 0x80ffu) != 0u ||
        kNvfp4E2m1HalfBits[m + 8] != (kNvfp4E2m1HalfBits[m] | 0x8000u)) {
      return false;
    }
  }
  return true;
}
static_assert(nvfp4_table_prmt_compatible(),
              "e2m1 table must have zero-low-byte magnitudes and mirrored "
              "negatives for the PRMT lookup");

// Four consecutive e2m1 codes (low 16 bits of `w16`, code i in nibble i) ->
// two half2 (dims 0,1 and 2,3), unscaled. One PRMT resolves the four magnitude
// high bytes; the sign nibble bits are moved to the fp16 sign positions.
__device__ __forceinline__ void nvfp4_data2_to_half2x2(const uint32_t w16,
                                                       uint32_t& h01,
                                                       uint32_t& h23) {
  const uint32_t hi = __byte_perm(kNvfp4MagHiBytesLo, kNvfp4MagHiBytesHi,
                                  w16 & 0x7777u);
  const uint32_t s = w16 & 0x8888u;
  h01 = __byte_perm(hi, 0u, 0x1404u) | ((s << 12) & 0x00008000u) |
        ((s << 24) & 0x80000000u);
  h23 = __byte_perm(hi, 0u, 0x3424u) | ((s << 4) & 0x00008000u) |
        ((s << 16) & 0x80000000u);
}

__device__ __forceinline__ uint32_t nvfp4_half2_scale(const uint32_t h2,
                                                      const __half2 scale2) {
  union {
    uint32_t u;
    __half2 h;
  } v;
  v.u = h2;
  v.h = __hmul2(v.h, scale2);
  return v.u;
}

// Decode 8 consecutive e2m1 codes (one 32-bit data word) with one fp16 block
// scale into 8 fp16 values. The e2m1 magnitude (<= 2 significant bits) times an
// E4M3FN scale (<= 4 significant bits) fits in 6 significant bits within the
// normal fp16 range [2^-10, 2688], so __hmul2 is exact (equal to the float
// path of nvfp4_data4_to_half8). A NaN scale byte propagates (0 * NaN = NaN).
__device__ __forceinline__ uint4 nvfp4_word_to_half8(const uint32_t word,
                                                     const __half2 scale2) {
  uint32_t a, b, c, d;
  nvfp4_data2_to_half2x2(word & 0xffffu, a, b);
  nvfp4_data2_to_half2x2(word >> 16, c, d);
  return make_uint4(nvfp4_half2_scale(a, scale2), nvfp4_half2_scale(b, scale2),
                    nvfp4_half2_scale(c, scale2), nvfp4_half2_scale(d, scale2));
}

// S1_DEQUANT_DIET (opt-in compile-time switch; default 0 = unchanged code):
//   0 = BASE, 1 = CS   (Diet C sign-bit pre-align word decode; scale function untouched),
//   3 = CSEP-NS (CSEP + NaN-preserving select, csep-h1v3-v1).
//   4 = L25 (NS + x256 folded into the e2m1 code table, no scale HMUL; l25-lut-fold-v1), 5 = L26 (L25 + page-id carry across KV tiles in s1_verify.cuh).
//   2 = CSEP (CS + raw_store scale decoded per scale byte via scale_u8_to_half2 + HMUL2 by 256).
// Source: agy wave2 live-TU arms (agy-sass-diet-wave2-livetu-20261024 live_s1_arms.cu / test_spill_fix.cu).
#ifndef S1_DEQUANT_DIET
#define S1_DEQUANT_DIET 0
#endif
#if S1_DEQUANT_DIET >= 4
// L25 (lut-fold-v1): the x256 of the E4M3 scale decode is folded into the e2m1 code table. Every magnitude is multiplied by 256
// (0.5..6 -> 128..1536, still zero low byte, still within the fp16 range) and the scale operand is the unscaled shifted-bits half
// (scale/256). code*256 * scale/256 = code*scale exactly (<= 6 significant bits, result in [2^-10, 2688]) so the HMUL2 output is bit-identical.
constexpr uint16_t kNvfp4E2m1HalfBitsX256[16] = {
    0x0000, 0x5800, 0x5c00, 0x5e00, 0x6000, 0x6200, 0x6400, 0x6600,
    0x8000, 0xd800, 0xdc00, 0xde00, 0xe000, 0xe200, 0xe400, 0xe600};
constexpr uint32_t nvfp4_mag_hi_bytes_x256(const int first) {
  return static_cast<uint32_t>(kNvfp4E2m1HalfBitsX256[first] >> 8) |
         (static_cast<uint32_t>(kNvfp4E2m1HalfBitsX256[first + 1] >> 8) << 8) |
         (static_cast<uint32_t>(kNvfp4E2m1HalfBitsX256[first + 2] >> 8) << 16) |
         (static_cast<uint32_t>(kNvfp4E2m1HalfBitsX256[first + 3] >> 8) << 24);
}
constexpr uint32_t kNvfp4MagHiBytesLoX256 = nvfp4_mag_hi_bytes_x256(0);
constexpr uint32_t kNvfp4MagHiBytesHiX256 = nvfp4_mag_hi_bytes_x256(4);
constexpr bool nvfp4_table_x256_consistent() {
  for (int m = 0; m < 8; ++m) {
    if ((kNvfp4E2m1HalfBitsX256[m] & 0x80ffu) != 0u ||
        kNvfp4E2m1HalfBitsX256[m + 8] != (kNvfp4E2m1HalfBitsX256[m] | 0x8000u)) return false;
  }
  // exponent field of every nonzero magnitude is the unscaled one + 8 (x256), mantissa unchanged
  for (int m = 1; m < 8; ++m) {
    if (kNvfp4E2m1HalfBitsX256[m] != kNvfp4E2m1HalfBits[m] + 0x2000u) return false;
  }
  return true;
}
static_assert(nvfp4_table_x256_consistent(), "x256 e2m1 table must be the base table with exponent +8");
#define S1_DIET_TBL_LO kNvfp4MagHiBytesLoX256
#define S1_DIET_TBL_HI kNvfp4MagHiBytesHiX256
#else
#define S1_DIET_TBL_LO kNvfp4MagHiBytesLo
#define S1_DIET_TBL_HI kNvfp4MagHiBytesHi
#endif
#if S1_DEQUANT_DIET != 0
// Diet C: sign bits pre-aligned with two PRMT (w<<4 merged with word), merged into the magnitude bytes with two lop3 (0x80808080).
__device__ __forceinline__ uint4 nvfp4_word_to_half8_diet(const uint32_t word, const __half2 scale2) {
  uint32_t hi_lo = __byte_perm(S1_DIET_TBL_LO, S1_DIET_TBL_HI, word & 0x7777u);
  uint32_t hi_hi = __byte_perm(S1_DIET_TBL_LO, S1_DIET_TBL_HI, (word >> 16) & 0x7777u);
  const uint32_t w_shl4 = word << 4;
  const uint32_t s_lo = __byte_perm(w_shl4, word, 0x5140u);
  const uint32_t s_hi = __byte_perm(w_shl4, word, 0x7362u);
  asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(hi_lo) : "r"(s_lo));
  asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(hi_hi) : "r"(s_hi));
  const uint32_t h01 = __byte_perm(hi_lo, 0u, 0x1404u);
  const uint32_t h23 = __byte_perm(hi_lo, 0u, 0x3424u);
  const uint32_t h45 = __byte_perm(hi_hi, 0u, 0x1404u);
  const uint32_t h67 = __byte_perm(hi_hi, 0u, 0x3424u);
  return make_uint4(nvfp4_half2_scale(h01, scale2), nvfp4_half2_scale(h23, scale2),
                    nvfp4_half2_scale(h45, scale2), nvfp4_half2_scale(h67, scale2));
}
#endif
#if S1_DEQUANT_DIET >= 2
// E4M3 scale byte -> fp16 pair: exponent/mantissa bits shifted into fp16 position, then x256 (HMUL by 0x5c00 = 256.0h).
// Scale bytes 0x7f / 0xff (NaN in E4M3) become +-480 here instead of NaN (the CSEP-only scan gate in the microbench covers this).
__device__ __forceinline__ __half2 scale_u8_to_half2(uint32_t raw) {
  uint16_t half_bits = uint16_t(((raw << 7) & 0x3f80u) | ((raw << 8) & 0x8000u));
#if S1_DEQUANT_DIET >= 4
  // L25: no HMUL; the x256 lives in the code table (kNvfp4E2m1HalfBitsX256). NaN select unchanged: 0x7f / 0xff -> 0x7fff.
  const uint16_t out_bits = ((raw & 0x7fu) == 0x7fu) ? uint16_t(0x7fffu) : half_bits;
  return __half2half2(__ushort_as_half(out_bits));
#else
  __half h = __hmul(__ushort_as_half(half_bits), __ushort_as_half(0x5c00u));
#if S1_DEQUANT_DIET == 3
  // CSEP-NS: low 7 bits all ones (E4M3 NaN, 0x7f / 0xff) -> the same half NaN the CS path yields
  // (CS: quiet_nan_f() = float 0x7fffffff -> __float2half_rn = 0x7fff, sign dropped).
  const uint16_t out_bits = ((raw & 0x7fu) == 0x7fu) ? uint16_t(0x7fffu) : __half_as_ushort(h);
  return __half2half2(__ushort_as_half(out_bits));
#else
  return __half2half2(h);
#endif
#endif
}
#endif

// Decode 32 consecutive dims: 16 data bytes + two scale bytes (low byte covers
// dims 0..15, high byte dims 16..31) -> 4 uint4 of fp16 (dims 8i..8i+7).
__device__ __forceinline__ void nvfp4_data16_to_half32(const uint4 data,
                                                       const uint32_t scales2,
                                                       uint4 (&out)[4]) {
  const __half2 s0 = __half2half2(
      __float2half_rn(fp8_e4m3fn_to_float(static_cast<uint8_t>(scales2))));
  const __half2 s1 = __half2half2(__float2half_rn(
      fp8_e4m3fn_to_float(static_cast<uint8_t>(scales2 >> 8))));
  out[0] = nvfp4_word_to_half8(data.x, s0);
  out[1] = nvfp4_word_to_half8(data.y, s0);
  out[2] = nvfp4_word_to_half8(data.z, s1);
  out[3] = nvfp4_word_to_half8(data.w, s1);
}

template <int KV_DTYPE, bool E4M3_BITS = false>
__device__ __forceinline__ float load_kv_cache_float_unscaled(
    const void* __restrict__ cache, const int64_t index) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    const __half* cache_h = reinterpret_cast<const __half*>(cache);
    return __half2float(cache_h[index]);
  } else {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    const uint8_t raw = cache_u8[index];
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 && E4M3_BITS) {
      return fp8_e4m3fn_to_float_bits(raw);
    }
    const float value = KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3
                            ? fp8_e4m3fn_to_float(raw)
                            : fp8_e5m2_to_float(raw);
    return value;
  }
}

template <int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  return load_kv_cache_float_unscaled<KV_DTYPE>(cache, index) * scale;
}

template <int KV_DTYPE>
__device__ __forceinline__ __half load_kv_cache_half(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2) {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    return __float2half_rn(__half2float(fp8_e5m2_to_half(cache_u8[index])) *
                           scale);
  }
  return __float2half_rn(load_kv_cache_float<KV_DTYPE>(cache, index, scale));
}

}  // namespace flash_v100
