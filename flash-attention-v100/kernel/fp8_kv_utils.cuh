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
