#pragma once
#include "pvh_loader_layout.h"

// Per thread: exactly two 32-dimension units for a complete N64/D256 panel.
// Fixed members rather than runtime-indexed arrays keep raw values in registers.
struct PVHRawUnit { uint4 data; uint32_t scales; };
struct PVHRawPair { PVHRawUnit first, second; };

template <bool FULL, int SLOT>
__device__ __forceinline__ PVHRawUnit pvh_load_unit(
    const uint8_t* page, int tile_offset, int valid, int tid) {
  const int unit_idx = pvh_loader::unit_index(tid, SLOT);
  const int row = pvh_loader::row(unit_idx);
  const int unit = pvh_loader::unit_in_row(unit_idx);
  PVHRawUnit raw{make_uint4(0,0,0,0),0u};
  if (FULL || row < valid) {
    raw.data = __ldg(reinterpret_cast<const uint4*>(
        page + pvh_loader::data_in_page_offset(tile_offset + row, unit)));
    raw.scales = __ldg(reinterpret_cast<const uint16_t*>(
        page + pvh_loader::scale_in_page_offset(tile_offset + row, unit)));
  }
  return raw;
}

template <bool FULL>
__device__ __forceinline__ PVHRawPair pvh_load_pair(
    const uint8_t* page, int tile_offset, int valid, int tid) {
  PVHRawPair raw{pvh_load_unit<FULL,0>(page,tile_offset,valid,tid),
                 pvh_load_unit<FULL,1>(page,tile_offset,valid,tid)};
  // Compiler-level dependency: finish both raw-load expressions before starting
  // any dequantization. This adds no SASS instruction; actual issue order and
  // V-load distance to QK are checked in disassembly, then NCU.
  asm volatile("" : "+r"(raw.first.data.x), "+r"(raw.first.data.y),
      "+r"(raw.first.data.z), "+r"(raw.first.data.w), "+r"(raw.first.scales),
      "+r"(raw.second.data.x), "+r"(raw.second.data.y),
      "+r"(raw.second.data.z), "+r"(raw.second.data.w), "+r"(raw.second.scales)
      : : "memory");
  return raw;
}

template <int SLOT>
__device__ __forceinline__ void pvh_store_unit(
    __half* shared, const PVHRawUnit& raw, int tid) {
  const int unit_idx=pvh_loader::unit_index(tid,SLOT);
  const int row=pvh_loader::row(unit_idx), unit=pvh_loader::unit_in_row(unit_idx);
  // Finite E4M3->float bits is exact, including subnormals and signed zeros;
  // NaNs use the same quiet_nan_f payload as the original branch decoder.
  const __half2 s0=__half2half2(__float2half_rn(flash_v100::fp8_e4m3fn_to_float_bits(
      static_cast<uint8_t>(raw.scales))));
  const __half2 s1=__half2half2(__float2half_rn(flash_v100::fp8_e4m3fn_to_float_bits(
      static_cast<uint8_t>(raw.scales >> 8))));
  uint4* out=reinterpret_cast<uint4*>(shared+
      pvh_loader::smem_half_offset(row,unit,0));
  out[0]=flash_v100::nvfp4_word_to_half8(raw.data.x,s0);
  out[1]=flash_v100::nvfp4_word_to_half8(raw.data.y,s0);
  out[2]=flash_v100::nvfp4_word_to_half8(raw.data.z,s1);
  out[3]=flash_v100::nvfp4_word_to_half8(raw.data.w,s1);
}

__device__ __forceinline__ void pvh_store_pair(
    __half* shared, const PVHRawPair& raw, int tid) {
  pvh_store_unit<0>(shared,raw.first,tid);
  pvh_store_unit<1>(shared,raw.second,tid);
}
