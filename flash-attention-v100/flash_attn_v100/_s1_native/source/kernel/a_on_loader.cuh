// Included within candidate A namespace; two raw slots are V(t) and K(t+1).
struct Raw { uint4 data; uint32_t scales; };
__device__ __forceinline__ Raw raw_load(const uint8_t* page,int token,int r,int u,int valid) {
  Raw x{make_uint4(0,0,0,0),0u};
  if(r<valid) {
    x.data=__ldg(reinterpret_cast<const uint4*>(page+int64_t(token+r)*128+u*16));
    x.scales=__ldg(reinterpret_cast<const uint16_t*>(page+int64_t(4096)*128+int64_t(token+r)*16+u*2));
  }
  return x;
}
__device__ __forceinline__ void raw_store(__half* dst,const Raw& x,int r,int u) {
  const __half2 a=__half2half2(__float2half_rn(flash_v100::fp8_e4m3fn_to_float_bits(uint8_t(x.scales))));
  const __half2 b=__half2half2(__float2half_rn(flash_v100::fp8_e4m3fn_to_float_bits(uint8_t(x.scales>>8))));
  auto* o=reinterpret_cast<uint4*>(dst);
  o[r*32+((u*4+0)^a1_mask(r))]=flash_v100::nvfp4_word_to_half8(x.data.x,a);
  o[r*32+((u*4+1)^a1_mask(r))]=flash_v100::nvfp4_word_to_half8(x.data.y,a);
  o[r*32+((u*4+2)^a1_mask(r))]=flash_v100::nvfp4_word_to_half8(x.data.z,b);
  o[r*32+((u*4+3)^a1_mask(r))]=flash_v100::nvfp4_word_to_half8(x.data.w,b);
}
__device__ __forceinline__ void pvb_load_vk_pair(__half* v,__half* k,
 const uint8_t* vp,const uint8_t* kp,int vt,int kt,int vv,int kv,int tid) {
 if(tid<256) {
   int r=tid>>3,u=tid&7;
   Raw vr=raw_load(vp,vt,r,u,vv),kr=raw_load(kp,kt,r,u,kv);
   // Pin both raw loads ahead of dequant; inspect scoreboard waits in SASS.
   asm volatile("" : "+r"(vr.data.x),"+r"(vr.data.y),"+r"(vr.data.z),"+r"(vr.data.w),"+r"(vr.scales),
                     "+r"(kr.data.x),"+r"(kr.data.y),"+r"(kr.data.z),"+r"(kr.data.w),"+r"(kr.scales) : : "memory");
   raw_store(v,vr,r,u);
   if(kv) raw_store(k,kr,r,u);
 }
}
