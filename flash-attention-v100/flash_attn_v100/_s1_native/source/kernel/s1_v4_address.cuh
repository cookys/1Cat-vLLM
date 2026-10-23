// S1-v4-A: byte-address algebra only. Physical layout and WMMA fragments unchanged.
__host__ __device__ constexpr unsigned v4_mask(int row) {
  return ((row&3)|((row>>1)&4))^((row&1)<<2);
}
__host__ __device__ constexpr unsigned v4_byte_offset(int row,int col,int stride) {
  return unsigned(row*stride*2)+((unsigned(col)*2)^(v4_mask(row)*16));
}
__host__ __device__ constexpr int v4_a_row(int lane) {
  return (lane&3)+((lane>>4)&1)*4+((lane>>2)&1)*8;
}
__host__ __device__ constexpr int v4_bc_row(int lane) {
  return (lane&3)+((lane>>4)&1)*4+((lane>>3)&1)*8;
}
__host__ __device__ constexpr int v4_v_col(int lane,int warp) {
  return warp*16+((lane>>3)&1)*8+((lane>>4)&1)*4;
}
// Two packed relative byte coordinates replace four live base registers.
// Every component is <65536 (host-exhaustive contract); shared base is added
// only in the phase, so no assumption about the dynamic shared base address.
__host__ __device__ constexpr unsigned v4_qk_coordinates(int warp,int lane) {
  return v4_byte_offset((warp>>1)*16+v4_a_row(lane),0,256)
       | (v4_byte_offset((warp&1)*32+v4_bc_row(lane),0,256)<<16);
}
__host__ __device__ constexpr unsigned v4_pv_coordinates(int warp,int lane) {
  return v4_byte_offset(v4_a_row(lane),0,64)
       | (v4_byte_offset(lane&3,v4_v_col(lane,warp),256)<<16);
}
template<int Offset>
__device__ __forceinline__ uint4 v4_lds128(unsigned address) {
  uint4 x;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4+%5];"
    : "=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"r"(address),"n"(Offset):"memory");
  return x;
}
template<int Offset>
__device__ __forceinline__ uint2 v4_lds64(unsigned address) {
  uint2 x;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2+%3];"
    : "=r"(x.x),"=r"(x.y):"r"(address),"n"(Offset):"memory");
  return x;
}
template<class F,int O>
__device__ __forceinline__ void v4_load_pair(F& f,unsigned a,unsigned b) {
  uint4 x=v4_lds128<O>(a),y=v4_lds128<O>(b);
  f.x[0]=x.x;f.x[1]=x.y;f.x[2]=x.z;f.x[3]=x.w;
  f.x[4]=y.x;f.x[5]=y.y;f.x[6]=y.z;f.x[7]=y.w;
}
template<int K>
__device__ __forceinline__ void v4_qk_step(Acc& x0,Acc& x1,unsigned qbase,unsigned kbase) {
  constexpr int low=(K%64)*2, high=(K/64)*128;
  unsigned qa=qbase^low,qb=qbase^(low+16);
  unsigned ka=kbase^low,kb=kbase^(low+16);
  AFrag a;BCFrag b;
  v4_load_pair<AFrag,high>(a,qa,qb);
  v4_load_pair<BCFrag,high>(b,ka,kb);
  volta::mma_sync(x0,a,b,x0);
  // K row+16 has identical swizzle; use a single immediate, no second address chain.
  v4_load_pair<BCFrag,high+16*256*2>(b,ka,kb);
  volta::mma_sync(x1,a,b,x1);
}
template<int K,int M>
__device__ __forceinline__ void v4_pv_m(Acc& out,const BRFrag& v,unsigned pa,unsigned pb) {
  AFrag p;v4_load_pair<AFrag,M*16*64*2>(p,pa,pb);volta::mma_sync(out,p,v,out);
}
template<int K>
__device__ __forceinline__ void v4_pv_step(Acc (&out)[3],unsigned pbase,unsigned vbase,unsigned vx64) {
  BRFrag v;
  uint2 v0=v4_lds64<K*256*2>(vbase);
  uint2 v1=v4_lds64<(K+4)*256*2>(vbase);
  uint2 v2=v4_lds64<(K+8)*256*2>(vx64);
  uint2 v3=v4_lds64<(K+12)*256*2>(vx64);
  v.x[0]=v0.x;v.x[1]=v0.y;v.x[2]=v1.x;v.x[3]=v1.y;
  v.x[4]=v2.x;v.x[5]=v2.y;v.x[6]=v3.x;v.x[7]=v3.y;
  unsigned pa=pbase^(K*2),pb=pbase^((K+8)*2);
  v4_pv_m<K,0>(out[0],v,pa,pb);v4_pv_m<K,1>(out[1],v,pa,pb);v4_pv_m<K,2>(out[2],v,pa,pb);
}
