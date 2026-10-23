// A1: physical shared layout only. WMMA m16n16k16 fragment ABI derived from
// CUDA12.9 sm70 probe/wmma_map.sass; all MMA and arithmetic stay unchanged.
__host__ __device__ constexpr int a1_mask(int row) {
  return ((row & 3) | ((row >> 1) & 4)) ^ ((row & 1) << 2);
}
__host__ __device__ constexpr int a1_offset(int row,int col,int stride) {
  return row*stride + (((col>>3)^a1_mask(row))<<3) + (col&7);
}
__device__ __forceinline__ uint4 a1_lds128(const half* p) {
  uint4 v;unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
    : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(a):"memory");
  return v;
}
__device__ __forceinline__ uint2 a1_lds64(const half* p) {
  uint2 v;unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
    : "=r"(v.x),"=r"(v.y):"r"(a):"memory");
  return v;
}
template<class Frag>
__device__ __forceinline__ void a1_contiguous(Frag& f,const half* base,int row,int col,int stride) {
  uint4 x=a1_lds128(base+a1_offset(row,col,stride));
  uint4 y=a1_lds128(base+a1_offset(row,col+8,stride));
  f.x[0]=x.x;f.x[1]=x.y;f.x[2]=x.z;f.x[3]=x.w;
  f.x[4]=y.x;f.x[5]=y.y;f.x[6]=y.z;f.x[7]=y.w;
}
__device__ __forceinline__ void a1_load(
 volta::fragment<volta::matrix_a,16,16,16,half,volta::row_major>& f,
 const half* base,int row,int col,int stride) {
 int lane=volta::get_lane_id();
 row+=(lane&3)+((lane>>4)&1)*4+((lane>>2)&1)*8;
 a1_contiguous(f,base,row,col,stride);
}
__device__ __forceinline__ void a1_load(
 volta::fragment<volta::matrix_b,16,16,16,half,volta::col_major>& f,
 const half* base,int row,int col,int stride) {
 int lane=volta::get_lane_id();
 // Transposed K is physically row-major [N,D].
 row+=(lane&3)+((lane>>4)&1)*4+((lane>>3)&1)*8;
 a1_contiguous(f,base,row,col,stride);
}
__device__ __forceinline__ void a1_load(
 volta::fragment<volta::matrix_b,16,16,16,half,volta::row_major>& f,
 const half* base,int row,int col,int stride) {
 int lane=volta::get_lane_id();row+=lane&3;
 col+=((lane>>3)&1)*8+((lane>>4)&1)*4;
 // B-row ABI holds 4 halves per physical row. Keep LDS.64, not an
 // unnecessary LDS.128 overfetch; Q/K/P use the requested LDS.128.
 #pragma unroll
 for(int i=0;i<4;++i) {
   uint2 v=a1_lds64(base+a1_offset(row+i*4,col,stride));
   f.x[2*i]=v.x;f.x[2*i+1]=v.y;
 }
}
