// S1 dense NVFP4/q8 only. Original BASE/A+A1B are unmodified references.
// N64, six QK warps reuse A over two N16 fragments; all sixteen PV warps
// reuse one V fragment across three M16 output tiles. No cross-user KV share.
namespace s1 {
using Acc = volta::fragment<volta::accumulator,16,16,16,float>;
using AFrag = volta::fragment<volta::matrix_a,16,16,16,half,volta::row_major>;
using BCFrag = volta::fragment<volta::matrix_b,16,16,16,half,volta::col_major>;
using BRFrag = volta::fragment<volta::matrix_b,16,16,16,half,volta::row_major>;
constexpr int Rows=48, N=64, D=256, Threads=512, Splits=80;
constexpr int ScoreStride=68, PStride=64;
struct alignas(256) Smem {
  union {
    struct {
      alignas(16) half q[Rows*D];
      // bank0: K(t) until the QK-dead barrier (named 1); then K(t+1) (H1: scores stay in registers).
      // bank1: V remains live until PV-done barrier.
      alignas(16) half bank0[N*D];
      alignas(16) half bank1[N*D];
      alignas(16) half probs[Rows*PStride];
    } x;
    alignas(16) float out[Rows*D];
  } u;
  float row_max[Rows], row_sum[Rows], row_scale[Rows];
  // H1 RS-KH: per-row partial max / sum of the two 32-key halves ([row][half]).
  float xmax[Rows*2], xsum[Rows*2];
};
static_assert(sizeof(Smem)<=96*1024,"sm70 shared budget");
static_assert(sizeof(Smem)==97792,"standalone P shared budget (H1: +768 B exchange slots)");

// Host-visible geometry definitions are also compiled by CPU contracts.
__host__ __device__ constexpr int active_splits(int length) {
  return length<=128 ? 1 : (length+63)/64<80 ? (length+63)/64 : 80;
}
__host__ __device__ constexpr int tile_start_for(int length,int split) {
  int a=active_splits(length),t=(length+63)/64,b=t/a,e=t-b*a;
  return (split*b+(split<e?split:e))*64;
}
__host__ __device__ constexpr int tile_end_for(int length,int split) {
  int a=active_splits(length),t=(length+63)/64,b=t/a,e=t-b*a;
  int end=tile_start_for(length,split)+(b+(split<e))*64;
  return end<length?end:length;
}

#include "s1_v4_address.cuh"

// H1V3 bit-equal row-sum: the K row a QK lane loads is remapped so the in-register fp32 sum
// tree equals the live v4_softmax tree. key = hf + 2*fc1 + 4*fc3 + 8*fc0 + 16*fc2 + 32*f.
__host__ __device__ constexpr int rk_key(int hf,int fc,int f) {
  return hf+2*((fc>>1)&1)+4*((fc>>3)&1)+8*(fc&1)+16*((fc>>2)&1)+32*f;
}
__host__ __device__ constexpr unsigned rk_qk_coordinates(int warp,int lane) {
  return v4_byte_offset((warp>>1)*16+v4_a_row(lane),0,256)
       | (v4_byte_offset(rk_key(warp&1,v4_bc_row(lane),0),0,256)<<16);
}
template<int K>
__device__ __forceinline__ void rk_qk_step(Acc& x0,Acc& x1,unsigned qbase,unsigned kbase) {
  constexpr int low=(K%64)*2, high=(K/64)*128;
  unsigned qa=qbase^low,qb=qbase^(low+16);
  unsigned ka=kbase^low,kb=kbase^(low+16);
  AFrag a;BCFrag b;
  v4_load_pair<AFrag,high>(a,qa,qb);
  v4_load_pair<BCFrag,high>(b,ka,kb);
  volta::mma_sync(x0,a,b,x0);
  // key+32 has the same swizzle (depends on row bits 0,1,3 only): one immediate.
  v4_load_pair<BCFrag,high+32*256*2>(b,ka,kb);
  volta::mma_sync(x1,a,b,x1);
}

// H1: QK keeps the (scaled) scores in the wmma accumulators. Named barrier 1 (512 threads:
// 6 QK warps arrive, 10 producer warps wait) publishes "K(t) dead" so bank0 may take K(t+1).
__device__ __forceinline__ void qk_group(Acc& x0,Acc& x1,unsigned qbase,unsigned kbase,float scale) {
  volta::fill_fragment(x0,0.f);volta::fill_fragment(x1,0.f);
  rk_qk_step<0>(x0,x1,qbase,kbase);
  rk_qk_step<16>(x0,x1,qbase,kbase);
  rk_qk_step<32>(x0,x1,qbase,kbase);
  rk_qk_step<48>(x0,x1,qbase,kbase);
  rk_qk_step<64>(x0,x1,qbase,kbase);
  rk_qk_step<80>(x0,x1,qbase,kbase);
  rk_qk_step<96>(x0,x1,qbase,kbase);
  rk_qk_step<112>(x0,x1,qbase,kbase);
  rk_qk_step<128>(x0,x1,qbase,kbase);
  rk_qk_step<144>(x0,x1,qbase,kbase);
  rk_qk_step<160>(x0,x1,qbase,kbase);
  rk_qk_step<176>(x0,x1,qbase,kbase);
  rk_qk_step<192>(x0,x1,qbase,kbase);
  rk_qk_step<208>(x0,x1,qbase,kbase);
  rk_qk_step<224>(x0,x1,qbase,kbase);
  rk_qk_step<240>(x0,x1,qbase,kbase);
  #pragma unroll
  for(int j=0;j<8;++j){x0.x[j]*=scale;x1.x[j]*=scale;}
  asm volatile("barrier.arrive 1, 512;" ::: "memory"); // 1: K-dead (all K LDS consumed above).
}

// H1 RS-KH: softmax directly on the wmma accumulators (sm_70 m16n16 fp32 lane map, see
// memory sm70-wmma-acc-layout-and-shfl-accounting): lane L holds, per fragment,
//   row0=8*((L>>2)&1)+(L&1)+4*((L>>4)&1), col0=8*((L>>3)&1)+(L&2)
//   x[0,1]@(row0,col0..+1) x[2,3]@(row0+2,..) x[4,5]@(row0,col0+4..+5) x[6,7]@(row0+2,col0+4..+5)
// so a row lives in the 4 lanes that differ in L1/L3 (SHFL xor 2, xor 8). Warp w = (m-tile, key half):
// the partner warp w^1 holds the other 32 keys of the same 16 rows. Max is exchanged through
// sm.xmax with named barrier 2+m (64 threads); the partial sums are published in sm.xsum and
// folded into row_sum by tid<Rows after barrier 2 (see the main loop). Per-row max / exp operations
// are identical to v4_softmax (P fp16 bit-equal); only the fp32 row-sum order differs.
template<bool Full>
__device__ __forceinline__ void rs_softmax(Smem& sm,Acc& x0,Acc& x1,half* probs,int t,int valid,int length,int w,int lane) {
  // lever 2: make the loop-invariant row/limit terms opaque so ptxas recomputes them here
  // instead of hoisting + spilling them (v1 hot LDL [R1+4/8/12] were max(0,length-8) and rA/6, rB/6).
  asm volatile("" : "+r"(length),"+r"(lane),"+r"(w));
  const int m=w>>1,hf=w&1;
  const int kb=hf+(lane&2)+((lane>>1)&4); // H1V3: key base, key = kb+32*f+16*(j>>1)+8*(j&1)
  const int rA=m*16+8*((lane>>2)&1)+(lane&1)+4*((lane>>4)&1),rB=rA+2;
  // Row-wise column validity: col valid <=> col <= cmax (col<valid && t+col<=max(0,length-8)+row/6).
  const int cmA=Full?N:min(valid-1,max(0,length-8)+rA/6-t);
  const int cmB=Full?N:min(valid-1,max(0,length-8)+rB/6-t);
  float sA[8],sB[8];
  #pragma unroll
  for(int f=0;f<2;++f){
    const Acc& x=f?x1:x0;
    #pragma unroll
    for(int j=0;j<4;++j){
      const int col=kb+32*f+16*(j>>1)+8*(j&1);
      const int ia=(j>>1)*4+(j&1),ib=ia+2;
      sA[f*4+j]=(Full||col<=cmA)?x.x[ia]:kXQANegInf;
      sB[f*4+j]=(Full||col<=cmB)?x.x[ib]:kXQANegInf;
    }
  }
  float mA=sA[0],mB=sB[0];
  #pragma unroll
  for(int j=1;j<8;++j){mA=fmaxf(mA,sA[j]);mB=fmaxf(mB,sB[j]);}
  mA=fmaxf(mA,__shfl_xor_sync(0xffffffffu,mA,2));mB=fmaxf(mB,__shfl_xor_sync(0xffffffffu,mB,2));
  mA=fmaxf(mA,__shfl_xor_sync(0xffffffffu,mA,8));mB=fmaxf(mB,__shfl_xor_sync(0xffffffffu,mB,8));
  const bool wr=(lane&10)==0; // one writer lane per row
  const float oldA=sm.row_max[rA],oldB=sm.row_max[rB]; // read before the pair barrier: the half-0 warp rewrites row_max after it
  if(wr){sm.xmax[rA*2+hf]=mA;sm.xmax[rB*2+hf]=mB;}
  // compile-time barrier ids (warp-uniform switch) so no register holds the id
  if(m==0)asm volatile("barrier.sync 2, 64;" ::: "memory");
  else if(m==1)asm volatile("barrier.sync 3, 64;" ::: "memory");
  else asm volatile("barrier.sync 4, 64;" ::: "memory");
  mA=fmaxf(mA,sm.xmax[rA*2+(hf^1)]);mB=fmaxf(mB,sm.xmax[rB*2+(hf^1)]);
  const float nmA=fmaxf(oldA,mA),nmB=fmaxf(oldB,mB);
  float pA[8],pB[8],psA,psB;
  #pragma unroll
  for(int f=0;f<2;++f){
    #pragma unroll
    for(int j=0;j<4;++j){
      const int col=kb+32*f+16*(j>>1)+8*(j&1);
      pA[f*4+j]=(Full||col<=cmA)?__expf(fmaxf(sA[f*4+j]-nmA,-80.f)):0.f;
      pB[f*4+j]=(Full||col<=cmB)?__expf(fmaxf(sB[f*4+j]-nmB,-80.f)):0.f;
    }
  }
  {
    // live tree: key bits 5,4,3 in-lane, then bit 2 (xor 8), bit 1 (xor 2); bit 0 = xsum[0]+xsum[1].
    float tA[4],tB[4];
    #pragma unroll
    for(int j=0;j<4;++j){tA[j]=pA[j]+pA[4+j];tB[j]=pB[j]+pB[4+j];}
    psA=(tA[0]+tA[2])+(tA[1]+tA[3]);psB=(tB[0]+tB[2])+(tB[1]+tB[3]);
  }
  psA+=__shfl_xor_sync(0xffffffffu,psA,8);psB+=__shfl_xor_sync(0xffffffffu,psB,8);
  psA+=__shfl_xor_sync(0xffffffffu,psA,2);psB+=__shfl_xor_sync(0xffffffffu,psB,2);
  #pragma unroll
  for(int f=0;f<2;++f){
    #pragma unroll
    for(int j=0;j<4;++j){
      const int key=kb+32*f+16*(j>>1)+8*(j&1); // partner keys sit in the other warp: scalar stores
      probs[a1::a1_offset(rA,key,PStride)]=__float2half_rn(pA[f*4+j]);
      probs[a1::a1_offset(rB,key,PStride)]=__float2half_rn(pB[f*4+j]);
    }
  }
  if(wr){
    sm.xsum[rA*2+hf]=psA;sm.xsum[rB*2+hf]=psB;
    if(hf==0){
      // sum>0 <=> some column of the row is valid <=> column 0 is valid (cmax>=0).
      const bool anyA=Full||cmA>=0,anyB=Full||cmB>=0;
      if(anyA)sm.row_max[rA]=nmA;
      if(anyB)sm.row_max[rB]=nmB;
      sm.row_scale[rA]=anyA?__expf(fmaxf(oldA-nmA,-80.f)):1.f;
      sm.row_scale[rB]=anyB?__expf(fmaxf(oldB-nmB,-80.f)):1.f;
    }
  }
}

// Same native launch argument ABI as the old partial kernel. Dispatch checks
// dense q8/Hq6/Hkv1/D256/page4096/NVFP4 before entering this specialized body.
__global__ __launch_bounds__(Threads,1)
void flash_attention_s1_verify_partial_kernel(
 const half* __restrict__ q,const void* __restrict__ kc,const void* __restrict__ vc,
 const int* __restrict__ bt,const int* __restrict__ sl,
 half* __restrict__ po,float* __restrict__ pl,int query_len,int max_blocks,int page_size,
 int64_t kbs,int64_t kts,int64_t khs,int64_t vbs,int64_t vts,int64_t vhs,
 float qscale,float vscale,const uint32_t* masks=nullptr,int batch=1,const int* row_lengths=nullptr) {
  int b=blockIdx.z,s=blockIdx.y,tid=threadIdx.x,w=tid>>5,lane=tid&31;
  if(blockIdx.x || b>=batch || query_len!=8)return;
  int length=sl[b];
  if(length<=0 || s>=active_splits(length))return;
  int start=tile_start_for(length,s),end=tile_end_for(length,s);
  if(start>=end)return;
  const int* pages=bt+int64_t(b)*max_blocks;
  extern __shared__ char s1_smem[];
  auto& sm=*reinterpret_cast<Smem*>(s1_smem);
  half* sq=sm.u.x.q;half* sk=sm.u.x.bank0;half* sv=sm.u.x.bank1;
  half* probs=sm.u.x.probs;
  auto* qv=reinterpret_cast<const uint4*>(q)+int64_t(b)*8*6*32;
  auto* qdst=reinterpret_cast<uint4*>(sq);
  for(int i=tid;i<Rows*32;i+=Threads) {
    int r=i>>5,c=i&31;qdst[r*32+(c^a1::a1_mask(r))]=__ldg(qv+i);
  }
  if(tid<Rows){sm.row_max[tid]=kXQANegInf;sm.row_sum[tid]=0.f;sm.row_scale[tid]=1.f;}
  // Every lane owns exactly one 32-dim raw unit in a complete N64 tile.
  int r=(tid>>5)*4+(tid&3),u=(tid>>2)&7; // Same 4-row warp footprint, conflict-free STS.128
  int kp=__ldg(pages+(start>>12));
  auto kr=a1::raw_load(reinterpret_cast<const uint8_t*>(kc)+int64_t(kp)*kbs,
                       start&4095,r,u,min(N,end-start));
  a1::raw_store(sk,kr,r,u);
  __syncthreads(); // Initial Q/K publication.
  Acc output[3];
  #pragma unroll
  for(int m=0;m<3;++m)volta::fill_fragment(output[m],0.f);
  const unsigned q_shared=static_cast<unsigned>(__cvta_generic_to_shared(sq));
  const unsigned k_shared=static_cast<unsigned>(__cvta_generic_to_shared(sk));
  const unsigned p_shared=static_cast<unsigned>(__cvta_generic_to_shared(probs));
  const unsigned v_shared=static_cast<unsigned>(__cvta_generic_to_shared(sv));
  const unsigned qk_coordinates=rk_qk_coordinates(w,lane);
#if S1_DEQUANT_DIET >= 5
  int vp_carry=kp; // L26: page of tile t; later tiles take it from the previous iteration's K(t+1) page id
#endif
  for(int t=start;t<end;t+=N) {
#if S1_DEQUANT_DIET >= 5
    int valid=min(N,end-t),vp=vp_carry;
#else
    int valid=min(N,end-t),vp=__ldg(pages+(t>>12));
#endif
    const int next=t+N,nvalid=max(0,min(N,end-next));
    // Six QK warps (QK + register softmax + P) run concurrently with ten producer warps (V(t), then K(t+1)).
    if(w<6) {
      // Keep derived XOR/immediate addresses phase-local; otherwise ptxas
      // hoists every K-step address across the loop and spills them.
      unsigned coordinates=qk_coordinates;
      asm volatile("" : "+r"(coordinates));
      unsigned qb=q_shared+(coordinates&65535u),kb=k_shared+(coordinates>>16);
      Acc x0,x1;
      qk_group(x0,x1,qb,kb,qscale);
      if(valid==N && t+N-1<=max(0,length-8))
        rs_softmax<true>(sm,x0,x1,probs,t,valid,length,w,lane);
      else
        rs_softmax<false>(sm,x0,x1,probs,t,valid,length,w,lane);
    } else {
      const int x=tid-192, x1=x+320;
      const int r0=(x>>5)*4+(x&3), u0=(x>>2)&7;
      const int r1=(x1>>5)*4+(x1&3), u1=(x1>>2)&7;
      const auto* page=reinterpret_cast<const uint8_t*>(vc)+int64_t(vp)*vbs;
      auto vr0=a1::raw_load(page,t&4095,r0,u0,valid);
      a1::Raw vr1{make_uint4(0,0,0,0),0u};
      if(x<192)vr1=a1::raw_load(page,t&4095,r1,u1,valid);
      // Fixed scalar packets: both raw loads precede dequant; no indexed array.
      asm volatile("" : "+r"(vr0.data.x),"+r"(vr0.data.y),"+r"(vr0.data.z),"+r"(vr0.data.w),"+r"(vr0.scales),
                          "+r"(vr1.data.x),"+r"(vr1.data.y),"+r"(vr1.data.z),"+r"(vr1.data.w),"+r"(vr1.scales) : : "memory");
      a1::raw_store(sv,vr0,r0,u0);
      if(x<192)a1::raw_store(sv,vr1,r1,u1);
      // K(t+1): raw loads issued now, dequant+STS into bank0 only after the QK warps released K(t).
#if S1_DEQUANT_DIET >= 5
      // L26: the table is only read when K(t+1) starts a new page; otherwise it is the page of V(t).
      const int np=nvalid?(((next>>12)==(t>>12))?vp:__ldg(pages+(next>>12))):0;
      vp_carry=np;
      const auto* kpage=reinterpret_cast<const uint8_t*>(kc)+int64_t(np)*kbs;
#else
      const auto* kpage=reinterpret_cast<const uint8_t*>(kc)+int64_t(nvalid?__ldg(pages+(next>>12)):0)*kbs;
#endif
      a1::Raw kr0{make_uint4(0,0,0,0),0u},kr1{make_uint4(0,0,0,0),0u};
      if(nvalid) {
        kr0=a1::raw_load(kpage,next&4095,r0,u0,nvalid);
        if(x<192)kr1=a1::raw_load(kpage,next&4095,r1,u1,nvalid);
      }
      asm volatile("" : "+r"(kr0.data.x),"+r"(kr0.data.y),"+r"(kr0.data.z),"+r"(kr0.data.w),"+r"(kr0.scales),
                          "+r"(kr1.data.x),"+r"(kr1.data.y),"+r"(kr1.data.z),"+r"(kr1.data.w),"+r"(kr1.scales) : : "memory");
      asm volatile("barrier.sync 1, 512;" ::: "memory"); // 1: K(t) dead.
      if(nvalid) {
        a1::raw_store(sk,kr0,r0,u0);
        if(x<192)a1::raw_store(sk,kr1,r1,u1);
      }
    }
    __syncthreads(); // 2: P, row_scale, V(t) and K(t+1) ready. (old sync3 removed)
    #pragma unroll
    for(int m=0;m<3;++m)a1::grouped_verify_scale_output_fragment(output[m],sm.row_scale,m*16);
    // V fragment reused across all three M16 tiles (A+A1B reloaded it 3x).
    // lever 2: recompute the packed PV coordinates per tile (v1 hot LDL [R1] was pv_coordinates).
    int pw=w,pl=lane;
    asm volatile("" : "+r"(pw),"+r"(pl));
    unsigned coordinates=v4_pv_coordinates(pw,pl);
    asm volatile("" : "+r"(coordinates));
    unsigned pb=p_shared+(coordinates&65535u),vb=v_shared+(coordinates>>16);
    const unsigned vx=vb^64;
    v4_pv_step<0>(output,pb,vb,vx);
    v4_pv_step<16>(output,pb,vb,vx);
    v4_pv_step<32>(output,pb,vb,vx);
    v4_pv_step<48>(output,pb,vb,vx);
    // Fold the two half-row partial sums into row_sum (row_scale published before barrier 2).
    if(tid<Rows) {
      const float sum=sm.xsum[tid*2]+sm.xsum[tid*2+1];
      if(sum>0.f)sm.row_sum[tid]=sm.row_sum[tid]*sm.row_scale[tid]+sum;
    }
    __syncthreads(); // 4: PV-dead (K(t+1) already in bank0); also protects final epilogue.
  }
  #pragma unroll
  for(int m=0;m<3;++m)volta::store_matrix_sync(sm.u.out+m*16*D+w*16,output[m],D,volta::mem_row_major);
  __syncthreads();
  int64_t base=(int64_t(b)*80+s)*Rows;
  for(int i=tid;i<Rows*D;i+=Threads) {
    int row=i/D;float sum=sm.row_sum[row];
    po[base*D+i]=__float2half_rn(sm.u.out[i]*(sum>0.f?vscale/sum:0.f));
  }
  if(tid<Rows){float sum=sm.row_sum[tid];pl[base+tid]=sum>0.f?sm.row_max[tid]+logf(sum):kXQANegInf;}
}
} // namespace s1
