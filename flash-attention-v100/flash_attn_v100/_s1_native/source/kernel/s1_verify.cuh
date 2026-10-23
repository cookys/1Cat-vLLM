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
      // bank0: K until QK-dead barrier; then FP32 score. P is independent.
      // bank1: V remains live until PV-done barrier.
      alignas(16) half bank0[N*D];
      alignas(16) half bank1[N*D];
      alignas(16) half probs[Rows*PStride];
    } x;
    alignas(16) float out[Rows*D];
  } u;
  float row_max[Rows], row_sum[Rows], row_scale[Rows];
};
static_assert(sizeof(Smem)<=96*1024,"sm70 shared budget");
static_assert(Rows*ScoreStride*4<=N*D*2,"dead K score alias budget");
static_assert(sizeof(Smem)==97024,"standalone P shared budget");

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

// Publish QK only after every warp has finished reading aliased K.
__device__ __forceinline__ void qk_group(unsigned qbase,unsigned kbase,float* scores,float scale) {
  int w=threadIdx.x>>5;
  Acc x0,x1;
  if(w<6) {
    volta::fill_fragment(x0,0.f);volta::fill_fragment(x1,0.f);
    int m=w>>1,n=(w&1)*32;
    v4_qk_step<0>(x0,x1,qbase,kbase);
    v4_qk_step<16>(x0,x1,qbase,kbase);
    v4_qk_step<32>(x0,x1,qbase,kbase);
    v4_qk_step<48>(x0,x1,qbase,kbase);
    v4_qk_step<64>(x0,x1,qbase,kbase);
    v4_qk_step<80>(x0,x1,qbase,kbase);
    v4_qk_step<96>(x0,x1,qbase,kbase);
    v4_qk_step<112>(x0,x1,qbase,kbase);
    v4_qk_step<128>(x0,x1,qbase,kbase);
    v4_qk_step<144>(x0,x1,qbase,kbase);
    v4_qk_step<160>(x0,x1,qbase,kbase);
    v4_qk_step<176>(x0,x1,qbase,kbase);
    v4_qk_step<192>(x0,x1,qbase,kbase);
    v4_qk_step<208>(x0,x1,qbase,kbase);
    v4_qk_step<224>(x0,x1,qbase,kbase);
    v4_qk_step<240>(x0,x1,qbase,kbase);
    #pragma unroll
    for(int j=0;j<8;++j){x0.x[j]*=scale;x1.x[j]*=scale;}
  }
  asm volatile("barrier.sync 1, 192;" ::: "memory"); // 1: six QK warps, K-dead.
  if(w<6) {
    int m=w>>1,n=(w&1)*32;
    volta::store_matrix_sync(scores+m*16*ScoreStride+n,x0,ScoreStride,volta::mem_row_major);
    volta::store_matrix_sync(scores+m*16*ScoreStride+n+16,x1,ScoreStride,volta::mem_row_major);
  }
}

template<bool Full>
__device__ __forceinline__ void v4_softmax_row(Smem& sm,float* scores,half* probs,int t,int valid,int length,int w,int lane,int row) {
      bool v0=Full || (lane<valid && t+lane<=max(0,length-8)+row/6);
      bool v1=Full || (lane+32<valid && t+lane+32<=max(0,length-8)+row/6);
      float s0=v0?scores[row*ScoreStride+lane]:kXQANegInf;
      float s1=v1?scores[row*ScoreStride+lane+32]:kXQANegInf;
      float mx=warp_reduce_max(fmaxf(s0,s1));mx=__shfl_sync(0xffffffffu,mx,0);
      float old=sm.row_max[row],nm=fmaxf(old,mx);
      float p0=v0?__expf(fmaxf(s0-nm,-80.f)):0.f;
      float p1=v1?__expf(fmaxf(s1-nm,-80.f)):0.f;
      float sum=warp_reduce_sum(p0+p1);sum=__shfl_sync(0xffffffffu,sum,0);
      float a=sum>0.f?__expf(fmaxf(old-nm,-80.f)):1.f;
      probs[a1::a1_offset(row,lane,PStride)]=__float2half_rn(p0);
      probs[a1::a1_offset(row,lane+32,PStride)]=__float2half_rn(p1);
      if(lane==0) {
        if(sum>0.f){sm.row_sum[row]=sm.row_sum[row]*a+sum;sm.row_max[row]=nm;}
        sm.row_scale[row]=a;
      }

}
template<bool Full>
__device__ __forceinline__ void v4_softmax(Smem& sm,float* scores,half* probs,int t,int valid,int length,int w,int lane) {
  #pragma unroll 1
  for(int row=w;row<Rows;row+=16)
    v4_softmax_row<Full>(sm,scores,probs,t,valid,length,w,lane,row);
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
  float* scores=reinterpret_cast<float*>(sk);
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
  const unsigned qk_coordinates=v4_qk_coordinates(w,lane);
  const unsigned pv_coordinates=v4_pv_coordinates(w,lane);
  for(int t=start;t<end;t+=N) {
    int valid=min(N,end-t),vp=__ldg(pages+(t>>12));
    // Six QK warps run concurrently with ten V producer warps.
    if(w<6) {
      // Keep derived XOR/immediate addresses phase-local; otherwise ptxas
      // hoists every K-step address across the loop and spills them.
      unsigned coordinates=qk_coordinates;
      asm volatile("" : "+r"(coordinates));
      unsigned qb=q_shared+(coordinates&65535u),kb=k_shared+(coordinates>>16);
      qk_group(qb,kb,scores,qscale);
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
    }
    __syncthreads(); // 2: all scores AND V ready; separates named generations.
    // Only interior complete tiles omit validity/causal guards. Last two tiles retain them.
    if(valid==N && t+N-1<=max(0,length-8))
      v4_softmax<true>(sm,scores,probs,t,valid,length,w,lane);
    else
      v4_softmax<false>(sm,scores,probs,t,valid,length,w,lane);
    __syncthreads(); // 3: P and scale ready; V published at barrier 2. Scores dead.
    int next=t+N,nvalid=max(0,min(N,end-next));
    int np=nvalid?__ldg(pages+(next>>12)):0;
    auto nr=a1::raw_load(reinterpret_cast<const uint8_t*>(kc)+int64_t(np)*kbs,
                         next&4095,r,u,nvalid);
    #pragma unroll
    for(int m=0;m<3;++m)a1::grouped_verify_scale_output_fragment(output[m],sm.row_scale,m*16);
    // V fragment reused across all three M16 tiles (A+A1B reloaded it 3x).
    unsigned coordinates=pv_coordinates;
    asm volatile("" : "+r"(coordinates));
    unsigned pb=p_shared+(coordinates&65535u),vb=v_shared+(coordinates>>16);
    const unsigned vx=vb^64;
    v4_pv_step<0>(output,pb,vb,vx);
    v4_pv_step<16>(output,pb,vb,vx);
    v4_pv_step<32>(output,pb,vb,vx);
    v4_pv_step<48>(output,pb,vb,vx);
    // P is independent: each warp may store Knext after its own PV.
    if(nvalid)a1::raw_store(sk,nr,r,u);
    __syncthreads(); // 4: Knext-ready AND PV-dead; also protects final epilogue.
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
