// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Experimental multi-row native-NVFP4 decode. Python dispatch defaults off.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
constexpr int kMaxRoutes = 160;
constexpr int kPack = 8;
constexpr int kExperts = 512;
constexpr int kChunks = kMaxRoutes / kPack;

// Integer atomics only: route order within a pack is immaterial because both
// projections scatter back to the original route before the unchanged W2.
__global__ void plan_kernel(const int32_t* ids, int32_t* rows, int32_t* experts,
                            int32_t* sizes, int32_t* total, int routes) {
  __shared__ int counts[kExperts + 1];
  __shared__ int groups[kExperts + 1][kChunks];
  const int t = threadIdx.x;
  for (int e = t; e <= kExperts; e += blockDim.x) counts[e] = 0;
  if (t == 0) *total = 0;
  __syncthreads();
  int expert = 0, ordinal = 0;
  if (t < routes) {
    expert = ids[t];
    if (expert < 0 || expert >= kExperts) expert = kExperts;
    ordinal = atomicAdd(counts + expert, 1);
  }
  __syncthreads();
  if (t < routes && ordinal % kPack == 0) {
    const int group = atomicAdd(total, 1);
    groups[expert][ordinal / kPack] = group;
    experts[group] = expert;
    sizes[group] = min(kPack, counts[expert] - ordinal);
  }
  __syncthreads();
  if (t < routes) {
    const int group = groups[expert][ordinal / kPack];
    rows[group * kPack + ordinal % kPack] = t;
  }
}

__device__ __forceinline__ void decode(unsigned packed, half2 scale,
                                       half2* out) {
  constexpr unsigned sign = 0x80008000u, em = 0x0e000e00u;
  unsigned v[4] = {((packed << 12) & sign) | ((packed << 9) & em),
                   ((packed << 8) & sign) | ((packed << 5) & em),
                   ((packed << 4) & sign) | ((packed << 1) & em),
                   (packed & sign) | ((packed >> 3) & em)};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[i] = __hmul2(*reinterpret_cast<half2*>(v + i), scale);
}

#define PACKED_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// ---------------------------------------------------------------------------
// NoPlan (Q2.16 option C, VLLM_SM70_MOE_QPN_NO_PLAN=1, <=64 routes, M=5 only
// from Python).  plan_kernel is not launched; every w13 CTA derives its own
// group from the call's route ids instead.  Reading of the old plan:
//  * plan_kernel packs the routes of one expert into groups of <=8 rows
//    (ordinal / 8); group numbers and the row order inside a group come from
//    atomicAdd races, i.e. they are already unspecified.  Nothing the kernels
//    compute depends on them: every output row is a function of that row's
//    input and the expert's weights only, each row is written back to its own
//    route slot, and W2 merges rows in top-k order afterwards.
//  * Here route slot t (= blockIdx.y; slots 0..31 live in lane t of idA, slots
//    32..63 in lane t-32 of idB) is the leader of its group iff it is the
//    8k-th occurrence of its expert in slot order.  Non-leader CTAs exit before
//    any barrier (the decision is identical for all warps of the CTA).  The
//    leader's rows are the set bits of `rest` in ascending slot order.
//  * W2 still reads rows/experts/sizes/total, so the leader CTAs with
//    blockIdx.x == 0 write them (group number = leader rank in slot order,
//    total written once by slot 0, which is always a leader).  W2 is unchanged.
//  * Invalid ids (<0 or >=kExperts) share the expert-kExperts bucket, as in
//    plan_kernel: zero accumulators, zero rows written.
// What this buys, honestly: only the removal of plan_kernel's launch boundary
// and, inside w13, three dependent global loads (total -> sizes/experts ->
// rows) becoming one ids load plus ballots. W2 is untouched, so its three
// dependent L2 loads of the plan arrays remain. The expected gain of
// -0.5..-3 us per layer is a design estimate (plan 072 Q2.16); no GPU
// measurement exists yet. Leader CTAs of tile 0 add a small plan-publishing
// tail in warp 0, which may eat part of it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int sanitize_expert(int expert) {
  return (expert < 0 || expert >= kExperts) ? kExperts : expert;
}

// Position of the n-th (0-based) set bit; mask must have more than n bits.
__device__ __forceinline__ int nth_set_bit(unsigned long long mask, int n) {
#pragma unroll 1  // n <= 7 in practice; compact code beats unrolled 64-bit ops
  for (int i = 0; i < n; ++i) mask &= mask - 1;
  return __ffsll(static_cast<long long>(mask)) - 1;
}

template <int Split, bool Interleaved, bool NoPlan = false>
__global__ void w13_kernel(const half* x, const uint32_t* weights,
                           const half* scales, int32_t* rows, int32_t* experts,
                           int32_t* sizes, int32_t* total, half* out,
                           const int32_t* ids, int routes) {
  // Split within the CTA: no floating-point atomics or global partial tensor.
  __shared__ float partial[2][Split][kPack][32];
  __shared__ half projected[2][kPack][32];
  __shared__ int slot_rows[NoPlan ? kPack : 1];
  const int group_id = blockIdx.y;
  int count, expert;
  int idA = -1, idB = -1;  // NoPlan only: ids of route slots lane / lane + 32
  unsigned long long rest = 0;
  if constexpr (NoPlan) {
    const int plane = threadIdx.x % 32;
    idA = plane < routes ? sanitize_expert(ids[plane]) : -1;
    idB = plane + 32 < routes ? sanitize_expert(ids[plane + 32]) : -1;
    expert = __shfl_sync(0xffffffffu, group_id < 32 ? idA : idB, group_id & 31);
    const unsigned long long same =
        static_cast<unsigned long long>(__ballot_sync(0xffffffffu, idA == expert)) |
        (static_cast<unsigned long long>(
             __ballot_sync(0xffffffffu, idB == expert))
         << 32);
    const unsigned long long below = (1ull << group_id) - 1ull;
    const int ordinal = __popcll(same & below);
    if (ordinal % kPack != 0) return;
    count = min(kPack, __popcll(same) - ordinal);
    rest = same & ~below;
  } else {
    if (group_id >= *total) return;
    count = sizes[group_id];
    expert = experts[group_id];
  }
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int projection = warp / Split, split = warp % Split;
  const int tile =
      Interleaved ? blockIdx.x * 2 + projection : blockIdx.x + projection * 5;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + mma_row;
  int route = 0;
  if constexpr (NoPlan) {
    if (mma_row < count) route = nth_set_bit(rest, mma_row);
    if (warp == 0 && quad == 0 && mma_row < count) slot_rows[mma_row] = route;
  } else {
    route = mma_row < count ? rows[group_id * kPack + mma_row] : 0;
  }
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 2560 * 40;
    const half* s = scales + static_cast<size_t>(expert) * 160 * 320;
    const half* input = x + static_cast<size_t>(route / 10) * 2560;
#pragma unroll 4
    for (int g = split * (160 / Split); g < (split + 1) * (160 / Split); ++g) {
      const size_t offset =
          (static_cast<size_t>(tile) * 320 + g * 2) * 32 + col;
      const half scalar = __hmul(__ldg(s + (g * 10 + tile) * 32 + col),
                                 __float2half_rn(16384.0f));
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (mma_row < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }
      PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
      PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
      PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
      PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
    }
  }
  if constexpr (NoPlan) {
    // Publish the plan W2 consumes (see header comment). Warp 0 only, once
    // per leader CTA of tile 0; other warps proceed to the barrier below.
    if (blockIdx.x == 0 && warp == 0) {
      const unsigned below = (1u << lane) - 1u;
      const unsigned same_a = __match_any_sync(0xffffffffu, idA);
      const unsigned same_b = __match_any_sync(0xffffffffu, idB);
      int earlier_a = 0;  // slots 0..31 holding this lane's slot-32 expert
#pragma unroll 1  // runs once per leader CTA; keep the code small
      for (int j = 0; j < 32; ++j)
        earlier_a += __shfl_sync(0xffffffffu, idA, j) == idB;
      const bool lead_a = idA >= 0 && __popc(same_a & below) % kPack == 0;
      const bool lead_b =
          idB >= 0 && (earlier_a + __popc(same_b & below)) % kPack == 0;
      const unsigned lead_lo = __ballot_sync(0xffffffffu, lead_a);
      const unsigned lead_hi = __ballot_sync(0xffffffffu, lead_b);
      const int g = group_id < 32
                        ? __popc(lead_lo & ((1u << group_id) - 1u))
                        : __popc(lead_lo) +
                              __popc(lead_hi & ((1u << (group_id - 32)) - 1u));
      if (quad == 0 && mma_row < count) rows[g * kPack + mma_row] = route;
      if (lane == 0) {
        experts[g] = expert;
        sizes[g] = count;
        if (group_id == 0) *total = __popc(lead_lo) + __popc(lead_hi);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[projection][split][r][quad * 8 + c] = accum[i];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * kPack * 32; idx += blockDim.x) {
    const int p = idx / (kPack * 32), r = idx / 32 % kPack, c = idx % 32;
    // FP16 materialization is retained before SiLU, then again before the
    // multiplication. Split>1 changes FP32 association, not quantization.
    float value = 0;
#pragma unroll
    for (int s = 0; s < Split; ++s) value += partial[p][s][r][c];
    projected[p][r][c] = __float2half_rn(value);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < count * 32; idx += blockDim.x) {
    const int r = idx / 32, c = idx % 32;
    const int p = Interleaved ? c / 16 : 0;
    const int pc = Interleaved ? c % 16 * 2 : c;
    const half gate = projected[p][r][pc];
    const half up = Interleaved ? projected[p][r][pc + 1] : projected[1][r][c];
    const float gf = __half2float(gate);
    const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
    int dst;
    if constexpr (NoPlan) {
      dst = slot_rows[r];
    } else {
      dst = rows[group_id * kPack + r];
    }
    out[static_cast<size_t>(dst) * 160 + blockIdx.x * 32 + c] =
        __hmul(activated, up);
  }
}

void run_impl(torch::Tensor out, torch::Tensor x, torch::Tensor w,
              torch::Tensor s, torch::Tensor ids, torch::Tensor rows,
              torch::Tensor experts, torch::Tensor sizes, torch::Tensor total,
              int64_t split, bool interleaved, bool no_plan) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int routes = x.size(0) * 10;
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 2560 && routes > 0 && routes <= 160);
  // NoPlan keeps route slots in two 32-lane ballot words.
  TORCH_CHECK(!no_plan || routes <= 64, "NoPlan supports at most 64 routes");
  for (const auto& t : {out, x, w, s, ids, rows, experts, sizes, total}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  }
  TORCH_CHECK(x.scalar_type() == at::kHalf && s.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kHalf && w.scalar_type() == at::kInt);
  for (const auto& t : {ids, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(ids.numel() == routes && rows.numel() >= routes * kPack &&
              experts.numel() >= routes && sizes.numel() >= routes &&
              total.numel() == 1 && out.numel() == routes * 160 &&
              w.numel() == 512 * 2560 * 40 && s.numel() == 512 * 160 * 320);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (!no_plan) {
    plan_kernel<<<1, 256, 0, stream>>>(
        ids.data_ptr<int32_t>(), rows.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
        total.data_ptr<int32_t>(), routes);
  }
#define LAUNCH(S, I, N)                                                      \
  w13_kernel<S, I, N><<<dim3(5, routes), 64 * S, 0, stream>>>(               \
      reinterpret_cast<const half*>(x.data_ptr()),                           \
      reinterpret_cast<const uint32_t*>(w.data_ptr()),                       \
      reinterpret_cast<const half*>(s.data_ptr()), rows.data_ptr<int32_t>(), \
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),                \
      total.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr()),    \
      ids.data_ptr<int32_t>(), routes)
#define CASE(S)                  \
  case S:                        \
    if (no_plan) {               \
      if (interleaved) {         \
        LAUNCH(S, true, true);   \
      } else {                   \
        LAUNCH(S, false, true);  \
      }                          \
    } else if (interleaved) {    \
      LAUNCH(S, true, false);    \
    } else {                     \
      LAUNCH(S, false, false);   \
    }                            \
    break
  switch (split) {
    CASE(1);
    CASE(2);
    CASE(4);
    CASE(5);
    CASE(8);
    default:
      TORCH_CHECK(false, "Unsupported split");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef CASE
#undef LAUNCH
}

void run(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor s,
         torch::Tensor ids, torch::Tensor rows, torch::Tensor experts,
         torch::Tensor sizes, torch::Tensor total, int64_t split,
         bool interleaved) {
  run_impl(out, x, w, s, ids, rows, experts, sizes, total, split, interleaved,
           false);
}

// Same signature and outputs as run(); rows/experts/sizes/total are written by
// the w13 CTAs themselves (no plan_kernel launch). Q2.16 option C.
void run_noplan(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                torch::Tensor s, torch::Tensor ids, torch::Tensor rows,
                torch::Tensor experts, torch::Tensor sizes,
                torch::Tensor total, int64_t split, bool interleaved) {
  run_impl(out, x, w, s, ids, rows, experts, sizes, total, split, interleaved,
           true);
}

// All groups, including singletons, use one kernel. The earlier prototype
// launched separate repeated/singleton kernels; their overhead erased reuse.
__global__ void w2_kernel(const half* x, const uint32_t* weights,
                          const half* scales, const int32_t* rows,
                          const int32_t* experts, const int32_t* sizes,
                          const int32_t* total, half* out) {
  const int group = blockIdx.y * 4 + threadIdx.x / 32;
  if (group >= *total) return;
  const int count = sizes[group], expert = experts[group];
  const int lane = threadIdx.x % 32, quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), col = quad * 8 + r;
  const int route = r < count ? rows[group * kPack + r] : 0;
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 160 * 320;
    const half* s = scales + static_cast<size_t>(expert) * 10 * 2560;
    const half* input = x + static_cast<size_t>(route) * 160;
#pragma unroll
    for (int g = 0; g < 10; ++g) {
      const int offset = (blockIdx.x * 20 + g * 2) * 32 + col;
      const half scalar = __hmul(__ldg(s + (g * 80 + blockIdx.x) * 32 + col),
                                 __float2half_rn(16384.0f));
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (r < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }
      PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
      PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
      PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
      PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    if (row < count) {
      const int dst = rows[group * kPack + row];
      out[static_cast<size_t>(dst) * 2560 + blockIdx.x * 32 + quad * 8 + c] =
          __float2half_rn(accum[i]);
    }
  }
}

__global__ void reduce_kernel(const half* routed, const float* weights,
                              half* out) {
  const int token = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  float result = 0;
#pragma unroll
  for (int slot = 0; slot < 10; ++slot)
    result = fmaf(__half2float(routed[(token * 10 + slot) * 2560 + col]),
                  weights[token * 10 + slot], result);
  out[token * 2560 + col] = __float2half_rn(result);
}

// Each CTA owns 32 output columns for the entire token batch. A warp reuses
// one expert's weights across up to eight routed rows, and loops over actual
// groups rather than token slots. Keep the FP16 W2 boundary in CTA memory;
// after one barrier, reduce each token in the original top-k order. No global
// scatter tensor, floating-point atomics, or inter-CTA synchronization.
__global__ __launch_bounds__(1024) void w2_batch_reduce_kernel(
    const half* x, const uint32_t* weights, const half* scales,
    const float* topk, const int32_t* rows, const int32_t* experts,
    const int32_t* sizes, const int32_t* total, half* out, int tokens) {
  __shared__ half values[kMaxRoutes][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quad * 8 + r, tile = blockIdx.x;
  const int groups = *total;
  for (int group = warp; group < groups; group += 32) {
    const int count = sizes[group], expert = experts[group];
    const int route = r < count ? rows[group * kPack + r] : 0;
    float accum[8] = {};
    if (expert < kExperts) {
      const uint32_t* w = weights + static_cast<size_t>(expert) * 160 * 320;
      const half* s = scales + static_cast<size_t>(expert) * 10 * 2560;
      const half* input = x + static_cast<size_t>(route) * 160;
#pragma unroll
      for (int g = 0; g < 10; ++g) {
        const int offset = (tile * 20 + g * 2) * 32 + col;
        const half scalar = __hmul(__ldg(s + (g * 80 + tile) * 32 + col),
                                   __float2half_rn(16384.0f));
        const half2 scale = __halves2half2(scalar, scalar);
        half2 decoded[8];
        decode(__ldcs(w + offset), scale, decoded);
        decode(__ldcs(w + offset + 32), scale, decoded + 4);
        const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
        uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
        if (r < count) {
          lo = *reinterpret_cast<const uint4*>(input + g * 16);
          hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
        }
        PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
        PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
        PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
        PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < count) {
        values[rows[group * kPack + row]][quad * 8 + c] =
            __float2half_rn(accum[i]);
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    const int token = idx / 32, c = idx % 32;
    float weighted = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 10; ++slot) {
      weighted = fmaf(__half2float(values[token * 10 + slot][c]),
                      __ldg(topk + token * 10 + slot), weighted);
    }
    out[token * 2560 + tile * 32 + c] = __float2half_rn(weighted);
  }
}

void w2_dispatch(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                 torch::Tensor w, torch::Tensor s, torch::Tensor topk,
                 torch::Tensor rows, torch::Tensor experts, torch::Tensor sizes,
                 torch::Tensor total, bool batch_reduce) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 160 && x.size(0) % 10 == 0);
  const int routes = x.size(0), tokens = routes / 10;
  TORCH_CHECK(tokens >= 1 && tokens <= 16);
  for (const auto& t :
       {out, routed, x, w, s, topk, rows, experts, sizes, total})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {out, routed, x, s})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  for (const auto& t : {w, rows, experts, sizes, total})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(topk.scalar_type() == at::kFloat && topk.numel() == routes &&
              out.numel() == tokens * 2560 && routed.numel() == routes * 2560 &&
              rows.numel() >= routes * 8 && experts.numel() >= routes &&
              sizes.numel() >= routes && total.numel() == 1 &&
              w.numel() == 512 * 160 * 320 && s.numel() == 512 * 10 * 2560);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (batch_reduce) {
    TORCH_CHECK(tokens == 5 || tokens == 8 || tokens == 16,
                "Grouped batch reduction is screened for M5/M8/M16 only");
    w2_batch_reduce_kernel<<<80, 1024, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(w.data_ptr()),
        reinterpret_cast<const half*>(s.data_ptr()), topk.data_ptr<float>(),
        rows.data_ptr<int32_t>(), experts.data_ptr<int32_t>(),
        sizes.data_ptr<int32_t>(), total.data_ptr<int32_t>(),
        reinterpret_cast<half*>(out.data_ptr()), tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  w2_kernel<<<dim3(80, (routes + 3) / 4), 128, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const uint32_t*>(w.data_ptr()),
      reinterpret_cast<const half*>(s.data_ptr()), rows.data_ptr<int32_t>(),
      experts.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      total.data_ptr<int32_t>(), reinterpret_cast<half*>(routed.data_ptr()));
  reduce_kernel<<<dim3(10, tokens), 256, 0, stream>>>(
      reinterpret_cast<const half*>(routed.data_ptr()), topk.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
        torch::Tensor w, torch::Tensor s, torch::Tensor topk,
        torch::Tensor rows, torch::Tensor experts, torch::Tensor sizes,
        torch::Tensor total) {
  w2_dispatch(out, routed, x, w, s, topk, rows, experts, sizes, total, false);
}

void w2_batch_reduce(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                     torch::Tensor w, torch::Tensor s, torch::Tensor topk,
                     torch::Tensor rows, torch::Tensor experts,
                     torch::Tensor sizes, torch::Tensor total) {
  w2_dispatch(out, routed, x, w, s, topk, rows, experts, sizes, total, true);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "nvfp4_grouped_w13_sm70_out(Tensor(a!) out, Tensor x, Tensor w, Tensor "
      "s, Tensor ids, "
      "Tensor(b!) rows, Tensor(c!) experts, Tensor(d!) sizes, Tensor(e!) "
      "total, "
      "int split, bool interleaved) -> ()");
  m.def(
      "nvfp4_grouped_w13_noplan_sm70_out(Tensor(a!) out, Tensor x, Tensor w, "
      "Tensor s, Tensor ids, "
      "Tensor(b!) rows, Tensor(c!) experts, Tensor(d!) sizes, Tensor(e!) "
      "total, "
      "int split, bool interleaved) -> ()");
  m.def(
      "nvfp4_grouped_w2_sm70_out(Tensor(a!) out, Tensor(b!) routed, Tensor x, "
      "Tensor w, Tensor s, "
      "Tensor topk, Tensor rows, Tensor experts, Tensor sizes, Tensor total) "
      "-> ()");
  m.def(
      "nvfp4_grouped_w2_batch_reduce_sm70_out(Tensor(a!) out, Tensor routed, "
      "Tensor x, Tensor w, Tensor s, Tensor topk, Tensor rows, Tensor experts, "
      "Tensor sizes, Tensor total) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("nvfp4_grouped_w13_sm70_out", &run);
  m.impl("nvfp4_grouped_w13_noplan_sm70_out", &run_noplan);
  m.impl("nvfp4_grouped_w2_sm70_out", &w2);
  m.impl("nvfp4_grouped_w2_batch_reduce_sm70_out", &w2_batch_reduce);
}
