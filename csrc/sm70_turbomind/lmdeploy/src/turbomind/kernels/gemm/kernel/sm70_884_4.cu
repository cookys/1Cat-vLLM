// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/batch_kernel_sm70.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

#include <cstdlib>

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

// K-GEMM P1 (plan 075, Fable DESIGN daf9a9b7): whole-M decode tiles so a
// team decode step with M = 8*c (c = 10/12/16/20 -> M = 80/96/128/160) is
// one CTA row instead of a 64+16 pair.  Context::Filter keeps only the
// smallest CTA_M covering M, so M <= 64 problems never see these tiles.
// Prescaled counterparts are registered by the same list below.
// X(CTA_M, CTA_N, CTA_K, TG_M, TG_N, Stages, TILE_C_M, TILE_C_N)
#define SM70_NVF4_WHOLE_M_TILES(X)              \
  X(80, 128, 64, 2, 4, 2, -1, -1)               \
  X(80, 128, 64, 2, 2, 2, -1, -1)               \
  X(80, 128, 32, 2, 4, 4, -1, -1)               \
  X(80, 64, 64, 2, 2, 2, -1, -1)                \
  X(80, 64, 64, 2, 2, 4, -1, -1)                \
  X(96, 128, 64, 2, 4, 2, -1, -1)               \
  X(96, 64, 64, 2, 2, 2, -1, -1)                \
  X(128, 128, 64, 4, 2, 2, 64, 128)             \
  X(128, 64, 64, 4, 2, 2, -1, -1)               \
  X(160, 128, 64, 4, 2, 2, 80, 128)             \
  X(160, 64, 64, 4, 2, 2, 80, 64)               \
  X(160, 128, 32, 4, 2, 4, 80, 128) \
  /* P1b: warp 40x64, 2-warp small-N, wide-N, deeper stages (epilogue needs CTA_M*CTA_N <= 128*THREADS) */ \
  X(80, 64, 32, 2, 1, 4, -1, -1)                \
  X(80, 64, 64, 2, 1, 2, -1, -1)                \
  X(80, 256, 32, 2, 4, 4, 40, 256)              \
  X(80, 256, 64, 2, 4, 2, 40, 256)              \
  X(80, 128, 64, 2, 2, 4, -1, -1)               \
  X(80, 128, 128, 2, 2, 2, -1, -1)              \
  X(96, 256, 32, 2, 4, 4, 48, 256)              \
  X(96, 128, 32, 2, 4, 4, -1, -1)               \
  X(96, 128, 64, 2, 2, 2, -1, -1)

#define SM70_NVF4_WHOLE_M_TYPE(Cfg, M, N, K, TM, TN, ST, CM, CN) \
  Cfg::Type<M, N, K, TM, TN, 1, D, S, ST, true, 1, 16, CM, CN>

inline bool Sm70Nvfp4WholeMTilesEnabled() {
  const char* raw = std::getenv("VLLM_SM70_NVFP4_WHOLE_M_TILES");
  return raw == nullptr || std::atoi(raw) != 0;
}

namespace {

// Keep shape-specific small-N tactics out of prefill and unrelated CUDA graph
// shapes.
template <class Gemm, int ExactM>
class ExactMKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == ExactM && KernelImpl<Gemm>::is_feasible(desc);
  }
};

template <class Gemm, int ExactM, int ExactN, int ExactK>
class ExactMnkKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == ExactM && desc.n == ExactN && desc.k == ExactK &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

// Default-cache-B kernel for the exact Qwen3.8 TP4 W2 prefill descriptor.
// Expert-sorted prefill gives each expert several adjacent M tiles, so keeping
// B cacheable may reuse its FP4 weights across those tiles. The exact contract
// and default-on feature gate keep it out of unrelated descriptors and provide
// an operational rollback.
template <class Gemm>
class Qwen38Nvfp4W2CacheBKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    const char* enabled =
        std::getenv("VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL");
    return (!enabled || std::atoi(enabled) != 0) && desc.m >= 1280 &&
           desc.num == 512 && desc.n == 2560 && desc.k == 160 &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

// The full W13 N=320 shape is faster with N128 tiles, but its final tile does
// half-empty work. This N64 kernel is exposed only for the exact split-W13
// tail, while the first 256 columns retain the established N128 kernel.
template <class Gemm>
class Qwen38Nvfp4W13TailN64KernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    const char* enabled =
        std::getenv("VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL");
    return (!enabled || std::atoi(enabled) != 0) && desc.m >= 1280 &&
           desc.num == 512 && desc.n == 64 && desc.k == 2560 &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

}  // namespace

void Registry::sm70_884_4() {
  {
    using B = Config_QuantizedBatch<fp4_e2m1_t, kColMajor>;
    using Rows32 = B::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>;
    using Full64 =
        B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true>;
    using Full48K32 =
        B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>;
    using Full48K64 =
        B::Type<48, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>;
    using Tail48K32 =
        B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true, true>;
    using Tail64 =
        B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true, true>;
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Rows32::Kernel>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full64::Kernel, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full48K32::Kernel, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full48K64::Kernel, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Tail48K32::Kernel, true, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Tail64::Kernel, true, true>>());

  }

  if constexpr (1) {
    // clang-format off
        using C = Config_U4_d<kColMajor>;
        Add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 128, 64, 128>>();
        Add<C::Type<128, 128, 16, 2, 2, 1, D, S, 2, true, 1, 128, 64, 128>>();
        Add<C::Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 48, 128>>();
        Add<C::Type< 64, 128, 32, 2, 2, 1, D, D, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128, 64, 128>>();
        Add<C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256, 64, 1, 4, 2, D, S, 2, true, 1, 128>>();

        using CS = Config_U4_d_A8x64Swizzle<kColMajor>;
        using C32 = CS::Type< 8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 128>;
        using C64 = CS::Type< 8, 64, 64, 1, 2, 1, D, S, 2, true, 1, 128, -1, -1, 2>;
        Add(std::make_unique<ExactMKernelImpl<typename C32::Kernel, 5>>());
        Add(std::make_unique<ExactMKernelImpl<typename C64::Kernel, 5>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C64::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C64::Kernel, 1, 4096, 5120>>());

    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        // GroupSizeV=128
        using C = Config_U4_g<kColMajor>;
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // GroupSizeV=64
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 64,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 64>>();
        // GroupSizeV=32
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_MXF4<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_NVF4<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 16,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 16>>();
        if (Sm70Nvfp4WholeMTilesEnabled()) {
#define SM70_NVF4_ADD_WHOLE_M(M, N, K, TM, TN, ST, CM, CN) \
          Add<SM70_NVF4_WHOLE_M_TYPE(C, M, N, K, TM, TN, ST, CM, CN)>();
          SM70_NVF4_WHOLE_M_TILES(SM70_NVF4_ADD_WHOLE_M)
#undef SM70_NVF4_ADD_WHOLE_M
        }
        using Qwen38CacheB = C::Type<32, 128, 32, 1, 4, 1, D, D, 2, true, 1, 16>;
        Add(std::make_unique<Qwen38Nvfp4W2CacheBKernelImpl<typename Qwen38CacheB::Kernel>>());
        using Qwen38W13TailN64 = C::Type<32, 64, 32, 1, 2, 1, D, S, 2, true, 1, 16>;
        Add(std::make_unique<Qwen38Nvfp4W13TailN64KernelImpl<typename Qwen38W13TailN64::Kernel>>());
        using C32K64L1 = C::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 16>;
        using C32K64L2 = C::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 16, -1, -1, 2>;
        using C32K128L1 = C::Type<8, 32, 128, 1, 1, 1, D, S, 2, true, 1, 16>;
        using C32K128L2 = C::Type<8, 32, 128, 1, 1, 1, D, S, 2, true, 1, 16, -1, -1, 2>;
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K64L1::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K64L2::Kernel, 1, 5120, 4352>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K128L1::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K128L2::Kernel, 1, 5120, 4352>>());
    // clang-format on
  }
}

namespace {

template <class Gemm>
class PrescaledNvfp4KernelImpl final : public KernelImpl<Gemm> {
 public:
  PrescaledNvfp4KernelImpl() { this->info_.name += "_sm70_nvfp4_prescaled"; }
};

template <class Ordinary, class Prescaled>
Kernel* MatchPrescaled(const Kernel& control, bool batch = false,
                       bool mask_m = false) {
  static thread_local KernelImpl<typename Ordinary::Kernel> ordinary;
  const std::string name =
      ordinary.name() + (mask_m ? "_sm70_batch_supply_mtail"
                               : (batch ? "_sm70_batch_supply" : ""));
  if (control.name() != name) {
    return nullptr;
  }
  static thread_local PrescaledNvfp4KernelImpl<typename Prescaled::Kernel>
      scaled;
  return &scaled;
}

}  // namespace

Kernel* Sm70Nvfp4PrescaledCounterpart(const Kernel& control) {
  using C = Config_NVF4<kColMajor, 0>;
  using P = Config_NVF4_Prescaled<kColMajor>;
  using B = Config_QuantizedBatch<fp4_e2m1_t, kColMajor>;
  using BP = Config_QuantizedBatch<fp4_e2m1_t, kColMajor,
                                   Transform_HMMA_SIMT_B_PrescaledE2M1>;
  // Match the ordinary tile and split partition before changing its transform.
  if (auto* k = MatchPrescaled<B::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>, BP::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true>, BP::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>, BP::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<48, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>, BP::Type<48, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true, true>, BP::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true, true>>(control, true, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true, true>, BP::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true, true>>(control, true, true))
    return k;
#define SM70_NVF4_MATCH_WHOLE_M(M, N, K, TM, TN, ST, CM, CN)               \
  if (auto* k = MatchPrescaled<                                            \
          SM70_NVF4_WHOLE_M_TYPE(C, M, N, K, TM, TN, ST, CM, CN),          \
          SM70_NVF4_WHOLE_M_TYPE(P, M, N, K, TM, TN, ST, CM, CN)>(control)) \
    return k;
  SM70_NVF4_WHOLE_M_TILES(SM70_NVF4_MATCH_WHOLE_M)
#undef SM70_NVF4_MATCH_WHOLE_M
  if (auto* k = MatchPrescaled<
          C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>,
          P::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>>(
          control))
    return k;
  if (auto* k = MatchPrescaled<
          C::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>,
          P::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>>(
          control))
    return k;
  if (auto* k =
          MatchPrescaled<C::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>,
                         P::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>(
              control))
    return k;
  if (auto* k =
          MatchPrescaled<C::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>,
                         P::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>(
              control))
    return k;
  return MatchPrescaled<C::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 16>,
                        P::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 16>>(
      control);
}

}  // namespace turbomind::gemm
