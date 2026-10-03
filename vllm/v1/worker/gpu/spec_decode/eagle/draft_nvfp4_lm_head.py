# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only NVFP4 lm_head with an exact FP16 rerank for SM70 MTP drafts.

The MTP draft samples one greedy token per step.  On a V100 the FP16 lm_head
GEMM for that step is HBM-bound (a 318 MB TP shard per rank), so reading a
4-bit copy of the same weights is the only lever left.  This module keeps the
FP16 shard untouched (the target verify path never sees any of this) and adds
a second, 89 MB NVFP4 copy that is used only to *screen* candidates:

1. ``q = QPN2(hidden)`` over the shard-local vocabulary (NVFP4 weights, FP16
   output).
2. ``cand = topk(q, R)``; ``R == 0`` skips 3-4 and returns the raw NVFP4
   argmax (diagnostic tier only).
3. Exact fp32 logits for the ``R`` candidate rows gathered from the FP16 shard.
4. Shard-local argmax over the exact candidate logits, then the same TP pair
   all-gather reduction as ``LogitsProcessor.get_top_tokens``.

The draft token can differ from the FP16 path for two reasons: (a) the FP16
argmax is outside the NVFP4 top-R, or (b) the top FP16 logits are within FP16
rounding of each other (the FP16 GEMM rounds every logit to FP16, so a gap
below one FP16 ulp is invisible to it) and the FP32 rerank resolves that
near-tie differently.  Both are a valid greedy proposal.  Greedy drafting
makes the proposal distribution one-hot, so the verifier's rejection sampling
stays exact for any proposal (quality tier E2: the target distribution is
unchanged, only the acceptance rate may move).

Ties.  Among candidates whose exact FP32 logits are equal the smallest
shard-local id wins, whatever order ``topk(sorted=False)`` returned them in.
The raw ``R == 0`` path (``q.max``) also returns the first, i.e. smallest,
index, and the TP pair reduction (``argmax`` over the rank-ordered gather)
returns the lowest rank, i.e. the lowest vocabulary range.  All three stages
agree with ``torch.max`` / ``LogitsProcessor.get_top_tokens``: an exact tie
resolves to the smallest global token id.

Nothing here is allowed to synchronize with the host inside
``local_top_tokens`` / ``top_tokens``: ``EagleSpeculator._sample_draft`` is
recorded inside the FULL CUDA graphs of the draft prefill and decode steps.
The hot-path temporaries are served from the CUDA-graph pool; there is no
host-side allocation or synchronization.

Quantization convention (see ``nvfp4_emulation_utils``)
-------------------------------------------------------
``dequant = fp4 * e4m3_block_scale * g``.  ``g`` is the *multiplier* consumed
by ``nvfp4_qpn2_gemm_sm70_out`` (the kernel computes
``fp16_rn(float(e4m3) * g)``).  The repo reference ``ref_nvfp4_quant`` takes the
reciprocal, ``global_scale = gq = 1 / g``.  ``g`` is always a float32-exact
python float so both sides agree on the value.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX_RECIPROCAL,
    _quantize_e4m3fn_scale,
)

logger = init_logger(__name__)

BLOCK_SIZE = 16
FP4_MAX = 6.0
E4M3_MAX = 448.0
# Smallest positive E4M3 value.  Blocks whose scale would round to zero (all
# zero blocks, or blocks 2**-10 below the shard maximum) get this scale so the
# division in the quantizer is always finite.
E4M3_MIN_POSITIVE = 2.0**-9
# nvfp4_qpn2_sm70.cu kQpn2MaxRows.
MAX_ROWS = 64
RERANK_K_CHOICES = (0, 8, 16, 32, 64, 128)
SUPPORTED_SPLIT_K = (8, 16, 32)
# Token ids travel through an fp32 pair all-gather; fp32 is exact below 2**24.
_MAX_EXACT_FP32_VOCAB = 1 << 24
# Below this the shard is treated as numerically empty.
_MIN_AMAX = 1e-12
# Sentinel id for non-maximal candidates in the smallest-id tie-break.
_INT64_MAX = torch.iinfo(torch.int64).max

# Load-time TP consensus (see ``build_draft_nvfp4_lm_head``): what a rank
# reports to its peers.
_STATE_OK = 1  # head built
_STATE_FALLBACK = 2  # recoverable: ineligible, bad value, missing op, Triton, ...
_STATE_FATAL = 3  # the CUDA context is not trustworthy (OOM, sticky error)
# Case-sensitive substrings of the RuntimeError text of a CUDA failure
# after which the rank must not touch the device again.  ``torch.cuda.
# OutOfMemoryError`` is matched by type.  The one place to extend the list.
_FATAL_CUDA_MARKERS = (
    "CUDA error",
    "illegal",
    "device-side assert",
    "unspecified launch failure",
)

_REQUIRED_OPS = ("nvfp4_qpn2_prepare_sm70", "nvfp4_qpn2_gemm_sm70_out")
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Thin seams over the native ops (monkeypatched by the CPU tests).
# ---------------------------------------------------------------------------
def _prepare_qpn2(
    packed: torch.Tensor, scales_e4m3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    codes, scales = sm70_ops.nvfp4_qpn2_prepare_sm70(packed, scales_e4m3)
    return codes, scales


def _qpn2_gemm_out(
    out: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
) -> None:
    sm70_ops.nvfp4_qpn2_gemm_sm70_out(
        out, x, codes, scales, global_scale, split_k, accumulator_chains
    )


def _use_triton_rerank(x: torch.Tensor) -> bool:
    return x.is_cuda


def _exact_candidate_logits(
    x: torch.Tensor,
    weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out[i, j] = x[i] . weight[candidate_ids[i, j]]`` in float32.

    On CUDA this is the in-tree Triton kernel ``indexed_fp32_logits`` (one
    program per (row, candidate), FP32 dot read straight from the FP16 rows).
    It takes our shapes as is: ``x`` FP16 [M, K] contiguous, ``weight`` FP16
    [N, K] row-major contiguous, ``candidate_ids`` int64 [M, R] contiguous
    shard-local ids, ``out`` FP32 [M, R] contiguous.  Other devices (the CPU
    unit tests) use the equivalent gather + FP32 batched matmul.
    """
    if _use_triton_rerank(x):
        from vllm.model_executor.layers.sm70_fp32_lm_head import indexed_fp32_logits

        indexed_fp32_logits(x, weight, candidate_ids, out)
        return
    m, r = candidate_ids.shape
    rows = weight.index_select(0, candidate_ids.reshape(-1)).view(m, r, -1)
    out.copy_(
        torch.bmm(x.float().unsqueeze(1), rows.float().transpose(1, 2)).squeeze(1)
    )


def _missing_ops() -> list[str]:
    return [name for name in _REQUIRED_OPS if not hasattr(torch.ops._C, name)]


def is_fatal_cuda_error(err: BaseException) -> bool:
    """Whether ``err`` leaves the CUDA context unusable for further work.

    ``torch.cuda.OutOfMemoryError`` and a ``RuntimeError`` carrying one of
    ``_FATAL_CUDA_MARKERS`` are fatal; everything else (ineligible shard,
    ``ValueError``, missing native op, Triton compile failure) is recoverable.
    """
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(err, RuntimeError):
        text = str(err)
        return any(marker in text for marker in _FATAL_CUDA_MARKERS)
    return False


def _tp_cpu_all_reduce(states: torch.Tensor) -> torch.Tensor:
    """Sum the CPU int tensor ``states`` over the TP group's CPU (gloo) group.

    Deliberately not ``tensor_model_parallel_all_reduce``: that one runs on the
    device group and needs a healthy CUDA context on every rank.
    """
    torch.distributed.all_reduce(
        states, op=torch.distributed.ReduceOp.SUM, group=get_tp_group().cpu_group
    )
    return states


def _tp_consensus(state: int) -> list[int]:
    """Every TP rank's ``_STATE_*``, indexed by TP rank; no collective at TP=1.

    Each rank writes its state into its own slot of a zero int32 CPU vector and
    one SUM all-reduce on the CPU group fills in all the others, so the result
    also tells which rank failed.  Never touches the GPU.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return [state]
    # Explicit device: an outer default-device context (cuda, meta) must not
    # move this vector, the gloo collective needs it on the CPU.
    table = torch.zeros(tp_size, dtype=torch.int32, device="cpu")
    table[get_tensor_model_parallel_rank()] = state
    return [int(v) for v in _tp_cpu_all_reduce(table).tolist()]


def _sm70_device_reason(device: torch.device) -> str | None:
    """None when ``device`` is an SM70 CUDA device, otherwise the reason."""
    if device.type != "cuda":
        return f"device {device} is not CUDA"
    capability = torch.cuda.get_device_capability(device)
    if tuple(capability) != (7, 0):
        return f"compute capability {tuple(capability)} is not (7, 0)"
    return None


def default_split_config(k: int, n: int) -> tuple[int, int]:
    """(split_k, accumulator_chains) the QPN2 linear path uses for this shape.

    ``_qpn2_config`` has no (2560, 62080) entry, so it falls back to
    ``(8 if k % 256 else 16, 2)``, i.e. ``(16, 2)`` at K=2560.
    """
    try:
        from vllm.model_executor.kernels.linear.nvfp4.sm70 import _qpn2_config

        return _qpn2_config(k, n, False)
    except Exception:  # pragma: no cover - import-time environment issues
        return (8 if k % 256 else 16, 2)


# ---------------------------------------------------------------------------
# Quantization (pure torch, device agnostic)
# ---------------------------------------------------------------------------
def ref_global_scale(global_scale: float, device: torch.device | str) -> torch.Tensor:
    """The ``ref_nvfp4_quant`` ``global_scale`` argument for multiplier ``g``.

    ``gq = 1 / g`` in float32.  Used by the quantizer itself and by the tests
    that compare against the repo reference.
    """
    g32 = torch.tensor(global_scale, dtype=torch.float32, device=device)
    return 1.0 / g32


def _fp4_codes(scaled: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (index | sign << 3) of values already scaled into [-6, 6].

    Round-half-even onto the code grid, bit-identical to ``cast_to_fp4``:
    ties go to the even code, i.e. 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2,
    2.5 -> 2, 3.5 -> 4, 5.0 -> 4.
    """
    magnitude = scaled.abs()
    index = (magnitude > 0.25).to(torch.uint8)
    index += magnitude >= 0.75
    index += magnitude > 1.25
    index += magnitude >= 1.75
    index += magnitude > 2.5
    index += magnitude >= 3.5
    index += magnitude > 5.0
    # A negative value that rounds to zero stays +0 (code 0, never code 8).
    negative = ((scaled < 0) & (index > 0)).to(torch.uint8)
    return index | (negative << 3)


@torch.no_grad()
def quantize_fp16_to_nvfp4_packed(
    weight: torch.Tensor, chunk_rows: int = 4096
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Quantize ``weight`` [N, K] (f16 / bf16 / f32) to packed NVFP4.

    Returns ``(packed uint8 [N, K/2], scales float8_e4m3fn [N, K/16], g)``.
    The low nibble of ``packed[:, j]`` is K index ``2j``, the high nibble
    ``2j + 1``; a nibble is ``index | sign << 3`` into
    ``{0, .5, 1, 1.5, 2, 3, 4, 6}``.  ``g = amax / (6 * 448)`` (a float32-exact
    python float) is the global multiplier, so the largest block scale is 448.

    Rows are processed in chunks so the temporaries stay bounded (about 6
    float32 copies of one chunk, ~250 MB at 4096 x 2560).
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported weight dtype {weight.dtype}")
    n, k = weight.shape
    if n == 0 or k == 0 or k % BLOCK_SIZE:
        raise ValueError(
            f"weight shape {(n, k)} must be non-empty with K % {BLOCK_SIZE} == 0"
        )
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    device = weight.device

    # Pass 1: shard-wide amax without materializing |W| for the whole shard.
    amax = torch.zeros((), dtype=torch.float32, device=device)
    for begin in range(0, n, chunk_rows):
        chunk_amax = weight[begin : begin + chunk_rows].abs().amax().float()
        amax = torch.maximum(amax, chunk_amax)
    amax_value = float(amax)
    if amax_value != amax_value or amax_value == float("inf"):
        raise ValueError("weight contains NaN or inf")
    if amax_value == 0.0:
        global_scale = 1.0
    elif amax_value < _MIN_AMAX:
        raise ValueError(f"weight amax {amax_value:g} is numerically empty")
    else:
        global_scale = float((amax / (FP4_MAX * E4M3_MAX)).item())

    gq = ref_global_scale(global_scale, device)
    inv_gq = 1.0 / gq
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((n, k // BLOCK_SIZE), dtype=torch.float8_e4m3fn, device=device)
    # Pass 2: same arithmetic as ref_nvfp4_quant, in float32, per row chunk.
    for begin in range(0, n, chunk_rows):
        end = min(begin + chunk_rows, n)
        rows = end - begin
        blocks = weight[begin:end].to(torch.float32).reshape(rows, -1, BLOCK_SIZE)
        vec_max = blocks.abs().amax(dim=-1, keepdim=True)
        scale = gq * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
        scale = torch.clamp(scale, min=0.0, max=E4M3_MAX)
        scale = _quantize_e4m3fn_scale(scale)
        scale = torch.clamp(scale, min=E4M3_MIN_POSITIVE)
        output_scale = 1.0 / (scale * inv_gq)
        scaled = torch.clamp(blocks * output_scale, -FP4_MAX, FP4_MAX)
        codes = _fp4_codes(scaled).view(rows, k)
        packed[begin:end] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        # Exact: ``scale`` already lies on the E4M3 grid.
        scales[begin:end] = scale.squeeze(-1).to(torch.float8_e4m3fn)
    return packed, scales, global_scale


def unpack_nvfp4_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    *,
    kernel_exact: bool = False,
) -> torch.Tensor:
    """Dequantize to float32 [N, K].

    Default: ``fp4 * (e4m3 * g)`` in float32, the formula of
    ``dequantize_to_dtype``.  With ``kernel_exact`` the scale product and the
    weight are rounded to FP16 the way the QPN2 kernel does
    (``half(fp4 * half(e4m3 * g))``).
    """
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise ValueError("packed must be a 2-D uint8 tensor")
    n, half_k = packed.shape
    k = half_k * 2
    if scales.shape != (n, k // BLOCK_SIZE):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} != {(n, k // BLOCK_SIZE)}"
        )
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(n, k)
    values = table[(codes & 0x07).long()]
    values = torch.where(codes & 0x08 != 0, -values, values)
    scale = scales.to(torch.float32) * global_scale
    if kernel_exact:
        scale = scale.half().float()
    out = (values.view(n, -1, BLOCK_SIZE) * scale.unsqueeze(-1)).reshape(n, k)
    if kernel_exact:
        out = out.half().float()
    return out


# ---------------------------------------------------------------------------
# The draft head
# ---------------------------------------------------------------------------
class DraftNvfp4LMHead:
    """NVFP4 screening copy of one lm_head shard plus the FP16 rerank.

    Draft only (quality tier E2).  The token can differ from the FP16 path
    when the FP16 argmax is outside the NVFP4 top-R, or when the top FP16
    logits sit within FP16 rounding of each other and the FP32 rerank resolves
    the near-tie differently; exact ties go to the smallest id.
    """

    @staticmethod
    def eligible(
        lm_head: Any, speculative_config: Any, device: torch.device
    ) -> tuple[bool, str]:
        """Whether the draft can use the NVFP4 head; ``(False, reason)`` if not.

        The kernel's own runtime asserts (nvfp4_qpn2_sm70.cu) are CUDA tensors,
        fp16 contiguous input/output, ``M in [1, 64]``, ``K % 64 == 0``,
        ``N % 32 == 0``, ``split_k in {8, 16, 32}`` with ``(K / 16) % split_k
        == 0`` and ``accumulator_chains in {1, 2}``.  The gate is stricter on K
        (``% 128``, the QPN2 linear contract) and checks the rest up front.
        """
        method = getattr(speculative_config, "method", None)
        if method not in ("mtp", "eagle"):
            return False, f"speculative method {method!r} is not mtp/eagle"
        sample_method = getattr(speculative_config, "draft_sample_method", "greedy")
        if sample_method != "greedy":
            return False, (
                f"draft_sample_method={sample_method!r}; the NVFP4 head only "
                "serves greedy drafts (probabilistic drafts need exact logits)"
            )
        reason = _sm70_device_reason(device)
        if reason is not None:
            return False, reason
        missing = _missing_ops()
        if missing:
            return False, f"missing native QPN2 operators: {missing}"
        weight = getattr(lm_head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            return False, "lm_head.weight is not a 2-D tensor"
        if weight.dtype != torch.float16:
            return False, f"lm_head weight dtype {weight.dtype} is not float16"
        if not weight.is_contiguous():
            return False, "lm_head weight is not contiguous (rerank reads rows)"
        if getattr(lm_head, "bias", None) is not None:
            return False, "lm_head has a bias"
        n, k = weight.shape
        if k % 128 or k <= 0:
            return False, f"K={k} is not a positive multiple of 128"
        if n % 32 or n < max(RERANK_K_CHOICES):
            return False, (
                f"N={n} must be a multiple of 32 and at least {max(RERANK_K_CHOICES)}"
            )
        split_k, chains = default_split_config(k, n)
        if split_k not in SUPPORTED_SPLIT_K or (k // BLOCK_SIZE) % split_k:
            return False, f"split_k={split_k} does not divide K/16={k // BLOCK_SIZE}"
        if chains not in (1, 2):
            return False, f"accumulator_chains={chains} is not 1 or 2"
        indices = getattr(lm_head, "shard_indices", None)
        if indices is None:
            return False, "lm_head has no shard_indices"
        if indices.num_org_vocab_padding != 0:
            return False, (
                f"shard has {indices.num_org_vocab_padding} vocab padding rows"
            )
        vocab = getattr(lm_head, "org_vocab_size", None)
        if vocab is not None and vocab >= _MAX_EXACT_FP32_VOCAB:
            return False, f"vocab size {vocab} is not exact in an fp32 id pair"
        return True, "ok"

    @torch.no_grad()
    def __init__(
        self,
        lm_head: Any,
        *,
        rerank_k: int,
        split_k: int | None = None,
        accumulator_chains: int | None = None,
        chunk_rows: int = 4096,
    ) -> None:
        if rerank_k not in RERANK_K_CHOICES:
            raise ValueError(f"rerank_k={rerank_k} not in {RERANK_K_CHOICES}")
        weight = lm_head.weight
        n, k = weight.shape
        default_split, default_chains = default_split_config(k, n)
        split_k = default_split if split_k is None else split_k
        accumulator_chains = (
            default_chains if accumulator_chains is None else accumulator_chains
        )
        if split_k not in SUPPORTED_SPLIT_K or (k // BLOCK_SIZE) % split_k:
            raise ValueError(f"unsupported split_k={split_k} for K={k}")
        if accumulator_chains not in (1, 2):
            raise ValueError(f"unsupported accumulator_chains={accumulator_chains}")

        device = weight.device
        cuda = device.type == "cuda"
        if cuda:
            torch.accelerator.synchronize(device)
            torch.accelerator.reset_peak_memory_stats(device)
            allocated_before = torch.accelerator.memory_allocated(device)
        started = time.perf_counter()

        self._device = device
        self.rows = n
        self.hidden_size = k
        self.split_k = split_k
        self.accumulator_chains = accumulator_chains
        self.org_vocab_start_index = int(lm_head.shard_indices.org_vocab_start_index)
        self.num_org_vocab_padding = int(lm_head.shard_indices.num_org_vocab_padding)
        # Shares storage with the target lm_head: the exact rerank reads it.
        self.weight = weight.detach()

        packed, scales_e4m3, self.global_scale = quantize_fp16_to_nvfp4_packed(
            weight, chunk_rows
        )
        self.codes, self.scales = _prepare_qpn2(packed, scales_e4m3)
        del packed, scales_e4m3
        # The big per-call buffers are allocated here and only sliced
        # afterwards (the QPN8 rerank template does the same).  The small
        # hot-path temporaries (max / masked_fill / min / all-gather results) are
        # served from the CUDA-graph pool while ``top_tokens`` is captured:
        # no host-side allocation or synchronization.
        self.out = torch.empty((MAX_ROWS, n), dtype=torch.float16, device=device)
        self._pair = torch.empty((MAX_ROWS, 2), dtype=torch.float32, device=device)
        self.rerank_k = rerank_k  # property: allocates the candidate buffers

        if cuda:
            torch.accelerator.synchronize(device)
        self.quantize_seconds = time.perf_counter() - started
        self.peak_extra_bytes = (
            torch.accelerator.max_memory_allocated(device) - allocated_before
            if cuda
            else 0
        )
        if cuda:
            torch.accelerator.empty_cache()
            # Compile/launch every kernel of the path now (Triton rerank, topk,
            # QPN2) so a problem surfaces as a load-time fallback and nothing
            # is JIT-compiled inside CUDA graph capture. No collective here.
            self.local_top_tokens(
                torch.zeros((1, k), dtype=torch.float16, device=device)
            )
            torch.accelerator.synchronize(device)
        logger.info(
            "SM70 MTP draft NVFP4 lm_head active: rows=%d K=%d rerank_k=%d "
            "split_k=%d chains=%d quantize=%.2fs peak_extra=%.1f MB "
            "resident=%.1f MB",
            n,
            k,
            rerank_k,
            split_k,
            accumulator_chains,
            self.quantize_seconds,
            self.peak_extra_bytes / 1e6,
            self.resident_bytes() / 1e6,
        )

    @property
    def rerank_k(self) -> int:
        return self._rerank_k

    @rerank_k.setter
    def rerank_k(self, value: int) -> None:
        """Change R (diagnostics only); reallocates the candidate buffers.

        Never call this while a CUDA graph that uses the head is captured.
        """
        if value not in RERANK_K_CHOICES:
            raise ValueError(f"rerank_k={value} not in {RERANK_K_CHOICES}")
        self._rerank_k = value
        if value == 0:
            self._topk_values = self._topk_ids = self._exact = None
            logger.warning_once(
                "SM70 MTP draft NVFP4 lm_head: rerank_k=0 is active. The draft "
                "token is the raw 4-bit argmax with no FP16 rerank (diagnostic "
                "tier; it can flip near-ties). Use VLLM_SM70_MTP_DRAFT_NVFP4_"
                "RERANK_K=64 for serving."
            )
            return
        # Separate exactly-sized tensors (not column slices of a wider one):
        # topk(out=...) and the Triton kernel need contiguous [M, R] rows.
        device = self._device
        self._topk_values = torch.empty(
            (MAX_ROWS, value), dtype=torch.float16, device=device
        )
        self._topk_ids = torch.empty(
            (MAX_ROWS, value), dtype=torch.int64, device=device
        )
        self._exact = torch.empty((MAX_ROWS, value), dtype=torch.float32, device=device)

    def resident_bytes(self) -> int:
        """Bytes held by the NVFP4 copy and every preallocated buffer."""
        tensors = (
            self.codes,
            self.scales,
            self.out,
            self._pair,
            self._topk_values,
            self._topk_ids,
            self._exact,
        )
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def local_top_tokens(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shard-local winner per row: ``(value float32 [M], global id int64 [M])``.

        Exact ties go to the smallest shard-local id.  No host synchronization:
        shapes and python ints only.
        """
        m, k = hidden_states.shape
        if not 1 <= m <= MAX_ROWS:
            raise ValueError(f"M={m} outside [1, {MAX_ROWS}]")
        if k != self.hidden_size or hidden_states.dtype != torch.float16:
            raise ValueError(
                f"hidden_states must be float16 [M, {self.hidden_size}], got "
                f"{hidden_states.dtype} {tuple(hidden_states.shape)}"
            )
        x = (
            hidden_states
            if hidden_states.is_contiguous()
            else hidden_states.contiguous()
        )
        q = self.out[:m]
        _qpn2_gemm_out(
            q,
            x,
            self.codes,
            self.scales,
            self.global_scale,
            self.split_k,
            self.accumulator_chains,
        )
        pad = self.num_org_vocab_padding
        if pad > 0:
            q[:, -pad:] = float("-inf")
        r = self._rerank_k
        if r == 0:
            values, local_ids = q.max(dim=-1)
            return values.float(), local_ids + self.org_vocab_start_index
        # The rerank is permutation invariant, so skip sorting the candidates.
        candidate_ids = self._topk_ids[:m]
        torch.topk(
            q, r, dim=-1, sorted=False, out=(self._topk_values[:m], candidate_ids)
        )
        # fp32 products against the FP16 rows: at least as accurate as the FP16
        # GEMM the target uses.
        exact = self._exact[:m]
        _exact_candidate_logits(x, self.weight, candidate_ids, exact)
        # The candidates arrive in topk's arbitrary order, so take the maximum
        # value and, among the candidates equal to it, the smallest id (what
        # torch.max / get_top_tokens return).  A NaN row keeps its NaN
        # candidates, so the sentinel can never leak into the result.
        best = exact.max(dim=-1, keepdim=True).values
        not_best = (exact != best) & (exact == exact)
        local_ids = candidate_ids.masked_fill(not_best, _INT64_MAX).min(dim=-1).values
        return best.squeeze(1), local_ids + self.org_vocab_start_index

    def top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Global draft token per row, int64 [M], identical on all TP ranks."""
        local_values, global_ids = self.local_top_tokens(hidden_states)
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return global_ids
        # Same reduction as LogitsProcessor.get_top_tokens: fp32 (value, id)
        # pairs are exact for ids below 2**24.
        pair = self._pair[: hidden_states.shape[0]]
        pair[:, 0] = local_values
        pair[:, 1] = global_ids
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        winner = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        tokens = gathered[:, :, 1].gather(dim=-1, index=winner)
        return tokens.squeeze(-1).to(torch.int64)


def build_draft_nvfp4_lm_head(
    model: Any,
    speculative_config: Any,
    device: torch.device,
    hidden_dtype: torch.dtype,
    rerank_k: int,
) -> DraftNvfp4LMHead | None:
    """Build the head for ``model`` or return None (always logging why).

    The caller has already checked the enabling env var.  A recoverable
    failure (ineligible shard, bad value, missing native op, Triton compile
    failure) never raises and keeps the FP16 path.

    With TP > 1 every rank joins one tiny all-reduce on the TP *CPU* (gloo)
    group, so a failure on any rank disables the head on all of them: the head
    adds collectives to the draft path and a mixed state would deadlock.  The
    consensus never touches the GPU: after a sticky CUDA error a rank could not
    take part in a device collective and its peers would hang.

    A fatal CUDA failure (``is_fatal_cuda_error``: OOM or a sticky error) on
    any rank, or at TP=1 on the only rank, makes every rank raise
    ``RuntimeError`` so the process group exits consistently instead of falling
    back on a context that may be unusable.
    """
    head: DraftNvfp4LMHead | None = None
    reason = ""
    fatal_error: Exception | None = None
    try:
        lm_head = getattr(model, "lm_head", None)
        scale = getattr(getattr(model, "logits_processor", None), "scale", 1.0)
        if lm_head is None:
            reason = "draft model exposes no lm_head"
        elif hidden_dtype != torch.float16:
            reason = f"model dtype {hidden_dtype} is not float16"
        elif scale <= 0.0:
            reason = f"logits scale {scale} is not positive (argmax not invariant)"
        else:
            ok, reason = DraftNvfp4LMHead.eligible(lm_head, speculative_config, device)
            if ok:
                head = DraftNvfp4LMHead(lm_head, rerank_k=rerank_k)
    except Exception as err:
        head = None
        reason = f"construction failed: {type(err).__name__}: {err}"
        if is_fatal_cuda_error(err):
            fatal_error = err

    # From here on a failed rank does no GPU work at all.
    if fatal_error is not None:
        state = _STATE_FATAL
    elif head is not None:
        state = _STATE_OK
    else:
        state = _STATE_FALLBACK
    states = _tp_consensus(state)

    if _STATE_FATAL in states:
        culprit = states.index(_STATE_FATAL)
        if fatal_error is not None:
            logger.error("SM70 MTP draft NVFP4 lm_head: %s", reason)
        raise RuntimeError(
            f"draft NVFP4 lm_head: fatal CUDA error on rank {culprit}; aborting startup"
        ) from fatal_error
    if head is not None and any(s != _STATE_OK for s in states):
        first = next(i for i, s in enumerate(states) if s != _STATE_OK)
        head = None  # drops the buffers; no device synchronization
        reason = (
            f"TP rank {first} could not build the head; every rank falls back "
            "to the FP16 path"
        )
    if head is None:
        logger.info("SM70 MTP draft NVFP4 lm_head disabled: %s", reason)
    return head
