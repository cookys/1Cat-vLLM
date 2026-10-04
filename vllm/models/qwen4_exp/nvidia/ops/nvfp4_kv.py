# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch reference for the Qwen4Exp QSA NVFP4 main K/V cache (plan 071).

This module is the executable specification of the format. It has no kernels,
no CUDA dependency and no platform imports, so the format, the zero-fill rule
and the page layout can be pinned down on CPU before any Triton/CUDA store or
sparse-gather kernel exists. Later kernels are tested against these functions.

Format (one K row or one V row = ``head_size`` values of one KV head)::

    16 consecutive values            -> 8 data bytes + 1 scale byte
    data byte j  = (nib[2j+1] << 4) | nib[2j]        low nibble = EVEN element
    nib          = sign (bit 3) | E2M1 code (bits 2..0)
    E2M1 code    -> 0, .5, 1, 1.5, 2, 3, 4, 6
    scale byte   = E4M3FN (non-negative; 0x7F/0xFF decode to NaN)

    sf    = (1 / layer_scale) * (amax16 * fp32(1/6))   per 16-value group
    sf_q  = E4M3(min(sf, 448))                         round-to-nearest-even
    out   = 1 / (sf_q * layer_scale)    (0 when sf_q is 0)
    nib   = E2M1(clamp(x * out, -6, 6))
    x_hat = E2M1(nib) * sf_q * layer_scale

This is the operation order of upstream's SM100/SM120 store
(``csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`` and
``quantization/fp4/nvfp4_utils.cuh:221-285``), so the SM100 unit tests apply to
it. ``layer_scale`` is the checkpoint's per-layer scalar K or V scale, the
dequantization multiplier of ``reshape_and_cache_flash`` (the kernel is given
``1 / k_scale``; the FlashInfer reader folds ``k_scale`` into ``bmm1_scale``).
Upstream's tests use ``amax / 448`` per tensor, and the E4M3 QSA overlay scales
are of that kind. The block scale is capped at 448, so the scale only has to
be at least ``amax / 2688``; with 1.0 the block scale alone carries the dynamic
range and the error is the same (see the tests).

Choices that keep the later fused kernel simple:

* The low nibble holds the even element. This is the convention of the repo's
  NVFP4 *weight* emulation path (``nvfp4_emulation_utils.break_fp4_bytes`` and
  ``_dequantize_nvfp4_kernel``) and of upstream's SM100 KV store, so a kernel
  can reuse ``_e2m1_inline`` and the E4M3 software decoder unchanged. It is
  *not* the QPN2 tensor-core weight order ``[0,2,4,6,1,3,5,7]``.
* A K or V side of one cache page is two regions, data first and scales second
  (``[K_data | K_scale | V_data | V_scale]``, the layout of
  ``vllm.utils.torch_utils.nvfp4_kv_cache_split_views``). The logical cache
  shape is ``(num_blocks, 2, block_size, num_kv_heads, 144)`` for ``head_size``
  256, where 144 = 128 data + 16 scale bytes; the last dimension is the size of
  the pair, not a stride.
* Scales are linear (not SM100-swizzled), for K and V alike.
* The writer never emits the "negative zero" nibble 0x8: a group element that
  rounds to zero is stored as 0x0, so equal values are equal bytes.
* Illegal top-k indices (negative, past the block table, unmapped or
  out-of-range block, bad request) are zero-filled in the gather, data and scale
  alike, and never read. The SGLang QSA NVFP4 patch documents why: a stale
  scale byte can be an E4M3 NaN (0x7F/0xFF) and a masked-but-read NaN poisons
  the softmax-weighted sum.

Everything here computes in float32 with IEEE division and rounds to nearest-even.
The SM100 store uses ``rcp.approx.ftz.f32`` for the three reciprocals, which is
within one ulp of the IEEE result, so a byte can differ only when an fp32
intermediate lies within an ulp of an E4M3 or E2M1 rounding boundary. That
residual cannot be reproduced on V100 and is not tested.
"""

from __future__ import annotations

import math

import torch

from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim, nvfp4_kv_cache_split_views

NVFP4_KV_CACHE_DTYPE = "nvfp4"
NVFP4_KV_GROUP_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9
E4M3_MIN_NORMAL = 2.0**-6
# Largest per-layer scale the loader accepts. A group's block scale is
# amax16 / (6 * layer_scale); below 2**-6 it is an E4M3 subnormal with fewer than
# three mantissa bits. For a layer whose amax is at least 1 that happens for
# layer_scale > 1 / (6 * 2**-6) = 10.67. The calibrated K/V amax of the
# Flash-Next overlay is above 3 (its scalars times 200 or 448), so 10 leaves
# room and still rejects a scale that was meant for another cache format.
NVFP4_KV_LAYER_SCALE_MAX = 10.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# fp32(1/6): what the GPU's rcp.approx.ftz(6.0f) returns when it rounds correctly.
_ONE_SIXTH = torch.tensor(1.0 / 6.0, dtype=torch.float32)

__all__ = [
    "E2M1_GRID",
    "E2M1_MAX",
    "E4M3_MAX",
    "E4M3_MIN_NORMAL",
    "E4M3_MIN_SUBNORMAL",
    "NVFP4_KV_LAYER_SCALE_MAX",
    "NVFP4_KV_CACHE_DTYPE",
    "NVFP4_KV_GROUP_SIZE",
    "check_nvfp4_layer_scale",
    "dequantize_kv_nvfp4",
    "e2m1_decode",
    "e2m1_encode",
    "e4m3_decode",
    "e4m3_encode",
    "gather_dequant_nvfp4_kv",
    "nvfp4_block_scale_range",
    "nvfp4_kv_bytes_per_token",
    "nvfp4_kv_data_bytes",
    "nvfp4_kv_page_bytes",
    "nvfp4_kv_page_regions",
    "nvfp4_kv_row_bytes",
    "nvfp4_kv_scale_bytes",
    "nvfp4_kv_split_views",
    "nvfp4_layer_scale_from_amax",
    "nvfp4_subnormal_group_fraction",
    "pack_nibbles",
    "quantize_kv_nvfp4",
    "reshape_and_cache_nvfp4_reference",
    "unpack_nibbles",
]


# --------------------------------------------------------------------------- #
# Byte accounting
# --------------------------------------------------------------------------- #
def _check_head_size(head_size: int) -> None:
    if head_size <= 0 or head_size % NVFP4_KV_GROUP_SIZE:
        raise ValueError(
            f"NVFP4 KV head size must be a positive multiple of "
            f"{NVFP4_KV_GROUP_SIZE}, got {head_size}"
        )


def nvfp4_kv_data_bytes(head_size: int) -> int:
    """Packed E2M1 bytes of one K or V row: two values per byte."""
    _check_head_size(head_size)
    return head_size // 2


def nvfp4_kv_scale_bytes(head_size: int) -> int:
    """E4M3 scale bytes of one K or V row: one per 16-value group."""
    _check_head_size(head_size)
    return head_size // NVFP4_KV_GROUP_SIZE


def nvfp4_kv_row_bytes(head_size: int) -> int:
    """Data plus scale bytes of one K (or V) row; 144 for ``head_size`` 256."""
    _check_head_size(head_size)
    return nvfp4_kv_cache_full_dim(head_size)


def nvfp4_kv_bytes_per_token(head_size: int, num_kv_heads: int = 1) -> int:
    """K and V bytes of one token in one layer: 288 for 256 x 1 head."""
    return 2 * num_kv_heads * nvfp4_kv_row_bytes(head_size)


def nvfp4_kv_page_bytes(block_size: int, num_kv_heads: int, head_size: int) -> int:
    """Bytes of one cache page of ``block_size`` tokens (K and V)."""
    return block_size * nvfp4_kv_bytes_per_token(head_size, num_kv_heads)


def nvfp4_kv_page_regions(
    block_size: int, num_kv_heads: int, head_size: int
) -> dict[str, tuple[int, int]]:
    """Byte ``(offset, size)`` of the four regions of one page.

    ``[K_data | K_scale | V_data | V_scale]``: each side is data then scale.
    """
    data = block_size * num_kv_heads * nvfp4_kv_data_bytes(head_size)
    scale = block_size * num_kv_heads * nvfp4_kv_scale_bytes(head_size)
    side = data + scale
    return {
        "k_data": (0, data),
        "k_scale": (data, scale),
        "v_data": (side, data),
        "v_scale": (side + data, scale),
    }


# --------------------------------------------------------------------------- #
# E4M3 (scale) and E2M1 (data) scalar codecs
# --------------------------------------------------------------------------- #
def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """float -> E4M3FN byte, round-to-nearest-even, saturating at +-448.

    Native ``float -> float8_e4m3fn`` conversion turns anything beyond the
    range into NaN, so the input is clamped first (the CUDA converters used for
    the E4M3 KV cache saturate the same way). NaN input stays NaN.
    """
    clamped = x.float().clamp(-E4M3_MAX, E4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_decode(code: torch.Tensor) -> torch.Tensor:
    """E4M3FN byte (uint8 or float8_e4m3fn tensor) -> float32."""
    if code.dtype == torch.float8_e4m3fn:
        return code.to(torch.float32)
    if code.dtype != torch.uint8:
        raise TypeError(f"E4M3 codes must be uint8 or float8_e4m3fn, got {code.dtype}")
    return code.view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """float -> 4-bit E2M1 code (sign << 3 | magnitude), round-to-nearest-even.

    Magnitudes saturate at 6. Ties go to the even code: 0.25 -> 0, 0.75 -> 1,
    1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4. A result of zero never
    carries a sign (0x8 is not produced).
    """
    magnitude = x.float().abs()
    code = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )
    negative = (x.float() < 0) & (code != 0)
    return code | (negative.to(torch.uint8) << 3)


def e2m1_decode(nibble: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 code (low 4 bits of a uint8) -> float32.

    Code 0x8 decodes to -0.0, as IEEE would; the writer never produces it.
    """
    grid = torch.tensor(E2M1_GRID, dtype=torch.float32, device=nibble.device)
    magnitude = grid[(nibble & 0x7).long()]
    return torch.where((nibble & 0x8).bool(), -magnitude, magnitude)


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """``[..., D]`` 4-bit codes -> ``[..., D // 2]`` bytes, low nibble = even."""
    if nibbles.shape[-1] % 2:
        raise ValueError("an even number of nibbles is required")
    pairs = nibbles.reshape(*nibbles.shape[:-1], -1, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """``[..., D // 2]`` bytes -> ``[..., D]`` 4-bit codes, even element first."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed data must be uint8, got {packed.dtype}")
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], -1
    )


# --------------------------------------------------------------------------- #
# Global (per-layer) scale
# --------------------------------------------------------------------------- #
def check_nvfp4_layer_scale(layer_scale: float) -> float:
    """Validate a per-layer K or V scale and return it as a float.

    It must be finite and positive, and at most ``NVFP4_KV_LAYER_SCALE_MAX``:
    a larger scale pushes the block scales of an ordinary layer into the E4M3
    subnormal range, where the scale loses mantissa bits.
    """
    value = float(layer_scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"NVFP4 KV layer scale must be finite and positive: {value}")
    if value > NVFP4_KV_LAYER_SCALE_MAX:
        raise ValueError(
            f"NVFP4 KV layer scale {value} exceeds {NVFP4_KV_LAYER_SCALE_MAX}: "
            "the block scales of a layer with amax near 1 would be E4M3 "
            "subnormals"
        )
    return value


def nvfp4_layer_scale_from_amax(amax: float) -> float:
    """Layer scale that puts the layer's largest block scale at the E4M3 maximum.

    ``amax / (6 * 448)``: the smallest scale that never saturates a block scale
    for values up to ``amax``. It uses the whole normal range of the block scale
    (``2**-6 .. 448``), at the price of no headroom above ``amax``.
    """
    if not math.isfinite(amax) or amax <= 0.0:
        raise ValueError(f"amax must be finite and positive: {amax}")
    return float(amax) / (E2M1_MAX * E4M3_MAX)


def nvfp4_block_scale_range(layer_scale: float, amax: float) -> tuple[float, float]:
    """``(largest block scale, smallest E4M3-normal group amax)`` of a layer.

    The first is the block scale of a group whose amax is the layer's ``amax``;
    the second is the smallest group amax whose block scale is still an E4M3
    normal number (at least ``2**-6``).
    """
    scale = check_nvfp4_layer_scale(layer_scale)
    return amax / (E2M1_MAX * scale), E2M1_MAX * scale * E4M3_MIN_NORMAL


def nvfp4_subnormal_group_fraction(
    x: torch.Tensor, layer_scale: float | torch.Tensor = 1.0
) -> float:
    """Fraction of non-zero 16-value groups whose block scale is below ``2**-6``.

    Such groups store a subnormal (or flushed) E4M3 scale. A diagnostic for
    choosing a layer scale; the quantizer itself accepts them.
    """
    _check_head_size(x.shape[-1])
    groups = x.float().reshape(*x.shape[:-1], -1, NVFP4_KV_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    scale = torch.as_tensor(layer_scale, dtype=torch.float32, device=x.device)
    block = (1.0 / scale) * (amax * _ONE_SIXTH)
    nonzero = amax > 0
    if not bool(nonzero.any()):
        return 0.0
    return float((block[nonzero] < E4M3_MIN_NORMAL).float().mean())


# --------------------------------------------------------------------------- #
# Row quantizer
# --------------------------------------------------------------------------- #
def quantize_kv_nvfp4(
    x: torch.Tensor, *, layer_scale: float | torch.Tensor = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K or V rows ``[..., head_size]`` to NVFP4.

    Returns ``(packed, scales)``: ``packed`` is uint8 ``[..., head_size // 2]``,
    ``scales`` is ``float8_e4m3fn`` ``[..., head_size // 16]`` (one byte per
    group; ``scales.view(torch.uint8)`` is what the cache stores). ``x`` must
    be finite. ``layer_scale`` is the checkpoint's per-layer K or V scalar.
    """
    head_size = x.shape[-1]
    _check_head_size(head_size)
    values = x.float()
    if not torch.isfinite(values).all():
        raise ValueError("NVFP4 KV quantization requires finite inputs")
    scale = torch.as_tensor(layer_scale, dtype=torch.float32, device=x.device)
    if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise ValueError("NVFP4 KV layer scale must be finite and positive")

    groups = values.reshape(*x.shape[:-1], -1, NVFP4_KV_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    # The SM100 store's order: SFScaleVal = 1 / k_scale; SFValue = SFScaleVal *
    # (vecMax * rcp(6)); outputScale = rcp(SFValue * rcp(SFScaleVal)).
    scale_inverse = 1.0 / scale
    block_scale = e4m3_encode(scale_inverse * (amax * _ONE_SIXTH))
    block_value = e4m3_decode(block_scale)
    inverse = torch.where(
        block_value > 0,
        1.0 / (block_value * (1.0 / scale_inverse)),
        torch.zeros_like(block_value),
    )
    scaled = (groups * inverse.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    nibbles = e2m1_encode(scaled).reshape(*x.shape[:-1], head_size)
    return pack_nibbles(nibbles), block_scale.view(torch.float8_e4m3fn)


def dequantize_kv_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    layer_scale: float | torch.Tensor = 1.0,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Inverse of :func:`quantize_kv_nvfp4`; ``scales`` may be uint8 or float8.

    A NaN scale byte (0x7F/0xFF) decodes to NaN, exactly as the hardware
    converters would; the gather below never reads such bytes for invalid
    entries.
    """
    if packed.shape[-1] * 2 != scales.shape[-1] * NVFP4_KV_GROUP_SIZE:
        raise ValueError(
            f"packed {tuple(packed.shape)} and scales {tuple(scales.shape)} "
            "do not describe the same row"
        )
    values = e2m1_decode(unpack_nibbles(packed))
    groups = values.reshape(*values.shape[:-1], -1, NVFP4_KV_GROUP_SIZE)
    scale = torch.as_tensor(layer_scale, dtype=torch.float32, device=packed.device)
    block = (e4m3_decode(scales) * scale).unsqueeze(-1)
    return (groups * block).reshape(values.shape).to(out_dtype)


# --------------------------------------------------------------------------- #
# Paged cache: writer and sparse gather
# --------------------------------------------------------------------------- #
def _cache_views(kv_cache: torch.Tensor):
    """``(k_data, v_data), (k_scale, v_scale)`` of a 5-D NVFP4 cache (uint8)."""
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "NVFP4 cache must be (num_blocks, 2, block_size, heads, row_bytes), "
            f"got {tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"NVFP4 cache must be uint8, got {kv_cache.dtype}")
    data, scales = nvfp4_kv_split_views(kv_cache)
    return data, scales


def nvfp4_kv_split_views(kv_cache: torch.Tensor):
    """Data (uint8) and scale (uint8) views of each side of a 5-D cache."""
    (k_data, v_data), (k_scale, v_scale) = nvfp4_kv_cache_split_views(kv_cache)
    return (k_data, v_data), (k_scale.view(torch.uint8), v_scale.view(torch.uint8))


def reshape_and_cache_nvfp4_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale: float | torch.Tensor = 1.0,
    v_scale: float | torch.Tensor = 1.0,
) -> None:
    """Quantize ``key``/``value`` ``[T, heads, head_size]`` into the cache.

    The store contract of the QSA owner (``do_kv_cache_update``):

    * ``slot = block * block_size + offset``; only the first
      ``slot_mapping.numel()`` rows of ``key`` and ``value`` are written (the
      forward pass pads them for CUDA graphs, the slot mapping is not padded);
    * a negative slot (padding, or a token another DCP rank owns) is skipped, and
      so is a slot past ``num_blocks * block_size`` (upstream's SM100 store does
      not bound-check it);
    * if a slot repeats the *last* token wins, which is how an accepted row
      replaces a rejected draft row. A real kernel leaves repeated slots
      unordered; the engine never repeats a slot inside one call.
    """
    (k_data, v_data), (k_scales, v_scales) = _cache_views(kv_cache)
    block_size = kv_cache.shape[2]
    capacity = kv_cache.shape[0] * block_size
    num_tokens = slot_mapping.numel()
    key, value = key[:num_tokens], value[:num_tokens]

    live = (slot_mapping >= 0) & (slot_mapping < capacity)
    order = torch.arange(num_tokens, device=slot_mapping.device)
    # For each distinct slot keep the largest token index (the last writer).
    last = torch.full((capacity,), -1, dtype=torch.long, device=order.device)
    last.scatter_reduce_(0, slot_mapping[live].long(), order[live], "amax")
    winners = live & (last[slot_mapping.clamp(0, capacity - 1).long()] == order)
    slots = slot_mapping[winners].long()
    block, offset = slots // block_size, slots % block_size

    for side, (data, scales), layer_scale in (
        (key[winners], (k_data, k_scales), k_scale),
        (value[winners], (v_data, v_scales), v_scale),
    ):
        packed, group_scale = quantize_kv_nvfp4(side, layer_scale=layer_scale)
        data[block, offset] = packed
        scales[block, offset] = group_scale.view(torch.uint8)


def gather_dequant_nvfp4_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_indices: torch.Tensor,
    *,
    k_scale: float | torch.Tensor = 1.0,
    v_scale: float | torch.Tensor = 1.0,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sparse gather: top-k logical tokens -> dequantized K and V.

    Mirrors the validity rule of the QSA sparse kernel
    (``_qsa_sparse_paged_gqa_splitk_kernel``): an entry is read only if its
    request is in range, its logical token is non-negative, its logical page is
    inside the block table and the mapped physical block is in range. Every
    other entry is **zero-filled** in both outputs (exact +0.0) without reading
    the cache, so stale or poisoned bytes (an E4M3 NaN scale, say) cannot leak.

    Shapes: ``logical_indices`` ``[rows, topk]`` int, ``token_to_req``
    ``[rows]`` int, ``block_table`` ``[requests, pages]`` int. Returns
    ``(keys, values)``, each ``[rows, topk, heads, head_size]``.
    """
    (k_data, v_data), (k_scales, v_scales) = _cache_views(kv_cache)
    num_blocks, _, block_size, heads, _ = kv_cache.shape
    num_requests, table_width = block_table.shape

    index = logical_indices.long()
    request = token_to_req.long().unsqueeze(1)
    page = index.div(block_size, rounding_mode="floor")

    valid = (index >= 0) & (request >= 0) & (request < num_requests)
    valid &= page < table_width
    physical = block_table.long()[
        request.clamp(0, num_requests - 1), page.clamp(0, table_width - 1)
    ]
    valid &= (physical >= 0) & (physical < num_blocks)

    safe_block = physical.clamp(0, num_blocks - 1)
    offset = index.clamp_min(0) % block_size
    keep = valid[..., None, None]

    outputs = []
    for data, scales, layer_scale in (
        (k_data, k_scales, k_scale),
        (v_data, v_scales, v_scale),
    ):
        dequantized = dequantize_kv_nvfp4(
            data[safe_block, offset],
            scales[safe_block, offset],
            layer_scale=layer_scale,
            out_dtype=out_dtype,
        )
        outputs.append(torch.where(keep, dequantized, torch.zeros_like(dequantized)))
    assert outputs[0].shape[2] == heads
    return outputs[0], outputs[1]
