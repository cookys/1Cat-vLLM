# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels of the NVFP4 QSA KV cache: store and fused unpack + scale gather.

Plan 071. Both are tested against the pure-torch reference in ``nvfp4_kv.py``:

* ``store_nvfp4_kv_triton`` quantizes, packs and writes the K and V rows of a
  forward pass into the cache pages (the NVFP4 counterpart of
  ``reshape_and_cache_flash``). It evaluates a group in the operation order of
  upstream's SM100 store, with a software E4M3 encoder and a software E2M1
  rounding, so nothing uses an FP4/FP8 instruction.
* ``gather_dequant_nvfp4_kv_triton`` reads the packed pages of the selected
  top-k tokens and writes dequantized K and V rows, with the validity and
  zero-fill rule of the QSA sparse kernel. It is the unpack-and-scale half of
  the later fused attention kernel, kept on its own so the decode arithmetic can
  be checked in isolation.

Everything works on SM70: E2M1 magnitudes come from integer arithmetic, E4M3
scale bytes from the software decoder of the E4M3 QSA path and from an integer
encoder, and the stores are even then odd positions, so no register interleave
is needed. On a machine without a GPU, run it under ``TRITON_INTERPRET=1``.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import fp8_e4m3fn_bits_to_fp32
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv import (
    NVFP4_KV_GROUP_SIZE,
    nvfp4_kv_split_views,
)
from vllm.triton_utils import tl, triton

__all__ = ["gather_dequant_nvfp4_kv_triton", "store_nvfp4_kv_triton"]


@triton.jit
def _e2m1_magnitude(code):
    """3-bit E2M1 magnitude code -> 0, .5, 1, 1.5, 2, 3, 4, 6 as float32."""
    exponent = code >> 1
    mantissa = code & 1
    subnormal = mantissa.to(tl.float32) * 0.5
    normal = (1.0 + mantissa.to(tl.float32) * 0.5) * tl.exp2(
        exponent.to(tl.float32) - 1.0
    )
    return tl.where(exponent == 0, subnormal, normal)


@triton.jit
def _e2m1_value(nibble):
    # Multiply by +-1 instead of negating: IEEE gives -0.0 for the code 0x8
    # with either lowering, as the torch reference does.
    sign = tl.where((nibble & 0x8) != 0, -1.0, 1.0)
    return _e2m1_magnitude(nibble & 0x7) * sign


@triton.jit
def _nvfp4_row(
    data_ptr,
    scale_ptr,
    block,
    token,
    head,
    layer_scale,
    valid,
    stride_data_block,
    stride_data_token,
    stride_data_head,
    stride_scale_block,
    stride_scale_token,
    stride_scale_head,
    HALF: tl.constexpr,
):
    """Even and odd dequantized values of one K or V row (``HALF`` bytes)."""
    byte = tl.arange(0, HALF)
    data_base = (
        data_ptr
        + block * stride_data_block
        + token * stride_data_token
        + head * stride_data_head
    )
    scale_base = (
        scale_ptr
        + block * stride_scale_block
        + token * stride_scale_token
        + head * stride_scale_head
    )
    packed = tl.load(data_base + byte, mask=valid, other=0).to(tl.int32)
    # Eight packed bytes (16 values) share one scale byte.
    scale_bits = tl.load(scale_base + byte // 8, mask=valid, other=0)
    scale = fp8_e4m3fn_bits_to_fp32(scale_bits) * layer_scale
    even = _e2m1_value(packed & 0xF) * scale
    odd = _e2m1_value(packed >> 4) * scale
    return even, odd


@triton.jit
def _gather_dequant_nvfp4_kernel(
    k_data_ptr,
    k_scale_ptr,
    v_data_ptr,
    v_scale_ptr,
    block_table_ptr,
    token_to_req_ptr,
    indices_ptr,
    k_out_ptr,
    v_out_ptr,
    k_layer_scale,
    v_layer_scale,
    stride_data_block,
    stride_data_token,
    stride_data_head,
    stride_scale_block,
    stride_scale_token,
    stride_scale_head,
    stride_table_req,
    stride_indices_row,
    stride_out_row,
    stride_out_col,
    stride_out_head,
    num_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HALF: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col = tl.program_id(1)
    head = tl.program_id(2)

    request = tl.load(token_to_req_ptr + row)
    token = tl.load(indices_ptr + row * stride_indices_row + col)
    safe_token = tl.maximum(token, 0)
    page = safe_token // PAGE_SIZE
    offset = safe_token % PAGE_SIZE
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    valid = (request >= 0) & (request < num_requests) & (token >= 0)
    valid &= page < TABLE_WIDTH
    physical = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(page, TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical >= 0) & (physical < num_blocks)
    block = tl.maximum(physical, 0).to(tl.int64)

    k_even, k_odd = _nvfp4_row(
        k_data_ptr,
        k_scale_ptr,
        block,
        offset,
        head,
        k_layer_scale,
        valid,
        stride_data_block,
        stride_data_token,
        stride_data_head,
        stride_scale_block,
        stride_scale_token,
        stride_scale_head,
        HALF,
    )
    v_even, v_odd = _nvfp4_row(
        v_data_ptr,
        v_scale_ptr,
        block,
        offset,
        head,
        v_layer_scale,
        valid,
        stride_data_block,
        stride_data_token,
        stride_data_head,
        stride_scale_block,
        stride_scale_token,
        stride_scale_head,
        HALF,
    )

    byte = tl.arange(0, HALF)
    out = row * stride_out_row + col * stride_out_col + head * stride_out_head
    # Invalid entries read nothing (masked loads return zero data and a zero
    # scale), so these stores write exact +0.0.
    tl.store(k_out_ptr + out + 2 * byte, k_even.to(k_out_ptr.dtype.element_ty))
    tl.store(k_out_ptr + out + 2 * byte + 1, k_odd.to(k_out_ptr.dtype.element_ty))
    tl.store(v_out_ptr + out + 2 * byte, v_even.to(v_out_ptr.dtype.element_ty))
    tl.store(v_out_ptr + out + 2 * byte + 1, v_odd.to(v_out_ptr.dtype.element_ty))


def gather_dequant_nvfp4_kv_triton(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_indices: torch.Tensor,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton counterpart of ``nvfp4_kv.gather_dequant_nvfp4_kv``.

    ``kv_cache`` is the 5-D uint8 ``(blocks, 2, block_size, heads, row_bytes)``
    cache; ``block_table`` and ``logical_indices`` are int32. Returns
    ``(keys, values)`` of shape ``[rows, topk, heads, head_size]``.
    """
    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv_split_views(kv_cache)
    num_blocks, _, block_size, heads, _ = kv_cache.shape
    head_size = k_data.shape[-1] * 2
    if head_size % NVFP4_KV_GROUP_SIZE:
        raise ValueError("NVFP4 KV head size must be a multiple of 16")
    if block_table.dtype != torch.int32 or logical_indices.dtype != torch.int32:
        raise TypeError("block table and logical indices must be int32")
    if token_to_req.dtype != torch.int32:
        raise TypeError("token_to_req must be int32")
    if (
        block_table.stride(1) != 1
        or logical_indices.stride(1) != 1
        or token_to_req.stride(0) != 1
    ):
        raise ValueError("block table, indices and token_to_req need unit inner stride")
    if k_data.stride(3) != 1 or k_scales.stride(3) != 1:
        raise ValueError("cache rows must be contiguous in their last dimension")

    rows, topk = logical_indices.shape
    keys = torch.empty(
        (rows, topk, heads, head_size), dtype=out_dtype, device=kv_cache.device
    )
    values = torch.empty_like(keys)
    if keys.numel() == 0:
        return keys, values
    _gather_dequant_nvfp4_kernel[(rows, topk, heads)](
        k_data,
        k_scales,
        v_data,
        v_scales,
        block_table,
        token_to_req,
        logical_indices,
        keys,
        values,
        float(k_scale),
        float(v_scale),
        k_data.stride(0),
        k_data.stride(1),
        k_data.stride(2),
        k_scales.stride(0),
        k_scales.stride(1),
        k_scales.stride(2),
        block_table.stride(0),
        logical_indices.stride(0),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        num_blocks,
        block_table.shape[0],
        PAGE_SIZE=block_size,
        TABLE_WIDTH=block_table.shape[1],
        HALF=head_size // 2,
        num_warps=2,
    )
    return keys, values


@triton.jit
def _e4m3_encode_nonneg(value):
    """Non-negative fp32 -> E4M3FN byte (int32), round-to-nearest-even, saturating.

    Integer arithmetic on the fp32 bits for the normal range, a rounded multiple
    of 2**-9 below 2**-6. Matches ``nvfp4_kv.e4m3_encode`` for finite input.
    """
    v = tl.minimum(value, 448.0)
    bits = v.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    top = mantissa >> 20  # the three kept mantissa bits
    rest = mantissa & 0xFFFFF
    round_up = (rest > 0x80000) | ((rest == 0x80000) & ((top & 1) == 1))
    top = top + round_up.to(tl.int32)
    carry = top >> 3  # the mantissa rounded up to 1.0: bump the exponent
    normal = ((exponent - 127 + carry + 7) << 3) | (top & 7)
    scaled = v * 512.0  # subnormals are multiples of 2**-9
    floor = tl.floor(scaled)
    fraction = scaled - floor
    floor_int = floor.to(tl.int32)
    subnormal = (
        floor_int
        + (fraction > 0.5).to(tl.int32)
        + ((fraction == 0.5) & ((floor_int & 1) == 1)).to(tl.int32)
    )
    code = tl.where(v < 0.015625, subnormal, normal)
    return tl.where(v > 0.0, code, 0)


@triton.jit
def _e2m1_code(scaled):
    """fp32 -> 4-bit E2M1 code, round-to-nearest-even, saturating; no -0."""
    magnitude = tl.abs(scaled)
    code = (
        (magnitude > 0.25).to(tl.int32)
        + (magnitude >= 0.75).to(tl.int32)
        + (magnitude > 1.25).to(tl.int32)
        + (magnitude >= 1.75).to(tl.int32)
        + (magnitude > 2.5).to(tl.int32)
        + (magnitude >= 3.5).to(tl.int32)
        + (magnitude > 5.0).to(tl.int32)
    )
    negative = (scaled < 0.0) & (code != 0)
    return code | (negative.to(tl.int32) << 3)


@triton.jit
def _store_nvfp4_kernel(
    x_ptr,
    slots_ptr,
    data_ptr,
    scale_ptr,
    layer_scale_ptr,
    stride_x_token,
    stride_x_head,
    stride_data_block,
    stride_data_token,
    stride_data_head,
    stride_scale_block,
    stride_scale_token,
    stride_scale_head,
    capacity,
    PAGE_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots_ptr + token)
    valid = (slot >= 0) & (slot < capacity)
    safe_slot = tl.maximum(slot, 0)
    block = (safe_slot // PAGE_SIZE).to(tl.int64)
    offset = safe_slot % PAGE_SIZE

    group = tl.arange(0, GROUPS)[:, None]
    pair = tl.arange(0, 8)[None, :]
    row = x_ptr + token * stride_x_token + head * stride_x_head
    # Element 16 * group + 2 * pair is the even one, the next is the odd one.
    even = tl.load(row + group * 16 + pair * 2, mask=valid, other=0.0).to(tl.float32)
    odd = tl.load(row + group * 16 + pair * 2 + 1, mask=valid, other=0.0).to(tl.float32)

    # The SM100 store's order: SFScaleVal = 1 / k_scale;
    # SFValue = SFScaleVal * (vecMax * (1 / 6)); outputScale = 1 / (SFValue_q * k).
    layer_scale = tl.load(layer_scale_ptr)
    scale_inverse = 1.0 / layer_scale
    amax = tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), axis=1)
    block_code = _e4m3_encode_nonneg(scale_inverse * (amax * 0.16666667163372040))
    block_value = fp8_e4m3fn_bits_to_fp32(block_code)
    out_scale = tl.where(
        block_value > 0.0, 1.0 / (block_value * (1.0 / scale_inverse)), 0.0
    )[:, None]
    low = _e2m1_code(even * out_scale)  # the even element is the low nibble
    high = _e2m1_code(odd * out_scale)
    packed = (low | (high << 4)).to(tl.uint8)

    data = (
        data_ptr
        + block * stride_data_block
        + offset * stride_data_token
        + head * stride_data_head
    )
    scale = (
        scale_ptr
        + block * stride_scale_block
        + offset * stride_scale_token
        + head * stride_scale_head
    )
    tl.store(data + group * 8 + pair, packed, mask=valid)
    tl.store(scale + tl.arange(0, GROUPS), block_code.to(tl.uint8), mask=valid)


def _layer_scale_tensor(scale, device: torch.device) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise TypeError("a layer scale tensor must be one float32 element")
        return scale.reshape(1)
    # A host float builds a device tensor now, so it is not CUDA-graph safe:
    # capture with the layer's own device scalar (``layer._k_scale``).
    return torch.tensor([float(scale)], dtype=torch.float32, device=device)


def _store_side(x, slots, data, scales, layer_scale, capacity, block_size) -> None:
    heads, head_size = x.shape[1], x.shape[2]
    _store_nvfp4_kernel[(slots.numel(), heads)](
        x,
        slots,
        data,
        scales,
        layer_scale,
        x.stride(0),
        x.stride(1),
        data.stride(0),
        data.stride(1),
        data.stride(2),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        capacity,
        PAGE_SIZE=block_size,
        GROUPS=head_size // NVFP4_KV_GROUP_SIZE,
        num_warps=1,
    )


def store_nvfp4_kv_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale: float | torch.Tensor = 1.0,
    v_scale: float | torch.Tensor = 1.0,
) -> None:
    """Triton counterpart of ``nvfp4_kv.reshape_and_cache_nvfp4_reference``.

    ``key`` and ``value`` are ``[rows, heads, head_size]`` fp16 or fp32 with a
    contiguous last dimension (the rows may be padded, and ``value`` is often a
    strided slice of the QKV projection); ``slot_mapping`` is int64
    ``[num_tokens]`` and only that many rows are written; ``kv_cache`` is the
    layer's 5-D uint8 cache (a packed member view is fine). ``k_scale`` and
    ``v_scale`` are the layer's device scalars (``layer._k_scale``) or floats.
    One program handles one (token, head); slots must not repeat.
    """
    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv_split_views(kv_cache)
    num_blocks, _, block_size, heads, _ = kv_cache.shape
    head_size = k_data.shape[-1] * 2
    if head_size % NVFP4_KV_GROUP_SIZE or (head_size // NVFP4_KV_GROUP_SIZE) & (
        head_size // NVFP4_KV_GROUP_SIZE - 1
    ):
        raise ValueError("NVFP4 KV head size must be 16 times a power of two")
    for name, x in (("key", key), ("value", value)):
        if x.ndim != 3 or x.shape[1:] != (heads, head_size):
            raise ValueError(f"{name} must be [rows, {heads}, {head_size}]")
        if x.stride(2) != 1:
            raise ValueError(f"{name} needs a contiguous last dimension")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError(f"{name} must be fp16 or fp32, got {x.dtype}")
        if x.shape[0] < slot_mapping.numel():
            raise ValueError(f"{name} has fewer rows than slots")
    if slot_mapping.dtype != torch.int64 or slot_mapping.stride(0) != 1:
        raise TypeError("slot_mapping must be a contiguous int64 vector")
    if k_data.stride(3) != 1 or k_scales.stride(3) != 1:
        raise ValueError("cache rows must be contiguous in their last dimension")
    if slot_mapping.numel() == 0:
        return
    capacity = num_blocks * block_size
    _store_side(
        key,
        slot_mapping,
        k_data,
        k_scales,
        _layer_scale_tensor(k_scale, key.device),
        capacity,
        block_size,
    )
    _store_side(
        value,
        slot_mapping,
        v_data,
        v_scales,
        _layer_scale_tensor(v_scale, value.device),
        capacity,
        block_size,
    )


@triton.jit
def _encode_probe_kernel(x_ptr, out_ptr, n, KIND: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < n
    x = tl.load(x_ptr + index, mask=mask, other=0.0)
    if KIND == 0:  # noqa: SIM108 - a statement `if` on a constexpr, safe in Triton
        code = _e4m3_encode_nonneg(x)
    else:
        code = _e2m1_code(x)
    tl.store(out_ptr + index, code, mask=mask)


def _probe_encoder(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Run one of the in-kernel software encoders on a vector (test support)."""
    x = x.contiguous().float()
    out = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    block = 256
    _encode_probe_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), KIND=0 if kind == "e4m3" else 1, BLOCK=block
    )
    return out
