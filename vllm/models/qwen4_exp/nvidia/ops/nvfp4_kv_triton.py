# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused unpack + scale sparse gather for the NVFP4 QSA main K/V cache.

First kernel of plan 071: it reads the packed cache pages and writes the
dequantized K and V rows of the selected top-k tokens, with the same validity
and zero-fill rule as ``nvfp4_kv.gather_dequant_nvfp4_kv`` (the reference it is
tested against). It is the unpack-and-scale half of the later fused attention
kernel, kept separate so the decode arithmetic can be checked on its own.

Everything the kernel needs works on SM70: E2M1 magnitudes come from integer
arithmetic, E4M3 scale bytes from the software decoder used by the E4M3 QSA
path, and nothing uses a native FP4/FP8 type. The output is written even
positions then odd positions, so no register interleave is needed.

On a machine without a GPU, run it under ``TRITON_INTERPRET=1``.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import fp8_e4m3fn_bits_to_fp32
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv import (
    NVFP4_KV_GROUP_SIZE,
    nvfp4_kv_split_views,
)
from vllm.triton_utils import tl, triton

__all__ = ["gather_dequant_nvfp4_kv_triton"]


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
