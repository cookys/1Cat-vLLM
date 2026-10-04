# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton fused unpack + scale gather equals the pure-torch reference.

Without a GPU, run with ``TRITON_INTERPRET=1`` (the kernel then executes on the
CPU interpreter); with a GPU the same tests compile and run it for real::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \\
        python -m pytest --noconftest tests/models/qwen4_exp/test_nvfp4_kv_triton.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs TRITON_INTERPRET=1 (CPU interpreter) or a CUDA GPU",
)

if not torch.cuda.is_available():
    # Importing the QSA backend asks the GPU for a FlashAttention version (see
    # test_nvfp4_kv_admission.py); Volta runs FA2.
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton import (  # noqa: E402
    gather_dequant_nvfp4_kv_triton as triton_gather,
)

BLOCK = 16
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def _cache(
    num_blocks=6, tokens=50, heads=2, head_size=256, seed=0, k_scale=1.0, v_scale=1.0
):
    generator = torch.Generator().manual_seed(seed)
    row = nv.nvfp4_kv_row_bytes(head_size)
    cache = torch.zeros((num_blocks, 2, BLOCK, heads, row), dtype=torch.uint8)
    slots = torch.randperm(num_blocks * BLOCK, generator=generator)[:tokens]
    key = torch.randn((tokens, heads, head_size), generator=generator).half()
    value = torch.randn((tokens, heads, head_size), generator=generator).half()
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, slots, k_scale=k_scale, v_scale=v_scale
    )
    return cache


def _both(cache, table, requests, indices, **kwargs):
    reference = nv.gather_dequant_nvfp4_kv(cache, table, requests, indices, **kwargs)
    actual = triton_gather(
        cache.to(DEVICE),
        table.to(DEVICE),
        requests.to(DEVICE),
        indices.to(DEVICE),
        **kwargs,
    )
    return reference, tuple(x.cpu() for x in actual)


def _single(num_blocks):
    return torch.arange(num_blocks, dtype=torch.int32).view(1, -1)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_the_kernel_matches_the_reference_bit_for_bit() -> None:
    cache = _cache()
    generator = torch.Generator().manual_seed(1)
    indices = torch.randint(0, 6 * BLOCK, (3, 20), generator=generator).int()
    reference, actual = _both(
        cache, _single(6), torch.zeros(3, dtype=torch.int32), indices
    )
    for want, got in zip(reference, actual):
        assert _same_bits(want, got)


@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_the_kernel_handles_other_head_sizes(head_size) -> None:
    cache = _cache(head_size=head_size, tokens=30, heads=1, seed=2)
    indices = torch.arange(0, 30, dtype=torch.int32).view(1, -1)
    reference, actual = _both(
        cache, _single(6), torch.zeros(1, dtype=torch.int32), indices
    )
    for want, got in zip(reference, actual):
        assert got.shape == (1, 30, 1, head_size) and _same_bits(want, got)


def test_the_kernel_applies_the_layer_scales() -> None:
    cache = _cache(k_scale=0.5, v_scale=2.0, seed=3)
    indices = torch.arange(0, 40, dtype=torch.int32).view(1, -1)
    reference, actual = _both(
        cache,
        _single(6),
        torch.zeros(1, dtype=torch.int32),
        indices,
        k_scale=0.5,
        v_scale=2.0,
    )
    for want, got in zip(reference, actual):
        assert _same_bits(want, got)
    wrong = triton_gather(
        cache.to(DEVICE),
        _single(6).to(DEVICE),
        torch.zeros(1, dtype=torch.int32).to(DEVICE),
        indices.to(DEVICE),
    )[0].cpu()
    assert not _same_bits(reference[0], wrong)


def test_the_kernel_follows_a_permuted_block_table() -> None:
    cache = _cache(seed=4)
    table = torch.tensor([[3, 0, 5, 1, 4, 2]], dtype=torch.int32)
    indices = torch.arange(6 * BLOCK, dtype=torch.int32).view(1, -1)
    reference, actual = _both(cache, table, torch.zeros(1, dtype=torch.int32), indices)
    for want, got in zip(reference, actual):
        assert _same_bits(want, got)


def _assert_positive_zero(x: torch.Tensor) -> None:
    assert not x.any() and not torch.signbit(x).any()


@pytest.mark.parametrize(
    ("indices", "table", "request_id"),
    [
        ([[-1, -7, -(2**30)]], [[0, 1, 2, 3]], 0),  # negative logical tokens
        ([[4 * BLOCK, 5 * BLOCK + 3, 10**6]], [[0, 1, 2, 3]], 0),  # past the table
        ([[0, 3, BLOCK - 1]], [[-1, 1, 2, 3]], 0),  # unmapped block
        ([[0, 3, BLOCK - 1]], [[6, 1, 2, 3]], 0),  # block out of range
        ([[0, 3, BLOCK - 1]], [[0, 1, 2, 3]], -1),  # negative request
        ([[0, 3, BLOCK - 1]], [[0, 1, 2, 3]], 4),  # request past the table
    ],
)
def test_the_kernel_zero_fills_every_illegal_entry(indices, table, request_id) -> None:
    cache = _cache(seed=5)
    reference, actual = _both(
        cache,
        torch.tensor(table, dtype=torch.int32),
        torch.tensor([request_id], dtype=torch.int32),
        torch.tensor(indices, dtype=torch.int32),
    )
    for want, got in zip(reference, actual):
        _assert_positive_zero(want)
        _assert_positive_zero(got)


def test_the_kernel_zero_fills_interleaved_invalid_entries_only() -> None:
    cache = _cache(seed=6)
    indices = torch.tensor(
        [[3, -1, 17, 10**6, 40, -5, 41, 5, -1, 60]], dtype=torch.int32
    )
    reference, actual = _both(
        cache, _single(6), torch.zeros(1, dtype=torch.int32), indices
    )
    for want, got in zip(reference, actual):
        assert _same_bits(want, got)
        _assert_positive_zero(got[0, [1, 3, 5, 8]])


def test_the_kernel_never_reads_the_poisoned_bytes_of_invalid_entries() -> None:
    cache = torch.full((4, 2, BLOCK, 1, 144), 0xFF, dtype=torch.uint8)
    key = torch.randn((2, 1, 256), generator=torch.Generator().manual_seed(7)).half()
    nv.reshape_and_cache_nvfp4_reference(key, key, cache, torch.tensor([0, 1]))
    indices = torch.tensor([[0, -1, 10**6, 1]], dtype=torch.int32)
    _, actual = _both(cache, _single(4), torch.zeros(1, dtype=torch.int32), indices)
    for got in actual:
        assert torch.isfinite(got[0, [0, 3]]).all()
        _assert_positive_zero(got[0, [1, 2]])


def test_every_scale_code_and_every_nibble_decode_like_the_reference() -> None:
    # One token whose 16 scale bytes sweep all 256 E4M3 codes over 16 rows, and
    # whose data bytes sweep all 256 byte values.
    cache = torch.zeros((16, 2, BLOCK, 1, 144), dtype=torch.uint8)
    (k_data, v_data), (k_scale, v_scale) = nv.nvfp4_kv_split_views(cache)
    codes = torch.arange(256, dtype=torch.uint8)
    for block in range(16):
        k_scale[block, 0, 0] = codes[block * 16 : (block + 1) * 16]
        v_scale[block, 0, 0] = codes.flip(0)[block * 16 : (block + 1) * 16]
        k_data[block, 0, 0] = (torch.arange(128, dtype=torch.int64) * 7 + block).to(
            torch.uint8
        )
        v_data[block, 0, 0] = (torch.arange(128, dtype=torch.int64) * 5 + block).to(
            torch.uint8
        )
    indices = (torch.arange(16, dtype=torch.int32) * BLOCK).view(1, -1)
    reference, actual = _both(
        cache, _single(16), torch.zeros(1, dtype=torch.int32), indices
    )
    for want, got in zip(reference, actual):
        nan = torch.isnan(want)
        assert torch.equal(nan, torch.isnan(got))
        assert _same_bits(want.masked_fill(nan, 0), got.masked_fill(nan, 0))
        assert nan.any()  # the sweep includes the NaN codes 0x7F and 0xFF


def test_the_kernel_reads_a_strided_packed_member_view(monkeypatch) -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.v1.attention.backends import flash_attn
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
    from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
    from vllm.v1.worker.utils import AttentionGroup

    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    spec = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    members = ["a", "b"]
    raw = torch.zeros(4 * 2 * spec.page_size_bytes, dtype=torch.int8)
    views = _reshape_kv_cache(
        attn_groups=[
            AttentionGroup(Qwen4ExpQSAFlashAttentionBackend, members, spec, 0)
        ],
        kv_cache_raw_tensors={name: raw for name in members},
        cache_dtype="nvfp4",
        kernel_block_sizes=[BLOCK],
        shared_kv_cache_layers={},
        packed_members={name: (i, 2) for i, name in enumerate(members)},
    )
    second = views["b"]
    key = torch.randn((30, 1, 256), generator=torch.Generator().manual_seed(8)).half()
    slots = torch.randperm(4 * BLOCK, generator=torch.Generator().manual_seed(9))[:30]
    nv.reshape_and_cache_nvfp4_reference(key, key, second, slots)
    indices = slots.int().view(1, -1)
    reference = nv.gather_dequant_nvfp4_kv(
        second, _single(4), torch.zeros(1, dtype=torch.int32), indices
    )
    actual = triton_gather(
        second.to(DEVICE),
        _single(4).to(DEVICE),
        torch.zeros(1, dtype=torch.int32).to(DEVICE),
        indices.to(DEVICE),
    )
    for want, got in zip(reference, actual):
        assert _same_bits(want, got.cpu())


def test_the_kernel_can_write_float32() -> None:
    cache = _cache(seed=10)
    indices = torch.arange(0, 30, dtype=torch.int32).view(1, -1)
    reference, actual = _both(
        cache,
        _single(6),
        torch.zeros(1, dtype=torch.int32),
        indices,
        out_dtype=torch.float32,
    )
    for want, got in zip(reference, actual):
        assert got.dtype == torch.float32 and torch.equal(want, got)


def test_the_kernel_returns_empty_outputs_for_no_rows() -> None:
    cache = _cache(seed=11)
    keys, values = triton_gather(
        cache.to(DEVICE),
        _single(6).to(DEVICE),
        torch.zeros(0, dtype=torch.int32).to(DEVICE),
        torch.zeros((0, 8), dtype=torch.int32).to(DEVICE),
    )
    assert keys.shape == values.shape == (0, 8, 2, 256)


def test_the_kernel_rejects_non_int32_metadata() -> None:
    cache = _cache(seed=12)
    with pytest.raises(TypeError, match="int32"):
        triton_gather(
            cache.to(DEVICE),
            _single(6).long().to(DEVICE),
            torch.zeros(1, dtype=torch.int32).to(DEVICE),
            torch.zeros((1, 4), dtype=torch.int32).to(DEVICE),
        )
