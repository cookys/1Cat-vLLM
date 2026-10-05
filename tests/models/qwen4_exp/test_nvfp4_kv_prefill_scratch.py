# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan 071 option B': decode an NVFP4 prefix once into an FP16 scratch.

``dequant_nvfp4_prefix_triton`` writes every cached token of a request exactly once
into an FP16 scratch laid out like the FP16 paged cache the grouped page4 CUDA
route reads (``[pages, block_size, heads, head_size]``, K and V separate). The
decode must be bit for bit the gather dequant without the layer scale; tokens at
or past the sequence length, and unmapped pages, must be exact ``+0.0`` so a stale
or poisoned byte cannot reach the tensor-core P @ V product.

Without a GPU, run with ``TRITON_INTERPRET=1``::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python -m pytest \\
        --noconftest tests/models/qwen4_exp/test_nvfp4_kv_prefill_scratch.py
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
    # See test_nvfp4_kv_admission.py: importing the QSA backend asks the GPU for a
    # FlashAttention version.
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa_nvfp4  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton import (  # noqa: E402
    dequant_nvfp4_prefix_triton,
    gather_dequant_nvfp4_sides_triton,
)

# 36 = 9 microblocks of 4 and not a multiple of the 16-token decode tile, so the
# last tile of every page is partial.
BLOCK = 36
HEAD = 256
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def _cache(blocks=10, heads=1, seed=0, k_scale=1.0, v_scale=1.0, amplitude=2.0):
    generator = torch.Generator().manual_seed(seed)
    tokens = blocks * BLOCK
    key = (torch.randn(tokens, heads, HEAD, generator=generator) * amplitude).half()
    value = (torch.randn(tokens, heads, HEAD, generator=generator) * amplitude).half()
    cache = torch.zeros(
        (blocks, 2, BLOCK, heads, nv.nvfp4_kv_row_bytes(HEAD)), dtype=torch.uint8
    )
    slots = torch.randperm(tokens, generator=generator)
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, slots, k_scale=k_scale, v_scale=v_scale
    )
    return cache


def _table(rows):
    return torch.tensor(rows, dtype=torch.int32)


def _lens(*values):
    return torch.tensor(values, dtype=torch.int32)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _decode(
    cache,
    table,
    seq_lens,
    *,
    scratch=None,
    start_tokens=None,
    scratch_pages=None,
    table_hook=None,
):
    """Run the scratch decode; returns ``(scratch_k, scratch_v, compact, offsets)``."""
    heads = cache.shape[3]
    compact, offsets = qsa_nvfp4.nvfp4_prefix_scratch_table(
        table.to(DEVICE), seq_lens.to(DEVICE), BLOCK, scratch_pages or 10**6
    )
    if table_hook is not None:
        compact, offsets = table_hook(compact, offsets)
    pages = scratch_pages or int(compact.max().item()) + 1
    if scratch is None:
        scratch = (
            torch.full((pages, BLOCK, heads, HEAD), 7.0, dtype=torch.float16),
            torch.full((pages, BLOCK, heads, HEAD), 7.0, dtype=torch.float16),
        )
    scratch_k, scratch_v = (s.to(DEVICE) for s in scratch)
    max_pages = max(1, int(torch.ceil(seq_lens.float().max() / BLOCK).item()))
    dequant_nvfp4_prefix_triton(
        cache[:, 0].to(DEVICE),
        cache[:, 1].to(DEVICE),
        table.to(DEVICE),
        seq_lens.to(DEVICE),
        offsets,
        scratch_k,
        scratch_v,
        start_tokens=None if start_tokens is None else start_tokens.to(DEVICE),
        max_pages=max_pages,
    )
    return scratch_k.cpu(), scratch_v.cpu(), compact.cpu(), offsets.cpu()


def _reference(cache, table, request, seq_len):
    """Gather dequant of one request's tokens ``0..seq_len-1``, layer scale 1."""
    indices = torch.arange(seq_len, dtype=torch.int32).view(1, -1)
    keys, values = nv.gather_dequant_nvfp4_kv(
        cache,
        table,
        torch.tensor([request], dtype=torch.int32),
        indices,
        k_scale=1.0,
        v_scale=1.0,
    )
    return keys[0], values[0]


def _assert_scratch_equals_gather(cache, table, seq_lens, scratch_k, scratch_v, offs):
    """Every live token equals the gather bit for bit; the rest is exact +0.0."""
    for request in range(table.shape[0]):
        seq_len = int(seq_lens[request])
        if seq_len == 0:
            continue
        pages = -(-seq_len // BLOCK)
        want_k, want_v = _reference(cache, table, request, seq_len)
        base = int(offs[request])
        got_k = scratch_k[base : base + pages].reshape(pages * BLOCK, -1, HEAD)
        got_v = scratch_v[base : base + pages].reshape(pages * BLOCK, -1, HEAD)
        assert torch.equal(_bits(got_k[:seq_len]), _bits(want_k))
        assert torch.equal(_bits(got_v[:seq_len]), _bits(want_v))
        assert not _bits(got_k[seq_len:]).any()
        assert not _bits(got_v[seq_len:]).any()


# --------------------------------------------------------------------------- table


def test_scratch_table_matches_a_hand_computed_layout() -> None:
    table = _table([[3, 1, 4, 0], [2, 5, 6, 7], [9, 8, 0, 0], [0, 0, 0, 0]])
    seq_lens = _lens(2 * BLOCK + 1, BLOCK, 0, 3 * BLOCK)
    compact, offsets = qsa_nvfp4.nvfp4_prefix_scratch_table(table, seq_lens, BLOCK, 100)
    # pages per request: 3, 1, 0, 3 -> exclusive cumsum 0, 3, 4, 4
    assert offsets.tolist() == [0, 3, 4, 4]
    assert compact.tolist() == [
        [0, 1, 2, -1],
        [3, -1, -1, -1],
        [-1, -1, -1, -1],
        [4, 5, 6, -1],
    ]
    assert compact.dtype == offsets.dtype == torch.int32
    assert compact.stride(1) == 1


def test_scratch_table_clamps_to_the_block_table_width() -> None:
    table = _table([[0, 1]])
    compact, offsets = qsa_nvfp4.nvfp4_prefix_scratch_table(
        table, _lens(5 * BLOCK), BLOCK, 100
    )
    assert compact.tolist() == [[0, 1]] and offsets.tolist() == [0]


def test_scratch_table_pages_exact_multiple_and_single_token() -> None:
    table = _table([[0, 1, 2], [3, 4, 5]])
    compact, offsets = qsa_nvfp4.nvfp4_prefix_scratch_table(
        table, _lens(2 * BLOCK, 1), BLOCK, 100
    )
    assert offsets.tolist() == [0, 2]
    assert compact.tolist() == [[0, 1, -1], [2, -1, -1]]


# --------------------------------------------------------------------------- decode


@pytest.mark.parametrize(
    ("rows", "lens"),
    [
        ([[3, 1, 4, 0]], (3 * BLOCK - 5,)),  # one request, partial last page
        ([[0, 1, 2, 3]], (BLOCK,)),  # exactly one full page
        ([[2, 0, 4, 1]], (1,)),  # a single token
        ([[5, 3, 9, 7], [1, 8, 2, 6]], (2 * BLOCK + 7, 3 * BLOCK)),  # two requests
        ([[5, 3, 9, 7], [1, 8, 2, 6], [0, 4, 0, 0]], (BLOCK + 2, 0, 2 * BLOCK + 3)),
    ],
)
def test_scratch_decode_equals_the_gather_bit_for_bit(rows, lens) -> None:
    cache = _cache(seed=1)
    table, seq_lens = _table(rows), _lens(*lens)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens)
    _assert_scratch_equals_gather(cache, table, seq_lens, scratch_k, scratch_v, offsets)


def test_scratch_decode_equals_the_triton_gather_kernel_too() -> None:
    cache = _cache(seed=2)
    table, seq_lens = _table([[4, 2, 6, 0]]), _lens(3 * BLOCK - 1)
    scratch_k, scratch_v, _, _ = _decode(cache, table, seq_lens)
    indices = torch.arange(int(seq_lens[0]), dtype=torch.int32).view(1, -1)
    keys, values = gather_dequant_nvfp4_sides_triton(
        cache[:, 0].to(DEVICE),
        cache[:, 1].to(DEVICE),
        table.to(DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        indices.to(DEVICE),
    )
    n = int(seq_lens[0])
    got_k = scratch_k[:3].reshape(-1, 1, HEAD)[:n]
    got_v = scratch_v[:3].reshape(-1, 1, HEAD)[:n]
    assert torch.equal(_bits(got_k), _bits(keys[0].cpu()))
    assert torch.equal(_bits(got_v), _bits(values[0].cpu()))


def test_the_layer_scale_is_not_part_of_the_decode() -> None:
    # The cache is written with layer scales != 1. The scratch must hold the values
    # a gather with scale 1 returns (the attention folds the scales), not the real
    # K/V a gather with the layer scale returns.
    cache = _cache(seed=3, k_scale=0.0213623046875, v_scale=0.0404924675822258)
    table, seq_lens = _table([[1, 2, 3, 4]]), _lens(2 * BLOCK + 11)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens)
    _assert_scratch_equals_gather(cache, table, seq_lens, scratch_k, scratch_v, offsets)
    n = int(seq_lens[0])
    scaled, _ = nv.gather_dequant_nvfp4_kv(
        cache,
        table,
        torch.zeros(1, dtype=torch.int32),
        torch.arange(n, dtype=torch.int32).view(1, -1),
        k_scale=0.0213623046875,
        v_scale=0.0404924675822258,
    )
    got_k = scratch_k[:3].reshape(-1, 1, HEAD)[:n]
    assert not torch.equal(_bits(got_k), _bits(scaled[0]))


def test_multi_head_caches_decode_every_head() -> None:
    cache = _cache(seed=4, heads=2)
    table, seq_lens = _table([[2, 0, 5, 1]]), _lens(2 * BLOCK + 3)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens)
    _assert_scratch_equals_gather(cache, table, seq_lens, scratch_k, scratch_v, offsets)


def test_a_block_table_with_unmapped_or_out_of_range_blocks_decodes_to_zero() -> None:
    cache = _cache(seed=5)
    table = _table([[3, -1, 4, 99]])
    seq_lens = _lens(4 * BLOCK)
    scratch_k, scratch_v, _, _ = _decode(cache, table, seq_lens)
    live_k, live_v = _reference(cache, table.clamp(0, 9), 0, 4 * BLOCK)
    flat_k = scratch_k[:4].reshape(-1, 1, HEAD)
    flat_v = scratch_v[:4].reshape(-1, 1, HEAD)
    for page in (0, 2):
        span = slice(page * BLOCK, (page + 1) * BLOCK)
        assert torch.equal(_bits(flat_k[span]), _bits(live_k[span]))
        assert torch.equal(_bits(flat_v[span]), _bits(live_v[span]))
    for page in (1, 3):
        span = slice(page * BLOCK, (page + 1) * BLOCK)
        assert not _bits(flat_k[span]).any() and not _bits(flat_v[span]).any()


def test_requests_that_share_physical_pages_each_get_a_correct_copy() -> None:
    cache = _cache(seed=6)
    # Prefix sharing: both requests start with physical blocks 2 and 7.
    table = _table([[2, 7, 1, 0], [2, 7, 5, 0]])
    seq_lens = _lens(2 * BLOCK + 9, 2 * BLOCK + 20)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens)
    assert offsets.tolist() == [0, 3]
    _assert_scratch_equals_gather(cache, table, seq_lens, scratch_k, scratch_v, offsets)


def _poison(cache, table, seq_lens):
    """Overwrite every slot at or past the sequence length with hostile bytes.

    The logical ``(blocks, 2, block_size, heads, 144)`` shape of the cache is not its
    byte layout (a page is ``[K_data | K_scale | V_data | V_scale]``), so the slots
    are reached through the data and scale views.
    """
    poisoned = cache.clone()
    (k_data, v_data), (k_scale, v_scale) = nv.nvfp4_kv_split_views(poisoned)
    generator = torch.Generator().manual_seed(99)
    for request in range(table.shape[0]):
        for token in range(int(seq_lens[request]), table.shape[1] * BLOCK):
            block = int(table[request, token // BLOCK])
            if block < 0:
                continue
            offset = token % BLOCK
            for data, scale in ((k_data, k_scale), (v_data, v_scale)):
                data[block, offset] = torch.randint(
                    0, 256, data[block, offset].shape, generator=generator
                ).to(torch.uint8)
                scale[block, offset] = 0x7F if token % 2 else 0xFF  # E4M3 NaN
    return poisoned


def test_bytes_past_the_sequence_length_cannot_reach_the_scratch() -> None:
    cache = _cache(seed=7)
    table, seq_lens = _table([[6, 2, 8, 1]]), _lens(2 * BLOCK + 5)
    clean = _decode(cache, table, seq_lens)
    dirty = _decode(_poison(cache, table, seq_lens), table, seq_lens)
    assert torch.equal(_bits(clean[0]), _bits(dirty[0]))
    assert torch.equal(_bits(clean[1]), _bits(dirty[1]))
    assert torch.isfinite(dirty[0]).all() and torch.isfinite(dirty[1]).all()
    # The tail of the last page is the exact +0.0 the NaN-free P @ V needs.
    tail = dirty[0][2, 5:]
    assert not _bits(tail).any() and not torch.signbit(tail).any()


def test_a_page_is_never_written_outside_its_scratch_slot() -> None:
    cache = _cache(seed=8)
    table, seq_lens = _table([[0, 1, 2, 3], [4, 5, 6, 7]]), _lens(BLOCK + 3, BLOCK)
    sentinel = (
        torch.full((6, BLOCK, 1, HEAD), 7.0, dtype=torch.float16),
        torch.full((6, BLOCK, 1, HEAD), 7.0, dtype=torch.float16),
    )
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens, scratch=sentinel)
    assert offsets.tolist() == [0, 2]
    # 3 pages are live; pages 3..5 keep the sentinel.
    assert (scratch_k[3:] == 7.0).all() and (scratch_v[3:] == 7.0).all()


def test_a_scratch_that_is_too_small_is_never_written_past_its_end() -> None:
    cache = _cache(seed=9)
    table, seq_lens = _table([[0, 1, 2, 3]]), _lens(3 * BLOCK)
    sentinel = tuple(
        torch.full((2, BLOCK, 1, HEAD), 7.0, dtype=torch.float16) for _ in "kv"
    )
    scratch_k, scratch_v, _, _ = _decode(
        cache, table, seq_lens, scratch=sentinel, scratch_pages=2
    )
    want_k, _ = _reference(cache, table, 0, 2 * BLOCK)
    assert torch.equal(_bits(scratch_k.reshape(-1, 1, HEAD)), _bits(want_k))


# ----------------------------------------------------------------------- watermark


def test_an_incremental_decode_equals_a_full_decode() -> None:
    cache = _cache(seed=10)
    table = _table([[3, 5, 1, 7]])
    first, second = _lens(BLOCK + 13), _lens(3 * BLOCK - 4)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, first, scratch_pages=3)
    # The next chunk appends tokens; only tokens from the old length on are new.
    incremental = _decode(
        cache,
        table,
        second,
        scratch=(scratch_k, scratch_v),
        start_tokens=first,
        scratch_pages=scratch_k.shape[0],
    )
    full = _decode(cache, table, second, scratch_pages=scratch_k.shape[0])
    assert torch.equal(_bits(incremental[0][:3]), _bits(full[0][:3]))
    assert torch.equal(_bits(incremental[1][:3]), _bits(full[1][:3]))
    _assert_scratch_equals_gather(
        cache, table, second, incremental[0], incremental[1], offsets
    )


def test_an_incremental_decode_really_skips_the_old_tokens() -> None:
    # Guards the guard: a start token that is ignored would pass the test above.
    cache = _cache(seed=11)
    table, seq_lens = _table([[0, 1, 2, 3]]), _lens(3 * BLOCK)
    sentinel = tuple(
        torch.full((3, BLOCK, 1, HEAD), 7.0, dtype=torch.float16) for _ in "kv"
    )
    scratch_k, _, _, _ = _decode(
        cache,
        table,
        seq_lens,
        scratch=sentinel,
        start_tokens=_lens(2 * BLOCK),
        scratch_pages=3,
    )
    assert (scratch_k[0] == 7.0).all() and (scratch_k[1] == 7.0).all()
    assert (scratch_k[2] != 7.0).any()


def test_a_start_inside_a_tile_redecodes_that_whole_tile_identically() -> None:
    cache = _cache(seed=12)
    table, seq_lens = _table([[2, 4, 6, 8]]), _lens(2 * BLOCK + 20)
    full = _decode(cache, table, seq_lens)
    redo = _decode(
        cache,
        table,
        seq_lens,
        scratch=(full[0].clone(), full[1].clone()),
        start_tokens=_lens(BLOCK + 17),
        scratch_pages=full[0].shape[0],
    )
    assert torch.equal(_bits(full[0]), _bits(redo[0]))
    assert torch.equal(_bits(full[1]), _bits(redo[1]))


# ------------------------------------------------------------- the watermark rule


def _tracker():
    return qsa_nvfp4.Nvfp4ScratchWatermark()


def test_the_watermark_never_grants_without_a_proof() -> None:
    mark = _tracker()
    layer = object()
    # Nothing recorded yet.
    assert mark.start_tokens(layer, ("a",), [0], [100]) == [0]
    # Unknown request identity: recorded state is not trusted, nothing is recorded.
    mark.record(layer, None, [100])
    assert mark.start_tokens(layer, None, [100], [200]) == [0]
    assert mark.start_tokens(layer, ("a",), [100], [200]) == [0]


def test_the_watermark_grants_a_contiguous_chunk_of_the_same_owner() -> None:
    mark, layer = _tracker(), object()
    mark.record(layer, ("a",), [BLOCK + 1])
    granted = mark.start_tokens(layer, ("a",), [BLOCK + 1], [3 * BLOCK], BLOCK)
    assert granted == [BLOCK + 1]


@pytest.mark.parametrize(
    "case",
    [
        "other layer wrote the scratch since",
        "other request",
        "query start is not the recorded length (cache hit or preempt)",
        "request set changed",
        "earlier request grew a page so the offsets moved",
    ],
)
def test_the_watermark_refuses_every_case_it_cannot_prove(case) -> None:
    mark, layer = _tracker(), object()
    mark.record(layer, ("a", "b"), [BLOCK, BLOCK])
    ids, starts, lens = ("a", "b"), [BLOCK, BLOCK], [2 * BLOCK, 2 * BLOCK]
    want = [0, 0]
    if case.startswith("other layer"):
        mark.record(object(), ("a", "b"), [BLOCK, BLOCK])
    elif case == "other request":
        ids = ("a", "c")
    elif case.startswith("query start"):
        # Request "b" restarts below its recorded length; "a" is still contiguous.
        starts, want = [BLOCK, BLOCK - 1], [BLOCK, 0]
    elif case == "request set changed":
        ids, starts, lens, want = ("a",), [BLOCK], [2 * BLOCK], [0]
    else:
        # "a" grows from 1 to 2 pages, which moves "b" from scratch page 1 to 2.
        want = [BLOCK, 0]
    assert mark.start_tokens(layer, ids, starts, lens, BLOCK) == want


# --------------------------------------------------------------------- mutations


def _swap_nibbles(cache):
    out = cache.clone()
    (k_data, v_data), _ = nv.nvfp4_kv_split_views(out)
    for data in (k_data, v_data):
        data.copy_(((data & 0x0F) << 4) | (data >> 4))
    return out


def _scale_placement_wrong(cache):
    """The scale of group g is read from slot g // 2 (what a //16 for //8 would do)."""
    out = cache.clone()
    _, (k_scale, v_scale) = nv.nvfp4_kv_split_views(out)
    for scale in (k_scale, v_scale):
        scale.copy_(scale[..., torch.arange(scale.shape[-1]) // 2].clone())
    return out


@pytest.mark.parametrize("mutation", ["nibble_order_swapped", "scale_placement_wrong"])
def test_the_parity_check_catches_decode_layout_mutations(mutation) -> None:
    # A kernel that swapped the nibbles (or placed the scales wrongly) decodes the
    # clean cache like the correct kernel decodes the mutated cache.
    cache = _cache(seed=13)
    mutated = {
        "nibble_order_swapped": _swap_nibbles,
        "scale_placement_wrong": _scale_placement_wrong,
    }[mutation](cache)
    table, seq_lens = _table([[1, 3, 5, 7]]), _lens(2 * BLOCK + 9)
    _decode(cache, table, seq_lens)  # the unmutated decode passes (see above)
    scratch_k, scratch_v, _, offsets = _decode(mutated, table, seq_lens)
    with pytest.raises(AssertionError):
        _assert_scratch_equals_gather(
            cache, table, seq_lens, scratch_k, scratch_v, offsets
        )


def test_the_parity_check_catches_a_tail_that_is_not_zeroed() -> None:
    cache = _cache(seed=14)
    table, seq_lens = _table([[1, 3, 5, 7]]), _lens(2 * BLOCK + 9)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens)
    dirty = scratch_k.clone()
    dirty[2, 9:] = 1.0
    with pytest.raises(AssertionError):
        _assert_scratch_equals_gather(cache, table, seq_lens, dirty, scratch_v, offsets)


@pytest.mark.parametrize("mutation", ["compact_off_by_one", "offset_not_added"])
def test_the_parity_check_catches_table_mutations(mutation) -> None:
    cache = _cache(seed=15)
    table, seq_lens = _table([[1, 3, 5, 7], [2, 4, 6, 8]]), _lens(BLOCK + 5, 2 * BLOCK)

    def hook(compact, offsets):
        if mutation == "compact_off_by_one":
            return torch.where(compact >= 0, compact + 1, compact).int(), offsets + 1
        return compact, torch.zeros_like(offsets)

    scratch_k, scratch_v, _, offsets = _decode(
        cache, table, seq_lens, scratch_pages=8, table_hook=hook
    )
    reference_offsets = qsa_nvfp4.nvfp4_prefix_scratch_table(
        table, seq_lens, BLOCK, 8
    )[1]
    with pytest.raises(AssertionError):
        _assert_scratch_equals_gather(
            cache, table, seq_lens, scratch_k, scratch_v, reference_offsets
        )


def test_the_layer_scale_inside_the_decode_would_be_caught() -> None:
    cache = _cache(seed=16, k_scale=0.0213623046875)
    table, seq_lens = _table([[1, 3, 5, 7]]), _lens(BLOCK + 2)
    scratch_k, scratch_v, _, offsets = _decode(cache, table, seq_lens)
    scaled_k = (scratch_k.float() * 0.0213623046875).half()  # mutation m8
    with pytest.raises(AssertionError):
        _assert_scratch_equals_gather(
            cache, table, seq_lens, scaled_k, scratch_v, offsets
        )
