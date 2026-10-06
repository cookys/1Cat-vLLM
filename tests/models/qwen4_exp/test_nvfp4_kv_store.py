# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton NVFP4 store equals the pure-torch reference writer, byte for byte.

Without a GPU, run with ``TRITON_INTERPRET=1`` (the kernel then executes on the
CPU interpreter); with a GPU the same tests compile and run it for real::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \\
        python -m pytest --noconftest tests/models/qwen4_exp/test_nvfp4_kv_store.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

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
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv_triton as nvt  # noqa: E402

BLOCK = 16
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def _cache(blocks=6, heads=2, head_size=256):
    row = nv.nvfp4_kv_row_bytes(head_size)
    return torch.zeros((blocks, 2, BLOCK, heads, row), dtype=torch.uint8)


def _rows(shape, seed=0, scale=1.0, dtype=torch.float16):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


def _store_both(key, value, slots, *, k_scale=1.0, v_scale=1.0, blocks=6, heads=2):
    """Write with the reference and with the kernel; return both caches."""
    head_size = key.shape[-1]
    reference = _cache(blocks, heads, head_size)
    nv.reshape_and_cache_nvfp4_reference(
        key, value, reference, slots, k_scale=k_scale, v_scale=v_scale
    )
    actual = _cache(blocks, heads, head_size).to(DEVICE)
    scales = []
    for scale in (k_scale, v_scale):
        scales.append(
            scale.to(DEVICE) if isinstance(scale, torch.Tensor) else float(scale)
        )
    nvt.store_nvfp4_kv_triton(
        key.to(DEVICE),
        value.to(DEVICE),
        actual,
        slots.to(DEVICE),
        k_scale=scales[0],
        v_scale=scales[1],
    )
    return reference, actual.cpu()


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)  # 0-dim, as layer._k_scale


def _slots(tokens, blocks=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(blocks * BLOCK, generator=generator)[:tokens]


def test_the_store_matches_the_reference_byte_for_byte() -> None:
    key, value = _rows((40, 2, 256), 1, 3.0), _rows((40, 2, 256), 2, 3.0)
    reference, actual = _store_both(key, value, _slots(40))
    assert torch.equal(reference, actual)
    assert actual.any()


@pytest.mark.parametrize(
    "scales", [(1.0, 1.0), (0.0213623046875, 0.0404924675822258), (7.0, 0.5)]
)
def test_the_store_applies_the_layer_scales(scales) -> None:
    key, value = _rows((30, 2, 256), 3, 4.0), _rows((30, 2, 256), 4, 4.0)
    reference, actual = _store_both(
        key, value, _slots(30), k_scale=_scalar(scales[0]), v_scale=_scalar(scales[1])
    )
    assert torch.equal(reference, actual)
    # A different scale produces different bytes, so the scale is really used.
    other, _ = _store_both(key, value, _slots(30), k_scale=_scalar(scales[0] * 3))
    assert not torch.equal(reference, other)


def test_a_host_float_scale_is_the_same_as_a_device_scalar() -> None:
    key, value = _rows((20, 2, 256), 5), _rows((20, 2, 256), 6)
    _, from_tensor = _store_both(
        key, value, _slots(20), k_scale=_scalar(0.5), v_scale=_scalar(2.0)
    )
    _, from_float = _store_both(key, value, _slots(20), k_scale=0.5, v_scale=2.0)
    assert torch.equal(from_tensor, from_float)


def test_padding_and_out_of_range_slots_and_surplus_rows_are_skipped() -> None:
    slots = _slots(12)
    slots[2], slots[5], slots[9] = -1, 6 * BLOCK, 10**6
    key, value = _rows((17, 2, 256), 7), _rows((17, 2, 256), 8)  # 5 padded rows
    reference, actual = _store_both(key, value, slots)
    assert torch.equal(reference, actual)
    # Nine slots are valid, the other three left no bytes anywhere.
    (k_data, _), _ = nv.nvfp4_kv_split_views(actual)
    touched = int(k_data.reshape(6 * BLOCK, -1).any(dim=1).sum())
    assert touched == 9


def test_a_strided_value_slice_of_the_qkv_output_is_stored_correctly() -> None:
    qkv = _rows((24, 3, 2, 256), 9)  # [rows, q|k|v, heads, dim]
    key, value = qkv[:, 1], qkv[:, 2]
    assert value.stride(0) == 3 * 2 * 256 and value.stride(2) == 1
    reference, actual = _store_both(key.contiguous(), value.contiguous(), _slots(24))
    strided = _cache().to(DEVICE)
    nvt.store_nvfp4_kv_triton(
        key.to(DEVICE), value.to(DEVICE), strided, _slots(24).to(DEVICE)
    )
    assert torch.equal(reference, actual) and torch.equal(strided.cpu(), reference)


def test_fp32_inputs_are_accepted() -> None:
    key = _rows((16, 2, 256), 10, dtype=torch.float32)
    value = _rows((16, 2, 256), 11, dtype=torch.float32)
    reference, actual = _store_both(key, value, _slots(16))
    assert torch.equal(reference, actual)


@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_other_head_sizes(head_size) -> None:
    key = _rows((20, 1, head_size), 12)
    value = _rows((20, 1, head_size), 13)
    reference, actual = _store_both(key, value, _slots(20), heads=1)
    assert torch.equal(reference, actual)


def test_the_store_reproduces_the_rounding_boundary_cases_of_the_sm100_order() -> None:
    # fp16 group maxima where amax / (6 k) and the SM100 order give different
    # scale bytes at layer scale 7 (see the reference tests).
    amax = (
        torch.arange(0, 0x7C00, dtype=torch.int32).to(torch.int16).view(torch.float16)
    )
    values = amax.float().numpy()
    import numpy as np

    f32 = np.float32
    plain = nv.e4m3_encode(torch.from_numpy(values / f32(6.0 * 7.0)))
    order = nv.e4m3_encode(
        torch.from_numpy((f32(1) / f32(7.0)) * (values * (f32(1) / f32(6.0))))
    )
    picks = (plain != order).nonzero().flatten()[:8].tolist()
    key = torch.zeros(len(picks), 1, 256, dtype=torch.float16)
    for row, index in enumerate(picks):
        key[row, 0, 0] = float(values[index])
    reference, actual = _store_both(
        key,
        key,
        torch.arange(len(picks)),
        k_scale=_scalar(7.0),
        v_scale=_scalar(7.0),
        heads=1,
    )
    assert torch.equal(reference, actual)


def test_zero_tiny_and_saturating_groups() -> None:
    key = torch.zeros(6, 1, 256, dtype=torch.float16)
    key[1, 0, :16] = 0.001  # below the subnormal floor at scale 1
    key[2, 0, 0] = 6.0 * 448.0  # exactly the largest block scale
    key[3, 0, 0] = 30000.0  # saturates the block scale
    key[4, 0, :16] = torch.linspace(-3, 3, 16)
    key[5, 0, 5] = -1e-4
    reference, actual = _store_both(key, key, torch.arange(6), heads=1)
    assert torch.equal(reference, actual)


def test_groups_over_a_wide_dynamic_range() -> None:
    generator = torch.Generator().manual_seed(14)
    magnitude = 10.0 ** (torch.rand(64, 1, 16, 1, generator=generator) * 9 - 6)
    key = (torch.randn(64, 1, 16, 16, generator=generator) * magnitude).reshape(
        64, 1, 256
    )
    key = key.half()
    reference, actual = _store_both(key, key, torch.arange(64), heads=1)
    assert torch.equal(reference, actual)


def test_storing_twice_is_idempotent_and_deterministic() -> None:
    key, value = _rows((20, 2, 256), 15), _rows((20, 2, 256), 16)
    _, once = _store_both(key, value, _slots(20))
    cache = once.to(DEVICE)
    nvt.store_nvfp4_kv_triton(
        key.to(DEVICE), value.to(DEVICE), cache, _slots(20).to(DEVICE)
    )
    assert torch.equal(cache.cpu(), once)


def test_a_later_call_overwrites_only_its_own_slots() -> None:
    first_key, first_value = _rows((20, 2, 256), 17), _rows((20, 2, 256), 18)
    second_key, second_value = _rows((4, 2, 256), 19), _rows((4, 2, 256), 20)
    first_slots, second_slots = _slots(20), _slots(20)[:4]
    cache = _cache().to(DEVICE)
    for key, value, slots in (
        (first_key, first_value, first_slots),
        (second_key, second_value, second_slots),
    ):
        nvt.store_nvfp4_kv_triton(
            key.to(DEVICE), value.to(DEVICE), cache, slots.to(DEVICE)
        )
    expected = _cache()
    nv.reshape_and_cache_nvfp4_reference(first_key, first_value, expected, first_slots)
    nv.reshape_and_cache_nvfp4_reference(
        second_key, second_value, expected, second_slots
    )
    assert torch.equal(cache.cpu(), expected)


def test_a_store_followed_by_the_triton_gather_equals_the_reference_roundtrip() -> None:
    key, value = _rows((40, 2, 256), 21, 2.0), _rows((40, 2, 256), 22, 2.0)
    slots = _slots(40)
    cache = _cache().to(DEVICE)
    nvt.store_nvfp4_kv_triton(key.to(DEVICE), value.to(DEVICE), cache, slots.to(DEVICE))
    table = torch.arange(6, dtype=torch.int32).view(1, -1)
    indices = slots.int().view(1, -1)
    got = nvt.gather_dequant_nvfp4_kv_triton(
        cache,
        table.to(DEVICE),
        torch.zeros(1, dtype=torch.int32).to(DEVICE),
        indices.to(DEVICE),
    )
    expected_k = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(key))
    expected_v = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(value))
    assert torch.equal(got[0].cpu()[0], expected_k)
    assert torch.equal(got[1].cpu()[0], expected_v)


def _packed_member_views(monkeypatch):
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
    return raw, views["a"], views["b"]


@pytest.mark.skipif(
    "packed_members"
    not in __import__("inspect")
    .signature(
        __import__(
            "vllm.v1.worker.gpu.attn_utils", fromlist=["_reshape_kv_cache"]
        )._reshape_kv_cache
    )
    .parameters,
    reason="runner packed_members (p070 packs) is not on the d30469863 base; "
    "covered on p071-fable-prefill-bprime",
)
def test_a_packed_member_view_is_written_without_touching_its_neighbour(
    monkeypatch,
) -> None:
    raw, first, second = _packed_member_views(monkeypatch)
    key = _rows((30, 1, 256), 23)
    slots = torch.randperm(4 * BLOCK, generator=torch.Generator().manual_seed(24))[:30]
    nvt.store_nvfp4_kv_triton(key.to(DEVICE), key.to(DEVICE), second, slots.to(DEVICE))
    assert not first.any()  # the other member's pages are untouched
    expected = torch.zeros_like(second)
    nv.reshape_and_cache_nvfp4_reference(key, key, expected, slots)
    assert torch.equal(second, expected)
    assert raw.view(torch.uint8).count_nonzero() == expected.count_nonzero()


def _impl(dtype):
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl
    from vllm.v1.attention.backend import AttentionType

    impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
    impl.kv_cache_dtype = dtype
    impl.attn_type = AttentionType.DECODER
    return impl


@pytest.mark.skipif(
    "packed_members"
    not in __import__("inspect")
    .signature(
        __import__(
            "vllm.v1.worker.gpu.attn_utils", fromlist=["_reshape_kv_cache"]
        )._reshape_kv_cache
    )
    .parameters,
    reason="runner packed_members (p070 packs) is not on the d30469863 base; "
    "covered on p071-fable-prefill-bprime",
)
def test_the_owner_contract_stores_through_do_kv_cache_update() -> None:
    key, value = _rows((24, 2, 256), 25), _rows((24, 2, 256), 26)
    slots = _slots(24)
    layer = SimpleNamespace(
        _k_scale=_scalar(0.0213).to(DEVICE), _v_scale=_scalar(0.0405).to(DEVICE)
    )
    cache = _cache().to(DEVICE)
    _impl("nvfp4").do_kv_cache_update(
        layer, key.to(DEVICE), value.to(DEVICE), cache, slots.to(DEVICE)
    )
    expected = _cache()
    nv.reshape_and_cache_nvfp4_reference(
        key, value, expected, slots, k_scale=_scalar(0.0213), v_scale=_scalar(0.0405)
    )
    assert torch.equal(cache.cpu(), expected)


@pytest.mark.parametrize("dtype", ["auto", "fp8_e4m3"])
def test_other_cache_dtypes_still_use_reshape_and_cache_flash(
    monkeypatch, dtype
) -> None:
    from vllm.v1.attention.backends import flash_attn

    calls = []
    # The optional FlashAttention import is absent with CUDA hidden. This
    # delegation test supplies its own callback and never invokes a GPU op.
    monkeypatch.setattr(
        flash_attn,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    layer = SimpleNamespace(_k_scale="k", _v_scale="v")
    cache = torch.zeros(2, 2, BLOCK, 1, 256, dtype=torch.uint8)
    _impl(dtype).do_kv_cache_update(layer, "key", "value", cache, "slots")
    assert len(calls) == 1
    assert calls[0][4:] == ("slots", dtype, "k", "v")


def test_a_store_with_no_tokens_is_a_noop() -> None:
    cache = _cache().to(DEVICE)
    nvt.store_nvfp4_kv_triton(
        torch.zeros(3, 2, 256, dtype=torch.float16, device=DEVICE),
        torch.zeros(3, 2, 256, dtype=torch.float16, device=DEVICE),
        cache,
        torch.zeros(0, dtype=torch.int64, device=DEVICE),
    )
    assert not cache.any()


@pytest.mark.parametrize(
    ("what", "error"),
    [
        ("key_dtype", TypeError),
        ("slot_dtype", TypeError),
        ("key_heads", ValueError),
        ("last_stride", ValueError),
        ("head_size", ValueError),
        ("rows", ValueError),
        ("scale_dtype", TypeError),
    ],
)
def test_the_store_validates_its_inputs(what, error) -> None:
    heads, head_size = 2, 256
    key = torch.zeros(8, heads, head_size, dtype=torch.float16, device=DEVICE)
    value = key.clone()
    slots = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cache = _cache(heads=heads).to(DEVICE)
    kwargs = {}
    if what == "key_dtype":
        key = key.to(torch.bfloat16)
    elif what == "slot_dtype":
        slots = slots.int()
    elif what == "key_heads":
        key = torch.zeros(8, 1, head_size, dtype=torch.float16, device=DEVICE)
    elif what == "last_stride":
        key = torch.zeros(8, heads, head_size * 2, dtype=torch.float16, device=DEVICE)[
            ..., ::2
        ]
    elif what == "head_size":
        cache = _cache(heads=heads, head_size=48).to(DEVICE)
        key = torch.zeros(8, heads, 48, dtype=torch.float16, device=DEVICE)
        value = key.clone()
    elif what == "rows":
        slots = torch.arange(9, dtype=torch.int64, device=DEVICE)
    elif what == "scale_dtype":
        kwargs["k_scale"] = torch.ones(1, dtype=torch.float16, device=DEVICE)
    with pytest.raises(error):
        nvt.store_nvfp4_kv_triton(key, value, cache, slots, **kwargs)


# --------------------------------------------------------------------------- #
# The in-kernel software encoders, against the torch ones, exhaustively
# --------------------------------------------------------------------------- #
def _e4m3_probe_values() -> torch.Tensor:
    codes = torch.arange(0, 0x7F, dtype=torch.uint8)  # every finite positive code
    grid = nv.e4m3_decode(codes)
    middles = (grid[:-1] + grid[1:]) / 2  # the round-to-nearest-even ties
    around = torch.cat([middles, grid])
    up, down = (
        torch.nextafter(around, torch.tensor(1e9)),
        torch.nextafter(around, torch.tensor(-1e9)),
    )
    halves = (
        torch.arange(0, 0x7C00, dtype=torch.int32).to(torch.int16).view(torch.float16)
    )
    return torch.cat(
        [around, up, down, halves.float(), torch.tensor([448.0, 464.0, 1e9])]
    )


def test_the_in_kernel_e4m3_encoder_matches_torch_on_every_tie_and_every_fp16() -> None:
    x = _e4m3_probe_values().clamp_min(0.0)
    got = nvt._probe_encoder(x.to(DEVICE), "e4m3").cpu()
    assert torch.equal(got, nv.e4m3_encode(x).to(torch.int32))


def test_the_in_kernel_e2m1_encoder_matches_torch_on_every_tie_and_a_dense_sweep() -> (
    None
):
    grid = torch.tensor(nv.E2M1_GRID)
    middles = (grid[:-1] + grid[1:]) / 2
    around = torch.cat([middles, grid, torch.tensor([6.5, 100.0])])
    probes = torch.cat(
        [
            around,
            torch.nextafter(around, torch.tensor(1e9)),
            torch.nextafter(around, torch.tensor(-1e9)),
            torch.linspace(-7.0, 7.0, 4001),
            torch.tensor([-0.0, 0.0, -1e-9]),
        ]
    )
    x = torch.cat([probes, -probes])
    got = nvt._probe_encoder(x.to(DEVICE), "e2m1").cpu()
    assert torch.equal(got, nv.e2m1_encode(x).to(torch.int32))
