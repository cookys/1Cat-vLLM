# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the pure-torch NVFP4 KV reference (plan 071 stage 1).

They pin the format down before any kernel exists: scalar codecs, nibble
packing, the row quantizer and its error bounds, the page layout, the cache
writer, and the sparse gather with its illegal-index zero-fill rule.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import regex as re
import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv

GRID = nv.E2M1_GRID
D = 256


def _rows(shape, kind="gaussian", seed=0, dtype=torch.float16):
    generator = torch.Generator().manual_seed(seed)
    if kind == "gaussian":
        values = torch.randn(shape, generator=generator)
    elif kind == "heavy":
        # Student-t with 3 degrees of freedom: heavy tails, finite variance.
        normal = torch.randn(shape, generator=generator)
        chi2 = (torch.randn((3, *shape), generator=generator) ** 2).sum(0)
        values = normal / (chi2 / 3).sqrt()
    else:
        raise AssertionError(kind)
    return values.to(dtype)


def _group_amax(x):
    groups = x.float().reshape(*x.shape[:-1], -1, nv.NVFP4_KV_GROUP_SIZE)
    return groups.abs().amax(-1, keepdim=True)


def _sm100_store_reference(x: torch.Tensor, layer_scale: float):
    """Per-group scale byte and nibbles in the SM100 store's order, in numpy fp32.

    ``nvfp4_utils.cuh:221-285`` with IEEE division for ``rcp.approx.ftz``:
    SFScaleVal = 1 / k_scale; SFValue = SFScaleVal * (vecMax * (1 / 6));
    byte = E4M3(SFValue); outputScale = 1 / (SFValue_q * (1 / SFScaleVal));
    nibble = E2M1(x * outputScale).
    """
    f32 = np.float32
    groups = x.float().numpy().reshape(-1, 16)
    sf_scale = f32(1.0) / f32(layer_scale)
    amax = np.abs(groups).max(axis=1).astype(f32)
    sf = sf_scale * (amax * (f32(1.0) / f32(6.0)))
    byte = nv.e4m3_encode(torch.from_numpy(sf))
    sf_q = nv.e4m3_decode(byte).numpy().astype(f32)
    with np.errstate(divide="ignore"):
        out = np.where(sf_q != 0, f32(1.0) / (sf_q * (f32(1.0) / sf_scale)), f32(0))
    nibbles = nv.e2m1_encode(torch.from_numpy((groups * out[:, None]).astype(f32)))
    return byte.numpy(), nibbles.numpy()


def _empty_cache(blocks, block_size, heads=1, head_size=D):
    row = nv.nvfp4_kv_row_bytes(head_size)
    return torch.zeros((blocks, 2, block_size, heads, row), dtype=torch.uint8)


# --------------------------------------------------------------------------- #
# E4M3 scale codec
# --------------------------------------------------------------------------- #
def _e4m3_formula(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def test_e4m3_decode_matches_the_bit_formula_for_every_code() -> None:
    codes = torch.arange(256, dtype=torch.uint8)
    decoded = nv.e4m3_decode(codes)
    for code in range(256):
        expected = _e4m3_formula(code)
        if math.isnan(expected):
            assert torch.isnan(decoded[code])
        else:
            assert decoded[code].item() == expected, hex(code)


def test_e4m3_encode_is_the_inverse_of_decode_on_all_finite_codes() -> None:
    codes = torch.arange(256, dtype=torch.uint8)
    finite = codes[~torch.isnan(nv.e4m3_decode(codes))]
    reencoded = nv.e4m3_encode(nv.e4m3_decode(finite))
    # -0.0 and +0.0 both decode to a zero; every other code is preserved.
    nonzero = nv.e4m3_decode(finite) != 0
    assert torch.equal(reencoded[nonzero], finite[nonzero])
    assert torch.equal(nv.e4m3_decode(reencoded), nv.e4m3_decode(finite))


@pytest.mark.parametrize("value", [448.0, 449.0, 464.0, 1e4, float("inf")])
def test_e4m3_encode_saturates_instead_of_producing_nan(value: float) -> None:
    for sign, code in ((1.0, 0x7E), (-1.0, 0xFE)):
        encoded = nv.e4m3_encode(torch.tensor([sign * value]))
        assert encoded.item() == code
        assert nv.e4m3_decode(encoded).item() == sign * 448.0


def test_e4m3_encode_rounds_to_nearest_even() -> None:
    # Midpoints of neighbouring codes go to the even code (even mantissa bit).
    assert nv.e4m3_encode(torch.tensor([1.0625])).item() == 0x38  # 1.0 | 1.125
    assert nv.e4m3_encode(torch.tensor([1.1875])).item() == 0x3A  # 1.125 | 1.25
    assert nv.e4m3_encode(torch.tensor([1.0624])).item() == 0x38
    assert nv.e4m3_encode(torch.tensor([1.0626])).item() == 0x39


def test_e4m3_subnormal_scale_edge_cases() -> None:
    tiny = nv.E4M3_MIN_SUBNORMAL
    assert nv.e4m3_encode(torch.tensor([tiny])).item() == 0x01
    assert nv.e4m3_encode(torch.tensor([tiny / 2])).item() == 0x00  # tie -> even
    assert nv.e4m3_encode(torch.tensor([tiny / 2 * 1.01])).item() == 0x01
    assert nv.e4m3_encode(torch.tensor([tiny / 4])).item() == 0x00
    assert nv.e4m3_encode(torch.tensor([2.0**-6])).item() == 0x08  # min normal
    assert nv.e4m3_decode(torch.tensor([7], dtype=torch.uint8)).item() == 7 * tiny


def test_e4m3_nan_codes_decode_to_nan() -> None:
    for code in (0x7F, 0xFF):
        assert torch.isnan(nv.e4m3_decode(torch.tensor([code], dtype=torch.uint8)))
    with pytest.raises(TypeError):
        nv.e4m3_decode(torch.tensor([1.0]))


# --------------------------------------------------------------------------- #
# E2M1 data codec and nibble packing
# --------------------------------------------------------------------------- #
def test_e2m1_decode_grid_and_sign() -> None:
    nibbles = torch.arange(16, dtype=torch.uint8)
    decoded = nv.e2m1_decode(nibbles)
    assert decoded[:8].tolist() == list(GRID)
    assert decoded[9:].tolist() == [-v for v in GRID[1:]]
    assert math.copysign(1.0, decoded[8].item()) == -1.0  # code 0x8 is -0.0


@pytest.mark.parametrize("magnitude", GRID)
def test_e2m1_encode_exact_grid_values_roundtrip(magnitude: float) -> None:
    for sign in (1.0, -1.0):
        nibble = nv.e2m1_encode(torch.tensor([sign * magnitude]))
        assert nv.e2m1_decode(nibble).item() == sign * magnitude or magnitude == 0.0


@pytest.mark.parametrize(
    ("midpoint", "expected"),
    [
        (0.25, 0.0),
        (0.75, 1.0),
        (1.25, 1.0),
        (1.75, 2.0),
        (2.5, 2.0),
        (3.5, 4.0),
        (5.0, 4.0),
    ],
)
def test_e2m1_encode_ties_go_to_the_even_code(midpoint, expected) -> None:
    for sign in (1.0, -1.0):
        decoded = nv.e2m1_decode(nv.e2m1_encode(torch.tensor([sign * midpoint])))
        assert abs(decoded.item()) == expected


def test_e2m1_encode_rounds_to_the_nearest_grid_point() -> None:
    x = torch.linspace(-7.0, 7.0, 14001)
    decoded = nv.e2m1_decode(nv.e2m1_encode(x))
    grid = torch.tensor([-g for g in GRID[:0:-1]] + list(GRID))
    nearest_distance = (x[:, None] - grid[None, :]).abs().min(dim=1).values
    assert torch.allclose((x - decoded).abs(), nearest_distance.clamp_min(0), atol=1e-6)


def test_e2m1_encode_saturates_at_six() -> None:
    nibbles = nv.e2m1_encode(torch.tensor([6.0, 6.1, 100.0, -100.0, float("inf")]))
    assert nv.e2m1_decode(nibbles).tolist() == [6.0, 6.0, 6.0, -6.0, 6.0]


def test_e2m1_encode_never_emits_negative_zero() -> None:
    tiny_negative = torch.tensor([-0.0, -0.1, -0.25, -1e-9])
    nibbles = nv.e2m1_encode(tiny_negative)
    assert nibbles.tolist() == [0, 0, 0, 0]


def test_pack_unpack_is_bit_exact_for_every_byte() -> None:
    every_byte = torch.arange(256, dtype=torch.uint8)
    assert torch.equal(nv.pack_nibbles(nv.unpack_nibbles(every_byte)), every_byte)
    nibbles = nv.unpack_nibbles(every_byte)
    assert nibbles.shape == (512,)
    assert nibbles.max().item() == 15


def test_low_nibble_is_the_even_element() -> None:
    nibbles = torch.tensor([0x1, 0x2, 0x3, 0x4], dtype=torch.uint8)
    assert nv.pack_nibbles(nibbles).tolist() == [0x21, 0x43]
    assert nv.unpack_nibbles(torch.tensor([0x21], dtype=torch.uint8)).tolist() == [
        0x1,
        0x2,
    ]


def test_pack_rejects_odd_length_and_unpack_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError):
        nv.pack_nibbles(torch.zeros(3, dtype=torch.uint8))
    with pytest.raises(TypeError):
        nv.unpack_nibbles(torch.zeros(4, dtype=torch.int32))


def test_packing_convention_matches_the_weight_emulation_unpacker() -> None:
    # The repo's NVFP4 weight path reads the low nibble first. A packed KV row
    # must unpack to the same values through that existing helper.
    x = _rows((8, D), seed=3)
    packed, _ = nv.quantize_kv_nvfp4(x)
    via_weights = break_fp4_bytes(packed, torch.float32)
    own = nv.e2m1_decode(nv.unpack_nibbles(packed))
    assert torch.equal(via_weights, own)


# --------------------------------------------------------------------------- #
# Row quantizer
# --------------------------------------------------------------------------- #
def test_quantize_output_shapes_and_dtypes() -> None:
    packed, scales = nv.quantize_kv_nvfp4(_rows((5, 2, D)))
    assert packed.shape == (5, 2, D // 2) and packed.dtype == torch.uint8
    assert scales.shape == (5, 2, D // 16) and scales.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("head_size", [0, 8, 17, 250])
def test_quantize_requires_a_multiple_of_the_group_size(head_size) -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        nv.quantize_kv_nvfp4(torch.zeros(2, head_size))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_quantize_rejects_non_finite_inputs(bad) -> None:
    x = torch.zeros(1, D)
    x[0, 7] = bad
    with pytest.raises(ValueError, match="finite"):
        nv.quantize_kv_nvfp4(x)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_quantize_rejects_a_bad_layer_scale(scale) -> None:
    with pytest.raises(ValueError, match="layer scale"):
        nv.quantize_kv_nvfp4(torch.ones(1, D), layer_scale=scale)


def test_all_zero_rows_have_zero_scale_and_zero_bytes() -> None:
    packed, scales = nv.quantize_kv_nvfp4(torch.zeros(3, D))
    assert not packed.any() and not scales.view(torch.uint8).any()
    assert not nv.dequantize_kv_nvfp4(packed, scales).any()


def test_values_on_the_grid_roundtrip_exactly() -> None:
    # Every group holds the 8 grid magnitudes with alternating signs, scaled
    # by a power of two: the block scale is exact and so is every element.
    pattern = torch.tensor(list(GRID) * 2) * torch.tensor([1.0, -1.0] * 8)
    for exponent in (-6, -1, 0, 3, 7):
        x = (pattern * 2.0**exponent).repeat(16).reshape(1, D)
        packed, scales = nv.quantize_kv_nvfp4(x)
        restored = nv.dequantize_kv_nvfp4(packed, scales, out_dtype=torch.float32)
        assert torch.equal(restored, x), exponent


def test_group_scale_is_the_rounded_amax_over_six() -> None:
    x = _rows((16, D), seed=5)
    _, scales = nv.quantize_kv_nvfp4(x)
    expected, _ = _sm100_store_reference(x, 1.0)
    assert torch.equal(scales.view(torch.uint8).reshape(-1), torch.from_numpy(expected))
    # Away from rounding boundaries this is the rounded amax / 6.
    plain = nv.e4m3_encode(_group_amax(x).squeeze(-1) / 6.0)
    assert (scales.view(torch.uint8) != plain).float().mean() < 0.01


def test_maximum_scale_group_is_not_clipped() -> None:
    # amax = 6 * 448 puts the block scale exactly at the E4M3 maximum.
    x = torch.zeros(1, D)
    x[0, 0] = 6.0 * 448.0
    packed, scales = nv.quantize_kv_nvfp4(x)
    assert scales.view(torch.uint8)[0, 0].item() == 0x7E
    restored = nv.dequantize_kv_nvfp4(packed, scales, out_dtype=torch.float32)
    assert restored[0, 0].item() == 6.0 * 448.0


def test_scale_beyond_the_e4m3_range_saturates_and_clips_the_group() -> None:
    x = torch.zeros(1, D)
    x[0, 0] = 10.0 * 448.0
    packed, scales = nv.quantize_kv_nvfp4(x)
    assert scales.view(torch.uint8)[0, 0].item() == 0x7E
    restored = nv.dequantize_kv_nvfp4(packed, scales, out_dtype=torch.float32)
    assert restored[0, 0].item() == 6.0 * 448.0  # clipped, finite


def test_a_group_below_the_subnormal_floor_flushes_to_zero() -> None:
    x = torch.zeros(1, D)
    x[0, :16] = 6.0 * nv.E4M3_MIN_SUBNORMAL / 4  # scale rounds to zero
    packed, scales = nv.quantize_kv_nvfp4(x)
    assert scales.view(torch.uint8)[0, 0].item() == 0
    assert not nv.dequantize_kv_nvfp4(packed, scales).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_size", [64, 256])
def test_per_group_error_bound_gaussian(dtype, head_size) -> None:
    x = _rows((512, head_size), "gaussian", seed=11, dtype=dtype)
    packed, scales = nv.quantize_kv_nvfp4(x)
    restored = nv.dequantize_kv_nvfp4(packed, scales, out_dtype=torch.float32)
    error = (restored - x.float()).reshape(*x.shape[:-1], -1, 16).abs()
    # Largest E2M1 step is 2 grid units (4 -> 6), so half a step is one scale;
    # the E4M3 scale is within 1/16 of amax/6 in the normal range.
    bound = _group_amax(x) / 6.0 * (1.0 + 1.0 / 16.0)
    assert (error <= bound + 1e-6).all()


def test_per_group_error_bound_heavy_tail() -> None:
    x = _rows((512, D), "heavy", seed=12)
    packed, scales = nv.quantize_kv_nvfp4(x)
    restored = nv.dequantize_kv_nvfp4(packed, scales, out_dtype=torch.float32)
    error = (restored - x.float()).reshape(*x.shape[:-1], -1, 16).abs()
    amax = _group_amax(x)
    normal_range = amax / 6.0 >= 2.0**-6
    bound = amax / 6.0 * (1.0 + 1.0 / 16.0)
    assert ((error <= bound + 1e-6) | ~normal_range).all()


def test_an_outlier_group_does_not_degrade_the_other_groups() -> None:
    x = _rows((64, D), seed=13)
    reference = nv.dequantize_kv_nvfp4(
        *nv.quantize_kv_nvfp4(x), out_dtype=torch.float32
    )
    spiked = x.clone()
    spiked[:, :16] *= 500.0  # first group only
    restored = nv.dequantize_kv_nvfp4(
        *nv.quantize_kv_nvfp4(spiked), out_dtype=torch.float32
    )
    assert torch.equal(restored[:, 16:], reference[:, 16:])


@pytest.mark.parametrize(
    ("kind", "low", "high"), [("gaussian", 0.07, 0.12), ("heavy", 0.07, 0.14)]
)
def test_relative_l2_error_statistics(kind, low, high) -> None:
    x = _rows((4096, D), kind, seed=21)
    restored = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(x), out_dtype=torch.float32)
    relative = (
        torch.linalg.norm(restored - x.float()) / torch.linalg.norm(x.float())
    ).item()
    assert low < relative < high, relative


def test_layer_scale_invariance_for_power_of_two_scales() -> None:
    x = _rows((32, D), seed=31)
    base = nv.quantize_kv_nvfp4(x)
    for scale in (0.25, 4.0, 64.0):
        scaled = nv.quantize_kv_nvfp4(
            (x.float() * scale).to(torch.float32), layer_scale=scale
        )
        assert torch.equal(scaled[0], base[0])
        assert torch.equal(scaled[1].view(torch.uint8), base[1].view(torch.uint8))


@pytest.mark.parametrize("scale", [0.01, 0.5, 7.0])
def test_layer_scale_is_divided_out_of_the_group_scale_and_multiplied_back(
    scale,
) -> None:
    x = _rows((64, D), seed=36)
    packed, scales = nv.quantize_kv_nvfp4(x, layer_scale=scale)
    expected, _ = _sm100_store_reference(x, scale)
    assert torch.equal(scales.view(torch.uint8).reshape(-1), torch.from_numpy(expected))
    plain = nv.e4m3_encode(_group_amax(x).squeeze(-1) / (6.0 * scale))
    assert (scales.view(torch.uint8) != plain).float().mean() < 0.01
    restored = nv.dequantize_kv_nvfp4(
        packed, scales, layer_scale=scale, out_dtype=torch.float32
    )
    relative = torch.linalg.norm(restored - x.float()) / torch.linalg.norm(x.float())
    assert relative < 0.12, (scale, relative.item())


def test_an_oversized_layer_scale_pushes_group_scales_into_the_subnormal_range() -> (
    None
):
    # Documents the calibration constraint: with N(0, 1) data the group scales
    # are about amax / (6 * layer_scale); once that falls below 2**-6 the E4M3
    # scale has fewer than three mantissa bits and the error grows.
    x = _rows((256, D), seed=38)

    def relative(scale: float) -> float:
        packed, scales = nv.quantize_kv_nvfp4(x, layer_scale=scale)
        restored = nv.dequantize_kv_nvfp4(
            packed, scales, layer_scale=scale, out_dtype=torch.float32
        )
        return (
            torch.linalg.norm(restored - x.float()) / torch.linalg.norm(x.float())
        ).item()

    assert relative(300.0) > 1.5 * relative(1.0)


def test_a_wrong_layer_scale_at_dequantization_rescales_the_result() -> None:
    x = _rows((8, D), seed=37)
    packed, scales = nv.quantize_kv_nvfp4(x, layer_scale=2.0)
    right = nv.dequantize_kv_nvfp4(
        packed, scales, layer_scale=2.0, out_dtype=torch.float32
    )
    wrong = nv.dequantize_kv_nvfp4(
        packed, scales, layer_scale=1.0, out_dtype=torch.float32
    )
    assert torch.equal(right, wrong * 2.0)


@pytest.mark.parametrize("scale", [1.0, 0.0213623046875, 0.0404924675822258, 3.0])
@pytest.mark.parametrize("kind", ["gaussian", "heavy"])
def test_the_bytes_equal_the_sm100_store_operation_order(scale, kind) -> None:
    # Scales and nibbles, byte for byte, against an independent numpy fp32
    # emulation of nvfp4_utils.cuh:221-285 (IEEE division for rcp.approx).
    x = _rows((256, D), kind, seed=41)
    packed, scales = nv.quantize_kv_nvfp4(x, layer_scale=scale)
    byte, nibbles = _sm100_store_reference(x, scale)
    assert torch.equal(scales.view(torch.uint8).reshape(-1), torch.from_numpy(byte))
    assert torch.equal(
        nv.unpack_nibbles(packed).reshape(-1, 16), torch.from_numpy(nibbles)
    )


def test_the_operation_order_decides_the_byte_at_rounding_boundaries() -> None:
    # For layer scale 7 some fp16 group maxima land where amax / (6 k) and the
    # SM100 order (1/k) * (amax * fp32(1/6)) round to different E4M3 bytes. The
    # quantizer must follow the SM100 order.
    f32 = np.float32
    amax = (
        torch.arange(0, 0x7C00, dtype=torch.int32).to(torch.int16).view(torch.float16)
    )
    values = amax.float().numpy()
    plain = nv.e4m3_encode(torch.from_numpy(values / f32(6.0 * 7.0)))
    sm100 = nv.e4m3_encode(
        torch.from_numpy((f32(1) / f32(7.0)) * (values * (f32(1) / f32(6.0))))
    )
    boundary = (plain != sm100).nonzero().flatten()
    assert boundary.numel() >= 8  # the discriminating cases exist
    for index in boundary[:8].tolist():
        x = torch.zeros(1, D)
        x[0, 0] = float(values[index])
        _, scales = nv.quantize_kv_nvfp4(x, layer_scale=7.0)
        assert scales.view(torch.uint8)[0, 0].item() == sm100[index].item()
        assert scales.view(torch.uint8)[0, 0].item() != plain[index].item()


@pytest.mark.parametrize("divisor", [2688.0, 448.0, 200.0, None])
def test_the_global_scale_choice_barely_changes_the_error(divisor) -> None:
    # k_scale = amax / divisor; None is the unit scale. 2688 = 6 * 448 is the
    # smallest scale whose largest block scale still fits E4M3. The E4M3 QSA
    # overlay (amax / 448 in upstream's convention) can be reused as is, and so
    # can 1.0: the block scale carries the range.
    x = _rows((1024, D), seed=42)
    scale = 1.0 if divisor is None else float(x.float().abs().max()) / divisor
    packed, scales = nv.quantize_kv_nvfp4(x, layer_scale=scale)
    restored = nv.dequantize_kv_nvfp4(
        packed, scales, layer_scale=scale, out_dtype=torch.float32
    )
    relative = (
        torch.linalg.norm(restored - x.float()) / torch.linalg.norm(x.float())
    ).item()
    assert 0.085 < relative < 0.105, (divisor, relative)
    assert nv.e4m3_decode(scales).max().item() <= 448.0


def test_an_amax_over_448_scale_leaves_the_block_scale_far_from_the_cap() -> None:
    x = _rows((512, D), seed=43)
    scale = float(x.float().abs().max()) / 448.0
    _, scales = nv.quantize_kv_nvfp4(x, layer_scale=scale)
    assert nv.e4m3_decode(scales).max().item() <= 448.0 / 6.0 * (1 + 1 / 16)


# --------------------------------------------------------------------------- #
# The SM100/SM120 store is the specification: pin its source conventions
# --------------------------------------------------------------------------- #
_CSRC = Path(__file__).resolve().parents[3] / "csrc" / "libtorch_stable"
_needs_csrc = pytest.mark.skipif(
    not (_CSRC / "nvfp4_kv_cache_kernels.cu").is_file(),
    reason="the csrc tree is not part of this install",
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


@_needs_csrc
def test_sm100_store_page_layout_and_scale_convention() -> None:
    source = _squash((_CSRC / "nvfp4_kv_cache_kernels.cu").read_text())
    assert "page = [K_data | K_scale | V_data | V_scale]" in source
    # The kernel is handed 1 / k_scale and 1 / v_scale.
    assert "1.0f / ((kv == 0) ? *k_scale_ptr : *v_scale_ptr)" in source
    # K scales are written linearly, V scales swizzled (V100 writes both linear).
    assert "K (kv==0): linear layout (no swizzle)" in source
    assert "V (kv==1): swizzled layout" in source
    # 16 values -> 8 data bytes: low word then high word of one uint64.
    assert "(uint64_t(packed.hi) << 32) | uint64_t(packed.lo)" in source
    assert "data_byte_offset = group_in_head * 8" in source


@_needs_csrc
def test_sm100_store_scale_math_and_nibble_order() -> None:
    source = _squash((_CSRC / "quantization/fp4/nvfp4_utils.cuh").read_text())
    assert "SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f))" in source
    assert "__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue)" in source
    assert (
        "reciprocal_approximate_ftz( SFValue * reciprocal_approximate_ftz(SFScaleVal))"
        in source
    )
    # cvt.rn.satfinite.e2m1x2 d, a, b puts a in the high nibble: the odd element
    # is the first source operand, so the even element is the low nibble.
    assert "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;" in source
    assert "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;" in source
    assert "// maximum value of e2m1 = 6.0." in source


@_needs_csrc
def test_the_only_sm100_only_intrinsic_is_the_e2m1_conversion() -> None:
    utils = (_CSRC / "quantization/fp4/nvfp4_utils.cuh").read_text()
    # Every cvt of the store is the e2m1x2 one, with no __CUDA_ARCH__ guard, so a
    # build for another architecture needs it replaced (e2m1_encode is the spec).
    assert set(re.findall(r"cvt\.[a-z.0-9]+", utils)) == {"cvt.rn.satfinite.e2m1x2.f32"}
    assert "__CUDA_ARCH__" not in utils


def test_default_layer_scale_is_one() -> None:
    x = _rows((8, D), seed=32)
    a = nv.quantize_kv_nvfp4(x)
    b = nv.quantize_kv_nvfp4(x, layer_scale=1.0)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1].view(torch.uint8), b[1].view(torch.uint8))


def test_quantization_is_deterministic_and_row_independent() -> None:
    x = _rows((16, 2, D), seed=33)
    batched = nv.quantize_kv_nvfp4(x)
    for row in (0, 5, 15):
        single = nv.quantize_kv_nvfp4(x[row : row + 1])
        assert torch.equal(single[0], batched[0][row : row + 1])
        assert torch.equal(
            single[1].view(torch.uint8), batched[1][row : row + 1].view(torch.uint8)
        )
    again = nv.quantize_kv_nvfp4(x)
    assert torch.equal(again[0], batched[0])


def test_heads_are_quantized_independently() -> None:
    x = _rows((8, 2, D), seed=34)
    x[:, 1] *= 100.0
    packed, scales = nv.quantize_kv_nvfp4(x)
    solo = nv.quantize_kv_nvfp4(x[:, 0])
    assert torch.equal(packed[:, 0], solo[0])


def test_nibble_signs_follow_the_input_signs() -> None:
    x = _rows((8, D), seed=35)
    packed, _ = nv.quantize_kv_nvfp4(x)
    nibbles = nv.unpack_nibbles(packed)
    sign_bit = (nibbles & 0x8) != 0
    assert (sign_bit <= (x < 0)).all()  # a set sign bit implies a negative input
    nonzero = (nibbles & 0x7) != 0
    assert (sign_bit[nonzero] == (x < 0)[nonzero]).all()


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_dequantize_output_dtype(out_dtype) -> None:
    packed, scales = nv.quantize_kv_nvfp4(_rows((2, D)))
    assert (
        nv.dequantize_kv_nvfp4(packed, scales, out_dtype=out_dtype).dtype == out_dtype
    )


def test_dequantize_validates_the_row_geometry() -> None:
    packed, scales = nv.quantize_kv_nvfp4(_rows((2, D)))
    with pytest.raises(ValueError, match="same row"):
        nv.dequantize_kv_nvfp4(packed, scales[..., :-1])


def test_a_nan_scale_byte_decodes_to_nan() -> None:
    packed, scales = nv.quantize_kv_nvfp4(_rows((1, D)))
    poisoned = scales.view(torch.uint8).clone()
    poisoned[0, 3] = 0x7F
    restored = nv.dequantize_kv_nvfp4(packed, poisoned, out_dtype=torch.float32)
    assert torch.isnan(restored[0, 48:64]).all()
    assert (
        torch.isfinite(restored[0, :48]).all()
        and torch.isfinite(restored[0, 64:]).all()
    )


# --------------------------------------------------------------------------- #
# Byte accounting and page layout
# --------------------------------------------------------------------------- #
def test_row_and_token_bytes_for_head_size_256() -> None:
    assert nv.nvfp4_kv_data_bytes(256) == 128
    assert nv.nvfp4_kv_scale_bytes(256) == 16
    assert nv.nvfp4_kv_row_bytes(256) == 144
    assert nv.nvfp4_kv_bytes_per_token(256, 1) == 288  # K 144 + V 144


@pytest.mark.parametrize(("block", "heads"), [(16, 1), (1616, 1), (2784, 1), (64, 2)])
def test_page_bytes_scale_with_block_and_heads(block, heads) -> None:
    assert nv.nvfp4_kv_page_bytes(block, heads, 256) == block * heads * 288


@pytest.mark.parametrize("head_size", [0, 15, 100])
def test_byte_math_rejects_a_bad_head_size(head_size) -> None:
    with pytest.raises(ValueError):
        nv.nvfp4_kv_row_bytes(head_size)


@pytest.mark.parametrize(("block", "heads"), [(16, 1), (32, 2)])
def test_page_regions_are_disjoint_and_cover_the_page(block, heads) -> None:
    regions = nv.nvfp4_kv_page_regions(block, heads, D)
    assert list(regions) == ["k_data", "k_scale", "v_data", "v_scale"]
    cursor = 0
    for offset, size in regions.values():
        assert offset == cursor
        cursor += size
    assert cursor == nv.nvfp4_kv_page_bytes(block, heads, D)


def test_split_views_write_through_at_the_documented_offsets() -> None:
    block, heads = 16, 2
    cache = _empty_cache(3, block, heads)
    (k_data, v_data), (k_scale, v_scale) = nv.nvfp4_kv_split_views(cache)
    k_data.fill_(1)
    k_scale.fill_(2)
    v_data.fill_(3)
    v_scale.fill_(4)
    page = cache[1].reshape(-1)  # one physical page, K side then V side
    for name, value in (("k_data", 1), ("k_scale", 2), ("v_data", 3), ("v_scale", 4)):
        offset, size = nv.nvfp4_kv_page_regions(block, heads, D)[name]
        assert (page[offset : offset + size] == value).all(), name


# --------------------------------------------------------------------------- #
# Cache writer and sparse gather
# --------------------------------------------------------------------------- #
BLOCK = 16


def _written_cache(num_blocks=6, tokens=40, heads=1, seed=0, k_scale=1.0, v_scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    cache = _empty_cache(num_blocks, BLOCK, heads)
    slots = torch.randperm(num_blocks * BLOCK, generator=generator)[:tokens]
    key = _rows((tokens, heads, D), seed=seed + 1)
    value = _rows((tokens, heads, D), seed=seed + 2)
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, slots, k_scale=k_scale, v_scale=v_scale
    )
    return cache, slots, key, value


def _single_request_table(num_blocks):
    return torch.arange(num_blocks, dtype=torch.int32).view(1, -1)


def test_cache_write_then_gather_matches_the_dequantized_rows() -> None:
    num_blocks, tokens = 6, 40
    cache, slots, key, value = _written_cache(num_blocks, tokens, heads=2)
    table = _single_request_table(num_blocks)
    keys, values = nv.gather_dequant_nvfp4_kv(
        cache, table, torch.zeros(1, dtype=torch.int32), slots.int().view(1, -1)
    )
    expected_k = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(key))
    expected_v = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(value))
    assert torch.equal(keys[0], expected_k)
    assert torch.equal(values[0], expected_v)


def test_gather_translates_logical_pages_through_the_block_table() -> None:
    cache, slots, key, _ = _written_cache(6, 40)
    # Physical blocks are visited in a permuted order by the logical sequence.
    order = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.int32).view(1, -1)
    logical = torch.arange(6 * BLOCK, dtype=torch.int32).view(1, -1)
    keys, _ = nv.gather_dequant_nvfp4_kv(
        cache, order, torch.zeros(1, dtype=torch.int32), logical
    )
    unpermuted, _ = nv.gather_dequant_nvfp4_kv(
        cache, _single_request_table(6), torch.zeros(1, dtype=torch.int32), logical
    )
    for logical_page in range(6):
        src = int(order[0, logical_page])
        got = keys[0, logical_page * BLOCK : (logical_page + 1) * BLOCK]
        want = unpermuted[0, src * BLOCK : (src + 1) * BLOCK]
        assert torch.equal(got, want)


def test_cache_write_skips_padding_slots() -> None:
    cache = _empty_cache(2, BLOCK)
    key = _rows((3, 1, D), seed=1)
    value = _rows((3, 1, D), seed=2)
    nv.reshape_and_cache_nvfp4_reference(key, value, cache, torch.tensor([-1, 5, -1]))
    touched = cache.reshape(2, -1).any(dim=1)
    assert touched.tolist() == [True, False]
    (k_data, _), _ = nv.nvfp4_kv_split_views(cache)
    assert k_data[0, 5].any() and not k_data[0, 4].any() and not k_data[0, 6].any()


def test_cache_write_skips_slots_past_the_cache_and_surplus_rows() -> None:
    # The store contract: a slot at or past num_blocks * block_size is skipped
    # (upstream's SM100 store does not bound-check it) and rows beyond the slot
    # mapping are padding.
    cache = _empty_cache(2, BLOCK)
    key = _rows((6, 1, D), seed=1)
    value = _rows((6, 1, D), seed=2)
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, torch.tensor([3, 2 * BLOCK, 10**6, -1])
    )
    (k_data, _), _ = nv.nvfp4_kv_split_views(cache)
    touched = k_data.reshape(2 * BLOCK, -1).any(dim=1)
    assert touched.nonzero().flatten().tolist() == [3]


def test_cache_write_last_writer_wins_for_a_repeated_slot() -> None:
    cache = _empty_cache(1, BLOCK)
    first = _rows((1, 1, D), seed=1)
    second = _rows((1, 1, D), seed=2)
    nv.reshape_and_cache_nvfp4_reference(
        torch.cat([first, second]),
        torch.cat([first, second]),
        cache,
        torch.tensor([7, 7]),
    )
    keys, _ = nv.gather_dequant_nvfp4_kv(
        cache,
        _single_request_table(1),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([[7]], dtype=torch.int32),
    )
    assert torch.equal(
        keys[0, 0], nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(second))[0]
    )


def test_cache_write_only_changes_the_written_slots() -> None:
    cache, slots, _, _ = _written_cache(6, 10)
    (k_data, v_data), (k_scale, v_scale) = nv.nvfp4_kv_split_views(cache)
    written = torch.zeros(6 * BLOCK, dtype=torch.bool)
    written[slots] = True
    for view in (k_data, v_data, k_scale, v_scale):
        touched = view.reshape(6 * BLOCK, -1).any(dim=1)
        assert (touched & ~written).sum() == 0


def test_cache_write_uses_the_layer_scales() -> None:
    cache, slots, key, value = _written_cache(4, 20, k_scale=0.5, v_scale=2.0)
    keys, values = nv.gather_dequant_nvfp4_kv(
        cache,
        _single_request_table(4),
        torch.zeros(1, dtype=torch.int32),
        slots.int().view(1, -1),
        k_scale=0.5,
        v_scale=2.0,
    )
    assert torch.equal(
        keys[0],
        nv.dequantize_kv_nvfp4(
            *nv.quantize_kv_nvfp4(key, layer_scale=0.5), layer_scale=0.5
        ),
    )
    assert torch.equal(
        values[0],
        nv.dequantize_kv_nvfp4(
            *nv.quantize_kv_nvfp4(value, layer_scale=2.0), layer_scale=2.0
        ),
    )
    # And against the original rows, independently of the codec under test.
    for restored, original in ((keys[0], key), (values[0], value)):
        relative = torch.linalg.norm(
            restored.float() - original.float()
        ) / torch.linalg.norm(original.float())
        assert relative < 0.12


def _gather(cache, indices, table=None, requests=None):
    table = _single_request_table(cache.shape[0]) if table is None else table
    requests = (
        torch.zeros(len(indices), dtype=torch.int32) if requests is None else requests
    )
    return nv.gather_dequant_nvfp4_kv(
        cache, table, requests, torch.tensor(indices, dtype=torch.int32)
    )


def _assert_positive_zero(x: torch.Tensor) -> None:
    assert not x.any()
    assert not torch.signbit(x).any()  # exact +0.0, never -0.0


def test_gather_zero_fills_negative_indices_with_positive_zero() -> None:
    cache, *_ = _written_cache(4, 30)
    keys, values = _gather(cache, [[-1, -7, -(2**30)]])
    _assert_positive_zero(keys)
    _assert_positive_zero(values)


def test_gather_zero_fills_pages_past_the_block_table() -> None:
    cache, *_ = _written_cache(4, 30)
    table = _single_request_table(2)  # only two logical pages are mapped
    keys, values = _gather(cache, [[2 * BLOCK, 3 * BLOCK + 5, 10**6]], table=table)
    _assert_positive_zero(keys)
    _assert_positive_zero(values)


@pytest.mark.parametrize("physical", [-1, 4, 99])
def test_gather_zero_fills_unmapped_and_out_of_range_blocks(physical) -> None:
    cache, *_ = _written_cache(4, 30)
    table = torch.tensor([[physical, 1]], dtype=torch.int32)
    keys, values = _gather(cache, [[0, 3, BLOCK - 1]], table=table)
    _assert_positive_zero(keys)
    _assert_positive_zero(values)


@pytest.mark.parametrize("request_id", [-1, 1, 5])
def test_gather_zero_fills_invalid_requests(request_id) -> None:
    cache, *_ = _written_cache(4, 30)
    keys, values = _gather(
        cache, [[0, 1, 2]], requests=torch.tensor([request_id], dtype=torch.int32)
    )
    _assert_positive_zero(keys)
    _assert_positive_zero(values)


def test_gather_zero_fills_interleaved_invalid_entries_only() -> None:
    num_blocks = 4
    cache, slots, _, _ = _written_cache(num_blocks, 30)
    valid = slots[:6].int().tolist()
    indices = [
        valid[0],
        -1,
        valid[1],
        10**6,
        valid[2],
        -5,
        valid[3],
        valid[4],
        -1,
        valid[5],
    ]
    keys, values = _gather(cache, [indices])
    solid, _ = _gather(cache, [valid])
    positions = [0, 2, 4, 6, 7, 9]
    assert torch.equal(keys[0, positions], solid[0])
    invalid = [1, 3, 5, 8]
    _assert_positive_zero(keys[0, invalid])
    _assert_positive_zero(values[0, invalid])


def test_gather_never_reads_the_poisoned_bytes_of_invalid_entries() -> None:
    cache = _empty_cache(4, BLOCK)
    cache.fill_(0xFF)  # every scale byte is an E4M3 NaN, every nibble is 0xF
    key = _rows((2, 1, D), seed=1)
    nv.reshape_and_cache_nvfp4_reference(key, key, cache, torch.tensor([0, 1]))
    # Slot 2 is poisoned and valid by the validity rule; -1 and 10**6 are not.
    keys, values = _gather(cache, [[0, -1, 10**6, 1]])
    assert torch.isfinite(keys[0, [0, 3]]).all()
    _assert_positive_zero(keys[0, [1, 2]])
    _assert_positive_zero(values[0, [1, 2]])
    # Reading a poisoned slot that the rule does admit returns NaN, which is
    # why the allocator must hand out zeroed blocks.
    poisoned, _ = _gather(cache, [[2]])
    assert torch.isnan(poisoned).all()


def test_gather_of_a_zeroed_slot_is_exactly_zero() -> None:
    cache = _empty_cache(2, BLOCK)
    keys, values = _gather(cache, [[0, 5, 17]])
    _assert_positive_zero(keys)
    _assert_positive_zero(values)


def test_gather_output_dtype_and_shape() -> None:
    cache, slots, *_ = _written_cache(4, 12, heads=2)
    keys, values = nv.gather_dequant_nvfp4_kv(
        cache,
        _single_request_table(4),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([[1, 2, 3, 4]] * 3, dtype=torch.int32),
        out_dtype=torch.float32,
    )
    assert keys.shape == values.shape == (3, 4, 2, D)
    assert keys.dtype == values.dtype == torch.float32


def test_gather_matches_a_naive_per_entry_implementation() -> None:
    generator = torch.Generator().manual_seed(7)
    num_blocks = 5
    cache, *_ = _written_cache(num_blocks, 60, heads=2, seed=4)
    table = torch.tensor([[2, 0, 4, -1], [1, 3, 9, 0]], dtype=torch.int32)
    rows, topk = 6, 24
    indices = torch.randint(-4, 4 * BLOCK + 8, (rows, topk), generator=generator).int()
    requests = torch.randint(-1, 3, (rows,), generator=generator).int()
    keys, values = nv.gather_dequant_nvfp4_kv(cache, table, requests, indices)
    (k_data, v_data), (k_scale, v_scale) = nv.nvfp4_kv_split_views(cache)
    for row in range(rows):
        for col in range(topk):
            token, request = int(indices[row, col]), int(requests[row])
            page, offset = divmod(token, BLOCK) if token >= 0 else (-1, 0)
            ok = token >= 0 and 0 <= request < 2 and page < table.shape[1]
            block = int(table[request, page]) if ok else -1
            ok = ok and 0 <= block < num_blocks
            if not ok:
                _assert_positive_zero(keys[row, col])
                _assert_positive_zero(values[row, col])
                continue
            want_k = nv.dequantize_kv_nvfp4(
                k_data[block, offset], k_scale[block, offset]
            )
            want_v = nv.dequantize_kv_nvfp4(
                v_data[block, offset], v_scale[block, offset]
            )
            assert torch.equal(keys[row, col], want_k)
            assert torch.equal(values[row, col], want_v)


# --------------------------------------------------------------------------- #
# The entry validity rule and the per-side views (plan 071 step D)
# --------------------------------------------------------------------------- #
def _validity_by_loops(indices, token_to_req, table, num_blocks, block_size):
    """The sparse kernel's rule, one entry at a time."""
    requests, width = table.shape
    valid = torch.zeros(indices.shape, dtype=torch.bool)
    for row in range(indices.shape[0]):
        request = int(token_to_req[row])
        for col in range(indices.shape[1]):
            token = int(indices[row, col])
            if request < 0 or request >= requests or token < 0:
                continue
            page = token // block_size
            if page >= width:
                continue
            block = int(table[request, page])
            valid[row, col] = 0 <= block < num_blocks
    return valid


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_entry_validity_equals_the_kernel_rule_written_as_loops(seed):
    generator = torch.Generator().manual_seed(seed)
    block_size, num_blocks, requests, width = 16, 7, 3, 4
    table = torch.randint(-2, num_blocks + 3, (requests, width), generator=generator)
    table = table.int()
    token_to_req = torch.randint(-2, requests + 2, (9,), generator=generator).int()
    indices = torch.randint(
        -5, width * block_size + 20, (9, 40), generator=generator
    ).int()
    indices[0, :4] = torch.tensor([2**30, -(2**30), 0, width * block_size - 1])
    got = nv.nvfp4_entry_validity(indices, token_to_req, table, num_blocks, block_size)
    expected = _validity_by_loops(indices, token_to_req, table, num_blocks, block_size)
    assert got.dtype == torch.bool and got.shape == indices.shape
    assert torch.equal(got, expected)
    assert expected.any() and not expected.all()  # both outcomes occur


def test_the_reference_gather_zero_fills_exactly_the_invalid_entries():
    generator = torch.Generator().manual_seed(7)
    blocks, block_size = 4, 8
    cache = _empty_cache(blocks, block_size)
    key, value = (
        _rows((blocks * block_size, 1, D), seed=1),
        _rows((blocks * block_size, 1, D), seed=2),
    )
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, torch.randperm(blocks * block_size, generator=generator)
    )
    table = torch.tensor([[0, 1, -1], [2, 9, 3]], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 1, -1], dtype=torch.int32)
    indices = torch.randint(-3, 3 * block_size + 4, (4, 12), generator=generator).int()
    keys, values = nv.gather_dequant_nvfp4_kv(cache, table, token_to_req, indices)
    valid = nv.nvfp4_entry_validity(indices, token_to_req, table, blocks, block_size)
    assert not keys[~valid].any() and not values[~valid].any()
    assert keys[valid].abs().sum() > 0


def test_side_views_are_the_views_of_the_five_dimensional_cache():
    cache = _empty_cache(5, 8, heads=2)
    cache.copy_(torch.randint(0, 256, cache.shape, dtype=torch.uint8))
    (k_data, v_data), (k_scale, v_scale) = nv.nvfp4_kv_split_views(cache)
    for side, data, scale in (
        (cache[:, 0], k_data, k_scale),
        (cache[:, 1], v_data, v_scale),
    ):
        got_data, got_scale = nv.nvfp4_side_views(side)
        assert got_data.dtype == got_scale.dtype == torch.uint8
        assert got_data.shape == data.shape and got_scale.shape == scale.shape
        assert got_data.data_ptr() == data.data_ptr()
        assert got_scale.data_ptr() == scale.data_ptr()
        assert (
            got_data.stride() == data.stride() and got_scale.stride() == scale.stride()
        )
        assert torch.equal(got_data, data) and torch.equal(got_scale, scale)


def test_side_views_reject_what_is_not_one_uint8_side():
    cache = _empty_cache(2, 4)
    with pytest.raises(ValueError, match="must be"):
        nv.nvfp4_side_views(cache)
    with pytest.raises(TypeError, match="uint8"):
        nv.nvfp4_side_views(cache[:, 0].to(torch.int16))
