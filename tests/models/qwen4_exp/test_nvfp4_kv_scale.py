# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-layer global scale of the NVFP4 QSA KV cache (plan 071 stage 1A).

The semantics are those of the fleet's SGLang implementation: the stored value
is ``e2m1 * block_scale * layer_scale``, the layer scale is the checkpoint's
``k_scale``/``v_scale`` when it carries one and 1.0 otherwise (SGLang's
acceptance runs used 1.0 because that checkpoint has no KV scales). E4M3
refuses to start without a calibrated overlay; NVFP4 never does. A layer scale
large enough to push the block scales into the E4M3 subnormal range is
rejected when it is loaded.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    # Importing the QSA backend asks the GPU for a FlashAttention version (see
    # test_nvfp4_kv_admission.py); Volta runs FA2.
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from vllm.model_executor.layers.attention.attention import (  # noqa: E402
    set_default_quant_scales,
)
from vllm.models.qwen4_exp.nvidia import model as model_mod  # noqa: E402
from vllm.models.qwen4_exp.nvidia.model import (  # noqa: E402
    _finalize_qsa_e4m3_scale_load,
    _validate_qsa_e4m3_scale_load,
)
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv  # noqa: E402
from vllm.models.qwen4_exp.nvidia.qsa import (  # noqa: E402
    Qwen4ExpQSAAttention,
    _qsa_loads_kv_scales,
)

D = 256
NVFP4 = "nvfp4"
E4M3 = "fp8_e4m3"
# The K and V scalars of the E4M3 overlay shipped with the Flash-Next checkpoint.
OVERLAY_K = (0.0186, 0.0213623046875, 0.0255998894572258, 0.0367257259786129)
OVERLAY_V = (0.0171, 0.0404924675822258, 0.0677315890789032, 0.0841238871216774)


def _rows(shape, seed=0, dtype=torch.float16, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


# --------------------------------------------------------------------------- #
# Reference helpers and the subnormal-range guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1.0, 0.0001, *OVERLAY_K, *OVERLAY_V, 10.0])
def test_a_plausible_layer_scale_is_accepted(scale) -> None:
    assert nv.check_nvfp4_layer_scale(scale) == scale
    assert isinstance(nv.check_nvfp4_layer_scale(torch.tensor(scale)), float)


@pytest.mark.parametrize(
    "scale", [0.0, -0.5, float("nan"), float("inf"), -float("inf")]
)
def test_a_non_finite_or_non_positive_layer_scale_is_rejected(scale) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        nv.check_nvfp4_layer_scale(scale)


@pytest.mark.parametrize("scale", [10.0001, 11.0, 300.0])
def test_a_layer_scale_in_the_subnormal_range_is_rejected(scale) -> None:
    with pytest.raises(ValueError, match="subnormal"):
        nv.check_nvfp4_layer_scale(scale)


def test_the_scale_limit_is_below_the_bound_it_protects() -> None:
    # For amax = 1 the block scale amax / (6 * s) drops below 2**-6 at s = 10.67.
    bound = 1.0 / (nv.E2M1_MAX * nv.E4M3_MIN_NORMAL)
    assert bound > nv.NVFP4_KV_LAYER_SCALE_MAX
    assert bound == pytest.approx(10.6667, rel=1e-4)
    assert nv.E4M3_MIN_NORMAL == 2.0**-6


def test_every_overlay_scalar_stays_in_the_normal_range_with_room() -> None:
    # If the overlay is amax / 448, the largest block scale is about 74.7.
    for scale in (*OVERLAY_K, *OVERLAY_V):
        largest, floor = nv.nvfp4_block_scale_range(scale, scale * 448.0)
        assert largest == pytest.approx(448.0 / 6.0)
        assert floor == pytest.approx(6.0 * scale * 2.0**-6)


def test_the_derived_scale_puts_the_largest_block_scale_at_the_e4m3_maximum() -> None:
    for amax in (3.0, 9.57, 37.7, 2688.0):
        scale = nv.nvfp4_layer_scale_from_amax(amax)
        assert scale == pytest.approx(amax / 2688.0)
        largest, _ = nv.nvfp4_block_scale_range(scale, amax)
        assert largest == pytest.approx(448.0)


def test_the_derived_scale_never_saturates_a_block_scale() -> None:
    x = _rows((512, D), seed=1, scale=4.0)
    amax = float(x.float().abs().max())
    scale = nv.nvfp4_layer_scale_from_amax(amax)
    _, scales = nv.quantize_kv_nvfp4(x, layer_scale=scale)
    decoded = nv.e4m3_decode(scales)
    assert decoded.max().item() <= 448.0
    assert decoded.max().item() >= 440.0  # the largest group sits at the top


@pytest.mark.parametrize("amax", [0.0, -1.0, float("nan"), float("inf")])
def test_the_derived_scale_rejects_a_bad_amax(amax) -> None:
    with pytest.raises(ValueError, match="amax"):
        nv.nvfp4_layer_scale_from_amax(amax)


def test_block_scale_range_rejects_an_implausible_scale() -> None:
    with pytest.raises(ValueError, match="subnormal"):
        nv.nvfp4_block_scale_range(300.0, 10.0)


def test_the_subnormal_fraction_is_zero_for_ordinary_data_and_one_when_oversized() -> (
    None
):
    x = _rows((256, D), seed=2)
    assert nv.nvfp4_subnormal_group_fraction(x, 1.0) == 0.0
    assert nv.nvfp4_subnormal_group_fraction(x, 300.0) == 1.0


def test_the_subnormal_fraction_grows_with_the_layer_scale() -> None:
    # Groups spread over four decades: the larger the scale, the more of them
    # have a block scale below 2**-6.
    generator = torch.Generator().manual_seed(3)
    magnitude = 10.0 ** (torch.rand(2048, 1, generator=generator) * -4)
    x = (torch.randn(2048, D, generator=generator) * magnitude).half()
    fractions = [nv.nvfp4_subnormal_group_fraction(x, s) for s in (0.01, 1.0, 8.0)]
    assert fractions == sorted(fractions) and fractions[0] < fractions[-1]
    # The derived scale from the data's amax leaves fewer subnormal groups than 1.0.
    derived = nv.nvfp4_layer_scale_from_amax(float(x.float().abs().max()))
    assert nv.nvfp4_subnormal_group_fraction(x, derived) < fractions[1]


def test_the_subnormal_fraction_ignores_all_zero_groups() -> None:
    assert nv.nvfp4_subnormal_group_fraction(torch.zeros(4, D), 1.0) == 0.0
    x = torch.zeros(1, D)
    x[0, :16] = 0.001  # one non-zero group with a subnormal block scale
    assert nv.nvfp4_subnormal_group_fraction(x, 1.0) == 1.0


# --------------------------------------------------------------------------- #
# Parity cases ported from the SGLang QSA NVFP4 patch tests
# --------------------------------------------------------------------------- #
def test_a_host_scale_and_a_one_element_tensor_write_identical_bytes() -> None:
    # SGLang test_qsa_nvfp4_gather.py::scale_source_equivalence.
    x = _rows((512, 1, D), seed=4, dtype=torch.bfloat16)
    host = nv.quantize_kv_nvfp4(x, layer_scale=1.0)
    device = nv.quantize_kv_nvfp4(x, layer_scale=torch.ones(1, dtype=torch.float32))
    assert torch.equal(host[0], device[0])
    assert torch.equal(host[1].view(torch.uint8), device[1].view(torch.uint8))


def test_the_unit_scale_is_the_default_of_the_writer_and_the_reader() -> None:
    x = _rows((16, D), seed=5)
    packed, scales = nv.quantize_kv_nvfp4(x)
    explicit = nv.quantize_kv_nvfp4(x, layer_scale=1.0)
    assert torch.equal(packed, explicit[0])
    restored = nv.dequantize_kv_nvfp4(packed, scales)
    assert torch.equal(
        restored, nv.dequantize_kv_nvfp4(packed, scales, layer_scale=1.0)
    )


@pytest.mark.parametrize(
    ("topk", "batch", "poison"), [(2051, 3, False), (2051, 4, True), (128, 2, True)]
)
def test_the_sglang_pool_scenario_valid_rows_exact_and_invalid_tail_zero(
    topk, batch, poison
) -> None:
    # The shapes of SGLang's run_case: one KV head of 256, global scale 1.0, K and
    # V three times a Gaussian, a valid prefix of each row then -1 entries.
    block, num_blocks = 64, 128
    generator = torch.Generator().manual_seed(6)
    cache = torch.zeros((num_blocks, 2, block, 1, 144), dtype=torch.uint8)
    key = (torch.randn((num_blocks * block, 1, D), generator=generator) * 3).half()
    value = (torch.randn((num_blocks * block, 1, D), generator=generator) * 3).half()
    if poison:
        cache.fill_(0xFF)  # every scale byte is an E4M3 NaN until it is written
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, torch.arange(num_blocks * block), k_scale=1.0, v_scale=1.0
    )
    indices = torch.full((batch, topk), -1, dtype=torch.int32)
    valid = []
    for row in range(batch):
        count = int(
            torch.randint(1, min(topk, num_blocks * block), (1,), generator=generator)
        )
        indices[row, :count] = torch.randperm(num_blocks * block, generator=generator)[
            :count
        ].int()
        valid.append(count)
    keys, values = nv.gather_dequant_nvfp4_kv(
        cache,
        torch.arange(num_blocks, dtype=torch.int32).view(1, -1),
        torch.zeros(batch, dtype=torch.int32),
        indices,
    )
    expected_k = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(key))
    expected_v = nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(value))
    assert torch.isfinite(keys.float()).all() and torch.isfinite(values.float()).all()
    for row, count in enumerate(valid):
        slots = indices[row, :count].long()
        assert torch.equal(keys[row, :count], expected_k[slots])
        assert torch.equal(values[row, :count], expected_v[slots])
        tail = keys[row, count:].float(), values[row, count:].float()
        assert all((t == 0).all() and not torch.signbit(t).any() for t in tail)


# --------------------------------------------------------------------------- #
# The owner's scale slots and the checkpoint loader
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("fp8", True),
        (E4M3, True),
        (NVFP4, True),
        ("auto", False),
        ("float16", False),
        ("bfloat16", False),
    ],
)
def test_the_cache_dtypes_that_read_layer_scales(dtype, expected) -> None:
    assert _qsa_loads_kv_scales(dtype) is expected


def _owner(k_value, v_value, *, dtype=NVFP4):
    """A real owner with only the scale slots populated, as test_e4m3_mtp does."""
    stub = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(stub)
    stub.kv_cache_dtype = dtype
    stub.layer_name = "model.layers.3.self_attn"
    stub._qsa_kv_scales_finalized = not _qsa_loads_kv_scales(dtype)
    set_default_quant_scales(stub, register_buffer=True)  # as the real __init__
    if not stub._qsa_kv_scales_finalized:
        stub.k_scale = nn.Parameter(torch.tensor(float(k_value)), requires_grad=False)
        stub.v_scale = nn.Parameter(torch.tensor(float(v_value)), requires_grad=False)
    return stub


def test_an_nvfp4_owner_waits_for_its_scales_like_an_e4m3_one() -> None:
    assert _owner(-1.0, -1.0)._qsa_kv_scales_finalized is False
    assert _owner(0.0, 0.0, dtype="float16")._qsa_kv_scales_finalized is True


@pytest.mark.parametrize(("k", "v"), list(zip(OVERLAY_K, OVERLAY_V)))
def test_loaded_overlay_scalars_become_the_layer_scales(k, v) -> None:
    owner = _owner(k, v)
    owner.validate_loaded_kv_scales()
    assert owner._qsa_kv_scales_finalized is True
    assert owner._k_scale_float == pytest.approx(k)
    assert owner._v_scale_float == pytest.approx(v)
    assert float(owner._k_scale) == pytest.approx(k)
    assert not hasattr(owner, "k_scale") and not hasattr(owner, "v_scale")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_an_invalid_loaded_scalar_is_rejected(bad) -> None:
    for k, v in ((bad, 0.02), (0.02, bad)):
        with pytest.raises(ValueError, match="invalid"):
            _owner(k, v).validate_loaded_kv_scales()


@pytest.mark.parametrize("which", ["K", "V"])
def test_an_oversized_loaded_scalar_is_rejected_with_the_layer_name(which) -> None:
    k, v = (300.0, 0.02) if which == "K" else (0.02, 300.0)
    with pytest.raises(
        ValueError, match=rf"NVFP4 {which} scale of model.layers.3.*subnormal"
    ):
        _owner(k, v).validate_loaded_kv_scales()


def test_the_loaded_scalar_limit_is_inclusive_at_ten() -> None:
    _owner(10.0, 10.0).validate_loaded_kv_scales()
    with pytest.raises(ValueError, match="subnormal"):
        _owner(10.5, 1.0).validate_loaded_kv_scales()


def test_e4m3_does_not_get_the_nvfp4_limit() -> None:
    owner = _owner(300.0, 300.0, dtype=E4M3)  # fp8 scales are not capped here
    owner.validate_loaded_kv_scales()
    assert owner._k_scale_float == 300.0


def test_the_default_scale_of_an_nvfp4_owner_is_one() -> None:
    owner = _owner(-1.0, -1.0)
    owner.adopt_default_kv_scales()
    assert owner._qsa_kv_scales_finalized is True
    assert owner._k_scale_float == owner._v_scale_float == 1.0
    assert not hasattr(owner, "k_scale")


class _Layer(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.self_attn = owner


class _Model(nn.Module):
    def __init__(self, *owners):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(owner) for owner in owners])


def _names(*indices, kinds=("k", "v")):
    return {f"layers.{i}.self_attn.{kind}_scale" for i in indices for kind in kinds}


@pytest.fixture(autouse=True)
def _not_the_offload_process(monkeypatch):
    monkeypatch.setattr(model_mod, "is_offload_process", lambda: False)


def test_the_loader_takes_every_checkpoint_scalar_for_nvfp4() -> None:
    first, second = _owner(0.0213, 0.0405), _owner(0.0256, 0.0319)
    _finalize_qsa_e4m3_scale_load(_Model(first, second), _names(0, 1), NVFP4)
    assert first._k_scale_float == pytest.approx(0.0213)
    assert second._v_scale_float == pytest.approx(0.0319)
    assert first._qsa_kv_scales_finalized and second._qsa_kv_scales_finalized


def test_the_loader_uses_the_unit_scale_when_the_checkpoint_has_none() -> None:
    first, second = _owner(-1.0, -1.0), _owner(-1.0, -1.0)
    _finalize_qsa_e4m3_scale_load(_Model(first, second), set(), NVFP4)
    for owner in (first, second):
        assert owner._qsa_kv_scales_finalized is True
        assert owner._k_scale_float == owner._v_scale_float == 1.0


def test_nvfp4_never_demands_calibration_unlike_e4m3() -> None:
    owner = _owner(-1.0, -1.0)
    _finalize_qsa_e4m3_scale_load(
        _Model(owner),
        set(),
        NVFP4,
        require_calibrated_target=True,
        require_calibrated_speculative_draft=True,
    )
    assert owner._k_scale_float == 1.0
    with pytest.raises(ValueError, match="requires complete calibrated"):
        _finalize_qsa_e4m3_scale_load(
            _Model(_owner(-1.0, -1.0, dtype=E4M3)),
            set(),
            E4M3,
            require_calibrated_target=True,
        )


def test_a_partly_loaded_overlay_mixes_checkpoint_and_unit_scales() -> None:
    loaded, unloaded = _owner(0.03, 0.05), _owner(-1.0, -1.0)
    _finalize_qsa_e4m3_scale_load(_Model(loaded, unloaded), _names(0), NVFP4)
    assert loaded._k_scale_float == pytest.approx(0.03)
    assert unloaded._k_scale_float == unloaded._v_scale_float == 1.0


def test_a_layer_with_only_one_loaded_scalar_falls_back_to_the_unit_scale() -> None:
    owner = _owner(0.03, -1.0)
    _finalize_qsa_e4m3_scale_load(_Model(owner), _names(0, kinds=("k",)), NVFP4)
    assert owner._k_scale_float == owner._v_scale_float == 1.0


def test_the_loader_still_rejects_an_oversized_nvfp4_scalar() -> None:
    with pytest.raises(ValueError, match="subnormal"):
        _finalize_qsa_e4m3_scale_load(_Model(_owner(300.0, 0.02)), _names(0), NVFP4)


def test_the_loader_skips_the_offload_process(monkeypatch) -> None:
    monkeypatch.setattr(model_mod, "is_offload_process", lambda: True)
    owner = _owner(-1.0, -1.0)
    _finalize_qsa_e4m3_scale_load(_Model(owner), set(), NVFP4)
    assert owner._qsa_kv_scales_finalized is False


def test_the_scale_check_reports_missing_names_only_for_nvfp4_as_a_set() -> None:
    required = _names(0, 1)
    assert _validate_qsa_e4m3_scale_load(required, set(), NVFP4) == required
    assert _validate_qsa_e4m3_scale_load(required, required, NVFP4) == set()
    assert _validate_qsa_e4m3_scale_load(required, set(), "float16") == set()


def test_an_unfinalized_nvfp4_owner_cannot_run_until_it_adopts_its_scales() -> None:
    owner = _owner(-1.0, -1.0)
    with pytest.raises(RuntimeError, match="were not finalized"):
        owner._run_qsa(None, None, None, None, None, None)
    owner.adopt_default_kv_scales()
    # Finalized: the scale check passes. A DCP-sharded cache is the next refusal.
    owner.qsa_dcp_sharded = True
    with pytest.raises(NotImplementedError, match="DCP"):
        owner._run_qsa(None, None, None, None, None, None)


def test_layer_scales_reach_the_reference_writer_and_reader() -> None:
    # What the loader finalizes is what the writer and the reader are given.
    owner = _owner(0.0213623046875, 0.0404924675822258)
    owner.validate_loaded_kv_scales()
    key, value = _rows((32, 1, D), seed=7), _rows((32, 1, D), seed=8)
    cache = torch.zeros((2, 2, 16, 1, 144), dtype=torch.uint8)
    nv.reshape_and_cache_nvfp4_reference(
        key,
        value,
        cache,
        torch.arange(32),
        k_scale=owner._k_scale_float,
        v_scale=owner._v_scale_float,
    )
    keys, values = nv.gather_dequant_nvfp4_kv(
        cache,
        torch.arange(2, dtype=torch.int32).view(1, -1),
        torch.zeros(1, dtype=torch.int32),
        torch.arange(32, dtype=torch.int32).view(1, -1),
        k_scale=owner._k_scale_float,
        v_scale=owner._v_scale_float,
    )
    for restored, original in ((keys[0], key), (values[0], value)):
        relative = torch.linalg.norm(
            restored.float() - original.float()
        ) / torch.linalg.norm(original.float())
        assert relative < 0.12
    assert math.isfinite(owner._k_scale_float)
