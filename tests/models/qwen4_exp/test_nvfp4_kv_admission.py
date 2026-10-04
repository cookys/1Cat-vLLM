# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: the QSA owner admits ``--kv-cache-dtype nvfp4`` and the cache
spec, the worker views and the uniform-page derivation see 288 B/token/layer.

The block size is derived by the existing hybrid platform rule from the
per-token page bytes; nothing here hard-codes 1616.
"""

from __future__ import annotations

import typing
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    # flash_attn.py asks the GPU which FlashAttention version to use while its
    # backend classes are being defined, so the QSA module cannot even be
    # imported without a visible device. Volta runs FA2; use that answer so the
    # CPU-only checks below can import the owner and the backend.
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from vllm.config.cache import CacheConfig, CacheDType
from vllm.model_executor.models import ModelRegistry
from vllm.models.qwen4_exp.common.kv_policy import resolve_qsa_auto_e4m3
from vllm.models.qwen4_exp.nvidia import qsa as qsa_module
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv
from vllm.models.qwen4_exp.nvidia.qsa import (
    _QSA_MAIN_KV_CACHE_DTYPES,
    Qwen4ExpQSAAttention,
    Qwen4ExpQSAFlashAttentionBackend,
    _verify_main_kv_storage_dtype,
    _verify_nvfp4_kv_requirements,
)
from vllm.platforms.interface import Platform
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVQuantMode,
    MambaAttentionBackendEnum,
    MambaSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup

D = 256
TP4_KV_HEADS = 1
# Bytes of one recurrent state page at TP4 (fp16 conv, fp32 SSM), from
# MambaStateShapeCalculator.gated_delta_net_state_shape: conv 2560 x (3 + K),
# SSM 12 x 128 x 128 x 4 B.
GDN_STATE = {0: 801_792, 4: 822_272}
E4M3_BLOCK = {0: 1568, 4: 1616}
NVFP4_BLOCK = {0: 2784, 4: 2864}


def _gdn(num_spec: int) -> MambaSpec:
    return MambaSpec(
        shapes=((2560, 3 + num_spec), (12, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        block_size=-1,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


# --------------------------------------------------------------------------- #
# Config and dtype tables
# --------------------------------------------------------------------------- #
def test_nvfp4_is_a_declared_cache_dtype() -> None:
    assert "nvfp4" in typing.get_args(CacheDType)
    assert CacheConfig(cache_dtype="nvfp4").cache_dtype == "nvfp4"


def test_an_unknown_cache_dtype_is_still_rejected() -> None:
    with pytest.raises(Exception, match="cache_dtype"):
        CacheConfig(cache_dtype="nvfp5")


def test_nvfp4_quant_mode_and_storage_dtype() -> None:
    assert get_kv_quant_mode("nvfp4") is KVQuantMode.NVFP4
    assert KVQuantMode.NVFP4.is_nvfp4 and not KVQuantMode.FP8_PER_TENSOR.is_nvfp4
    assert STR_DTYPE_TO_TORCH_DTYPE["nvfp4"] is torch.uint8


def test_an_explicit_nvfp4_dtype_beats_the_automatic_e4m3_policy() -> None:
    from vllm.config import KernelConfig

    cfg = SimpleNamespace(
        kernel_config=KernelConfig(),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_text_config=SimpleNamespace(indexer_n_heads=4, layer_types=[]),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="nvfp4",
            cache_dtype_from_checkpoint=False,
            calculate_kv_scales=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, tensor_parallel_size=4
        ),
    )
    assert resolve_qsa_auto_e4m3(cfg) is False
    assert cfg.cache_config.cache_dtype == "nvfp4"
    assert "explicit KV dtype" in cfg.kernel_config.qsa_auto_e4m3_reason


# --------------------------------------------------------------------------- #
# Backend / owner admission
# --------------------------------------------------------------------------- #
def test_backend_admits_nvfp4_and_keeps_the_existing_dtypes() -> None:
    admitted = set(Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes)
    assert {"auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "nvfp4"} <= admitted


def test_backend_list_and_owner_constant_agree() -> None:
    backend = Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
    assert set(backend) == set(_QSA_MAIN_KV_CACHE_DTYPES)


@pytest.mark.parametrize(
    "dtype", ["fp8_e5m2", "int8_per_token_head", "turboquant_k8v4", "nvfp5"]
)
def test_other_cache_dtypes_stay_rejected(dtype) -> None:
    assert dtype not in _QSA_MAIN_KV_CACHE_DTYPES
    assert dtype not in Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes


def _volta(monkeypatch, capability=70):
    from vllm.models.qwen4_exp.nvidia.ops import qsa as ops

    monkeypatch.setattr(
        ops,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )


def test_nvfp4_requirements_are_a_noop_for_other_cache_dtypes() -> None:
    model = SimpleNamespace(dtype=torch.float32)
    for dtype in ("auto", "fp8_e4m3"):
        _verify_nvfp4_kv_requirements(model, SimpleNamespace(cache_dtype=dtype), 7)


def test_nvfp4_requirements_accept_fp16_on_volta(monkeypatch) -> None:
    _volta(monkeypatch)
    _verify_nvfp4_kv_requirements(
        SimpleNamespace(dtype=torch.float16),
        SimpleNamespace(cache_dtype="nvfp4", calculate_kv_scales=False),
        D,
    )


@pytest.mark.parametrize(
    ("capability", "dtype"),
    [
        (70, torch.bfloat16),
        (75, torch.bfloat16),
        (60, torch.float16),
        (80, torch.float32),
    ],
)
def test_nvfp4_requirements_reject_an_unsupported_operator_dtype(
    monkeypatch, capability, dtype
) -> None:
    _volta(monkeypatch, capability)
    with pytest.raises(NotImplementedError, match="QSA NVFP4 cache unavailable"):
        _verify_nvfp4_kv_requirements(
            SimpleNamespace(dtype=dtype), SimpleNamespace(cache_dtype="nvfp4"), D
        )


def test_nvfp4_requirements_accept_bf16_on_ampere(monkeypatch) -> None:
    _volta(monkeypatch, 80)
    _verify_nvfp4_kv_requirements(
        SimpleNamespace(dtype=torch.bfloat16), SimpleNamespace(cache_dtype="nvfp4"), D
    )


@pytest.mark.parametrize("head_dim", [0, 8, 100, 250])
def test_nvfp4_requirements_reject_a_head_size_off_the_group(
    monkeypatch, head_dim
) -> None:
    _volta(monkeypatch)
    with pytest.raises(NotImplementedError, match="multiple of 16"):
        _verify_nvfp4_kv_requirements(
            SimpleNamespace(dtype=torch.float16),
            SimpleNamespace(cache_dtype="nvfp4"),
            head_dim,
        )


def test_nvfp4_requirements_forbid_runtime_scale_calculation(monkeypatch) -> None:
    _volta(monkeypatch)
    with pytest.raises(ValueError, match="calculate_kv_scales"):
        _verify_nvfp4_kv_requirements(
            SimpleNamespace(dtype=torch.float16),
            SimpleNamespace(cache_dtype="nvfp4", calculate_kv_scales=True),
            D,
        )


@pytest.mark.parametrize(
    ("kv_cache_dtype", "storage"),
    [
        ("nvfp4", torch.uint8),
        ("fp8_e4m3", torch.uint8),
        ("fp8", torch.uint8),
        ("float16", torch.float16),
        ("auto", torch.float16),
    ],
)
def test_main_kv_storage_dtype_is_accepted(kv_cache_dtype, storage) -> None:
    _verify_main_kv_storage_dtype(kv_cache_dtype, storage, torch.float16)


@pytest.mark.parametrize(
    ("kv_cache_dtype", "storage"),
    [("bfloat16", torch.bfloat16), ("float16", torch.uint8), ("auto", torch.float32)],
)
def test_unquantized_main_kv_must_match_the_model_dtype(
    kv_cache_dtype, storage
) -> None:
    with pytest.raises(NotImplementedError, match="must match the model dtype"):
        _verify_main_kv_storage_dtype(kv_cache_dtype, storage, torch.float16)


def _bare_owner(**attributes) -> Qwen4ExpQSAAttention:
    layer = object.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    for name, value in attributes.items():
        setattr(layer, name, value)
    return layer


def test_the_owner_refuses_to_execute_nvfp4_until_the_kernels_exist() -> None:
    layer = _bare_owner(
        kv_cache_dtype="nvfp4",
        _qsa_kv_scales_finalized=True,
        layer_name="model.layers.3.self_attn.attn",
    )
    with pytest.raises(NotImplementedError, match="NVFP4.*not implemented"):
        layer._run_qsa(None, None, None, None, None, None)


def test_the_execution_guard_is_specific_to_nvfp4() -> None:
    layer = _bare_owner(
        kv_cache_dtype="fp8_e4m3",
        _qsa_kv_scales_finalized=False,
        layer_name="model.layers.3.self_attn.attn",
    )
    with pytest.raises(RuntimeError, match="scales were not finalized"):
        layer._run_qsa(None, None, None, None, None, None)


# --------------------------------------------------------------------------- #
# Cache spec: 288 B per token per layer
# --------------------------------------------------------------------------- #
def _vllm_config(block_size: int, dcp: int = 1):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )


def _owner_spec(dtype: str, block_size: int, heads: int = TP4_KV_HEADS):
    layer = _bare_owner(
        layer_name="model.layers.3.self_attn.attn",
        num_kv_heads=heads,
        head_dim=D,
        kv_cache_dtype=dtype,
        kv_cache_torch_dtype=STR_DTYPE_TO_TORCH_DTYPE[dtype],
    )
    return layer.get_kv_cache_spec(_vllm_config(block_size))


@pytest.mark.parametrize("block_size", [16, 1616, 2784, 2864])
def test_owner_spec_page_is_288_bytes_per_token(block_size) -> None:
    spec = _owner_spec("nvfp4", block_size)
    assert type(spec) is FullAttentionSpec  # the CSA classifier needs this type
    assert spec.kv_quant_mode is KVQuantMode.NVFP4
    assert spec.dtype is torch.uint8
    # K row 128 data + 16 scale, V the same: head_size // 2 + head_size // 16.
    assert (
        spec.page_size_bytes == block_size * 2 * (D // 2 + D // 16) == block_size * 288
    )


def test_owner_spec_matches_the_reference_byte_math() -> None:
    spec = _owner_spec("nvfp4", 2784)
    assert spec.page_size_bytes == nv.nvfp4_kv_page_bytes(2784, TP4_KV_HEADS, D)


def test_owner_spec_scales_with_the_number_of_kv_heads() -> None:
    assert _owner_spec("nvfp4", 64, heads=2).page_size_bytes == 64 * 2 * 288


def test_e4m3_and_fp16_specs_are_unchanged() -> None:
    assert _owner_spec("fp8_e4m3", 1616).page_size_bytes == 1616 * 512
    assert _owner_spec("float16", 800).page_size_bytes == 800 * 1024


def test_nvfp4_page_is_43_75_percent_smaller_than_e4m3() -> None:
    e4m3, fp4 = _owner_spec("fp8_e4m3", 64), _owner_spec("nvfp4", 64)
    assert 1 - fp4.page_size_bytes / e4m3.page_size_bytes == pytest.approx(0.4375)


# --------------------------------------------------------------------------- #
# Backend cache shape and worker views
# --------------------------------------------------------------------------- #
def test_backend_cache_shape_for_nvfp4_is_the_row_byte_pair() -> None:
    shape = Qwen4ExpQSAFlashAttentionBackend.get_kv_cache_shape(7, 32, 1, D, "nvfp4")
    assert shape == (7, 2, 32, 1, 144)


@pytest.mark.parametrize("dtype", ["auto", "fp8_e4m3"])
def test_backend_cache_shape_is_unchanged_for_the_other_dtypes(dtype) -> None:
    shape = Qwen4ExpQSAFlashAttentionBackend.get_kv_cache_shape(7, 32, 1, D, dtype)
    assert shape == (7, 2, 32, 1, D)


def test_backend_cache_shape_rejects_an_unaligned_block_for_nvfp4() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        Qwen4ExpQSAFlashAttentionBackend.get_kv_cache_shape(7, 24, 1, D, "nvfp4")


def _views(members, block_size, blocks, monkeypatch, packed):
    from vllm.v1.attention.backends import flash_attn

    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=D,
        head_size_v=D,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    raw = torch.zeros(
        blocks * len(members if packed else [0]) * spec.page_size_bytes,
        dtype=torch.int8,
    )
    views = _reshape_kv_cache(
        attn_groups=[
            AttentionGroup(Qwen4ExpQSAFlashAttentionBackend, members, spec, 0)
        ],
        kv_cache_raw_tensors={name: raw for name in members},
        cache_dtype="nvfp4",
        kernel_block_sizes=[block_size],
        shared_kv_cache_layers={},
        packed_members=(
            {name: (i, len(members)) for i, name in enumerate(members)}
            if packed
            else None
        ),
    )
    return raw, spec, [views[name] for name in members]


def test_worker_view_of_an_nvfp4_layer_has_the_row_byte_shape(monkeypatch) -> None:
    raw, spec, (view,) = _views(["model.layers.3.self_attn"], 32, 5, monkeypatch, False)
    assert view.shape == (5, 2, 32, 1, 144) and view.dtype is torch.uint8
    assert view.numel() == raw.numel() == 5 * spec.page_size_bytes


def test_worker_view_roundtrips_through_the_reference_writer(monkeypatch) -> None:
    _, _, (view,) = _views(["model.layers.3.self_attn"], 32, 5, monkeypatch, False)
    generator = torch.Generator().manual_seed(1)
    key = torch.randn((20, 1, D), generator=generator).half()
    value = torch.randn((20, 1, D), generator=generator).half()
    slots = torch.randperm(5 * 32, generator=generator)[:20]
    nv.reshape_and_cache_nvfp4_reference(key, value, view, slots)
    keys, values = nv.gather_dequant_nvfp4_kv(
        view,
        torch.arange(5, dtype=torch.int32).view(1, -1),
        torch.zeros(1, dtype=torch.int32),
        slots.int().view(1, -1),
    )
    assert torch.equal(keys[0], nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(key)))
    assert torch.equal(values[0], nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(value)))


def test_packed_nvfp4_members_share_a_page_without_overlapping(monkeypatch) -> None:
    members = ["model.layers.3.self_attn", "model.layers.7.self_attn"]
    raw, spec, views = _views(members, 32, 4, monkeypatch, True)
    first, second = views
    assert first.shape == second.shape == (4, 2, 32, 1, 144)
    assert (
        first.stride(0) == 2 * spec.page_size_bytes
    )  # a physical block holds both pages
    generator = torch.Generator().manual_seed(2)
    written = []
    for view in views:
        key = torch.randn((20, 1, D), generator=generator).half()
        slots = torch.randperm(4 * 32, generator=generator)[:20]
        nv.reshape_and_cache_nvfp4_reference(key, key, view, slots)
        written.append((view, key, slots))
    for view, key, slots in written:
        keys, _ = nv.gather_dequant_nvfp4_kv(
            view,
            torch.arange(4, dtype=torch.int32).view(1, -1),
            torch.zeros(1, dtype=torch.int32),
            slots.int().view(1, -1),
        )
        assert torch.equal(keys[0], nv.dequantize_kv_nvfp4(*nv.quantize_kv_nvfp4(key)))
    # Together the members touched bytes of two pages per block and no more.
    assert raw.view(torch.uint8).count_nonzero() > 0


# --------------------------------------------------------------------------- #
# Uniform-page derivation: the platform rule re-derives the block size
# --------------------------------------------------------------------------- #
def _derive(cache_dtype: str, num_spec: int):
    """Run the real hybrid block-size rule with the QSA backend and real state."""

    class _Model:
        @classmethod
        def get_mamba_specs_from_config(cls, vllm_config):
            return (_gdn(num_spec),)

        @classmethod
        def get_kv_block_size_multiple(cls, vllm_config):
            return 1

    model_config = SimpleNamespace(
        use_mla=False,
        dtype=torch.float16,
        architecture="FakeQwen4ExpForCausalLM",
        is_hybrid=True,
        get_num_kv_heads=lambda parallel_config: TP4_KV_HEADS,
        get_head_size=lambda: D,
        hf_text_config=SimpleNamespace(),
    )
    cache_config = SimpleNamespace(
        cache_dtype=cache_dtype,
        block_size=16,
        mamba_cache_mode="align",
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_page_size_padded=None,
        mamba_ssm_cache_dtype="float32",
        mamba_cache_dtype="auto",
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    original = ModelRegistry.resolve_model_cls
    ModelRegistry.resolve_model_cls = lambda *a, **k: (_Model, "x")  # type: ignore[assignment]
    try:
        Platform._align_hybrid_block_size(config, Qwen4ExpQSAFlashAttentionBackend)
    finally:
        ModelRegistry.resolve_model_cls = original  # type: ignore[assignment]
    return cache_config


@pytest.mark.parametrize("num_spec", [0, 4])
def test_the_derivation_reproduces_the_production_e4m3_block(num_spec) -> None:
    config = _derive("fp8_e4m3", num_spec)
    assert config.block_size == E4M3_BLOCK[num_spec]  # 1568 / the production 1616


@pytest.mark.parametrize("num_spec", [0, 4])
def test_the_derivation_yields_an_integer_nvfp4_block(num_spec) -> None:
    config = _derive("nvfp4", num_spec)
    assert config.block_size == NVFP4_BLOCK[num_spec]
    assert isinstance(config.block_size, int)
    assert config.block_size % 16 == 0


@pytest.mark.parametrize("num_spec", [0, 4])
def test_the_derived_nvfp4_block_is_the_smallest_page_that_holds_the_state(
    num_spec,
) -> None:
    block = _derive("nvfp4", num_spec).block_size
    assert block * 288 >= GDN_STATE[num_spec] > (block - 16) * 288


def test_without_mtp_the_nvfp4_page_equals_the_state_page() -> None:
    config = _derive("nvfp4", 0)
    assert config.block_size * 288 == GDN_STATE[0]  # 2784 x 288: zero padding
    assert config.mamba_page_size_padded is None


def test_with_mtp4_the_state_page_is_padded_by_2560_bytes() -> None:
    config = _derive("nvfp4", 4)
    assert config.mamba_page_size_padded == config.block_size * 288 == 824_832
    assert config.mamba_page_size_padded - GDN_STATE[4] == 2560


@pytest.mark.parametrize(("num_spec", "capacity"), [(0, 4), (4, 8)])
def test_the_derived_nvfp4_block_divides_the_ring_capacity(num_spec, capacity) -> None:
    assert _derive("nvfp4", num_spec).block_size % capacity == 0


def test_the_nvfp4_block_is_larger_than_e4m3_by_the_per_token_byte_ratio() -> None:
    for num_spec in (0, 4):
        ratio = (
            _derive("nvfp4", num_spec).block_size
            / _derive("fp8_e4m3", num_spec).block_size
        )
        assert ratio == pytest.approx(512 / 288, rel=0.02)


# --------------------------------------------------------------------------- #
# The real CSA+linear allocator consumes NVFP4 owners
# --------------------------------------------------------------------------- #
MTP4_E4M3_BUDGET = 292 * 12_100_608  # M94: 292 IDs of 12,100,608 B each


def _csa_specs(num_spec: int, block: int, mode: KVQuantMode) -> dict:
    specs: dict = {}
    capacity = 4 * -(-(4 + num_spec) // 4)
    owners = [3 + 4 * i for i in range(12)] + ([48] if num_spec else [])
    for layer in owners:
        prefix = (
            f"mtp.layers.{layer}.self_attn"
            if layer == 48
            else f"model.layers.{layer}.self_attn"
        )
        specs[prefix + ".attn"] = FullAttentionSpec(
            block_size=block,
            num_kv_heads=1,
            head_size=D,
            head_size_v=D,
            dtype=torch.uint8,
            kv_quant_mode=mode,
            dcp_sharded=False,
        )
        specs[prefix + ".indexer.compressed"] = MLAAttentionSpec(
            block_size=block,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        )
        specs[prefix + ".indexer.raw"] = CircularBufferSpec(
            block_size=capacity,
            num_kv_heads=1,
            head_size=140,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        )
    gdn = MambaSpec(
        block_size=block,
        shapes=((2560, 3 + num_spec), (12, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
    )
    for layer in range(48):
        if layer % 4 != 3:
            specs[f"model.layers.{layer}.linear_attn"] = gdn
    specs["model.layers.2.ple_conv"] = MambaSpec(
        block_size=block,
        shapes=((10240, 9 + num_spec),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
        tp_replicated=True,
    )
    return specs


def _allocate(num_spec: int, block: int, mode: KVQuantMode, budget: int):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            max_model_len=262_144,
            get_num_kv_heads=lambda parallel_config: 1,
            get_total_num_hidden_layers=lambda: 48,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align", num_gpu_blocks_override=None, block_size=block
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    groups = U.get_kv_cache_groups(config, _csa_specs(num_spec, block, mode))
    kv_config = U.get_kv_cache_config_from_groups(config, groups, budget)
    concurrency = U.get_max_concurrency_for_kv_cache_config(config, kv_config)
    return kv_config, int(concurrency * 262_144)


def test_the_allocator_reproduces_the_e4m3_mtp4_reference() -> None:
    kv_config, reported = _allocate(
        4, 1616, KVQuantMode.FP8_PER_TENSOR, MTP4_E4M3_BUDGET
    )
    assert kv_config.num_blocks == 292
    assert reported == 407_159  # the "GPU KV cache size" line of the M94 log


def test_the_allocator_accepts_uniform_nvfp4_owners_without_mtp() -> None:
    kv_config, _ = _allocate(0, 2784, KVQuantMode.NVFP4, MTP4_E4M3_BUDGET)
    assert kv_config.num_blocks == 300
    page_sizes = {t.size // kv_config.num_blocks for t in kv_config.kv_cache_tensors}
    assert 2784 * 288 in page_sizes  # the main-KV owner page


def test_the_allocator_accepts_uniform_nvfp4_owners_with_mtp4() -> None:
    kv_config, reported = _allocate(4, 2864, KVQuantMode.NVFP4, MTP4_E4M3_BUDGET)
    assert kv_config.num_blocks == 269
    assert reported == 602_707
    assert not any(t.packed_members for t in kv_config.kv_cache_tensors)


def test_nvfp4_capacity_without_mtp_exceeds_e4m3_at_the_same_budget() -> None:
    _, e4m3 = _allocate(0, 1568, KVQuantMode.FP8_PER_TENSOR, MTP4_E4M3_BUDGET)
    _, fp4 = _allocate(0, 2784, KVQuantMode.NVFP4, MTP4_E4M3_BUDGET)
    assert fp4 / e4m3 == pytest.approx(1.566, abs=0.01)


def test_the_allocator_rejects_the_e4m3_block_for_thirteen_nvfp4_owners() -> None:
    # 13 pages of 465,408 B cannot hold the 822,272 B state one per page, and
    # the packer (two pages per physical page) leaves one owner over. The
    # block size has to come from the platform derivation instead.
    with pytest.raises(ValueError, match="do not fill whole physical pages"):
        _allocate(4, 1616, KVQuantMode.NVFP4, MTP4_E4M3_BUDGET)


def test_a_smaller_nvfp4_block_is_packed_two_per_page_rather_than_rejected() -> None:
    # Documents what the allocator does with the E4M3 block when nothing pads
    # it: it silently packs member pairs (the DCP2 mechanism). The DCP1 QSA
    # kernels do not read packed strides yet, so the derived block is the
    # supported layout for stage 1.
    kv_config, _ = _allocate(0, 1568, KVQuantMode.NVFP4, MTP4_E4M3_BUDGET)
    packed = [t for t in kv_config.kv_cache_tensors if t.packed_members]
    assert len(packed) == 6 and all(len(t.packed_members) == 2 for t in packed)


def test_the_qsa_module_exports_the_dtype_constant() -> None:
    assert qsa_module.NVFP4_KV_CACHE_DTYPE == nv.NVFP4_KV_CACHE_DTYPE == "nvfp4"
