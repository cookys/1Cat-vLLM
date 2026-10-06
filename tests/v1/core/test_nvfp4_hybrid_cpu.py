# SPDX-License-Identifier: Apache-2.0
"""Real allocator, platform alignment and worker storage tests; CPU only."""

import os
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from benchmarks.sm70_nvfp4_27b_capacity import capacity, geometry
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Backend as Backend
from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVQuantMode,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    assert not torch.cuda.is_initialized()
    from vllm.v1.attention.backends import triton_attn

    monkeypatch.setattr(triton_attn, "get_kv_cache_layout", lambda: "NHD")
    yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "cli,main,sw,page", [(16, 2912, 416, 838656), (2048, 4096, 1024, 1179648)]
)
def test_real_unifier_padding_and_hash_grid(cli, main, sw, page):
    specs, config = geometry("dflash7", "nvfp4", cli)
    original = specs.copy()
    unified = U.unify_kv_cache_spec_page_size(specs, config)
    assert specs == original  # no input spec mutation
    assert {s.page_size_bytes for s in unified.values()} == {page}
    assert unified["model.layers.3"] == original["model.layers.3"]
    draft = unified["draft.layers.0"]
    assert draft.block_size == sw
    assert draft.sliding_window == 2048
    assert draft.dtype == torch.float16 and draft.kv_quant_mode == KVQuantMode.NONE
    assert unified["model.layers.0"].shapes == original["model.layers.0"].shapes
    assert U.unify_kv_cache_spec_page_size(unified, config) == unified
    groups = U.get_kv_cache_groups(config, specs)
    cache = KVCacheConfig(num_blocks=10, kv_cache_tensors=[], kv_cache_groups=groups)
    assert U.resolve_kv_cache_block_sizes(cache, config) == (main, sw)


@pytest.mark.parametrize(
    "cli,users,ids,per_req", [(16, 10, 2645, 262), (2048, 9, 1880, 193)]
)
def test_real_dflash_allocator_capacity(cli, users, ids, per_req):
    result = capacity("dflash7", "nvfp4", cli, 16.53)
    assert (
        result["users_at_256k"],
        result["pool_ids"],
        result["ids_per_256k_request"],
    ) == (users, ids, per_req)
    after_reserve = capacity("dflash7", "nvfp4", cli, 16.53, True)
    assert 256 * 2**20 <= after_reserve["bridge_reserve_bytes"] < 260 * 2**20
    assert after_reserve["pool_ids"] < ids


def test_fp8_unifier_legacy_behavior_is_unchanged():
    specs, config = geometry("dflash7", "fp8", 2048)
    assert U._pad_nvfp4_hybrid_draft_pages(specs) is specs
    out = U.unify_kv_cache_spec_page_size(specs, config)
    assert out["draft.layers.0"] == specs["draft.layers.0"]
    assert out["model.layers.3"].block_size == 4096


@pytest.mark.parametrize("mode", ["nospec", "mtp4", "dflash7"])
@pytest.mark.parametrize("cli", [16, 2048])
def test_block_math_matches_actual_platform_alignment(monkeypatch, mode, cli):
    from vllm.config import vllm as config_module
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms.interface import Platform

    specs, config = geometry(mode, "nvfp4", cli)
    state = specs["model.layers.0"]
    expected = config.cache_config.block_size
    config.cache_config.cache_dtype = "nvfp4"
    config.cache_config.block_size = cli
    config.cache_config.user_specified_mamba_block_size = False
    config.cache_config.mamba_block_size = None
    config.cache_config.mamba_page_size_padded = None
    config.model_config.use_mla = False
    config.model_config.get_head_size = lambda: 256
    config.model_config.architecture = "cpu_test_geometry"
    model = NS(
        get_mamba_specs_from_config=lambda _: [replace(state, page_size_padded=None)]
    )
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *a, **kw: (model, None)
    )
    monkeypatch.setattr(
        config_module, "set_current_vllm_config", lambda _: nullcontext()
    )
    monkeypatch.setenv("VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16", "0")
    Platform._align_hybrid_block_size(config, Backend)
    assert config.cache_config.block_size == expected
    assert config.cache_config.mamba_block_size == expected
    padded = config.cache_config.mamba_page_size_padded
    assert (padded or state.real_page_size_bytes) == expected * 288


@pytest.mark.parametrize("runner_version", [1, 2])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("cli", [16, 2048])
def test_real_reshape_keeps_fp16_draft_stride_and_mamba_state(
    cli, layout, runner_version, monkeypatch
):
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
    from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

    monkeypatch.setattr(triton_attn, "get_kv_cache_layout", lambda: layout)

    specs, config = geometry("dflash7", "nvfp4", cli)
    specs = U.unify_kv_cache_spec_page_size(specs, config)
    names = ["model.layers.3", "draft.layers.0", "model.layers.0"]
    specs = {n: specs[n] for n in names}
    groups = [AttentionGroup(Backend, [n], specs[n], i) for i, n in enumerate(names)]
    cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([n], specs[n]) for n in names],
    )
    kernel_blocks = prepare_kernel_block_sizes(cfg, [[g] for g in groups])
    assert kernel_blocks == [specs[n].block_size for n in names]
    raw = {
        n: torch.full((2 * s.page_size_bytes,), 0x5A, dtype=torch.uint8)
        for n, s in specs.items()
    }
    if runner_version == 2:
        caches = _reshape_kv_cache(groups, raw, "nvfp4", kernel_blocks, {})
    else:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        runner = NS(
            cache_config=NS(cache_dtype="nvfp4"),
            runner_only_attn_layers=set(),
            _kv_cache_spec_attn_group_iterator=lambda: iter(groups),
        )
        runner._update_hybrid_attention_mamba_layout = lambda *args: (
            GPUModelRunner._update_hybrid_attention_mamba_layout(runner, *args)
        )
        caches = GPUModelRunner._reshape_kv_cache_tensors(runner, raw, kernel_blocks)
    assert caches[names[0]].shape[-1] == 144
    from vllm.v1.attention.backends.flash_attn_v100 import _validate_nvfp4_xqa_cache

    if layout == "NHD":
        _validate_nvfp4_xqa_cache(caches[names[0]], 1)
    else:
        # HND's singleton head has a different stride even at TP4/H=1.
        # Native P1 explicitly requires head stride144, so never admit it.
        with pytest.raises(ValueError, match="NHD"):
            _validate_nvfp4_xqa_cache(caches[names[0]], 1)
    draft = caches[names[1]]
    assert draft.dtype == torch.float16 and draft.shape[-1] == 128
    assert draft.stride(0) * 2 == specs[names[1]].page_size_bytes
    page = specs[names[1]].page_size_bytes
    draft[0].fill_(1)
    assert (raw[names[1]][page:] == 0x5A).all()
    assert (raw[names[1]][draft[0].numel() * 2 : page] == 0x5A).all()
    assert caches[names[2]][1].shape == (2, 12, 128, 128)
    assert caches[names[2]][1].stride(0) * 4 == page


def test_padded_pages_cannot_be_virtually_split():
    from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
    from vllm.v1.worker.utils import AttentionGroup

    specs, config = geometry("dflash7", "nvfp4", 2048)
    spec = U.unify_kv_cache_spec_page_size(specs, config)["draft.layers.0"]
    group = AttentionGroup(Backend, ["draft"], spec, 0)
    with pytest.raises(ValueError, match="padded physical pages"):
        _reshape_kv_cache(
            [group],
            {"draft": torch.empty(spec.page_size_bytes, dtype=torch.uint8)},
            "nvfp4",
            [16],
            {},
        )


@pytest.mark.parametrize(
    "explicit,expected", [(None, "auto"), ("auto", "auto"), ("fp8_e5m2", "fp8_e5m2")]
)
def test_real_dflash_loader_defaults_draft_to_fp16(monkeypatch, explicit, expected):
    from dataclasses import dataclass

    from vllm.compilation import backends
    from vllm.v1.worker.gpu.spec_decode.dflash import utils

    @dataclass
    class Cache:
        cache_dtype: str = "nvfp4"

    @dataclass
    class Attention:
        use_non_causal: bool = False
        backend: object = None

    @dataclass
    class Config:
        cache_config: Cache
        attention_config: Attention
        speculative_config: object

    config = Config(
        Cache(),
        Attention(),
        NS(
            kv_cache_dtype=explicit,
            attention_backend=None,
            draft_model_config=NS(hf_config=NS(is_causal=True, num_hidden_layers=5)),
        ),
    )
    monkeypatch.setattr(
        utils,
        "current_platform",
        NS(is_cuda=lambda: True, is_device_capability=lambda cap: cap == 70),
    )
    monkeypatch.setattr(backends, "set_model_tag", lambda _: nullcontext())
    observed = []

    class StopBeforeModelLoad(Exception):
        pass

    def get_model(**kw):
        observed.append(kw["vllm_config"])
        raise StopBeforeModelLoad

    monkeypatch.setattr(utils, "get_model", get_model)
    with pytest.raises(StopBeforeModelLoad):
        utils.load_dflash_model(None, config)
    assert config.cache_config.cache_dtype == "nvfp4"
    assert observed[0].cache_config.cache_dtype == expected
