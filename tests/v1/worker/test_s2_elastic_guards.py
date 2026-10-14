# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2-fix: startup guards for the elastic KV tail (block-major
layout, FlexAttention) and the EngineCore span decorator (B2). CPU-only."""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.engine.core as core_mod
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, MambaSpec
from vllm.v1.worker.gpu import attn_utils

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _BlockMajor(AttentionBackend):
    @staticmethod
    def get_name():
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class _KvPlaneOuter(_BlockMajor):
    @staticmethod
    def get_name():
        return "KV_OUTER"

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _Flex(_BlockMajor):
    @staticmethod
    def get_name():
        return "FLEX_ATTENTION"



def _attn_spec():
    from vllm.v1.kv_cache_interface import AttentionSpec

    s = AttentionSpec.__new__(AttentionSpec)
    object.__setattr__(s, "num_kv_heads", 2)
    object.__setattr__(s, "head_size", 64)
    object.__setattr__(s, "dtype", torch.float16)
    return s


def _group(backend, spec):
    return SimpleNamespace(backend=backend, kv_cache_spec=spec, kv_cache_group_id=0)


def _cfg(elastic=True):
    t = KVCacheTensor(size=16, shared_by=["l0"], max_size=32 if elastic else None)
    return KVCacheConfig(num_blocks=1, kv_cache_tensors=[t], kv_cache_groups=[])


def _run(backend, spec=None, elastic=True):
    attn_utils.check_elastic_kv_layout(
        _cfg(elastic), [_group(backend, spec or _attn_spec())], [16], "auto"
    )


def test_block_major_backend_passes():
    _run(_BlockMajor)


def test_non_hybrid_kv_plane_outermost_is_refused():
    with pytest.raises(ValueError, match="block-major.*KV_OUTER.*dim 1"):
        _run(_KvPlaneOuter)


def test_hybrid_kv_plane_outermost_is_allowed_because_layout_is_rewritten():
    mamba = MambaSpec.__new__(MambaSpec)
    attn_utils.check_elastic_kv_layout(
        _cfg(),
        [_group(_KvPlaneOuter, _attn_spec()), _group(_BlockMajor, mamba)],
        [16, 16],
        "auto",
    )


def test_flex_attention_is_refused():
    with pytest.raises(ValueError, match="FlexAttention"):
        _run(_Flex)


def test_guards_are_off_without_an_elastic_tensor():
    _run(_KvPlaneOuter, elastic=False)
    _run(_Flex, elastic=False)


def test_init_kv_cache_calls_the_guard_before_allocating(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attn_utils, "check_elastic_kv_layout", lambda *a: calls.append("check")
    )
    monkeypatch.setattr(
        attn_utils, "get_shared_kv_cache_layers", lambda _c: {}
    )

    def boom(*a, **k):
        calls.append("alloc")
        raise RuntimeError("stop")

    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", boom)
    with pytest.raises(RuntimeError, match="stop"):
        attn_utils.init_kv_cache([], {}, _cfg(), [[]], torch.device("cpu"), "auto",
                                 [16], None)
    assert calls == ["check", "alloc"]


def test_prepare_model_span_is_on_initialize_kv_caches_only():
    """B2: the @instrument(span_name="Prepare model") decorator must stay glued
    to _initialize_kv_caches (flag-off behaviour = S1)."""
    tree = ast.parse(pathlib.Path(core_mod.__file__).read_text())
    spans = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            for dec in fn.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and getattr(dec.func, "id", "") == "instrument"
                    and any(
                        k.arg == "span_name"
                        and getattr(k.value, "value", None) == "Prepare model"
                        for k in dec.keywords
                    )
                ):
                    spans[f"{cls.name}.{fn.name}"] = True
    assert spans == {"EngineCore._initialize_kv_caches": True}
