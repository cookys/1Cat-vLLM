# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test: env-gated pre-quantization fp16 KV dump (plan 072 Q1.23)."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

import vllm.v1.attention.backends.flash_attn_v100 as mod
import vllm.v1.attention.backends.sm70_kv_dump as kv_dump

HEAD_DIM = 8


def _impl_cls():
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and "do_kv_cache_update" in vars(obj):
            return obj
    pytest.skip("impl class with do_kv_cache_update not found")


def _md(q_lens, seq_lens):
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    return types.SimpleNamespace(
        query_start_loc_cpu=qsl,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_size=16,
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    state = types.SimpleNamespace(md=None, stored=[])
    monkeypatch.setattr(kv_dump, "_DUMPER", None)
    monkeypatch.setattr(kv_dump, "_metadata_for_layer", lambda name: state.md)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_DIR", str(tmp_path),
                        raising=False)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_MAX_TOKENS", 6,
                        raising=False)
    fake = types.ModuleType("vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton")
    fake.store_nvfp4_kv_triton = lambda *a, **k: state.stored.append((a, k))
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton", fake
    )
    cls = _impl_cls()
    impl = types.SimpleNamespace(
        kv_cache_dtype="nvfp4",
        nvfp4_kv_available=True,
        attn_type=mod.AttentionType.DECODER,
        num_kv_heads=1,
        head_size=HEAD_DIM,
    )
    state.call = lambda layer, k, v: cls.do_kv_cache_update(
        impl, layer, k, v, None, None
    )
    state.dir = tmp_path
    return state


def _layer(i):
    return types.SimpleNamespace(
        layer_name=f"model.layers.{i}.self_attn.attn",
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )


def test_disabled_by_default(monkeypatch, tmp_path):
    assert kv_dump.ENABLED is False
    assert kv_dump._DUMPER is None
    monkeypatch.setattr(kv_dump, "_DUMPER", None)
    cls = _impl_cls()
    fake = types.ModuleType("vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton")
    seen = []
    fake.store_nvfp4_kv_triton = lambda *a, **k: seen.append(1)
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton", fake
    )
    impl = types.SimpleNamespace(
        kv_cache_dtype="nvfp4", nvfp4_kv_available=True,
        attn_type=mod.AttentionType.DECODER, num_kv_heads=1, head_size=HEAD_DIM,
    )
    k = torch.zeros(4, 1, HEAD_DIM, dtype=torch.float16)
    cls.do_kv_cache_update(impl, _layer(0), k, k, None, None)
    assert seen == [1]
    assert kv_dump._DUMPER is None
    assert list(tmp_path.iterdir()) == []


def test_dump_first_request_only(monkeypatch, env):
    monkeypatch.setattr(kv_dump, "ENABLED", True)
    torch.manual_seed(0)
    layer = _layer(3)
    # chunk 1: request 0 (4 tok, no prefix) + request 1 (3 tok) -> only req 0.
    k1 = torch.randn(7, 1, HEAD_DIM).half()
    v1 = torch.randn(7, 1, HEAD_DIM).half()
    env.md = _md([4, 3], [4, 3])
    env.call(layer, k1, v1)
    # chunk 2: request 0 continues (4 tok, ctx 4 -> 8), cap = 6 -> take 2.
    k2 = torch.randn(4, 1, HEAD_DIM).half()
    v2 = torch.randn(4, 1, HEAD_DIM).half()
    env.md = _md([4], [8])
    k2_copy = k2.clone()
    env.call(layer, k2, v2)
    # next call (any later forward) emits the 'written' log and does not re-collect.
    env.md = _md([1], [9])
    env.call(layer, k2, v2)

    d = env.dir / "rank0"
    kk = np.load(d / "layer3_k.npy")
    vv = np.load(d / "layer3_v.npy")
    assert kk.shape == (6, HEAD_DIM) and kk.dtype == np.float16
    assert vv.shape == (6, HEAD_DIM) and vv.dtype == np.float16
    np.testing.assert_array_equal(kk[:4], k1[:4, 0].numpy())
    np.testing.assert_array_equal(kk[4:], k2[:2, 0].numpy())
    np.testing.assert_array_equal(vv[:4], v1[:4, 0].numpy())
    meta = json.loads((d / "meta.json").read_text())
    assert meta["tokens"] == 6 and meta["layers"] == [3]
    assert meta["head_dim"] == HEAD_DIM and meta["num_kv_heads"] == 1
    assert meta["kv_cache_dtype"] == "nvfp4" and meta["tp_rank"] == 0
    assert meta["block_size"] == 16
    assert meta["k_scale"]["3"] == 1.0
    # the quantized store saw the original tensors, untouched, every call
    assert len(env.stored) == 3
    assert env.stored[0][0][0] is k1 and env.stored[1][0][0] is k2
    assert torch.equal(k2, k2_copy)


def test_new_request_finalizes_short_dump(monkeypatch, env):
    monkeypatch.setattr(kv_dump, "ENABLED", True)
    layer = _layer(0)
    k = torch.randn(3, 1, HEAD_DIM).half()
    env.md = _md([3], [3])
    env.call(layer, k, k)
    # a different request now sits at index 0: ctx != collected -> finalize.
    env.md = _md([5], [5])
    env.call(layer, torch.randn(5, 1, HEAD_DIM).half(), k)
    meta = json.loads((env.dir / "rank0" / "meta.json").read_text())
    assert meta["tokens"] == 3
    assert np.load(env.dir / "rank0" / "layer0_k.npy").shape == (3, HEAD_DIM)
