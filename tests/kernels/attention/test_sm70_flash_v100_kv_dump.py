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
_REAL_MD = kv_dump._metadata_for_layer


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
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_MIN_TOKENS", 3,
                        raising=False)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_LAYER_PREFIX",
                        "language_model.model.layers.", raising=False)
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
        layer_name=f"language_model.model.layers.{i}.self_attn.attn",
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
    assert meta["complete"] is True and meta["tokens_min"] == 6
    info = meta["layers_info"]["3"]
    assert info["k_shape"] == [7, 1, HEAD_DIM] and info["saved_shape"] == [6, HEAD_DIM]
    assert info["kv_cache_dtype"] == "nvfp4" and info["complete"] is True
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
    assert meta["tokens"] == 3 and meta["complete"] is False
    assert meta["layers_info"]["0"]["complete"] is False
    assert np.load(env.dir / "rank0" / "layer0_k.npy").shape == (3, HEAD_DIM)


def _draft_layer(i):
    layer = _layer(i)
    layer.layer_name = f"model.layers.{i}.self_attn.attn"
    return layer


def test_v2_warmup_draft_and_32k_request(monkeypatch, env):
    """relay 14 D regression: 8-token warm-up and draft layers must not be dumped."""
    cap, chunk, hd = 32768, 4096, 4
    monkeypatch.setattr(kv_dump, "ENABLED", True)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_MAX_TOKENS", cap)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_MIN_TOKENS", 4096)
    main = [_layer(3 + 4 * i) for i in range(16)]
    draft = [_draft_layer(64 + i) for i in range(5)]
    impl_draft = types.SimpleNamespace(
        kv_cache_dtype="auto", nvfp4_kv_available=False,
        attn_type=mod.AttentionType.DECODER, num_kv_heads=2, head_size=128,
    )
    cls = _impl_cls()

    def call_main(layer, k):
        env.call(layer, k, k)

    # 1) warm-up: 8 tokens on every main layer -> ignored (< MIN_TOKENS)
    env.md = _md([8], [8])
    w = torch.zeros(8, 1, hd).half()
    for layer in main:
        call_main(layer, w)
    assert not (env.dir / "rank0").exists()
    # 2) draft layer calls (2 heads x 128 view) -> ignored by prefix, even if big
    env.md = _md([chunk], [chunk])
    dk = torch.zeros(chunk, 2, 128).half()
    for layer in draft:
        monkeypatch.setattr(kv_dump, "_metadata_for_layer", lambda n: env.md)
        # draft attention impl is a different impl instance (kv_cache_dtype=auto)
        kv_dump.on_kv_update(layer, impl_draft, dk, dk)
    assert not (env.dir / "rank0").exists()
    # 3) the real 32K request in 4096-token chunks on all 16 main layers
    lines = []
    monkeypatch.setattr(
        kv_dump.logger, "info", lambda fmt, *args, **kw: lines.append(fmt % args)
    )
    g = torch.Generator().manual_seed(1)
    ref = {}
    for c in range(cap // chunk):
        env.md = _md([chunk], [(c + 1) * chunk])
        for layer in main:
            k = torch.randn(chunk, 1, hd, generator=g).half()
            ref.setdefault(layer.layer_name, []).append(k[:, 0].clone())
            call_main(layer, k)
        for layer in draft:  # draft keeps being called every forward
            kv_dump.on_kv_update(layer, impl_draft, dk, dk)
    # v3: the written line was already printed at finalize (captured in `lines`)
    env.md = _md([1], [cap + 1])
    call_main(main[0], torch.zeros(1, 1, hd).half())
    written = [l for l in lines if "KV dump written" in l]
    assert len(written) == 1
    assert f"layers=16 tokens={cap} tokens_max={cap} complete=16 skipped=5" in written[0]
    meta = json.loads((env.dir / "rank0" / "meta.json").read_text())
    assert meta["layers"] == [3 + 4 * i for i in range(16)]
    assert meta["tokens"] == cap and meta["tokens_min"] == cap
    assert meta["complete"] is True
    assert sorted(meta["skipped_layers"]) == sorted(l.layer_name for l in draft)
    assert meta["head_dim"] == hd and meta["num_kv_heads"] == 1
    assert meta["kv_cache_dtype"] == "nvfp4"
    for layer in main:
        i = meta["layer_names"]
        idx = [k for k, n in i.items() if n == layer.layer_name][0]
        info = meta["layers_info"][idx]
        assert info["tokens"] == cap and info["complete"] is True
        assert info["k_shape"] == [chunk, 1, hd] and info["saved_shape"] == [cap, hd]
        assert info["dtype"] == "float16"
        kk = np.load(env.dir / "rank0" / f"layer{idx}_k.npy")
        np.testing.assert_array_equal(kk, torch.cat(ref[layer.layer_name]).numpy())
    # no draft files at all
    assert not list((env.dir / "rank0").glob("layer6[4-8]_*"))
    dumper = kv_dump._DUMPER
    assert dumper._logged_written


def test_v2_prefixed_cached_chunk_does_not_latch(monkeypatch, env):
    """A big chunk that is not a request start (cached prefix) cannot latch."""
    monkeypatch.setattr(kv_dump, "ENABLED", True)
    env.md = _md([4], [100])
    k = torch.zeros(4, 1, HEAD_DIM).half()
    env.call(_layer(3), k, k)
    assert not (env.dir / "rank0").exists()


def test_v3_no_forward_context_ignored_and_written_line_at_finalize(
    monkeypatch, env, caplog
):
    """relay 16 D0 regression: calls without a usable forward context (before,
    between, after the chunks) are ignored without ERROR logs, and the written
    line is printed exactly once, synchronously, when the cap is reached."""
    import logging

    import vllm.forward_context as fc

    cap, chunk, hd = 32768, 4096, 4
    monkeypatch.setattr(kv_dump, "ENABLED", True)
    monkeypatch.setattr(kv_dump, "_NO_CTX_LOGGED", False)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_MAX_TOKENS", cap)
    monkeypatch.setattr(kv_dump.envs, "VLLM_FLASH_V100_KV_DUMP_MIN_TOKENS", 4096)
    # use the REAL _metadata_for_layer on top of a controllable forward context
    monkeypatch.setattr(kv_dump, "_metadata_for_layer", _REAL_MD)
    mode = {"m": "raise", "md": None}

    def fake_ctx():
        if mode["m"] == "raise":
            raise AssertionError("Forward context is not set.")
        if mode["m"] == "none":
            return types.SimpleNamespace(attn_metadata=None)
        if mode["m"] == "missing":
            return types.SimpleNamespace(attn_metadata={"other.layer": mode["md"]})
        if mode["m"] == "empty_list":
            return types.SimpleNamespace(attn_metadata=[])
        return types.SimpleNamespace(
            attn_metadata={l.layer_name: mode["md"] for l in main + draft}
        )

    monkeypatch.setattr(fc, "get_forward_context", fake_ctx)
    main = [_layer(3 + 4 * i) for i in range(16)]
    draft = [_draft_layer(64 + i) for i in range(5)]
    impl_main = types.SimpleNamespace(
        kv_cache_dtype="nvfp4", num_kv_heads=1, head_size=hd
    )
    impl_draft = types.SimpleNamespace(
        kv_cache_dtype="auto", num_kv_heads=2, head_size=128
    )
    dk = torch.zeros(chunk, 2, 128).half()
    caplog.set_level(logging.DEBUG)
    old_level = kv_dump.logger.level
    kv_dump.logger.setLevel(logging.DEBUG)
    try:

        def junk_calls():
            kk = torch.zeros(chunk, 1, hd).half()
            for m in ("raise", "none", "missing", "empty_list"):
                mode["m"] = m
                for layer in main:
                    kv_dump.on_kv_update(layer, impl_main, kk, kk)
                for layer in draft:
                    kv_dump.on_kv_update(layer, impl_draft, dk, dk)
            mode["m"] = "ok"

        junk_calls()  # before
        g = torch.Generator().manual_seed(2)
        ref = {}
        n_chunks = cap // chunk
        for c in range(n_chunks):
            mode["md"] = _md([chunk], [(c + 1) * chunk])
            mode["m"] = "ok"
            for i, layer in enumerate(main):
                k = torch.randn(chunk, 1, hd, generator=g).half()
                ref.setdefault(layer.layer_name, []).append(k[:, 0].clone())
                kv_dump.on_kv_update(layer, impl_main, k, k)
                if i < len(draft):  # draft layers run every forward, ctx ok
                    kv_dump.on_kv_update(draft[i], impl_draft, dk, dk)
                last = c == n_chunks - 1
                n_written = sum("KV dump written" in r.getMessage()
                                for r in caplog.records)
                # nothing before the very last layer of the last chunk
                assert n_written == (1 if last and i == len(main) - 1 else 0)
            if c < n_chunks - 1:
                junk_calls()  # between chunks
        # printed already, with no further call needed
        written = [r.getMessage() for r in caplog.records
                   if "KV dump written" in r.getMessage()]
        assert len(written) == 1
        assert (f"layers=16 tokens={cap} tokens_max={cap} complete=16 skipped=5"
                in written[0])
        assert written[0].startswith("FLASH_ATTN_V100 KV dump written dir=")
        junk_calls()  # after finalize
        mode["md"] = _md([1], [cap + 1])
        kv_dump.on_kv_update(main[0], impl_main, torch.zeros(1, 1, hd).half(),
                             torch.zeros(1, 1, hd).half())
    finally:
        kv_dump.logger.setLevel(old_level)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert sum("KV dump written" in r.getMessage() for r in caplog.records) == 1
    assert sum("no usable forward context" in r.getMessage()
               for r in caplog.records) == 1  # debug, first time only
    meta = json.loads((env.dir / "rank0" / "meta.json").read_text())
    assert meta["tokens"] == cap and meta["complete"] is True
    assert len(meta["layers"]) == 16
    for layer in main:
        idx = [k for k, n in meta["layer_names"].items() if n == layer.layer_name][0]
        kk = np.load(env.dir / "rank0" / f"layer{idx}_k.npy")
        np.testing.assert_array_equal(kk, torch.cat(ref[layer.layer_name]).numpy())


def test_v3_early_finalize_prints_written_once(monkeypatch, env, caplog):
    import logging

    monkeypatch.setattr(kv_dump, "ENABLED", True)
    caplog.set_level(logging.INFO)
    old_level = kv_dump.logger.level
    kv_dump.logger.setLevel(logging.INFO)
    try:
        k = torch.randn(3, 1, HEAD_DIM).half()
        env.md = _md([3], [3])
        for i in (0, 1):
            env.call(_layer(i), k, k)
        env.md = _md([5], [5])  # new request before cap
        env.call(_layer(0), torch.randn(5, 1, HEAD_DIM).half(), k)
        assert not [r for r in caplog.records if "KV dump written" in r.getMessage()]
        env.call(_layer(1), torch.randn(5, 1, HEAD_DIM).half(), k)
        env.call(_layer(1), torch.randn(5, 1, HEAD_DIM).half(), k)
    finally:
        kv_dump.logger.setLevel(old_level)
    written = [r.getMessage() for r in caplog.records
               if "KV dump written" in r.getMessage()]
    assert len(written) == 1
    assert "layers=2 tokens=3 tokens_max=3 complete=0 skipped=0" in written[0]
    meta = json.loads((env.dir / "rank0" / "meta.json").read_text())
    assert meta["complete"] is False
