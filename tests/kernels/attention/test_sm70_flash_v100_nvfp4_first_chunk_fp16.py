# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing test: NVFP4 first (no-prefix) chunk uses fp16 K/V.

VLLM_FLASH_V100_NVFP4_FIRST_CHUNK_FP16 (default 1): a prefill batch whose
requests all have seq_len == query_len goes through _flash_v100_prefill with
the fresh fp16 q/k/v; any request with prefix context, or knob=0, keeps the
paged NVFP4 bridge. Kernels are mocked, so no GPU is needed.
"""

from __future__ import annotations

import types

import pytest
import torch

import vllm.v1.attention.backends.flash_attn_v100 as mod


def _impl_cls():
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and "_forward_nvfp4" in vars(obj):
            return obj
    pytest.skip("impl class with _forward_nvfp4 not found")


def _run(monkeypatch, knob, seq_lens, q_lens):
    cls = _impl_cls()
    if knob is None:
        monkeypatch.delenv("VLLM_FLASH_V100_NVFP4_FIRST_CHUNK_FP16", raising=False)
    else:
        monkeypatch.setenv("VLLM_FLASH_V100_NVFP4_FIRST_CHUNK_FP16", knob)
    calls = {"prefill": [], "bridge": 0}
    total = sum(q_lens)
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    q = torch.zeros(total, 2, 256, dtype=torch.float16)
    k = torch.zeros(total, 1, 256, dtype=torch.float16)
    v = torch.zeros(total, 1, 256, dtype=torch.float16)
    out = torch.zeros(total, 2, 256, dtype=torch.float16)
    meta = types.SimpleNamespace(
        causal=True,
        ddtree_parent_ids=None,
        num_actual_tokens=total,
        max_query_len=max(q_lens),
        query_start_loc_cpu=qsl,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.zeros(len(q_lens), 4, dtype=torch.int32),
    )

    def fake_prefill(query, key, value, attn_metadata, output):
        calls["prefill"].append((query.dtype, key.dtype, value.dtype))
        return output

    def fake_bridge(**kwargs):
        calls["bridge"] += 1
        return kwargs["out"], True

    self = types.SimpleNamespace(
        nvfp4_kv_available=True,
        attn_type=mod.AttentionType.DECODER,
        alibi_slopes=None,
        logits_soft_cap=None,
        sinks=None,
        prefix_anchored_decode_window=None,
        num_kv_heads=1,
        nvfp4_paged_kv_to_fp16=object(),
        use_dflash2_grouped_verify=False,
        _flash_v100_has_sliding_window=lambda: False,
        _flash_v100_prefill=fake_prefill,
        _run_fp8_prefill_bridge=fake_bridge,
    )
    self._nvfp4_first_chunk_fp16_applies = types.MethodType(
        cls._nvfp4_first_chunk_fp16_applies, self
    )
    monkeypatch.setattr(mod, "_validate_nvfp4_xqa_cache", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_split_paged_kv_cache", lambda c: (c, c))
    monkeypatch.setattr(mod, "_is_cuda_graph_capturing", lambda q: False)
    monkeypatch.setattr(mod, "_record_route", lambda *a, **k: None)
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    cls._forward_nvfp4(
        self, layer, q, k, v, torch.zeros(1), meta, out, None, None
    )
    return calls


def test_knob_on_no_prefix_uses_fp16_prefill(monkeypatch):
    calls = _run(monkeypatch, "1", seq_lens=[32, 20], q_lens=[32, 20])
    assert calls["prefill"] == [(torch.float16,) * 3]
    assert calls["bridge"] == 0


def test_default_is_on(monkeypatch):
    calls = _run(monkeypatch, None, seq_lens=[32], q_lens=[32])
    assert len(calls["prefill"]) == 1 and calls["bridge"] == 0


def test_knob_on_prefix_chunk_uses_bridge(monkeypatch):
    calls = _run(monkeypatch, "1", seq_lens=[4128], q_lens=[32])
    assert calls["prefill"] == []
    assert calls["bridge"] == 1


def test_knob_on_mixed_batch_uses_bridge(monkeypatch):
    calls = _run(monkeypatch, "1", seq_lens=[32, 100], q_lens=[32, 20])
    assert calls["prefill"] == []
    assert calls["bridge"] == 2


def test_knob_off_first_chunk_uses_bridge(monkeypatch):
    calls = _run(monkeypatch, "0", seq_lens=[32], q_lens=[32])
    assert calls["prefill"] == []
    assert calls["bridge"] == 1
