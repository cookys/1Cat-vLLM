# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing test: NVFP4 mixed-batch decode-row split.

VLLM_FLASH_V100_NVFP4_PREFIX_DECODE_ROWS (default 1): in a batch with
max_query_len > 8, rows with q <= smallq_decode_max_query_len and prior
context run through the paged NVFP4 small-query decode path (one decode row
per query token); only the remaining rows use the FP16 prefix bridge.
Kernels are mocked, so no GPU is needed.
"""

from __future__ import annotations

import types

import pytest
import torch

import vllm.v1.attention.backends.flash_attn_v100 as mod

KNOB = "VLLM_FLASH_V100_NVFP4_PREFIX_DECODE_ROWS"


def _impl_cls():
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and "_forward_nvfp4" in vars(obj):
            return obj
    pytest.skip("impl class with _forward_nvfp4 not found")


def _run(monkeypatch, knob, seq_lens, q_lens, max_q=16):
    cls = _impl_cls()
    if knob is None:
        monkeypatch.delenv(KNOB, raising=False)
    else:
        monkeypatch.setenv(KNOB, knob)
    monkeypatch.setattr(mod, "_logged_nvfp4_mixed_split_fired", False)
    calls = {"decode": [], "bridge": [], "prefill": 0, "smallq": 0, "decode1": 0}
    total = sum(q_lens)
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    # Row r's tokens are filled with r + 1 so scatter offsets are checkable.
    q = torch.zeros(total, 2, 256, dtype=torch.float16)
    for r in range(len(q_lens)):
        q[int(qsl[r]) : int(qsl[r + 1])] = r + 1
    k = torch.zeros(total, 1, 256, dtype=torch.float16)
    out = torch.zeros(total, 2, 256, dtype=torch.float16)
    bt = torch.arange(len(q_lens) * 4, dtype=torch.int32).reshape(len(q_lens), 4)
    meta = types.SimpleNamespace(
        causal=True,
        ddtree_parent_ids=None,
        num_actual_tokens=total,
        max_query_len=max(q_lens),
        query_start_loc_cpu=qsl,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=bt,
    )

    def fake_decode(layer, qq, kc, vc, block_table, sl, am, *, out, **kw):
        calls["decode"].append(
            dict(
                rows=int(qq.shape[0]),
                q_vals=qq[:, 0, 0].tolist(),
                seq_lens=sl.tolist(),
                bt_first=block_table[:, 0].tolist(),
                max_seq=kw["max_seq_len_hint"],
            )
        )
        out.copy_(qq * 10)

    def fake_bridge(**kwargs):
        calls["bridge"].append(int(kwargs["query"].shape[1]))
        kwargs["out"].fill_(-1)
        return kwargs["out"], True

    def fake_prefill(*a, **k):
        calls["prefill"] += 1
        return a[-1]

    def fake_smallq(*a, **k):
        calls["smallq"] += 1
        return a[5]

    def fake_decode1(*a, **k):
        calls["decode1"] += 1
        return a[-1]

    self = types.SimpleNamespace(
        _flash_v100_decode=fake_decode1,
        nvfp4_kv_available=True,
        attn_type=mod.AttentionType.DECODER,
        alibi_slopes=None,
        logits_soft_cap=None,
        sinks=None,
        prefix_anchored_decode_window=None,
        num_kv_heads=1,
        nvfp4_paged_kv_to_fp16=object(),
        use_dflash2_grouped_verify=False,
        smallq_decode_max_query_len=max_q,
        _flash_v100_has_sliding_window=lambda: False,
        _flash_v100_prefill=fake_prefill,
        _run_fp8_prefill_bridge=fake_bridge,
        _call_flash_attn_smallq_decode_paged=fake_decode,
        _flash_v100_small_query_prefill_as_decode=fake_smallq,
    )
    self._nvfp4_first_chunk_fp16_applies = types.MethodType(
        cls._nvfp4_first_chunk_fp16_applies, self
    )
    self._run_nvfp4_mixed_decode_rows = types.MethodType(
        cls._run_nvfp4_mixed_decode_rows, self
    )
    monkeypatch.setattr(mod, "_validate_nvfp4_xqa_cache", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_split_paged_kv_cache", lambda c: (c, c))
    monkeypatch.setattr(mod, "_is_cuda_graph_capturing", lambda q: False)
    monkeypatch.setattr(mod, "_record_route", lambda *a, **k: None)
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    cls._forward_nvfp4(self, layer, q, k, k, torch.zeros(4, 16, 1, 144), meta, out, None, None)
    return calls, out, qsl


# 3 decode rows (q=1), 1 DFlash2 verify row (q=8), 1 append chunk (q=1500)
Q = [1, 1, 1, 8, 1500]
S = [5000, 6000, 7000, 8000, 9000]


def test_mixed_batch_splits_decode_rows(monkeypatch, caplog):
    with caplog.at_level("INFO"):
        calls, out, qsl = _run(monkeypatch, "1", S, Q)
    assert len(calls["decode"]) == 1
    d = calls["decode"][0]
    # 3 x1 + 8 expanded verify tokens = 11 decode rows, one call
    assert d["rows"] == 11
    assert d["seq_lens"] == [5000, 6000, 7000] + list(range(8000 - 8 + 1, 8001))
    assert d["bt_first"] == [0, 4, 8] + [12] * 8
    assert d["max_seq"] == 8000
    # bridge only for the append chunk
    assert calls["bridge"] == [1500]
    # scatter: decode rows x10, append rows overwritten by bridge (-1)
    assert out[0, 0, 0].item() == 10
    assert out[1, 0, 0].item() == 20
    assert out[2, 0, 0].item() == 30
    assert (out[3:11, 0, 0] == 40).all()
    assert (out[11:, 0, 0] == -1).all()
    assert any("split fired: decode_rows=4 prefill_rows=1" in r.message for r in caplog.records)


def test_knob_off_bridge_for_every_request(monkeypatch):
    calls, out, qsl = _run(monkeypatch, "0", S, Q)
    assert calls["decode"] == []
    assert calls["bridge"] == Q


def test_pure_decode_batch_unchanged(monkeypatch, caplog):
    with caplog.at_level("INFO"):
        calls, out, qsl = _run(monkeypatch, "1", [5000, 6000, 7000], [1, 1, 1])
    assert calls["decode1"] == 1
    assert calls["decode"] == [] and calls["bridge"] == []
    assert not any("mixed-batch decode-row split" in r.message for r in caplog.records)


def test_pure_verify_batch_unchanged(monkeypatch):
    calls, out, qsl = _run(monkeypatch, "1", [5000, 6000], [8, 5])
    assert calls["smallq"] == 1
    assert calls["decode"] == [] and calls["bridge"] == []


def test_pure_first_chunk_batch_still_fp16(monkeypatch):
    calls, out, qsl = _run(monkeypatch, "1", [32, 20], [32, 20])
    assert calls["prefill"] == 1
    assert calls["decode"] == [] and calls["bridge"] == []


def test_prefill_only_with_prefix_unchanged(monkeypatch):
    calls, out, qsl = _run(monkeypatch, "1", [4128, 3000], [32, 1500])
    assert calls["decode"] == []
    assert calls["bridge"] == [32, 1500]
