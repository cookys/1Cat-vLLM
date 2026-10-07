# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing test: VLLM_FLASH_V100_GROUPED_VERIFY_PER_REQUEST_FALLBACK.

When the whole-batch DFlash2 grouped-verify gate rejects (reqs=10 is not in
{2,4,8}), the single-request B1 grouped one-pass runs once per eligible q=8
request instead of expanding every request to per-row XQA. The NVFP4 mixed
batch split sends q=8 decoders through the same per-request call. The native
kernels are replaced by Python fakes; no GPU is used.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/bench/relay11-pytest-shim \
  /data/venvs/1cat-main-integp6/bin/python -m pytest \
  tests/kernels/attention/test_sm70_flash_v100_grouped_per_request_fallback.py
"""

from __future__ import annotations

import types

import pytest
import torch

import vllm.v1.attention.backends.flash_attn_v100 as mod

KNOB = "VLLM_FLASH_V100_GROUPED_VERIFY_PER_REQUEST_FALLBACK"


def _impl_cls():
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and "_forward_nvfp4" in vars(obj):
            return obj
    pytest.skip("impl class with _forward_nvfp4 not found")


def _set_knob(monkeypatch, knob):
    if knob is None:
        monkeypatch.delenv(KNOB, raising=False)
    else:
        monkeypatch.setenv(KNOB, knob)
    monkeypatch.setattr(mod, "_logged_grouped_per_request_fallback_capture", False)
    monkeypatch.setattr(mod, "_logged_grouped_per_request_fallback_eager", False)
    monkeypatch.setattr(mod, "_logged_nvfp4_mixed_split_fired", False)
    monkeypatch.setattr(mod, "_logged_prefill_smallq_grouped_verify", False)
    monkeypatch.setattr(mod, "_logged_prefill_smallq_grouped_verify_gate", False)
    monkeypatch.setattr(mod, "_record_route", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_log_fp8_kv_cache_route", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_is_cuda_graph_capturing", lambda q: False)


def _make_self(cls, calls, *, marker=True):
    def fake_grouped(q, kc, vc, bt, sl, *, out, **kw):
        calls["grouped"].append(
            dict(
                rows=int(q.shape[0]),
                q_val=q[0, 0, 0].item(),
                bt_first=bt[:, 0].tolist(),
                seq_lens=sl.tolist(),
                one_pass=kw["one_pass"],
            )
        )
        out.copy_(q * 3)

    def fake_xqa(layer, qq, kc, vc, block_table, sl, am, *, out, **kw):
        calls["xqa"].append(
            dict(rows=int(qq.shape[0]), q_vals=qq[:, 0, 0].tolist(), seq_lens=sl.tolist())
        )
        out.copy_(qq * 10)

    def fake_scalar(qq, kc, vc, block_table, sl, *, out, **kw):
        calls["xqa"].append(
            dict(rows=int(qq.shape[0]), q_vals=qq[:, 0, 0].tolist(), seq_lens=sl.tolist())
        )
        out.copy_(qq * 10)

    def fake_bridge(**kwargs):
        calls["bridge"].append(int(kwargs["query"].shape[1]))
        kwargs["out"].fill_(-1)
        return kwargs["out"], True

    self = types.SimpleNamespace(
        use_dflash2_grouped_verify=True,
        use_dflash2_batched_grouped_verify=False,
        dflash2_grouped_verify_request_major_abi_version=0,
        dflash2_grouped_verify_max_query_tokens=8,
        dflash2_grouped_verify_min_model_len=32768,
        dflash2_grouped_verify_extra_pages=(),
        flash_attn_grouped_verify_paged=fake_grouped,
        kv_cache_dtype="fp8_e5m2",
        scale=1.0,
        smallq_decode_max_query_len=16,
        _flash_v100_window_size=lambda causal=True: (-1, -1),
        _call_flash_attn_smallq_decode_paged=fake_xqa,
        _run_fp8_prefill_bridge=fake_bridge,
        _nvfp4_first_chunk_fp16_applies=None,
        use_decode_xqa=False,
        flash_attn_decode_paged_xqa=None,
        _call_flash_attn_decode_paged=fake_scalar,
    )
    for name in (
        "_dflash2_grouped_verify_allowed",
        "_call_dflash2_grouped_verify",
        "_grouped_verify_per_request_enabled",
        "_grouped_verify_request_eligible",
        "_call_grouped_verify_request",
        "_log_grouped_per_request_fallback",
        "_try_grouped_verify_per_request",
        "_run_nvfp4_mixed_decode_rows",
        "_run_grouped_per_request_rows",
        "_run_prefill_prefix_decode_rows",
        "_run_prefill_paged_call",
        "_flash_v100_small_query_prefill_as_decode",
        "_nvfp4_first_chunk_fp16_applies",
    ):
        setattr(self, name, types.MethodType(getattr(cls, name), self))
    return self


def _meta(seq_lens, q_lens, *, marker=True, padded_reqs=None, actual=None):
    n = len(q_lens)
    total = sum(q_lens)
    if padded_reqs is not None:
        # FULL-cudagraph padding: tables/seq_lens carry dummy rows beyond the
        # real requests; num_actual_tokens counts only real tokens.
        pad = padded_reqs - n
        seq_lens = list(seq_lens) + [0] * pad
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    return types.SimpleNamespace(
        causal=True,
        ddtree_parent_ids=None,
        is_dflash_selector_target=marker,
        is_mtp_verify_target=False,
        max_model_len=131072,
        num_actual_tokens=total if actual is None else actual,
        num_reqs=n,
        max_query_len=max(q_lens),
        query_start_loc_cpu=qsl,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.arange(
            (n if padded_reqs is None else padded_reqs) * 4, dtype=torch.int32
        ).reshape(-1, 4),
        # persistent smallq metadata so the legacy per-row path is reachable
        smallq_decode_block_table=torch.zeros(total, 4, dtype=torch.int32),
        smallq_decode_seq_lens=torch.ones(total, dtype=torch.int32),
        smallq_query_start_loc=qsl,
    ), qsl


def _query(q_lens, qsl):
    total = sum(q_lens)
    q = torch.zeros(total, 6, 256, dtype=torch.float16)
    for r in range(len(q_lens)):
        q[int(qsl[r]) : int(qsl[r + 1])] = r + 1
    return q


def _cache():
    return torch.zeros(2, 1648, 1, 256, dtype=torch.uint8)


def _run_pure(monkeypatch, knob, q_lens, seq_lens, *, marker=True):
    cls = _impl_cls()
    _set_knob(monkeypatch, knob)
    calls = {"grouped": [], "xqa": [], "bridge": []}
    self = _make_self(cls, calls)
    meta, qsl = _meta(seq_lens, q_lens, marker=marker)
    q = _query(q_lens, qsl)
    out = torch.zeros_like(q)
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    kc = _cache()
    res = self._flash_v100_small_query_prefill_as_decode(
        layer, q, kc, kc, meta, out, qsl, meta.seq_lens
    )
    assert res is out
    return calls, out


SEQ10 = [4096, 70000, 131000, 4096, 70000, 131000, 4096, 70000, 131000, 5000]


def test_knob0_pure_decode_reqs10_is_todays_path(monkeypatch):
    calls, out = _run_pure(monkeypatch, "0", [8] * 10, SEQ10)
    assert calls["grouped"] == []
    assert len(calls["xqa"]) == 1 and calls["xqa"][0]["rows"] == 80


def test_knob1_pure_decode_reqs10_one_grouped_per_request(monkeypatch, caplog):
    with caplog.at_level("INFO"):
        calls, out = _run_pure(monkeypatch, "1", [8] * 10, SEQ10)
    assert calls["xqa"] == []
    assert len(calls["grouped"]) == 10
    for i, c in enumerate(calls["grouped"]):
        assert c["rows"] == 8 and c["q_val"] == i + 1
        assert c["bt_first"] == [4 * i]
        assert c["seq_lens"] == [SEQ10[i]]
        assert c["one_pass"] is True
        assert (out[8 * i : 8 * i + 8] == 3 * (i + 1)).all()
    msgs = [r.message for r in caplog.records]
    assert sum("per-request grouped fallback: n=10" in m for m in msgs) == 1
    assert any("during_cuda_graph_capture=False" in m for m in msgs)


def test_log_once_per_process_and_capture_flag(monkeypatch, caplog):
    cls = _impl_cls()
    _set_knob(monkeypatch, "1")
    monkeypatch.setattr(mod, "_is_cuda_graph_capturing", lambda q: True)
    self = types.SimpleNamespace()
    f = types.MethodType(cls._log_grouped_per_request_fallback, self)
    with caplog.at_level("INFO"):
        f(10, None)
        f(10, None)
    assert sum("per-request grouped fallback: n=10" in r.message for r in caplog.records) == 1
    assert "during_cuda_graph_capture=True" in caplog.records[-1].message


def test_knob1_marker_missing_keeps_xqa(monkeypatch):
    calls, out = _run_pure(monkeypatch, "1", [8] * 10, SEQ10, marker=False)
    assert calls["grouped"] == []
    assert len(calls["xqa"]) == 1


def test_knob1_non_uniform_batch_keeps_xqa(monkeypatch):
    # q=8 and q=5 mix: not the uniform capture-safe shape -> unchanged path
    calls, out = _run_pure(monkeypatch, "1", [8, 8, 5], [5000, 6000, 7000])
    assert calls["grouped"] == []
    assert len(calls["xqa"]) == 1


def test_knob1_batch_already_allowed_uses_whole_batch_gate_only(monkeypatch):
    # reqs=1 is B1 itself: handled by the existing gate, fallback not involved
    calls, out = _run_pure(monkeypatch, "1", [8], [5000])
    assert len(calls["grouped"]) == 1 and calls["xqa"] == []


# ---- mixed batch (NVFP4 P6 split) ------------------------------------------


def _run_mixed(monkeypatch, knob, q_lens, seq_lens):
    cls = _impl_cls()
    _set_knob(monkeypatch, knob)
    calls = {"grouped": [], "xqa": [], "bridge": []}
    self = _make_self(cls, calls)
    meta, qsl = _meta(seq_lens, q_lens)
    q = _query(q_lens, qsl)
    out = torch.zeros_like(q)
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    kc = _cache()
    consumed = self._run_nvfp4_mixed_decode_rows(
        layer, q, kc, kc, meta, out, qsl, meta.seq_lens
    )
    return calls, out, consumed


MQ = [1, 8, 1, 8, 1500]
MS = [5000, 70000, 6000, 131000, 9000]


def test_mixed_knob0_expands_q8_to_xqa_rows(monkeypatch):
    calls, out, consumed = _run_mixed(monkeypatch, "0", MQ, MS)
    assert calls["grouped"] == []
    assert len(calls["xqa"]) == 1 and calls["xqa"][0]["rows"] == 18
    assert consumed == {0, 1, 2, 3}


def test_mixed_knob1_q8_grouped_q1_still_split(monkeypatch):
    calls, out, consumed = _run_mixed(monkeypatch, "1", MQ, MS)
    assert consumed == {0, 1, 2, 3}
    assert [c["q_val"] for c in calls["grouped"]] == [2, 4]
    assert [c["bt_first"] for c in calls["grouped"]] == [[4], [12]]
    assert [c["seq_lens"] for c in calls["grouped"]] == [[70000], [131000]]
    # only the two q=1 rows remain on the XQA expansion (P6 unchanged)
    assert len(calls["xqa"]) == 1
    assert calls["xqa"][0]["rows"] == 2
    assert calls["xqa"][0]["seq_lens"] == [5000, 6000]
    assert out[0, 0, 0].item() == 10 and out[9, 0, 0].item() == 30
    assert (out[1:9, 0, 0] == 6).all()
    assert (out[10:18, 0, 0] == 12).all()


def test_mixed_knob1_only_q8_decoders_no_xqa_call(monkeypatch):
    calls, out, consumed = _run_mixed(monkeypatch, "1", [8, 8, 1500], [70000, 80000, 9000])
    assert consumed == {0, 1}
    assert len(calls["grouped"]) == 2 and calls["xqa"] == []


# ---- padding (FULL cudagraph) ------------------------------------------------


def test_knob1_padded_batch_runs_only_real_requests(monkeypatch):
    cls = _impl_cls()
    _set_knob(monkeypatch, "1")
    calls = {"grouped": [], "xqa": [], "bridge": []}
    self = _make_self(cls, calls)
    meta, qsl = _meta(SEQ10, [8] * 10, padded_reqs=16)
    assert meta.block_table.shape[0] == 16 and meta.seq_lens.shape[0] == 16
    q = _query([8] * 10, qsl)
    # query/output buffers are padded too (16*8 rows); padded rows untouched
    qp = torch.cat([q, torch.zeros(48, 6, 256, dtype=torch.float16)])
    out = torch.full_like(qp, -7)
    kc = _cache()
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    self._flash_v100_small_query_prefill_as_decode(
        layer, qp, kc, kc, meta, out, qsl, meta.seq_lens
    )
    assert len(calls["grouped"]) == 10 and calls["xqa"] == []
    assert [c["bt_first"] for c in calls["grouped"]] == [[4 * i] for i in range(10)]
    assert (out[80:] == -7).all()  # padded rows left untouched


def test_knob1_padded_log_prints_real_and_padded(monkeypatch, caplog):
    cls = _impl_cls()
    _set_knob(monkeypatch, "1")
    self = types.SimpleNamespace()
    f = types.MethodType(cls._log_grouped_per_request_fallback, self)
    with caplog.at_level("INFO"):
        f(10, None, 16)
    assert "n=10 (padded=16," in caplog.records[-1].message


def test_knob1_actual_tokens_disagree_with_uniform_keeps_xqa(monkeypatch):
    cls = _impl_cls()
    _set_knob(monkeypatch, "1")
    calls = {"grouped": [], "xqa": [], "bridge": []}
    self = _make_self(cls, calls)
    meta, qsl = _meta(SEQ10, [8] * 10, actual=75)  # 75 != real*8
    q = _query([8] * 10, qsl)
    out = torch.zeros_like(q)
    kc = _cache()
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    assert not self._try_grouped_verify_per_request(layer, q, kc, kc, meta, out, 75)


# ---- e5m2 mixed batch (_run_prefill_prefix_decode_rows) ----------------------


def _run_e5m2_mixed(monkeypatch, knob, q_lens, seq_lens):
    cls = _impl_cls()
    _set_knob(monkeypatch, knob)
    monkeypatch.setattr(mod, "_logged_prefill_prefix_decode_rows", False)
    calls = {"grouped": [], "xqa": [], "bridge": []}
    self = _make_self(cls, calls)
    meta, qsl = _meta(seq_lens, q_lens)
    q = _query(q_lens, qsl)
    out = torch.zeros_like(q)
    kc = _cache()
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    consumed = self._run_prefill_prefix_decode_rows(
        layer, q, kc, kc, meta, out, qsl, meta.seq_lens, (-1, -1)
    )
    return calls, out, consumed


def test_e5m2_mixed_knob0_expands_q8(monkeypatch):
    calls, out, consumed = _run_e5m2_mixed(monkeypatch, "0", MQ, MS)
    assert calls["grouped"] == [] and consumed == {0, 1, 2, 3}
    assert len(calls["xqa"]) == 1 and calls["xqa"][0]["rows"] == 18


def test_e5m2_mixed_knob1_q8_grouped_q1_decode_rows(monkeypatch):
    calls, out, consumed = _run_e5m2_mixed(monkeypatch, "1", MQ, MS)
    assert consumed == {0, 1, 2, 3}
    assert [c["q_val"] for c in calls["grouped"]] == [2, 4]
    assert [c["seq_lens"] for c in calls["grouped"]] == [[70000], [131000]]
    assert len(calls["xqa"]) == 1 and calls["xqa"][0]["rows"] == 2
    assert calls["xqa"][0]["seq_lens"] == [5000, 6000]
    assert (out[1:9, 0, 0] == 6).all() and (out[10:18, 0, 0] == 12).all()
    assert out[0, 0, 0].item() == 10 and out[9, 0, 0].item() == 30
