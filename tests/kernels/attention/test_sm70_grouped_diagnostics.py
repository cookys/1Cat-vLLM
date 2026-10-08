# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only diagnostics against serving metadata, never GPU math validation."""

import json
import types
from dataclasses import fields

import numpy as np
import pytest
import torch
from test_sm70_flash_v100_grouped_any_batch import setup_case
from test_sm70_flash_v100_grouped_per_request_fallback import _run_mixed

import vllm.v1.attention.backends.flash_attn_v100 as mod
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends import sm70_grouped_diagnostics as diag
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch


@pytest.fixture(autouse=True)
def diagnostic_env(monkeypatch):
    monkeypatch.setenv(diag.KNOB, "1")
    monkeypatch.setattr(diag, "_seen", set())


def records(caplog):
    prefix = "SM70 grouped route diagnostic: "
    return [
        json.loads(r.message.removeprefix(prefix))
        for r in caplog.records
        if r.message.startswith(prefix)
    ]


def test_off_reads_no_metadata_and_logs_nothing(monkeypatch, caplog):
    monkeypatch.setenv(diag.KNOB, "0")
    with caplog.at_level("INFO"):
        diag.dispatch(mod.logger, None, None, dummy_run=False, is_profile=False)
        diag.emit(mod.logger, "unused")
        impl, _, meta, qsl, q, cache = setup_case(monkeypatch, 10, "1")
        impl._flash_v100_small_query_prefill_as_decode(
            types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
            q,
            cache,
            cache,
            meta,
            torch.empty_like(q),
            qsl,
            meta.seq_lens,
        )
    assert records(caplog) == []


def test_real_metadata_once_per_capture_state_no_device_reads(monkeypatch, caplog):
    impl, _, meta, _, q, cache = setup_case(monkeypatch, 12, "1")

    def forbidden(*a, **kw):
        raise AssertionError("diagnostic must not read device values")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with caplog.at_level("INFO"):
        for capturing in (False, False, True, True):
            monkeypatch.setattr(
                mod, "_is_cuda_graph_capturing", lambda q, value=capturing: value
            )
            assert impl._dflash2_grouped_verify_allowed(
                q, cache, cache, meta, num_query_tokens=96
            )
    rows = records(caplog)
    assert len(rows) == 2
    assert [r["capture"] for r in rows] == [False, True]
    assert all(r["B"] == 12 and r["num_actual_tokens"] == 96 for r in rows)
    assert all(r["route"] == "gate_accept" and r["reasons"] == [] for r in rows)


def test_padded_eager_actual_tokens_rejection_and_b1_route(monkeypatch, caplog):
    impl, calls, meta, _, q, cache = setup_case(monkeypatch, 12, "1")
    meta.num_actual_tokens = 80
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    with caplog.at_level("INFO"):
        assert not impl._dflash2_grouped_verify_allowed(
            q, cache, cache, meta, num_query_tokens=80
        )
        assert impl._try_grouped_verify_per_request(
            layer, q, cache, cache, meta, torch.empty_like(q), 80
        )
    rows = records(caplog)
    reject = next(r for r in rows if r["route"] == "gate_reject")
    assert reject["B"] == 12 and reject["gate_tokens"] == 80
    assert {"tokens_ne_B_times_8", "query_shape"} <= set(reject["reasons"])
    selected = next(r for r in rows if r["route"] == "per_request_uniform")
    assert selected["selected_requests"] == 10 and selected["block_table_rows"] == 12
    assert any(
        r["route"] == "native_grouped_call"
        and r["metadata_scope"] == "B1_view"
        and r["B"] == 1
        for r in rows
    )
    assert len(calls["grouped"]) == 10


@pytest.mark.parametrize(
    "guard,reason",
    [
        ("marker", "target_marker"),
        ("shape", "max_query_len_ne_8"),
        ("parent", "batched_disabled"),
        ("abi", "request_major_abi"),
        ("dtype", "kv_dtype_or_probe"),
    ],
)
def test_named_gate_reasons(monkeypatch, caplog, guard, reason):
    impl, _, meta, _, q, cache = setup_case(monkeypatch, 10, "1")
    if guard == "marker":
        meta.is_dflash_selector_target = False
    elif guard == "shape":
        meta.max_query_len = 9
    elif guard == "parent":
        impl.use_dflash2_batched_grouped_verify = False
    elif guard == "abi":
        impl.dflash2_grouped_verify_request_major_abi_version = 0
    else:
        impl.kv_cache_dtype = "fp8_e4m3"
    with caplog.at_level("INFO"):
        assert not impl._dflash2_grouped_verify_allowed(
            q, cache, cache, meta, num_query_tokens=80
        )
    assert reason in records(caplog)[0]["reasons"]


def test_mixed_log_keeps_parent_batch_and_selected_count(monkeypatch, caplog):
    with caplog.at_level("INFO"):
        calls, _, _ = _run_mixed(monkeypatch, "1", [8, 1, 1500], [40000] * 3)
    row = next(r for r in records(caplog) if r["route"] == "per_request_mixed")
    assert row["B"] == 3 and row["selected_requests"] == 1
    assert row["parent_query_lens"] == [8, 1, 1500]
    assert len(calls["grouped"]) == 1


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE]
)
def test_real_input_batch_dispatch_logs_live_vs_bucket(caplog, mode):
    # Construct the real dataclass without initializing any GPU buffers. Only
    # these declared host fields are read; every other field is a poison None.
    values = {f.name: None for f in fields(InputBatch)}
    values.update(
        num_reqs=10,
        num_tokens=80,
        num_scheduled_tokens=np.full(10, 8),
        is_prefilling_np=np.zeros(10, dtype=np.bool_),
    )
    batch = InputBatch(**values)
    desc = BatchExecutionDescriptor(mode, 96, 12, 8)
    with caplog.at_level("INFO"):
        diag.dispatch(mod.logger, desc, batch, dummy_run=False, is_profile=False)
        diag.dispatch(mod.logger, desc, batch, dummy_run=False, is_profile=False)
    (row,) = records(caplog)
    assert (
        row["B_live"],
        row["B_bucket"],
        row["num_actual_tokens"],
        row["padded_tokens"],
    ) == (10, 12, 80, 96)
    assert row["runtime_mode"] == str(mode)
    assert row["query_length_histogram"] == [[8, 10]]
    assert row["prefill_requests"] == 0 and not row["capture"]
