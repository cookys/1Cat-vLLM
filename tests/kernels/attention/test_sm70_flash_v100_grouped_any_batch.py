# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P9 CPU dispatch/metadata tests. Native math is mocked, not GPU parity."""

import types

import pytest
import torch
from test_sm70_flash_v100_grouped_per_request_fallback import (
    _cache,
    _impl_cls,
    _make_self,
    _meta,
    _query,
    _run_mixed,
    _set_knob,
)

import vllm.envs as envs
import vllm.v1.attention.backends.flash_attn_v100 as mod

ANY = "VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY_ANY_BATCH"


def setup_case(monkeypatch, n, knob, *, padded=False):
    _set_knob(monkeypatch, "1")
    if knob is None:
        monkeypatch.delenv(ANY, raising=False)
    else:
        monkeypatch.setenv(ANY, knob)
    calls = {"grouped": [], "xqa": [], "bridge": []}
    impl = _make_self(_impl_cls(), calls)
    impl.use_dflash2_batched_grouped_verify = True
    impl.dflash2_grouped_verify_request_major_abi_version = 1
    impl.dflash2_grouped_verify_any_batch_max_reqs = 16
    monkeypatch.setattr(mod, "_logged_grouped_verify_any_batches", set())
    seqs = [40000 + i for i in range(n)]
    if padded:
        seqs[10:] = [0] * (n - 10)
    meta, qsl = _meta(seqs, [8] * n)
    q = _query([8] * n, qsl)
    return impl, calls, meta, qsl, q, _cache()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 10, 16])
@pytest.mark.parametrize("knob", [None, "0", "1"])
def test_dispatch_metadata_and_mock_output_equal_b1(monkeypatch, n, knob):
    impl, calls, meta, qsl, q, cache = setup_case(monkeypatch, n, knob, padded=n == 16)
    out = torch.full_like(q, -99)
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    impl._flash_v100_small_query_prefill_as_decode(
        layer, q, cache, cache, meta, out, qsl, meta.seq_lens
    )
    whole = n in (1, 2, 4, 8) or knob == "1"
    assert len(calls["grouped"]) == (1 if whole else n)
    assert calls["xqa"] == calls["bridge"] == []
    if whole:
        assert calls["grouped"][0]["rows"] == n * 8
        assert calls["grouped"][0]["bt_first"] == meta.block_table[:, 0].tolist()
        assert calls["grouped"][0]["seq_lens"] == meta.seq_lens.tolist()
    else:
        assert [c["bt_first"] for c in calls["grouped"]] == [[i * 4] for i in range(n)]
    assert torch.equal(out, q * 3)  # fake math only; CUDA equality is a GPU gate


@pytest.mark.parametrize(
    "guard", ["parent", "abi", "marker", "q", "rows", "dtype", "table", "probe"]
)
def test_any_batch_keeps_other_guards(monkeypatch, guard):
    impl, _, meta, _, q, cache = setup_case(monkeypatch, 10, "1")
    if guard == "parent":
        impl.use_dflash2_batched_grouped_verify = False
    elif guard == "abi":
        impl.dflash2_grouped_verify_request_major_abi_version = 0
    elif guard == "marker":
        meta.is_dflash_selector_target = False
    elif guard == "q":
        meta.max_query_len = 16
    elif guard == "rows":
        q = q[:-1]
    elif guard == "dtype":
        impl.kv_cache_dtype = "fp8_e4m3"  # repaired FP32 path stays separate
    elif guard == "table":
        meta.block_table = meta.block_table[:-1]
    else:
        impl.kv_cache_dtype = "nvfp4"
        cache = torch.zeros(2, 1648, 1, 144, dtype=torch.uint8)
        monkeypatch.setattr(mod, "_grouped_verify_nvfp4_available", lambda: False)
    assert not impl._dflash2_grouped_verify_allowed(
        q, cache, cache, meta, num_query_tokens=q.shape[0]
    )


def test_nvfp4_uses_existing_v3_probe(monkeypatch):
    impl, _, meta, _, q, _ = setup_case(monkeypatch, 10, "1")
    impl.kv_cache_dtype = "nvfp4"
    cache = torch.zeros(2, 4096, 1, 144, dtype=torch.uint8)
    impl.dflash2_grouped_verify_extra_pages = (4096,)
    monkeypatch.setattr(mod, "_grouped_verify_nvfp4_available", lambda: True)
    assert impl._dflash2_grouped_verify_allowed(
        q, cache, cache, meta, num_query_tokens=80
    )


def test_mixed_request_paths_unchanged(monkeypatch):
    monkeypatch.setenv(ANY, "0")
    off_calls, off, off_consumed = _run_mixed(
        monkeypatch, "1", [8, 1, 1500], [40000] * 3
    )
    monkeypatch.setenv(ANY, "1")
    on_calls, on, on_consumed = _run_mixed(monkeypatch, "1", [8, 1, 1500], [40000] * 3)
    assert on_calls == off_calls
    assert on_consumed == off_consumed
    assert torch.equal(on, off)


def test_real_env_registration(monkeypatch):
    monkeypatch.delenv(ANY, raising=False)
    assert getattr(envs, ANY) is False
    monkeypatch.setenv(ANY, "1")
    assert getattr(envs, ANY) is True


def test_above_configured_capacity_rejected(monkeypatch):
    impl, _, meta, _, q, cache = setup_case(monkeypatch, 17, "1")
    assert not impl._dflash2_grouped_verify_allowed(
        q, cache, cache, meta, num_query_tokens=q.shape[0]
    )


def test_new_route_log_once_per_bucket_and_capture_phase(monkeypatch, caplog):
    impl, calls, meta, qsl, q, cache = setup_case(monkeypatch, 16, "1", padded=True)
    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    with caplog.at_level("INFO"):
        for capturing in (False, False, True, True):
            monkeypatch.setattr(
                mod, "_is_cuda_graph_capturing", lambda q, value=capturing: value
            )
            impl._flash_v100_small_query_prefill_as_decode(
                layer, q, cache, cache, meta, torch.empty_like(q), qsl, meta.seq_lens
            )
    messages = [
        r.message for r in caplog.records if "any-batch grouped verify:" in r.message
    ]
    assert len(messages) == 2
    assert all("B=16 bucket=16" in m and "configured_max=16" in m for m in messages)
    assert "capture=False" in messages[0] and "capture=True" in messages[1]
    assert all(c["rows"] == 128 for c in calls["grouped"])
