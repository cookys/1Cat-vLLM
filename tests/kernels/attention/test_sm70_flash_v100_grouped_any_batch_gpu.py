# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P9 GPU gate, collected/skipped in the CPU-only development session.

Real-row outputs must be bit-identical to B1. Padding output has no contract:
the existing native combine can read stale partials for zero-seqlen rows.
Capture/replay 16 -> 10 -> 16 verifies that this cannot affect real rows.
"""

import pytest
import torch
from test_sm70_flash_v100_grouped_per_request_fallback_gpu import (
    CTX10,
    HEAD_DIM,
    KNOB,
    LAYER,
    MIXED_ROWS,
    Q_HEADS,
    _build_cache,
    _make_impl,
    _meta,
    _reference,
    _require,
    _verify,
)

ANY = "VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY_ANY_BATCH"


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8_e5m2"])
@pytest.mark.parametrize("bucket", [10, 16])
def test_batched_q8_and_graph_replay_equal_b1(monkeypatch, dtype, bucket):
    _require(dtype)
    monkeypatch.setenv(KNOB, "1")
    monkeypatch.setenv("VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY", "1")
    monkeypatch.setenv(ANY, "1")  # constructor freezes the configured capacity
    impl, calls = _make_impl(monkeypatch, dtype)
    assert impl.use_dflash2_batched_grouped_verify
    assert impl.dflash2_grouped_verify_request_major_abi_version >= 1
    seqs = CTX10 + [4096] * (bucket - 10)
    cache, table, _ = _build_cache(dtype, seqs, seed=7901)
    q = torch.randn(bucket * 8, Q_HEADS, HEAD_DIM, device="cuda").mul_(0.5).half()
    meta, qsl = _meta(table, seqs, [8] * bucket)
    monkeypatch.setenv(ANY, "0")
    calls.clear()
    off = _verify(impl, cache, q, meta, qsl)
    assert calls == [1] * bucket
    monkeypatch.setenv(ANY, "1")
    calls.clear()
    on = _verify(impl, cache, q, meta, qsl)
    assert calls == [bucket]
    assert torch.equal(on, off)

    out = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl._flash_v100_small_query_prefill_as_decode(
                LAYER, q, cache[:, 0], cache[:, 1], meta, out, qsl, meta.seq_lens
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    calls.clear()
    with torch.cuda.graph(graph, stream=stream):
        impl._flash_v100_small_query_prefill_as_decode(
            LAYER, q, cache[:, 0], cache[:, 1], meta, out, qsl, meta.seq_lens
        )
    assert calls == [bucket]
    for active in (bucket, 10, bucket):
        meta.seq_lens.copy_(
            torch.tensor(
                seqs[:active] + [0] * (bucket - active),
                device="cuda",
                dtype=torch.int32,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out[: active * 8]).all()
        assert torch.equal(out[: active * 8], off[: active * 8])


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8_e5m2"])
def test_mixed_p9_append_equal_and_q1_p7g_bound(monkeypatch, dtype):
    _require(dtype)
    monkeypatch.setenv(KNOB, "1")
    monkeypatch.setenv("VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY", "1")
    monkeypatch.setenv(ANY, "1")
    impl, _ = _make_impl(monkeypatch, dtype)
    seqs = [s for _, s in MIXED_ROWS]
    qlens = [q for q, _ in MIXED_ROWS]
    cache, table, kv = _build_cache(dtype, seqs, seed=7902)
    meta, qsl = _meta(table, seqs, qlens)
    query = torch.randn(int(qsl[-1]), Q_HEADS, HEAD_DIM, device="cuda").mul_(0.5).half()
    if dtype == "nvfp4":
        prof = torch.empty_like(query[: qlens[-1]])
        impl.forward(LAYER, prof, prof[:, :1], prof[:, :1], cache, None, prof.clone())
        kvz = torch.zeros(
            query.shape[0], 1, HEAD_DIM, dtype=torch.float16, device="cuda"
        )

        def run():
            out = torch.full_like(query, -99)
            impl.forward(LAYER, query, kvz, kvz, cache, meta, out)
            return out
    else:
        stacked = torch.stack([cache[:, 0], cache[:, 1]])

        def run():
            out = torch.full_like(query, -99)
            impl._flash_v100_prefill_with_prefix(
                LAYER, query, None, None, stacked, meta, out
            )
            return out

    monkeypatch.setenv(ANY, "0")
    off = run()
    monkeypatch.setenv(ANY, "1")
    on = run()
    torch.cuda.synchronize()
    assert torch.equal(on[int(qsl[-2]) :], off[int(qsl[-2]) :])
    start = int(qsl[3])
    ref = _reference(query[start : start + 1], *kv[3], seqs[3])
    err_off = ((off[start : start + 1].float() - ref).norm() / ref.norm()).item()
    err_on = ((on[start : start + 1].float() - ref).norm() / ref.norm()).item()
    assert err_on <= 2e-3 and err_on <= 1.25 * err_off
    assert torch.equal(on, off)  # P9 does not reroute mixed batches
