# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity: VLLM_FLASH_V100_GROUPED_VERIFY_PER_REQUEST_FALLBACK (plan 072 P7).

GPU-only (SM70, NVFP4 probe >= 3 for the nvfp4 cases, grouped-verify extension);
collected and skipped on CPU. NOT RUN when it was written (CPU-only session).

Real NVFP4 and E5M2 paged caches (page 1728 = a page size the grouped gate
admits natively), a batch of 10 requests with q=8 at ctx 4K / 70K / 131K mixed
(reqs=10 is rejected by the whole-batch gate, which only admits 2/4/8), and an
NVFP4 and E5M2 mixed batches (three q=8 decoders + one q=1 decoder + one 1500-token prefix
append).

  * knob=1, every q=8 request is BIT-IDENTICAL to running that request alone
    through the single-request B1 grouped one-pass (and the native op was
    called once per request, i.e. the fallback really engaged);
  * knob=0 (per-row XQA) and knob=1 both stay within rel-L2 <= 2e-3 of a float32
    causal reference over the dequantized cache.

Graph-baked loop count (test_graph_baked_16_loop_with_6_zero_seqlen_rows): the
loop count ``real = num_actual_tokens // 8`` is a Python int fixed when the
CUDA graph is captured, while ``block_table`` / ``seq_lens`` are device tensors
refreshed on every replay. A graph captured at 16 requests and replayed with 10
live ones therefore still issues 16 B1 launches, and the 6 padded tail requests
carry ``seq_len == 0`` (block_table rows of 0). The native kernel's behaviour
for ``seq_len == 0`` could not be read from source (only the .so ships), so the
test pins it empirically: no exception, the 10 real outputs bit-identical to
B1 alone, no NaN/Inf in the real slices, and what B1 writes into the padded
slices is printed (-s) for the record.

Run in a GPU window (needs an idle V100; never on :8001's cards):

  CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=/data/src/1cat-wt-grpfb \\
    /data/venvs/1cat-main-integp6/bin/python -m pytest -q -p no:cacheprovider -s \\
    tests/kernels/attention/test_sm70_flash_v100_grouped_per_request_fallback_gpu.py
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

KNOB = "VLLM_FLASH_V100_GROUPED_VERIFY_PER_REQUEST_FALLBACK"
HEAD_DIM = 256
Q_HEADS = 6
PAGE = 1728  # in the grouped gate's native page set (1648/1728/3296/3456)
MAX_MODEL_LEN = 262144
SCALE = HEAD_DIM**-0.5
CTX10 = [4096, 70000, 131072, 4096, 70000, 131072, 4096, 70000, 131072, 20000]
MIXED_ROWS = [(8, 4096), (8, 70000), (8, 131072), (1, 20000), (1500, 8192 + 1500)]


def _require(dtype: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Flash-V100 grouped verify is SM70-only")
    fa = pytest.importorskip("flash_attn_v100")
    if not hasattr(fa, "flash_attn_grouped_verify_paged"):
        pytest.skip("Flash-V100 extension lacks grouped verification")
    if dtype == "nvfp4":
        if not hasattr(fa, "flash_attn_nvfp4_kv_available"):
            pytest.skip("Flash-V100 lacks the NVFP4 probe")
        if not fa.flash_attn_nvfp4_kv_available(min_version=3):
            pytest.skip("Flash-V100 extension lacks the NVFP4 grouped verifier (v3)")
    return fa


def _make_impl(monkeypatch, dtype: str):
    import vllm.config
    from vllm.v1.attention.backends import flash_attn_v100 as f

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            cache_config=SimpleNamespace(block_size=PAGE),
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        ),
    )
    impl = f.FlashAttnV100Impl(
        num_heads=Q_HEADS,
        head_size=HEAD_DIM,
        scale=SCALE,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=dtype,
    )
    assert impl.use_dflash2_grouped_verify, "grouped verifier not available"
    calls = []
    real = impl.flash_attn_grouped_verify_paged

    def spy(*a, **k):
        calls.append(int(a[4].shape[0]))  # request rows of this native call
        return real(*a, **k)

    impl.flash_attn_grouped_verify_paged = spy
    return impl, calls


def _build_cache(dtype: str, seq_lens: list[int], seed: int):
    """Returns cache [blocks,2,PAGE,1,row], block_table, per-request K/V (fp32)."""
    torch.manual_seed(seed)
    dev = "cuda"
    pages = [math.ceil(s / PAGE) for s in seq_lens]
    blocks = sum(pages) + 3
    physical = torch.randperm(blocks, dtype=torch.int32, device=dev)
    table = torch.zeros(len(seq_lens), max(pages), dtype=torch.int32, device=dev)
    off = 0
    for r, n in enumerate(pages):
        table[r, :n] = physical[off : off + n]
        off += n
    if dtype == "nvfp4":
        from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv

        cache = torch.zeros(
            blocks, 2, PAGE, 1, nvfp4_kv.nvfp4_kv_row_bytes(HEAD_DIM),
            dtype=torch.uint8, device=dev,
        )
        for r, s in enumerate(seq_lens):
            token = torch.arange(s, device=dev)
            k = torch.randn(s, 1, HEAD_DIM, device=dev).mul_(0.5).half()
            v = torch.randn(s, 1, HEAD_DIM, device=dev).mul_(0.5).half()
            slots = table[r].long()[token // PAGE] * PAGE + token % PAGE
            nvfp4_kv.reshape_and_cache_nvfp4_reference(
                k, v, cache, slots.to(torch.int32)
            )
        (kd, vd), (ks, vs) = nvfp4_kv.nvfp4_kv_split_views(cache)
        kfull = nvfp4_kv.dequantize_kv_nvfp4(kd, ks, out_dtype=torch.float16)
        vfull = nvfp4_kv.dequantize_kv_nvfp4(vd, vs, out_dtype=torch.float16)
    else:
        src = torch.randn(blocks, 2, PAGE, 1, HEAD_DIM, device=dev).mul_(0.25).half()
        cache = src.to(torch.float8_e5m2).view(torch.uint8)
        kfull = cache[:, 0].view(torch.float8_e5m2).half()
        vfull = cache[:, 1].view(torch.float8_e5m2).half()
    kv = []
    for r, s in enumerate(seq_lens):
        idx = table[r, : pages[r]].long()
        kv.append(
            (
                kfull[idx].reshape(-1, 1, HEAD_DIM)[:s, 0].float(),
                vfull[idx].reshape(-1, 1, HEAD_DIM)[:s, 0].float(),
            )
        )
    return cache, table, kv


def _reference(q8, k, v, seq_len):
    out = torch.empty(q8.shape, dtype=torch.float32, device=q8.device)
    for t in range(q8.shape[0]):
        n = seq_len - q8.shape[0] + 1 + t
        scores = (q8[t].float() @ k[:n].T) * SCALE
        out[t] = torch.softmax(scores, dim=-1) @ v[:n]
    return out


def _meta(table, seq_lens_cpu, q_lens):
    qsl = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32)
    total = int(qsl[-1])
    return (
        SimpleNamespace(
            causal=True,
            ddtree_parent_ids=None,
            is_dflash_selector_target=True,
            is_mtp_verify_target=False,
            num_actual_tokens=total,
            num_reqs=len(q_lens),
            max_query_len=max(q_lens),
            max_model_len=MAX_MODEL_LEN,
            query_start_loc=qsl.cuda(),
            query_start_loc_cpu=qsl,
            seq_lens=torch.tensor(seq_lens_cpu, dtype=torch.int32).cuda(),
            seq_lens_cpu=torch.tensor(seq_lens_cpu, dtype=torch.int32),
            block_table=table,
        ),
        qsl,
    )


LAYER = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)


def _verify(impl, cache, query, meta, qsl):
    out = torch.full_like(query, -99.0)
    impl._flash_v100_small_query_prefill_as_decode(
        LAYER, query, cache[:, 0], cache[:, 1], meta, out, qsl, meta.seq_lens
    )
    torch.cuda.synchronize()
    return out


def _alone(impl, cache, table, query, seq_lens, i):
    """Request i alone through the existing single-request B1 route."""
    meta, qsl = _meta(table[i : i + 1], [seq_lens[i]], [8])
    return _verify(impl, cache, query[8 * i : 8 * i + 8], meta, qsl)


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8_e5m2"])
def test_reqs10_knob1_bitwise_equals_b1_alone(monkeypatch, dtype):
    _require(dtype)
    impl, calls = _make_impl(monkeypatch, dtype)
    cache, table, _ = _build_cache(dtype, CTX10, seed=7201)
    query = torch.randn(80, Q_HEADS, HEAD_DIM, device="cuda").mul_(0.5).half()
    meta, qsl = _meta(table, CTX10, [8] * 10)

    monkeypatch.setenv(KNOB, "1")
    calls.clear()
    got = _verify(impl, cache, query, meta, qsl)
    assert calls == [1] * 10, f"expected 10 B1 native calls, got {calls}"
    assert not (got == -99.0).any()
    for i in range(10):
        calls.clear()
        alone = _alone(impl, cache, table, query, CTX10, i)
        assert calls == [1]
        assert torch.equal(got[8 * i : 8 * i + 8], alone), f"request {i} differs"


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8_e5m2"])
def test_reqs10_knob0_and_knob1_match_float32_reference(monkeypatch, dtype):
    _require(dtype)
    impl, calls = _make_impl(monkeypatch, dtype)
    cache, table, kv = _build_cache(dtype, CTX10, seed=7202)
    query = torch.randn(80, Q_HEADS, HEAD_DIM, device="cuda").mul_(0.5).half()
    meta, qsl = _meta(table, CTX10, [8] * 10)
    ref = torch.cat(
        [_reference(query[8 * i : 8 * i + 8], *kv[i], CTX10[i]) for i in range(10)]
    )
    for knob in ("0", "1"):
        monkeypatch.setenv(KNOB, knob)
        calls.clear()
        got = _verify(impl, cache, query, meta, qsl).float()
        assert (calls == [1] * 10) == (knob == "1"), (knob, calls)
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"{dtype} knob={knob}: rel-L2 vs float32 = {rel:.3e}")
        assert rel <= 2e-3, f"knob={knob} rel-L2 {rel:.3e}"


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8_e5m2"])
def test_mixed_batch_q8_rows_use_per_request_grouped(monkeypatch, dtype):
    _require(dtype)
    impl, calls = _make_impl(monkeypatch, dtype)
    seq_lens = [s for _, s in MIXED_ROWS]
    q_lens = [q for q, _ in MIXED_ROWS]
    cache, table, kv = _build_cache(dtype, seq_lens, seed=7203)
    meta, qsl = _meta(table, seq_lens, q_lens)
    query = torch.randn(int(qsl[-1]), Q_HEADS, HEAD_DIM, device="cuda").mul_(0.5).half()

    if dtype == "nvfp4":
        # Reserve the bridge workspace the way profile_run does.
        prof = torch.empty_like(query[: MIXED_ROWS[-1][0]])
        impl.forward(LAYER, prof, prof[:, :1], prof[:, :1], cache, None, prof.clone())
        kvz = torch.zeros(query.shape[0], 1, HEAD_DIM, dtype=torch.float16, device="cuda")

        def forward():
            out = torch.full_like(query, -99.0)
            impl.forward(LAYER, query, kvz, kvz, cache, meta, out)
            torch.cuda.synchronize()
            return out
    else:
        # e5m2 mixed batch: the chunked-prefill-with-prefix route, which owns
        # _run_prefill_prefix_decode_rows. kv layout there is [2, blocks, ...].
        kv_cache = torch.stack([cache[:, 0], cache[:, 1]])

        def forward():
            out = torch.full_like(query, -99.0)
            impl._flash_v100_prefill_with_prefix(
                LAYER, query, None, None, kv_cache, meta, out
            )
            torch.cuda.synchronize()
            return out

    monkeypatch.setenv(KNOB, "0")
    calls.clear()
    off = forward()
    assert calls == []  # q=8 decoders were expanded to per-row decode
    monkeypatch.setenv(KNOB, "1")
    calls.clear()
    on = forward()
    assert calls == [1, 1, 1], calls  # exactly the three q=8 decoders
    assert not (off == -99.0).any() and not (on == -99.0).any()

    for i in range(3):  # bit-identical to B1 alone
        s = int(qsl[i])
        alone = _alone(impl, cache, table, _pad(query, s, i), seq_lens, i)
        assert torch.equal(on[s : s + 8], alone), f"decoder {i} differs"
        ref = _reference(query[s : s + 8], *kv[i], seq_lens[i])
        for name, o in (("knob0", off), ("knob1", on)):
            rel = ((o[s : s + 8].float() - ref).norm() / ref.norm()).item()
            print(f"mixed decoder {i} {name}: rel-L2 {rel:.3e}")
            assert rel <= 2e-3
    # the q=1 decoder and the append chunk are untouched by the knob
    s1 = int(qsl[3])
    assert torch.equal(on[s1:], off[s1:])


def _pad(query, start, i):
    """Place the 8 query rows of request i at rows [8*i, 8*i+8) of a buffer so
    ``_alone`` (which slices ``8*i``) reads exactly those rows."""
    buf = torch.zeros(8 * (i + 1), Q_HEADS, HEAD_DIM, dtype=query.dtype, device=query.device)
    buf[8 * i : 8 * i + 8] = query[start : start + 8]
    return buf


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8_e5m2"])
def test_graph_baked_16_loop_with_6_zero_seqlen_rows(monkeypatch, dtype):
    _require(dtype)
    impl, calls = _make_impl(monkeypatch, dtype)
    cache, table10, _ = _build_cache(dtype, CTX10, seed=7204)
    # Padded tail as after capture@16 / replay with 10 live requests.
    table = torch.zeros(16, table10.shape[1], dtype=torch.int32, device="cuda")
    table[:10] = table10
    seq_lens = CTX10 + [0] * 6
    query = torch.randn(128, Q_HEADS, HEAD_DIM, device="cuda").mul_(0.5).half()
    meta, qsl = _meta(table, seq_lens, [8] * 16)
    assert meta.num_actual_tokens == 128  # loop count baked at 16

    monkeypatch.setenv(KNOB, "1")
    calls.clear()
    out = torch.full_like(query, -99.0)
    ok = impl._try_grouped_verify_per_request(
        LAYER, query, cache[:, 0], cache[:, 1], meta, out, 128
    )
    torch.cuda.synchronize()
    assert ok and calls == [1] * 16, calls

    real, tail = out[:80], out[80:]
    assert torch.isfinite(real).all(), "NaN/Inf leaked into the real requests' rows"
    for i in range(10):
        meta1, qsl1 = _meta(table[i : i + 1], [seq_lens[i]], [8])
        alone = _verify(impl, cache, query[8 * i : 8 * i + 8], meta1, qsl1)
        assert torch.equal(real[8 * i : 8 * i + 8], alone), f"request {i} differs"
    untouched = (tail == -99.0).all().item()
    print(
        f"[{dtype}] B1 on seq_len=0 rows: untouched={untouched} "
        f"nan={torch.isnan(tail).sum().item()} inf={torch.isinf(tail).sum().item()} "
        f"all_zero={(tail == 0).all().item()} "
        f"min={tail.float().min().item():.4g} max={tail.float().max().item():.4g}"
    )
