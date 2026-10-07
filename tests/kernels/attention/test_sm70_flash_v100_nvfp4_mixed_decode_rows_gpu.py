# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2 parity: NVFP4 mixed-batch decode-row split on a real NVFP4 KV cache.

GPU-only (SM70, native NVFP4 bridge version >= 2 and XQA reader); collected and
skipped on CPU. Builds one real mixed batch on the NVFP4 page layout:

  3 decode rows (q=1) at ctx 4096 / 70000 / 131072,
  1 DFlash2-style verify row (q=8, ctx 20000),
  1 append chunk (q=1500 over an 8192-token prefix).

It runs ``FlashAttnV100Impl.forward`` with VLLM_FLASH_V100_NVFP4_PREFIX_DECODE_ROWS=1
(decode/verify rows on the paged NVFP4 decode path) and =0 (every request through
the FP16 prefix bridge). Decode/verify rows must agree within the tolerance of the
existing NVFP4-vs-fp16-dequant tests (atol/rtol 2e-3, fp32 rel-L2 <= 2e-3); the
append row takes the bridge in both runs and must be bit-identical.

Run in a GPU window (needs an idle V100):

  CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=/data/src/1cat-wt-fable-p6 \\
    /data/venvs/1cat-p070/bin/python -m pytest -q -p no:cacheprovider -s \\
    tests/kernels/attention/test_sm70_flash_v100_nvfp4_mixed_decode_rows_gpu.py
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

KNOB = "VLLM_FLASH_V100_NVFP4_PREFIX_DECODE_ROWS"
HEAD_DIM = 256
Q_HEADS = 6
PAGE = 2048
MAX_MODEL_LEN = 262144

# (query_len, seq_len) per request, in batch order.
ROWS = [(1, 4096), (1, 70000), (1, 131072), (8, 20000), (1500, 8192 + 1500)]


def _require_gpu_nvfp4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Flash-V100 NVFP4 is SM70-only")
    fa = pytest.importorskip("flash_attn_v100")
    if not hasattr(fa, "flash_attn_nvfp4_kv_available"):
        pytest.skip("Flash-V100 Python package lacks the NVFP4 probe")
    if not fa.flash_attn_nvfp4_kv_available(min_version=2):
        pytest.skip("Flash-V100 extension lacks the NVFP4 KV bridge")
    return fa


def _make_impl(monkeypatch):
    import vllm.config
    from vllm.v1.attention.backends import flash_attn_v100 as f

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            cache_config=SimpleNamespace(block_size=PAGE),
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        ),
    )
    return f.FlashAttnV100Impl(
        num_heads=Q_HEADS,
        head_size=HEAD_DIM,
        scale=HEAD_DIM**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
    )


def _build_batch(seed: int):
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv

    torch.manual_seed(seed)
    dev = "cuda"
    pages = [math.ceil(s / PAGE) for _, s in ROWS]
    blocks = sum(pages) + 3
    cache = torch.zeros(
        blocks, 2, PAGE, 1, nvfp4_kv.nvfp4_kv_row_bytes(HEAD_DIM),
        dtype=torch.uint8, device=dev,
    )
    physical = torch.randperm(blocks, dtype=torch.int32, device=dev)
    table = torch.zeros(len(ROWS), max(pages), dtype=torch.int32, device=dev)
    off = 0
    for r, n in enumerate(pages):
        table[r, :n] = physical[off : off + n]
        off += n
    for r, (_, seq_len) in enumerate(ROWS):
        token = torch.arange(seq_len, device=dev)
        keys = torch.randn(seq_len, 1, HEAD_DIM, device=dev).mul_(0.5).half()
        values = torch.randn(seq_len, 1, HEAD_DIM, device=dev).mul_(0.5).half()
        slots = table[r].long()[token // PAGE] * PAGE + token % PAGE
        nvfp4_kv.reshape_and_cache_nvfp4_reference(
            keys, values, cache, slots.to(torch.int32)
        )
    q_lens = [q for q, _ in ROWS]
    qsl_cpu = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32)
    seq_cpu = torch.tensor([s for _, s in ROWS], dtype=torch.int32)
    total = int(qsl_cpu[-1])
    query = torch.randn(total, Q_HEADS, HEAD_DIM, device=dev).half()
    meta = SimpleNamespace(
        causal=True,
        ddtree_parent_ids=None,
        num_actual_tokens=total,
        max_query_len=max(q_lens),
        max_model_len=MAX_MODEL_LEN,
        query_start_loc=qsl_cpu.to(dev),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq_cpu.to(dev),
        seq_lens_cpu=seq_cpu,
        block_table=table,
    )
    return cache, query, meta, qsl_cpu


def _forward(impl, query, cache, meta):
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    out = torch.full_like(query, -99.0)
    kv = torch.zeros(query.shape[0], 1, HEAD_DIM, dtype=torch.float16, device="cuda")
    impl.forward(layer, query, kv, kv, cache, meta, out)
    torch.cuda.synchronize()
    return out


def test_nvfp4_mixed_decode_rows_match_bridge(monkeypatch):
    _require_gpu_nvfp4()
    impl = _make_impl(monkeypatch)
    cache, query, meta, qsl = _build_batch(seed=20261007)

    # Reserve the bridge workspace the way profile_run does (attn_metadata=None).
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    prof = torch.empty_like(query[: ROWS[-1][0]])
    impl.forward(layer, prof, prof[:, :1], prof[:, :1], cache, None, prof.clone())

    monkeypatch.setenv(KNOB, "0")
    off = _forward(impl, query, cache, meta)
    monkeypatch.setenv(KNOB, "1")
    on = _forward(impl, query, cache, meta)

    assert not (off == -99.0).any() and not (on == -99.0).any()
    split = int(qsl[-2])  # first token of the append chunk
    ref, got = off[:split].float(), on[:split].float()
    rel_l2 = ((got - ref).norm() / ref.norm()).item()
    max_diff = (got - ref).abs().max().item()
    print(f"decode/verify rows: max|d|={max_diff:.3e} rel-L2={rel_l2:.3e}")
    torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)
    assert rel_l2 <= 2e-3, f"rel-L2 knob=1 vs knob=0: {rel_l2:.3e}"
    # The append chunk takes the bridge under both knobs.
    assert torch.equal(on[split:], off[split:])
