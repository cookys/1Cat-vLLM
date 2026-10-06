# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact grouped DFlash2/MTP4 verifier reading an NVFP4 KV cache (Q1.23 P4).

GPU-only (SM70); every test skips on CPU or on an extension older than probe
version 3. The cache is built with the plan-071 reference store, so the kernel
is checked against the format's executable specification (a float32 causal
attention over the dequantized values) and against the per-row XQA NVFP4
decode (route independence).
"""

from __future__ import annotations

import math

import pytest
import torch

HEAD_DIM = 256
Q_HEADS = 6
PAGE_CASES = [2048, 4096]


def _require_grouped_nvfp4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Flash-V100 grouped verify is SM70-only")
    fa = pytest.importorskip("flash_attn_v100")
    if not hasattr(fa, "flash_attn_nvfp4_kv_available"):
        pytest.skip("Flash-V100 Python package lacks the NVFP4 probe")
    if not fa.flash_attn_nvfp4_kv_available(min_version=3):
        pytest.skip("Flash-V100 extension lacks the NVFP4 grouped verifier (v3)")
    return fa


def _build(batch: int, ctx_len: int, page: int, seed: int):
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv

    torch.manual_seed(seed)
    dev = "cuda"
    pages = math.ceil(ctx_len / page)
    blocks = batch * pages + 3
    cache = torch.zeros(
        blocks, 2, page, 1, nvfp4_kv.nvfp4_kv_row_bytes(HEAD_DIM),
        dtype=torch.uint8, device=dev,
    )
    physical = torch.randperm(blocks, dtype=torch.int32, device=dev)[: batch * pages]
    block_table = physical.view(batch, pages).contiguous()
    token = torch.arange(ctx_len, device=dev)
    k_deq, v_deq = [], []
    for b in range(batch):
        keys = torch.randn(ctx_len, 1, HEAD_DIM, device=dev).mul_(0.5).half()
        values = torch.randn(ctx_len, 1, HEAD_DIM, device=dev).mul_(0.5).half()
        slots = block_table[b].long()[token // page] * page + token % page
        nvfp4_kv.reshape_and_cache_nvfp4_reference(
            keys, values, cache, slots.to(torch.int32)
        )
    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv.nvfp4_kv_split_views(cache)

    def deq(data, scales, b):
        full = nvfp4_kv.dequantize_kv_nvfp4(data, scales, out_dtype=torch.float16)
        return full[block_table[b].long()].reshape(-1, 1, HEAD_DIM)[:ctx_len]

    for b in range(batch):
        k_deq.append(deq(k_data, k_scales, b))
        v_deq.append(deq(v_data, v_scales, b))
    return cache, block_table, k_deq, v_deq


def _reference(q, k_deq, v_deq, ctx_len, query_len, scale):
    """q [B*Q, 6, 256] -> float32 causal attention on the dequantized values."""
    batch = len(k_deq)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(batch):
        k = k_deq[b][:, 0].float()
        v = v_deq[b][:, 0].float()
        for t in range(query_len):
            n = ctx_len - query_len + 1 + t
            row = b * query_len + t
            scores = (q[row].float() @ k[:n].T) * scale
            out[row] = torch.softmax(scores, dim=-1) @ v[:n]
    return out


CASES = [(1, 8), (1, 5), (1, 16), (4, 8)]


@pytest.mark.parametrize("page", PAGE_CASES)
@pytest.mark.parametrize("ctx_len", [4096, 70000])
@pytest.mark.parametrize("batch,query_len", CASES)
def test_nvfp4_grouped_verify_matches_float32_and_xqa(batch, query_len, ctx_len, page):
    fa = _require_grouped_nvfp4()
    cache, block_table, k_deq, v_deq = _build(
        batch, ctx_len, page, seed=77 + 13 * batch + query_len
    )
    scale = HEAD_DIM**-0.5
    q = torch.randn(batch * query_len, Q_HEADS, HEAD_DIM, device="cuda").half()
    seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")

    out = fa.flash_attn_grouped_verify_paged(
        q, cache[:, 0], cache[:, 1], block_table, seq_lens,
        softmax_scale=scale, kv_cache_dtype="nvfp4",
        k_scale=1.0, v_scale=1.0, one_pass=True,
    )
    reference = _reference(q, k_deq, v_deq, ctx_len, query_len, scale)
    max_diff = (out.float() - reference).abs().max().item()
    rel_l2 = ((out.float() - reference).norm() / reference.norm()).item()
    print(
        f"B{batch} q{query_len} ctx={ctx_len} page={page}: "
        f"max|d|={max_diff:.3e} rel-L2={rel_l2:.3e}"
    )
    assert rel_l2 <= 2e-3, f"rel-L2 vs float32 reference: {rel_l2:.3e}"

    # Route independence: per-row XQA NVFP4 decode over the same rows.
    rows = batch * query_len
    row_tables = block_table.repeat_interleave(query_len, dim=0).contiguous()
    row_lens = torch.tensor(
        [ctx_len - query_len + 1 + t for _ in range(batch) for t in range(query_len)],
        dtype=torch.int32, device="cuda",
    )
    xqa = fa.flash_attn_decode_paged_xqa(
        q, cache[:, 0], cache[:, 1], row_tables, row_lens,
        softmax_scale=scale, kv_cache_dtype="nvfp4",
        k_scale=1.0, v_scale=1.0, max_seq_len_hint=ctx_len,
    )
    assert xqa.shape[0] == rows
    torch.testing.assert_close(out.float(), xqa.float(), atol=2e-3, rtol=2e-3)


def test_nvfp4_grouped_verify_rejects_misshaped_cache():
    fa = _require_grouped_nvfp4()
    q = torch.zeros(8, Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    bad = torch.zeros(4, 2048, 1, 256, device="cuda", dtype=torch.uint8)
    table = torch.zeros(1, 2, device="cuda", dtype=torch.int32)
    lens = torch.full((1,), 2048, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        fa.flash_attn_grouped_verify_paged(
            q, bad, bad, table, lens, kv_cache_dtype="nvfp4", one_pass=True
        )
