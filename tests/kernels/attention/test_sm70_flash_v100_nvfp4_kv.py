# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 (e2m1 + E4M3 block scale) KV read path of the Flash-V100 XQA decode.

GPU-only (SM70). The cache is built with the plan-071 reference store, so the
kernel is checked against the format's executable specification.
"""

from __future__ import annotations

import math

import pytest
import torch

HEAD_DIM = 256
Q_HEADS = 6
NUM_KV_HEADS = 1
PAGE_SIZE = 2048


def _require_nvfp4_xqa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Flash-V100 NVFP4 XQA is SM70-only")
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    if not hasattr(flash_attn_v100, "flash_attn_nvfp4_kv_available"):
        pytest.skip("Flash-V100 Python package lacks the NVFP4 probe")
    if not flash_attn_v100.flash_attn_nvfp4_kv_available():
        pytest.skip("Flash-V100 extension lacks the NVFP4 KV read path")
    return flash_attn_v100


def test_nvfp4_kv_probe_reports_a_version():
    flash_attn_v100 = _require_nvfp4_xqa()
    assert flash_attn_v100.flash_attn_nvfp4_kv_available(min_version=1)
    assert not flash_attn_v100.flash_attn_nvfp4_kv_available(min_version=10**6)


def _build_case(ctx_len: int, rows: int, seed: int):
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv

    torch.manual_seed(seed)
    device = "cuda"
    pages = math.ceil(ctx_len / PAGE_SIZE)
    blocks = pages + 3
    keys = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM, device=device).mul_(0.5)
    values = torch.randn(ctx_len, NUM_KV_HEADS, HEAD_DIM, device=device).mul_(0.5)
    keys, values = keys.half(), values.half()

    physical = torch.randperm(blocks, dtype=torch.int32, device=device)[:pages]
    token = torch.arange(ctx_len, device=device)
    slots = physical.long()[token // PAGE_SIZE] * PAGE_SIZE + token % PAGE_SIZE

    cache = torch.zeros(
        blocks,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        nvfp4_kv.nvfp4_kv_row_bytes(HEAD_DIM),
        dtype=torch.uint8,
        device=device,
    )
    nvfp4_kv.reshape_and_cache_nvfp4_reference(
        keys, values, cache, slots.to(torch.int32)
    )

    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv.nvfp4_kv_split_views(cache)

    def dequant_tokens(data, scales):
        full = nvfp4_kv.dequantize_kv_nvfp4(data, scales, out_dtype=torch.float16)
        return full[physical.long()].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:ctx_len]

    k_deq = dequant_tokens(k_data, k_scales)
    v_deq = dequant_tokens(v_data, v_scales)

    # Rows of one decode / small-query step: row i sees ctx_len - (rows-1-i).
    seq_lens = torch.tensor(
        [ctx_len - (rows - 1 - i) for i in range(rows)],
        dtype=torch.int32,
        device=device,
    )
    block_table = physical.unsqueeze(0).repeat(rows, 1).contiguous()
    q = torch.randn(rows, Q_HEADS * NUM_KV_HEADS, HEAD_DIM, device=device).half()

    # FP16 cache holding exactly the dequantized values (same page order).
    fp16_cache_k = torch.zeros(
        blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    fp16_cache_v = torch.zeros_like(fp16_cache_k)
    fp16_cache_k.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = k_deq
    fp16_cache_v.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = v_deq
    return cache, fp16_cache_k, fp16_cache_v, block_table, seq_lens, q, k_deq, v_deq


def _torch_reference(q, k_deq, v_deq, seq_lens, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for row in range(q.shape[0]):
        n = int(seq_lens[row])
        k = k_deq[:n, 0].float()
        v = v_deq[:n, 0].float()
        scores = (q[row].float() @ k.T) * scale
        out[row] = torch.softmax(scores, dim=-1) @ v
    return out


@pytest.mark.parametrize("ctx_len", [4096, 70000])
@pytest.mark.parametrize("rows", [1, 5, 8])
def test_nvfp4_xqa_matches_fp16_dequant_and_float32_reference(ctx_len, rows):
    flash_attn_v100 = _require_nvfp4_xqa()
    (
        cache,
        fp16_k,
        fp16_v,
        block_table,
        seq_lens,
        q,
        k_deq,
        v_deq,
    ) = _build_case(ctx_len, rows, seed=1234 + rows)
    scale = HEAD_DIM**-0.5

    out_nvfp4 = flash_attn_v100.flash_attn_decode_paged_xqa(
        q,
        cache[:, 0],
        cache[:, 1],
        block_table,
        seq_lens,
        softmax_scale=scale,
        kv_cache_dtype="nvfp4",
        k_scale=1.0,
        v_scale=1.0,
        max_seq_len_hint=ctx_len,
    )
    out_fp16 = flash_attn_v100.flash_attn_decode_paged_xqa(
        q,
        fp16_k,
        fp16_v,
        block_table,
        seq_lens,
        softmax_scale=scale,
        kv_cache_dtype="auto",
        max_seq_len_hint=ctx_len,
    )
    reference = _torch_reference(q, k_deq, v_deq, seq_lens, scale)

    max_diff = (out_nvfp4.float() - out_fp16.float()).abs().max().item()
    print(f"ctx={ctx_len} rows={rows} max|nvfp4 - fp16(dequant)| = {max_diff:.3e}")
    torch.testing.assert_close(
        out_nvfp4.float(), out_fp16.float(), atol=2e-3, rtol=2e-3
    )
    rel_l2 = (out_nvfp4.float() - reference).norm() / reference.norm()
    assert rel_l2.item() <= 1e-3, f"rel-L2 vs float32 reference: {rel_l2.item():.3e}"


def _decode_nvfp4(flash_attn_v100, cache, block_table, seq_lens, q, ctx_len):
    return flash_attn_v100.flash_attn_decode_paged_xqa(
        q,
        cache[:, 0],
        cache[:, 1],
        block_table,
        seq_lens,
        softmax_scale=HEAD_DIM**-0.5,
        kv_cache_dtype="nvfp4",
        k_scale=1.0,
        v_scale=1.0,
        max_seq_len_hint=ctx_len,
    )


@pytest.mark.parametrize("ctx_len", [4096, 70000])
@pytest.mark.parametrize("rows", [5, 8])
def test_nvfp4_smallq_192_variant_and_half_p_knobs(ctx_len, rows, monkeypatch):
    """Q1.23 P1.2: the 192-thread 2-blocks/SM smallq variant (default ON) and
    the half-P contrast build (default OFF), both read via getenv at dispatch.

    192 ON vs OFF runs the same math in a different block shape, so only the
    cross-partition merge order may differ (expected: bit-identical or within
    fp16 output rounding). HALF_P rounds P to half before PV (E5M2 contract),
    so it is allowed a looser bound against the fp32-P result.
    """
    flash_attn_v100 = _require_nvfp4_xqa()
    cache, _, _, block_table, seq_lens, q, k_deq, v_deq = _build_case(
        ctx_len, rows, seed=4321 + rows
    )
    reference = _torch_reference(q, k_deq, v_deq, seq_lens, HEAD_DIM**-0.5)

    def run(smallq_192: str, half_p: str):
        monkeypatch.setenv("VLLM_FLASH_V100_NVFP4_SMALLQ_192", smallq_192)
        monkeypatch.setenv("VLLM_FLASH_V100_NVFP4_HALF_P", half_p)
        return _decode_nvfp4(
            flash_attn_v100, cache, block_table, seq_lens, q, ctx_len
        ).float()

    on = run("1", "0")
    off = run("0", "0")
    half_p_on = run("1", "1")
    half_p_off_route0 = run("0", "1")

    torch.testing.assert_close(on, off, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(half_p_on, on, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(half_p_off_route0, off, atol=5e-3, rtol=5e-3)
    for name, out in (("192-on", on), ("192-off", off), ("half-p", half_p_on)):
        rel_l2 = (out - reference).norm() / reference.norm()
        assert rel_l2.item() <= 1e-3, f"{name} rel-L2 vs fp32 ref {rel_l2.item():.3e}"
