# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused NVFP4 reader of the QSA split-K kernel (``KV_NVFP4``), plan 071.

``nvfp4_fused_reader=True`` reads the packed e2m1 bytes and E4M3 block scales of the
cache inside ``_qsa_sparse_paged_gqa_splitk_kernel`` instead of gathering the
selected rows to FP16 first. The two routes decode identically (exactly, in FP16),
but QK is two K = 128 dots instead of one K = 256 dot, so their sums differ in FP32
rounding order: the comparisons below use a tolerance of FP16 output rounding, not
bit equality. Run with ``TRITON_INTERPRET=1`` on a machine without a GPU::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \\
        python -m pytest --noconftest tests/models/qwen4_exp/test_nvfp4_kv_fused.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs TRITON_INTERPRET=1 (CPU interpreter) or a CUDA GPU",
)

from tests.models.qwen4_exp.test_nvfp4_kv_decode import (  # noqa: E402
    BLOCK,
    DEVICE,
    HEAD_DIM,
    NVFP4,
    ON_INTERPRETER,
    Env,
    _TorchMerge,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa_nvfp4  # noqa: E402


@pytest.fixture(autouse=True)
def _interpreter_merge(monkeypatch):
    if ON_INTERPRETER:
        monkeypatch.setattr(qsa_ops, "_qsa_merge_splitk_kernel", _TorchMerge())


def _fused(env, **kwargs):
    return env.nvfp4(nvfp4_fused_reader=True, **kwargs)


def _gathered(env, **kwargs):
    return env.nvfp4(nvfp4_fused_reader=False, **kwargs)


def _close(actual, expected, rtol=2e-3, atol=2e-3) -> float:
    """FP16 rounding level: the outputs are rounded to FP16 (2**-11 relative) and the
    two dots add the same products in a different FP32 order."""
    a, e = actual.float().cpu(), expected.float().cpu()
    scale = float(e.abs().max())
    error = float((a - e).abs().max())
    assert torch.allclose(a, e, rtol=rtol, atol=atol * max(scale, 1e-3)), error
    return error


@pytest.mark.parametrize(
    ("heads", "rows", "topk"),
    [(1, 3, 48), (1, 2, 16), (2, 2, 40), (1, 1, 33)],
    ids=["three_tiles", "one_tile_no_merge", "two_kv_heads", "ragged_last_tile"],
)
def test_fused_matches_the_gathered_route_and_the_fp16_kernel(heads, rows, topk):
    env = Env(heads=heads, rows=rows, topk=topk, seed=heads + rows + topk)
    fused = _fused(env)
    assert fused.abs().sum() > 0
    _close(fused, _gathered(env))
    _close(fused, env.fp16())


def test_fused_with_layer_scales_not_one():
    env = Env(rows=4, topk=48, seed=71)
    for k_scale, v_scale in ((0.0213623046875, 0.0404924675822258), (3.7, 0.9)):
        fused = _fused(env, k_scale=k_scale, v_scale=v_scale)
        _close(fused, _gathered(env, k_scale=k_scale, v_scale=v_scale))
        expected = env.float32_reference(k_scale, v_scale)
        _close(fused, expected, rtol=3e-2, atol=3e-3)
    # The scales act where the gathered route puts them: V only scales the output.
    base = _fused(env)
    _close(_fused(env, v_scale=0.5), (base.float() * 0.5).half(), atol=1e-3)
    assert not torch.equal(_fused(env, k_scale=0.5).cpu(), base.cpu())


def test_fused_masks_illegal_entries_and_ignores_poisoned_bytes():
    env = Env(rows=5, topk=48, requests=2, width=4, blocks=8, seed=72)
    table = env.table.clone()
    table[0, 1] = -1  # a page the request never got
    table[1, 2] = 8  # a physical block past the cache
    table[1, 3] = 10**6
    env.table = table
    env.token_to_req = torch.tensor([0, 1, -1, 2, 1], dtype=torch.int32, device=DEVICE)
    indices = env.indices.clone()
    indices[0, :4] = torch.tensor([-1, -7, 4 * BLOCK, 10**6])
    indices[0, 4:8] = torch.tensor([BLOCK, BLOCK + 3, 2 * BLOCK, 0])
    indices[1, :6] = torch.tensor([2 * BLOCK, 3 * BLOCK + 1, 3 * BLOCK, 0, 5, 9])
    indices[3, :] = 7  # request 2 does not exist
    indices[4, :] = -1  # a row with nothing selected
    env.indices = indices
    # Blocks that no entry maps carry NaN scale bytes and saturated nibbles.
    mapped = {int(b) for b in env.table.flatten().tolist() if 0 <= b < 8}
    for block in set(range(8)) - mapped:
        env.cache[block] = 0xFF
    fused = _fused(env)
    assert torch.isfinite(fused.float()).all()
    assert not fused[2].any() and not fused[3].any() and not fused[4].any()
    assert fused[0].any() and fused[1].any()
    _close(fused, _gathered(env))


def test_fused_edge_tokens_shared_pages_and_repeats():
    table = torch.tensor([[0, 1, 2], [3, 1, 0], [4, 5, 6]], dtype=torch.int32)
    env = Env(
        rows=6,
        topk=36,
        blocks=8,
        table=table,
        token_to_req=torch.tensor([0, 2, 1, 0, 1, 2], dtype=torch.int32),
        seed=73,
    )
    indices = env.indices.clone()
    edge = [0, BLOCK - 1, BLOCK, 2 * BLOCK - 1, 2 * BLOCK, 3 * BLOCK - 1]
    indices[0, : len(edge)] = torch.tensor(edge)
    indices[1, 10:14] = indices[1, 3]  # the same token four times
    indices[4, 20:] = -1  # a short selection
    env.indices = indices
    _close(_fused(env), _gathered(env))


def test_fused_output_gate_lse_and_the_single_split_path():
    env = Env(rows=3, topk=16, seed=74)  # one tile: one split, no merge kernel
    gate = torch.randn(env.q.shape, dtype=torch.float16, device=DEVICE)
    _close(_fused(env, output_gate=gate), _gathered(env, output_gate=gate))
    env = Env(rows=3, topk=48, seed=75)
    results = []
    for run in (_fused, _gathered):
        out = torch.empty(env.q.shape, dtype=torch.float32, device=DEVICE)
        lse = torch.empty(env.q.shape[:2], dtype=torch.float32, device=DEVICE)
        run(env, out=out, lse=lse)
        results.append((out, lse))
    _close(results[0][0], results[1][0])
    _close(results[0][1], results[1][1], rtol=1e-4, atol=1e-4)


def test_fused_never_calls_the_gather_or_the_page4_routes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the fused reader must not use this route")

    for name in (
        "_use_sm70_qsa_xqa_page4",
        "_qsa_sparse_paged_attention_sm70_xqa_page4",
        "_qsa_sparse_paged_attention_sm70_grouped_page4",
    ):
        monkeypatch.setattr(qsa_ops, name, refuse)
    monkeypatch.setattr(qsa_nvfp4, "gather_dequant_nvfp4_sides_triton", refuse)
    env = Env(rows=20, topk=16, seed=76)
    positions = torch.zeros(20, dtype=torch.int64, device=DEVICE)
    lengths = torch.full((2,), 100, dtype=torch.int32, device=DEVICE)
    out = _fused(env, query_positions=positions, sequence_lengths=lengths)
    assert out.shape == env.q.shape


def test_the_knob_defaults_to_the_gather_route(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_QSA_NVFP4_FUSED_READER", raising=False)
    assert qsa_nvfp4.nvfp4_fused_reader_enabled() is False
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_FUSED_READER", "1")
    assert qsa_nvfp4.nvfp4_fused_reader_enabled() is True
    assert qsa_nvfp4.nvfp4_fused_reader_enabled(False) is False
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_FUSED_READER", "0")
    assert qsa_nvfp4.nvfp4_fused_reader_enabled(True) is True
    # With no override the entry point follows the variable.
    env = Env(rows=2, topk=24, seed=77)
    called = []
    original = qsa_nvfp4.qsa_sparse_attention_nvfp4

    def spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(qsa_nvfp4, "qsa_sparse_attention_nvfp4", spy)
    env.nvfp4()
    assert called == [True]
    called.clear()
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_FUSED_READER", "1")
    env.nvfp4()
    assert called == []


def test_fused_keeps_the_nvfp4_admission_checks():
    env = Env(rows=1, topk=16, seed=78)
    with pytest.raises(ValueError, match="scales must be finite and positive"):
        _fused(env, k_scale=0.0)
    with pytest.raises(ValueError, match="require FP16 queries"):
        qsa_ops.qsa_sparse_paged_attention(
            env.q.bfloat16(),
            env.cache[:, 0],
            env.cache[:, 1],
            env.indices,
            env.table,
            env.token_to_req,
            kv_cache_dtype=NVFP4,
            nvfp4_fused_reader=True,
        )
