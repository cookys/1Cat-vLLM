# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA sparse attention over an NVFP4 cache: gather, decode, FP16 kernel.

``kv_cache_dtype="nvfp4"`` gathers each row's selected tokens to FP16 and runs the
FP16 split-K kernel on them. The tests compare it with the same kernel run on a
plain FP16 paged cache that holds the dequantized values: bit for bit wherever the
two paths run identical arithmetic, and with a stated tolerance where they do not.

Without a GPU, run with ``TRITON_INTERPRET=1`` (the kernels then execute on the CPU
interpreter)::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \\
        python -m pytest --noconftest tests/models/qwen4_exp/test_nvfp4_kv_decode.py

The Triton interpreter cannot execute ``_qsa_merge_splitk_kernel`` (it applies ``&``
to a scalar and a block, which compiled code accepts), so on the CPU a torch
function with the same arithmetic stands in for it. Every comparison below uses the
same merge on both sides. The real merge kernel with ``FOLD_SCALES=True`` is covered
by the SM70 compile gate and by ``benchmarks/sm70_nvfp4_kv_kernel_check.py`` on a GPU.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs TRITON_INTERPRET=1 (CPU interpreter) or a CUDA GPU",
)

if not torch.cuda.is_available():
    # Importing the QSA backend asks the GPU for a FlashAttention version (see
    # test_nvfp4_kv_admission.py); Volta runs FA2.
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from vllm.models.qwen4_exp.nvidia import qsa as owner_module  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa_nvfp4  # noqa: E402

BLOCK = 32
HEAD_DIM = 256
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
ON_INTERPRETER = DEVICE == "cpu"
NVFP4 = nv.NVFP4_KV_CACHE_DTYPE


class _TorchMerge:
    """``_qsa_merge_splitk_kernel`` in torch, for the CPU interpreter."""

    def __getitem__(self, grid):
        return self._run

    @staticmethod
    def _run(
        partial_output,
        partial_lse,
        output,
        final_lse,
        gate,
        stride_output_row,
        stride_output_head,
        stride_gate_row,
        stride_gate_head,
        num_rows,
        v_scale,
        *,
        HEAD_DIM,
        NUM_QUERY_HEADS,
        NUM_SPLITS,
        BLOCK_SPLITS,
        FOLD_SCALES,
        num_warps=None,
        num_stages=None,
    ):
        inf = float("inf")
        lse_max = partial_lse.max(0).values
        has_values = lse_max > -inf
        shifted = torch.where(
            has_values[None], partial_lse - lse_max, torch.full_like(partial_lse, -inf)
        )
        weights = torch.exp2(shifted)
        denominator = weights.sum(0)
        if final_lse is not None:
            merged_lse = lse_max + torch.log2(denominator.clamp_min(1.0e-20))
            final_lse.copy_(
                torch.where(has_values, merged_lse, torch.full_like(lse_max, -inf))
            )
        merged = (partial_output * weights[..., None]).sum(0)
        merged = torch.where(
            denominator[..., None] > 0,
            merged / denominator[..., None],
            torch.zeros_like(merged),
        )
        if FOLD_SCALES:
            merged = merged * v_scale
        if gate is not None:
            merged = merged.to(output.dtype).float() * torch.sigmoid(gate.float())
        output.copy_(merged.to(output.dtype))


@pytest.fixture(autouse=True)
def _interpreter_merge(monkeypatch):
    if ON_INTERPRETER:
        monkeypatch.setattr(qsa_ops, "_qsa_merge_splitk_kernel", _TorchMerge())


def _scalar(value: float) -> float:
    return float(value)


class Env:
    """An NVFP4 cache, the FP16 cache of its dequantized values, and a batch."""

    def __init__(
        self,
        *,
        seed: int = 0,
        blocks: int = 8,
        heads: int = 1,
        rows: int = 3,
        topk: int = 48,
        requests: int = 2,
        width: int = 4,
        table: torch.Tensor | None = None,
        token_to_req: torch.Tensor | None = None,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        tokens = blocks * BLOCK
        key = (torch.randn(tokens, heads, HEAD_DIM, generator=generator) * 2).half()
        value = (torch.randn(tokens, heads, HEAD_DIM, generator=generator) * 2).half()
        cache = torch.zeros(
            (blocks, 2, BLOCK, heads, nv.nvfp4_kv_row_bytes(HEAD_DIM)),
            dtype=torch.uint8,
        )
        slots = torch.randperm(tokens, generator=generator)
        nv.reshape_and_cache_nvfp4_reference(key, value, cache, slots)
        (k_data, v_data), (k_scales, v_scales) = nv.nvfp4_kv_split_views(cache)
        self.k16 = nv.dequantize_kv_nvfp4(k_data, k_scales).to(DEVICE)
        self.v16 = nv.dequantize_kv_nvfp4(v_data, v_scales).to(DEVICE)
        self.cache = cache.to(DEVICE)
        self.blocks, self.heads, self.rows, self.topk = blocks, heads, rows, topk
        if table is None:
            order = torch.randperm(blocks, generator=generator)[: requests * width]
            table = order.reshape(requests, width).int()
        self.table = table.to(DEVICE)
        self.width = self.table.shape[1]
        if token_to_req is None:
            token_to_req = (torch.arange(rows) % self.table.shape[0]).int()
        self.token_to_req = token_to_req.to(DEVICE)
        self.q = torch.randn(rows, 6 * heads, HEAD_DIM, generator=generator).half()
        self.q = self.q.to(DEVICE)
        self.indices = torch.randint(
            0, self.width * BLOCK, (rows, topk), generator=generator, dtype=torch.int32
        ).to(DEVICE)
        self.generator = generator

    def nvfp4(self, **kwargs):
        return qsa_ops.qsa_sparse_paged_attention(
            self.q,
            self.cache[:, 0],
            self.cache[:, 1],
            self.indices,
            self.table,
            self.token_to_req,
            kv_cache_dtype=NVFP4,
            **kwargs,
        )

    def fp16(self, k_cache=None, v_cache=None, **kwargs):
        return qsa_ops.qsa_sparse_paged_attention(
            self.q,
            self.k16 if k_cache is None else k_cache,
            self.v16 if v_cache is None else v_cache,
            self.indices,
            self.table,
            self.token_to_req,
            **kwargs,
        )

    def float32_reference(self, k_scale: float, v_scale: float) -> torch.Tensor:
        """Attention in FP32 on K and V dequantized with their layer scales."""
        keys, values = nv.gather_dequant_nvfp4_kv(
            self.cache.cpu(),
            self.table.cpu(),
            self.token_to_req.cpu(),
            self.indices.cpu(),
            k_scale=k_scale,
            v_scale=v_scale,
            out_dtype=torch.float32,
        )
        valid = nv.nvfp4_entry_validity(
            self.indices.cpu(),
            self.token_to_req.cpu(),
            self.table.cpu(),
            self.blocks,
            BLOCK,
        )
        output = torch.zeros(self.q.shape, dtype=torch.float32)
        repeats = self.q.shape[1] // self.heads
        for row in range(self.rows):
            keep = valid[row]
            if not keep.any():
                continue
            k = keys[row][keep].repeat_interleave(repeats, dim=1)
            v = values[row][keep].repeat_interleave(repeats, dim=1)
            scores = torch.einsum("hd,khd->hk", self.q[row].cpu().float(), k)
            probabilities = torch.softmax(scores * HEAD_DIM**-0.5, dim=-1)
            output[row] = torch.einsum("hk,khd->hd", probabilities, v)
        return output


def _equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.equal(a.cpu(), b.cpu()), float((a.float() - b.float()).abs().max())


# --------------------------------------------------------------------------- #
# Equal to the FP16 kernel on the dequantized cache
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("heads", "rows", "topk"),
    [(1, 3, 48), (1, 2, 16), (2, 2, 40)],
    ids=["tp4_three_tiles", "one_tile_no_merge", "two_kv_heads"],
)
def test_decode_equals_the_fp16_kernel_on_the_dequantized_cache(heads, rows, topk):
    env = Env(heads=heads, rows=rows, topk=topk, seed=heads + rows)
    _equal(env.nvfp4(), env.fp16())
    assert env.nvfp4().abs().sum() > 0


def test_illegal_top_k_entries_are_masked_like_the_fp16_kernel():
    env = Env(rows=5, topk=48, requests=2, width=4, blocks=8, seed=11)
    unmapped = env.table.clone()
    unmapped[0, 1] = -1  # a page the request never got
    unmapped[1, 2] = 8  # a physical block past the cache
    unmapped[1, 3] = 10**6
    env.table = unmapped
    env.token_to_req = torch.tensor([0, 1, -1, 2, 1], dtype=torch.int32, device=DEVICE)
    indices = env.indices.clone()
    indices[0, :4] = torch.tensor(
        [-1, -7, 4 * BLOCK, 10**6]
    )  # negative, past the table
    indices[0, 4:8] = torch.tensor(
        [BLOCK, BLOCK + 3, 2 * BLOCK, 0]
    )  # page 1 is unmapped
    indices[1, :6] = torch.tensor([2 * BLOCK, 3 * BLOCK + 1, 3 * BLOCK, 0, 5, 9])
    indices[3, :] = 7  # request 2 does not exist
    indices[4, :] = -1  # a row with nothing selected
    env.indices = indices
    expected = env.fp16()
    actual = env.nvfp4()
    _equal(actual, expected)
    assert not actual[2].any() and not actual[3].any() and not actual[4].any()
    assert actual[0].any() and actual[1].any()


def test_poisoned_bytes_behind_masked_entries_never_reach_the_output():
    env = Env(rows=2, topk=24, requests=1, width=4, blocks=8, seed=12)
    env.table = torch.tensor([[0, -1, 1, 2]], dtype=torch.int32, device=DEVICE)
    env.token_to_req = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    unused = torch.tensor([3, 4, 5, 6, 7])  # physical blocks no table entry maps
    env.cache[unused] = 0xFF  # NaN scale bytes and saturated nibbles everywhere
    indices = env.indices.clone()
    indices[:, ::3] = BLOCK + 2  # page 1 is unmapped: masked, never read
    indices[1, 1::5] = -3
    env.indices = indices
    actual = env.nvfp4()
    assert torch.isfinite(actual.float()).all()
    _equal(actual, env.fp16())


def test_the_first_and_last_token_of_every_page_and_the_last_table_page():
    env = Env(rows=2, topk=16, requests=1, width=4, blocks=8, seed=13)
    edge = [0, BLOCK - 1, BLOCK, 2 * BLOCK - 1, 2 * BLOCK, 3 * BLOCK - 1]
    edge += [3 * BLOCK, 4 * BLOCK - 1]
    env.indices = torch.tensor(
        [edge + edge[::-1], edge[::-1] + edge], dtype=torch.int32, device=DEVICE
    )
    env.token_to_req = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    _equal(env.nvfp4(), env.fp16())


def test_mixed_batch_rows_with_shared_pages_and_repeated_tokens():
    # Three requests, two of them sharing physical pages, rows of different
    # requests interleaved, a repeated token inside one selection.
    table = torch.tensor([[0, 1, 2], [3, 1, 0], [4, 5, 6]], dtype=torch.int32)
    env = Env(
        rows=6,
        topk=36,
        blocks=8,
        table=table,
        token_to_req=torch.tensor([0, 2, 1, 0, 1, 2], dtype=torch.int32),
        seed=14,
    )
    indices = env.indices.clone()
    indices[1, 10:14] = indices[1, 3]  # the same token four times
    indices[4, 20:] = -1  # a short selection
    env.indices = indices
    _equal(env.nvfp4(), env.fp16())


# --------------------------------------------------------------------------- #
# The layer scales fold like E4M3's
# --------------------------------------------------------------------------- #
def test_power_of_two_layer_scales_equal_pre_scaled_fp16_kv_exactly():
    # K * 2**-3 and V * 2**-2 are exact in FP16, and scaling by a power of two
    # commutes with every rounding in the kernel, so folding k_scale into the QK
    # scores and v_scale into the output must equal running the plain FP16 kernel on
    # pre-scaled K and V. This pins which side each scale multiplies.
    env = Env(rows=3, topk=48, seed=21)
    folded = env.nvfp4(k_scale=0.125, v_scale=0.25)
    reference = env.fp16(env.k16 * 0.125, env.v16 * 0.25)
    _equal(folded, reference)
    swapped = env.nvfp4(k_scale=0.25, v_scale=0.125)
    assert not torch.equal(swapped.cpu(), folded.cpu())


@pytest.mark.parametrize("rows", [1, 3])
def test_the_single_split_kernel_folds_both_scales_itself(rows):
    # One tile means one split: the kernel writes the output directly and applies
    # v_scale itself (the merge kernel, with its stand-in on the CPU, is not
    # launched). The same exactness argument as above holds.
    env = Env(rows=rows, topk=16, seed=25)
    folded = env.nvfp4(k_scale=0.125, v_scale=0.25)
    _equal(folded, env.fp16(env.k16 * 0.125, env.v16 * 0.25))
    assert not torch.equal(folded.cpu(), env.nvfp4().cpu())
    gated = torch.randn(env.q.shape, dtype=torch.float16, device=DEVICE)
    _equal(
        env.nvfp4(k_scale=0.125, v_scale=0.25, output_gate=gated),
        env.fp16(env.k16 * 0.125, env.v16 * 0.25, output_gate=gated),
    )


def test_k_scale_changes_the_attention_pattern_and_v_scale_only_the_magnitude():
    env = Env(rows=2, topk=32, seed=22)
    base = env.nvfp4()
    v_only = env.nvfp4(v_scale=0.5)
    _equal(v_only, (base.float() * 0.5).half())
    k_scaled = env.nvfp4(k_scale=0.5)
    assert not torch.equal(k_scaled.cpu(), base.cpu())


@pytest.mark.parametrize("scales", [(0.0213623046875, 0.0404924675822258), (3.7, 0.9)])
def test_general_layer_scales_match_float32_attention_on_the_dequantized_kv(scales):
    # Not bit-exact by construction: the reference dequantizes with the scale in
    # FP32 and runs softmax in FP32, while the kernel multiplies the FP16 decode by
    # the scale after the dot, casts the probabilities to FP16 and rounds the output
    # to FP16. The error is the FP16 rounding of those steps.
    k_scale, v_scale = scales
    env = Env(rows=4, topk=48, seed=23)
    actual = env.nvfp4(k_scale=k_scale, v_scale=v_scale).float().cpu()
    expected = env.float32_reference(k_scale, v_scale)
    scale = float(expected.abs().max())
    assert torch.allclose(actual, expected, rtol=3e-2, atol=3e-3 * max(scale, 1e-3))


def test_scales_that_are_not_finite_and_positive_are_refused():
    env = Env(rows=1, topk=16, seed=24)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scales must be finite and positive"):
            env.nvfp4(k_scale=bad)
        with pytest.raises(ValueError, match="scales must be finite and positive"):
            env.nvfp4(v_scale=bad)


# --------------------------------------------------------------------------- #
# Other outputs, row groups, routing
# --------------------------------------------------------------------------- #
def test_the_output_gate_is_applied_like_the_fp16_kernel_applies_it():
    env = Env(rows=3, topk=48, seed=31)
    gate = torch.randn(env.q.shape, dtype=torch.float16, device=DEVICE)
    _equal(env.nvfp4(output_gate=gate), env.fp16(output_gate=gate))


def test_the_lse_output_of_the_dcp_merge_equals_the_fp16_kernels():
    env = Env(rows=3, topk=48, seed=32)
    outputs = []
    for run in (env.nvfp4, env.fp16):
        out = torch.empty(env.q.shape, dtype=torch.float32, device=DEVICE)
        lse = torch.empty(env.q.shape[:2], dtype=torch.float32, device=DEVICE)
        run(out=out, lse=lse)
        outputs.append((out, lse))
    _equal(outputs[0][0], outputs[1][0])
    _equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("rows_per_group", [1, 2, 3])
def test_row_groups_do_not_change_the_result(monkeypatch, rows_per_group):
    # Up to 8 programs the launch profile (tile, splits, warps) depends only on the
    # selection width, so any grouping of rows is bit-identical to one group.
    env = Env(rows=5, topk=40, seed=33)
    expected = env.nvfp4()
    per_row = 2 * env.topk * env.heads * HEAD_DIM * 2
    monkeypatch.setattr(
        qsa_nvfp4, "NVFP4_GATHER_SCRATCH_BYTES", per_row * rows_per_group
    )
    seen = []
    original = qsa_nvfp4.gather_dequant_nvfp4_sides_triton

    def spy(*args, **kwargs):
        seen.append(args[4].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(qsa_nvfp4, "gather_dequant_nvfp4_sides_triton", spy)
    grouped = qsa_ops.qsa_sparse_paged_attention(
        env.q,
        env.cache[:, 0],
        env.cache[:, 1],
        env.indices,
        env.table,
        env.token_to_req,
        kv_cache_dtype=NVFP4,
    )
    assert max(seen) == rows_per_group and sum(seen) == env.rows
    _equal(grouped, expected)


def test_the_gather_scratch_budget_sets_the_rows_per_group():
    per_row = 2 * 2051 * 1 * 256 * 2
    assert qsa_nvfp4.nvfp4_gather_rows_per_group(2051, 1, 256) == (64 << 20) // per_row
    assert qsa_nvfp4.nvfp4_gather_rows_per_group(2051, 1, 256, 1) == 1
    assert qsa_nvfp4.nvfp4_gather_rows_per_group(2051, 1, 256, 3 * per_row) == 3
    assert qsa_nvfp4.nvfp4_gather_rows_per_group(2051, 2, 256, 3 * per_row) == 1


def test_the_page4_cuda_routes_never_see_an_nvfp4_cache(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an NVFP4 cache reached a page4 CUDA route")

    for name in (
        "_use_sm70_qsa_xqa_page4",
        "_qsa_sparse_paged_attention_sm70_xqa_page4",
        "_qsa_sparse_paged_attention_sm70_grouped_page4",
    ):
        monkeypatch.setattr(qsa_ops, name, refuse)
    env = Env(rows=20, topk=16, seed=34)
    positions = torch.zeros(20, dtype=torch.int64, device=DEVICE)
    lengths = torch.full((2,), 100, dtype=torch.int32, device=DEVICE)
    out = env.nvfp4(query_positions=positions, sequence_lengths=lengths)
    assert out.shape == env.q.shape


# --------------------------------------------------------------------------- #
# Validation of the entry point
# --------------------------------------------------------------------------- #
def test_the_entry_point_validates_an_nvfp4_cache():
    env = Env(rows=1, topk=16, seed=41)
    with pytest.raises(ValueError, match="require FP16 queries"):
        qsa_ops.qsa_sparse_paged_attention(
            env.q.bfloat16(),
            env.cache[:, 0],
            env.cache[:, 1],
            env.indices,
            env.table,
            env.token_to_req,
            kv_cache_dtype=NVFP4,
        )
    signed = env.cache.view(torch.int8)
    with pytest.raises(ValueError, match="uint8 storage"):
        qsa_ops.qsa_sparse_paged_attention(
            env.q,
            signed[:, 0],
            signed[:, 1],
            env.indices,
            env.table,
            env.token_to_req,
            kv_cache_dtype=NVFP4,
        )
    # A row of the wrong width does not describe a 256-wide head.
    narrow = env.cache[..., :-16]
    with pytest.raises(ValueError, match="grouped-query heads"):
        qsa_ops.qsa_sparse_paged_attention(
            env.q,
            narrow[:, 0],
            narrow[:, 1],
            env.indices,
            env.table,
            env.token_to_req,
            kv_cache_dtype=NVFP4,
        )


def test_the_other_cache_dtypes_keep_their_checks():
    env = Env(rows=1, topk=16, seed=42)
    raw = torch.zeros((8, BLOCK, 1, HEAD_DIM), dtype=torch.uint8, device=DEVICE)
    with pytest.raises(ValueError, match="must match query dtype"):
        qsa_ops.qsa_sparse_paged_attention(
            env.q, raw, raw, env.indices, env.table, env.token_to_req
        )
    with pytest.raises(ValueError, match="Unsupported QSA K/V cache dtype"):
        env.fp16(kv_cache_dtype="nvfp8")


# --------------------------------------------------------------------------- #
# The owner: forward_qsa and the gates that stay
# --------------------------------------------------------------------------- #
def _impl():
    impl = object.__new__(owner_module.Qwen4ExpQSAFlashAttentionImpl)
    impl.kv_cache_dtype = NVFP4
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    return impl


def _layer(env, *, sharded=False, k_scale=0.0213623046875, v_scale=0.0404924675822258):
    return SimpleNamespace(
        topk_indices_buffer=env.indices,
        qsa_dcp_sharded=sharded,
        _k_scale_float=k_scale,
        _v_scale_float=v_scale,
    )


def test_forward_qsa_runs_an_nvfp4_cache_through_the_gathered_route():
    env = Env(rows=3, topk=48, seed=51)
    metadata = SimpleNamespace(num_actual_tokens=3, block_table=env.table)
    output = torch.empty_like(env.q)
    returned = _impl().forward_qsa(
        _layer(env),
        env.q,
        None,
        None,
        env.cache,
        metadata,
        output,
        token_to_req=env.token_to_req,
    )
    assert returned is output
    expected = env.nvfp4(k_scale=0.0213623046875, v_scale=0.0404924675822258)
    _equal(output, expected)
    assert output.abs().sum() > 0


def test_forward_qsa_refuses_what_nvfp4_does_not_support_yet():
    env = Env(rows=1, topk=16, seed=52)
    metadata = SimpleNamespace(num_actual_tokens=1, block_table=env.table)
    output = torch.empty_like(env.q)
    args = (env.cache, metadata, output)
    with pytest.raises(NotImplementedError, match="DCP"):
        _impl().forward_qsa(
            _layer(env, sharded=True),
            env.q,
            None,
            None,
            *args,
            token_to_req=env.token_to_req,
        )
    with pytest.raises(NotImplementedError, match="FP16 queries"):
        _impl().forward_qsa(
            _layer(env),
            env.q.bfloat16(),
            None,
            None,
            *args,
            token_to_req=env.token_to_req,
        )
    with pytest.raises(RuntimeError, match="uint8 storage"):
        _impl().forward_qsa(
            _layer(env),
            env.q,
            None,
            None,
            env.cache.view(torch.int8),
            metadata,
            output,
            token_to_req=env.token_to_req,
        )


def test_run_qsa_still_requires_finalized_scales_for_nvfp4():
    owner = object.__new__(owner_module.Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.kv_cache_dtype = NVFP4
    owner._qsa_kv_scales_finalized = False
    owner.layer_name = "model.layers.3.self_attn.attn"
    with pytest.raises(RuntimeError, match="were not finalized"):
        owner._run_qsa(None, None, None, None, None, None)


@pytest.mark.parametrize(
    ("cache_dtype", "speculative", "refused"),
    [
        (NVFP4, SimpleNamespace(num_speculative_tokens=4), True),
        (NVFP4, SimpleNamespace(num_speculative_tokens=0), False),
        (NVFP4, None, False),
        ("fp8_e4m3", SimpleNamespace(num_speculative_tokens=4), False),
        ("auto", SimpleNamespace(num_speculative_tokens=4), False),
    ],
)
def test_the_mtp_draft_is_refused_for_nvfp4_only(cache_dtype, speculative, refused):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        speculative_config=speculative,
    )
    if refused:
        with pytest.raises(NotImplementedError, match="MTP draft"):
            owner_module._verify_nvfp4_kv_speculation(config)
    else:
        owner_module._verify_nvfp4_kv_speculation(config)


# --------------------------------------------------------------------------- #
# The halves gather equals the 5-D gather
# --------------------------------------------------------------------------- #
def test_the_halves_gather_equals_the_five_dimensional_gather():
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv_triton as nvt

    env = Env(rows=2, topk=24, seed=61)
    five = nvt.gather_dequant_nvfp4_kv_triton(
        env.cache, env.table, env.token_to_req, env.indices, k_scale=0.5, v_scale=2.0
    )
    halves = nvt.gather_dequant_nvfp4_sides_triton(
        env.cache[:, 0],
        env.cache[:, 1],
        env.table,
        env.token_to_req,
        env.indices,
        k_scale=0.5,
        v_scale=2.0,
    )
    _equal(five[0], halves[0])
    _equal(five[1], halves[1])
