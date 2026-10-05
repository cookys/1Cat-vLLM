# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan 071 option B': routing of NVFP4 prefill chunks through the FP16 grouped route.

With ``VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH=1`` a prefill chunk (rows at or above
``VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS``) of an NVFP4 cache is decoded once into the
FP16 scratch and attended by ``grouped_sparse_page4_fwd`` with
``kv_cache_dtype="auto"``.
The CUDA kernels are replaced here by torch stand-ins that follow the same contract
(an FP16 cache ignores ``k_scale``/``v_scale``; the softmax scale is a plain float), so
these tests pin down the wiring and the scale arithmetic, not the CUDA kernel::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python -m pytest \\
        --noconftest tests/models/qwen4_exp/test_nvfp4_kv_prefill_route.py
"""

from __future__ import annotations

import math
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs TRITON_INTERPRET=1 (CPU interpreter) or a CUDA GPU",
)

if not torch.cuda.is_available():
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from vllm.models.qwen4_exp.nvidia import qsa as owner_module  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa_nvfp4  # noqa: E402

BLOCK = 36
HEAD = 256
TOPK = 2051
K_SCALE = 0.0213623046875
V_SCALE = 0.0404924675822258
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


# ------------------------------------------------------------------ the fake CUDA


class FakeFlashV100:
    """Torch stand-ins for the grouped page4 planner/forward and the XQA batch."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._plan = None

    def grouped_sparse_page4_abi_version(self) -> int:
        return 2

    def grouped_sparse_page4_plan_fwd(
        self,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        page_size,
        physical_page_stride,
        num_cache_blocks,
    ) -> None:
        self._plan = dict(
            indices=logical_indices.clone(),
            table=block_table.clone(),
            requests=token_to_req.clone(),
            positions=query_positions.clone(),
            seq_lens=sequence_lengths.clone(),
            page_size=page_size,
            stride=physical_page_stride,
            blocks=num_cache_blocks,
        )
        self.calls.append(("plan", dict(self._plan)))

    def _attend(self, q, k_phys, v_phys, selected, scale):
        out = torch.zeros_like(q)
        for row, tokens in enumerate(selected):
            if not tokens:
                continue
            keys = torch.stack([k_phys[mb, slot, 0] for mb, slot in tokens]).float()
            values = torch.stack([v_phys[mb, slot, 0] for mb, slot in tokens]).float()
            scores = torch.einsum("hd,td->ht", q[row].float(), keys) * scale
            weights = torch.softmax(scores, dim=-1)
            out[row] = (weights @ values).to(q.dtype)
        return out

    def grouped_sparse_page4_fwd(
        self,
        q,
        k_cache,
        v_cache,
        out,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        lse,
        softmax_scale,
        kv_cache_dtype,
        k_scale,
        v_scale,
    ) -> None:
        # An FP16 cache ignores k_scale/v_scale (fdp.cu: e4m3_kv ? k_scale : 1.0f).
        assert kv_cache_dtype in ("auto", "float16")
        assert k_cache.dtype == torch.float16 and k_cache.shape[1:] == (4, 1, HEAD)
        self.calls.append(
            (
                "grouped",
                dict(
                    rows=q.shape[0],
                    softmax_scale=softmax_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    kv_cache_dtype=kv_cache_dtype,
                    k_ptr=k_cache.data_ptr(),
                    v_ptr=v_cache.data_ptr(),
                ),
            )
        )
        plan = self._plan
        selected = []
        for row in range(q.shape[0]):
            request = int(plan["requests"][row])
            seq_len = int(plan["seq_lens"][request])
            visible = min(int(plan["positions"][row]) + 1, seq_len)
            tokens = []
            for token in plan["indices"][row].tolist():
                if token < 0 or token >= visible:
                    continue
                page, offset = divmod(token, plan["page_size"])
                physical = int(plan["table"][request, page])
                if not 0 <= physical < plan["blocks"]:
                    continue
                tokens.append((physical * plan["stride"] + offset // 4, offset % 4))
            selected.append(tokens)
        out.copy_(self._attend(q, k_cache, v_cache, selected, softmax_scale))

    def decode_paged_xqa_fwd(
        self,
        q,
        k_cache,
        v_cache,
        out,
        virtual_block_table,
        sequence_lengths,
        temporary_output,
        max_logits,
        exp_sums,
        active_num_partitions,
        softmax_scale,
        partition_size,
        num_partitions,
        kv_cache_dtype,
        k_scale,
        v_scale,
        window_left,
        window_right,
        anchored_window,
    ) -> None:
        assert kv_cache_dtype in ("auto", "float16")
        self.calls.append(
            (
                "xqa",
                dict(
                    rows=q.shape[0],
                    softmax_scale=softmax_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    kv_cache_dtype=kv_cache_dtype,
                ),
            )
        )
        selected = []
        for row in range(q.shape[0]):
            microblocks = virtual_block_table[row].tolist()
            tokens = [
                (microblocks[position // 4], position % 4)
                for position in range(int(sequence_lengths[row]))
            ]
            selected.append(tokens)
        out.copy_(self._attend(q, k_cache, v_cache, selected, softmax_scale))


@pytest.fixture
def fake_cuda(monkeypatch):
    fake = FakeFlashV100()
    package = types.ModuleType("flash_attn_v100")
    interface = types.ModuleType("flash_attn_v100.flash_attn_interface")
    interface.flash_attn_v100_cuda = fake
    package.flash_attn_interface = interface
    monkeypatch.setitem(sys.modules, "flash_attn_v100", package)
    monkeypatch.setitem(sys.modules, "flash_attn_v100.flash_attn_interface", interface)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=77)
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4_ABI_CACHE", None)
    monkeypatch.setattr(
        qsa_ops.current_platform, "is_device_capability", lambda capability: True
    )
    workspaces = (
        qsa_ops._SM70_QSA_GROUPED_PAGE4_WORKSPACES,
        qsa_ops._SM70_QSA_XQA_PAGE4_WORKSPACES,
    )
    saved = [dict(w) for w in workspaces]
    for w in workspaces:
        w.clear()
    yield fake
    for w, s in zip(workspaces, saved):
        w.clear()
        w.update(s)


@pytest.fixture(autouse=True)
def clean_scratch():
    qsa_nvfp4.reset_nvfp4_prefill_scratch()
    yield
    qsa_nvfp4.reset_nvfp4_prefill_scratch()


@pytest.fixture
def knob(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", "1")


# ---------------------------------------------------------------------- a batch


class Batch:
    """An NVFP4 cache, a causal prefill chunk and its top-k selection."""

    def __init__(self, *, rows=64, seq_len=600, seed=0, blocks=24, first_row=None):
        generator = torch.Generator().manual_seed(seed)
        tokens = blocks * BLOCK
        key = (torch.randn(tokens, 1, HEAD, generator=generator) * 2).half()
        value = (torch.randn(tokens, 1, HEAD, generator=generator) * 2).half()
        cache = torch.zeros(
            (blocks, 2, BLOCK, 1, nv.nvfp4_kv_row_bytes(HEAD)), dtype=torch.uint8
        )
        slots = torch.randperm(tokens, generator=generator)
        nv.reshape_and_cache_nvfp4_reference(
            key, value, cache, slots, k_scale=K_SCALE, v_scale=V_SCALE
        )
        pages = -(-seq_len // BLOCK)
        order = torch.randperm(blocks, generator=generator)[:pages].int()
        table = torch.full((1, pages + 2), -1, dtype=torch.int32)
        table[0, :pages] = order
        first_row = seq_len - rows if first_row is None else first_row
        positions = torch.arange(first_row, first_row + rows, dtype=torch.int64)
        indices = torch.full((rows, TOPK), -1, dtype=torch.int32)
        for row, position in enumerate(positions.tolist()):
            visible = position + 1
            full = visible // 4
            count = min(full, 512)
            groups = torch.randperm(full, generator=generator)[:count].sort().values
            span = (groups[:, None] * 4 + torch.arange(4)).reshape(-1).int()
            indices[row, : span.numel()] = span
            tail = visible - full * 4
            if tail:
                indices[row, count * 4 : count * 4 + tail] = (
                    full * 4 + torch.arange(tail)
                ).int()
        self.cache = cache.to(DEVICE)
        self.table = table.to(DEVICE)
        self.indices = indices.to(DEVICE)
        self.positions = positions.to(DEVICE)
        self.seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        self.requests = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
        self.q = torch.randn(rows, 6, HEAD, generator=generator).half().to(DEVICE)
        self.gate = torch.randn(rows, 6, HEAD, generator=generator).half().to(DEVICE)
        self.rows, self.seq_len = rows, seq_len

    def run(self, *, gate=True, scratch_tokens=None, max_sequence_length="seq", **kw):
        if max_sequence_length == "seq":
            max_sequence_length = self.seq_len
        out = torch.empty_like(self.q)
        result = qsa_ops.qsa_sparse_paged_attention(
            self.q,
            self.cache[:, 0],
            self.cache[:, 1],
            self.indices,
            self.table,
            self.requests,
            out=out,
            output_gate=self.gate if gate else None,
            query_positions=self.positions,
            sequence_lengths=self.seq_lens,
            kv_cache_dtype="nvfp4",
            k_scale=K_SCALE,
            v_scale=V_SCALE,
            max_sequence_length=max_sequence_length,
            **kw,
        )
        return result

    def reserve(self, tokens=None):
        return qsa_nvfp4.ensure_nvfp4_prefill_scratch(
            torch.device(DEVICE),
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=HEAD,
            capacity_tokens=tokens or self.seq_len,
        )

    def reference(self, *, gate=True) -> torch.Tensor:
        """Dense FP32 attention over the valid top-k entries, layer scales applied."""
        keys, values = nv.gather_dequant_nvfp4_kv(
            self.cache.cpu(),
            self.table.cpu(),
            self.requests.cpu(),
            self.indices.cpu(),
            k_scale=K_SCALE,
            v_scale=V_SCALE,
            out_dtype=torch.float32,
        )
        keep = nv.nvfp4_entry_validity(
            self.indices.cpu(), self.requests.cpu(), self.table.cpu(), 24, BLOCK
        )
        out = torch.zeros(self.rows, 6, HEAD)
        for row in range(self.rows):
            k, v = keys[row][keep[row]][:, 0], values[row][keep[row]][:, 0]
            scores = torch.einsum("hd,td->ht", self.q[row].cpu().float(), k)
            out[row] = torch.softmax(scores / math.sqrt(HEAD), dim=-1) @ v
        if gate:
            out = out * torch.sigmoid(self.gate.cpu().float())
        return out


def _rel_l2(actual, expected) -> float:
    a, e = actual.float().cpu(), expected.float().cpu()
    return float((a - e).norm() / e.norm().clamp_min(1e-12))


# --------------------------------------------------------------------- the knob


def test_the_prefill_knob_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", raising=False)
    assert qsa_nvfp4.nvfp4_prefill_scratch_enabled() is False
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", "1")
    assert qsa_nvfp4.nvfp4_prefill_scratch_enabled() is True


class _Recorder:
    """Stands in for a Triton kernel launch: ``kernel[grid](...)``."""

    def __init__(self) -> None:
        self.launches = 0

    def __getitem__(self, grid):
        return self._launch

    def _launch(self, *args, **kwargs) -> None:
        self.launches += 1


@pytest.fixture
def fused_recorder(monkeypatch):
    """The fused reader's launches are counted, not run (topk 2051 is too slow on the
    CPU interpreter); the split-K merge is a no-op for the same reason."""
    recorder = _Recorder()
    monkeypatch.setattr(qsa_ops, "_qsa_sparse_paged_gqa_splitk_kernel", recorder)
    monkeypatch.setattr(qsa_ops, "_qsa_merge_splitk_kernel", _Recorder())
    return recorder


def test_with_the_knob_off_the_existing_routes_run_unchanged(
    monkeypatch, fake_cuda, fused_recorder
) -> None:
    monkeypatch.delenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("the prefill route must not run with the knob off")

    monkeypatch.setattr(qsa_nvfp4, "qsa_sparse_attention_nvfp4_prefill", forbidden)
    recorder = fused_recorder
    batch = Batch()
    batch.run(nvfp4_fused_reader=True)
    assert recorder.launches == 1 and fake_cuda.calls == []

    gathered = []
    monkeypatch.setattr(
        qsa_nvfp4, "qsa_sparse_attention_nvfp4", lambda *a, **k: gathered.append(1)
    )
    batch.run(nvfp4_fused_reader=False)
    assert gathered == [1]


# ------------------------------------------------------------------ the route


def test_the_route_matches_dense_attention_with_both_layer_scales(
    knob, fake_cuda
) -> None:
    batch = Batch(rows=64, seq_len=600)
    scratch = batch.reserve()
    out = batch.run(nvfp4_fused_reader=True)
    assert _rel_l2(out, batch.reference()) < 2e-3
    kinds = [kind for kind, _ in fake_cuda.calls]
    assert kinds == ["plan", "grouped"]
    grouped = fake_cuda.calls[1][1]
    assert grouped["kv_cache_dtype"] == "auto"
    assert grouped["k_scale"] == grouped["v_scale"] == 1.0  # nothing folded twice
    assert grouped["softmax_scale"] == pytest.approx(HEAD**-0.5 * K_SCALE, rel=1e-6)
    assert grouped["k_ptr"] == scratch.key.data_ptr()
    assert grouped["v_ptr"] == scratch.value.data_ptr()
    plan = fake_cuda.calls[0][1]
    assert plan["page_size"] == BLOCK and plan["stride"] == BLOCK // 4
    assert plan["blocks"] == scratch.pages


def test_the_route_without_an_output_gate_scales_v_once(knob, fake_cuda) -> None:
    batch = Batch(rows=64, seq_len=500, seed=1)
    batch.reserve()
    out = batch.run(nvfp4_fused_reader=True, gate=False)
    assert _rel_l2(out, batch.reference(gate=False)) < 2e-3


def test_rows_that_do_not_fill_a_group_go_through_xqa_with_the_same_scale(
    knob, fake_cuda
) -> None:
    batch = Batch(rows=67, seq_len=700, seed=2)
    batch.reserve()
    out = batch.run(nvfp4_fused_reader=True)
    assert _rel_l2(out, batch.reference()) < 2e-3
    kinds = [kind for kind, _ in fake_cuda.calls]
    assert kinds == ["plan", "grouped", "xqa"]
    xqa = fake_cuda.calls[2][1]
    assert xqa["rows"] == 3 and xqa["kv_cache_dtype"] == "auto"
    assert xqa["k_scale"] == xqa["v_scale"] == 1.0
    assert xqa["softmax_scale"] == pytest.approx(HEAD**-0.5 * K_SCALE, rel=1e-6)


def test_the_first_chunk_of_a_prompt_reads_the_prefix_it_just_wrote(
    knob, fake_cuda
) -> None:
    # rows start at position 0: every selected token is inside the chunk itself.
    batch = Batch(rows=96, seq_len=96, seed=3, first_row=0)
    batch.reserve()
    out = batch.run(nvfp4_fused_reader=True)
    assert _rel_l2(out, batch.reference()) < 2e-3


# ------------------------------------------------------------------ fallbacks


def _expect_fused_fallback(recorder, batch, fake_cuda, **kw):
    batch.run(nvfp4_fused_reader=True, **kw)
    assert recorder.launches == 1, "expected the fused reader"
    assert fake_cuda.calls == []


def test_a_missing_scratch_falls_back_to_the_fused_reader(
    knob, fake_cuda, fused_recorder
) -> None:
    # Reserved nowhere: never allocate lazily, the KV pool is already sized.
    _expect_fused_fallback(fused_recorder, Batch(), fake_cuda)
    assert qsa_nvfp4.get_nvfp4_prefill_scratch(torch.device(DEVICE)) is None


def test_few_rows_stay_on_the_fused_reader(knob, fake_cuda, fused_recorder) -> None:
    batch = Batch(rows=16, seq_len=600)
    batch.reserve()
    _expect_fused_fallback(fused_recorder, batch, fake_cuda)


def test_a_context_longer_than_the_scratch_falls_back(
    knob, fake_cuda, fused_recorder
) -> None:
    batch = Batch(rows=64, seq_len=600)
    batch.reserve(tokens=BLOCK * 10)
    _expect_fused_fallback(fused_recorder, batch, fake_cuda)


def test_an_unknown_context_length_falls_back(
    knob, fake_cuda, fused_recorder
) -> None:
    batch = Batch(rows=64, seq_len=600)
    batch.reserve()
    _expect_fused_fallback(
        fused_recorder, batch, fake_cuda, max_sequence_length=None
    )


def test_a_capturing_stream_falls_back(
    knob, fake_cuda, fused_recorder, monkeypatch
) -> None:
    batch = Batch(rows=64, seq_len=600)
    batch.reserve()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _expect_fused_fallback(fused_recorder, batch, fake_cuda)


def test_a_missing_extension_falls_back(knob, fused_recorder, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn_v100", None)
    monkeypatch.setitem(sys.modules, "flash_attn_v100.flash_attn_interface", None)
    batch = Batch(rows=64, seq_len=600)
    batch.reserve()
    batch.run(nvfp4_fused_reader=True)
    assert fused_recorder.launches == 1


def test_the_gather_reader_is_the_fallback_when_the_fused_knob_is_off(
    knob, fake_cuda, monkeypatch
) -> None:
    batch = Batch(rows=16, seq_len=600)
    batch.reserve()
    gathered = []
    monkeypatch.setattr(
        qsa_nvfp4, "qsa_sparse_attention_nvfp4", lambda *a, **k: gathered.append(1)
    )
    batch.run(nvfp4_fused_reader=False)
    assert gathered == [1]


# ----------------------------------------------------------- scratch accounting


def test_the_scratch_is_sized_and_counted_by_the_bytes_it_holds() -> None:
    pages = qsa_nvfp4.nvfp4_prefill_scratch_pages(131072, 2784)
    assert pages == 48  # ceil(131072 / 2784)
    assert qsa_nvfp4.nvfp4_prefill_scratch_bytes(131072, 2784, 1, 256) == (
        48 * 2784 * 1 * 256 * 2 * 2
    )
    # 1 KiB per token (K and V, FP16, one head of 256): the design's 128 MiB.
    assert qsa_nvfp4.nvfp4_prefill_scratch_bytes(131072, 2784, 1, 256) < 140 * 2**20
    assert qsa_nvfp4.nvfp4_prefill_scratch_bytes(2784, 2784, 1, 256) == 2784 * 1024


def test_reserving_allocates_once_and_is_idempotent() -> None:
    device = torch.device(DEVICE)
    first = qsa_nvfp4.ensure_nvfp4_prefill_scratch(
        device, block_size=BLOCK, num_kv_heads=1, head_size=HEAD, capacity_tokens=400
    )
    again = qsa_nvfp4.ensure_nvfp4_prefill_scratch(
        device, block_size=BLOCK, num_kv_heads=1, head_size=HEAD, capacity_tokens=400
    )
    assert again is first
    assert first.key.shape == first.value.shape == (first.pages, BLOCK, 1, HEAD)
    assert first.key.dtype == torch.float16
    assert first.key.is_contiguous() and first.key.stride(0) == BLOCK * HEAD
    assert first.key.data_ptr() != first.value.data_ptr()
    assert first.pages == -(-400 // BLOCK)
    assert qsa_nvfp4.get_nvfp4_prefill_scratch(device) is first
    # Finite from the first byte: the planner's null microblock may read page 0.
    assert torch.isfinite(first.key).all() and torch.isfinite(first.value).all()


def test_the_scratch_environment_override_sets_the_capacity(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH_TOKENS", "720")
    assert qsa_nvfp4.nvfp4_prefill_scratch_capacity_tokens(131072) == 720
    monkeypatch.delenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH_TOKENS")
    assert qsa_nvfp4.nvfp4_prefill_scratch_capacity_tokens(131072) == 131072
    assert qsa_nvfp4.nvfp4_prefill_scratch_capacity_tokens(262144) == 262144


def test_reserve_does_nothing_with_the_knob_off_or_another_dtype(monkeypatch) -> None:
    kw = dict(
        device=torch.device(DEVICE),
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=HEAD,
        max_model_len=400,
    )
    monkeypatch.delenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", raising=False)
    assert qsa_nvfp4.reserve_nvfp4_prefill_scratch(kv_cache_dtype="nvfp4", **kw) is None
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", "1")
    assert (
        qsa_nvfp4.reserve_nvfp4_prefill_scratch(kv_cache_dtype="fp8_e4m3", **kw) is None
    )
    assert qsa_nvfp4.get_nvfp4_prefill_scratch(kw["device"]) is None
    assert (
        qsa_nvfp4.reserve_nvfp4_prefill_scratch(kv_cache_dtype="nvfp4", **kw)
        is not None
    )


def test_the_profile_run_reserves_the_scratch_before_the_pool_is_sized(
    knob, monkeypatch
) -> None:
    """The profile run returns early in ``_run_qsa`` (no attention metadata): the
    reservation must happen there, or the scratch would be allocated after the
    KV pool was sized."""
    layer = object.__new__(owner_module.Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer._qsa_kv_scales_finalized = True
    layer.kv_cache_dtype = "nvfp4"
    layer.qsa_dcp_sharded = False
    layer.num_kv_heads = 1
    layer.head_dim = HEAD
    layer._nvfp4_cache_config = SimpleNamespace(block_size=BLOCK)
    layer._nvfp4_model_config = SimpleNamespace(max_model_len=500)
    monkeypatch.setattr(
        owner_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    output = torch.ones(4, 6, HEAD, dtype=torch.float16, device=DEVICE)
    layer._run_qsa(
        torch.zeros(4, 8), torch.zeros(4), output, output, output, output
    )
    assert not output.any()
    scratch = qsa_nvfp4.get_nvfp4_prefill_scratch(torch.device(DEVICE))
    assert scratch is not None and scratch.pages == -(-500 // BLOCK)


# ------------------------------------------------------------- the log evidence


def test_the_route_state_is_logged_once(knob, fake_cuda, monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        qsa_ops.logger,
        "info_once",
        lambda msg, *args, **kw: lines.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        qsa_nvfp4.logger,
        "info_once",
        lambda msg, *args, **kw: lines.append(msg % args if args else msg),
    )
    batch = Batch(rows=64, seq_len=600)
    batch.reserve()
    batch.run(nvfp4_fused_reader=True)
    text = "\n".join(lines)
    assert "VLLM_SM70_QSA_NVFP4_FUSED_READER=1" in text
    assert "VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH=1" in text
    assert "prefill scratch route" in text


# -------------------------------------------------------------------- mutations


def _run_with(monkeypatch, fake_cuda, **patches):
    for target, name, value in patches.values():
        monkeypatch.setattr(target, name, value)
    batch = Batch(rows=64, seq_len=600, seed=5)
    batch.reserve()
    return _rel_l2(batch.run(nvfp4_fused_reader=True), batch.reference())


def test_the_route_parity_check_has_the_power_to_fail(
    knob, fake_cuda, monkeypatch
) -> None:
    """Each wrong variant of the scale folding or the table must break the check."""
    assert _run_with(monkeypatch, fake_cuda) < 2e-3

    original_gate = qsa_ops._qsa_output_scale_gate
    original_table = qsa_nvfp4.nvfp4_prefix_scratch_table

    def v_not_folded(out, gate, scale):  # m6'/m7: V scale missing
        return original_gate(out, gate, 1.0)

    def v_folded_twice(out, gate, scale):  # m7
        return original_gate(out, gate, scale * scale)

    def table_off_by_one(block_table, seq_lens, block_size, pages):  # m4
        compact, offsets = original_table(block_table, seq_lens, block_size, pages)
        return torch.where(compact >= 0, compact + 1, compact).int(), offsets + 1

    cases = {
        "v_not_folded": ("v", (qsa_ops, "_qsa_output_scale_gate", v_not_folded)),
        "v_folded_twice": ("v", (qsa_ops, "_qsa_output_scale_gate", v_folded_twice)),
        "table_off_by_one": (
            "t",
            (qsa_nvfp4, "nvfp4_prefix_scratch_table", table_off_by_one),
        ),
    }
    for name, (_, patch) in cases.items():
        with monkeypatch.context() as local:
            error = _run_with(local, fake_cuda, only=patch)
        assert error > 2e-3, f"mutation {name} was not caught (error {error})"


def test_a_k_scale_that_is_not_folded_is_caught(knob, fake_cuda, monkeypatch) -> None:
    # m6: softmax scale without k_scale. Patch the wrapper's call to drop the fold.
    original = qsa_ops._qsa_sparse_paged_attention_sm70_xqa_page4

    def unfolded(*args, softmax_scale=None, **kwargs):
        return original(*args, softmax_scale=HEAD**-0.5, **kwargs)

    monkeypatch.setattr(qsa_ops, "_qsa_sparse_paged_attention_sm70_xqa_page4", unfolded)
    batch = Batch(rows=64, seq_len=600, seed=6)
    batch.reserve()
    error = _rel_l2(batch.run(nvfp4_fused_reader=True), batch.reference())
    assert error > 2e-3


def test_forward_qsa_passes_the_host_side_longest_sequence_for_nvfp4(monkeypatch):
    """``forward_qsa`` hands ``attn_metadata.max_seq_len`` to the ops entry point
    for an NVFP4 cache (the scratch route needs it to prove its capacity without a
    device sync); a metadata without it passes nothing."""
    seen: list[dict] = []

    def record(*args, **kwargs):
        seen.append(kwargs)
        return args[6] if len(args) > 6 else kwargs["out"]

    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", record)
    q = torch.zeros(4, 6, HEAD, dtype=torch.float16)
    cache = torch.zeros(3, 2, BLOCK, 1, nv.nvfp4_kv_row_bytes(HEAD), dtype=torch.uint8)
    layer = SimpleNamespace(
        topk_indices_buffer=torch.zeros(4, TOPK, dtype=torch.int32),
        qsa_dcp_sharded=False,
        _k_scale_float=K_SCALE,
        _v_scale_float=V_SCALE,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        block_table=torch.zeros(1, 3, dtype=torch.int32),
        max_seq_len=123,
    )

    def run(dtype):
        impl = object.__new__(owner_module.Qwen4ExpQSAFlashAttentionImpl)
        impl.kv_cache_dtype = dtype
        impl.alibi_slopes = None
        impl.sinks = None
        impl.sliding_window = (-1, -1)
        impl.forward_qsa(
            layer,
            q,
            None,
            None,
            cache,
            metadata,
            torch.zeros_like(q),
            token_to_req=torch.zeros(4, dtype=torch.int32),
        )

    run("nvfp4")
    assert seen[-1]["max_sequence_length"] == 123
    metadata.max_seq_len = None  # a metadata without it passes nothing
    run("nvfp4")
    assert "max_sequence_length" not in seen[-1]
