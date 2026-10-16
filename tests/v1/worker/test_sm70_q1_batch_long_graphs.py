# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the plan 075 M-fix12 E q1 batch long-context graph variants."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.cudagraph_utils import (
    SM70_Q1_BATCH_LONG_SEQ,
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)

BATCHES = tuple(range(2, 9))


def _manager(monkeypatch, *, long_graphs=True, q1_batch=True, max_reqs=8):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_Q1_BATCH_LONG_GRAPHS", "1" if long_graphs else "0")
    m = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    m.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    m._capture_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 48, 64]
    m.decode_query_len = 8
    m.decode_query_lens = (8, 7, 6, 5, 4, 3, 2, 1)
    m.max_num_reqs = max_reqs
    m._sm70_dflash2_tail_graphs = True
    m._sm70_dflash2_q1_batch_graphs = q1_batch
    m._capture_descs = {}
    m._candidates = []
    m._long_attention_graphs = {}
    m._q1_batch_long_graphs = {}
    m.graphs = {}
    m._init_candidates()
    m._init_q1_batch_long_graphs()
    return m


def _q1(b):
    return BatchExecutionDescriptor(CUDAGraphMode.FULL, b, b, 1)


def test_threshold_matches_eager_partition_switch():
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _select_default_decode_partition_size as sel,
    )

    assert sel(SM70_Q1_BATCH_LONG_SEQ - 1) == 256
    assert sel(SM70_Q1_BATCH_LONG_SEQ) == 1024


def test_variant_count_and_set(monkeypatch):
    m = _manager(monkeypatch)
    full = m._capture_descs[CUDAGraphMode.FULL]
    base = [d for d in full if d.decode_seq_hint is None]
    var = [d for d in full if d.decode_seq_hint is not None]
    # 6 q8 + 7 single-request tails + 7 q1 batches (B=2..8)
    assert len(base) == 20
    assert len(var) == len(BATCHES) == 7
    assert {d.num_reqs for d in var} == set(BATCHES)
    assert all(d.uniform_token_count == 1 and d.num_tokens == d.num_reqs for d in var)
    assert all(d.decode_seq_hint == SM70_Q1_BATCH_LONG_SEQ for d in var)
    assert set(m._q1_batch_long_graphs) == {_q1(b) for b in BATCHES}


def test_capture_order_base_then_variants_each_descending(monkeypatch):
    m = _manager(monkeypatch)
    full = m._capture_descs[CUDAGraphMode.FULL]
    base_tokens = [d.num_tokens for d in full if d.decode_seq_hint is None]
    var_tokens = [d.num_tokens for d in full if d.decode_seq_hint is not None]
    assert base_tokens == sorted(base_tokens, reverse=True)
    assert var_tokens == sorted(var_tokens, reverse=True)
    first_var = next(i for i, d in enumerate(full) if d.decode_seq_hint is not None)
    assert all(d.decode_seq_hint is not None for d in full[first_var:])


def test_variants_never_dispatchable(monkeypatch):
    m = _manager(monkeypatch)
    assert all(
        d.decode_seq_hint is None for cands in m._candidates for d in cands
    )


@pytest.mark.parametrize("long_graphs,q1_batch", [(False, True), (True, False)])
def test_off_keeps_descriptor_set(monkeypatch, long_graphs, q1_batch):
    m = _manager(monkeypatch, long_graphs=long_graphs, q1_batch=q1_batch)
    full = m._capture_descs[CUDAGraphMode.FULL]
    assert all(d.decode_seq_hint is None for d in full)
    assert m._q1_batch_long_graphs == {}


def test_max_num_seqs_16_adds_b16_variant(monkeypatch):
    m = _manager(monkeypatch, max_reqs=16)
    assert {d.num_reqs for d in m._q1_batch_long_graphs} == {*BATCHES, 16}


@pytest.mark.parametrize("b", BATCHES)
def test_select_table(monkeypatch, b):
    m = _manager(monkeypatch)
    base = _q1(b)
    long = m._q1_batch_long_graphs[base]
    m.graphs = {base: object(), long: object()}
    table = (
        (1, base),
        (SM70_Q1_BATCH_LONG_SEQ - 1, base),
        (SM70_Q1_BATCH_LONG_SEQ, long),
        (131072, long),
        (262144, long),
    )
    for upper, expected in table:
        assert m.select_attention_graph(base, torch.tensor([upper] * b)) == expected
    # One long request promotes the whole batch; padded requests are ignored.
    mixed = torch.tensor([100] * (b - 1) + [SM70_Q1_BATCH_LONG_SEQ])
    assert m.select_attention_graph(base, mixed) == long
    assert m.select_attention_graph(base, torch.tensor([SM70_Q1_BATCH_LONG_SEQ])) == long


def test_select_conservative_cases(monkeypatch):
    m = _manager(monkeypatch)
    base = _q1(4)
    long = m._q1_batch_long_graphs[base]
    hint = torch.tensor([40000] * 4)
    # variant not captured
    m.graphs = {base: object()}
    assert m.select_attention_graph(base, hint) == base
    m.graphs = {base: object(), long: object()}
    # device hint is never read, more hints than requests, empty hint
    dev = SimpleNamespace(device=torch.device("cuda"))
    assert m.select_attention_graph(base, dev) == base
    assert m.select_attention_graph(base, torch.tensor([40000] * 5)) == base
    assert m.select_attention_graph(base, torch.tensor([], dtype=torch.int64)) == base
    # q8 / single-request selection is untouched
    q8 = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 1, 8)
    assert m.select_attention_graph(q8, hint) == q8


def test_dummy_capture_seq_len(monkeypatch):
    seen = {}

    class _State:
        def prepare_attn(self, input_batch, *a, **k):
            seen["seq_lens"] = input_batch.seq_lens.clone()
            seen["qsl"] = input_batch.query_start_loc.clone()
            seen["ub"] = input_batch.seq_lens_cpu_upper_bound.clone()
            return {}

    n = 4
    buffers = SimpleNamespace(
        device=torch.device("cpu"),
        seq_lens=torch.zeros(8, dtype=torch.int32),
        query_start_loc=torch.zeros(9, dtype=torch.int32),
        input_ids=torch.zeros(64, dtype=torch.int64),
        positions=torch.zeros(64, dtype=torch.int64),
        dcp_local_seq_lens=None,
    )
    tables = SimpleNamespace(
        cp_size=1,
        get_dummy_block_tables=lambda r: (torch.zeros(r, 4, dtype=torch.int32),),
        get_dummy_slot_mappings=lambda t: torch.zeros(1, t, dtype=torch.int64),
    )
    monkeypatch.setattr(cg, "build_slot_mappings_by_layer", lambda *a, **k: {})
    for hint, want in ((None, 1), (SM70_Q1_BATCH_LONG_SEQ, SM70_Q1_BATCH_LONG_SEQ)):
        cg.prepare_inputs_to_capture(
            n, n, _State(), buffers, tables, [], None, decode_seq_hint=hint
        )
        assert seen["seq_lens"].tolist() == [want] * n
        assert seen["ub"].tolist() == [want] * n
        # query lengths (and so query_start_loc) are unchanged by the hint
        assert seen["qsl"].tolist() == [0, 1, 2, 3, 4]


def _decode_plan(seq_hint):
    """The plan a q1 decode capture/replay gets for ``seq_hint`` (meta tensors)."""
    from flash_attn_v100 import flash_attn_interface as fi

    q = torch.empty(4, 6, 256, dtype=torch.float16, device="meta")
    k_cache = torch.empty(2, 4096, 1, 144, dtype=torch.uint8, device="meta")
    block_table = torch.empty(4, 64, dtype=torch.int32, device="meta")
    active = torch.empty(1, dtype=torch.int32, device="meta")
    # max_model_len 262144 = 64 blocks x 4096; the capture workspace envelope.
    return fi._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=seq_hint,
        batch_size_hint=4,
        workspace_seq_capacity_hint=262144,
        active_num_partitions=active,
    )


@pytest.mark.parametrize(
    "hint,partition_size,launch",
    [
        (1, 256, 1024),  # base q1 graph: captured with a 1-token dummy
        (SM70_Q1_BATCH_LONG_SEQ - 1, 256, 1024),
        (SM70_Q1_BATCH_LONG_SEQ, 1024, 256),  # long variant
    ],
)
def test_decode_plan_at_capture_hint(hint, partition_size, launch):
    plan = _decode_plan(hint)
    assert plan.partition_size == partition_size
    assert plan.launch_num_partitions == launch
    assert plan.workspace_num_partitions == launch


def test_decode_plan_long_variant_covers_254k():
    plan = _decode_plan(254_000)
    assert plan.partition_size == 1024
    assert plan.actual_num_partitions <= 256
    assert plan.launch_num_partitions == 256
    # the variant's launch grid covers the full served window
    assert plan.launch_num_partitions * plan.partition_size >= 262144
