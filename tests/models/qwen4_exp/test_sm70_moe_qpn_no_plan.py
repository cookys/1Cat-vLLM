# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU bitwise test for Q2.16 option C (VLLM_SM70_MOE_QPN_NO_PLAN).

The planned path (plan_kernel -> w13_kernel -> w2_batch_reduce_kernel) and the
NoPlan path (w13 CTAs derive the grouping -> the unchanged w2_batch_reduce_kernel)
must agree bit for bit: ``intermediate`` after W13 and ``output`` after the W2
batch reduction, M=5, split 4, interleaved W13 (the production MTP4 call). Needs a
V100 and an extension built with ``nvfp4_grouped_w13_noplan_sm70_out``; otherwise
every test here is skipped. The group tables are compared as sets (plan_kernel's
group numbering is an atomicAdd race; NoPlan's is the leader rank).

    cd <worktree> && PYTHONPATH=$PWD python -m pytest \\
        tests/models/qwen4_exp/test_sm70_moe_qpn_no_plan.py -q
"""

import numpy as np
import pytest
import torch

from . import qpn_no_plan_ref as ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (V100)")

GROUP = 16
HIDDEN, INTER, TOP_K, ROWS = 2560, 160, 10, 5
POOL = 8
U_LIST = (10, 20, 35, 50)
MODES = ("uniform", "skewed", "staircase")


@pytest.fixture(scope="module")
def ops():
    from vllm import _sm70_ops as ops

    if not (
        ops.has_nvfp4_grouped_decode_dispatch()
        and ops.has_nvfp4_grouped_batch_reduce_dispatch()
        and ops.has_nvfp4_grouped_noplan_dispatch()
    ):
        pytest.skip("extension without the grouped / NoPlan ops; build this branch first")
    return ops


@pytest.fixture(scope="module")
def weights(ops):
    """POOL distinct random experts tiled to 512, prepared like production."""
    gen = torch.Generator().manual_seed(0)
    w13, s13, w2, s2 = [], [], [], []
    for _ in range(POOL):
        n13 = torch.randint(0, 16, (HIDDEN, 2 * INTER), dtype=torch.uint8, generator=gen).cuda()
        c13 = (torch.rand(HIDDEN // GROUP, 2 * INTER, generator=gen) * 0.008 + 0.002).half().cuda()
        n2 = torch.randint(0, 16, (INTER, HIDDEN), dtype=torch.uint8, generator=gen).cuda()
        c2 = (torch.rand(INTER // GROUP, HIDDEN, generator=gen) * 0.008 + 0.002).half().cuda()
        a = ops.nvfp4_sm70_prepare(n13, c13, GROUP, True)
        b = ops.nvfp4_sm70_prepare(n2, c2, GROUP, False)
        w13.append(a[0]), s13.append(a[1]), w2.append(b[0]), s2.append(b[1])
    reps = 512 // POOL
    stacked = [torch.stack(t) for t in (w13, s13, w2, s2)]
    return tuple(t.repeat(reps, *([1] * (t.dim() - 1))).contiguous() for t in stacked)


def new_meta():
    return (
        torch.full((160, 8), -7, dtype=torch.int32, device="cuda"),
        torch.full((160,), -7, dtype=torch.int32, device="cuda"),
        torch.full((160,), -7, dtype=torch.int32, device="cuda"),
        torch.full((1,), -7, dtype=torch.int32, device="cuda"),
    )


def groups_of(meta, ids):
    rows, experts, sizes, total = (t.cpu() for t in meta)
    n = int(total.item())
    flat = ids.reshape(-1).tolist()
    groups = []
    for g in range(n):
        c = int(sizes[g])
        groups.append((ref.sanitize(int(experts[g])), tuple(sorted(int(rows[g, j]) for j in range(c)))))
    assert all(ref.sanitize(flat[r]) == e for e, rs in groups for r in rs)
    return sorted(groups)


def run_pair(ops, ws, x, ids, tw, noplan, fill):
    """W13 then W2 batch-reduce; buffers pre-filled with ``fill`` so a missed write shows."""
    meta = new_meta()
    mid = torch.full((ROWS * TOP_K, INTER), fill, dtype=torch.float16, device="cuda")
    routed = torch.empty(ROWS * TOP_K, HIDDEN, dtype=torch.float16, device="cuda")
    out = torch.full((ROWS, HIDDEN), fill, dtype=torch.float16, device="cuda")
    w13 = ops.nvfp4_grouped_w13_noplan_sm70_out if noplan else ops.nvfp4_grouped_w13_sm70_out
    w13(mid, x, ws[0], ws[1], ids.reshape(-1), *meta, 4, True)
    ops.nvfp4_grouped_w2_batch_reduce_sm70_out(out, routed, mid, ws[2], ws[3], tw, *meta)
    torch.cuda.synchronize()
    return mid, out, meta


def check_case(ops, ws, ids_np, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, HIDDEN, generator=gen).half().cuda()
    tw = torch.softmax(torch.randn(ROWS, TOP_K, generator=gen), -1).cuda()
    ids = torch.as_tensor(np.asarray(ids_np, dtype=np.int32).reshape(ROWS, TOP_K)).cuda().contiguous()
    mid_a, out_a, meta_a = run_pair(ops, ws, x, ids, tw, noplan=False, fill=1.0)
    mid_a2, out_a2, _ = run_pair(ops, ws, x, ids, tw, noplan=False, fill=3.0)
    # the baseline must itself be reproducible, or equality below would mean nothing
    assert torch.equal(mid_a, mid_a2) and torch.equal(out_a, out_a2), "planned path is not run-to-run stable"
    mid_b, out_b, meta_b = run_pair(ops, ws, x, ids, tw, noplan=True, fill=2.0)
    assert groups_of(meta_a, ids.cpu().numpy()) == groups_of(meta_b, ids.cpu().numpy())
    assert torch.equal(mid_a, mid_b), "intermediate differs"
    assert torch.equal(out_a, out_b), "output differs"
    # NoPlan numbers groups by leader rank in slot order
    rows_b = meta_b[0].cpu()
    firsts = [int(rows_b[g, 0]) for g in range(int(meta_b[3].item()))]
    assert firsts == sorted(firsts)
    return ref.multiplicity_histogram(ids.cpu().numpy())


@pytest.mark.parametrize("u", U_LIST)
@pytest.mark.parametrize("mode", MODES)
def test_noplan_matches_planned_bitwise(ops, weights, u, mode):
    for seed in range(4):
        ids = ref.routing_with_multiplicity(u, seed, mode)
        check_case(ops, weights, ids, seed)


def test_every_row_count_one_to_five_is_covered(ops, weights):
    seen = set()
    for u in U_LIST:
        for mode in MODES:
            for seed in range(2):
                ids = ref.routing_with_multiplicity(u, seed, mode)
                seen |= set(check_case(ops, weights, ids, seed))
    assert seen == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("seed", range(3))
def test_noplan_matches_planned_with_repeated_and_invalid_ids(ops, weights, seed):
    """Beyond the router's contract: repeated experts inside a token (>8 rows in
    one group) and ids outside [0, 512) (zero rows) behave exactly like the planned path."""
    rng = np.random.default_rng(seed)
    pool = rng.choice(512, size=6, replace=False)
    ids = rng.choice(pool, size=ROWS * TOP_K)
    bad = rng.choice(ROWS * TOP_K, size=7, replace=False)
    ids[bad] = rng.choice([-1, 512, 100000], size=7)
    check_case(ops, weights, ids, seed)


def test_expert_with_more_than_eight_rows_splits_into_two_packs(ops, weights):
    """One expert in 12 slots (repeated inside tokens): packs of 8 and 4, as planned."""
    rng = np.random.default_rng(11)
    ids = np.concatenate([np.full(12, 9), rng.choice(np.arange(100, 400), size=38, replace=False)])
    rng.shuffle(ids)
    check_case(ops, weights, ids, 11)
    meta = None
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(ROWS, HIDDEN, generator=gen).half().cuda()
    tw = torch.softmax(torch.randn(ROWS, TOP_K, generator=gen), -1).cuda()
    t = torch.as_tensor(ids.astype(np.int32).reshape(ROWS, TOP_K)).cuda().contiguous()
    _, _, meta = run_pair(ops, weights, x, t, tw, noplan=True, fill=2.0)
    sizes = [len(rs) for e, rs in groups_of(meta, t.cpu().numpy()) if e == 9]
    assert sorted(sizes) == [4, 8]


@pytest.mark.parametrize("bad", [-1, 512, 100000])
def test_invalid_ids_use_zero_bucket_and_give_zero_rows(ops, weights, bad):
    rng = np.random.default_rng(bad & 0xFFFF)
    ids = ref.routing_with_multiplicity(20, 1, "uniform").reshape(-1).copy()
    slots = rng.choice(ROWS * TOP_K, size=9, replace=False)
    ids[slots] = bad
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(ROWS, HIDDEN, generator=gen).half().cuda()
    tw = torch.softmax(torch.randn(ROWS, TOP_K, generator=gen), -1).cuda()
    t = torch.as_tensor(ids.astype(np.int32).reshape(ROWS, TOP_K)).cuda().contiguous()
    for noplan in (False, True):
        mid, _, meta = run_pair(ops, weights, x, t, tw, noplan=noplan, fill=2.0)
        assert not mid[torch.as_tensor(slots).cuda()].any(), "invalid-id rows must be exactly zero"
        assert (512, tuple(sorted(slots.tolist()))) in groups_of(meta, t.cpu().numpy())
    check_case(ops, weights, ids, 3)


def test_fallback_shape_keeps_planned_op_and_runs(ops, weights, monkeypatch):
    """M=8 (80 routes, split 4) is outside NoPlan: knob on still selects the planned op."""
    from vllm import envs
    from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

    monkeypatch.setenv("VLLM_SM70_MOE_QPN_NO_PLAN", "1")
    envs.disable_envs_cache()
    seen = []
    monkeypatch.setattr(moe.logger, "warning_once", lambda *a, **k: seen.append(a[0]))
    x = torch.randn(8, HIDDEN).half().cuda()
    ids = ref.routing_with_multiplicity(10, 0, "uniform", rows=8, top_k=10).astype(np.int32)
    t = torch.as_tensor(ids).cuda().contiguous()
    op = moe._select_grouped_w13_op(x, t, 4)
    assert op is ops.nvfp4_grouped_w13_sm70_out
    assert seen and "NO_PLAN" in seen[0]
    mid = torch.zeros(80, INTER, dtype=torch.float16, device="cuda")
    op(mid, x, weights[0], weights[1], t.reshape(-1), *new_meta(), 4, True)
    torch.cuda.synchronize()
    assert mid.abs().sum() > 0


def test_python_op_wrapper_signature_matches_old_op(ops):
    import inspect

    new = list(inspect.signature(ops.nvfp4_grouped_w13_noplan_sm70_out).parameters)
    old = list(inspect.signature(ops.nvfp4_grouped_w13_sm70_out).parameters)
    assert new == old
