# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for Q2.16 option C (VLLM_SM70_MOE_QPN_NO_PLAN).

What runs here: the ballot-based expert grouping of ``w13_kernel<S, I, true>``
replayed lane by lane in numpy (``qpn_no_plan_ref``) against the grouping
``plan_kernel`` defines (unique group leaders, no row lost or duplicated, group
content equal to plan_kernel's up to its unspecified numbering), and the Python
dispatch (knob default, M != 5 / missing-op fallbacks). The kernel itself is only
compiled for checking (ptxas), never run here; its bitwise test is
``test_sm70_moe_qpn_no_plan.py`` (needs a GPU).

    cd <worktree> && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 \\
        nice -n 19 taskset -c 0-26:2 /data/venvs/1cat-p070/bin/python -m pytest \\
        tests/models/qwen4_exp/test_sm70_moe_qpn_no_plan_cpu.py -q
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

from . import qpn_no_plan_ref as ref

U_LIST = (10, 20, 35, 50)
MODES = ("uniform", "skewed", "staircase")


# ------------------------------------------------------------ routing makers
@pytest.mark.parametrize("u", U_LIST)
@pytest.mark.parametrize("mode", MODES)
def test_routing_maker_contract(u, mode):
    for seed in range(5):
        ids = ref.routing_with_multiplicity(u, seed, mode)
        assert ids.shape == (5, 10) and ids.dtype == np.int32
        assert len(np.unique(ids)) == u
        assert all(len(set(row)) == 10 for row in ids.tolist())
        assert ids.min() >= 0 and ids.max() < 512


def test_routing_makers_cover_every_row_count_1_to_5():
    seen: dict[int, set[str]] = {m: set() for m in range(1, 6)}
    for u in U_LIST:
        for mode in MODES:
            for seed in range(5):
                hist = ref.multiplicity_histogram(ref.routing_with_multiplicity(u, seed, mode))
                for m in hist:
                    seen[m].add(f"U{u}/{mode}")
    assert all(seen[m] for m in range(1, 6)), {m: sorted(v)[:3] for m, v in seen.items()}


# ------------------------------------------------------ ballot grouping mirror
@pytest.mark.parametrize("u", U_LIST)
@pytest.mark.parametrize("mode", MODES)
def test_m5_distinct_topk_grouping_matches_plan_kernel(u, mode):
    for seed in range(8):
        ids = ref.routing_with_multiplicity(u, seed, mode).reshape(-1).tolist()
        out = ref.check_plan(ids, 50)
        # distinct top-k per token: at most 5 rows per expert, one pack each
        assert out["total"] == u
        assert max(out["sizes"].values()) <= 5


def test_hand_checked_example():
    # tokens: [3,1,2,... ] style; expert 7 appears in slots 0, 3, 11 -> one group of 3
    ids = [7, 1, 2, 7, 3, 4, 5, 6, 8, 9, 10, 7] + list(range(100, 100 + 38))
    out = ref.check_plan(ids, 50)
    groups = ref.groups_from_arrays(out)
    assert (7, (0, 3, 11)) in groups
    assert out["total"] == 50 - 2  # slots 3 and 11 join slot 0's group
    # group numbers follow leader slot order: slot 0 -> 0, slot 1 -> 1, slot 2 -> 2, slot 4 -> 3 ...
    assert out["experts"][0] == 7 and out["experts"][1] == 1 and out["experts"][3] == 3


@pytest.mark.parametrize("routes", [1, 2, 8, 10, 31, 32, 33, 49, 50, 63, 64])
def test_every_route_count_up_to_64_with_duplicates(routes):
    rng = np.random.default_rng(routes)
    for trial in range(40):
        pool = rng.choice(512, size=int(rng.integers(1, 12)), replace=False)
        ids = rng.choice(pool, size=routes).tolist()  # heavy duplication, >8 rows/expert
        ref.check_plan(ids, routes)


def test_invalid_ids_share_one_zero_bucket():
    rng = np.random.default_rng(0)
    for trial in range(60):
        ids = rng.integers(0, 512, size=50)
        bad = rng.choice(50, size=int(rng.integers(1, 20)), replace=False)
        ids[bad] = rng.choice([-1, -7, 512, 513, 100000], size=len(bad))
        out = ref.check_plan(ids.tolist(), 50)
        # every invalid slot lands in an expert-512 group, 8 rows at a time
        invalid = sum(1 for i in ids.tolist() if i < 0 or i >= 512)
        assert sum(out["sizes"][g] for g, e in out["experts"].items() if e == 512) == invalid


def test_more_than_eight_rows_split_into_ordered_packs():
    ids = [5] * 20 + list(range(100, 130))
    out = ref.check_plan(ids, 50)
    sizes = sorted(out["sizes"][g] for g, e in out["experts"].items() if e == 5)
    assert sizes == [4, 8, 8]
    groups = [(e, rs) for e, rs in ref.groups_from_arrays(out) if e == 5]
    assert groups == [(5, tuple(range(0, 8))), (5, tuple(range(8, 16))), (5, tuple(range(16, 20)))]


def test_random_fuzz_matches_reference():
    rng = np.random.default_rng(2026)
    for trial in range(300):
        routes = int(rng.integers(1, 65))
        span = int(rng.choice([3, 10, 60, 512]))
        ids = rng.integers(0, span, size=routes)
        flip = rng.random(routes) < rng.choice([0.0, 0.1, 0.5])
        ids[flip] = rng.choice([-1, 512, 4096], size=int(flip.sum()))
        ref.check_plan(ids.tolist(), routes)


@pytest.mark.parametrize("u", U_LIST)
@pytest.mark.parametrize("mode", MODES)
def test_equivalent_to_every_old_plan_kernel_schedule(u, mode):
    """Group content equals plan_kernel's under random atomicAdd arrival orders."""
    rng = np.random.default_rng(u)
    for seed in range(4):
        ids = ref.routing_with_multiplicity(u, seed, mode).reshape(-1).tolist()
        new = ref.run_all(ids, 50)
        for _ in range(5):
            ref.equivalent_to_plan_kernel(new, ref.plan_kernel_mirror(ids, 50, rng))


def test_equivalent_to_old_plan_with_repeats_and_invalid_ids():
    rng = np.random.default_rng(7)
    for trial in range(100):
        routes = int(rng.integers(1, 65))
        ids = rng.integers(0, int(rng.choice([2, 5, 40])), size=routes)
        flip = rng.random(routes) < rng.choice([0.0, 0.2, 0.6])
        ids[flip] = rng.choice([-1, 512, 99999], size=int(flip.sum()))
        ids = ids.tolist()
        new = ref.check_plan(ids, routes)
        ref.equivalent_to_plan_kernel(new, ref.plan_kernel_mirror(ids, routes, rng))


def test_expert_with_more_than_eight_rows_splits_in_two_packs_like_old():
    ids = [9] * 12 + list(range(100, 138))  # one expert, 12 rows (repeated within tokens)
    new = ref.check_plan(ids, 50)
    sizes = sorted(new["sizes"][g] for g, e in new["experts"].items() if e == 9)
    assert sizes == [4, 8]
    old = ref.plan_kernel_mirror(ids, 50, np.random.default_rng(0))
    assert sorted(old["sizes"][g] for g, e in old["experts"].items() if e == 9) == [4, 8]


def test_nth_set_bit_matches_naive():
    rng = np.random.default_rng(1)
    for _ in range(300):
        mask = int(rng.integers(1, 1 << 62)) | (int(rng.integers(0, 2)) << 63)
        bits = [i for i in range(64) if mask >> i & 1]
        for n in range(min(len(bits), 8)):
            assert ref.nth_set_bit(mask, n) == bits[n]


# ------------------------------------------------------------ Python dispatch
def test_knob_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_MOE_QPN_NO_PLAN", raising=False)
    envs.disable_envs_cache()
    assert not envs.VLLM_SM70_MOE_QPN_NO_PLAN


def _knob(monkeypatch, value):
    monkeypatch.setenv("VLLM_SM70_MOE_QPN_NO_PLAN", value)
    envs.disable_envs_cache()


def test_dispatch_knob_off_never_selects_noplan(monkeypatch):
    _knob(monkeypatch, "0")
    monkeypatch.setattr(moe.sm70_ops, "has_nvfp4_grouped_noplan_dispatch", lambda: True)
    x = torch.empty(5, 2560, dtype=torch.float16)
    ids = torch.empty(5, 10, dtype=torch.int32)
    assert not moe._use_grouped_noplan(x, ids)


def test_dispatch_selects_noplan_only_for_m5_with_op(monkeypatch):
    _knob(monkeypatch, "1")
    monkeypatch.setattr(moe.sm70_ops, "has_nvfp4_grouped_noplan_dispatch", lambda: True)
    x = torch.empty(5, 2560, dtype=torch.float16)
    ids = torch.empty(5, 10, dtype=torch.int32)
    assert moe._use_grouped_noplan(x, ids)
    for tokens in (1, 2, 4, 6, 8, 16):
        xt = torch.empty(tokens, 2560, dtype=torch.float16)
        it = torch.empty(tokens, 10, dtype=torch.int32)
        assert not moe._use_grouped_noplan(xt, it), tokens


def test_dispatch_falls_back_when_build_has_no_op(monkeypatch):
    _knob(monkeypatch, "1")
    monkeypatch.setattr(moe.sm70_ops, "has_nvfp4_grouped_noplan_dispatch", lambda: False)
    x = torch.empty(5, 2560, dtype=torch.float16)
    ids = torch.empty(5, 10, dtype=torch.int32)
    assert not moe._use_grouped_noplan(x, ids)


def _warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(moe.logger, "warning_once", lambda *a, **k: seen.append(a[0]))
    return seen


@pytest.mark.parametrize(
    "tokens,split",
    [(8, 4), (16, 8), (5, 8), (5, 1), (5, 5), (6, 4), (1, 4)],  # routes > 64 or split != 4
)
def test_fallback_shapes_select_old_op_and_warn(monkeypatch, tokens, split):
    _knob(monkeypatch, "1")
    monkeypatch.setattr(moe.sm70_ops, "has_nvfp4_grouped_noplan_dispatch", lambda: True)
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_grouped_w13_noplan_sm70_out", lambda *a: "noplan", raising=False)
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_grouped_w13_sm70_out", lambda *a: "planned", raising=False)
    seen = _warnings(monkeypatch)
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    assert moe._select_grouped_w13_op(x, ids, split)() == "planned"
    assert seen and "NO_PLAN" in seen[0] and "plan_kernel" in seen[0]


def test_admitted_shape_selects_noplan_op_silently(monkeypatch):
    _knob(monkeypatch, "1")
    monkeypatch.setattr(moe.sm70_ops, "has_nvfp4_grouped_noplan_dispatch", lambda: True)
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_grouped_w13_noplan_sm70_out", lambda *a: "noplan", raising=False)
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_grouped_w13_sm70_out", lambda *a: "planned", raising=False)
    seen = _warnings(monkeypatch)
    x = torch.empty(5, 2560, dtype=torch.float16)
    ids = torch.empty(5, 10, dtype=torch.int32)
    assert moe._select_grouped_w13_op(x, ids, 4)() == "noplan"
    assert not seen


def test_missing_op_build_selects_old_op_and_warns(monkeypatch):
    _knob(monkeypatch, "1")
    monkeypatch.setattr(moe.sm70_ops, "has_nvfp4_grouped_noplan_dispatch", lambda: False)
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_grouped_w13_sm70_out", lambda *a: "planned", raising=False)
    seen = _warnings(monkeypatch)
    x = torch.empty(5, 2560, dtype=torch.float16)
    ids = torch.empty(5, 10, dtype=torch.int32)
    assert moe._select_grouped_w13_op(x, ids, 4)() == "planned"
    assert seen and "no" in seen[0]


def test_knob_off_selects_old_op_without_warning(monkeypatch):
    _knob(monkeypatch, "0")
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_grouped_w13_sm70_out", lambda *a: "planned", raising=False)
    seen = _warnings(monkeypatch)
    for tokens, split in ((5, 4), (8, 4), (16, 8)):
        x = torch.empty(tokens, 2560, dtype=torch.float16)
        ids = torch.empty(tokens, 10, dtype=torch.int32)
        assert moe._select_grouped_w13_op(x, ids, split)() == "planned"
    assert not seen


def test_grouped_branch_uses_noplan_op_object(monkeypatch):
    """The forward picks the op by the helper only; old op stays the default."""
    import inspect

    src = inspect.getsource(moe)
    assert "_select_grouped_w13_op(x, topk_ids, grouped_split)(" in src
    assert "grouped_split,\n                interleaved_w13," in src
