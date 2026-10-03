# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the helpers of benchmarks/sm70_sampling_graph_replay_check.py.

The script itself needs a GPU (UVA buffers, Triton kernels, graph capture).
What can be checked without one is the part that decides pass or fail: the
bit-for-bit round comparison, the draft-token generator that makes the rounds
mix accepted and rejected drafts, the argument validation and the exit status
when there is no CUDA device.

    cd <worktree> && CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> \
        uv run --no-project --python <venv>/bin/python --with pytest -- \
        python -m pytest --noconftest \
        tests/v1/sample/test_sm70_sampling_graph_replay_check.py -q
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "sm70_sampling_graph_replay_check.py"
)
PER_REQ = 5
NUM_SPEC = 4


@pytest.fixture(scope="module")
def chk():
    spec = importlib.util.spec_from_file_location("sm70_sampling_replay_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class FakeRig:
    def __init__(self, **state):
        self.state = {k: torch.as_tensor(v).clone() for k, v in state.items()}

    def state_tensors(self):
        return self.state


def outputs(chk, sampled, num_sampled=(5,), num_rejected=(0,)):
    return chk.RoundOutputs(
        sampled=torch.tensor(sampled),
        num_sampled=torch.tensor(num_sampled, dtype=torch.int32),
        num_rejected=torch.tensor(num_rejected, dtype=torch.int32),
        host_us=1.0,
        gpu_us=1.0,
    )


def test_identical_rounds_compare_equal(chk):
    rig = FakeRig(num_computed_tokens=[70], total_len=[70])
    out = outputs(chk, [[1, 2, 3, 4, 5]])
    assert chk.compare_round(0, rig, out, FakeRig(**rig.state), out) is None


def test_a_differing_token_is_reported_with_its_flat_index(chk):
    rig = FakeRig(num_computed_tokens=[70])
    eager = outputs(chk, [[1, 2, 3, 4, 5]])
    graph = outputs(chk, [[1, 2, 9, 4, 5]])
    got = chk.compare_round(7, rig, eager, FakeRig(**rig.state), graph)
    assert got == {
        "round": 7,
        "field": "sampled",
        "first_differing_flat_indices": [2],
        "eager": [3],
        "graph": [9],
    }


def test_the_unwritten_tail_after_a_rejection_is_not_compared(chk):
    """sampled is new_empty; entries past num_sampled hold recycled memory."""
    rig = FakeRig(num_computed_tokens=[70])
    eager = outputs(chk, [[1, 2, 111, 222, 333]], num_sampled=(2,), num_rejected=(3,))
    graph = outputs(chk, [[1, 2, 7, 8, 9]], num_sampled=(2,), num_rejected=(3,))
    assert chk.compare_round(0, rig, eager, FakeRig(**rig.state), graph) is None
    # A difference inside the valid prefix still counts.
    graph = outputs(chk, [[1, 4, 7, 8, 9]], num_sampled=(2,), num_rejected=(3,))
    got = chk.compare_round(0, rig, eager, FakeRig(**rig.state), graph)
    assert got["field"] == "sampled" and got["first_differing_flat_indices"] == [1]


def test_num_sampled_num_rejected_and_state_each_trip_the_comparison(chk):
    base = outputs(chk, [[1, 2, 3, 4, 5]])
    rig = FakeRig(total_len=[10], all_token_ids=[[1, 2, 3]])

    sampled_changed = outputs(chk, [[1, 2, 3, 4, 5]], num_sampled=(3,))
    assert (
        chk.compare_round(0, rig, base, FakeRig(**rig.state), sampled_changed)["field"]
        == "num_sampled"
    )

    rejected_changed = outputs(chk, [[1, 2, 3, 4, 5]], num_rejected=(2,))
    assert (
        chk.compare_round(0, rig, base, FakeRig(**rig.state), rejected_changed)["field"]
        == "num_rejected"
    )

    other = FakeRig(total_len=[10], all_token_ids=[[1, 2, 4]])
    got = chk.compare_round(0, rig, base, other, base)
    assert got["field"] == "all_token_ids"
    assert got["first_differing_flat_indices"] == [2]


def test_make_round_inputs_shapes_and_draft_membership(chk):
    args = argparse.Namespace(num_reqs=2, vocab_size=1000, logit_scale=2.0)
    generator = torch.Generator()
    generator.manual_seed(5)
    logits, drafts = chk.make_round_inputs(
        args, torch.device("cpu"), generator, np.random.default_rng(5)
    )
    assert logits.shape == (2 * PER_REQ, 1000) and logits.dtype == torch.float16
    assert drafts.shape == (2, NUM_SPEC) and drafts.dtype == torch.int64
    top5 = torch.topk(logits.float(), 5, dim=-1).indices.view(2, PER_REQ, 5)
    for r in range(2):
        for i in range(NUM_SPEC):
            # Draft i is verified by logits row i, so it comes from that row.
            assert int(drafts[r, i]) in top5[r, i].tolist()


def test_drafts_mix_the_argmax_and_other_top_tokens(chk):
    args = argparse.Namespace(num_reqs=1, vocab_size=500, logit_scale=2.0)
    generator = torch.Generator()
    generator.manual_seed(1)
    host_rng = np.random.default_rng(1)
    picks = []
    for _ in range(40):
        logits, drafts = chk.make_round_inputs(
            args, torch.device("cpu"), generator, host_rng
        )
        argmax = logits.float().argmax(dim=-1)[:NUM_SPEC]
        picks.extend((drafts[0] == argmax).tolist())
    # Accepting and rejecting drafts both have to occur for the check to mean
    # anything.
    assert 0.3 < sum(picks) / len(picks) < 0.9


def test_median_ignores_empty_input(chk):
    assert chk._median([3.0, 1.0, 2.0]) == 2.0
    assert chk._median([]) != chk._median([])  # nan


def test_arguments_are_validated(chk, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check", "--rounds", "5"])
    with pytest.raises(SystemExit) as exc:
        chk._parse_args()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, "argv", ["check", "--num-reqs", "99"])
    with pytest.raises(SystemExit):
        chk._parse_args()
    monkeypatch.setattr(sys, "argv", ["check"])
    args = chk._parse_args()
    assert (args.rounds, args.num_reqs, args.temperature) == (40, 1, 1.0)
    assert (args.top_k, args.top_p, args.min_p) == (20, 0.95, 0.0)
    assert args.model_state == "both" and args.eager_branchfree == 2


def test_main_exits_2_without_a_cuda_device(chk, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert chk.main() == 2
    assert "needs a CUDA device" in capsys.readouterr().out


def test_first_state_difference_reports_the_field_and_indices(chk):
    before = {
        "output_bin_counts": torch.zeros(2, 4, dtype=torch.int32),
        "total_len": torch.tensor([5, 6]),
    }
    after = {k: v.clone() for k, v in before.items()}
    assert chk.first_state_difference(before, after) is None

    # The residue the GPU run found: one bin count, flat index 5 of a 2 x 4 row set.
    after["output_bin_counts"][1, 1] = 1
    assert chk.first_state_difference(before, after) == {
        "field": "output_bin_counts",
        "first_differing_flat_indices": [5],
        "before_capture": [0],
        "after": [1],
    }
    # Fields are checked in order; the first differing one is reported.
    after["total_len"][0] = 9
    assert chk.first_state_difference(before, after)["field"] == "output_bin_counts"
    del before["output_bin_counts"], after["output_bin_counts"]
    assert chk.first_state_difference(before, after)["field"] == "total_len"


def test_pool_budget_allows_ten_percent_over_the_accepted_pool(chk):
    assert chk.POOL_ACCEPTED_MIB == 58.0
    assert chk.pool_within_budget(58.0, num_reqs=1)
    assert chk.pool_within_budget(63.79, num_reqs=1)
    assert not chk.pool_within_budget(63.81, num_reqs=1)
    # The budget was measured for the (1, 5) key only.
    assert chk.pool_within_budget(500.0, num_reqs=2)


def test_exit_status_separates_capture_residue_from_round_mismatches(chk):
    ok = {"capture_state_residue": None, "pass": True}
    mismatch = {"capture_state_residue": None, "pass": False}
    residue = {"capture_state_residue": {"field": "output_bin_counts"}, "pass": False}
    assert chk.EXIT_STATE_RESIDUE == 3
    assert chk.compute_exit_code([ok, ok]) == 0
    assert chk.compute_exit_code([ok, mismatch]) == 1
    assert chk.compute_exit_code([mismatch, residue]) == 3
    assert chk.compute_exit_code([residue, ok]) == 3


@pytest.mark.parametrize("tail", [False, True])
def test_the_rig_model_state_matches_what_the_manager_restores(chk, tail):
    """The stub must expose the buffer name the manager's restore list reads."""
    from vllm.v1.worker.gpu.sample import cudagraph as cg

    accepted = torch.ones(4, dtype=torch.int32)
    stub = chk._StubModelState(accepted, tail)
    assert cg.model_state_postprocess_is_capture_safe(stub) is (not tail)
    regions = cg.model_state_persistent_regions(stub, 2)
    assert [(t.data_ptr(), r) for t, r in regions] == [
        (accepted.data_ptr(), slice(0, 2))
    ]
