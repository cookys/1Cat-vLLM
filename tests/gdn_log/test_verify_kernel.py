# SPDX-License-Identifier: Apache-2.0
"""G-LOG verify kernel vs the classic 1+K snapshot kernel (CPU, interpreter)."""

import pytest
import torch

from tests.gdn_log.glog_fixtures import (
    LAYOUT,
    WIDTH,
    Pages,
    make_round_inputs,
    run_verify,
)

N = 2


def _chain(accepted_per_round: list[list[int]]):
    """Run the same rounds through both kernels and compare after each."""
    classic = Pages(N * WIDTH)
    glog = Pages(N)
    torch.manual_seed(0)
    h0 = torch.randn(N, *classic.ssm.shape[1:]) * 0.05
    glog.ssm[:] = h0
    c_idx = torch.arange(N * WIDTH, dtype=torch.int32).reshape(N, WIDTH)
    classic.ssm[c_idx[:, 0].long()] = h0
    g_idx = torch.full((N, WIDTH), -1, dtype=torch.int32)
    g_idx[:, 0] = torch.arange(N, dtype=torch.int32)
    prev = torch.ones(N, dtype=torch.int32)  # fresh pages: nothing pending
    for rnd, accepted in enumerate(accepted_per_round):
        inputs = make_round_inputs(N, seed=100 + rnd)
        out_c = run_verify(classic.ssm, c_idx, prev, inputs, gdn_log=None)
        e_before = [glog.epoch(p) for p in range(N)]
        out_g = run_verify(glog.ssm, g_idx, prev, inputs, gdn_log=LAYOUT)
        assert torch.equal(out_c, out_g), f"round {rnd}: attention output differs"
        for p in range(N):
            a = accepted[p]
            want = classic.ssm[c_idx[p, a - 1].long()]
            got = glog.replay(p, e_before[p] & 1, a - 1)
            assert torch.equal(want, got), f"round {rnd} req {p} accepted {a}"
            glog.set_epoch(p, e_before[p] + 1)  # the runner's post-step flip
        prev = torch.tensor(accepted, dtype=torch.int32)


@pytest.mark.parametrize("accepted", range(1, 9))
def test_accepted_count_roundtrip(accepted: int) -> None:
    # Round 0 produces logs; round 1 must replay `accepted - 1` of them as its
    # initial state and still match the classic kernel bit for bit.
    _chain([[accepted, 1 + (accepted % 8)], [1, 1]])


def test_mixed_accepted_counts_across_requests() -> None:
    _chain([[3, 8], [8, 2], [5, 1], [1, 7]])
