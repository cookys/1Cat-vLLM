# SPDX-License-Identifier: Apache-2.0
"""G-LOG request lifecycle through the real MRV2 align kernels (CPU, interpreter).

Every scenario drives a *classic* world (1+7 snapshot pages per request, the
stage-FIX behaviour) and a *G-LOG* world (one page per request) through the same
preprocess -> precopy -> verify -> postprocess (-> epoch flip) sequence with
identical inputs and compares attention outputs and every committed state
(boundary pages, accepted state) bit for bit.
"""

import pytest
import torch

from tests.gdn_log.glog_fixtures import (
    HV,
    K,
    LAYOUT,
    V,
    WIDTH,
    Pages,
    make_align_ctx,
    make_round_inputs,
    run_verify,
)
from vllm.v1.worker.gpu.mamba_align import (
    preprocess_mamba_align_fused_kernel,
    run_glog_epoch_flip,
    run_mamba_align_postprocess,
    run_mamba_align_precopy,
)

BS = 4096
NUM_COLS = 6
POOL = 40


class Req:
    """One request: a block-table row over a page pool, in classic or G-LOG mode."""

    def __init__(self, pages: Pages, page_of_col: list[int], glog: bool, state_idx):
        self.pages = pages
        self.glog = glog
        self.bt = torch.tensor([page_of_col], dtype=torch.int32)
        self.ctx = make_align_ctx(pages, self.bt, glog=glog, block_size=BS)
        self.state_idx = torch.tensor([state_idx], dtype=torch.int32)
        self.accepted = torch.ones(1, dtype=torch.int32)
        self.src_col = torch.full((1,), -1, dtype=torch.int32)
        self.bias = torch.zeros(1, dtype=torch.int32)
        self.mapping = torch.zeros(1, dtype=torch.int32)

    def page(self, col: int) -> int:
        return int(self.bt[0, col])

    def seed(self, col: int, h: torch.Tensor, conv: torch.Tensor) -> None:
        self.pages.ssm[self.page(col)] = h
        self.pages.conv[self.page(col)] = conv

    def indices(self, col: int, width: int) -> torch.Tensor:
        idx = torch.full((1, width), -1, dtype=torch.int32)
        if self.glog and width > 1:
            idx[0, 0] = self.page(col)
        else:
            for t in range(width):
                idx[0, t] = self.page(col + t)
        return idx

    def step(self, computed: int, accepted_next: int, qlen: int, seed: int):
        """One engine step; returns (attention output, new_computed)."""
        qsl = torch.tensor([0, qlen], dtype=torch.int32)
        num_computed = torch.tensor([computed], dtype=torch.int32)
        preprocess_mamba_align_fused_kernel[(1,)](
            self.mapping,
            self.state_idx,
            num_computed,
            qsl,
            self.accepted,
            self.src_col,
            self.bias,
            1,
            BLOCK_SIZE=1,
            MAMBA_BLOCK_SIZE=BS,
        )
        run_mamba_align_precopy(
            self.ctx, 1, self.state_idx, self.src_col, self.bias, self.mapping, qsl
        )
        col = int(self.state_idx[0])
        self.start_state = self.pages.ssm[self.page(col)].clone()
        inputs = make_round_inputs(1, seed)
        if qlen == WIDTH:
            sel = self.accepted.clone()
            out = run_verify(
                self.pages.ssm,
                self.indices(col, WIDTH),
                sel,
                inputs,
                gdn_log=LAYOUT if self.glog else None,
            )
        else:
            # Single-token tail: the ordinary decode kernel on the canonical
            # state of column 0 of the window, in both worlds.
            mixed, g, beta, _ = inputs
            inputs1 = (mixed[:1], g[:1], beta[:1], torch.tensor([0, 1], dtype=torch.int32))
            out = run_verify(
                self.pages.ssm,
                torch.tensor([[self.page(col)]], dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
                inputs1,
                gdn_log=None,
            )
        new_computed = computed + (accepted_next if qlen == WIDTH else 1)
        self.accepted[0] = accepted_next if qlen == WIDTH else 1
        run_mamba_align_postprocess(
            self.ctx,
            1,
            self.accepted,
            self.state_idx,
            torch.tensor([new_computed], dtype=torch.int32),
            self.mapping,
        )
        if self.glog:
            self.flip()
        return out, new_computed

    def flip(self) -> None:
        run_glog_epoch_flip(self.ctx, 1, self.state_idx, self.mapping)

    def committed(self, col: int, accepted: int) -> torch.Tensor:
        """State after the last accepted token of the round run at ``col``."""
        if not self.glog:
            return self.pages.ssm[self.page(col + accepted - 1)].clone()
        epoch = self.pages.epoch(self.page(col))  # the round's flip already ran
        return self.pages.replay(self.page(col), (epoch & 1) ^ 1, accepted - 1)


def _worlds(seed_col: int, state_idx: int, pool_seed: int = 0):
    """Two worlds sharing the same canonical state at ``seed_col``.

    Classic columns map to distinct pages (slot t of column c is column c+t);
    G-LOG uses the same table, touching only the window's first column.
    """
    gen = torch.Generator().manual_seed(pool_seed)
    h = torch.randn(HV, V, K, generator=gen) * 0.05
    conv = (torch.randn(10, 2560, generator=gen) * 0.1).to(torch.float16)
    worlds = []
    for glog in (False, True):
        pages = Pages(POOL)
        req = Req(pages, list(range(NUM_COLS + WIDTH)), glog, state_idx)
        req.seed(seed_col, h, conv)
        worlds.append(req)
    return worlds


def _same(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    assert torch.equal(a, b), f"{what} differs (max |d|={(a - b).abs().max():.3e})"


@pytest.mark.parametrize("accepted", range(1, 9))
def test_spec_round_then_single_token_tail(accepted: int) -> None:
    """accepted 1..8, then a one-token step: preprocess must commit the accepted
    state into the canonical slot (in place when the page does not change)."""
    classic, glog = _worlds(seed_col=0, state_idx=0)
    outs = []
    for world in (classic, glog):
        o1, computed = world.step(100, accepted, WIDTH, seed=11)
        # Oracle: the classic snapshot slot a-1 is the true accepted state.
        world.slot = world.pages.ssm[world.page(accepted - 1)].clone()
        o2, _ = world.step(computed, 1, 1, seed=12)
        outs.append((o1, o2, world.pages.ssm[world.page(0)].clone()))
        # The tail must have started from the true accepted state (a == 1
        # needs no commit; a > 1 is exactly the early-return case).
        _same(classic.slot, world.start_state, "tail start state vs oracle")
    _same(outs[0][0], outs[1][0], "verify output")
    _same(outs[0][1], outs[1][1], "tail output")
    _same(outs[0][2], outs[1][2], "canonical state after the tail")
    assert int(glog.accepted[0]) == 1


def test_boundary_hit_exactly_same_page_commits_the_log() -> None:
    """The round ends exactly on a 4096 multiple while still on that page
    (src == dst, bias == a - 1 == 7). The classic kernels' early return was fine
    because a snapshot slot made the copy implicit; with logs the in-place
    materialisation must not be skipped."""
    classic, glog = _worlds(seed_col=1, state_idx=1)
    for world in (classic, glog):
        world.trace = []
        _, c = world.step(8176, 8, WIDTH, seed=21)
        o, c = world.step(c, 8, WIDTH, seed=22)  # P=8184 -> 8192, same page
        assert c == 8192
        world.trace.append(o)
        # Canonical state at the boundary, the thing a prefix hit would read:
        world.trace.append(world.pages.ssm[world.page(1)].clone())
        # Independent oracle: the classic world's slot 7 holds the true state
        # after all 8 tokens (a source slot, untouched by the commit copy).
        world.slot = world.pages.ssm[world.page(1 + 7)].clone()
        assert int(world.accepted[0]) == 1  # logs are dead after the commit
        # The next round crosses into column 2 and must start from that state.
        o3, c = world.step(c, 3, WIDTH, seed=23)
        world.trace.append(o3)
        world.trace.append(world.committed(2, 3))
        world.trace.append(world.pages.conv[world.page(2)].clone().float())
    names = ["boundary out", "boundary page", "next out", "next state", "conv"]
    for i, name in enumerate(names):
        _same(classic.trace[i], glog.trace[i], name)
    # Both worlds must hold the *true* boundary state, not merely agree.
    _same(classic.slot, classic.trace[1], "classic boundary vs oracle")
    _same(classic.slot, glog.trace[1], "glog boundary vs oracle")
    assert int(glog.accepted[0]) == 3


def test_boundary_hit_same_page_without_skew() -> None:
    """Single-token step ending on the boundary: src == dst, token_bias == 0, so
    the early return is correct and the accepted count is reset to 1."""
    classic, glog = _worlds(seed_col=1, state_idx=1)
    for world in (classic, glog):
        o, c = world.step(8191, 1, 1, seed=31)
        assert c == 8192
        world.final = [o, world.pages.ssm[world.page(1)].clone()]
    _same(classic.final[0], glog.final[0], "tail output")
    _same(classic.final[1], glog.final[1], "boundary page")


@pytest.mark.parametrize("accepted", [4, 5, 8])
def test_boundary_inside_round_materialises_into_older_page(accepted: int) -> None:
    """P=8188: tokens 8188..8195 straddle 8192. The boundary state is slot
    `bias` (=3) of the round, committed into the older page (src != dst)."""
    classic, glog = _worlds(seed_col=1, state_idx=1)
    for world in (classic, glog):
        _, c = world.step(8188, accepted, WIDTH, seed=41)
        assert c == 8188 + accepted
        world.trace = [
            world.pages.ssm[world.page(1)].clone(),  # boundary (prefix cache) page
            world.pages.conv[world.page(1)].clone().float(),
        ]
        # The round straddles 8192, so preprocess moved it to column 2. The
        # state after the 4 tokens that precede 8192 is that round's slot 3.
        assert int(world.state_idx[0]) == 2
        world.slot = world.pages.ssm[world.page(2 + 3)].clone()
        # A further spec round on the new page must replay the pending log.
        o, c = world.step(c, 2, WIDTH, seed=42)
        world.trace += [o, world.committed(2, 2)]
    names = ["boundary page", "boundary conv", "next out", "next state"]
    for i, name in enumerate(names):
        _same(classic.trace[i], glog.trace[i], name)
    _same(classic.slot, classic.trace[0], "classic boundary vs oracle")
    _same(classic.slot, glog.trace[0], "glog boundary vs oracle")


@pytest.mark.parametrize("src_epoch_odd", [False, True])
@pytest.mark.parametrize("dst_epoch_odd", [False, True])
def test_page_relocation_across_epochs(src_epoch_odd: bool, dst_epoch_odd: bool) -> None:
    """Round 1 leaves 6 accepted on column 1; round 2 crosses into column 2, a
    different page whose epoch is unrelated (any parity). The relocation copies
    the replayed state; the index buffers are reused like a replayed graph's."""
    classic, glog = _worlds(seed_col=1, state_idx=1)
    glog.pages.set_epoch(glog.page(1), 7 if src_epoch_odd else 8)
    glog.pages.set_epoch(glog.page(2), 11 if dst_epoch_odd else 12)
    for world in (classic, glog):
        _, c = world.step(8180, 6, WIDTH, seed=51)
        assert c == 8186
        o, c = world.step(c, 3, WIDTH, seed=52)  # computed_after 8194 -> column 2
        assert int(world.state_idx[0]) == 2
        world.trace = [o, world.committed(2, 3)]
    _same(classic.trace[0], glog.trace[0], "relocated round output")
    _same(classic.trace[1], glog.trace[1], "relocated accepted state")


def test_prefix_resend_reads_the_canonical_boundary_page() -> None:
    """A second request hits the boundary page left by the first one (shared,
    read-only). It must start from the canonical state, never from logs."""
    classic, glog = _worlds(seed_col=1, state_idx=1)
    for world in (classic, glog):
        _, c = world.step(8188, 8, WIDTH, seed=61)
        hit_page = world.page(1)  # cached boundary page
        cols = list(range(20, 20 + NUM_COLS + WIDTH))
        cols[1] = hit_page
        other = Req(world.pages, cols, world.glog, -2)  # state_idx seeded from num_computed
        before = world.pages.ssm[hit_page].clone()
        o, _ = other.step(8192, 3, WIDTH, seed=62)
        world.trace = [o, other.committed(int(other.state_idx[0]), 3)]
        world.shared = (before, world.pages.ssm[hit_page].clone())
    _same(classic.trace[0], glog.trace[0], "resend output")
    _same(classic.trace[1], glog.trace[1], "resend state")
    for world in (classic, glog):
        _same(world.shared[0], world.shared[1], "shared page must stay read-only")
    _same(classic.shared[1], glog.shared[1], "shared page")


def test_preempt_resume_ignores_stale_logs_on_reused_pages() -> None:
    """Request A is preempted with 5 tokens of log pending; its pages go back to
    the pool and are reused, uncleaned, by the resumed request (fresh prefill
    state, accepted == 1). Stale logs and epoch must not leak into the round."""
    classic, glog = _worlds(seed_col=1, state_idx=1)
    for world in (classic, glog):
        _, _ = world.step(8000, 5, WIDTH, seed=71)  # A leaves a pending log
        # Preempt + resume: the same pages come back; prefill rewrites ssm/conv
        # (canonical) but not the log area; the new request starts at accepted 1.
        gen = torch.Generator().manual_seed(5)
        world.seed(
            1,
            torch.randn(HV, V, K, generator=gen) * 0.05,
            (torch.randn(10, 2560, generator=gen) * 0.1).to(torch.float16),
        )
        world.accepted[0] = 1
        world.state_idx[0] = -2
        o, c = world.step(8000, 4, WIDTH, seed=72)
        world.trace = [o, world.committed(int(world.state_idx[0]), 4)]
    _same(classic.trace[0], glog.trace[0], "resumed round output")
    _same(classic.trace[1], glog.trace[1], "resumed accepted state")


def test_epoch_flip_only_touches_live_rows_and_log_states() -> None:
    pages = Pages(6)
    bt = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    ctx = make_align_ctx(pages, bt, glog=True)
    before = [pages.epoch(p) for p in range(6)]
    state_idx = torch.tensor([1, 2], dtype=torch.int32)  # per persistent slot
    # batch row 0 -> slot 1 (column 2 of row 0 = page 2); row 1 -> slot -1 (padding)
    mapping = torch.tensor([1, -1], dtype=torch.int32)
    run_glog_epoch_flip(ctx, 2, state_idx, mapping)
    after = [pages.epoch(p) for p in range(6)]
    assert [a - b for a, b in zip(after, before)] == [0, 0, 1, 0, 0, 0]
    classic_ctx = make_align_ctx(pages, bt, glog=False)
    run_glog_epoch_flip(classic_ctx, 2, state_idx, mapping)  # no layout: no-op
    assert [pages.epoch(p) for p in range(6)] == after
