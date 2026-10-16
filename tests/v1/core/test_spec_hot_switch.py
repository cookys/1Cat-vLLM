# SPDX-License-Identifier: Apache-2.0
"""plan 075 S1: DFlash2 <-> nospec soft switch, CPU-only.

Covers the scheduler flag + per-request K (fixed for the request's life), K=7 /
K=0 requests coexisting in one allocator (GDN/Mamba align blocks and the draft
sliding-window group), the draft group leaving prefix-cache lookup, and the
``/spec_mode`` HTTP surface.
"""

import random
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.v1.core import utils
from tests.v1.core import test_mixed_prefill_budget as _mixed_prefill_tests
from vllm.entrypoints.serve.spec_mode.api_router import router as spec_mode_router
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Re-export the scheduler factory fixture (config-only OPT model, no weights).
factory = _mixed_prefill_tests.factory

B = 16  # one block size for every group keeps the lookup arithmetic obvious
NSPEC = 7


def make_cfg(num_blocks=600, draft_is_eagle=True):
    full = FullAttentionSpec(
        block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=B,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=NSPEC,
    )
    sw = SlidingWindowSpec(
        block_size=B,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
        sliding_window=2 * B,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["mamba"], mamba),
            KVCacheGroupSpec(["draft"], sw, is_eagle_group=draft_is_eagle),
        ],
    )


@pytest.fixture
def mgr():
    """The production shape: DFlash does not flag its draft group, the scheduler
    asks the manager to treat the sliding-window group as the draft group."""
    return KVCacheManager(
        make_cfg(draft_is_eagle=False),
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=False,
        draft_sliding_window_groups=True,
    )


_salt = iter(range(1, 10_000))


def new_req(rid, num_tokens, spec_off, same_prompt=True):
    """same_prompt=True -> every such request shares one prompt (prefix hits);
    False -> a prompt no other request has."""
    n = 1 if same_prompt else next(_salt) + 1
    r = utils.create_requests(
        n,
        num_tokens=num_tokens,
        req_ids=[rid] + [f"{rid}-pad{i}" for i in range(n - 1)],
        block_size=B,
        same_prompt=same_prompt,
        max_tokens=4096,
    )[-1]
    r.request_id = rid
    r.spec_off = spec_off
    return r


def lookahead(req):
    return 0 if req.spec_off else NSPEC


def prefill(m, req, total, chunk=4 * B):
    """Block-aligned chunked prefill, the way the Mamba-align scheduler splits it."""
    blocks, n_hit = m.get_computed_blocks(req)
    req.num_computed_tokens = n_hit
    first = True
    while req.num_computed_tokens < total:
        new = min(chunk, total - req.num_computed_tokens)
        got = m.allocate_slots(
            req,
            new,
            num_new_computed_tokens=n_hit if first else 0,
            new_computed_blocks=blocks if first else None,
            num_lookahead_tokens=lookahead(req),
        )
        assert got is not None
        first = False
        req.num_computed_tokens += new
    m.new_step_starts()  # Mamba refuses same-step reuse of freshly cached states
    return n_hit


def per_group_len(m, rid):
    return [len(mg.req_to_blocks.get(rid, [])) for mg in m.coordinator.single_type_managers]


# --------------------------------------------------------------------------
# allocator: K=7 and K=0 side by side
# --------------------------------------------------------------------------


def test_draft_group_is_found_and_marked(mgr):
    c = mgr.coordinator
    assert c.draft_group_ids == frozenset({2})
    assert [mg.is_draft_group for mg in c.single_type_managers] == [False, False, True]
    assert isinstance(c.single_type_managers[2], SlidingWindowManager)


def test_draft_group_explicit_eagle_flag_also_marks():
    m = KVCacheManager(
        make_cfg(draft_is_eagle=True),
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=True,
    )
    assert m.coordinator.draft_group_ids == frozenset({2})
    assert m.coordinator.single_type_managers[2].is_draft_group


def test_draft_group_not_marked_without_request():
    m = KVCacheManager(
        make_cfg(draft_is_eagle=False),
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=True,  # fallback flags every group; none may become "draft"
    )
    assert m.coordinator.draft_group_ids == frozenset()
    assert not any(mg.is_draft_group for mg in m.coordinator.single_type_managers)


def test_sole_sliding_window_group_is_never_the_draft_group():
    cfg = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["sw"],
                SlidingWindowSpec(
                    block_size=B,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                    sliding_window=2 * B,
                ),
            )
        ],
    )
    m = KVCacheManager(
        cfg,
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        draft_sliding_window_groups=True,
    )
    assert m.coordinator.draft_group_ids == frozenset()


def test_scheduler_requests_draft_marking_only_for_dflash_family():
    from vllm.v1.core.sched.scheduler import Scheduler

    has = Scheduler._has_draft_sliding_window_group
    assert has(None) is False
    assert has(SimpleNamespace(use_dflash_family=lambda: False)) is False
    assert has(SimpleNamespace(use_dflash_family=lambda: True)) is True


def test_k7_k0_coexist_allocator(mgr):
    c = mgr.coordinator
    free0 = mgr.block_pool.get_num_free_blocks()
    r7 = new_req("r7", 5 * B + 3, spec_off=False, same_prompt=False)
    r0 = new_req("r0", 5 * B + 3, spec_off=True, same_prompt=False)
    # interleave so both are live in the same allocator at the same time
    prefill(mgr, r7, 5 * B + 3)
    prefill(mgr, r0, 5 * B + 3)

    n7, n0 = per_group_len(mgr, "r7"), per_group_len(mgr, "r0")
    need = cdiv(5 * B + 3, B)
    # full attention: identical for both
    assert n7[0] == n0[0] == need
    # Mamba align: K=7 carries NSPEC extra state blocks, K=0 none
    assert n7[1] == need + NSPEC
    assert n0[1] == need
    # draft window group: only K=7 owns pages
    assert n7[2] > 0
    assert n0[2] == 0
    assert "r0" not in c.single_type_managers[2].num_cached_block
    mamba = c.single_type_managers[1]
    assert {"r7", "r0"} <= mamba._allocated_block_reqs

    # a K=0 request reserves nothing for the draft group in its estimate either
    probe = new_req("probe", 3 * B, spec_off=True, same_prompt=False)
    mgr.get_computed_blocks(probe)
    got = c.single_type_managers[2].get_num_blocks_to_allocate(
        "probe", 3 * B, [], 0, 3 * B
    )
    assert got == 0

    # freeing either order returns every block; the other request is untouched
    held_by_7 = [list(mg.req_to_blocks["r7"]) for mg in c.single_type_managers]
    mgr.free(r0)
    assert [list(mg.req_to_blocks["r7"]) for mg in c.single_type_managers] == held_by_7
    assert "r0" not in mamba._allocated_block_reqs
    assert not any("r0" in mg._spec_off_reqs for mg in c.single_type_managers)
    mgr.free(r7)
    mgr.free(probe)
    assert mgr.block_pool.get_num_free_blocks() == free0


def test_mamba_align_allocated_block_reqs_with_mixed_nspec(mgr):
    """R-6: ``_allocated_block_reqs`` / ``last_state_block_idx`` with nspec 0 and 7."""
    c = mgr.coordinator
    mamba: MambaManager = c.single_type_managers[1]
    reqs = {
        "a7": new_req("a7", 2 * B + 5, False, same_prompt=False),
        "b0": new_req("b0", 3 * B + 1, True, same_prompt=False),
        "c7": new_req("c7", 1 * B + 9, False, same_prompt=False),
        "d0": new_req("d0", 2 * B + 2, True, same_prompt=False),
    }
    totals = {k: r.num_tokens for k, r in reqs.items()}
    for k, r in reqs.items():
        prefill(mgr, r, totals[k])
    rng = random.Random(7)

    def check(live):
        for k in live:
            r = reqs[k]
            nspec = 0 if r.spec_off else NSPEC
            held = mamba.req_to_blocks[k]
            after = r.num_computed_tokens
            if r.spec_off:
                # K=0: exactly the token span, no speculative state blocks
                assert len(held) == cdiv(after, B), k
            else:
                # K=7: the allocation is sized for the scheduled 1+NSPEC tokens
                # (an over-estimate once drafts are rejected), plus NSPEC blocks
                assert cdiv(after, B) + nspec <= len(held), k
                assert len(held) <= cdiv(after + NSPEC, B) + nspec, k
            assert k in mamba._allocated_block_reqs
            # running state sits at the (1 + own nspec)-th block from the end
            if k in mamba.last_state_block_idx:
                assert mamba.last_state_block_idx[k] <= len(held) - 1 - nspec
            if r.spec_off:
                # a K=0 request never owns more blocks than its token span
                assert len(held) == cdiv(r.num_computed_tokens, B)

    live = list(reqs)
    for step in range(60):
        for k in list(live):
            r = reqs[k]
            if r.spec_off:
                new, adv = 1, 1
            else:
                new, adv = 1 + NSPEC, rng.randint(1, 1 + NSPEC)
            got = mgr.allocate_slots(r, new, num_lookahead_tokens=lookahead(r))
            assert got is not None
            r.num_computed_tokens += adv
        check(live)
        if step == 30:
            # drop one of each K mid-flight; the survivors keep their layout
            for k in ("b0", "c7"):
                mgr.free(reqs[k])
                live.remove(k)
                assert k not in mamba._allocated_block_reqs
                assert k not in mamba.last_state_block_idx
            check(live)
    for k in live:
        mgr.free(reqs[k])
    assert not mamba._allocated_block_reqs


def test_draft_group_k0_entry_points_are_noops(mgr):
    draft = mgr.coordinator.single_type_managers[2]
    draft.register_request_spec("z", True)
    assert draft.get_num_blocks_to_allocate("z", 10 * B, [], 0, 10 * B) == 0
    assert draft.allocate_new_blocks("z", 10 * B, 10 * B) == []
    assert draft.allocate_new_computed_blocks("z", [], 0, 0) is None
    assert draft.remove_skipped_blocks("z", 10 * B) is None
    assert "z" not in draft.req_to_blocks or draft.req_to_blocks["z"] == []
    assert "z" not in draft.num_cached_block
    # same request id, K=7 -> real work
    draft.register_request_spec("z", False)
    assert draft.get_num_blocks_to_allocate("z", 10 * B, [], 0, 10 * B) > 0
    assert len(draft.allocate_new_blocks("z", 10 * B, 10 * B)) > 0


def test_non_draft_sliding_window_group_ignores_k0():
    m = KVCacheManager(
        make_cfg(draft_is_eagle=False),
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=False,
    )
    sw = m.coordinator.single_type_managers[2]
    assert not sw.is_draft_group
    sw.register_request_spec("z", True)
    assert sw.get_num_blocks_to_allocate("z", 10 * B, [], 0, 10 * B) > 0
    assert len(sw.allocate_new_blocks("z", 10 * B, 10 * B)) > 0


# --------------------------------------------------------------------------
# prefix cache: the draft group leaves the lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize("producer_spec_off", [True, False], ids=["nospec-era", "dflash2-era"])
def test_skip_group_ids_hit_across_modes(mgr, producer_spec_off):
    c = mgr.coordinator
    total = 8 * B + 3
    prod = new_req("prod", total, producer_spec_off)
    prefill(mgr, prod, total)
    probe = new_req("probe", total, False)
    hashes, limit = probe.block_hashes, probe.num_tokens - 1

    # draft group taking part: a prefix cached in nospec mode has no draft
    # pages, so the whole lookup collapses to a miss; a dflash2-era prefix hits
    blocks, hit_all = c.find_longest_cache_hit(hashes, limit, skip_group_ids=frozenset())
    assert hit_all == (0 if producer_spec_off else 8 * B)

    # draft group skipped: main layers and the Mamba state hit the full prefix,
    # the draft group contributes nothing and does not shorten the hit
    blocks, hit = c.find_longest_cache_hit(
        hashes, limit, skip_group_ids=c.draft_group_ids
    )
    assert hit == 8 * B
    assert len(blocks[0]) == len(blocks[1]) == 8
    assert blocks[2] == []


@pytest.mark.parametrize("producer_spec_off", [True, False], ids=["nospec-era", "dflash2-era"])
@pytest.mark.parametrize("consumer_spec_off", [True, False], ids=["to-nospec", "to-dflash2"])
def test_get_computed_blocks_uses_skip_in_both_modes(
    mgr, producer_spec_off, consumer_spec_off
):
    total = 8 * B + 3
    prod = new_req("prod", total, producer_spec_off)
    prefill(mgr, prod, total)

    cons = new_req("cons", total, consumer_spec_off)
    n_hit = prefill(mgr, cons, total)
    # plan 075 S2 C1 fix: cached in either mode, hit in either mode. Only a K>0
    # reader of a nospec-era prefix is capped at N - W (= 99 -> the Mamba state at
    # 4 * B, this helper chunks 4 blocks) so the target rewrites its draft window;
    # test_c1_draft_prefix_hit.py covers that path in detail.
    capped = producer_spec_off and not consumer_spec_off
    assert n_hit == (4 * B if capped else 8 * B)
    c = mgr.coordinator
    # main layers share the producer's blocks
    n_shared = n_hit // B
    assert [b.block_id for b in c.single_type_managers[0].req_to_blocks["cons"][:n_shared]] == [
        b.block_id for b in c.single_type_managers[0].req_to_blocks["prod"][:n_shared]
    ]
    draft = c.single_type_managers[2]
    if consumer_spec_off:
        assert "cons" not in draft.req_to_blocks or not draft.req_to_blocks["cons"]
    else:
        held = [b for b in draft.req_to_blocks["cons"] if not b.is_null]
        assert held, "a K=7 consumer still gets its own draft window pages"
        prod_ids = {b.block_id for b in draft.req_to_blocks.get("prod", [])}
        # fresh pages, never the producer's (the lookup did not hand them over)
        assert not ({b.block_id for b in held} & prod_ids)
        assert all(b.ref_cnt == 1 for b in held)


# --------------------------------------------------------------------------
# scheduler: flag flips, per-request K, lookahead, skip_propose
# --------------------------------------------------------------------------


def complete(s, step, tok=42):
    ids = list(step.num_scheduled_tokens)
    toks = [[] if s.requests[r].is_prefill_chunk else [tok] for r in ids]
    out = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=toks,
    )
    return s.update_from_output(step, out)


def sync_scheduler(**kw):
    s = utils.create_scheduler(
        num_speculative_tokens=NSPEC, max_num_batched_tokens=2048, **kw
    )
    s.num_lookahead_tokens = NSPEC  # the ngram test config reserves none itself
    return s


def spy_lookahead(s, monkeypatch):
    seen: list[tuple[str, int]] = []
    orig = s.kv_cache_manager.allocate_slots

    def spy(request, num_new_tokens, *a, **kw):
        seen.append((request.request_id, kw.get("num_lookahead_tokens", 0)))
        return orig(request, num_new_tokens, *a, **kw)

    monkeypatch.setattr(s.kv_cache_manager, "allocate_slots", spy)
    return seen


def test_flag_flip_fixes_k_per_request_and_old_requests_keep_theirs(monkeypatch):
    s = sync_scheduler()
    seen = spy_lookahead(s, monkeypatch)
    expected: dict[str, bool] = {}
    nid = 0
    # K7 -> 0 -> 7 -> 0 -> 7, one request added per phase, repeated
    for phase_enabled in (True, False, True, False, True, False):
        prev = s.set_spec_enabled(phase_enabled)
        assert prev == (not phase_enabled) or nid == 0
        (r,) = utils.create_requests(
            1, num_tokens=40, req_ids=[f"q{nid}"], max_tokens=64
        )
        s.add_request(r)
        expected[r.request_id] = not phase_enabled
        nid += 1
        # every request seen so far keeps the K it was added with
        assert {k: v.spec_off for k, v in s.requests.items()} == expected
        complete(s, s.schedule())
        complete(s, s.schedule())
        assert {k: v.spec_off for k, v in s.requests.items()} == expected
    # drive a few more mixed steps; lookahead per request follows its own K only
    for _ in range(6):
        complete(s, s.schedule())
    assert {k: v.spec_off for k, v in s.requests.items()} == expected
    assert seen
    for rid, la in seen:
        assert la == (0 if expected[rid] else NSPEC), (rid, la)
    info = s.get_spec_mode_info()
    assert info["spec_enabled"] is False  # last phase was nospec
    assert info["running_spec"] == sum(not v for v in expected.values())
    assert info["running_nospec"] == sum(expected.values())


def test_skip_draft_propose_only_when_every_row_is_k0():
    s = sync_scheduler()
    (a,) = utils.create_requests(1, num_tokens=30, req_ids=["a"], max_tokens=64)
    s.add_request(a)  # K=7
    out = s.schedule()
    assert out.skip_draft_propose is False  # K>0 prefill chunk must hand off a draft
    complete(s, out)
    s.set_spec_enabled(False)
    (b,) = utils.create_requests(1, num_tokens=30, req_ids=["b"], max_tokens=64)
    s.add_request(b)  # K=0
    out = s.schedule()
    assert set(out.num_scheduled_tokens) == {"a", "b"}
    assert out.skip_draft_propose is False  # a is still K=7
    complete(s, out)
    s.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    out = s.schedule()
    assert set(out.num_scheduled_tokens) == {"b"}
    assert out.skip_draft_propose is True
    # nothing scheduled -> never skip
    assert s._all_scheduled_spec_off({}) is False


def test_engine_without_spec_never_skips_and_never_marks_spec_off():
    s = utils.create_scheduler(max_num_batched_tokens=2048)
    (a,) = utils.create_requests(1, num_tokens=30, req_ids=["a"], max_tokens=64)
    s.add_request(a)
    assert a.spec_off is False
    assert s.schedule().skip_draft_propose is False
    with pytest.raises(ValueError):
        s.set_spec_enabled(True)
    info = s.get_spec_mode_info()
    assert info["spec_enabled"] is False
    assert info["running_spec"] == 0


def test_update_draft_token_ids_ignores_k0_request():
    s = sync_scheduler()
    (a,) = utils.create_requests(1, num_tokens=30, req_ids=["a"], max_tokens=64)
    s.add_request(a)
    s.set_spec_enabled(False)
    (b,) = utils.create_requests(1, num_tokens=30, req_ids=["b"], max_tokens=64)
    s.add_request(b)
    for _ in range(2):
        complete(s, s.schedule())
    assert not a.is_prefill_chunk and not b.is_prefill_chunk
    s.update_draft_token_ids(
        DraftTokenIds(req_ids=["a", "b"], draft_token_ids=[[1, 2, 3], [4, 5, 6]])
    )
    assert a.spec_token_ids == [1, 2, 3]
    assert b.spec_token_ids == []


def test_async_scheduler_gives_k0_requests_no_spec_placeholders(factory):
    s = factory(
        async_scheduling=True,
        num_speculative_tokens=NSPEC,
        max_num_batched_tokens=2048,
        max_model_len=2048,
    )
    s.num_lookahead_tokens = NSPEC
    (a,) = utils.create_requests(1, num_tokens=30, req_ids=["a"], max_tokens=64)
    s.add_request(a)
    complete(s, s.schedule())
    s.set_spec_enabled(False)
    (b,) = utils.create_requests(1, num_tokens=30, req_ids=["b"], max_tokens=64)
    s.add_request(b)
    out = s.schedule()  # a decodes, b prefills
    complete(s, out)
    out = s.schedule()
    assert {"a", "b"} <= set(out.num_scheduled_tokens)
    assert len(a.spec_token_ids) == NSPEC
    assert b.spec_token_ids == []


# --------------------------------------------------------------------------
# /spec_mode
# --------------------------------------------------------------------------


class _Client:
    """EngineClient stand-in routing straight into the real EngineCore methods."""

    def __init__(self, scheduler):
        # plan 075 S2 D3: the real EngineCore methods now also read the config
        # (state machine off = the S1 soft switch) and the pool.
        self.core = EngineCore.__new__(EngineCore)
        self.core.scheduler = scheduler
        self.core.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(spec_hot_switch_elastic=False)
        )

    async def set_spec_mode(self, mode):
        return EngineCore.set_spec_mode(self.core, mode)

    async def get_spec_mode(self):
        return EngineCore.get_spec_mode(self.core)


def _app(scheduler):
    app = FastAPI()
    app.include_router(spec_mode_router)
    app.state.engine_client = _Client(scheduler)
    return TestClient(app)


def _lit(s, mode, running_spec, running_nospec, state, inflight=None):
    """Literal /spec_mode GET body (S2 D3: extended fields, state machine off)."""
    return {
        "mode": mode,
        "state": state,
        "running_spec": running_spec,
        "running_nospec": running_nospec,
        "spec_inflight": s.count_spec_inflight() if inflight is None else inflight,
        "drafter_resident": True,
        "pool_blocks": s.kv_cache_manager.block_pool.num_gpu_blocks,
        "spec_blocks": s.kv_cache_config.num_blocks,
        "tail_holder_reqs": 0,
        "tail_holder_blocks": 0,
        "error": None,
        "blocked": False,
        "last_transition": None,
    }


def _snapshot(s):
    return (
        s.spec_enabled,
        s.num_spec_tokens,
        s.num_lookahead_tokens,
        tuple(sorted((k, v.spec_off) for k, v in s.requests.items())),
        tuple(sorted(r.request_id for r in s.running)),
    )


def test_spec_mode_post_get_literals_and_same_state_is_zero_change():
    s = sync_scheduler()
    for rid in ("a", "b"):
        (r,) = utils.create_requests(1, num_tokens=30, req_ids=[rid], max_tokens=64)
        s.add_request(r)
    complete(s, s.schedule())
    c = _app(s)

    assert c.get("/spec_mode").json() == _lit(s, "dflash2", 2, 0, "DFLASH2", 2)
    before = _snapshot(s)
    r = c.post("/spec_mode", params={"mode": "dflash2"})
    assert r.status_code == 200
    assert r.json() == {
        "mode": "dflash2",
        "prev": "dflash2",
        "changed": False,
        "state": "DFLASH2",
    }
    assert _snapshot(s) == before  # same-mode resend: nothing moved

    r = c.post("/spec_mode", params={"mode": "nospec"})
    assert r.status_code == 200
    assert r.json() == {
        "mode": "nospec",
        "prev": "dflash2",
        "changed": True,
        "state": "NOSPEC_DRAINING",  # S1 soft switch: K=7 requests still run
    }
    # flipping the mode does not touch the K of requests already running
    assert c.get("/spec_mode").json() == _lit(
        s, "nospec", 2, 0, "NOSPEC_DRAINING", 2
    )
    again = _snapshot(s)
    r = c.post("/spec_mode", params={"mode": "nospec"})
    assert r.json() == {
        "mode": "nospec",
        "prev": "nospec",
        "changed": False,
        "state": "NOSPEC_DRAINING",
    }
    assert _snapshot(s) == again

    (late,) = utils.create_requests(1, num_tokens=30, req_ids=["c"], max_tokens=64)
    s.add_request(late)
    complete(s, s.schedule())
    assert c.get("/spec_mode").json() == _lit(
        s, "nospec", 2, 1, "NOSPEC_DRAINING", 2
    )
    r = c.post("/spec_mode", params={"mode": "dflash2"})
    assert r.json() == {
        "mode": "dflash2",
        "prev": "nospec",
        "changed": True,
        "state": "DFLASH2",
    }
    # GET is read-only
    snap = _snapshot(s)
    for _ in range(3):
        c.get("/spec_mode")
    assert _snapshot(s) == snap


@pytest.mark.parametrize("bad", [None, "", "dflash", "NOSPEC", "mtp"])
def test_spec_mode_rejects_bad_mode(bad):
    s = sync_scheduler()
    before = _snapshot(s)
    params = {} if bad is None else {"mode": bad}
    r = _app(s).post("/spec_mode", params=params)
    assert r.status_code == 400
    assert _snapshot(s) == before


def test_spec_mode_refused_when_engine_has_no_spec():
    s = utils.create_scheduler(max_num_batched_tokens=2048)
    c = _app(s)
    assert c.get("/spec_mode").json()["mode"] == "nospec"
    assert c.post("/spec_mode", params={"mode": "nospec"}).json()["changed"] is False
    r = c.post("/spec_mode", params={"mode": "dflash2"})
    assert r.status_code == 409
    assert "speculative" in r.json()["error"]


def test_spec_mode_router_mounted_without_dev_mode(monkeypatch):
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_SERVER_DEV_MODE", False, raising=False)
    from vllm.entrypoints.serve import register_vllm_serve_api_routers

    app = FastAPI()
    app.state.args = SimpleNamespace(profiler_config=None)
    register_vllm_serve_api_routers(app)
    paths = app.openapi()["paths"]
    assert set(paths["/spec_mode"]) == {"get", "post"}
    # and the dev-only endpoints really are absent in this mode
    assert "/sleep" not in paths


# D1 hardening: startup log + "draft group holds only drafter layers" assert
# --------------------------------------------------------------------------

TARGET_LAYERS = 40  # drafter layers are numbered from here (start_layer_id)


def _named_cfg(draft_names, full_names=("model.layers.0.self_attn.attn",)):
    cfg = make_cfg(draft_is_eagle=False)
    g = cfg.kv_cache_groups
    g[0].layer_names[:] = list(full_names)
    g[2].layer_names[:] = list(draft_names)
    return cfg


def _bare_scheduler(cfg, draft_ids):
    from vllm.v1.core.sched.scheduler import Scheduler

    s = object.__new__(Scheduler)
    s.kv_cache_config = cfg
    s.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(draft_group_ids=frozenset(draft_ids))
    )
    return s


def _vcfg(pp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        model_config=SimpleNamespace(get_num_layers=lambda pc: TARGET_LAYERS),
    )


DFLASH = SimpleNamespace(use_dflash_family=lambda: True)


def test_draft_group_assert_passes_for_all_drafter_layers(monkeypatch):
    names = [f"model.layers.{TARGET_LAYERS + i}.self_attn.attn" for i in range(3)]
    s = _bare_scheduler(_named_cfg(names), {2})
    s._log_and_check_draft_groups(_vcfg(), DFLASH)  # must not raise


@pytest.mark.parametrize(
    "bad",
    [
        "model.layers.7.self_attn.attn",  # a target layer mixed in
        "model.layers.39.self_attn.attn",  # last target layer
        "mamba",  # unrecognisable name
    ],
)
def test_draft_group_assert_fails_when_target_layer_mixed_in(bad):
    names = [f"model.layers.{TARGET_LAYERS}.self_attn.attn", bad]
    s = _bare_scheduler(_named_cfg(names), {2})
    with pytest.raises(RuntimeError, match="not a drafter layer"):
        s._log_and_check_draft_groups(_vcfg(), DFLASH)


def test_draft_group_startup_log_literal(monkeypatch):
    from vllm.v1.core.sched import scheduler as sched_mod

    seen = []
    monkeypatch.setattr(
        sched_mod.logger, "info_once", lambda fmt, *a: seen.append(fmt % a)
    )
    names = [f"model.layers.{TARGET_LAYERS + i}.self_attn.attn" for i in range(2)]
    s = _bare_scheduler(_named_cfg(names), {2})
    s._log_and_check_draft_groups(_vcfg(), DFLASH)
    assert seen == [
        "plan075 draft_groups=[2] layers="
        "['model.layers.40.self_attn.attn', 'model.layers.41.self_attn.attn']"
    ]
    # no draft group -> no log
    seen.clear()
    _bare_scheduler(_named_cfg(names), set())._log_and_check_draft_groups(
        _vcfg(), DFLASH
    )
    assert seen == []


# D1 wiring: Scheduler.__init__ really hands the flag to KVCacheManager
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [True, False])
def test_scheduler_init_passes_draft_sliding_window_groups_to_kv_manager(
    factory, monkeypatch, flag
):
    from vllm.v1.core.sched import scheduler as sched_mod

    seen = []
    real = sched_mod.KVCacheManager

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sched_mod, "KVCacheManager", spy)
    monkeypatch.setattr(
        sched_mod.Scheduler,
        "_has_draft_sliding_window_group",
        staticmethod(lambda speculative_config: flag),
    )
    factory()  # runs the real Scheduler.__init__
    assert len(seen) == 1
    assert seen[0]["draft_sliding_window_groups"] is flag


# F1: K=0 rows must never write draft KV into someone else's page
# --------------------------------------------------------------------------

DRAFT_BS = 16
MAX_BLOCKS = 8


_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _gather(batch_to_req, src_rows, num_blocks, dst, num_reqs_padded):
    """Run the real _gather_block_tables_kernel on cuda tensors (cpu only if no GPU) (one group)."""
    from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

    src = torch.zeros(4, MAX_BLOCKS, dtype=torch.int32, device=_DEV)
    for r, ids in src_rows.items():
        src[r, : len(ids)] = torch.tensor(ids, dtype=torch.int32, device=_DEV)
    nb = torch.zeros(1, 4, dtype=torch.int32, device=_DEV)
    for r, n in num_blocks.items():
        nb[0, r] = n
    ptr = lambda t: torch.tensor([t.data_ptr()], dtype=torch.uint64, device=_DEV)
    _gather_block_tables_kernel[(1, num_reqs_padded)](
        torch.tensor(batch_to_req, dtype=torch.int32, device=_DEV),
        ptr(src),
        ptr(dst),
        torch.tensor([src.stride(0)], dtype=torch.int64, device=_DEV),
        nb,
        nb.stride(0),
        len(batch_to_req),
        BLOCK_SIZE=4,
    )


def _dflash_slots(table, positions_per_req):
    """Run the real _prepare_dflash_inputs_kernel on cuda tensors (cpu only if no GPU); return (ctx, query) slots on CPU."""
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel,
    )

    nq = 3  # query tokens per request
    n_req = len(positions_per_req)
    qsl = [0]
    for p in positions_per_req:
        qsl.append(qsl[-1] + len(p))
    total = qsl[-1]
    pos = torch.tensor(
        [x for p in positions_per_req for x in p], dtype=torch.int64, device=_DEV
    )
    i32 = lambda n, v=0: torch.full((n,), v, dtype=torch.int32, device=_DEV)
    out_ctx = torch.full((total,), 12345, dtype=torch.int64, device=_DEV)
    out_q = torch.full((64,), 12345, dtype=torch.int64, device=_DEV)
    zeros = lambda n: torch.zeros(n, dtype=torch.int64, device=_DEV)
    max_reqs = 4
    _prepare_dflash_inputs_kernel[(n_req, 1)](
        i32(64), zeros(64), i32(max_reqs + 1), i32(max_reqs), out_q,
        zeros(total), out_ctx,
        i32(64), zeros(64), i32(64),
        torch.zeros(8, device=_DEV), zeros(8),
        pos, torch.tensor(qsl, dtype=torch.int32, device=_DEV),
        torch.arange(n_req, dtype=torch.int32, device=_DEV),
        i32(8, 1), i32(8, 1), i32(n_req, 1), i32(n_req, 0),
        torch.zeros(8, device=_DEV), zeros(8),
        table, table.stride(0),
        0, DRAFT_BS, nq, nq - 1, max_reqs, 64, 4096, 0,
        SAMPLE_FROM_ANCHOR=False, PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=1, CP_INTERLEAVE=1, BLOCK_SIZE=32,
    )
    return out_ctx.cpu(), out_q.cpu()


def test_k0_row_in_mixed_batch_never_writes_draft_kv():
    """Previous step: batch slot 1 was a K=7 request owning draft pages 9,10.
    This step slot 1 is a K=0 request (zero draft blocks) next to two K=7
    requests. Its stale pages 9,10 must not survive into the draft table."""
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    dst = torch.zeros(4, MAX_BLOCKS, dtype=torch.int32, device=_DEV)
    # step N-1: req0 [5,6], req1 [9,10], req2 [7,8]
    _gather(
        [0, 1, 2],
        {0: [5, 6], 1: [9, 10], 2: [7, 8]},
        {0: 2, 1: 2, 2: 2},
        dst,
        4,
    )
    assert dst[1, :2].tolist() == [9, 10]
    # step N: req1 finished; slot 1 now holds new K=0 request 3 with no pages
    _gather(
        [0, 3, 2],
        {0: [5, 6], 2: [7, 8]},
        {0: 2, 3: 0, 2: 2},
        dst,
        4,
    )
    assert dst[1].tolist() == [0] * MAX_BLOCKS  # null-block row
    assert dst[0, :2].tolist() == [5, 6] and dst[2, :2].tolist() == [7, 8]

    ctx, q = _dflash_slots(dst, [[20, 21], [18, 19], [4, 5]])
    # K=0 row (tokens 2..3, query idx 3..5): everything is padding
    assert ctx[2:4].tolist() == [PAD_SLOT_ID] * 2
    assert q[3:6].tolist() == [PAD_SLOT_ID] * 3
    # K=7 rows untouched: pos 20 -> block 1 (=6) offset 4; pos 4 -> block 0 (=7)
    assert ctx[0:2].tolist() == [6 * DRAFT_BS + 4, 6 * DRAFT_BS + 5]
    assert ctx[4:6].tolist() == [7 * DRAFT_BS + 4, 7 * DRAFT_BS + 5]
    assert q[0:3].tolist() == [6 * DRAFT_BS + 6 + k for k in range(3)]
    assert q[6:9].tolist() == [7 * DRAFT_BS + 6 + k for k in range(3)]


def test_all_live_rows_with_blocks_are_copied_unchanged():
    dst = torch.full((4, MAX_BLOCKS), 99, dtype=torch.int32, device=_DEV)
    _gather([1, 0], {0: [3], 1: [4, 5, 6]}, {0: 1, 1: 3}, dst, 3)
    assert dst[0, :3].tolist() == [4, 5, 6] and dst[0, 3] == 99  # tail untouched
    assert dst[1, :1].tolist() == [3]
    assert dst[2].tolist() == [0] * MAX_BLOCKS  # padded row zeroed


def test_scheduler_init_calls_draft_group_check(factory, monkeypatch):
    from vllm.v1.core.sched import scheduler as sched_mod

    calls = []
    monkeypatch.setattr(
        sched_mod.Scheduler,
        "_log_and_check_draft_groups",
        lambda self, vllm_config, speculative_config: calls.append(
            (vllm_config, speculative_config)
        ),
    )
    factory()  # real Scheduler.__init__
    assert len(calls) == 1


def test_update_from_output_routed_experts_branch_runs(factory):
    """Walks the real update_from_output through the routed_experts branch
    (the renamed local lives there; pyflakes cannot see a stale `re.` use)."""
    import numpy as np
    from vllm.v1.outputs import RoutedExpertsLists

    scheduler = factory()
    stored = []
    scheduler.routed_experts_mgr = SimpleNamespace(
        store_batch=lambda data, slots: stored.append((data, slots)),
        routed_experts_by_slot=np.zeros((8, 1, 1), dtype=np.int16),
    )
    req = utils.create_requests(1, num_tokens=8, req_ids=["r"], max_tokens=16)[0]
    scheduler.add_request(req)
    step = scheduler.schedule()
    n = step.num_scheduled_tokens["r"]
    routed = RoutedExpertsLists(
        np.ones((n, 1, 1), dtype=np.int32), np.arange(n, dtype=np.int64)
    )
    out = ModelRunnerOutput(
        req_ids=["r"],
        req_id_to_index={"r": 0},
        sampled_token_ids=[[42]],
        routed_experts=routed,
    )
    scheduler.update_from_output(step, out)
    assert len(stored) == 1 and stored[0][0] is routed.routing_data
