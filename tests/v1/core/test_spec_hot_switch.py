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
from tests.v1.core.test_mixed_prefill_budget import factory  # noqa: F401 (fixture)
from vllm.entrypoints.serve.spec_mode.api_router import router as spec_mode_router
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
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
    return KVCacheManager(
        make_cfg(),
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=True,
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


def test_draft_group_not_marked_without_explicit_flag():
    m = KVCacheManager(
        make_cfg(draft_is_eagle=False),
        max_model_len=4096,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=True,  # fallback flags every group; none may become "draft"
    )
    assert m.coordinator.draft_group_ids == frozenset()
    assert not any(mg.is_draft_group for mg in m.coordinator.single_type_managers)


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

    # draft group taking part: nospec-era prefixes never hit; dflash2-era only
    # as far as the draft window allows
    blocks, hit_all = c.find_longest_cache_hit(hashes, limit, skip_group_ids=frozenset())
    if producer_spec_off:
        assert hit_all == 0
    else:
        assert 0 < hit_all < 8 * B

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
    assert n_hit == 8 * B  # cached in either mode, hit in either mode
    c = mgr.coordinator
    # main layers share the producer's blocks
    assert [b.block_id for b in c.single_type_managers[0].req_to_blocks["cons"][:8]] == [
        b.block_id for b in c.single_type_managers[0].req_to_blocks["prod"][:8]
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
        self.core = SimpleNamespace(scheduler=scheduler)

    async def set_spec_mode(self, mode):
        return EngineCore.set_spec_mode(self.core, mode)

    async def get_spec_mode(self):
        return EngineCore.get_spec_mode(self.core)


def _app(scheduler):
    app = FastAPI()
    app.include_router(spec_mode_router)
    app.state.engine_client = _Client(scheduler)
    return TestClient(app)


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

    assert c.get("/spec_mode").json() == {
        "mode": "dflash2",
        "running_spec": 2,
        "running_nospec": 0,
    }
    before = _snapshot(s)
    r = c.post("/spec_mode", params={"mode": "dflash2"})
    assert r.status_code == 200
    assert r.json() == {"mode": "dflash2", "prev": "dflash2", "changed": False}
    assert _snapshot(s) == before  # same-mode resend: nothing moved

    r = c.post("/spec_mode", params={"mode": "nospec"})
    assert r.status_code == 200
    assert r.json() == {"mode": "nospec", "prev": "dflash2", "changed": True}
    # flipping the mode does not touch the K of requests already running
    assert c.get("/spec_mode").json() == {
        "mode": "nospec",
        "running_spec": 2,
        "running_nospec": 0,
    }
    again = _snapshot(s)
    r = c.post("/spec_mode", params={"mode": "nospec"})
    assert r.json() == {"mode": "nospec", "prev": "nospec", "changed": False}
    assert _snapshot(s) == again

    (late,) = utils.create_requests(1, num_tokens=30, req_ids=["c"], max_tokens=64)
    s.add_request(late)
    complete(s, s.schedule())
    assert c.get("/spec_mode").json() == {
        "mode": "nospec",
        "running_spec": 2,
        "running_nospec": 1,
    }
    r = c.post("/spec_mode", params={"mode": "dflash2"})
    assert r.json() == {"mode": "dflash2", "prev": "nospec", "changed": True}
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
