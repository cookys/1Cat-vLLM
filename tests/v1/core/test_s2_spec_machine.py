# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M3: the DFLASH2 <-> nospec state machine, end to end on CPU.

Stub server: the real ``EngineCore`` methods (POST/GET through the real FastAPI
router, ``_process_engine_step`` hook), the real ``Scheduler`` / ``BlockPool``,
and stub workers (``Cluster``) that hold a real ``ElasticKvRegion`` over a
``FakeVmmBackend`` per rank plus a fake drafter-weights mapping. Stub workload:
real requests added to the scheduler and stepped with ``schedule`` /
``update_from_output``. The worker RPC log is the F10 / F11 order oracle.
"""

import queue
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.v1.core import utils
from tests.v1.core.test_spec_hot_switch import complete, sync_scheduler
from vllm.device_allocator.elastic_region import ElasticKvRegion
from vllm.device_allocator.vmm import FakeVmmBackend
from vllm.entrypoints.serve.spec_mode.api_router import router as spec_mode_router
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.kv_cache_interface import KVCacheTensor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

GIB = 1 << 30
MIB = 1 << 20
N_SPEC, TAIL, BOUNDARY = 160, 64, 16
PAGE = 1179648
DRAFT = 1_200_000_000
TAIL_BYTES = TAIL * PAGE
MARGIN = EngineCore._SPEC_GROW_MARGIN_BYTES


class FakeRank:
    def __init__(self, free):
        self.free = free
        self.resident = True
        self.be = FakeVmmBackend()
        self.region = ElasticKvRegion(
            0, (N_SPEC + TAIL) * PAGE, N_SPEC * PAGE, BOUNDARY * PAGE, PAGE,
            backend=self.be,
        )
        self.sleep_partial_fail = False

    def sleep_tags(self, tags):
        assert tags == ("draft_weights",)
        if self.resident:
            self.resident = False
            self.free += DRAFT
        if self.sleep_partial_fail:
            raise RuntimeError("rank sleep failed midway")
        return {"freed_bytes": DRAFT, "free_after": self.free}

    def wake_tags(self, tags):
        assert tags == ("draft_weights",)
        if not self.resident:
            self.resident = True
            self.free -= DRAFT
        return {"mapped_bytes": DRAFT}

    def get_gpu_free_bytes(self):
        return {"free_bytes": self.free, "total_bytes": 32 * GIB}

    def grow_kv_cache(self, target):
        before = self.region.mapped_bytes
        self.region.grow_to(target * PAGE)
        self.free -= self.region.mapped_bytes - before
        return {
            "mapped_bytes": self.region.mapped_bytes,
            "mapped_blocks": self.region.mapped_blocks,
            "free_bytes": self.free,
        }

    def shrink_kv_cache(self, target):
        before = self.region.mapped_bytes
        self.region.shrink_to(target * PAGE)
        self.free += before - self.region.mapped_bytes
        return {
            "mapped_bytes": self.region.mapped_bytes,
            "mapped_blocks": self.region.mapped_blocks,
            "free_bytes": self.free,
        }


class Cluster:
    def __init__(self, free=40 * GIB, ranks=2):
        self.ranks = [FakeRank(free) for _ in range(ranks)]
        self.log: list[tuple] = []
        self.inject: dict[str, list[Exception]] = {}
        self.on_call = {}  # method -> callback run before the ranks see it

    def rpc(self, method, timeout=None, args=(), kwargs=None):
        self.log.append((method, args))
        if cb := self.on_call.get(method):
            cb()
        if self.inject.get(method):
            raise self.inject[method].pop(0)
        return [getattr(r, method)(*args) for r in self.ranks]

    def methods(self):
        return [m for m, _ in self.log]


def make(free=40 * GIB, tail=TAIL):
    s = sync_scheduler(num_blocks=N_SPEC, max_num_seqs=16)
    cfg = s.kv_cache_config
    cfg.kv_cache_tensors = [
        KVCacheTensor(size=PAGE * N_SPEC, shared_by=["l0"], max_size=PAGE * (N_SPEC + tail))
    ]
    cfg.elastic_tail_blocks = tail
    cfg.elastic_boundary_blocks = BOUNDARY
    cl = Cluster(free)
    core = EngineCore.__new__(EngineCore)
    core.scheduler = s
    core.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(spec_hot_switch_elastic=True)
    )
    core.collective_rpc = cl.rpc
    core.output_queue = queue.Queue()
    core.post_step = lambda executed: None

    def step_fn():
        out = s.schedule()
        complete(s, out)
        return {}, True

    core.step_fn = step_fn
    return core, cl, s


def step(core, n=1):
    for _ in range(n):
        EngineCoreProc._process_engine_step(core)  # the real loop body incl. the hook


def add(s, n, rid, tokens=30, max_tokens=64):
    reqs = utils.create_requests(
        n, num_tokens=tokens, req_ids=[f"{rid}{i}" for i in range(n)],
        max_tokens=max_tokens,
    )
    for r in reqs:
        s.add_request(r)
    return reqs


class Client:
    def __init__(self, core):
        self.core = core

    async def set_spec_mode(self, mode):
        return self.core.set_spec_mode(mode)

    async def get_spec_mode(self):
        return self.core.get_spec_mode()


def http(core):
    app = FastAPI()
    app.include_router(spec_mode_router)
    app.state.engine_client = Client(core)
    return TestClient(app)


def post(c, mode):
    r = c.post("/spec_mode", params={"mode": mode})
    return r.status_code, r.json()


# ---- whole loop -----------------------------------------------------------


def test_idle_round_trip_runs_the_f10_f11_order():
    core, cl, s = make()
    c = http(core)
    seen = {}
    cl.on_call["wake_tags"] = lambda: seen.update(spec_enabled=s.spec_enabled)
    code, out = post(c, "nospec")
    assert code == 200 and out["changed"] and out["state"] == "NOSPEC_GROWN"
    # release -> memory precheck -> grow (worker map first, then the pool)
    assert cl.methods() == ["sleep_tags", "get_gpu_free_bytes", "grow_kv_cache"]
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC + TAIL
    g = c.get("/spec_mode").json()
    assert g["state"] == "NOSPEC_GROWN" and g["mode"] == "nospec"
    assert g["drafter_resident"] is False and g["pool_blocks"] == N_SPEC + TAIL
    assert g["spec_blocks"] == N_SPEC and g["tail_holder_reqs"] == 0 and g["error"] is None
    tr = g["last_transition"]
    assert tr["from"] == "DFLASH2" and tr["to"] == "NOSPEC_GROWN"
    assert tr["steps"]["release"]["freed_bytes"] == [DRAFT, DRAFT]
    assert tr["steps"]["grow"]["added_blocks"] == TAIL and tr["duration_s"] >= 0

    cl.log.clear()
    code, out = post(c, "dflash2")
    assert out["changed"] and out["prev"] == "nospec" and out["state"] == "DFLASH2"
    assert cl.methods() == ["shrink_kv_cache", "wake_tags"]  # unmap tail, then weights
    assert seen["spec_enabled"] is False  # F11: flag only after every rank woke
    assert s.spec_enabled is True and s.drafter_resident is True
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC
    assert s.kv_cache_manager.block_pool.max_block_id is None
    assert c.get("/spec_mode").json()["last_transition"]["to"] == "DFLASH2"
    assert c.get("/spec_mode").json()["last_transition"]["steps"]["shrink"][
        "mapped_bytes"
    ] == [N_SPEC * PAGE] * 2  # per rank (D-fix-2)


def test_draining_waits_for_k7_requests_then_advances_from_the_step_loop():
    core, cl, s = make()
    c = http(core)
    add(s, 2, "k7", max_tokens=4)
    step(core)  # both running with K=7
    assert s.count_spec_inflight() == 2
    code, out = post(c, "nospec")
    assert out["changed"] and out["state"] == "NOSPEC_DRAINING"
    assert cl.log == []  # nothing is released while K=7 requests exist
    late = add(s, 1, "k0", max_tokens=4)
    assert late[0].spec_off is True  # new requests already run K=0
    for _ in range(12):
        step(core)
        if s.get_spec_mode_info()["state"] == "NOSPEC_GROWN":
            break
    assert s.get_spec_mode_info()["state"] == "NOSPEC_GROWN"
    assert cl.methods() == ["sleep_tags", "get_gpu_free_bytes", "grow_kv_cache"]


def _load_tail(core, s):
    """Grown engine with K=0 requests holding tail blocks."""
    c = http(core)
    post(c, "nospec")
    pool = s.kv_cache_manager.block_pool
    reqs = add(s, 14, "load", tokens=200, max_tokens=12)
    step(core, 2)  # 10 prefill in step 1; the rest do not fit below N_SPEC
    assert pool.count_blocks_in_use_above(N_SPEC) > 0, "workload must reach the tail"
    return c, reqs


def test_fencing_waits_for_tail_holders_then_shrinks_wakes_and_finishes():
    core, cl, s = make()
    c, _ = _load_tail(core, s)
    cl.log.clear()
    code, out = post(c, "dflash2")
    assert out["changed"] and out["state"] == "DFLASH2_FENCING"
    g = c.get("/spec_mode").json()
    assert g["tail_holder_reqs"] > 0 and g["tail_holder_blocks"] >= g["tail_holder_reqs"]
    assert "shrink_kv_cache" not in cl.methods()  # scheduler refused: no unmap yet
    assert g["mode"] == "nospec" and s.spec_enabled is False
    assert s.kv_cache_manager.block_pool.max_block_id == N_SPEC - 1  # fence up
    for _ in range(60):
        step(core)
        if s.get_spec_mode_info()["state"] == "DFLASH2":
            break
    assert s.get_spec_mode_info()["state"] == "DFLASH2" and s.spec_enabled
    assert cl.methods()[-2:] == ["shrink_kv_cache", "wake_tags"]
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC


# ---- reverse edges / idempotence -------------------------------------------


def test_dflash2_while_draining_only_flips_the_flag():
    core, cl, s = make()
    c = http(core)
    add(s, 1, "k7", max_tokens=50)
    step(core)
    post(c, "nospec")
    code, out = post(c, "dflash2")
    assert out["changed"] and out["state"] == "DFLASH2" and s.spec_enabled
    assert cl.log == []  # weights were never released


def test_nospec_while_fencing_lifts_the_fence_and_returns_to_grown():
    core, cl, s = make()
    c, _ = _load_tail(core, s)
    post(c, "dflash2")
    code, out = post(c, "nospec")
    assert out["changed"] and out["state"] == "NOSPEC_GROWN"
    assert s.kv_cache_manager.block_pool.max_block_id is None
    assert s.spec_enabled is False


def test_same_direction_posts_are_noops_in_every_state():
    core, cl, s = make()
    c = http(core)
    assert post(c, "dflash2")[1]["changed"] is False  # DFLASH2
    post(c, "nospec")
    cl.log.clear()
    out = post(c, "nospec")[1]
    assert out["changed"] is False and out["state"] == "NOSPEC_GROWN" and cl.log == []
    core2, cl2, s2 = make()
    c2, _ = _load_tail(core2, s2)
    post(c2, "dflash2")  # FENCING, waiting for holders
    n = len(cl2.log)
    assert post(c2, "dflash2")[1]["changed"] is False and len(cl2.log) == n


def test_state_machine_off_never_touches_workers():
    core, cl, s = make()
    core.vllm_config.cache_config.spec_hot_switch_elastic = False
    core.collective_rpc = lambda *a, **k: pytest.fail("S1 soft switch called a worker")
    c = http(core)
    assert post(c, "nospec")[1]["changed"] is True
    assert post(c, "dflash2")[1]["changed"] is True


# ---- failure edges ---------------------------------------------------------


def test_sleep_failure_restores_weights_blocks_and_retries_on_next_post():
    core, cl, s = make()
    c = http(core)
    cl.inject["sleep_tags"] = [RuntimeError("cuMemUnmap failed")]
    code, out = post(c, "nospec")
    assert out["state"] == "NOSPEC_DRAINING"  # weights resident again
    assert cl.methods() == ["sleep_tags", "wake_tags"]  # immediate wake of the tag
    g = c.get("/spec_mode").json()
    assert g["drafter_resident"] is True and g["blocked"] and "restored" in g["error"]
    step(core, 3)
    assert cl.methods() == ["sleep_tags", "wake_tags"]  # no per-step retry loop
    code, out = post(c, "nospec")  # explicit retry
    assert out["changed"] is False and out["state"] == "NOSPEC_GROWN"
    assert c.get("/spec_mode").json()["error"] is None


def test_second_sleep_failure_is_an_error_state_gate_closed_and_fail_loud():
    core, cl, s = make()
    c = http(core)
    cl.inject["sleep_tags"] = [RuntimeError("boom")]
    cl.inject["wake_tags"] = [RuntimeError("wake boom")]
    code, out = post(c, "nospec")
    g = c.get("/spec_mode").json()
    assert g["state"] == "error" and g["drafter_resident"] is False
    assert "wake_tags failed too" in g["error"]
    for mode in ("nospec", "dflash2"):
        code, body = post(c, mode)
        assert code == 409 and "error state" in body["error"]


def test_partial_rank_sleep_failure_is_recovered_by_wake_on_every_rank():
    core, cl, s = make()
    cl.ranks[0].sleep_partial_fail = True
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_DRAINING"
    assert all(r.resident for r in cl.ranks)  # rank 0 was half-released: put back


def test_grow_skipped_when_free_memory_leaves_no_whole_chunk_then_retried():
    # M-fix-9 (X2R-26): the target follows free - margin, so the skip is only
    # "not even one boundary chunk fits". After the release a rank has
    # base + DRAFT free; make it one byte short of margin + one chunk.
    base = BOUNDARY * PAGE + MARGIN - DRAFT - 1
    core, cl, s = make(free=base)
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_RELEASED"
    assert "grow_kv_cache" not in cl.methods()
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC
    err = core.get_spec_mode()["error"]
    assert "margin" in err and "grow skipped" in err
    gt = core.get_spec_mode()["last_transition"]["steps"]["grow_target"]
    assert gt["target_blocks"] == N_SPEC  # never below N_spec, never negative
    # exactly one chunk of free memory above the margin: grows one chunk
    cl.ranks[1].free += 1
    cl.ranks[0].free += 1
    out = core.set_spec_mode("nospec")
    assert out["changed"] is False and out["state"] == "NOSPEC_GROWN"
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC + BOUNDARY


def test_grow_target_is_min_of_tail_max_and_free_minus_margin():
    # branch 1: plenty of free memory -> N_spec + N_tail_max
    core, cl, s = make(free=40 * GIB)
    core.set_spec_mode("nospec")
    g = core.get_spec_mode()["last_transition"]["steps"]["grow"]
    assert g["target_blocks"] == g["pool_blocks"] == N_SPEC + TAIL
    assert g["n_tail_max"] == TAIL and g["margin_bytes"] == MARGIN
    assert g["min_rank_free_bytes"] == 40 * GIB + DRAFT
    # branch 2: free is the limit -> floor((free - margin)/page/boundary)*boundary
    chunks = 2
    base = chunks * BOUNDARY * PAGE + MARGIN + PAGE // 2 - DRAFT
    core, cl, s = make(free=base)
    core.set_spec_mode("nospec")
    g = core.get_spec_mode()["last_transition"]["steps"]["grow"]
    assert g["target_blocks"] == N_SPEC + chunks * BOUNDARY < N_SPEC + TAIL
    assert (g["target_blocks"] - N_SPEC) % BOUNDARY == 0
    assert cl.ranks[0].region.mapped_blocks == N_SPEC + chunks * BOUNDARY
    # the grow RPC carries the computed target, not N_spec + tail
    assert ("grow_kv_cache", (N_SPEC + chunks * BOUNDARY,)) in cl.log
    # restore: shrink back to N_spec still works from a partial grow
    assert core.set_spec_mode("dflash2")["state"] in ("DFLASH2", "DFLASH2_WAKING", "DFLASH2_SHRINKING")
    for _ in range(5):
        core.get_spec_mode()
    assert cl.ranks[0].region.mapped_blocks >= N_SPEC


def test_poorest_rank_decides_the_grow_precheck():
    core, cl, s = make(free=40 * GIB)
    cl.ranks[1].free = BOUNDARY * PAGE + MARGIN - DRAFT - 1
    assert core.set_spec_mode("nospec")["state"] == "NOSPEC_RELEASED"


def test_grow_rpc_failure_stays_released_and_retries():
    core, cl, s = make()
    cl.inject["grow_kv_cache"] = [RuntimeError("cuMemMap OOM")]
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_RELEASED"
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC  # F10: pool untouched
    assert "staying released" in core.get_spec_mode()["error"]
    assert core.set_spec_mode("nospec")["state"] == "NOSPEC_GROWN"


def test_no_elastic_tail_releases_but_reports_it_and_can_go_back():
    core, cl, s = make(tail=0)
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_RELEASED"
    assert "no elastic KV tail" in core.get_spec_mode()["error"]
    out = core.set_spec_mode("dflash2")
    assert out["state"] == "DFLASH2" and s.spec_enabled and s.drafter_resident
    assert cl.methods() == ["sleep_tags", "wake_tags"]  # nothing to shrink


def test_shrink_rpc_failure_after_scheduler_shrank_retries_only_the_unmap():
    core, cl, s = make()
    core.set_spec_mode("nospec")
    cl.inject["shrink_kv_cache"] = [RuntimeError("unmap failed")]
    out = core.set_spec_mode("dflash2")
    assert out["state"] == "DFLASH2_FENCING" and not s.spec_enabled
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC  # scheduler stopped first
    assert "wake_tags" not in cl.methods()
    out = core.set_spec_mode("dflash2")  # retry
    assert out["state"] == "DFLASH2" and out["changed"] is False
    assert cl.methods().count("shrink_kv_cache") == 2 and cl.methods()[-1] == "wake_tags"
    assert all(r.region.mapped_blocks == N_SPEC for r in cl.ranks)


def test_wake_failure_keeps_spec_off_and_retries():
    core, cl, s = make()
    core.set_spec_mode("nospec")
    cl.inject["wake_tags"] = [RuntimeError("H2D failed")]
    out = core.set_spec_mode("dflash2")
    assert out["state"] == "DFLASH2_FENCING"
    assert s.spec_enabled is False and s.drafter_resident is False  # F11
    assert core.set_spec_mode("dflash2")["state"] == "DFLASH2"
    assert s.spec_enabled is True


def test_twenty_round_trips_do_not_leak_memory_or_pool_blocks():
    core, cl, s = make()
    base = [r.free for r in cl.ranks]
    for _ in range(20):
        assert core.set_spec_mode("nospec")["state"] == "NOSPEC_GROWN"
        assert core.set_spec_mode("dflash2")["state"] == "DFLASH2"
    assert [r.free for r in cl.ranks] == base
    assert all(r.be.mapped_total() == N_SPEC * PAGE for r in cl.ranks)
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC


# ---- M3-fix ---------------------------------------------------------------

LOW_FREE = BOUNDARY * PAGE + MARGIN - DRAFT - 1  # one byte short of one chunk after the release


def _released_unblocked(core, cl):
    """RELEASED (grow skipped) -> dflash2 whose wake fails -> nospec cancels."""
    assert core.set_spec_mode("nospec")["state"] == "NOSPEC_RELEASED"
    cl.inject["wake_tags"] = [RuntimeError("H2D failed")]
    assert core.set_spec_mode("dflash2")["state"] == "DFLASH2_FENCING"
    out = core.set_spec_mode("nospec")  # cancel: lift the fence
    assert out["state"] == "NOSPEC_RELEASED" and core.get_spec_mode()["blocked"] is False
    for r in cl.ranks:
        r.free += TAIL_BYTES  # memory is available now


def test_released_unblocked_advances_to_grown_from_the_step_loop():
    core, cl, s = make(free=LOW_FREE)
    _released_unblocked(core, cl)
    step(core)  # RELEASED is not a resting state
    assert core.get_spec_mode()["state"] == "NOSPEC_GROWN"
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC + TAIL


def test_post_nospec_in_released_retries_the_grow_and_says_so():
    core, cl, s = make(free=LOW_FREE)
    _released_unblocked(core, cl)
    out = core.set_spec_mode("nospec")
    assert out["changed"] is False and out["retried"] is True
    assert out["state"] == "NOSPEC_GROWN"
    assert "retried" not in core.set_spec_mode("nospec")  # GROWN: plain no-op


def test_last_transition_follows_the_last_successful_round_with_attempts():
    core, cl, s = make(free=LOW_FREE)
    core.set_spec_mode("nospec")
    tr = core.get_spec_mode()["last_transition"]
    assert tr["to"] == "NOSPEC_RELEASED" and tr["attempts"] == 1
    for r in cl.ranks:
        r.free += TAIL_BYTES
    core.set_spec_mode("nospec")
    tr = core.get_spec_mode()["last_transition"]
    assert core.get_spec_mode()["state"] == "NOSPEC_GROWN"
    assert tr["to"] == "NOSPEC_GROWN" and tr["attempts"] == 2
    assert tr["duration_s"] >= 0 and set(tr["steps"]) >= {"release", "grow"}


def test_reply_missing_a_key_is_an_error_state_not_an_engine_crash():
    core, cl, s = make()
    cl.ranks[0].sleep_tags = lambda tags: {"free_after": 1}  # no freed_bytes
    cl.ranks[1].sleep_tags = lambda tags: {"free_after": 1}
    core.set_spec_mode("nospec")  # must not raise
    g = core.get_spec_mode()
    assert g["state"] == "error" and "freed_bytes" in g["error"]
    assert s.spec_enabled is False and s.drafter_resident is False
    with pytest.raises(RuntimeError):
        core.set_spec_mode("nospec")


def test_missing_key_in_the_grow_reply_is_an_error_state():
    core, cl, s = make()
    core.grow_kv_cache = lambda target=None: {"pool_blocks": 1}  # no added_blocks / workers
    core.set_spec_mode("nospec")
    assert core.get_spec_mode()["state"] == "error"


def test_rpc_timeout_is_an_error_state_and_every_rpc_carries_the_timeout():
    core, cl, s = make()
    core._SPEC_RPC_TIMEOUT_S = 0.05  # injectable short timeout
    seen = []
    orig = cl.rpc

    def rpc(method, timeout=None, args=(), kwargs=None):
        seen.append((method, timeout))
        if method == "wake_tags":
            raise TimeoutError("worker did not answer")
        return orig(method, timeout, args, kwargs)

    core.collective_rpc = rpc
    core.set_spec_mode("nospec")
    assert [t for _, t in seen] == [0.05] * len(seen) and seen
    core.set_spec_mode("dflash2")
    g = core.get_spec_mode()
    assert g["state"] == "error" and "timed out" in g["error"]
    assert ("wake_tags", 0.05) in seen and s.spec_enabled is False


def test_sleep_timeout_is_an_error_state():
    core, cl, s = make()
    cl.inject["sleep_tags"] = [TimeoutError("hung")]
    core.set_spec_mode("nospec")
    assert core.get_spec_mode()["state"] == "error"
    assert s.drafter_resident is False


def test_get_spec_mode_reports_tail_holder_reqs_and_blocks_side_by_side():
    core, cl, s = make()
    g = core.get_spec_mode()
    assert "tail_holders" not in g
    assert g["tail_holder_reqs"] == 0 and g["tail_holder_blocks"] == 0


def test_worker_region_shrink_failure_is_an_error_state_with_no_wake():
    core, cl, s = make()
    core.set_spec_mode("nospec")
    orig = cl.ranks[1].shrink_kv_cache

    def bad(target):
        out = orig(target)
        out["shrink_failed"] = ["region 1: VmmError('unmap failed')"]
        return out

    cl.ranks[1].shrink_kv_cache = bad
    cl.log.clear()
    core.set_spec_mode("dflash2")
    g = core.get_spec_mode()
    assert g["state"] == "error" and "rank(s) {1:" in g["error"]
    assert "wake_tags" not in cl.methods()
    assert s.spec_enabled is False and s.drafter_resident is False
    with pytest.raises(RuntimeError):
        core.set_spec_mode("nospec")


def test_shrink_region_failure_on_the_pending_retry_path_is_an_error_state():
    core, cl, s = make()
    core.set_spec_mode("nospec")
    cl.inject["shrink_kv_cache"] = [RuntimeError("unmap failed")]  # sets pending
    assert core.set_spec_mode("dflash2")["state"] == "DFLASH2_FENCING"
    orig = cl.ranks[1].shrink_kv_cache

    def bad(target):
        out = orig(target)
        out["shrink_failed"] = ["region 1: VmmError('unmap failed')"]
        return out

    cl.ranks[1].shrink_kv_cache = bad
    cl.log.clear()
    core.set_spec_mode("dflash2")  # retry: only the unmap is repeated
    g = core.get_spec_mode()
    assert g["state"] == "error" and "rank(s) {1:" in g["error"]
    assert g["last_transition"]["steps"]["shrink"]["mapped_bytes"] == [N_SPEC * PAGE] * 2
    assert "wake_tags" not in cl.methods()
    assert s.spec_enabled is False
