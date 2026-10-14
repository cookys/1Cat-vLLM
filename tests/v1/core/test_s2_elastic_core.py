# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2 (A2-7 engine side): EngineCore grow / shrink ordering (F10),
tail holders, fence plumbing and tail sizing from worker probes. CPU-only, fake
workers (collective_rpc is a recorder)."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

N_SPEC, TAIL, BOUNDARY = 32, 48, 16
PAGE = 1179648


def make_core(fail_rpc=None):
    pool = BlockPool(N_SPEC, True, 16)
    cfg = KVCacheConfig(
        num_blocks=N_SPEC,
        kv_cache_tensors=[KVCacheTensor(size=PAGE * N_SPEC, shared_by=["l0"])],
        kv_cache_groups=[],
        elastic_tail_blocks=TAIL,
        elastic_boundary_blocks=BOUNDARY,
    )
    sched = Scheduler.__new__(Scheduler)
    sched.kv_cache_config = cfg
    mgr = SimpleNamespace(req_to_blocks={})
    sched.kv_cache_manager = SimpleNamespace(
        block_pool=pool, coordinator=SimpleNamespace(single_type_managers=(mgr,))
    )
    core = EngineCore.__new__(EngineCore)
    core.scheduler = sched
    core.log = log = []

    def rpc(method, timeout=None, args=(), kwargs=None):
        log.append(("rpc", method, args, pool.num_gpu_blocks))
        if fail_rpc == method:
            raise RuntimeError("worker failed")
        return [{"rank": 0}]

    core.collective_rpc = rpc
    core.req_to_blocks = mgr.req_to_blocks
    return core, pool, log


def test_grow_maps_workers_first_then_grows_the_pool():
    core, pool, log = make_core()
    out = core.grow_kv_cache()
    assert log == [("rpc", "grow_kv_cache", (N_SPEC + TAIL,), N_SPEC)]  # pool still old
    assert pool.num_gpu_blocks == N_SPEC + TAIL
    assert out["pool_blocks"] == N_SPEC + TAIL and out["added_blocks"] == TAIL


def test_grow_worker_failure_leaves_the_pool_untouched():
    core, pool, _ = make_core(fail_rpc="grow_kv_cache")
    with pytest.raises(RuntimeError):
        core.grow_kv_cache()
    assert pool.num_gpu_blocks == N_SPEC


@pytest.mark.parametrize("bad", [N_SPEC - 16, N_SPEC + TAIL + 16, N_SPEC + 5])
def test_grow_rejects_out_of_range_or_unaligned_targets(bad):
    core, pool, log = make_core()
    with pytest.raises(ValueError):
        core.grow_kv_cache(bad)
    assert log == [] and pool.num_gpu_blocks == N_SPEC


def test_partial_grow_is_allowed_on_boundary_steps():
    core, pool, _ = make_core()
    core.grow_kv_cache(N_SPEC + 2 * BOUNDARY)
    assert pool.num_gpu_blocks == N_SPEC + 2 * BOUNDARY


def test_shrink_stops_scheduler_first_then_unmaps_workers():
    core, pool, log = make_core()
    core.grow_kv_cache()
    log.clear()
    out = core.shrink_kv_cache()
    # when the worker RPC ran the pool had already been cut back to N_spec
    assert log == [("rpc", "shrink_kv_cache", (N_SPEC,), N_SPEC)]
    assert out["ok"] and pool.num_gpu_blocks == N_SPEC and out["tail_holder_blocks"] == 0


def test_shrink_with_tail_holders_changes_nothing_and_reports_them():
    core, pool, log = make_core()
    core.grow_kv_cache()
    log.clear()
    held = pool.get_new_blocks(N_SPEC + TAIL - 1)  # every usable block
    tail = [b for b in held if b.block_id >= N_SPEC]
    core.req_to_blocks.update(
        {"low": [b for b in held if b.block_id < N_SPEC], "t1": tail[:10], "t2": tail[10:]}
    )
    out = core.shrink_kv_cache()
    assert out["ok"] is False and out["tail_holder_blocks"] == TAIL
    assert out["tail_holder_reqs"] == 2  # D8: requests, not blocks
    assert log == []  # workers were not asked to unmap
    assert pool.num_gpu_blocks == N_SPEC + TAIL
    pool.free_blocks(held)
    assert core.shrink_kv_cache()["ok"] is True


def test_shrink_below_nspec_is_rejected():
    core, _, _ = make_core()
    with pytest.raises(ValueError):
        core.shrink_kv_cache(N_SPEC - 1)


def test_shrink_worker_failure_after_scheduler_shrink_stays_consistent():
    core, pool, _ = make_core(fail_rpc="shrink_kv_cache")
    core.grow_kv_cache()
    with pytest.raises(RuntimeError):
        core.shrink_kv_cache()
    # scheduler already stopped using the tail: safe (workers just keep it mapped)
    assert pool.num_gpu_blocks == N_SPEC


def test_fence_and_pool_info_plumbing():
    core, pool, _ = make_core()
    core.grow_kv_cache()
    info = core.set_kv_fence(N_SPEC - 1)
    assert info["fence_max_block_id"] == N_SPEC - 1
    assert info["pool_blocks"] == N_SPEC + TAIL and info["spec_blocks"] == N_SPEC
    assert info["tail_blocks_max"] == TAIL and info["tail_holder_blocks"] == 0
    assert info["tail_holder_reqs"] == 0 and info["spec_blocks_lost_to_alignment"] == 0
    assert core.scheduler.get_kv_pool_info() is not None
    assert info["free_blocks"] == N_SPEC - 1  # null block excluded, tail fenced
    got = pool.get_new_blocks(N_SPEC - 1)
    assert all(b.block_id < N_SPEC for b in got)
    assert core.get_kv_pool_info()["free_blocks"] == 0
    assert core.set_kv_fence(None)["fence_max_block_id"] is None


def test_core_without_elastic_tail_refuses_grow():
    core, _, _ = make_core()
    core.scheduler.kv_cache_config.elastic_tail_blocks = 0
    with pytest.raises(RuntimeError, match="no elastic"):
        core.grow_kv_cache()


# ---- tail sizing from the workers' probes ----


def test_make_configs_elastic_uses_probes(monkeypatch):
    core = EngineCore.__new__(EngineCore)
    core.collective_rpc = lambda m, **k: [
        {"draft_weights_bytes": int(1.1675 * (1 << 30)), "granularity": 2 << 20}
    ] * 2
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=8192),
        cache_config=SimpleNamespace(block_size=16),
    )
    cfgs = [
        KVCacheConfig(
            num_blocks=1000,
            kv_cache_tensors=[
                KVCacheTensor(size=PAGE * 1000, shared_by=[f"l{i}"]) for i in range(16)
            ],
            kv_cache_groups=[],
        )
        for _ in range(2)
    ]
    from vllm.v1.engine import core as core_mod

    lines = []
    monkeypatch.setattr(
        core_mod.logger, "info", lambda fmt, *a, **k: lines.append(fmt % a)
    )
    core._make_kv_cache_configs_elastic(cfgs)
    assert cfgs[0].num_blocks == 992 and cfgs[0].elastic_tail_blocks == 64
    assert cfgs[0].elastic_spec_blocks_lost == 8  # D2: 1000 -> 992
    assert any("N_spec=992" in l and "8 blocks lost" in l for l in lines)
    assert cfgs[1].kv_cache_tensors[0].max_size == PAGE * (992 + 64)


# ---- walk the real _initialize_kv_caches branch ----


def _init_kv_caches(monkeypatch, elastic):
    from vllm.v1.engine import core as core_mod

    cfgs = [
        KVCacheConfig(
            num_blocks=1000,
            kv_cache_tensors=[
                KVCacheTensor(size=PAGE * 1000, shared_by=[f"l{i}"]) for i in range(16)
            ],
            kv_cache_groups=[],
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(core_mod, "get_kv_cache_configs", lambda *a: cfgs)
    monkeypatch.setattr(core_mod.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", False, raising=False)
    seen = {}
    log = []
    ex = SimpleNamespace(
        get_kv_cache_specs=lambda: [{"l": object()}] * 2,
        determine_available_memory=lambda: [10**9] * 2,
        initialize_from_config=lambda c: seen.setdefault("init", [x.num_blocks for x in c]),
    )
    core = EngineCore.__new__(EngineCore)
    core.model_executor = ex
    core.collective_rpc = lambda m, **k: (
        log.append(m)
        or [{"draft_weights_bytes": int(1.1675 * (1 << 30)), "granularity": 2 << 20}] * 2
    )
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(
            spec_hot_switch_elastic=elastic, num_gpu_blocks=None, block_size=16
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        compilation_config=SimpleNamespace(compilation_time=0.0, encoder_compilation_time=0.0),
        validate_block_size=lambda: None,
    )
    core.vllm_config = vc  # the real __init__ sets it before this call
    core._initialize_kv_caches(vc)
    return vc, cfgs, seen, log


def test_initialize_kv_caches_elastic_branch(monkeypatch):
    vc, cfgs, seen, log = _init_kv_caches(monkeypatch, elastic=True)
    assert log == ["get_elastic_kv_probe"]
    assert vc.cache_config.num_gpu_blocks == 992  # cache_config follows N_spec
    assert seen["init"] == [992, 992]  # workers get the elastic configs
    assert cfgs[0].kv_cache_tensors[0].max_size == PAGE * (992 + 64)


def test_initialize_kv_caches_flag_off_is_unchanged(monkeypatch):
    vc, cfgs, seen, log = _init_kv_caches(monkeypatch, elastic=False)
    assert log == []
    assert vc.cache_config.num_gpu_blocks == 1000 and seen["init"] == [1000, 1000]
    assert cfgs[0].kv_cache_tensors[0].max_size is None


def test_pool_info_reqs_only_scanned_while_fenced():
    core, pool, _ = make_core()
    core.grow_kv_cache()
    assert core.get_kv_pool_info()["tail_holder_reqs"] is None  # no fence: not scanned
    assert core.set_kv_fence(N_SPEC - 1)["tail_holder_reqs"] == 0


def test_shrink_rejects_non_boundary_target_symmetrically_to_grow():
    core, pool, log = make_core()
    core.grow_kv_cache()
    log.clear()
    with pytest.raises(ValueError, match="multiple of"):
        core.shrink_kv_cache(N_SPEC + 5)
    assert log == [] and pool.num_gpu_blocks == N_SPEC + TAIL
    out = core.shrink_kv_cache(N_SPEC + BOUNDARY)  # on-boundary partial shrink ok
    assert out["ok"] and pool.num_gpu_blocks == N_SPEC + BOUNDARY


def test_alignment_loss_reaches_log_and_pool_info():
    from vllm.v1.core.kv_cache_utils import apply_elastic_kv_tail

    t = KVCacheTensor(size=PAGE * 1000, shared_by=["l0"])
    cfg = KVCacheConfig(num_blocks=1000, kv_cache_tensors=[t], kv_cache_groups=[])
    tail = apply_elastic_kv_tail([cfg], [PAGE * 64 * 2], 2 << 20)
    assert tail > 0 and cfg.num_blocks == 992 and cfg.elastic_spec_blocks_lost == 8
    core, _, _ = make_core()
    core.scheduler.kv_cache_config.elastic_spec_blocks_lost = 8
    assert core.get_kv_pool_info()["spec_blocks_lost_to_alignment"] == 8


def test_core_passes_max_model_len_and_block_size_to_the_alignment_check():
    core = EngineCore.__new__(EngineCore)
    core.collective_rpc = lambda m, **k: [
        {"draft_weights_bytes": int(1.1675 * (1 << 30)), "granularity": 2 << 20}
    ] * 2
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16000),
        cache_config=SimpleNamespace(block_size=16),
    )
    cfgs = [
        KVCacheConfig(
            num_blocks=1000,
            kv_cache_tensors=[
                KVCacheTensor(size=PAGE * 1000, shared_by=[f"l{i}"]) for i in range(16)
            ],
            kv_cache_groups=[],
        )
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="max_model_len 16000"):
        core._make_kv_cache_configs_elastic(cfgs)
