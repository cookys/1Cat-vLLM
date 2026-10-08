# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: real allocator/manager/scheduler, mocked CUDA events only.

Run with the serving venv, CUDA_VISIBLE_DEVICES=, --noconftest.
No GPU allocation and no actual pinned pool allocation in this file.
"""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    SchedulerOffloadConfig,
    TransferJobStatus,
)
from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MambaAttentionBackendEnum,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.cpu.parking import (
    GIB,
    ParkingConfig,
    layout_namespace,
    parking_hash,
)
from vllm.v1.kv_offload.cpu.parking_manager import ParkingManager
from vllm.v1.kv_offload.cpu.parking_spec import HostParkingSpec
from vllm.v1.kv_offload.cpu.parking_transfer import RecoveringHandler
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


def real_layout(**extra):
    """27B config geometry; run the actual mixed-page allocator on CPU."""
    attn = FullAttentionSpec(
        block_size=4096,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    mamba = MambaSpec(
        block_size=4096,
        shapes=((2560, 10), (12, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=7,
        page_size_padded=attn.page_size_bytes,
    )
    specs = {
        f"model.layers.{i}." + ("self_attn.attn" if i % 4 == 3 else "linear_attn"): attn
        if i % 4 == 3
        else mamba
        for i in range(64)
    }
    sw = SlidingWindowSpec(
        block_size=4096,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float16,
        sliding_window=2048,
    )
    specs.update({f"draft.layers.{i}.self_attn.attn": sw for i in range(5)})
    config = NS(
        parallel_config=NS(
            world_size=4,
            tensor_parallel_size=4,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=NS(
            model="Qwen3.8-27B-QUASAR-NVFP4",
            revision="fixture",
            dtype=torch.float16,
            quantization="modelopt",
            max_model_len=262144,
            hf_config=NS(to_dict=lambda: {"layers": 64}),
            get_num_kv_heads=lambda pc: 1,
            get_total_num_hidden_layers=lambda: 64,
        ),
        cache_config=NS(
            user_specified_mamba_block_size=True,
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
            block_size=4096,
            hash_block_size=None,
            cache_dtype="nvfp4",
            calculate_kv_scales=False,
            enable_prefix_caching=True,
            prefix_cache_retention_interval=0,
        ),
        scheduler_config=NS(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=4096,
            mixed_prefill_step_latency_ms=0,
        ),
        speculative_config=NS(use_dflash=lambda: True),
        max_in_flight_tokens=8192,
        kv_transfer_config=NS(
            kv_connector_extra_config={"host_parking": True, **extra},
            kv_load_failure_policy="recompute",
            engine_id="test",
        ),
        kv_events_config=None,
        compilation_config=NS(static_forward_context={}),
    )
    groups = U.get_kv_cache_groups(config, specs)
    caches = U.get_kv_cache_config_from_groups(config, groups, int(16.53 * GIB))
    return config, caches


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"host_parking": 1},
        {"host_parking": True, "cpu_bytes_to_use": 33 * GIB},
        {"host_parking": True, "cpu_bytes_to_use": 26 * GIB},
        {"host_parking": True, "parking_mode": "idle"},
        {"host_parking": True, "offload_prompt_only": False},
        {"host_parking": True, "eviction_policy": "arc"},
        {"host_parking": True, "store_threshold": 2},
        {"host_parking": True, "parking_numa_node": -1},
    ],
)
def test_bad_configuration_rejected_before_allocation(extra):
    with pytest.raises(ValueError):
        ParkingConfig.parse(extra)


def test_default_and_notified_limit():
    assert ParkingConfig.parse({"host_parking": True}) == ParkingConfig(16 * GIB, 0)
    assert (
        ParkingConfig.parse(
            {
                "host_parking": True,
                "cpu_bytes_to_use": 32 * GIB,
                "parking_large_pool_ack": True,
            }
        ).total_bytes
        == 32 * GIB
    )


def test_real_mixed_allocator_budget_and_native_off_unchanged():
    config, caches = real_layout()
    assert len(caches.kv_cache_groups) == 9
    assert (
        sorted(g.kv_cache_spec.block_size for g in caches.kv_cache_groups)
        == [1024] + [4096] * 8
    )
    assert all(
        g.kv_cache_spec.page_size_bytes == 1179648 for g in caches.kv_cache_groups
    )
    spec = HostParkingSpec(config, caches)
    assert spec.partition_by_group and spec.block_size_factor == 1
    assert spec.hash_block_size == 1024
    assert 0 < spec.actual_pool_bytes <= 16 * GIB
    assert spec.actual_pool_bytes == 4 * sum(
        spec.cpu_group_page_sizes[g] * n for g, n in spec.cpu_group_num_blocks.items()
    )
    cfg = SchedulerOffloadConfig.from_spec(spec)
    assert sum(g.requires_exact_boundary_source for g in cfg.kv_group_configs) == 6
    assert len(cfg.kv_group_configs) == 9
    # Selecting the old spec still uses its original slab allocation policy.
    assert not CPUOffloadingSpec(config, caches).partition_by_group
    assert spec._handlers is None  # No torch.zeros(pinned)/CUDA work.


@pytest.mark.parametrize(
    "attr,value",
    [
        ("cache_dtype", "fp8"),
        ("calculate_kv_scales", True),
        ("enable_prefix_caching", False),
    ],
)
def test_unsupported_runtime_rejected(attr, value):
    config, caches = real_layout()
    setattr(config.cache_config, attr, value)
    with pytest.raises(ValueError):
        HostParkingSpec(config, caches)


def test_namespace_generation_scale_and_layout():
    config, caches = real_layout()
    first = layout_namespace(config, caches)
    assert first == layout_namespace(config, caches)
    config.model_config.revision = "other"
    second = layout_namespace(config, caches)
    assert first != second
    assert (
        len(
            {
                parking_hash(first, 0, b"x"),
                parking_hash(first, 1, b"x"),
                parking_hash(second, 0, b"x"),
                parking_hash(first, 0, b"y"),
            }
        )
        == 4
    )
    caches.kv_cache_groups[-1].kv_cache_spec = replace(
        caches.kv_cache_groups[-1].kv_cache_spec, sliding_window=1024
    )
    assert second != layout_namespace(config, caches)


def test_actual_layer_scales_verified():
    config, caches = real_layout()
    spec = HostParkingSpec(config, caches)
    layers = config.compilation_config.static_forward_context
    for group in caches.kv_cache_groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            continue
        for name in group.layer_names:
            layers[name] = NS(
                _k_scale_float=1.0,
                _v_scale_float=1.0,
                _k_scale=torch.tensor(1.0),
                _v_scale=torch.tensor(1.0),
            )
    spec.validate_layers()
    next(iter(layers.values()))._v_scale = torch.tensor(2.0)
    with pytest.raises(ValueError, match="fixed scale violated"):
        spec.validate_layers()


CTX = ReqContext("parking")


def key(g, n):
    return make_offload_key(n.to_bytes(8, "big"), g)


def test_lru_failed_store_invalid_load_and_inflight_pin():
    manager = ParkingManager(
        GroupedCPUOffloadingManager({g: CPUOffloadingManager(2) for g in (0, 1)})
    )
    keys = [key(g, n) for g in (0, 1) for n in (0, 1)]
    output = manager.prepare_store(keys, CTX)
    assert output is not None
    assert all(manager.lookup(k, CTX) is None for k in keys)
    manager.complete_store(keys, CTX)
    manager.prepare_load(keys, CTX)
    manager.invalidate(keys)
    assert all(manager.lookup(k, CTX) is False for k in keys)
    assert manager.prepare_store([key(0, 2)], CTX) is None  # Readers own slots.
    manager.complete_load(keys, CTX)
    assert manager.prepare_store([key(0, 2)], CTX) is not None
    manager.complete_store([key(0, 2)], CTX, success=False)
    assert manager.lookup(key(0, 2), CTX) is False
    with pytest.raises(ValueError, match="invalid"):
        manager.prepare_load([keys[-1]], CTX)
    assert len(manager.invalid_keys) < len(keys)  # Eviction removes tombstone.
    manager.reset_cache()
    assert not manager.invalid_keys


def parking_scheduler():
    from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
        _make_exact_boundary_scheduler,
    )

    scheduler = _make_exact_boundary_scheduler()
    scheduler.host_parking = True
    scheduler.config = scheduler.config._replace(
        num_workers=4, key_namespace=b"parking"
    )
    state = scheduler._req_status["req"]
    state.config = scheduler.config
    scheduler._parking_generation = 0
    scheduler._parking_failure_count = 0
    scheduler._parking_held_finished = set()
    return scheduler, state


def completion(job, ranks, failed=0):
    return KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            {job: ranks}, {job: failed} if failed else {}
        )
    )


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("aborted", [False, True])
def test_store_holds_finished_pages_until_all_four_ranks(failure, aborted):
    from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
        _empty_store_output,
    )

    scheduler, state = parking_scheduler()
    jobs = scheduler._build_boundary_state_store_jobs(
        _empty_store_output(
            num_scheduled_tokens={"req": 16},
            boundary_state_offloads={"req": [(1, 99, 16)]},
        )
    )
    job, transfer = next(iter(jobs.items()))
    assert list(transfer.transfer_spec[0].block_ids) == [99]  # Not stale mirror 21.
    state.req.is_finished.return_value = True
    state.req.status = (
        RequestStatus.FINISHED_ABORTED if aborted else RequestStatus.FINISHED_STOPPED
    )
    assert scheduler.request_finished(state.req) == (True, None)
    early = completion(job, 3, failed=int(failure))
    scheduler.update_connector_output(early)
    scheduler.manager.complete_store.assert_not_called()
    assert not early.finished_sending and "req" in scheduler._req_status
    last = completion(job, 1)
    scheduler.update_connector_output(last)
    assert last.finished_sending == {"req"}
    assert not scheduler._parking_held_finished and not scheduler._jobs
    assert "req" not in scheduler._req_status
    assert scheduler.manager.complete_store.call_args.kwargs == (
        {"success": False} if failure else {}
    )


def test_failed_load_invalidates_only_after_all_ranks_and_hybrid_recomputes():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler, state = parking_scheduler()
    keys = {key(0, 1), key(1, 1)}
    scheduler.manager = MagicMock(spec=ParkingManager)
    scheduler._jobs[7] = TransferJobStatus("req", 4, keys, False)
    state.transfer_jobs.add(7)
    scheduler.update_connector_output(completion(7, 1, failed=1))
    scheduler.manager.invalidate.assert_not_called()
    scheduler.update_connector_output(completion(7, 3))
    scheduler.manager.invalidate.assert_called_once_with(keys)
    core = object.__new__(Scheduler)
    core.connector = NS(connector_scheduler=scheduler)
    core.kv_cache_manager = MagicMock()
    core.kv_cache_manager.get_block_ids.return_value = ([2, 3], [0, 9], [0, 4])
    req = NS(
        request_id="req",
        num_computed_tokens=8192,
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
    )
    affected, tokens, evict = core._update_requests_with_invalid_blocks(
        [req], {9}, {}, False
    )
    assert affected == {"req"} and tokens == 8192 and evict == set()
    assert req.num_computed_tokens == 0
    core.kv_cache_manager.free.assert_not_called()  # Still owned until all TP events.
    core.failed_recving_kv_req_ids = {"req"}
    core.finished_recving_kv_req_ids = {"req"}
    core._update_waiting_for_remote_kv(req)
    core.kv_cache_manager.free.assert_called_once_with(req)
    core.kv_cache_manager.cache_blocks.assert_not_called()


def test_generation_reset_ignores_stale_completion():
    scheduler, state = parking_scheduler()
    state.group_states[0].offload_keys.clear()
    state.update_offload_keys()
    before = state.group_states[0].offload_keys[:]
    scheduler._job_counter = 8
    scheduler._jobs[7] = TransferJobStatus("req", 4, set(before), True)
    state.transfer_jobs.add(7)
    scheduler.reset_cache()
    assert scheduler._current_batch_jobs_to_flush == {7}
    assert not state.transfer_jobs
    state.update_offload_keys()
    assert before != state.group_states[0].offload_keys
    scheduler.update_connector_output(completion(7, 4))
    scheduler.manager.complete_store.assert_not_called()


class FakeHandler:
    def __init__(self):
        self._transfers = deque()
        self._transfer_events = {}
        self.result = []
        self.error = None
        self.refuse = False

    def transfer_async(self, job, spec):
        self._transfers.append(job)
        self._transfer_events[job] = object()
        return not self.refuse

    def get_finished(self):
        if self.error:
            raise self.error
        results, self.result = self.result, []
        return results

    def wait(self, jobs):
        if self.error:
            raise self.error

    def shutdown(self):
        self._transfers.clear()
        self._transfer_events.clear()


@pytest.mark.parametrize("stage", ["submit", "poll", "wait"])
def test_copy_failure_drains_and_returns_failed_jobs_without_engine_failure(
    monkeypatch, stage
):
    drain = MagicMock()
    monkeypatch.setattr(torch.cuda, "synchronize", drain)
    native = FakeHandler()
    handler = RecoveringHandler(native)
    handler.transfer_async(1, None)
    if stage == "submit":
        native.refuse = True
        handler.transfer_async(2, None)
    else:
        native.error = RuntimeError("test copy fault")
        if stage == "poll":
            results = handler.get_finished()
        else:
            handler.wait({1})
        native.error = None
    if stage != "poll":
        results = handler.get_finished()
    assert [(r.job_id, r.success) for r in results] == (
        [(1, False), (2, False)] if stage == "submit" else [(1, False)]
    )
    assert (
        drain.call_count == 1 and not native._transfers and not native._transfer_events
    )
    assert handler.failure_count == 1 and handler.fatal_count == 0
    native.refuse = False
    handler.transfer_async(3, None)
    native.result = [TransferResult(3, True)]
    assert handler.get_finished()[0].success  # Later unrelated request works.


def test_unrecoverable_cuda_context_fails_loudly(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "synchronize", MagicMock(side_effect=RuntimeError("lost rank"))
    )
    native = FakeHandler()
    native.refuse = True
    handler = RecoveringHandler(native)
    with pytest.raises(RuntimeError, match="unrecoverable rank"):
        handler.transfer_async(1, None)
    assert handler.fatal_count == 1
    assert handler.pending == {1} and native._transfers  # No unsafe reclamation.


def test_failure_metadata_aggregates_rank_counts():
    all_ranks = OffloadingWorkerMetadata()
    for rank in range(4):
        one = OffloadingWorkerMetadata()
        if rank == 1:
            one.mark_failed(7)
        else:
            one.mark_completed(7)
        all_ranks = all_ranks.aggregate(one)
    assert all_ranks.completed_jobs == {7: 4} and all_ranks.failed_jobs == {7: 1}


def test_worker_reports_failed_load_destination_and_receiving_completion():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    worker = object.__new__(OffloadingConnectorWorker)
    worker.host_parking = True
    worker.worker = NS(get_finished=lambda: [TransferResult(7, False)])
    worker._load_jobs = {7: "req"}
    worker._load_block_ids = {7: {5, 9}}
    worker._invalid_block_ids = set()
    worker._connector_worker_meta = OffloadingWorkerMetadata()
    assert worker.get_finished(set()) == (set(), {"req"})
    assert worker.get_invalid_blocks() == {5, 9}
    assert worker.get_invalid_blocks() == set()
    meta = worker.build_connector_worker_meta()
    assert meta.failed_jobs == {7: 1} and meta.completed_jobs == {7: 1}


def test_exact_pool_quota_two_direction_lifetime_and_registration(monkeypatch):
    from vllm.v1.kv_offload.cpu.parking_pool import ParkingPool

    register = MagicMock(return_value=NS(value=0))
    unregister = MagicMock(return_value=NS(value=0))
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: NS(cudaHostRegister=register, cudaHostUnregister=unregister),
    )
    pool = ParkingPool(8192)  # Real anonymous CPU mapping, fake CUDA registration.
    a = pool.allocate(3, 1024)
    b = pool.allocate(5, 1024)
    assert b.data_ptr() - a.data_ptr() == 3072
    assert register.call_args.args[1:] == (8192, 0)
    a.fill_(7)
    b.fill_(9)
    assert pool.base.tolist() == [7] * 3072 + [9] * 5120
    with pytest.raises(ValueError, match="quota"):
        pool.allocate(1, 1)
    first = RecoveringHandler(FakeHandler(), pool)
    second = RecoveringHandler(FakeHandler(), pool)
    first.shutdown()
    unregister.assert_not_called()
    assert pool.pinned
    del a, b
    second.shutdown()
    unregister.assert_called_once()
    assert not pool.pinned and pool.mapping.closed
    second.shutdown()  # Idempotent wrapper shutdown.


def test_pool_registration_failure_and_page_guard(monkeypatch):
    from vllm.v1.kv_offload.cpu.parking_pool import ParkingPool

    monkeypatch.setattr(
        torch.cuda, "cudart", lambda: NS(cudaHostRegister=lambda *args: NS(value=2))
    )
    with pytest.raises(ValueError, match="whole OS pages"):
        ParkingPool(4097)
    with pytest.raises(RuntimeError, match="Register failed"):
        ParkingPool(4096)


def test_group_rollback_prunes_failed_snapshot_tombstones():
    m = ParkingManager(
        GroupedCPUOffloadingManager({g: CPUOffloadingManager(1) for g in (0, 1)})
    )
    a, b = key(0, 1), key(1, 1)
    m.prepare_store([a, b], CTX)
    m.complete_store([a, b], CTX)
    m.invalidate([a])
    m.prepare_load([b], CTX)
    assert m.prepare_store([key(0, 2), key(1, 2)], CTX) is None
    assert a not in m.invalid_keys  # Group 0 evicted a before group 1 refused.
    m.complete_load([b], CTX)


def test_abort_h2d_owned_until_all_rank_receive_notice():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler, state = parking_scheduler()
    state.req.is_finished.return_value = True
    state.req.status = RequestStatus.FINISHED_ABORTED
    scheduler._jobs[7] = TransferJobStatus("req", 4, {key(0, 1)}, False)
    state.transfer_jobs.add(7)
    # Core's finish_requests supplies delay_free_blocks for remote waits.
    core = object.__new__(Scheduler)
    core.requests = {"req": state.req}
    core._free_blocks = MagicMock()
    core.connector = NS(update_connector_output=scheduler.update_connector_output)
    core._update_from_kv_xfer_finished(completion(7, 3))
    core._free_blocks.assert_not_called()
    last = completion(7, 1)
    last.finished_recving = {"req"}  # Executor emits only after TP aggregation.
    core._update_from_kv_xfer_finished(last)
    core._free_blocks.assert_called_once_with(state.req)


def test_reset_refused_while_finished_stores_hold_pages():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    connector = object.__new__(OffloadingConnector)
    scheduler, state = parking_scheduler()
    connector.connector_scheduler = scheduler
    scheduler._parking_held_finished.add("req")
    assert connector.reset_cache() is False
    assert scheduler._parking_generation == 0


def test_real_worker_canonicalizes_nvfp4_scales_gdn_and_draft_bytes_on_cpu():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import KVCacheTensor

    config, caches = real_layout()
    old_n = caches.num_blocks
    caches.kv_cache_tensors = [
        KVCacheTensor(size=t.size // old_n * 2, shared_by=t.shared_by)
        for t in caches.kv_cache_tensors
    ]
    caches.num_blocks = 2
    spec = HostParkingSpec(config, caches)
    kv, owners = {}, []
    layer_specs = {
        name: g.kv_cache_spec for g in caches.kv_cache_groups for name in g.layer_names
    }
    for t in caches.kv_cache_tensors:
        raw = torch.zeros((2, t.size // 2), dtype=torch.int8)
        owners.append(raw)
        for name in t.shared_by:
            kv[name] = [raw] if isinstance(layer_specs[name], MambaSpec) else raw
    worker = object.__new__(OffloadingConnectorWorker)
    worker.spec = spec
    canonical = []
    worker._register_handlers = canonical.append
    worker.register_kv_caches(kv)
    result = canonical[0]
    assert len(result.tensors) == 8 and len(result.group_data_refs) == 9
    for g, refs in zip(caches.kv_cache_groups, result.group_data_refs):
        expected = g.kv_cache_spec.real_page_size_bytes
        assert len(refs) == len(g.layer_names)
        assert all(r.page_size_bytes == expected for r in refs)
        assert all(
            result.tensors[r.tensor_idx].page_size_bytes == 1179648 for r in refs
        )
    assert {r.page_size_bytes for refs in result.group_data_refs for r in refs} == {
        837632,
        1048576,
        1179648,
    }


@pytest.mark.parametrize("fail_body", [False, True])
def test_numa_policy_restored_without_actual_syscall(monkeypatch, fail_body):
    import ctypes

    from vllm.v1.kv_offload.cpu.parking import bind_memory_node

    def read(mode, mask, *unused):
        ctypes.cast(mode, ctypes.POINTER(ctypes.c_int))[0] = 0
        ctypes.cast(mask, ctypes.POINTER(ctypes.c_ulong))[0] = 0
        return 0

    calls = []

    def write(mode, mask, maxnode):
        calls.append(
            (
                mode,
                ctypes.cast(mask, ctypes.POINTER(ctypes.c_ulong))[0] if mask else None,
                maxnode,
            )
        )
        return 0

    lib = NS(
        get_mempolicy=MagicMock(side_effect=read),
        set_mempolicy=MagicMock(side_effect=write),
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: lib)
    try:
        with bind_memory_node(1):
            if fail_body:
                raise LookupError("body")
    except LookupError:
        assert fail_body
    assert calls == [(2, 2, 64), (0, None, 64)]


def test_p8_must_remain_off():
    config, caches = real_layout()
    config.scheduler_config.mixed_prefill_step_latency_ms = 10
    with pytest.raises(ValueError, match="P8 disabled"):
        HostParkingSpec(config, caches)
