# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only native quota/LRU and diagnostic regressions; serving venv required."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_offload.cpu.test_host_parking import CTX, key, real_layout
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.cpu.parking_manager import ParkingManager
from vllm.v1.kv_offload.cpu.parking_quota import proportional_group_slots
from vllm.v1.kv_offload.cpu.parking_spec import HostParkingSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from host_parking_capacity import analyze  # noqa: E402


def test_real_32k_geometry_sparse_state_and_sw_quota():
    config, caches = real_layout(mamba_state_slots_reference_tokens=32768)
    report = analyze(HostParkingSpec(config, caches))
    assert [g["allocated_slots"] for g in report["groups"]] == [36] * 6 + [91] * 3
    assert [g["producer_history_slots"] for g in report["groups"]] == (
        [20] * 6 + [80] * 2 + [160]
    )
    assert [g["deficit_slots"] for g in report["groups"]] == [0] * 8 + [69]
    assert report["logical_capacity_fits_budget"]
    assert not report["current_group_quotas_fit"]
    assert report["producer_history_bytes"] == 14_344_519_680  # 13.359375 GiB
    assert report["final_restore_bytes"] == 8_021_606_400  # 7.470703125 GiB


def test_native_gdn_handoffs_are_two_terminal_states_not_eight():
    _, caches = real_layout()
    pool = BlockPool(100, True, 1024)
    spec = caches.kv_cache_groups[0].kv_cache_spec
    manager = MambaManager(spec, pool, enable_caching=True, kv_cache_group_id=0)
    req = NS(
        request_id="cpu",
        num_tokens=32768,
        shared_prefix_boundary=0,
        block_hashes=[i.to_bytes(32, "big") for i in range(32)],
    )
    boundaries = KVCacheCoordinator.get_replay_boundaries(
        NS(eagle_group_ids=set()),
        req,
        4096,
    )
    offered = []
    for start in range(0, 32768, 4096):
        end = start + 4096
        manager.new_step_starts()
        manager.remove_skipped_blocks(req.request_id, start)
        manager.allocate_new_blocks(req.request_id, end, end)
        manager.cache_blocks(
            req,
            end,
            alignment_tokens=4096,
            retention_interval=0,
            replay_boundaries=boundaries,
        )
        offered.extend(manager.take_pending_boundary_state_offloads())
    assert [entry[3] for entry in offered] == [28672, 32768]


def native_cycle(capacity, reqs=10):
    """Conditional sequential-return counterexample, not a c10 GPU replay."""
    m = CPUOffloadingManager(capacity)

    def all_keys(r):
        return [key(8, r * 100 + i) for i in range(32) if i % 4 >= 2]

    def store(r):
        output = m.prepare_store(all_keys(r), CTX)
        assert output is not None
        m.complete_store(output.keys_to_store, CTX)

    for r in range(reqs):
        store(r)
    hits = []
    for r in range(reqs):
        # Identical resend uses boundary 28672 -> draft pages 26,27.
        hits.append(all(m.lookup(key(8, r * 100 + i), CTX) is True for i in (26, 27)))
        store(r)  # Miss causes full producer history to be stored again.
    return hits


def test_native_sw_lru_cyclic_miss_and_capacity_counterfactual():
    assert native_cycle(91) == [False] * 10
    assert native_cycle(160) == [True] * 10


def test_diagnostics_preserve_lookup_order_and_distinguish_pending(monkeypatch):
    records = []
    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.parking_manager.logger.info",
        lambda fmt, *args: records.append(fmt % args),
    )
    plain = ParkingManager(GroupedCPUOffloadingManager({0: CPUOffloadingManager(2)}))
    diag = ParkingManager(
        GroupedCPUOffloadingManager({0: CPUOffloadingManager(2)}),
        {0: 1024},
        diagnostics=True,
    )
    for manager in (plain, diag):
        assert manager.lookup(key(0, 0), CTX) is False
        store = manager.prepare_store([key(0, 0), key(0, 1)], CTX)
        assert manager.lookup(key(0, 0), CTX) is None
        manager.complete_store(store.keys_to_store, CTX)
        assert manager.lookup(key(0, 0), CTX) is True
        manager.touch([key(0, 0)], CTX)
        store = manager.prepare_store([key(0, 2)], CTX)
        assert store.evicted_keys == [key(0, 1)]
        manager.complete_store(store.keys_to_store, CTX)
    assert list(plain.inner.managers[0]._policy.blocks) == list(
        diag.inner.managers[0]._policy.blocks
    )
    diag.sample_stats()
    sample = json.loads(records[-1].split("HOST_PARKING step_stats ")[1])
    assert sample["group_capacity_slots"] == {"0": 2}
    assert sample["group_slots"] == {"0": 2}
    assert sample["lookup_misses"] == [
        dict(
            group=0,
            reason="absent",
            count=1,
            capacity_slots=2,
            used_slots_at_first_lookup=0,
        ),
        dict(
            group=0,
            reason="pending",
            count=1,
            capacity_slots=2,
            used_slots_at_first_lookup=2,
        ),
    ]
    assert sample["reported_evicted_slots_since_sample"] == {"0": 1}
    diag.sample_stats()
    sample = json.loads(records[-1].split("HOST_PARKING step_stats ")[1])
    assert (
        sample["lookup_misses"] == []
        and sample["reported_evicted_slots_since_sample"] == {}
    )
    plain.stats = MagicMock(side_effect=AssertionError("scan when disabled"))
    plain.sample_stats()
    plain.stats.assert_not_called()


def test_diagnostics_do_not_hide_partial_failed_reservation(monkeypatch):
    records = []
    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.parking_manager.logger.info",
        lambda fmt, *args: records.append(fmt % args),
    )
    m = ParkingManager(
        GroupedCPUOffloadingManager({g: CPUOffloadingManager(1) for g in (0, 1)}),
        diagnostics=True,
    )
    keys = [key(g, 0) for g in (0, 1)]
    m.prepare_store(keys, CTX)
    m.complete_store(keys, CTX)
    m.prepare_load([key(1, 0)], CTX)
    assert m.prepare_store([key(g, 1) for g in (0, 1)], CTX) is None
    m.sample_stats()
    data = json.loads(records[-1].split("HOST_PARKING step_stats ")[1])
    assert data["failed_reservations_since_sample"] == 1
    # Native API does not return already-performed evictions on rollback.
    assert data["reported_evicted_slots_since_sample"] == {}
    assert data["group_slots"] == {"0": 0, "1": 1}
    m.complete_load([key(1, 0)], CTX)


def test_spec_diagnostic_is_opt_in():
    for enabled in (False, True):
        config, caches = real_layout(parking_diagnostics=enabled)
        manager = HostParkingSpec(config, caches).get_manager()
        assert manager.diagnostics is enabled


RATIOS = {str(g): 2 if g < 6 else 8 if g < 8 else 16 for g in range(9)}


def test_balanced_native_spec_fits_ten_and_defaults_unchanged():
    c, k = real_layout(mamba_state_slots_reference_tokens=32768)
    before = HostParkingSpec(c, k)
    assert list(before.cpu_group_num_blocks.values()) == [36] * 6 + [91] * 3
    c.kv_transfer_config.kv_connector_extra_config["parking_group_slot_ratios"] = RATIOS
    after = HostParkingSpec(c, k)
    assert list(after.cpu_group_num_blocks.values()) == [24] * 6 + [96, 95, 192]
    assert after.actual_pool_bytes == 17_175_674_880 < 16 << 30
    assert after.key_namespace == before.key_namespace  # KV format/scales unchanged.
    report = analyze(after)
    assert report["current_group_quotas_fit"]
    manager = after.get_manager()

    def pages(g):
        return (
            (6, 7)
            if g < 6
            else range(8)
            if g < 8
            else [i for i in range(32) if i % 4 >= 2]
        )

    for req in range(10):
        keys = [key(g, req * 100 + p) for g in range(9) for p in pages(g)]
        job = manager.prepare_store(keys, CTX)
        assert job is not None and not job.evicted_keys
        manager.complete_store(job.keys_to_store, CTX)
    assert list(manager.stats()["group_slots"].values()) == [20] * 6 + [80, 80, 160]
    for req in range(10):
        keys = [
            key(g, req * 100 + p)
            for g in range(9)
            for p in ((6,) if g < 6 else range(7) if g < 8 else (26, 27))
        ]
        assert all(manager.lookup(k, CTX) is True for k in keys)
        manager.prepare_load(keys, CTX)
        manager.complete_load(keys, CTX)
    manager.reset_cache()
    assert manager.stats()["in_use_slots"] == manager.stats()["read_write_refs"] == 0


@pytest.mark.parametrize(
    "ratios",
    [
        {},
        {"0": 1},
        {**RATIOS, "9": 1},
        {**RATIOS, "0": 0},
        {**RATIOS, "0": -1},
        {**RATIOS, "0": 2.0},
        {**RATIOS, "0": True},
        {**RATIOS, "0": "2"},
        list(RATIOS.values()),
    ],
)
def test_bad_ratio_configuration_fails_before_pool_allocation(ratios):
    c, k = real_layout(parking_group_slot_ratios=ratios)
    with pytest.raises(ValueError, match="parking_group_slot_ratios"):
        HostParkingSpec(c, k)


def test_ratio_rounding_is_deterministic_and_bounded():
    sizes = {0: 9, 1: 9, 2: 5}
    ratios = {"0": 2, "1": 8, "2": 16}
    for budget in range(200, 500):
        a = proportional_group_slots(sizes, ratios, budget)
        b = proportional_group_slots(
            dict(reversed(list(sizes.items()))), ratios, budget
        )
        assert a == b and sum(sizes[g] * a[g] for g in sizes) <= budget
        assert min(a.values()) > 0
    with pytest.raises(ValueError, match="at least one"):
        proportional_group_slots(sizes, ratios, 1)


def test_worker_tensor_slot_counts_follow_each_group_not_fallback(monkeypatch):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )
    from vllm.v1.kv_offload.cpu import gpu_worker as worker

    c, k = real_layout(parking_group_slot_ratios=RATIOS)
    spec = HostParkingSpec(c, k)
    # Meta storage preserves full native byte geometry without any allocation.
    tensors = [
        CanonicalKVCacheTensor(torch.empty((2, p), dtype=torch.int8, device="meta"), p)
        for p in spec.cpu_group_page_sizes.values()
    ]
    canonical = CanonicalKVCaches(
        tensors,
        [[CanonicalKVCacheRef(g, p)] for g, p in spec.cpu_group_page_sizes.items()],
    )
    allocations = []

    def allocate(slots, size):
        allocations.append((slots, size))
        return torch.empty((slots, size), dtype=torch.int8, device="meta")

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    # Only the CUDA directions are mocked; actual partition/factory code runs.
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: NS(**kw)
    )
    handlers = worker.CpuGpuOffloadingHandlers(
        canonical,
        1,
        spec.num_blocks,
        group_page_sizes=spec.cpu_group_page_sizes,
        group_num_blocks=spec.cpu_group_num_blocks,
        cpu_tensor_factory=allocate,
    )
    assert allocations == [
        (spec.cpu_group_num_blocks[g], p) for g, p in spec.cpu_group_page_sizes.items()
    ]
    assert 4 * sum(n * p for n, p in allocations) == spec.actual_pool_bytes
    assert handlers.cpu_to_gpu_handler.cpu_tensors is not None
    assert not torch.cuda.is_initialized()
