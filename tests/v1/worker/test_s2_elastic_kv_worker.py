# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2 (A2-4, A2-7 worker side): elastic KV allocation branch,
runner grow/shrink with rollback, worker RPC methods, and the "elastic KV does
not use the cumem kv_cache pool" rule (F8). CPU-only, FakeVmmBackend."""

from types import SimpleNamespace

import pytest
import torch

from vllm.device_allocator import elastic_region as er
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.device_allocator.vmm import FakeVmmBackend, VmmError
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MIB = 1 << 20
PAGE = 1179648
N_SPEC = 32
TAIL = 48
BOUNDARY_BLOCKS = 16


@pytest.fixture(autouse=True)
def _cpu_tensor(monkeypatch):
    # the region's tensor view needs CUDA; give it a CPU stand-in
    monkeypatch.setattr(
        er.ElasticKvRegion,
        "as_tensor",
        lambda self: torch.zeros(1, dtype=torch.int8),
    )


def elastic_cfg(n_tensors=2, elastic=True):
    tensors = [
        KVCacheTensor(
            size=PAGE * N_SPEC,
            shared_by=[f"l{i}"],
            max_size=PAGE * (N_SPEC + TAIL) if elastic else None,
        )
        for i in range(n_tensors)
    ]
    return KVCacheConfig(
        num_blocks=N_SPEC,
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"l{i}" for i in range(n_tensors)], kv_cache_spec=None
            )
        ],
        elastic_tail_blocks=TAIL if elastic else 0,
        elastic_boundary_blocks=BOUNDARY_BLOCKS if elastic else 0,
    )


# ---- A2-4: allocation branch ----


def test_elastic_tensor_allocates_region_not_torch_zeros(monkeypatch):
    be = FakeVmmBackend()
    regions: list = []
    cfg = elastic_cfg()
    # device.index is None on cpu -> the allocator asks torch for the device
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    raw = attn_utils._allocate_kv_cache(
        cfg, {}, torch.device("cpu"), elastic_regions=regions, vmm_backend=be
    )
    assert set(raw) == {"l0", "l1"} and len(regions) == 2
    for r in regions:
        assert r.n_spec_bytes == PAGE * N_SPEC
        assert r.n_max_bytes == PAGE * (N_SPEC + TAIL)
        assert r.boundary == BOUNDARY_BLOCKS * PAGE and r.block_bytes == PAGE
        assert r.mapped_bytes == PAGE * N_SPEC  # tail not mapped yet
    # F9: the initial region is zeroed by memset, not by torch.zeros
    assert len(be.ops("memset")) == 2


def test_non_elastic_tensor_still_uses_torch_zeros():
    regions: list = []
    cfg = elastic_cfg(n_tensors=1, elastic=False)
    cfg.kv_cache_tensors[0].size = 64
    cfg.num_blocks = 1
    raw = attn_utils._allocate_kv_cache(
        cfg, {}, torch.device("cpu"), elastic_regions=regions
    )
    assert regions == [] and raw["l0"].numel() == 64


def test_elastic_tensor_without_region_owner_is_rejected():
    with pytest.raises(RuntimeError, match="elastic_regions"):
        attn_utils._allocate_kv_cache(elastic_cfg(), {}, torch.device("cpu"))


# ---- A2-7: runner grow / shrink ----


def runner(be=None, n=2):
    be = be or FakeVmmBackend()
    regions = [
        er.ElasticKvRegion(
            0, PAGE * (N_SPEC + TAIL), PAGE * N_SPEC, BOUNDARY_BLOCKS * PAGE, PAGE,
            backend=be,
        )
        for _ in range(n)
    ]
    r = SimpleNamespace(elastic_kv_regions=regions)
    r._elastic_regions = lambda: GPUModelRunner._elastic_regions(r)
    r._elastic_report = lambda: GPUModelRunner._elastic_report(r)
    return r, regions, be


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)


def test_runner_grow_and_shrink_every_region(no_cuda):
    r, regions, be = runner()
    rep = GPUModelRunner.grow_kv_cache(r, N_SPEC + TAIL)
    assert rep["mapped_blocks"] == N_SPEC + TAIL and rep["regions"] == 2
    assert rep["mapped_bytes"] == 2 * PAGE * (N_SPEC + TAIL)
    rep = GPUModelRunner.shrink_kv_cache(r, N_SPEC)
    assert rep["mapped_blocks"] == N_SPEC and rep["max_blocks"] == N_SPEC + TAIL
    assert all(x.mapped_bytes == PAGE * N_SPEC for x in regions)


def test_runner_grow_is_all_or_nothing_across_tensors(no_cuda):
    be = FakeVmmBackend()
    r, regions, _ = runner(be)
    # init maps 2 chunks (one per region); first region grows 3 chunks fine,
    # second region's first tail chunk fails
    be.fail_map_after = 2 + 3
    with pytest.raises(VmmError):
        GPUModelRunner.grow_kv_cache(r, N_SPEC + TAIL)
    assert [x.mapped_bytes for x in regions] == [PAGE * N_SPEC] * 2
    assert be.mapped_total() == 2 * PAGE * N_SPEC


def test_runner_without_regions_fails_loud(no_cuda):
    r = SimpleNamespace(elastic_kv_regions=[])
    r._elastic_regions = lambda: GPUModelRunner._elastic_regions(r)
    with pytest.raises(RuntimeError, match="elastic"):
        GPUModelRunner.grow_kv_cache(r, 1)


# ---- worker RPC wrappers ----


class _Runner:
    def __init__(self):
        self.calls = []

    def grow_kv_cache(self, n):
        self.calls.append(("grow", n))
        return {"mapped_blocks": n}

    def shrink_kv_cache(self, n):
        self.calls.append(("shrink", n))
        return {"mapped_blocks": n}


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (123, 456))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w.model_runner = _Runner()
    w.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_cumem_allocator=True)
    )
    return w


def test_worker_grow_shrink_report_mem_get_info(worker):
    out = worker.grow_kv_cache(80)
    assert out == {"mapped_blocks": 80, "free_bytes": 123, "total_bytes": 456}
    out = worker.shrink_kv_cache(32)
    assert out["mapped_blocks"] == 32 and out["free_bytes"] == 123
    assert worker.model_runner.calls == [("grow", 80), ("shrink", 32)]


def test_worker_probe_reports_resident_draft_bytes_and_granularity(worker, monkeypatch):
    class _Alloc:
        def get_current_usage(self, tag=None, only_mapped=False):
            assert tag == "draft_weights" and only_mapped is True
            return 1_200_000_000

    monkeypatch.setattr(CuMemAllocator, "get_instance", staticmethod(lambda: _Alloc()))
    monkeypatch.setattr(
        "vllm.device_allocator.vmm.get_vmm_backend",
        lambda: FakeVmmBackend(granularity=2 * MIB),
    )
    assert worker.get_elastic_kv_probe() == {
        "draft_weights_bytes": 1_200_000_000,
        "granularity": 2 * MIB,
        "device": 3,
    }


def test_worker_probe_requires_cumem(worker):
    worker.vllm_config.model_config.enable_cumem_allocator = False
    with pytest.raises(RuntimeError, match="cumem"):
        worker.get_elastic_kv_probe()


# ---- F8: elastic KV bypasses the cumem kv_cache pool ----


@pytest.mark.parametrize("elastic", [True, False])
def test_initialize_from_config_pool_use(monkeypatch, elastic):
    tags = []
    log = []
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w.cache_config = SimpleNamespace(num_gpu_blocks=None)
    w.model_config = SimpleNamespace(enable_return_routed_experts=False)
    w.vllm_config = SimpleNamespace()
    w.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda cfg: log.append("init_kv")
    )

    from contextlib import contextmanager

    @contextmanager
    def pool(tag):
        tags.append(tag)
        yield

    w._maybe_get_memory_pool_context = pool
    monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_initialized", lambda *a: None)
    gpu_worker.Worker.initialize_from_config(w, elastic_cfg(elastic=elastic))
    assert log == ["init_kv"]
    assert tags == ([] if elastic else ["kv_cache"])
