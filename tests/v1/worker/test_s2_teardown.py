# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M3: worker shutdown puts released tags back and closes the
elastic KV regions before the runner (and the cumem free callback) go away;
plus the grow-precheck probe. CPU-only."""

from types import SimpleNamespace

import pytest

from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Alloc:
    def __init__(self, total, mapped):
        self.total, self.mapped = total, mapped

    def get_current_usage(self, tag=None, only_mapped=False):
        return self.mapped if only_mapped else self.total


def worker(monkeypatch, alloc, log, runner=None, cumem=True):
    monkeypatch.setattr(CuMemAllocator, "get_instance", staticmethod(lambda: alloc))
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_cumem_allocator=cumem))
    w.wake_tags = lambda tags: log.append(("wake", tuple(tags)))
    w.model_runner = runner or SimpleNamespace(close_elastic_kv=lambda: log.append(("close",)))
    return w


def test_teardown_wakes_released_tag_before_closing_regions(monkeypatch):
    log = []
    w = worker(monkeypatch, _Alloc(total=100, mapped=40), log)
    w._s2_teardown()
    assert log == [("wake", ("draft_weights",)), ("close",)]  # wake first


def test_teardown_skips_wake_when_everything_is_mapped(monkeypatch):
    log = []
    w = worker(monkeypatch, _Alloc(total=100, mapped=100), log)
    w._s2_teardown()
    assert log == [("close",)]


def test_teardown_survives_a_failing_wake_and_still_closes(monkeypatch):
    log = []
    w = worker(monkeypatch, _Alloc(total=100, mapped=0), log)

    def boom(tags):
        raise RuntimeError("H2D failed")

    w.wake_tags = boom
    w._s2_teardown()  # must not raise: shutdown cannot be blocked
    assert log == [("close",)]


def test_teardown_without_cumem_or_elastic_runner_is_a_noop(monkeypatch):
    log = []
    w = worker(monkeypatch, _Alloc(100, 0), log, cumem=False, runner=SimpleNamespace())
    w._s2_teardown()
    assert log == []


def test_shutdown_calls_teardown_before_the_runner_shutdown(monkeypatch):
    order = []
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w.profiler = None
    w._ple_offload_worker_handle = None
    w._s2_teardown = lambda: order.append("teardown")
    w.model_runner = SimpleNamespace(shutdown=lambda: order.append("runner"))
    monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_shutdown", None)
    monkeypatch.setattr(gpu_worker, "ensure_ec_transfer_shutdown", None)
    w.shutdown()
    assert order == ["teardown", "runner"]


def test_runner_close_elastic_kv_closes_every_region_and_empties_the_list(monkeypatch):
    closed = []
    regions = [SimpleNamespace(close=lambda i=i: closed.append(i)) for i in range(3)]
    monkeypatch.setattr("torch.cuda.synchronize", lambda *a: closed.append("sync"))
    r = GPUModelRunner.__new__(GPUModelRunner)
    r.elastic_kv_regions = regions
    r.close_elastic_kv()
    assert closed == ["sync", 0, 1, 2] and r.elastic_kv_regions == []
    r.close_elastic_kv()  # second call: nothing left, no sync either
    assert closed == ["sync", 0, 1, 2]


def test_get_gpu_free_bytes(monkeypatch):
    monkeypatch.setattr(gpu_worker.torch.cuda, "mem_get_info", lambda *a: (123, 456))
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    assert w.get_gpu_free_bytes() == {"free_bytes": 123, "total_bytes": 456}
