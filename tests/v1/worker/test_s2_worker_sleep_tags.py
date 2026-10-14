# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1 (P2-3): Worker.sleep_tags / wake_tags, CPU-only (fake
allocator, fake runner; torch.cuda calls are patched)."""

from types import SimpleNamespace

import pytest

from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker import gpu_worker

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _FakeAlloc:
    def __init__(self, log):
        self.log = log
        self.mapped = 1000  # bytes, shared with the free-memory fake
        self.free = 5000

    def get_current_usage(self, tag=None, only_mapped=False):
        return self.mapped

    def sleep(self, offload_tags=None, unmap_tags=None):
        self.log.append(("sleep", offload_tags, unmap_tags))
        self.mapped -= 400
        self.free += 400

    def wake_up(self, tags=None):
        self.log.append(("wake", tags))
        self.mapped += 400
        self.free -= 400


@pytest.fixture
def env(monkeypatch):
    log: list[tuple] = []
    alloc = _FakeAlloc(log)
    monkeypatch.setattr(CuMemAllocator, "get_instance", staticmethod(lambda: alloc))
    monkeypatch.setattr(gpu_worker.torch.cuda, "synchronize", lambda *a: log.append(("sync",)))
    monkeypatch.setattr(gpu_worker.torch.cuda, "mem_get_info", lambda *a: (alloc.free, 10**6))
    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_pin_memory_available", lambda: True
    )
    runner = SimpleNamespace(
        set_drafter_resident=lambda r: log.append(("resident", r))
    )
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_cumem_allocator=True)
    )
    w.model_runner = runner
    return w, alloc, log


def test_sleep_tags_order_and_accounting(env):
    w, alloc, log = env
    out = w.sleep_tags(("draft_weights",))
    # F12: sync first, then close the gate, then unmap
    assert log == [
        ("sync",),
        ("resident", False),
        ("sleep", ("draft_weights",), ("draft_weights",)),
    ]
    assert out["freed_bytes"] == 400 and out["unmapped_bytes"] == 400
    assert out["free_after"] - out["free_before"] == 400


def test_wake_tags_reopens_gate_only_after_copy(env):
    w, alloc, log = env
    w.sleep_tags(["draft_weights"])
    log.clear()
    out = w.wake_tags(["draft_weights"])
    assert log == [("wake", ["draft_weights"]), ("sync",), ("resident", True)]
    assert out["mapped_bytes"] == 400


def test_wake_failure_keeps_gate_closed(env):
    w, alloc, log = env
    w.sleep_tags(["draft_weights"])
    log.clear()

    def boom(tags=None):
        raise RuntimeError("map failed")

    alloc.wake_up = boom
    with pytest.raises(RuntimeError):
        w.wake_tags(["draft_weights"])
    assert ("resident", True) not in log


@pytest.mark.parametrize("tags", [(), ("weights",), ("kv_cache",), ("draft_weights", "weights")])
def test_rejects_other_tags(env, tags):
    w, _, log = env
    with pytest.raises(ValueError):
        w.sleep_tags(tags)
    with pytest.raises(ValueError):
        w.wake_tags(tags)
    assert log == []


def test_sleep_tags_requires_cumem(env):
    w, _, log = env
    w.vllm_config.model_config.enable_cumem_allocator = False
    with pytest.raises(RuntimeError, match="cumem"):
        w.sleep_tags(("draft_weights",))
    assert log == []


def test_sleep_tags_fails_loud_without_pinned_memory(env, monkeypatch):
    w, _, log = env
    monkeypatch.setattr("vllm.utils.platform_utils.is_pin_memory_available", lambda: False)
    with pytest.raises(RuntimeError, match="pinned"):
        w.sleep_tags(("draft_weights",))
    assert log == []


def test_legacy_sleep_wake_untouched():
    # S2 must not change the legacy RPC surface.
    import inspect

    assert list(inspect.signature(gpu_worker.Worker.sleep).parameters) == ["self", "level"]
    assert list(inspect.signature(gpu_worker.Worker.wake_up).parameters) == ["self", "tags"]


# ---- M1-fix-1: legacy sleep()/wake_up() are hot-switch aware ----


def _legacy_env(env, monkeypatch, resident=True):
    w, alloc, log = env
    spec = SimpleNamespace(drafter_resident=resident)
    w.model_runner = SimpleNamespace(
        speculator=spec,
        post_kv_cache_wake_up=lambda: log.append(("post_kv",)),
    )
    w._sleep_saved_buffers = {}
    return w, alloc, log, spec


def test_legacy_sleep_level1_offloads_drafter_weights_too(env, monkeypatch):
    w, alloc, log, _ = _legacy_env(env, monkeypatch)
    w.sleep(1)
    assert ("sleep", ("weights", "draft_weights"), None) in log


def test_legacy_sleep_refused_while_drafter_released(env, monkeypatch):
    w, alloc, log, _ = _legacy_env(env, monkeypatch, resident=False)
    with pytest.raises(RuntimeError, match="DFLASH2"):
        w.sleep(1)
    assert not [e for e in log if e[0] == "sleep"]  # nothing was unmapped


def test_legacy_wake_weights_also_wakes_draft_weights(env, monkeypatch):
    w, alloc, log, _ = _legacy_env(env, monkeypatch)
    w.wake_up(["weights"])
    assert ("wake", ["weights", "draft_weights"]) in log
    log.clear()
    w.wake_up(["weights", "draft_weights"])  # no duplicate tag
    assert ("wake", ["weights", "draft_weights"]) in log
    log.clear()
    w.wake_up(["kv_cache"])  # unrelated tag untouched
    assert ("wake", ["kv_cache"]) in log


def test_legacy_sleep_refused_with_elastic_kv_regions(env, monkeypatch):
    w, alloc, log, _ = _legacy_env(env, monkeypatch)
    w.model_runner.elastic_kv_regions = [object()]
    with pytest.raises(RuntimeError, match="elastic"):
        w.sleep(1)
    assert not [e for e in log if e[0] == "sleep"]
