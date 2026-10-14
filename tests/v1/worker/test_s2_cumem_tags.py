# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1 (P2-2): cumem tag-scoped sleep / wake bookkeeping, CPU-only.

The cumem C extension is replaced by recording fakes, so nothing here touches
the GPU.
"""


import pytest

from vllm.device_allocator import cumem
from vllm.device_allocator.cumem import AllocationData, CuMemAllocator

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _FakeCudart:
    def __init__(self, log):
        self.log = log

    def cudaMemcpy(self, dst, src, n):
        self.log.append(("memcpy", dst, src, n))


@pytest.fixture
def alloc(monkeypatch):
    log: list[tuple] = []
    monkeypatch.setattr(cumem, "unmap_and_release", lambda h: log.append(("unmap", h[2])))
    monkeypatch.setattr(cumem, "create_and_map", lambda h: log.append(("map", h[2])))
    monkeypatch.setattr(cumem, "libcudart", _FakeCudart(log))
    monkeypatch.setattr(cumem.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cumem, "is_pin_memory_available", lambda: False)
    a = CuMemAllocator()
    # (tag, address, size): two drafter blocks, two target blocks, one kv block
    for tag, addr, size in [
        ("weights", 1000, 4096),
        ("weights", 2000, 8192),
        ("draft_weights", 3000, 1024),
        ("draft_weights", 4000, 2048),
        ("kv_cache", 5000, 16384),
    ]:
        a.pointer_to_data[addr] = AllocationData((0, size, addr, 0), tag)
    return a, log


def _mapped(a):
    return {p: d.mapped for p, d in a.pointer_to_data.items()}


def test_tag_scoped_sleep_only_unmaps_that_tag(alloc):
    a, log = alloc
    a.sleep(offload_tags=("draft_weights",), unmap_tags=("draft_weights",))
    assert [e for e in log if e[0] == "unmap"] == [("unmap", 3000), ("unmap", 4000)]
    assert _mapped(a) == {1000: True, 2000: True, 3000: False, 4000: False, 5000: True}
    # offloaded: both drafter blocks backed up to host, nothing else
    assert sorted(e[3] for e in log if e[0] == "memcpy") == [1024, 2048]
    assert a.pointer_to_data[3000].cpu_backup_tensor is not None
    assert a.pointer_to_data[1000].cpu_backup_tensor is None


def test_second_sleep_of_same_tag_is_noop(alloc):
    a, log = alloc
    a.sleep(offload_tags=("draft_weights",), unmap_tags=("draft_weights",))
    n = len(log)
    backup = a.pointer_to_data[3000].cpu_backup_tensor
    a.sleep(offload_tags=("draft_weights",), unmap_tags=("draft_weights",))
    assert len(log) == n
    assert a.pointer_to_data[3000].cpu_backup_tensor is backup


def test_wake_up_skips_mapped_and_restores_content(alloc):
    a, log = alloc
    a.sleep(offload_tags=("draft_weights",), unmap_tags=("draft_weights",))
    log.clear()
    a.wake_up(tags=["weights", "draft_weights"])
    # weights were never unmapped -> not re-mapped; only the drafter maps back
    assert [e for e in log if e[0] == "map"] == [("map", 3000), ("map", 4000)]
    assert sorted(e[3] for e in log if e[0] == "memcpy") == [1024, 2048]
    assert all(_mapped(a).values())
    assert a.pointer_to_data[3000].cpu_backup_tensor is None
    log.clear()
    a.wake_up(tags=["draft_weights"])
    assert log == []


def test_legacy_sleep_unmaps_everything_and_skips_already_unmapped(alloc):
    a, log = alloc
    a.sleep(offload_tags=("draft_weights",), unmap_tags=("draft_weights",))
    log.clear()
    a.sleep(offload_tags=("weights",))
    assert sorted(e[1] for e in log if e[0] == "unmap") == [1000, 2000, 5000]
    assert not any(_mapped(a).values())


def test_get_current_usage_tag_and_only_mapped(alloc):
    a, _ = alloc
    total = 4096 + 8192 + 1024 + 2048 + 16384
    assert a.get_current_usage() == total
    assert a.get_current_usage(tag="draft_weights") == 3072
    a.sleep(offload_tags=("draft_weights",), unmap_tags=("draft_weights",))
    assert a.get_current_usage() == total  # reserved, still counted
    assert a.get_current_usage(only_mapped=True) == total - 3072
    assert a.get_current_usage(tag="draft_weights", only_mapped=True) == 0
    assert a.get_current_usage(tag="weights", only_mapped=True) == 12288


def test_malloc_callback_creates_mapped_allocation(alloc):
    a, _ = alloc
    a.current_tag = "draft_weights"
    a._python_malloc_callback((0, 512, 7000, 0))
    assert a.pointer_to_data[7000].mapped is True
    assert a.pointer_to_data[7000].tag == "draft_weights"
