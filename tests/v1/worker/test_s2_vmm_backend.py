# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2 (A2-1): VmmBackend implementations, CPU-only. The real
backends run against a recording fake of the driver (no CUDA)."""

from types import SimpleNamespace

import pytest

from vllm.device_allocator.vmm import (
    CtypesVmmBackend,
    CudaPythonVmmBackend,
    FakeVmmBackend,
    VmmError,
    get_vmm_backend,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MIB = 1 << 20


# ---- FakeVmmBackend enforces the real driver's rules ----


def test_fake_rejects_unaligned_overlapping_and_foreign_ranges():
    b = FakeVmmBackend(granularity=2 * MIB)
    va = b.reserve(16 * MIB, 0)
    with pytest.raises(AssertionError):
        b.map(va, 1 * MIB, 2 * MIB, 0)  # offset not aligned
    with pytest.raises(AssertionError):
        b.map(va, 0, 3 * MIB, 0)  # size not aligned
    h = b.map(va, 0, 4 * MIB, 0)
    with pytest.raises(VmmError):
        b.map(va, 2 * MIB, 2 * MIB, 0)  # overlaps
    with pytest.raises(VmmError):
        b.map(va, 16 * MIB, 2 * MIB, 0)  # outside the reservation
    with pytest.raises(VmmError):
        b.unmap(va, 0, 2 * MIB, h)  # sub-range unmap is illegal
    b.unmap(va, 0, 4 * MIB, h)
    assert b.mapped_total() == 0


def test_fake_release_va_requires_everything_unmapped_and_memset_mapped_only():
    b = FakeVmmBackend()
    va = b.reserve(8 * MIB, 0)
    h = b.map(va, 0, 2 * MIB, 0)
    b.memset(va, 0, 2 * MIB)
    with pytest.raises(VmmError):
        b.memset(va, 2 * MIB, 2 * MIB)  # unmapped
    with pytest.raises(VmmError):
        b.release_va(va, 8 * MIB)
    b.unmap(va, 0, 2 * MIB, h)
    b.release_va(va, 8 * MIB)


# ---- cuda-python backend: plumbing against a fake driver module ----


class _Rec:
    def __init__(self):
        self.calls = []
        self.fail = {}

    def make(self, name, ret=()):
        def fn(*a):
            self.calls.append((name, *a))
            if name in self.fail:
                return (self.fail[name],)
            return (0, *ret)

        return fn


def _fake_driver(rec):
    ns = SimpleNamespace()
    ns.CUmemAllocationProp = lambda: SimpleNamespace(
        type=None, location=SimpleNamespace(type=None, id=None)
    )
    ns.CUmemAccessDesc = lambda: SimpleNamespace(
        location=SimpleNamespace(type=None, id=None), flags=None
    )
    ns.CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED="pinned")
    ns.CUmemLocationType = SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE="dev")
    ns.CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE="rw")
    ns.CUmemAllocationGranularity_flags = SimpleNamespace(
        CU_MEM_ALLOC_GRANULARITY_MINIMUM="min"
    )
    ns.cuMemGetAllocationGranularity = rec.make("gran", (2 * MIB,))
    ns.cuMemAddressReserve = rec.make("reserve", (0x1000,))
    ns.cuMemCreate = rec.make("create", ("H",))
    ns.cuMemMap = rec.make("map")
    ns.cuMemSetAccess = rec.make("access")
    ns.cuMemUnmap = rec.make("unmap")
    ns.cuMemRelease = rec.make("release")
    ns.cuMemAddressFree = rec.make("free")
    ns.cuMemsetD8 = rec.make("memset")
    ns.cuCtxSynchronize = rec.make("sync")
    return ns


def test_cuda_python_backend_call_sequence():
    rec = _Rec()
    b = CudaPythonVmmBackend(driver=_fake_driver(rec))
    assert b.granularity(0) == 2 * MIB
    va = b.reserve(64 * MIB, 0)
    assert va == 0x1000
    h = b.map(va, 4 * MIB, 2 * MIB, 0)
    b.memset(va, 4 * MIB, 2 * MIB)
    b.synchronize()
    b.unmap(va, 4 * MIB, 2 * MIB, h)
    b.release_va(va, 64 * MIB)
    names = [c[0] for c in rec.calls]
    assert names == [
        "gran", "reserve", "create", "map", "access",
        "memset", "sync", "unmap", "release", "free",
    ]
    # offsets are applied to the VA for map / memset / unmap
    assert rec.calls[3][1] == 0x1000 + 4 * MIB
    assert rec.calls[5][1] == 0x1000 + 4 * MIB
    assert rec.calls[7][1] == 0x1000 + 4 * MIB
    # the physical handle is released only by unmap
    assert rec.calls[8] == ("release", "H")


def test_cuda_python_map_failure_releases_handle_and_unmaps():
    rec = _Rec()
    rec.fail["access"] = 2
    b = CudaPythonVmmBackend(driver=_fake_driver(rec))
    with pytest.raises(VmmError, match="cuMemSetAccess"):
        b.map(0x1000, 0, 2 * MIB, 0)
    names = [c[0] for c in rec.calls]
    assert names[-2:] == ["unmap", "release"]


def test_cuda_python_error_code_raises():
    rec = _Rec()
    rec.fail["reserve"] = 1
    b = CudaPythonVmmBackend(driver=_fake_driver(rec))
    with pytest.raises(VmmError, match="cuMemAddressReserve"):
        b.reserve(MIB, 0)


# ---- ctypes backend: plumbing against a fake libcuda ----


class _FakeLib:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("cu"):
            raise AttributeError(name)

        def fn(*a):
            self.calls.append(name)
            if name == "cuMemGetAllocationGranularity":
                a[0]._obj.value = 2 * MIB
            elif name in ("cuMemAddressReserve", "cuMemCreate"):
                a[0]._obj.value = 0x2000
            return 0

        return fn


def test_ctypes_backend_call_sequence():
    lib = _FakeLib()
    b = CtypesVmmBackend(lib=lib)
    assert b.granularity(0) == 2 * MIB
    va = b.reserve(8 * MIB, 0)
    assert va == 0x2000
    h = b.map(va, 0, 2 * MIB, 0)
    b.memset(va, 0, 2 * MIB)
    b.unmap(va, 0, 2 * MIB, h)
    b.release_va(va, 8 * MIB)
    assert lib.calls == [
        "cuMemGetAllocationGranularity", "cuMemAddressReserve", "cuMemCreate",
        "cuMemMap", "cuMemSetAccess", "cuMemsetD8_v2", "cuMemUnmap",
        "cuMemRelease", "cuMemAddressFree",
    ]


def test_get_vmm_backend_prefers_cuda_python():
    pytest.importorskip("cuda.bindings.driver")
    assert isinstance(get_vmm_backend(), CudaPythonVmmBackend)
