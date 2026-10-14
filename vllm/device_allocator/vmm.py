# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA virtual memory management (VMM) bindings for the plan 075 S2 elastic
KV tail.

``VmmBackend`` is the small interface ``ElasticKvRegion`` needs: reserve a
virtual address range once, then map / unmap physical chunks inside it. Three
implementations:

* ``CudaPythonVmmBackend`` -- ``cuda.bindings.driver`` (cuda-python), the
  preferred real backend (F3 (a)). Importing this module never touches CUDA.
* ``CtypesVmmBackend`` -- ``libcuda.so.1`` through ``ctypes`` (F3 (b)), used
  only when cuda-python is missing.
* ``FakeVmmBackend`` -- records calls and checks alignment / overlap; the CPU
  test double.

No C++ and no ``vllm/cumem_allocator`` involvement (F3): the cumem allocator's
``.so`` is a symlink into a shared venv and is deliberately left alone.
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from typing import Any


class VmmError(RuntimeError):
    """A CUDA VMM driver call failed."""


class VmmReleaseError(VmmError):
    """``cuMemUnmap`` succeeded but ``cuMemRelease`` of the handle failed: the
    range is already unmapped (the physical handle leaked)."""


class VmmBackend(ABC):
    """Reserve one VA range, then map / unmap physical chunks inside it."""

    @abstractmethod
    def granularity(self, device: int) -> int:
        """Minimum VMM allocation granularity in bytes (typically 2 MiB)."""

    @abstractmethod
    def reserve(self, size: int, device: int) -> int:
        """Reserve ``size`` bytes of VA (no physical memory) and return it."""

    @abstractmethod
    def map(self, va: int, offset: int, size: int, device: int) -> Any:
        """Create ``size`` bytes of physical memory and map it at
        ``va + offset`` with read/write access for ``device``. Returns an
        opaque handle that must be passed back to ``unmap``."""

    @abstractmethod
    def unmap(self, va: int, offset: int, size: int, handle: Any) -> None:
        """Unmap exactly the range a previous ``map`` created and free its
        physical memory. Sub-ranges are not allowed (cuMemUnmap rule)."""

    @abstractmethod
    def release_va(self, va: int, size: int) -> None:
        """Free the VA reservation. Everything must be unmapped first."""

    @abstractmethod
    def memset(self, va: int, offset: int, size: int, value: int = 0) -> None:
        """Fill ``size`` bytes at ``va + offset`` (device memory, mapped)."""

    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all outstanding device work (called before any unmap)."""


# --------------------------------------------------------------------------
# cuda-python
# --------------------------------------------------------------------------


def _check(result: Any, what: str) -> Any:
    """cuda-python returns ``(err, *values)``; raise on a non-zero error."""
    err, *values = result
    if int(err) != 0:
        raise VmmError(f"{what} failed: CUresult {int(err)} ({err})")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class CudaPythonVmmBackend(VmmBackend):
    """VMM through ``cuda.bindings.driver``. The driver module may be injected
    (tests); by default it is imported lazily on first use."""

    def __init__(self, driver: Any | None = None) -> None:
        self._driver = driver

    @property
    def drv(self) -> Any:
        if self._driver is None:
            from cuda.bindings import driver  # lazy: no CUDA init on import

            self._driver = driver
        return self._driver

    def _prop(self, device: int) -> Any:
        d = self.drv
        prop = d.CUmemAllocationProp()
        prop.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        return prop

    def granularity(self, device: int) -> int:
        d = self.drv
        flag = d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        return int(
            _check(
                d.cuMemGetAllocationGranularity(self._prop(device), flag),
                "cuMemGetAllocationGranularity",
            )
        )

    def reserve(self, size: int, device: int) -> int:
        d = self.drv
        return int(
            _check(d.cuMemAddressReserve(size, 0, 0, 0), "cuMemAddressReserve")
        )

    def map(self, va: int, offset: int, size: int, device: int) -> Any:
        d = self.drv
        handle = _check(d.cuMemCreate(size, self._prop(device), 0), "cuMemCreate")
        try:
            _check(d.cuMemMap(va + offset, size, 0, handle, 0), "cuMemMap")
            try:
                access = d.CUmemAccessDesc()
                access.location.type = (
                    d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                )
                access.location.id = device
                access.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check(
                    d.cuMemSetAccess(va + offset, size, [access], 1),
                    "cuMemSetAccess",
                )
            except BaseException:
                d.cuMemUnmap(va + offset, size)
                raise
        except BaseException:
            d.cuMemRelease(handle)
            raise
        return handle

    def unmap(self, va: int, offset: int, size: int, handle: Any) -> None:
        d = self.drv
        _check(d.cuMemUnmap(va + offset, size), "cuMemUnmap")
        try:
            _check(d.cuMemRelease(handle), "cuMemRelease")
        except VmmError as exc:
            raise VmmReleaseError(str(exc)) from exc

    def release_va(self, va: int, size: int) -> None:
        _check(self.drv.cuMemAddressFree(va, size), "cuMemAddressFree")

    def memset(self, va: int, offset: int, size: int, value: int = 0) -> None:
        _check(self.drv.cuMemsetD8(va + offset, value, size), "cuMemsetD8")

    def synchronize(self) -> None:
        _check(self.drv.cuCtxSynchronize(), "cuCtxSynchronize")


# --------------------------------------------------------------------------
# ctypes fallback
# --------------------------------------------------------------------------


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationProp(ctypes.Structure):
    # CUmemAllocationProp_v1: type, requestedHandleTypes, location, win32HandleMetaData,
    # allocFlags{compressionType, gpuDirectRDMACapable, usage, reserved[4]}
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0


class CtypesVmmBackend(VmmBackend):
    """VMM through ``libcuda.so.1`` (F3 (b)). The library object may be
    injected (tests)."""

    def __init__(self, lib: Any | None = None) -> None:
        self._lib = lib

    @property
    def lib(self) -> Any:
        if self._lib is None:
            self._lib = ctypes.CDLL("libcuda.so.1")
        return self._lib

    @staticmethod
    def _ck(err: int, what: str) -> None:
        if err != 0:
            raise VmmError(f"{what} failed: CUresult {err}")

    def _prop(self, device: int) -> _CUmemAllocationProp:
        prop = _CUmemAllocationProp()
        prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        return prop

    def granularity(self, device: int) -> int:
        out = ctypes.c_size_t(0)
        self._ck(
            self.lib.cuMemGetAllocationGranularity(
                ctypes.byref(out),
                ctypes.byref(self._prop(device)),
                ctypes.c_int(_CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            ),
            "cuMemGetAllocationGranularity",
        )
        return int(out.value)

    def reserve(self, size: int, device: int) -> int:
        out = ctypes.c_ulonglong(0)
        self._ck(
            self.lib.cuMemAddressReserve(
                ctypes.byref(out),
                ctypes.c_size_t(size),
                ctypes.c_size_t(0),
                ctypes.c_ulonglong(0),
                ctypes.c_ulonglong(0),
            ),
            "cuMemAddressReserve",
        )
        return int(out.value)

    def map(self, va: int, offset: int, size: int, device: int) -> Any:
        lib = self.lib
        handle = ctypes.c_ulonglong(0)
        self._ck(
            lib.cuMemCreate(
                ctypes.byref(handle),
                ctypes.c_size_t(size),
                ctypes.byref(self._prop(device)),
                ctypes.c_ulonglong(0),
            ),
            "cuMemCreate",
        )
        ptr = ctypes.c_ulonglong(va + offset)
        try:
            self._ck(
                lib.cuMemMap(
                    ptr,
                    ctypes.c_size_t(size),
                    ctypes.c_size_t(0),
                    handle,
                    ctypes.c_ulonglong(0),
                ),
                "cuMemMap",
            )
            try:
                access = _CUmemAccessDesc()
                access.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = device
                access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                self._ck(
                    lib.cuMemSetAccess(
                        ptr,
                        ctypes.c_size_t(size),
                        ctypes.byref(access),
                        ctypes.c_size_t(1),
                    ),
                    "cuMemSetAccess",
                )
            except BaseException:
                lib.cuMemUnmap(ptr, ctypes.c_size_t(size))
                raise
        except BaseException:
            lib.cuMemRelease(handle)
            raise
        return handle

    def unmap(self, va: int, offset: int, size: int, handle: Any) -> None:
        self._ck(
            self.lib.cuMemUnmap(
                ctypes.c_ulonglong(va + offset), ctypes.c_size_t(size)
            ),
            "cuMemUnmap",
        )
        try:
            self._ck(self.lib.cuMemRelease(handle), "cuMemRelease")
        except VmmError as exc:
            raise VmmReleaseError(str(exc)) from exc

    def release_va(self, va: int, size: int) -> None:
        self._ck(
            self.lib.cuMemAddressFree(
                ctypes.c_ulonglong(va), ctypes.c_size_t(size)
            ),
            "cuMemAddressFree",
        )

    def memset(self, va: int, offset: int, size: int, value: int = 0) -> None:
        self._ck(
            self.lib.cuMemsetD8_v2(
                ctypes.c_ulonglong(va + offset),
                ctypes.c_ubyte(value),
                ctypes.c_size_t(size),
            ),
            "cuMemsetD8",
        )

    def synchronize(self) -> None:
        self._ck(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


def get_vmm_backend() -> VmmBackend:
    """F3 selection order: cuda-python, then ctypes. (A C++ extension is the
    last resort and is intentionally not implemented here.)"""
    import importlib.util

    try:
        have_cuda_python = importlib.util.find_spec("cuda.bindings.driver") is not None
    except ImportError:
        have_cuda_python = False
    return CudaPythonVmmBackend() if have_cuda_python else CtypesVmmBackend()


# --------------------------------------------------------------------------
# CPU test double
# --------------------------------------------------------------------------


class FakeVmmBackend(VmmBackend):
    """Records every call and enforces the real driver's rules: granularity
    alignment, mapping only inside a reservation, no overlapping maps, and
    ``unmap`` only of exactly a previously mapped range."""

    def __init__(self, granularity: int = 2 << 20, base: int = 0x7F0000000000):
        self.g = granularity
        self.base = base
        self.calls: list[tuple] = []
        self.reservations: dict[int, int] = {}  # va -> size
        self.mapped: dict[tuple[int, int], int] = {}  # (abs_start, size) -> handle
        self.fail_map_after: int | None = None  # raise after N successful maps
        self.fail_memset_after: int | None = None  # raise after N ok memsets
        self._memsets_done = 0
        self.fail_release_on_unmap = False  # unmap ok, handle release fails
        self._maps_done = 0
        self._next_va = base
        self._next_handle = 1

    def granularity(self, device: int) -> int:
        self.calls.append(("granularity", device))
        return self.g

    def reserve(self, size: int, device: int) -> int:
        self.calls.append(("reserve", size, device))
        assert size % self.g == 0, f"reserve size {size} not a multiple of {self.g}"
        va = self._next_va
        self._next_va += size + self.g  # leave a guard gap
        self.reservations[va] = size
        return va

    def _owner(self, start: int, size: int) -> int:
        for va, sz in self.reservations.items():
            if va <= start and start + size <= va + sz:
                return va
        raise VmmError(f"range [{start:#x}, +{size:#x}) is outside any reservation")

    def map(self, va: int, offset: int, size: int, device: int) -> Any:
        self.calls.append(("map", va, offset, size, device))
        assert offset % self.g == 0 and size % self.g == 0 and size > 0, (
            f"map offset={offset} size={size} not aligned to {self.g}"
        )
        start = va + offset
        self._owner(start, size)
        for (s, n) in self.mapped:
            if start < s + n and s < start + size:
                raise VmmError(f"map [{start:#x}, +{size:#x}) overlaps a mapping")
        if self.fail_map_after is not None and self._maps_done >= self.fail_map_after:
            raise VmmError("injected map failure")
        self._maps_done += 1
        handle = self._next_handle
        self._next_handle += 1
        self.mapped[(start, size)] = handle
        return handle

    def unmap(self, va: int, offset: int, size: int, handle: Any) -> None:
        self.calls.append(("unmap", va, offset, size))
        key = (va + offset, size)
        if self.mapped.get(key) != handle:
            raise VmmError(f"unmap {key} is not exactly one mapped range")
        del self.mapped[key]
        if self.fail_release_on_unmap:
            raise VmmReleaseError("injected cuMemRelease failure")

    def release_va(self, va: int, size: int) -> None:
        self.calls.append(("release_va", va, size))
        if self.reservations.get(va) != size:
            raise VmmError("release_va of an unknown reservation")
        if any(va <= s < va + size for (s, _n) in self.mapped):
            raise VmmError("release_va while ranges are still mapped")
        del self.reservations[va]

    def memset(self, va: int, offset: int, size: int, value: int = 0) -> None:
        self.calls.append(("memset", va, offset, size, value))
        start = va + offset
        if not self._covered(start, size):
            raise VmmError(f"memset [{start:#x}, +{size:#x}) hits unmapped memory")
        if self.fail_memset_after is not None and (
            self._memsets_done >= self.fail_memset_after
        ):
            raise VmmError("injected memset failure")
        self._memsets_done += 1

    def _covered(self, start: int, size: int) -> bool:
        """True when [start, start+size) is fully covered by mapped chunks."""
        pos = start
        end = start + size
        for s, n in sorted(self.mapped):
            if s <= pos < s + n:
                pos = s + n
            if pos >= end:
                return True
        return pos >= end

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))

    # test helpers
    def mapped_total(self) -> int:
        return sum(n for (_s, n) in self.mapped)

    def ops(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]
