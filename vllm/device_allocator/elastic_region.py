# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""plan 075 S2 A2: one KV tensor whose tail can be mapped and unmapped.

The region reserves ``n_max_bytes`` of virtual address space once and maps the
first ``n_spec_bytes`` (the part the engine always owns). ``grow_to`` maps more
physical memory behind the same base address (so every pointer, block table
and cuda graph that already captured the base stays valid); ``shrink_to``
gives it back. The tensor view always has ``n_max_bytes`` elements; only the
mapped prefix may be touched.

Mapping unit: the tail is mapped / unmapped in ``boundary``-byte chunks
(``boundary = lcm(page_size, granularity)``), because ``cuMemUnmap`` can only
unmap a range exactly as it was mapped. The initial ``n_spec`` region is one
chunk that is never unmapped.

Errors: a failing grow rolls back the chunks it mapped in that call and
re-raises, so the region stays in its previous steady state (F10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.device_allocator.vmm import VmmBackend, get_vmm_backend


@dataclass
class _Chunk:
    offset: int
    size: int
    handle: Any


class ElasticKvRegion:
    def __init__(
        self,
        device: int,
        n_max_bytes: int,
        n_spec_bytes: int,
        boundary: int,
        block_bytes: int,
        backend: VmmBackend | None = None,
    ) -> None:
        self.device = device
        self.backend = backend if backend is not None else get_vmm_backend()
        g = self.backend.granularity(device)
        if boundary <= 0 or boundary % g != 0:
            raise ValueError(
                f"boundary {boundary} must be a positive multiple of the "
                f"VMM granularity {g}"
            )
        if boundary % block_bytes != 0:
            raise ValueError(
                f"boundary {boundary} must be a multiple of block_bytes "
                f"{block_bytes} (block edges must coincide with chunk edges)"
            )
        if n_spec_bytes <= 0 or n_spec_bytes % boundary != 0:
            raise ValueError(
                f"n_spec_bytes {n_spec_bytes} must be a positive multiple of "
                f"boundary {boundary}"
            )
        if n_max_bytes < n_spec_bytes or (n_max_bytes - n_spec_bytes) % boundary != 0:
            raise ValueError(
                f"n_max_bytes {n_max_bytes} must be n_spec_bytes "
                f"{n_spec_bytes} plus a whole number of {boundary}-byte chunks"
            )
        self.granularity = g
        self.boundary = boundary
        self.block_bytes = block_bytes
        self.n_max_bytes = n_max_bytes
        self.n_spec_bytes = n_spec_bytes
        self._chunks: list[_Chunk] = []
        self.va = self.backend.reserve(n_max_bytes, device)
        try:
            self._map_chunk(0, n_spec_bytes, zero=True)
        except BaseException:
            self.backend.release_va(self.va, n_max_bytes)
            raise

    # ---- state -------------------------------------------------------

    @property
    def mapped_bytes(self) -> int:
        return sum(c.size for c in self._chunks)

    @property
    def mapped_blocks(self) -> int:
        return self.mapped_bytes // self.block_bytes

    @property
    def max_blocks(self) -> int:
        return self.n_max_bytes // self.block_bytes

    @property
    def spec_blocks(self) -> int:
        return self.n_spec_bytes // self.block_bytes

    # ---- tensor view -------------------------------------------------

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return {
            "shape": (self.n_max_bytes,),
            "typestr": "|i1",
            "data": (self.va, False),
            "version": 2,
            "strides": None,
        }

    def as_tensor(self) -> Any:
        """int8 tensor of ``n_max_bytes`` elements over the whole VA range."""
        import torch

        return torch.as_tensor(self, device=f"cuda:{self.device}")

    # ---- map / unmap -------------------------------------------------

    def _map_chunk(self, offset: int, size: int, zero: bool) -> None:
        handle = self.backend.map(self.va, offset, size, self.device)
        self._chunks.append(_Chunk(offset, size, handle))
        if zero:
            # F9: never rely on cuMemCreate returning zeroed memory.
            self.backend.memset(self.va, offset, size, 0)

    def grow_to(self, target_bytes: int) -> int:
        """Map chunks until ``mapped_bytes >= target_bytes`` (rounded up to a
        chunk edge, at most ``n_max_bytes``). Returns the new mapped size.
        Idempotent when already large enough."""
        if target_bytes > self.n_max_bytes:
            raise ValueError(
                f"grow_to({target_bytes}) exceeds n_max_bytes {self.n_max_bytes}"
            )
        added = 0
        try:
            while self.mapped_bytes < target_bytes:
                self._map_chunk(self.mapped_bytes, self.boundary, zero=True)
                added += 1
        except BaseException:
            for _ in range(added):
                self._unmap_last()
            raise
        return self.mapped_bytes

    def _unmap_last(self) -> None:
        chunk = self._chunks.pop()
        try:
            self.backend.unmap(self.va, chunk.offset, chunk.size, chunk.handle)
        except BaseException:
            self._chunks.append(chunk)
            raise

    def shrink_to(self, target_bytes: int) -> int:
        """Unmap tail chunks until ``mapped_bytes <= max(target_bytes,
        n_spec_bytes)`` and return the new mapped size. The caller must have
        stopped using the tail first (F10: scheduler before worker)."""
        floor = max(target_bytes, self.n_spec_bytes)
        if self.mapped_bytes > floor:
            self.backend.synchronize()  # F12
        while self.mapped_bytes - self._chunks[-1].size >= floor and len(self._chunks) > 1:
            self._unmap_last()
        return self.mapped_bytes

    def close(self) -> None:
        """Unmap everything and free the VA range."""
        self.backend.synchronize()
        while self._chunks:
            self._unmap_last()
        self.backend.release_va(self.va, self.n_max_bytes)
