# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-size private host region; bypass PyTorch pinned allocator rounding."""

import mmap

import torch


class ParkingPool:
    def __init__(self, size_bytes: int):
        if size_bytes <= 0 or size_bytes % mmap.PAGESIZE:
            raise ValueError("parking pool must contain whole OS pages")
        self.size_bytes = size_bytes
        self.offset = 0
        self.owners = 2  # One per transfer direction; both must drain.
        self.pinned = False
        self.base = None
        self.mapping = mmap.mmap(
            -1, size_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        try:
            # Fault on the allocating thread under its MPOL_BIND policy.
            # No cached pinned allocation or parallel torch.zeros first touch.
            self.mapping.madvise(getattr(mmap, "MADV_POPULATE_WRITE", 23))
            self.base = torch.frombuffer(self.mapping, dtype=torch.int8)
            result = torch.cuda.cudart().cudaHostRegister(
                self.base.data_ptr(), size_bytes, 0
            )
            if result.value != 0:
                raise RuntimeError(f"parking cudaHostRegister failed: {result.value}")
            self.pinned = True
        except Exception:
            self.base = None
            self.mapping.close()
            raise

    def allocate(self, slots: int, page: int):
        end = self.offset + slots * page
        if end > self.size_bytes:
            raise ValueError("parking tensor exceeds fixed pool quota")
        view = self.base[self.offset : end].view(slots, page)
        self.offset = end
        return view

    def release_direction(self):
        self.owners -= 1
        if self.owners == 0:
            self.close()

    def close(self):
        if self.pinned:
            result = torch.cuda.cudart().cudaHostUnregister(self.base.data_ptr())
            if result.value != 0:
                # Keep the registered owner alive; engine teardown is safer
                # than unmapping pages whose registration cannot be released.
                raise RuntimeError(f"HOST_PARKING fatal_unregister code={result.value}")
            self.pinned = False
        self.base = None
        self.mapping.close()
