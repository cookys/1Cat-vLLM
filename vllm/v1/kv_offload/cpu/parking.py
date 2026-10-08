# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, process-local host parking policy. No CUDA work at import time."""

import ctypes
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass

GIB = 1 << 30


@dataclass(frozen=True)
class ParkingConfig:
    # All amounts are aggregate across the TP workers, not per rank.
    total_bytes: int = 16 * GIB
    numa_node: int = 0

    @classmethod
    def parse(cls, extra):
        if extra.get("host_parking", False) is not True:
            raise ValueError("HostParkingSpec requires host_parking=true (default off)")
        size = extra.get("cpu_bytes_to_use", 16 * GIB)
        node = extra.get("parking_numa_node", 0)
        if type(size) is not int or not 0 < size <= 32 * GIB:
            raise ValueError("parking budget must be integer bytes in (0, 32 GiB]")
        if size > 25 * GIB and extra.get("parking_large_pool_ack") is not True:
            raise ValueError("parking >25 GiB requires lead/las notice and ack")
        if type(node) is not int or not 0 <= node < 64:
            raise ValueError("parking_numa_node must be an integer in [0, 64)")
        if extra.get("parking_mode", "completed_boundary") != "completed_boundary":
            raise ValueError("v1 supports completed_boundary, not an idle timer")
        if extra.get("offload_prompt_only", True) is not True:
            raise ValueError("parking v1 only stores committed prompt boundaries")
        if extra.get("store_threshold", 0) not in (0, 1):
            raise ValueError("parking must not filter one-time boundary handoffs")
        if extra.get("eviction_policy", "lru") != "lru":
            raise ValueError("parking v1 requires LRU")
        return cls(size, node)


def layout_namespace(config, cache_config) -> bytes:
    """Hash model/quant/group geometry and fixed layer-scale contract.

    No persistent files or cross-engine cache sharing: reset generation is
    added separately. The worker verifies unit scales before registering any
    host memory. In-row NVFP4 scales are copied as part of the raw page.
    """
    model = config.model_config
    identity = {
        "version": 1,
        "model": model.model,
        "revision": getattr(model, "revision", None),
        "dtype": str(model.dtype),
        "quantization": getattr(model, "quantization", None),
        "hf_config": model.hf_config.to_dict(),
        "tp": config.parallel_config.tensor_parallel_size,
        "groups": [
            {"layers": g.layer_names, "spec": asdict(g.kv_cache_spec)}
            for g in cache_config.kv_cache_groups
        ],
        "physical_pages": [
            (t.shared_by, t.size // cache_config.num_blocks)
            for t in cache_config.kv_cache_tensors
        ],
        "layer_scales": "verified-fixed-k=v=1.0",
        "state": "prompt-only-exact-committed-mamba-boundary",
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).digest()


def parking_hash(namespace: bytes, generation: int, block_hash: bytes) -> bytes:
    return hashlib.sha256(
        namespace + generation.to_bytes(8, "big") + block_hash
    ).digest()


@contextmanager
def bind_memory_node(node: int):
    """Bind only this thread's allocation policy, then restore it exactly.

    Fail before pinning if Linux refuses the policy. First touch is performed
    by anonymous mmap MADV_POPULATE_WRITE inside this context. No process-wide CPU
    affinity or existing-memory migration is performed.
    """
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    mode = ctypes.c_int()
    old_mask = ctypes.c_ulong()
    lib.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    lib.get_mempolicy.restype = ctypes.c_int
    lib.set_mempolicy.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
    ]
    lib.set_mempolicy.restype = ctypes.c_int
    if lib.get_mempolicy(ctypes.byref(mode), ctypes.byref(old_mask), 64, None, 0):
        raise OSError(ctypes.get_errno(), "get_mempolicy before host parking")
    mask = ctypes.c_ulong(1 << node)
    if lib.set_mempolicy(2, ctypes.byref(mask), 64):  # MPOL_BIND
        raise OSError(ctypes.get_errno(), "bind host parking pool")
    try:
        yield
    finally:
        restored_mask = ctypes.byref(old_mask) if old_mask.value else None
        if lib.set_mempolicy(mode.value, restored_mask, 64):
            raise OSError(ctypes.get_errno(), "restore host parking memory policy")
