# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""27B NVFP4 + DFlash2 native-connector parking proof of concept."""

import fcntl
import json
import os
from pathlib import Path

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.kv_offload.cpu.parking import (
    ParkingConfig,
    bind_memory_node,
    layout_namespace,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

logger = init_logger(__name__)


class HostParkingSpec(CPUOffloadingSpec):
    """Explicit spec selection AND host_parking=true are required.

    Uses native group pools/LRU and committed boundary stores. No disk tier,
    quantization changes, or speculative recurrent-state snapshots.
    """

    host_parking = True

    def __init__(self, vllm_config, kv_cache_config):
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.parking = ParkingConfig.parse(extra)
        extra.setdefault("cpu_bytes_to_use", self.parking.total_bytes)
        pc = vllm_config.parallel_config
        if pc.world_size != 4 or pc.tensor_parallel_size != 4:
            raise ValueError("parking PoC requires one TP4 replica")
        if getattr(pc, "data_parallel_size", 1) != 1:
            raise ValueError("parking PoC does not support DP replicas")
        cc = vllm_config.cache_config
        if cc.cache_dtype != "nvfp4" or not cc.enable_prefix_caching:
            raise ValueError("parking PoC requires NVFP4 and prefix caching")
        if cc.calculate_kv_scales:
            raise ValueError("parking PoC requires fixed layer KV scales")
        if vllm_config.scheduler_config.mixed_prefill_step_latency_ms != 0:
            raise ValueError("parking baseline requires P8 disabled")
        sc = vllm_config.speculative_config
        if sc is None or not sc.use_dflash():
            raise ValueError("parking PoC requires DFlash2")
        if vllm_config.kv_transfer_config.kv_load_failure_policy != "recompute":
            raise ValueError("parking requires kv_load_failure_policy=recompute")
        groups = kv_cache_config.kv_cache_groups
        if not any(isinstance(g.kv_cache_spec, MambaSpec) for g in groups):
            raise ValueError("parking PoC requires hybrid committed GDN states")
        for g in groups:
            if not isinstance(g.kv_cache_spec, (AttentionSpec, MambaSpec)):
                raise ValueError("unsupported parking cache spec")
            if (
                isinstance(g.kv_cache_spec, MambaSpec)
                and g.kv_cache_spec.mamba_cache_mode != "align"
            ):
                raise ValueError("parking requires Mamba align mode")
        state_grids = {
            g.kv_cache_spec.block_size
            for g in groups
            if isinstance(g.kv_cache_spec, MambaSpec)
        }
        if len(state_grids) != 1 or any(
            next(iter(state_grids)) % g.kv_cache_spec.block_size for g in groups
        ):
            raise ValueError("parking requires one state grid divisible by group grids")
        super().__init__(vllm_config, kv_cache_config)
        if self.block_size_factor != 1:
            raise ValueError("parking requires native per-group block granularity")
        if not self.partition_by_group:
            raise ValueError("parking requires native disjoint grouped CPU pools")
        self.key_namespace = layout_namespace(vllm_config, kv_cache_config)
        self.actual_pool_bytes = pc.world_size * sum(
            self.cpu_group_page_sizes[g] * n
            for g, n in self.cpu_group_num_blocks.items()
        )
        if not 0 < self.actual_pool_bytes <= self.parking.total_bytes:
            raise ValueError("parking physical pool exceeds aggregate budget")
        logger.info(
            "HOST_PARKING enabled mode=completed_boundary total_bytes=%d "
            "allocated_bytes=%d numa=%d group_slots=%s namespace=%s P8=off",
            self.parking.total_bytes,
            self.actual_pool_bytes,
            self.parking.numa_node,
            self.cpu_group_num_blocks,
            self.key_namespace.hex(),
        )

    def get_manager(self):
        from vllm.v1.kv_offload.cpu.parking_manager import ParkingManager

        manager = super().get_manager()
        if not isinstance(manager, ParkingManager):
            self._manager = ParkingManager(manager, self.cpu_group_page_sizes)
        return self._manager

    def validate_layers(self):
        layers = self.vllm_config.compilation_config.static_forward_context
        for group in self.kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, AttentionSpec):
                continue
            for name in group.layer_names:
                layer = layers[name]
                # Startup only; verify both kernel-side and host-side scale
                # representations before any entry can become reusable.
                for attr in (
                    "_k_scale_float",
                    "_v_scale_float",
                    "_k_scale",
                    "_v_scale",
                ):
                    value = getattr(layer, attr)
                    if float(value) != 1.0:
                        raise ValueError(f"parking fixed scale violated: {name}.{attr}")
                if getattr(layer, "calculate_kv_scales", False):
                    raise ValueError("parking does not allow dynamic KV scales")

    def create_handlers(self, kv_caches):
        from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
        from vllm.v1.kv_offload.cpu.parking_pool import ParkingPool

        if not is_pin_memory_available():
            raise RuntimeError("parking requires pinned memory; no pageable fallback")
        self.validate_layers()
        # The engine ID makes the lock private to this replica's TP workers.
        engine_id = self.vllm_config.kv_transfer_config.engine_id
        lock = Path("/data/tmp") / (
            "host-parking-"
            + self.key_namespace.hex()[:16]
            + "-"
            + str(engine_id).replace("/", "_")
            + ".lock"
        )
        with lock.open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            with bind_memory_node(self.parking.numa_node):
                pool = ParkingPool(self.actual_pool_bytes // 4)
            try:
                handlers = CpuGpuOffloadingHandlers(
                    kv_caches,
                    self.block_size_factor,
                    self.num_blocks,
                    group_page_sizes=self.cpu_group_page_sizes,
                    group_num_blocks=self.cpu_group_num_blocks,
                    cpu_tensor_factory=pool.allocate,
                )
                logger.info(
                    "HOST_PARKING pool_usage %s",
                    json.dumps({"pid": os.getpid(), **pool.stats()}, sort_keys=True),
                )
            except Exception:
                pool.close()
                raise
        from vllm.v1.kv_offload.cpu.parking_transfer import (
            RecoveringHandler,
            ValidationFaultHandler,
        )

        fault = self.extra_config.get("parking_test_fail_direction")
        if fault:
            from vllm.distributed import get_tensor_model_parallel_rank

            if get_tensor_model_parallel_rank() != self.extra_config.get(
                "parking_test_fail_rank", 0
            ):
                fault = None
        for name, direction in (
            ("gpu_to_cpu_handler", "GPU_to_CPU"),
            ("cpu_to_gpu_handler", "CPU_to_GPU"),
        ):
            inner = getattr(handlers, name)
            if fault == direction:
                inner = ValidationFaultHandler(
                    inner,
                    self.extra_config["parking_test_fail_mode"],
                    nth=self.extra_config.get("parking_test_fail_nth", 1),
                )
            setattr(handlers, name, RecoveringHandler(inner, pool))
        return handlers
