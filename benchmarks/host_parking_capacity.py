# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU quota arithmetic from a native HostParkingSpec; allocates no KV tensors.

This is an independent-prefix, completed-32K-producer scenario, not replay of
GPU scheduling. It preserves the distinction between native sparse GDN states
and historical draft SW suffixes. Tests exercise native managers separately.
"""


def analyze(spec, requests=10, prompt_tokens=32768):
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        SchedulerOffloadConfig,
    )
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    cfg = SchedulerOffloadConfig.from_spec(spec)
    grids = {
        g.offloaded_block_size
        for g in cfg.kv_group_configs
        if g.sliding_window_size_in_blocks is None
    }
    if len(grids) != 1 or requests <= 0:
        raise ValueError("require one full-attention grid and positive request count")
    alignment = grids.pop()
    if prompt_tokens < 2 * alignment or prompt_tokens % alignment:
        raise ValueError("audit expects an aligned prompt of at least two pages")
    boundaries = KVCacheCoordinator.get_replay_boundaries(
        SimpleNamespace(eagle_group_ids=set()),
        SimpleNamespace(num_tokens=prompt_tokens),
        alignment,
    )
    restore = (prompt_tokens - 1) // alignment * alignment
    groups = []
    for group in cfg.kv_group_configs:
        g = group.group_idx
        pages = prompt_tokens // group.offloaded_block_size
        if group.requires_exact_boundary_source:
            mask = MambaManager.reachable_block_mask(
                0,
                pages,
                alignment,
                spec.kv_cache_config.kv_cache_groups[g].kv_cache_spec,
                spec.retention_interval,
                boundaries,
            )
            per_request = pages if mask is None else sum(mask)
            restore_pages = 1
            kind = "GDN"
        elif group.sliding_window_size_in_blocks is not None:
            tail = group.sliding_window_size_in_blocks
            segment = group.alignment_block_count
            per_request = sum(
                segment is None or i % segment >= segment - tail for i in range(pages)
            )
            restore_pages = tail
            kind = "draft_SW"
        else:
            per_request = pages
            restore_pages = restore // group.offloaded_block_size
            kind = "full_attention"
        groups.append(
            dict(
                group=g,
                kind=kind,
                block_tokens=group.offloaded_block_size,
                bytes_per_slot_per_rank=spec.cpu_group_page_sizes[g],
                allocated_slots=spec.cpu_group_num_blocks[g],
                producer_history_slots=requests * per_request,
                final_restore_slots=requests * restore_pages,
                deficit_slots=max(
                    0, requests * per_request - spec.cpu_group_num_blocks[g]
                ),
            )
        )
    world = spec.vllm_config.parallel_config.world_size

    def total(field):
        return world * sum(g[field] * g["bytes_per_slot_per_rank"] for g in groups)

    return dict(
        requests=requests,
        prompt_tokens=prompt_tokens,
        restore_boundary=restore,
        tp=world,
        retention_interval=spec.retention_interval,
        groups=groups,
        allocated_bytes=total("allocated_slots"),
        producer_history_bytes=total("producer_history_slots"),
        final_restore_bytes=total("final_restore_slots"),
        logical_capacity_fits_budget=total("producer_history_slots")
        <= spec.parking.total_bytes,
        current_group_quotas_fit=not any(g["deficit_slots"] for g in groups),
        assumptions=[
            f"{requests} independent prompts; no shared hashes, old sessions or "
            "in-flight headroom",
            "DFlash2, no EAGLE drop; sparse GDN handoffs, not every boundary",
            "SW tail at every completed attention segment stays until LRU eviction",
            "Group fit is stricter than total-byte fit; allocation unchanged",
        ],
    )
