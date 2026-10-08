# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
from collections import Counter

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadingManager, get_offload_group_idx

logger = init_logger(__name__)


class ParkingManager(OffloadingManager):
    """Do not retry a failed snapshot; let native LRU reclaim its slots.

    In-flight readers keep their original refcounts until their own events
    finish. Invalidating never frees storage that a copy could still access.
    The tombstones are bounded by the underlying pool, and removed on eviction.
    """

    def __init__(self, inner, page_sizes=None, diagnostics=False):
        self.inner = inner
        self.invalid_keys = set()
        self.sweep_count = self.sweep_keys = self.sweep_ns = 0
        self.page_sizes = page_sizes or {}
        self.invalid_lookup_count = 0
        self.diagnostics = diagnostics
        self._lookup_samples = {}
        self._evictions = Counter()
        self._failed_reservations = 0
        self._sample_step = 0

    def invalidate(self, keys):
        self.invalid_keys.update(keys)

    def lookup(self, key, req_context):
        if key in self.invalid_keys:
            self.invalid_lookup_count += 1
            logger.info(
                "HOST_PARKING invalid_lookup_miss count=%d", self.invalid_lookup_count
            )
            result = False
            reason = "invalid"
        else:
            result = self.inner.lookup(key, req_context)
            reason = "absent" if result is False else "pending"
        if self.diagnostics and result is not True:
            group = get_offload_group_idx(key)
            manager = getattr(self.inner, "managers", {0: self.inner})[group]
            sample = self._lookup_samples.setdefault(
                (group, reason),
                dict(
                    group=group,
                    reason=reason,
                    count=0,
                    capacity_slots=manager._num_blocks,
                    used_slots_at_first_lookup=(
                        manager._num_blocks - manager._get_num_free_blocks()
                    ),
                ),
            )
            sample["count"] += 1
        return result

    def on_new_request(self, req_context):
        return self.inner.on_new_request(req_context)

    def on_request_finished(self, req_context):
        return self.inner.on_request_finished(req_context)

    def prepare_load(self, keys, req_context):
        if self.invalid_keys.intersection(keys):
            raise ValueError("cannot load an invalid parking snapshot")
        return self.inner.prepare_load(keys, req_context)

    def complete_load(self, keys, req_context):
        return self.inner.complete_load(keys, req_context)

    def prepare_store(self, keys, req_context):
        output = self.inner.prepare_store(keys, req_context)
        if self.diagnostics and output is not None:
            self._evictions.update(
                get_offload_group_idx(k) for k in output.evicted_keys
            )
        elif self.diagnostics:
            self._failed_reservations += 1
        # A grouped reservation can evict from one group, then roll back when
        # another group is full and return None (no evicted_keys available).
        # Sweep rare failure tombstones against the authoritative native pool.
        if self.invalid_keys:
            started = time.perf_counter_ns()
            self.sweep_count += 1
            self.sweep_keys += len(self.invalid_keys)
            self.invalid_keys = {
                k
                for k in self.invalid_keys
                if self.inner.lookup(k, req_context) is not False
            }
            self.sweep_ns += time.perf_counter_ns() - started
        return output

    def complete_store(self, keys, req_context, success=True):
        self.inner.complete_store(keys, req_context, success)
        if not success:
            self.invalid_keys.difference_update(keys)

    def touch(self, keys, req_context):
        return self.inner.touch(keys, req_context)

    def take_events(self):
        return self.inner.take_events()

    def reset_cache(self):
        self.inner.reset_cache()
        self.invalid_keys.clear()
        logger.info("HOST_PARKING quota %s", json.dumps(self.stats(), sort_keys=True))

    def stats(self):
        groups = getattr(self.inner, "managers", {0: self.inner})
        slots = {g: m._num_blocks - m._get_num_free_blocks() for g, m in groups.items()}
        # LRU is required by ParkingConfig; include in-flight ref counts too.
        refs = sum(
            b.ref_cnt for m in groups.values() for b in m._policy.blocks.values()
        )
        return {
            "in_use_slots": sum(slots.values()),
            "group_slots": slots,
            "group_capacity_slots": {g: m._num_blocks for g, m in groups.items()},
            "in_use_bytes_per_rank": sum(slots[g] * self.page_sizes[g] for g in slots)
            if self.page_sizes
            else None,
            "read_write_refs": refs,
            "invalid_keys": len(self.invalid_keys),
        }

    def sample_stats(self):
        """Diagnostic-only scheduler-step sample; never logs keys or request IDs.

        Miss occupancy is captured at lookup time, before that step's stores.
        Final occupancy includes those stores. Normal execution does no scan.
        """
        if not self.diagnostics:
            return
        self._sample_step += 1
        logger.info(
            "HOST_PARKING step_stats %s",
            json.dumps(
                dict(
                    step=self._sample_step,
                    **self.stats(),
                    lookup_misses=list(self._lookup_samples.values()),
                    reported_evicted_slots_since_sample=dict(self._evictions),
                    failed_reservations_since_sample=self._failed_reservations,
                ),
                sort_keys=True,
            ),
        )
        self._lookup_samples.clear()
        self._evictions.clear()
        self._failed_reservations = 0

    def shutdown(self):
        logger.info(
            "HOST_PARKING tombstone_sweep count=%d keys=%d cpu_ns=%d remaining=%d",
            self.sweep_count,
            self.sweep_keys,
            self.sweep_ns,
            len(self.invalid_keys),
        )
        self.inner.shutdown()
        self.invalid_keys.clear()
