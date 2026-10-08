# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import OffloadingManager


class ParkingManager(OffloadingManager):
    """Do not retry a failed snapshot; let native LRU reclaim its slots.

    In-flight readers keep their original refcounts until their own events
    finish. Invalidating never frees storage that a copy could still access.
    The tombstones are bounded by the underlying pool, and removed on eviction.
    """

    def __init__(self, inner):
        self.inner = inner
        self.invalid_keys = set()

    def invalidate(self, keys):
        self.invalid_keys.update(keys)

    def lookup(self, key, req_context):
        if key in self.invalid_keys:
            return False
        return self.inner.lookup(key, req_context)

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
        # A grouped reservation can evict from one group, then roll back when
        # another group is full and return None (no evicted_keys available).
        # Sweep rare failure tombstones against the authoritative native pool.
        self.invalid_keys = {
            k
            for k in self.invalid_keys
            if self.inner.lookup(k, req_context) is not False
        }
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

    def shutdown(self):
        self.inner.shutdown()
        self.invalid_keys.clear()
