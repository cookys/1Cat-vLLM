# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2 (A2-5): BlockPool.grow / shrink_to / max_block_id fence,
CPU-only."""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def pool(n=20, caching=True):
    return BlockPool(n, caching, 16)


def cache(p, block, tag):
    h = make_block_hash_with_group_id(tag.to_bytes(8, "big"), 0)
    block.block_hash = h
    p.cached_block_hash_to_block.insert(h, block)
    return h


def free_ids(p):
    return [b.block_id for b in p.free_block_queue.get_all_free_blocks()]


# ---- grow ----


def test_grow_adds_consecutive_ids_at_free_queue_tail():
    p = pool(10)
    assert p.grow(16) == 6
    assert len(p.blocks) == 16 and p.num_gpu_blocks == 16
    assert [b.block_id for b in p.blocks[10:]] == list(range(10, 16))
    assert free_ids(p) == list(range(1, 16))  # null block 0 is not free
    assert p.get_num_free_blocks() == 15
    assert p.free_block_queue.num_free_blocks == 15


def test_grow_goes_behind_blocks_freed_later_in_use_order():
    p = pool(6)
    got = p.get_new_blocks(2)  # ids 1, 2
    p.grow(10)
    p.free_blocks(got)  # uncached -> prepended (front)
    assert free_ids(p)[:2] == [1, 2]
    assert free_ids(p)[-4:] == [6, 7, 8, 9]


def test_grow_is_idempotent_and_never_shrinks():
    p = pool(8)
    assert p.grow(8) == 0
    with pytest.raises(ValueError):
        p.grow(7)
    assert len(p.blocks) == 8


def test_get_usage_denominator_follows_grow():
    p = pool(11)  # 10 usable
    p.get_new_blocks(5)
    assert p.get_usage() == pytest.approx(0.5)
    p.grow(21)  # 20 usable
    assert p.get_usage() == pytest.approx(0.25)


def test_grown_blocks_are_allocatable_after_low_ids_run_out():
    p = pool(4)
    p.grow(8)
    ids = [b.block_id for b in p.get_new_blocks(7)]
    assert ids == [1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        p.get_new_blocks(1)


# ---- shrink_to ----


def test_shrink_removes_free_tail_and_fixes_counts():
    p = pool(10)
    p.grow(16)
    assert p.shrink_to(10) == 0
    assert len(p.blocks) == 10 and p.num_gpu_blocks == 10
    assert free_ids(p) == list(range(1, 10))
    assert p.get_num_free_blocks() == 9
    p.grow(16)  # a second elastic cycle works
    assert p.get_num_free_blocks() == 15


def test_shrink_evicts_cached_tail_blocks():
    p = pool(10)
    p.grow(16)
    got = p.get_new_blocks(15)  # take everything
    tail = [b for b in got if b.block_id >= 10]
    hashes = [cache(p, b, i) for i, b in enumerate(tail)]
    p.free_blocks(got)  # cached tail blocks go back to the queue
    assert p.shrink_to(10) == 0
    for h in hashes:
        assert p.cached_block_hash_to_block.get_one_block(h) is None
    assert p.get_num_free_blocks() == 9


def test_shrink_refuses_while_a_tail_block_is_referenced():
    p = pool(10)
    p.grow(16)
    got = p.get_new_blocks(15)
    p.free_blocks([b for b in got if b.block_id < 10])
    before = (len(p.blocks), p.get_num_free_blocks(), free_ids(p))
    holders = p.shrink_to(10)
    assert holders == 6  # blocks 10..15 still referenced
    assert (len(p.blocks), p.get_num_free_blocks(), free_ids(p)) == before


def test_shrink_never_removes_the_null_block():
    p = pool(10)
    with pytest.raises(ValueError):
        p.shrink_to(0)


# ---- fence ----


def test_fence_get_new_blocks_never_hands_out_tail():
    p = pool(8)
    p.grow(16)
    p.set_max_block_id(7)
    ids = [b.block_id for b in p.get_new_blocks(7)]
    assert ids == [1, 2, 3, 4, 5, 6, 7]
    assert p.get_num_free_blocks() == 0  # tail is free but fenced
    with pytest.raises(ValueError):
        p.get_new_blocks(1)
    assert p.free_block_queue.num_free_blocks == 8  # the 8 fenced blocks remain


def test_fence_skips_tail_blocks_sitting_at_the_queue_head():
    p = pool(8)
    p.grow(16)
    got = p.get_new_blocks(14)  # ids 1..14
    p.free_blocks(got[::-1])  # uncached -> prepend; order puts tail ids first
    assert free_ids(p)[:3] == [14, 13, 12]  # tail ids really are at the head
    p.set_max_block_id(7)
    assert p.get_num_free_blocks() == 7
    ids = [b.block_id for b in p.get_new_blocks(7)]
    assert sorted(ids) == [1, 2, 3, 4, 5, 6, 7]
    assert all(b.ref_cnt == 1 for b in p.blocks[1:8])
    assert all(b.ref_cnt == 0 for b in p.blocks[8:15])


def test_fence_counts_tail_holders_and_lifts():
    p = pool(8)
    p.grow(16)
    got = p.get_new_blocks(15)
    p.set_max_block_id(7)
    assert p.count_blocks_in_use_above(8) == 8
    p.free_blocks([b for b in got if b.block_id >= 8][:5])
    assert p.count_blocks_in_use_above(8) == 3
    p.set_max_block_id(None)
    assert p.max_block_id is None
    with pytest.raises(ValueError):
        p.set_max_block_id(16)


def test_shrink_after_fence_then_lift_resets_fence():
    p = pool(8)
    p.grow(16)
    p.set_max_block_id(7)
    assert p.shrink_to(8) == 0
    assert p.max_block_id is None
    assert p.get_num_free_blocks() == 7


def test_reset_prefix_cache_ignores_fence():
    p = pool(8)
    p.grow(16)
    p.set_max_block_id(7)
    assert p.reset_prefix_cache() is True  # nothing in use except the null block
