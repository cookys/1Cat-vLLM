# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2 (A2-3): N_tail_max arithmetic and per-rank config rewrite,
CPU-only."""

import pytest

from vllm.v1.core.kv_cache_utils import (
    apply_elastic_kv_tail,
    elastic_tail_layout,
    generate_scheduler_kv_cache_config,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MIB = 1 << 20
GIB = 1 << 30
PAGE = 1179648  # 1.125 MiB


def cfg(num_blocks, group_size=16, page=PAGE):
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=page * num_blocks, shared_by=[f"l{i}"])
            for i in range(group_size)
        ],
        kv_cache_groups=[],
    )


def test_tail_layout_matches_brief_formula_and_16_block_boundary():
    # 1.1675 GiB of drafter weights, 16 tensors, 1.125 MiB page, 2 MiB gran.
    draft = int(1.1675 * GIB)
    boundary, tail = elastic_tail_layout([PAGE] * 16, draft, 2 * MIB)
    assert boundary == 16
    raw = draft // (16 * PAGE)
    assert raw == 66 and tail == 64  # floor(66 / 16) * 16
    assert tail % boundary == 0 and tail * 16 * PAGE <= draft


def test_tail_layout_rounds_down_never_up():
    # exactly one boundary's worth of bytes -> 16 blocks; one byte less -> 0
    one = 16 * 16 * PAGE
    assert elastic_tail_layout([PAGE] * 16, one, 2 * MIB)[1] == 16
    assert elastic_tail_layout([PAGE] * 16, one - 1, 2 * MIB)[1] == 0


@pytest.mark.parametrize(
    "gran,expected_boundary",
    [(2 * MIB, 16), (4 * MIB, 32), (1 * MIB, 8), (PAGE, 1)],
)
def test_boundary_is_lcm_of_page_and_granularity(gran, expected_boundary):
    boundary, _ = elastic_tail_layout([PAGE], 0, gran)
    assert boundary == expected_boundary


def test_tail_layout_mixed_pages_uses_lcm_and_sum():
    p1, p2 = PAGE, 2 * MIB  # p2 already granularity aligned -> boundary 1
    boundary, tail = elastic_tail_layout([p1, p2], 10 * GIB, 2 * MIB)
    assert boundary == 16
    assert tail == (10 * GIB // (p1 + p2)) // 16 * 16


def test_apply_rewrites_nspec_sizes_and_max_size():
    cfgs = [cfg(1000), cfg(1000)]
    tail = apply_elastic_kv_tail(cfgs, [int(1.1675 * GIB)] * 2, 2 * MIB)
    assert tail == 64
    for c in cfgs:
        assert c.num_blocks == 992  # 1000 rounded down to a multiple of 16
        assert c.elastic_tail_blocks == 64 and c.elastic_boundary_blocks == 16
        for t in c.kv_cache_tensors:
            assert t.size == PAGE * 992
            assert t.max_size == PAGE * (992 + 64)
            assert t.size % (2 * MIB) == 0  # N_spec ends on a granularity edge


def test_apply_takes_the_poorest_rank():
    cfgs = [cfg(1024), cfg(1024)]
    rich, poor = int(1.2 * GIB), int(0.6 * GIB)
    tail = apply_elastic_kv_tail(cfgs, [rich, poor], 2 * MIB)
    assert tail == (poor // (16 * PAGE)) // 16 * 16
    assert cfgs[0].elastic_tail_blocks == cfgs[1].elastic_tail_blocks == tail


def test_apply_without_enough_draft_bytes_leaves_configs_untouched():
    cfgs = [cfg(1000)]
    assert apply_elastic_kv_tail(cfgs, [10 * MIB], 2 * MIB) == 0
    assert cfgs[0].num_blocks == 1000
    assert cfgs[0].kv_cache_tensors[0].max_size is None
    assert cfgs[0].elastic_tail_blocks == 0


def test_scheduler_config_keeps_nspec():
    cfgs = [cfg(1000), cfg(1000)]
    apply_elastic_kv_tail(cfgs, [GIB] * 2, 2 * MIB)
    sched = generate_scheduler_kv_cache_config(cfgs)
    assert sched.num_blocks == cfgs[0].num_blocks == 992


def test_defaults_are_not_elastic():
    c = cfg(100)
    assert c.elastic_tail_blocks == 0 and c.kv_cache_tensors[0].max_size is None
