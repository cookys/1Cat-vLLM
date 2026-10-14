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


def test_alignment_below_max_model_len_fails_at_startup_with_both_numbers():
    # 1000 blocks x 16 tokens = 16000 fit max_model_len; aligned 992 x 16 = 15872 do not
    cfgs = [cfg(1000), cfg(1000)]
    with pytest.raises(ValueError) as e:
        apply_elastic_kv_tail(
            cfgs, [GIB] * 2, 2 * MIB, max_model_len=16000, block_size=16
        )
    msg = str(e.value)
    assert "992" in msg and "1000" in msg and "16000" in msg
    assert cfgs[0].num_blocks == 1000 and cfgs[0].elastic_tail_blocks == 0  # untouched
    # exactly enough after alignment: fine
    assert apply_elastic_kv_tail(
        cfgs, [GIB] * 2, 2 * MIB, max_model_len=15872, block_size=16
    ) > 0 and cfgs[0].num_blocks == 992


# ---- plan 075 S3 M-fix-9 (X2R-26): VA for the whole card, grow by free memory
from vllm.device_allocator.elastic_region import ElasticKvRegion  # noqa: E402
from vllm.device_allocator.vmm import FakeVmmBackend  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    ELASTIC_GROW_MARGIN_BYTES,
    elastic_grow_target,
    elastic_n_total,
)

PAGE_SUM = 8 * PAGE  # 9 MiB per block over 8 KV tensors
TOTAL = int(31.65 * GIB)


def test_n_total_is_floor_of_total_over_page_aligned_down_to_boundary():
    n = elastic_n_total(TOTAL, PAGE_SUM, 16)
    assert n % 16 == 0
    assert n * PAGE_SUM <= TOTAL < (n + 16) * PAGE_SUM
    assert 3590 <= n <= 3610  # X2R-35: ~3600
    assert elastic_n_total(16 * PAGE_SUM - 1, PAGE_SUM, 16) == 0  # never rounds up


def test_apply_v2_reserves_va_for_the_whole_card():
    cfgs = [cfg(1696, group_size=8), cfg(1696, group_size=8)]
    # a 1.17 GiB drafter pays only 64 blocks (v1), v2 must not be capped by it
    tail = apply_elastic_kv_tail(
        cfgs, [int(1.1675 * GIB)] * 2, 2 * MIB, total_vram_bytes=[TOTAL, TOTAL + GIB]
    )
    n_total = elastic_n_total(TOTAL, PAGE_SUM, 16)  # the poorest rank decides
    assert tail == n_total - 1696 and tail > 1800
    for c in cfgs:
        assert c.num_blocks == 1696 and c.elastic_tail_blocks == tail
        assert c.elastic_boundary_blocks == 16 and (c.num_blocks + tail) % 16 == 0
        for t in c.kv_cache_tensors:
            assert t.max_size == PAGE * n_total  # VA = page x N_total
    # the worker reserves exactly that much VA (no physical memory: FakeVmm)
    be = FakeVmmBackend()
    t = cfgs[0].kv_cache_tensors[0]
    ElasticKvRegion(0, t.max_size, t.size, 16 * PAGE, PAGE, backend=be)
    assert ("reserve", PAGE * n_total, 0) in be.calls


def test_apply_v2_keeps_the_drafter_gate_and_never_returns_negative_tail():
    cfgs = [cfg(1000)]
    assert apply_elastic_kv_tail(cfgs, [10 * MIB], 2 * MIB, total_vram_bytes=[TOTAL]) == 0
    assert cfgs[0].kv_cache_tensors[0].max_size is None
    # card smaller than N_spec: nothing to reserve
    cfgs = [cfg(1000, group_size=8)]
    assert apply_elastic_kv_tail(cfgs, [GIB], 2 * MIB, total_vram_bytes=[900 * PAGE_SUM]) == 0


@pytest.mark.parametrize(
    "free_blocks, expected_extra",
    [(10_000, 1904), (500, 496), (100, 96), (15, 0), (0, 0)],
)
def test_grow_target_is_min_of_tail_max_and_free_minus_margin(free_blocks, expected_extra):
    m = ELASTIC_GROW_MARGIN_BYTES
    free = free_blocks * PAGE_SUM + m + 1  # +1: strictly inside the last block
    t = elastic_grow_target(1696, 1904, 16, PAGE_SUM, free)
    assert t == 1696 + expected_extra
    assert (t - 1696) % 16 == 0 and 1696 <= t <= 1696 + 1904


def test_grow_target_free_below_margin_never_shrinks_below_nspec():
    assert elastic_grow_target(1696, 1904, 16, PAGE_SUM, 0) == 1696
    assert elastic_grow_target(1696, 1904, 16, PAGE_SUM, ELASTIC_GROW_MARGIN_BYTES - 1) == 1696
    assert ELASTIC_GROW_MARGIN_BYTES == 64 << 20


def test_grow_target_matches_x2r26_prediction_for_8_34_gib_free():
    t = elastic_grow_target(1696, 1904, 16, PAGE_SUM, int(8.34 * GIB))
    assert 2600 <= t <= 2640  # prereg predicts grown ~2624
