# SPDX-License-Identifier: Apache-2.0
"""plan 075 S3 M-fix-10 (X2R-154): the nospec grow keeps, per rank, at least
the startup-profiled forward activation peak + 64 MiB free (M-fix-9 kept only
64 MiB and the first post-grow forward OOMed with ~130 MiB left)."""

from types import SimpleNamespace

import pytest

from tests.v1.core.test_s2_elastic_layout import cfg
from tests.v1.core.test_s2_spec_machine import (
    BOUNDARY, DRAFT, GIB, MARGIN, MIB, N_SPEC, PAGE, TAIL, FakeRank, make,
)
from vllm.v1.core.kv_cache_utils import (
    ELASTIC_GROW_MARGIN_BYTES,
    elastic_grow_headroom,
    elastic_grow_target,
)

PEAK = 700 * MIB  # ~ the 0.69 GiB per-rank peak activation of the 27B NVFP4 startup
CHUNK = BOUNDARY * PAGE


def test_margin_constant_unchanged():
    assert ELASTIC_GROW_MARGIN_BYTES == 64 << 20 == MARGIN


def test_headroom_is_min_over_ranks_of_free_minus_peak():
    assert elastic_grow_headroom([10, 10, 10, 10], [1, 2, 3, 4]) == 6
    # the rank with the least free is not the one that decides when its peak is smaller
    assert elastic_grow_headroom([100, 50, 100, 100], [1, 40, 1, 1]) == 10
    assert elastic_grow_headroom([100, 90], [1, 80]) == 10
    assert elastic_grow_headroom([5, 5], [9, 9]) == 0  # never negative
    assert elastic_grow_headroom([5, 7], None) == 5  # unknown peaks count as 0
    assert elastic_grow_headroom([5, 7], []) == 5
    assert elastic_grow_headroom([], [1]) == 0
    with pytest.raises(ValueError):
        elastic_grow_headroom([1, 2, 3, 4], [1, 2])


def test_target_boundary_peak_plus_margin_exactly():
    k = 3
    free = k * CHUNK + PEAK + MARGIN
    h = elastic_grow_headroom([free] * 4, [PEAK] * 4)
    assert elastic_grow_target(1696, 1904, 16, CHUNK // 16, h, MARGIN) == 1696 + k * 16
    h = elastic_grow_headroom([free - 1] * 4, [PEAK] * 4)
    assert elastic_grow_target(1696, 1904, 16, CHUNK // 16, h, MARGIN) == 1696 + (k - 1) * 16


def _grow_machine(frees_after_release, peaks):
    core, cl, s = make(free=0)
    cl.ranks = [FakeRank(f - DRAFT) for f in frees_after_release]  # sleep adds DRAFT back
    core._spec_peak_activation_bytes = list(peaks)
    return core, cl, s


def test_machine_exact_peak_plus_margin_and_one_byte_short():
    k = 2
    free = k * CHUNK + PEAK + MARGIN
    core, cl, s = _grow_machine([free] * 4, [PEAK] * 4)
    assert core.set_spec_mode("nospec")["state"] == "NOSPEC_GROWN"
    g = core.get_spec_mode()["last_transition"]["steps"]["grow"]
    assert g["target_blocks"] == N_SPEC + k * BOUNDARY
    assert g["peak_activation_bytes"] == [PEAK] * 4 and g["headroom_bytes"] == free - PEAK
    assert g["n_tail_grow"] == k * BOUNDARY and g["min_rank_free_bytes"] == free
    # after the grow every rank still has >= peak + margin free
    assert all(f >= PEAK + MARGIN for f in g["free_bytes"])
    assert min(g["free_bytes"]) == PEAK + MARGIN
    # one byte short: one chunk fewer
    core, cl, s = _grow_machine([free - 1] * 4, [PEAK] * 4)
    core.set_spec_mode("nospec")
    g = core.get_spec_mode()["last_transition"]["steps"]["grow"]
    assert g["target_blocks"] == N_SPEC + (k - 1) * BOUNDARY
    assert all(f >= PEAK + MARGIN for f in g["free_bytes"])


def test_w2a_f5_shape_m_fix_9_vs_m_fix_10():
    # numbers from the W2a v2 F5 cell (out-w2a/01-F5-nospec-c16/get-grown.json):
    # page 9437184 B, min_rank_free before the grow 9492496384 B, N_spec 1696,
    # N_tail_max 1904, boundary 16. M-fix-9 target = 2688 (left ~130 MiB, forward OOM);
    # peak 0.69 GiB per rank is the S3-C startup profiling of the same config.
    page, free, peak = 9437184, 9492496384, int(0.69 * GIB)
    t9 = elastic_grow_target(1696, 1904, 16, page, free, MARGIN)
    assert t9 == 2688
    assert free - (t9 - 1696) * page < peak
    h = elastic_grow_headroom([free] * 4, [peak] * 4)
    t10 = elastic_grow_target(1696, 1904, 16, page, h, MARGIN)
    assert t10 == 1696 + 912  # 57 chunks of 16 blocks
    assert free - (t10 - 1696) * page >= peak + MARGIN
    # and one more chunk would break the rule
    assert free - (t10 - 1696 + 16) * page < peak + MARGIN


def test_machine_four_ranks_the_poorest_after_peak_decides():
    free = 5 * CHUNK + PEAK + MARGIN
    frees = [free + 9 * CHUNK, free + 4 * CHUNK, free + 9 * CHUNK, free + 9 * CHUNK]
    peaks = [PEAK, PEAK, PEAK, PEAK + 3 * CHUNK]  # rank 3 has the biggest peak: 6 chunks left
    core, cl, s = _grow_machine(frees, peaks)
    core.set_spec_mode("nospec")
    g = core.get_spec_mode()["last_transition"]["steps"]["grow"]
    # rank0: 14 chunks left, rank1: 9, rank2: 14, rank3: 5+9-3 = 11 -> rank1 decides (floor 9 chunks > TAIL cap?)
    expect = min(TAIL // BOUNDARY, 9)
    assert g["target_blocks"] == N_SPEC + expect * BOUNDARY
    assert all(f >= p + MARGIN for f, p in zip(g["free_bytes"], peaks))


def test_machine_skip_when_peak_leaves_no_whole_chunk_then_retry():
    free = CHUNK + PEAK + MARGIN - 1
    core, cl, s = _grow_machine([free] * 2, [PEAK] * 2)
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_RELEASED" and "grow_kv_cache" not in cl.methods()
    assert "grow skipped" in core.get_spec_mode()["error"]
    gt = core.get_spec_mode()["last_transition"]["steps"]["grow_target"]
    assert gt["target_blocks"] == N_SPEC and gt["peak_activation_bytes"] == [PEAK] * 2
    for r in cl.ranks:
        r.free += 1
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_GROWN"
    assert s.kv_cache_manager.block_pool.num_gpu_blocks == N_SPEC + BOUNDARY


def test_machine_peak_rank_count_mismatch_fails_the_grow_not_the_engine():
    core, cl, s = _grow_machine([40 * GIB] * 4, [PEAK] * 2)
    out = core.set_spec_mode("nospec")
    assert out["state"] == "NOSPEC_RELEASED" and "grow_kv_cache" not in cl.methods()
    assert "activation peaks" in core.get_spec_mode()["error"]


def test_startup_stores_per_rank_peak_from_the_probe():
    from vllm.v1.engine.core import EngineCore

    core = EngineCore.__new__(EngineCore)
    probes = [
        {"draft_weights_bytes": GIB, "granularity": 2 * MIB, "total_vram_bytes": 32 * GIB,
         "peak_activation_bytes": PEAK + i} for i in range(4)
    ]
    core.collective_rpc = lambda m: probes
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=None),
        cache_config=SimpleNamespace(block_size=16),
    )
    cfgs = [cfg(1000) for _ in range(4)]
    core._make_kv_cache_configs_elastic(cfgs)
    assert core._spec_peak_activation_bytes == [PEAK + i for i in range(4)]
    assert cfgs[0].elastic_tail_blocks > 0
    # a worker that predates the field (or skipped profiling) reports 0
    probes2 = [{k: v for k, v in p.items() if k != "peak_activation_bytes"} for p in probes]
    core.collective_rpc = lambda m: probes2
    core._make_kv_cache_configs_elastic([cfg(1000) for _ in range(4)])
    assert core._spec_peak_activation_bytes == [0] * 4
