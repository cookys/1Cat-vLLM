# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M2 (A2-2): ElasticKvRegion over FakeVmmBackend, CPU-only."""

import pytest

from vllm.device_allocator.elastic_region import ElasticKvRegion
from vllm.device_allocator.vmm import FakeVmmBackend, VmmError, VmmReleaseError

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MIB = 1 << 20
PAGE = 1179648  # 1.125 MiB
BOUNDARY = 18 * MIB  # lcm(1.125 MiB, 2 MiB) = 16 blocks


def make(n_spec_chunks=2, tail_chunks=3, **kw):
    be = kw.pop("backend", None) or FakeVmmBackend()
    n_spec = n_spec_chunks * BOUNDARY
    r = ElasticKvRegion(
        0, n_spec + tail_chunks * BOUNDARY, n_spec, BOUNDARY, PAGE, backend=be, **kw
    )
    return r, be


def test_boundary_is_16_blocks_of_18_mib():
    assert BOUNDARY // PAGE == 16 and BOUNDARY % (2 * MIB) == 0


def test_init_reserves_nmax_maps_nspec_and_zeroes_it():
    r, be = make()
    assert be.ops("reserve") == [("reserve", 5 * BOUNDARY, 0)]
    assert r.mapped_bytes == 2 * BOUNDARY
    assert len(be.ops("map")) == 1
    assert be.ops("memset") == [("memset", r.va, 0, 2 * BOUNDARY, 0)]
    assert r.spec_blocks == 32 and r.max_blocks == 80 and r.mapped_blocks == 32


def test_cuda_array_interface_covers_nmax():
    r, _ = make()
    d = r.__cuda_array_interface__
    assert d["shape"] == (5 * BOUNDARY,) and d["typestr"] == "|i1"
    assert d["data"] == (r.va, False)


def test_grow_maps_boundary_chunks_aligned_and_zeroes_each():
    r, be = make()
    assert r.grow_to(2 * BOUNDARY + 1) == 3 * BOUNDARY  # rounds up to a chunk edge
    maps = be.ops("map")[1:]
    assert [m[2] for m in maps] == [2 * BOUNDARY] and maps[0][3] == BOUNDARY
    assert all(m[2] % (2 * MIB) == 0 and m[3] % (2 * MIB) == 0 for m in be.ops("map"))
    assert be.ops("memset")[-1] == ("memset", r.va, 2 * BOUNDARY, BOUNDARY, 0)


def test_grow_is_idempotent_and_capped_at_nmax():
    r, be = make()
    r.grow_to(5 * BOUNDARY)
    n = len(be.calls)
    assert r.grow_to(4 * BOUNDARY) == 5 * BOUNDARY
    assert r.grow_to(5 * BOUNDARY) == 5 * BOUNDARY
    assert len(be.calls) == n
    with pytest.raises(ValueError):
        r.grow_to(5 * BOUNDARY + 1)


def test_shrink_never_below_nspec_and_syncs_first():
    r, be = make()
    r.grow_to(5 * BOUNDARY)
    be.calls.clear()
    assert r.shrink_to(3 * BOUNDARY) == 3 * BOUNDARY
    assert [c[0] for c in be.calls] == ["synchronize", "unmap", "unmap"]
    assert r.shrink_to(0) == 2 * BOUNDARY  # floor = n_spec
    assert r.shrink_to(0) == 2 * BOUNDARY  # idempotent
    assert be.mapped_total() == 2 * BOUNDARY


def test_shrink_noop_does_not_sync():
    r, be = make()
    be.calls.clear()
    r.shrink_to(0)
    assert be.calls == []


def test_regrow_after_shrink_remaps_the_same_va():
    r, be = make()
    va = r.va
    r.grow_to(5 * BOUNDARY)
    r.shrink_to(0)
    r.grow_to(4 * BOUNDARY)
    assert r.va == va and r.mapped_bytes == 4 * BOUNDARY
    assert len(be.ops("reserve")) == 1  # VA reserved once, never re-reserved


def test_failed_grow_rolls_back_to_previous_state():
    be = FakeVmmBackend()
    r, _ = make(backend=be)
    be.fail_map_after = 3  # init map + 2 tail chunks ok, third tail chunk fails
    with pytest.raises(VmmError):
        r.grow_to(5 * BOUNDARY)
    assert r.mapped_bytes == 2 * BOUNDARY
    assert be.mapped_total() == 2 * BOUNDARY


def test_close_unmaps_and_releases():
    r, be = make()
    r.grow_to(4 * BOUNDARY)
    r.close()
    assert be.mapped_total() == 0 and be.reservations == {}


@pytest.mark.parametrize(
    "n_max,n_spec,boundary,block",
    [
        (5 * BOUNDARY, 2 * BOUNDARY, 3 * MIB, PAGE),  # boundary not a granularity multiple
        (5 * BOUNDARY, 2 * BOUNDARY, BOUNDARY, 4 * MIB),  # boundary not a block multiple
        (5 * BOUNDARY, BOUNDARY // 2, BOUNDARY, PAGE),  # n_spec not a boundary multiple
        (BOUNDARY, 2 * BOUNDARY, BOUNDARY, PAGE),  # n_max < n_spec
        (5 * BOUNDARY + MIB, 2 * BOUNDARY, BOUNDARY, PAGE),  # tail not whole chunks
    ],
)
def test_invalid_geometry_is_rejected(n_max, n_spec, boundary, block):
    with pytest.raises(ValueError):
        ElasticKvRegion(0, n_max, n_spec, boundary, block, backend=FakeVmmBackend())


def test_non_2mib_granularity_changes_the_boundary_requirement():
    # g = 4 MiB (hypothetical): lcm(1.125 MiB, 4 MiB) = 36 MiB = 32 blocks;
    # an 18 MiB boundary is no longer legal.
    with pytest.raises(ValueError):
        ElasticKvRegion(
            0, 5 * BOUNDARY, 2 * BOUNDARY, BOUNDARY, PAGE,
            backend=FakeVmmBackend(granularity=4 * MIB),
        )
    ElasticKvRegion(
        0, 5 * 2 * BOUNDARY, 2 * 2 * BOUNDARY, 2 * BOUNDARY, PAGE,
        backend=FakeVmmBackend(granularity=4 * MIB),
    )


def test_memset_failure_during_grow_rolls_back_and_retry_zeroes():
    be = FakeVmmBackend()
    r, _ = make(backend=be)
    be.fail_memset_after = 1  # first tail chunk ok, second chunk's memset fails
    with pytest.raises(VmmError, match="injected memset"):
        r.grow_to(5 * BOUNDARY)
    assert r.mapped_bytes == 2 * BOUNDARY
    assert be.mapped_total() == 2 * BOUNDARY
    be.fail_memset_after = None
    before = len(be.ops("memset"))
    assert r.grow_to(4 * BOUNDARY) == 4 * BOUNDARY
    ms = be.ops("memset")[before:]
    assert [m[2] for m in ms] == [2 * BOUNDARY, 3 * BOUNDARY]  # every new chunk zeroed


def test_memset_failure_in_init_keeps_original_error_and_frees_va():
    be = FakeVmmBackend()
    be.fail_memset_after = 0
    with pytest.raises(VmmError, match="injected memset"):
        ElasticKvRegion(0, 5 * BOUNDARY, 2 * BOUNDARY, BOUNDARY, PAGE, backend=be)
    assert be.mapped_total() == 0 and be.reservations == {}


def test_release_failure_after_unmap_does_not_resurrect_chunk():
    r, be = make()
    r.grow_to(4 * BOUNDARY)
    be.fail_release_on_unmap = True
    with pytest.raises(VmmReleaseError):
        r.shrink_to(0)
    be.fail_release_on_unmap = False
    assert r.mapped_bytes == 3 * BOUNDARY == be.mapped_total()
    assert r.shrink_to(0) == 2 * BOUNDARY  # retry does not re-unmap a dead range


def test_startup_and_grow_synchronize_after_memset():
    r, be = make()
    names = [c[0] for c in be.calls]
    assert names.index("synchronize") > names.index("memset")
    n = len(be.ops("synchronize"))
    r.grow_to(3 * BOUNDARY)
    assert len(be.ops("synchronize")) == n + 1
