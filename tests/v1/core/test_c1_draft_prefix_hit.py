# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 C1 fix: K>0 requests read draft KV, so their prefix-cache hit must
(i) let the draft group take part (only DFlash2-era blocks can match) and
(ii) fall back to a main+Mamba hit capped at N - W so the target rewrites the
last W prompt tokens' draft KV. CPU-only, one test per S1 hot-switch invariant.

Geometry (test_spec_hot_switch.py): block 16, draft sliding window W = 32 (2 blocks).
"""

import pytest

from tests.v1.core.test_spec_hot_switch import B, NSPEC, mgr, new_req  # noqa: F401

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

W = 2 * B
DRAFT = 2  # kv cache group id of the draft sliding-window group in make_cfg()


def prefill(m, req, total, chunk=B):
    """Like test_spec_hot_switch.prefill but with the scheduler's accounting: a new
    request has num_computed_tokens == 0 when allocate_slots sees its hit (the shared
    helper pre-sets it to the hit length, which double-counts the hit)."""
    blocks, n_hit = m.get_computed_blocks(req)
    done = n_hit
    first = True
    while done < total:
        new = min(chunk, total - done)
        got = m.allocate_slots(
            req,
            new,
            num_new_computed_tokens=n_hit if first else 0,
            new_computed_blocks=blocks if first else None,
            num_lookahead_tokens=0 if req.spec_off else NSPEC,
        )
        assert got is not None
        first = False
        req.num_computed_tokens = done = done + new
    m.new_step_starts()
    return n_hit


def draft_cached(m, req, i):
    return m.block_pool.get_cached_block(req.block_hashes[i], [DRAFT]) is not None


def draft_pages(m, rid):
    return [b for b in m.coordinator.single_type_managers[DRAFT].req_to_blocks.get(rid, []) if not b.is_null]


def probe_hit(m, total, spec_off=False):
    r = new_req("probe", total, spec_off)
    _, n = m.get_computed_blocks(r)
    return n


def test_inv1_nospec_writes_no_draft_kv(mgr):  # noqa: F811
    total = 8 * B + 3
    prod = new_req("prod", total, True)
    prefill(mgr, prod, total)
    assert not draft_pages(mgr, "prod")
    assert not any(draft_cached(mgr, prod, i) for i in range(8))  # no draft hash entry at all


def test_inv2_cross_mode_still_hits_main_and_mamba(mgr):  # noqa: F811
    # nospec-era prefix, tail 3 < W: K>0 reader gets the capped main+Mamba hit,
    # K=0 reader gets the whole prefix.
    total = 8 * B + 3
    prod = new_req("prod", total, True)
    prefill(mgr, prod, total)
    blocks, n = mgr.get_computed_blocks(new_req("c1", total, False))
    assert n == 6 * B  # largest aligned hit <= N - W = 99
    assert len(blocks.blocks[0]) == 6 and len(blocks.blocks[1]) == 6  # full-attn + Mamba blocks
    assert probe_hit(mgr, total, spec_off=True) == 8 * B
    # the same prefix cached by a long-tailed nospec request: K>0 reader with tail >= W is uncapped
    long_total = 12 * B + 3
    assert probe_hit(mgr, long_total, spec_off=False) == 8 * B


def test_inv3_nospec_blocks_never_count_as_draft_content(mgr):  # noqa: F811
    total = 8 * B + 3
    prod = new_req("prod", total, True)
    prefill(mgr, prod, total)
    c = mgr.coordinator
    probe = new_req("probe", total, False)
    # draft group taking part finds nothing in a nospec-era prefix
    _, hit = c.find_longest_cache_hit(probe.block_hashes, probe.num_tokens - 1, skip_group_ids=frozenset())
    assert hit == 0
    # a K>0 request that rides the capped hit must not register its unwritten draft pages
    cons = new_req("cons", total, False)
    n_hit = prefill(mgr, cons, total)
    assert n_hit == 6 * B
    assert not any(draft_cached(mgr, cons, i) for i in range(4, 6))  # window pages below the hit: unwritten
    assert all(draft_cached(mgr, cons, i) for i in (6, 7))  # target processed these: written, cached
    # a later K>0 lookup bounded by the garbage pages must not hit them
    _, hit = c.find_longest_cache_hit(probe.block_hashes, 6 * B, skip_group_ids=frozenset())
    assert hit < 6 * B
    # ...while the written pages let the next K>0 turn hit in full (b1 behaviour)
    assert probe_hit(mgr, total) == 8 * B


def test_inv4_short_tail_caps_hit_at_n_minus_w(mgr):  # noqa: F811
    prod = new_req("prod", 8 * B + 3, False)
    prefill(mgr, prod, 8 * B + 3)
    # same-mode prefix: uncapped when the draft pages exist (cap is only the fallback)
    assert probe_hit(mgr, 8 * B + 3) == 8 * B
    # K=0 request path is untouched: full hit
    assert probe_hit(mgr, 8 * B + 3, spec_off=True) == 8 * B
    # nospec-only prefix: cap is the only way in
    mgr2 = type(mgr)(mgr.kv_cache_config, max_model_len=4096, hash_block_size=B, enable_caching=True, use_eagle=False, draft_sliding_window_groups=True)
    p2 = new_req("p2", 8 * B + 3, True)
    prefill(mgr2, p2, 8 * B + 3)
    for tail in (3, W - 1, W, W + 5):
        total = 8 * B + tail
        n = probe_hit(mgr2, total)
        assert n == min(8 * B, (total - W) // B * B), (tail, n)


def test_same_mode_hit_is_b1_behaviour_with_real_draft_pages(mgr):  # noqa: F811
    total = 8 * B + 3
    prod = new_req("prod", total, False)
    prefill(mgr, prod, total)
    prod_ids = {b.block_id for b in draft_pages(mgr, "prod")}
    blocks, n = mgr.get_computed_blocks(new_req("cons", total, False))
    assert n == 8 * B
    hit_ids = {b.block_id for b in blocks.blocks[DRAFT] if not b.is_null}
    assert hit_ids and hit_ids <= prod_ids, "K>0 reader hits the producer's written draft window pages"


def test_k0_reader_registers_nothing_in_draft_group(mgr):  # noqa: F811
    total = 8 * B + 3
    prod = new_req("prod", total, False)
    prefill(mgr, prod, total)
    cons = new_req("cons", total, True)
    assert prefill(mgr, cons, total) == 8 * B
    assert not draft_pages(mgr, "cons")
