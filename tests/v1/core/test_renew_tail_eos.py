# SPDX-License-Identifier: Apache-2.0
"""renew-tail-eos-v1: tail hash at request end, non-aligned resume, sampler buffer."""

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_test

BS = 16


@pytest.fixture(autouse=True)
def _hash_init():
    init_none_hash(sha256)


def _config(num_blocks=40, nspec=0, draft=False) -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=BS,
                shapes=(1, 1),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=nspec,
            ),
        ),
    ]
    if draft:
        groups.append(
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=BS // 2,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=BS,
                ),
            )
        )
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def _manager(monkeypatch, flag="1", **kw) -> KVCacheManager:
    if flag is None:
        monkeypatch.delenv("C4140_RENEW_TAIL_EOS", raising=False)
    else:
        monkeypatch.setenv("C4140_RENEW_TAIL_EOS", flag)
    draft = kw.get("draft", False)
    return KVCacheManager(
        _config(**kw),
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=BS // 2 if draft else BS,
        draft_sliding_window_groups=draft,
    )


def _request(rid: str, tokens: list[int], hash_size: int = BS) -> Request:
    sp = SamplingParams(max_tokens=64)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=rid,
        prompt_token_ids=list(tokens),
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_size, sha256),
    )


def _schedule(manager, req, n, lookahead=0):
    """One scheduler step: allocate then advance num_computed_tokens."""
    blocks = manager.allocate_slots(req, n, num_lookahead_tokens=lookahead)
    assert blocks is not None
    req.num_computed_tokens += n


def _finish_turn(manager, rid, prompt, out, nspec_gen=1, hash_size=BS, kept=None):
    """Prefill `prompt` in two chunks, decode `out` tokens one by one, finish.

    Returns the finished request (its tokens are prompt + out; the last
    sampled token has no KV/state, so num_computed_tokens == num_tokens - 1).
    """
    req = _request(rid, prompt, hash_size)
    computed, n = manager.get_computed_blocks(req)
    assert n == 0
    first = (len(prompt) // BS) * BS or len(prompt)
    manager.allocate_slots(req, first, 0, computed)
    req.num_computed_tokens = first
    if len(prompt) > first:
        _schedule(manager, req, len(prompt) - first)
    for i, tok in enumerate(out):
        req.append_output_token_ids(tok)  # sampled at the end of the last step
        if i < len(out) - 1:
            _schedule(manager, req, 1)  # fed to the model in the next step
    # the finished request's last token was never fed to the model
    assert req.num_computed_tokens == req.num_tokens - 1
    kept = nspec_gen if kept is None else kept
    # the scheduler rolled num_computed back to c0 + g, the stop kept only m of the g accepted tokens
    req.num_computed_tokens += nspec_gen - kept
    req.status = RequestStatus.FINISHED_STOPPED
    req.renew_tail_hint = (nspec_gen, 0, kept)
    manager.free(req)
    return req


PROMPT = list(range(1000, 1040))  # 40 tokens: 2 full blocks + 8
PROMPT2 = list(range(2000, 2040))
OUT = [7, 8, 9, 10, 11]  # N = 45, L = 44 -> aligned 32, tail 12 tokens


def test_a_tail_registered_at_free_queue_tail(monkeypatch):
    m = _manager(monkeypatch)
    assert m.renew_tail is not None
    old = _finish_turn(m, "0", PROMPT, OUT)
    rt = m.renew_tail
    assert m.tail_stats["tail_reg"] == 1
    assert len(rt.index) == 1
    (parent, ent), = rt.index.items()
    assert ent.length == 44
    assert set(ent.group_ids) == {0, 1}
    free = m.block_pool.free_block_queue.get_all_free_blocks()
    hashed = [b for b in free if b.block_hash is not None]
    first_hashed = next(i for i, b in enumerate(free) if b.block_hash is not None)
    # every cached block (aligned + tail) sits behind every unhashed block
    assert all(b.block_hash is not None for b in free[first_hashed:])
    for gid in ent.group_ids:
        key = make_block_hash_with_group_id(ent.tail_hash, gid)
        blk = m.block_pool.cached_block_hash_to_block.get_one_block(key)
        assert blk is not None and blk in hashed
    assert old.num_computed_tokens == 44


def test_b_resume_exactly_at_L_and_consume(monkeypatch):
    m = _manager(monkeypatch)
    old = _finish_turn(m, "0", PROMPT, OUT)
    ent = next(iter(m.renew_tail.index.values()))
    tokens = list(old.all_token_ids) + list(range(2000, 2010))
    new = _request("1", tokens)
    computed, n = m.get_computed_blocks(new)
    assert n == 44
    assert m.tail_stats["tail_hit"] == 1
    assert len(computed.blocks[0]) == 3  # 2 aligned + tail
    assert len(computed.blocks[1]) == 3 and computed.blocks[1][0].is_null
    assert computed.blocks[0][2].block_hash is not None  # still hashed: read-only
    m.allocate_slots(new, len(tokens) - n, n, computed)
    # consume-on-hit: tail unhashed, index entry gone, second request cannot hit
    assert not m.renew_tail.index
    for gid in ent.group_ids:
        key = make_block_hash_with_group_id(ent.tail_hash, gid)
        assert m.block_pool.cached_block_hash_to_block.get_one_block(key) is None
    # allocate_slots already filled block 2 (44 + 11 = 55 tokens): it carries
    # the ordinary full-block hash now, never the tail hash
    assert computed.blocks[0][2].block_hash == make_block_hash_with_group_id(
        new.block_hashes[2], 0
    )
    again = _request("2", tokens)
    _, n2 = m.get_computed_blocks(again)
    assert n2 == 48  # ordinary aligned hit on the new request's own blocks, not a tail hit
    assert m.tail_stats["tail_miss_absent"] == 2  # old turn-1 lookup + this one
    # the mamba page of the resumed block is hashed as a normal full block too
    assert computed.blocks[1][2].block_hash == make_block_hash_with_group_id(
        new.block_hashes[2], 1
    )


def test_c_tokens_differ_in_tail(monkeypatch):
    m = _manager(monkeypatch)
    old = _finish_turn(m, "0", PROMPT, OUT)
    tokens = list(old.all_token_ids) + list(range(2000, 2010))
    tokens[35] = 4242  # inside the tail region [32, 44)
    new = _request("1", tokens)
    _, n = m.get_computed_blocks(new)
    assert n == 32  # aligned hit only
    assert m.tail_stats["tail_miss_hash"] == 1
    assert m.tail_stats["tail_hit"] == 0
    assert len(m.renew_tail.index) == 1  # not consumed


def test_c2_new_prompt_shorter_than_L_is_a_length_miss(monkeypatch):
    m = _manager(monkeypatch)
    old = _finish_turn(m, "0", PROMPT, OUT)
    new = _request("1", list(old.all_token_ids)[:44])  # hit cap is 43 < L
    _, n = m.get_computed_blocks(new)
    assert n <= 43
    assert m.tail_stats["tail_miss_len"] == 1


def test_d_eviction_removes_index_entry(monkeypatch):
    m = _manager(monkeypatch)
    _finish_turn(m, "0", PROMPT, OUT)
    ent = next(iter(m.renew_tail.index.values()))
    m.block_pool.get_new_blocks(m.block_pool.get_num_free_blocks())
    assert not m.renew_tail.index
    assert not m.renew_tail._by_hash
    for gid in ent.group_ids:
        key = make_block_hash_with_group_id(ent.tail_hash, gid)
        assert m.block_pool.cached_block_hash_to_block.get_one_block(key) is None


def test_e_flag_off_is_unchanged(monkeypatch):
    m = _manager(monkeypatch, flag=None)
    assert m.renew_tail is None and m.tail_stats == {}
    old = _finish_turn(m, "0", PROMPT, OUT)
    free = m.block_pool.free_block_queue.get_all_free_blocks()
    # only the two aligned full blocks (full + mamba state at 32) are hashed
    n_hashed = sum(b.block_hash is not None for b in free)
    m2 = _manager(monkeypatch, flag="1")
    _finish_turn(m2, "0", PROMPT, OUT)
    free2 = m2.block_pool.free_block_queue.get_all_free_blocks()
    n_hashed2 = sum(b.block_hash is not None for b in free2)
    assert n_hashed2 == n_hashed + 2  # + full tail block + mamba tail block
    new = _request("1", list(old.all_token_ids) + [5, 6, 7])
    _, n = m.get_computed_blocks(new)
    assert n == 32


def test_flag_value_zero_is_off(monkeypatch):
    m = _manager(monkeypatch, flag="0")
    assert m.renew_tail is None


def _async_finish(m, rid, prompt, in_flight, nspec_gen=1, kept=None):
    """Stop seen in step k while step k+1 (feeds EOS + drafts) is in flight."""
    req = _request(rid, prompt)
    computed, _ = m.get_computed_blocks(req)
    m.allocate_slots(req, len(prompt), 0, computed)
    req.num_computed_tokens = len(prompt)
    req.append_output_token_ids(1)  # EOS sampled by step k
    # step k+1 already scheduled: feeds EOS (+ drafts), optimistic count
    m.allocate_slots(req, in_flight, num_lookahead_tokens=0)
    req.num_computed_tokens += in_flight
    req.status = RequestStatus.FINISHED_STOPPED
    req.renew_tail_hint = (nspec_gen, in_flight, nspec_gen if kept is None else kept)
    m.free(req)
    return req


def test_async_inflight_registers_L_plus_1(monkeypatch):
    m = _manager(monkeypatch)
    old = _async_finish(m, "0", PROMPT, 1)
    assert m.tail_stats["tail_reg_async"] == 1 and m.tail_stats["tail_reg_sync"] == 0
    ent = next(iter(m.renew_tail.index.values()))
    assert ent.length == old.num_tokens  # L+1: the EOS state
    new = _request("1", list(old.all_token_ids) + [5, 5, 5])
    computed, n = m.get_computed_blocks(new)
    assert n == old.num_tokens
    assert m.tail_stats["tail_hit"] == 1


def test_sync_registers_at_L(monkeypatch):
    m = _manager(monkeypatch)
    old = _finish_turn(m, "0", PROMPT, OUT)
    ent = next(iter(m.renew_tail.index.values()))
    assert ent.length == old.num_tokens - 1
    assert m.tail_stats["tail_reg_sync"] == 1


def test_glog_multi_accept_sync_registers_with_replay_bias(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH2_GDN_LOG", "1")
    m = _manager(monkeypatch, nspec=2)
    old = _finish_turn(m, "0", PROMPT, OUT, nspec_gen=3)
    assert m.tail_stats["tail_reg_sync"] == 1 and m.tail_stats["tail_reg_bias"] == 1
    assert next(iter(m.renew_tail.index.values())).bias == 2  # g = m = 3: two logged updates
    new = _request("1", list(old.all_token_ids) + [9, 9, 9])
    computed, n = m.get_computed_blocks(new)
    assert n == 44 and m.tail_stats["tail_hit"] == 1
    m.allocate_slots(new, 3, 0, computed)
    assert new.renew_tail_bias == 2  # consume() hands the bias to NewRequestData


def test_glog_sync_stop_inside_the_accepted_run_registers_kept_state(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH2_GDN_LOG", "1")
    m = _manager(monkeypatch, nspec=2)
    _finish_turn(m, "0", PROMPT, OUT, nspec_gen=3, kept=1)  # EOS was the first accepted token: m = 1
    assert m.tail_stats["tail_reg_sync"] == 1 and m.tail_stats["tail_reg_bias"] == 0
    assert next(iter(m.renew_tail.index.values())).bias == 0
    _finish_turn(m, "1", PROMPT2, OUT, nspec_gen=3, kept=2)  # max_tokens cut the run after 2 of 3
    assert sorted(e.bias for e in m.renew_tail.index.values()) == [0, 1]
    assert m.tail_stats["tail_reg_bias"] == 1


def test_sync_accounting_mismatch_is_skip_acct_with_a_full_log_line(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("VLLM_DFLASH2_GDN_LOG", "1")
    monkeypatch.setenv("C4140_RENEW_TAIL_VERBOSE", "1")
    m = _manager(monkeypatch, nspec=2)
    req = _request("0", PROMPT)
    computed, _ = m.get_computed_blocks(req)
    m.allocate_slots(req, 40, 0, computed)
    req.num_computed_tokens = 40
    for t in OUT:
        req.append_output_token_ids(t)
    req.num_computed_tokens += 5  # not what any accepted/kept pair explains
    req.status = RequestStatus.FINISHED_STOPPED
    req.renew_tail_hint = (1, 0, 1)
    with caplog.at_level(logging.INFO, logger="vllm.v1.core.renew_tail"):
        m.free(req)
    assert m.tail_stats["tail_skip_acct"] == 1 and not m.renew_tail.index
    line = [r.getMessage() for r in caplog.records if "tail_skip_acct" in r.getMessage()][-1]
    assert "n=" in line and "nc=" in line and "in_flight=0" in line and "g=1" in line and "m=1" in line


def test_async_stop_inside_the_accepted_run_is_not_recoverable(monkeypatch):
    m = _manager(monkeypatch)
    _async_finish(m, "0", PROMPT, 4, nspec_gen=4, kept=2)
    assert m.tail_stats["tail_skip_async_trunc"] == 1 and m.tail_stats["tail_reg"] == 0
    _async_finish(m, "1", PROMPT2, 4, nspec_gen=4, kept=4)  # nothing truncated: registers as before
    assert m.tail_stats["tail_reg_async"] == 1


def test_aborted_request_registers_nothing(monkeypatch):
    m = _manager(monkeypatch)
    req = _request("0", PROMPT)
    computed, _ = m.get_computed_blocks(req)
    m.allocate_slots(req, 40, 0, computed)
    m.free(req)  # no hint: abort / preemption path
    assert m.tail_stats["tail_reg"] == 0 and not m.renew_tail.index


def test_speculative_slot_selects_the_state_page(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH2_GDN_LOG", "0")
    m = _manager(monkeypatch, nspec=2)
    old = _finish_turn(m, "0", PROMPT, OUT, nspec_gen=2)
    ent = next(iter(m.renew_tail.index.values()))
    key = make_block_hash_with_group_id(ent.tail_hash, 1)
    page = m.block_pool.cached_block_hash_to_block.get_one_block(key)
    # blocks: [null.., running r, spec r+1, spec r+2]; 2 tokens accepted ->
    # the state lives in slot r + 1
    tokens = list(old.all_token_ids) + [9, 9, 9]
    new = _request("1", tokens)
    computed, n = m.get_computed_blocks(new)
    assert n == 44
    assert computed.blocks[1][-1].block_id == page.block_id
    assert m.tail_stats["tail_hit"] == 1


def test_draft_group_participates_in_tail(monkeypatch):
    m = _manager(monkeypatch, draft=True)
    assert m.renew_tail is not None
    hs = BS // 2
    old = _finish_turn(m, "0", PROMPT, OUT, hash_size=hs)
    assert m.tail_stats["tail_reg"] == 1
    ent = next(iter(m.renew_tail.index.values()))
    assert ent.draft is not None
    tokens = list(old.all_token_ids) + list(range(3000, 3040))
    new = _request("1", tokens, hs)
    computed, n = m.get_computed_blocks(new)
    assert n == 44
    assert m.tail_stats["tail_hit"] >= 1
    # draft pages cover the last W-1 tokens before L
    assert any(not b.is_null for b in computed.blocks[2])
    assert m.renew_tail.pending["1"].draft_group_ids == (2,)


def test_draft_tail_miss_when_interior_draft_page_gone(monkeypatch):
    m = _manager(monkeypatch, draft=True)
    hs = BS // 2
    old = _finish_turn(m, "0", PROMPT, OUT, hash_size=hs)
    ent = next(iter(m.renew_tail.index.values()))
    run = ent.draft[2]
    # the interior page carries an ordinary hash (not owned by the entry)
    pages = [j for j in range(run.first_idx, run.last_idx + 1) if j not in run.run_keys]
    assert pages
    tokens = list(old.all_token_ids) + list(range(3000, 3040))
    hashes = m.renew_tail._page_hashes(_request("x", tokens, hs), hs)
    blk = m.block_pool.get_cached_block(hashes[pages[0]], [2])[0]
    m.block_pool._maybe_evict_cached_block(blk)
    new = _request("1", tokens, hs)
    computed, n = m.get_computed_blocks(new)
    assert m.tail_stats["tail_miss_draft"] == 1
    assert n == 44  # branch (ii): draft skipped, cap N-W still allows L
    assert m.renew_tail.pending["1"].draft_group_ids == ()


# ---------------------------------------------------------------- sampler


def test_f_sampler_buffer_bytes_identical(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_triton as tt

    torch.manual_seed(0)
    rows, vocab = 6, 40000
    logits = torch.randn(rows, vocab)
    k = torch.tensor([5, 20, 50, 100, 1, 127], dtype=torch.int32)
    p = torch.tensor([0.9, 0.5, 0.99, 0.8, 0.7, 0.95])

    new = tt._apply_top_k_top_p_compact(logits.clone(), k, p, -float("inf"))
    monkeypatch.setattr(
        tt,
        "_persistent_neg_inf_buffer",
        lambda x: torch.full_like(x, -float("inf")),
    )
    old = tt._apply_top_k_top_p_compact(logits.clone(), k, p, -float("inf"))
    assert new is not None and old is not None
    assert new.dtype == old.dtype and new.shape == old.shape
    assert new.numpy().tobytes() == old.numpy().tobytes()


def test_f2_sampler_buffer_is_persistent_and_grow_only():
    from vllm.v1.sample.ops import topk_topp_triton as tt

    tt._NEG_INF_BUFFERS.clear()
    a = tt._persistent_neg_inf_buffer(torch.zeros(4, 128))
    ptr = a.untyped_storage().data_ptr()
    a.fill_(3.0)
    b = tt._persistent_neg_inf_buffer(torch.zeros(2, 128))  # fewer rows
    assert b.untyped_storage().data_ptr() == ptr  # no new allocation
    assert torch.isneginf(b).all() and b.shape == (2, 128)  # re-filled
    c = tt._persistent_neg_inf_buffer(torch.zeros(8, 128))  # grows
    assert c.shape == (8, 128) and torch.isneginf(c).all()
    d = tt._persistent_neg_inf_buffer(torch.zeros(3, 128))
    assert d.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()


def test_verbose_event_lines_have_fixed_format(monkeypatch, caplog):
    import logging
    import re

    monkeypatch.setenv("C4140_RENEW_TAIL_VERBOSE", "1")
    m = _manager(monkeypatch)
    with caplog.at_level(logging.INFO, logger="vllm.v1.core.renew_tail"):
        old = _finish_turn(m, "0", PROMPT, OUT)
        new = _request("1", list(old.all_token_ids) + [5, 5, 5])
        m.get_computed_blocks(new)
    lines = [r.getMessage() for r in caplog.records if "RENEW_TAIL event=" in r.getMessage()]
    pat = re.compile(r"^RENEW_TAIL event=\w+ req=\S+ L=-?\d+ groups=\S+( \w+=\S+)*$")
    assert len(lines) >= 2 and all(pat.match(x) for x in lines), lines
    assert any("event=tail_reg_sync req=0 L=44 groups=0,1" in x for x in lines)
    assert any("event=tail_hit req=1 L=44" in x for x in lines)
