"""Mamba align mode: num_required_blocks must never shrink below len(req_blocks).

Source: m51 (2026-10-09) k1 assertion RCA.  In async scheduling a decoder that is
skipped for one step (or starved by a cold-prefill token budget) comes back with
fewer placeholder tokens than it had, so cdiv(num_tokens_main_model) + spec blocks
drops below the blocks it already holds.  Before the fix MambaManager.allocate_new_blocks
asserted (log line MAMBA_ALIGN_SHRINK, added in d83d526c6, then AssertionError
"num_required_blocks N < len(req_blocks) M").  Required behaviour: tolerate the
shrink, allocate nothing, and never lose or reorder blocks.
"""
import pytest, torch
from tests.v1.core import utils
from tests.v1.core.test_mixed_prefill_budget import factory, complete  # noqa
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
B = 4096

def out(s, step, n_acc=1):
    ids = list(step.num_scheduled_tokens)
    toks = [[] if s.requests[r].is_prefill_chunk else [42] * n_acc for r in ids]
    return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)}, sampled_token_ids=toks)

@pytest.mark.parametrize("skip_at", [None, 0, 1, 2, 3, 4, 5, 6])
def test_skipped_decoder_required_blocks_must_not_shrink(factory, skip_at):
    s = factory(target=0, async_scheduling=True, num_speculative_tokens=7,
                block_size=B, max_num_batched_tokens=B, max_model_len=262144,
                num_blocks=400,
                kv_cache_spec=MambaSpec(block_size=B, shapes=((1,),), dtypes=(torch.float16,),
                                        mamba_cache_mode="align", num_speculative_blocks=7))
    s.need_mamba_block_aligned_split = True
    s.mamba_state_block_size = B
    d = utils.create_requests(1, num_tokens=20 * B - 12, req_ids=["d"], max_tokens=512, block_size=B)[0]
    s.add_request(d)
    while d.num_output_tokens == 0:
        s.update_from_output(*(lambda st: (st, out(s, st)))(s.schedule()))
    mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
    inflight = None
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    orig = MambaManager.allocate_new_blocks
    log = []
    def spy(self, rid, nt, ntm):
        before = list(self.req_to_blocks[rid])
        req = -(-ntm // B) + 7
        ret = orig(self, rid, nt, ntm)
        after = list(self.req_to_blocks[rid])
        log.append((rid, ntm, len(before), req, ret, before == after))
        return ret
    MambaManager.allocate_new_blocks = spy
    for i in range(40):
        if skip_at is not None and i == skip_at:
            d.next_decode_eligible_step = s.current_step + 2   # one skipped step
        st = s.schedule()
        nb = len(mgr.req_to_blocks["d"]) if "d" in mgr.req_to_blocks else 0
        if inflight is not None:
            s.update_from_output(*inflight)
        inflight = (st, out(s, st)) if st.num_scheduled_tokens else None
    MambaManager.allocate_new_blocks = orig
    assert log, "allocate_new_blocks was never exercised"
    # (rid, ntm, len_before, required, returned, blocks_unchanged)
    shrinks = [l for l in log if l[3] < l[2]]
    # A shrink must be a pure no-op: nothing allocated, held blocks untouched.
    bad = [l for l in shrinks if l[4] != [] or not l[5]]
    assert not bad, f"shrink call allocated or mutated req_blocks: {bad}"
    assert mgr.num_align_shrink_events == len(shrinks), (mgr.num_align_shrink_events, shrinks)
    if skip_at in (2, 3, 4):  # known to reproduce the shrink (verified pre-fix)
        assert shrinks, "scenario did not reproduce a shrink; test is vacuous"


@pytest.mark.parametrize("skip_at", [2, 3, 4])
def test_starved_by_cold_prefill_budget(factory, skip_at):
    s = factory(target=0, async_scheduling=True, num_speculative_tokens=7,
                block_size=B, max_num_batched_tokens=B, max_model_len=262144,
                num_blocks=400,
                kv_cache_spec=MambaSpec(block_size=B, shapes=((1,),), dtypes=(torch.float16,),
                                        mamba_cache_mode="align", num_speculative_blocks=7))
    s.need_mamba_block_aligned_split = True
    s.mamba_state_block_size = B
    d = utils.create_requests(1, num_tokens=20 * B - 12, req_ids=["d"], max_tokens=512, block_size=B)[0]
    s.add_request(d)
    while d.num_output_tokens == 0:
        s.update_from_output(*(lambda st: (st, out(s, st)))(s.schedule()))
    mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
    inflight = None
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    orig = MambaManager.allocate_new_blocks
    log = []
    def spy(self, rid, nt, ntm):
        before = list(self.req_to_blocks[rid])
        req = -(-ntm // B) + 7
        ret = orig(self, rid, nt, ntm)
        after = list(self.req_to_blocks[rid])
        log.append((rid, ntm, len(before), req, ret, before == after))
        return ret
    MambaManager.allocate_new_blocks = spy
    for i in range(40):
        if skip_at is not None and i == skip_at:
            c = utils.create_requests(1, num_tokens=8 * B + 5, req_ids=["cold"], max_tokens=4, block_size=B)[0]
            s.add_request(c)
            st0 = s.schedule()   # admits cold behind d (d still scheduled)
            if inflight is not None:
                s.update_from_output(*inflight)
            inflight = (st0, out(s, st0))
            s.running.sort(key=lambda r: r.request_id != "cold")   # cold ahead of d, like live order
        st = s.schedule()
        nb = len(mgr.req_to_blocks["d"]) if "d" in mgr.req_to_blocks else 0
        if inflight is not None:
            s.update_from_output(*inflight)
        inflight = (st, out(s, st)) if st.num_scheduled_tokens else None
    MambaManager.allocate_new_blocks = orig
    assert log, "allocate_new_blocks was never exercised"
    # (rid, ntm, len_before, required, returned, blocks_unchanged)
    shrinks = [l for l in log if l[3] < l[2]]
    # A shrink must be a pure no-op: nothing allocated, held blocks untouched.
    bad = [l for l in shrinks if l[4] != [] or not l[5]]
    assert not bad, f"shrink call allocated or mutated req_blocks: {bad}"
    assert mgr.num_align_shrink_events == len(shrinks), (mgr.num_align_shrink_events, shrinks)
    if skip_at == 2:  # known to reproduce the shrink (verified pre-fix)
        assert shrinks, "scenario did not reproduce a shrink; test is vacuous"
