# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-block prefill: real allocator + worker copy decisions, CPU scalar state.

The recurrence checks continuation/ownership, not CUDA numerical parity.
Adapted from upstream 1cabc7ce9 test_mamba_sub_block_state.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker import mamba_utils

from . import utils
from .test_mamba_sparse_retention import _request
from .test_mixed_prefill_budget import complete, factory, resident  # noqa: F401

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("block_size", [2048, 4096])
@pytest.mark.parametrize("chunk", [512, 1024])
@pytest.mark.parametrize("spec", [0, 7])
def test_subblock_prefill_preserves_retained_boundary_state(
    monkeypatch, block_size, chunk, spec
):
    alignment = 4096
    pool = BlockPool(48, True, 8)
    manager = MambaManager(
        MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=spec,
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    req = _request("partial", 3 * alignment + 32)
    req.num_computed_tokens = 0
    boundaries = KVCacheCoordinator.get_replay_boundaries(
        SimpleNamespace(eagle_group_ids={0} if spec else set()), req, alignment
    )
    states, state_idx, snapshots = {}, {}, {}
    copy_bufs = SimpleNamespace(
        mamba_group_ids=[0], mamba_spec=manager.kv_cache_spec, offset=0
    )
    batch = SimpleNamespace(
        req_ids=[req.request_id],
        num_accepted_tokens_cpu=[1],
        spec_num_accepted_tokens_cpu=[1],
    )

    def collect(buf, config, funcs, groups, prev, curr, bias, request, ctx, slot):
        assert bias == slot == 0  # A true prefill, not an accepted-draft slice.
        blocks = manager.req_to_blocks[request.request_id]
        states[blocks[curr].block_id] = states[blocks[prev].block_id]

    monkeypatch.setattr(mamba_utils, "collect_mamba_copy_meta", collect)
    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", lambda buf: None)
    monkeypatch.setattr(mamba_utils, "_debug_mamba_align", lambda *a, **kw: None)
    hashes = BlockHashListWithBlockSize(req.block_hashes, 8, block_size)
    while req.num_computed_tokens < req.num_tokens:
        start = req.num_computed_tokens
        count = min(chunk, block_size - start % block_size, req.num_tokens - start)
        end = start + count
        manager.new_step_starts()
        manager.remove_skipped_blocks(req.request_id, start)
        manager.allocate_new_blocks(req.request_id, end, end)
        output = SchedulerOutput.make_empty()
        output.num_scheduled_tokens = {req.request_id: count}
        mamba_utils.preprocess_mamba(
            output,
            None,
            SimpleNamespace(enable_prefix_caching=True),
            state_idx,
            batch,
            {req.request_id: req},
            {},
            None,
            copy_bufs,
        )
        block = manager.req_to_blocks[req.request_id][state_idx[req.request_id]]
        value = states.get(block.block_id, 0)
        for token in range(start, end):
            value = (value * 31 + token + 1) % 1000000007
        states[block.block_id] = value
        manager.cache_blocks(
            req,
            end,
            alignment_tokens=alignment,
            retention_interval=0,
            replay_boundaries=boundaries,
        )
        if end % block_size == 0:
            cached = pool.get_cached_block(hashes[end // block_size - 1], [0])
            if cached is not None:
                snapshots[end] = (cached[0].block_id, value)
        for block_id, snapshot in snapshots.values():
            assert states[block_id] == snapshot
        req.num_computed_tokens = end

    expected = 0
    for token in range(req.num_tokens):
        expected = (expected * 31 + token + 1) % 1000000007
    assert value == expected
    manager.free(req.request_id)
    hit = manager.find_longest_cache_hit(
        hashes,
        req.num_tokens - 1,
        [0],
        pool,
        manager.kv_cache_spec,
        bool(spec),
        alignment,
    )[0]
    boundary = (3 - bool(spec)) * alignment
    assert len(hit) * block_size == boundary
    assert states[hit[-1].block_id] == snapshots[boundary][1]


@pytest.mark.parametrize("block_size", [2048, 4096])
def test_real_scheduler_caps_prefix_hit_before_eagle_boundary(factory, block_size):  # noqa: F811
    from vllm.v1.request import RequestStatus

    scheduler = factory(
        cap=320,
        block_size=block_size,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        kv_cache_spec=MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=7,
        ),
    )
    # Tiny local OPT config supplies model-independent scheduler settings;
    # the real cache group is Mamba with EAGLE's last-page-drop contract.
    scheduler.use_eagle = True
    scheduler.need_mamba_block_aligned_split = True
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.eagle_group_ids = {0}
    coordinator.single_type_managers[0].use_eagle = True
    producer = utils.create_requests(
        1, num_tokens=3 * block_size + 32, req_ids=["producer"], block_size=block_size
    )[0]
    scheduler.add_request(producer)
    while producer.num_output_tokens == 0:
        complete(scheduler, scheduler.schedule())
    scheduler.finish_requests([producer.request_id], RequestStatus.FINISHED_STOPPED)
    resident(scheduler)
    consumer = utils.create_requests(
        1, num_tokens=4 * block_size + 197, req_ids=["consumer"], block_size=block_size
    )[0]
    scheduler.add_request(consumer)
    starts, ends, counts = [], [], []
    while consumer.num_output_tokens == 0:
        step = scheduler.schedule()
        count = step.num_scheduled_tokens["consumer"]
        end = consumer.num_computed_tokens
        starts.append(end - count)
        ends.append(end)
        counts.append(count)
        assert step.num_scheduled_tokens["decode"] == 1
        assert 0 < count <= 320
        assert end <= ((end - count) // block_size + 1) * block_size
        complete(scheduler, step)
    assert starts[0] == 2 * block_size  # A real prefix lookup, with EAGLE drop.
    assert 3 * block_size in ends  # last_cache_position, not an extra tail cut.
    assert 4 * block_size in ends
    assert 197 in counts  # Non-quantum final chunk reaches the sampler.
