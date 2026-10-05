# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the align-mode Mamba steady-state block accounting.

A running request holds, per Mamba group, its running state, the
``num_speculative_blocks`` and the superseded state blocks that still await
release (1 under sync scheduling, 2 under async scheduling; measured in
``test_mamba_align_state_block_leak.py``). Admission
(``allocate_slots(full_sequence_must_fit=True)``) and the capacity report
(``MambaSpec.max_memory_usage_bytes``) used to count 1 + spec and 2 + spec.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .test_mamba_align_state_block_leak import (
    BLOCK_SIZE,
    RID,
    _drive,
    _manager,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _spec(num_speculative_blocks: int, block_size: int = BLOCK_SIZE) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )


def _admission_count(manager: MambaManager, num_tokens: int) -> int:
    return manager.get_num_blocks_to_allocate(
        RID,
        num_tokens,
        [],
        0,
        num_tokens,
        apply_admission_cap=True,
    )


@pytest.mark.parametrize("spec_blocks", [0, 1, 4])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_admission_count_equals_measured_steady_state_holding(
    spec_blocks, async_scheduling
):
    # The leak-fix drive measures how many blocks a running request holds
    # after each allocation; admission must reserve exactly that maximum.
    pool, manager = _manager(spec_blocks)
    manager.async_scheduling = async_scheduling
    trace = _drive(manager, pool, 40, lag=int(async_scheduling), chunk_tokens=100)
    max_held = max(row[2] for row in trace.rows)
    assert max_held == 1 + spec_blocks + (2 if async_scheduling else 1)

    _, fresh = _manager(spec_blocks)
    fresh.async_scheduling = async_scheduling
    assert _admission_count(fresh, 40 * 100) == max_held


@pytest.mark.parametrize("spec_blocks", [0, 1, 4])
def test_admission_count_defaults_to_async_and_step_allocation_is_unchanged(
    spec_blocks,
):
    _, manager = _manager(spec_blocks)
    # default: the conservative (async) figure
    assert _admission_count(manager, 1000) == 3 + spec_blocks
    # the per-step allocation arithmetic (no admission cap) keeps counting only
    # what a first chunk allocates
    assert (
        manager.get_num_blocks_to_allocate(RID, 1000, [], 0, 1000) == 1 + spec_blocks
    )


def _kv_cache_manager(num_gpu_blocks: int, async_scheduling: bool):
    config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], _spec(4))],
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 16,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
        async_scheduling=async_scheduling,
    )


def _request(num_tokens: int) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=RID,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def test_request_fitting_the_old_count_but_not_the_steady_state_is_refused():
    """Async: 1 + 4 + 2 = 7 blocks are needed, the pool has 6 free (5 is what
    the old count asked for). ``allocate_slots`` returns None and the scheduler
    leaves the request at the head of the waiting queue (``scheduler.py``
    ``if new_blocks is None: ... break``, popped only after a successful
    allocation): it is retried every step and waits until a running request
    frees blocks, instead of being admitted and preempted in a loop."""
    num_tokens = 3 * BLOCK_SIZE
    # the null block is not allocatable: 7 blocks => 6 free
    manager = _kv_cache_manager(7, async_scheduling=True)
    assert manager.block_pool.get_num_free_blocks() == 6
    assert (
        manager.allocate_slots(
            _request(num_tokens), num_tokens, full_sequence_must_fit=True
        )
        is None
    )
    # nothing was taken from the pool while refusing
    assert manager.block_pool.get_num_free_blocks() == 6
    # without the admission gate the first chunk (1 + 4) still fits, which is
    # exactly the admit-then-preempt path the gate prevents
    assert manager.allocate_slots(_request(num_tokens), num_tokens) is not None


def test_sync_scheduling_admits_the_same_request():
    num_tokens = 3 * BLOCK_SIZE
    manager = _kv_cache_manager(7, async_scheduling=True)
    manager.coordinator.configure_async_scheduling(False)
    assert (
        manager.allocate_slots(
            _request(num_tokens), num_tokens, full_sequence_must_fit=True
        )
        is not None
    )
    # one block short of 1 + 4 + 1 is refused under sync as well
    manager = _kv_cache_manager(6, async_scheduling=False)
    assert manager.block_pool.get_num_free_blocks() == 5
    assert (
        manager.allocate_slots(
            _request(num_tokens), num_tokens, full_sequence_must_fit=True
        )
        is None
    )


# Production geometry (plan 072 section 3.1): 1616-token blocks, 262144-token
# requests, attention + QSA key ring + 4 Mamba groups, MTP 4.
MAX_MODEL_LEN = 262144
PROD_BLOCK = 1616
PROD_NUM_BLOCKS = 292
PROD_SPEC_BLOCKS = 4
PROD_MAMBA_GROUPS = 4


def _production_config(async_scheduling: bool):
    attention = FullAttentionSpec(
        block_size=PROD_BLOCK, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    ring = CircularBufferSpec(
        block_size=PROD_BLOCK,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.float16,
    )
    mamba = _spec(PROD_SPEC_BLOCKS, block_size=PROD_BLOCK)
    kv_cache_config = KVCacheConfig(
        num_blocks=PROD_NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], attention),
            KVCacheGroupSpec(["ring"], ring),
            *[KVCacheGroupSpec([f"mamba{i}"], mamba) for i in range(4)],
        ],
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    return vllm_config, kv_cache_config


@pytest.mark.parametrize(
    ("async_scheduling", "mamba_blocks_per_group"), [(False, 6), (True, 7)]
)
def test_capacity_report_counts_the_pending_state_blocks(
    async_scheduling, mamba_blocks_per_group
):
    vllm_config, kv_cache_config = _production_config(async_scheduling)
    attention_blocks = -(-MAX_MODEL_LEN // PROD_BLOCK)
    assert attention_blocks == 163
    per_request = attention_blocks + 1 + PROD_MAMBA_GROUPS * mamba_blocks_per_group
    concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
    assert concurrency == pytest.approx(PROD_NUM_BLOCKS / per_request)

    with mock.patch.object(kv_cache_utils, "logger") as logger:
        kv_cache_utils._report_kv_cache_config(vllm_config, kv_cache_config)
    (size_call, concurrency_call) = logger.info_once.call_args_list
    num_tokens = int(PROD_NUM_BLOCKS / per_request * MAX_MODEL_LEN)
    assert size_call.args[1] == f"{num_tokens:,}"
    assert concurrency_call.args[2] == pytest.approx(concurrency)

    if async_scheduling:
        assert per_request == 192
        assert f"{concurrency:.2f}" == "1.52"
        assert size_call.args[1] == "398,677"
    else:
        # unchanged from before: the old 2 + spec was the sync figure
        assert per_request == 188
        assert f"{concurrency:.2f}" == "1.55"
        assert size_call.args[1] == "407,159"
