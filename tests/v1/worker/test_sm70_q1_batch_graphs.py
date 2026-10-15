# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for plan 075 M-fix12: FULL graphs for request-major q1 batches.

A hot-switched nospec process keeps decode_query_len=8, so before this change a
B>=2 uniform q1 batch had no FULL descriptor and was dispatched to PIECEWISE.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.sm70_e4m3_long import BUILTIN_MAX_CONTEXT
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

CAPTURE_SIZES = [8, 16, 24, 32, 48, 64, 96, 128]


def _manager(monkeypatch, *, q1_batch, max_num_seqs, tail=True, method="dflash"):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", str(int(tail)))
    if q1_batch is None:
        monkeypatch.delenv("VLLM_SM70_DFLASH2_Q1_BATCH_CUDAGRAPHS", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_DFLASH2_Q1_BATCH_CUDAGRAPHS", str(int(q1_batch)))
    monkeypatch.setenv("VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS", "0")
    monkeypatch.delenv("VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST", raising=False)
    monkeypatch.setattr(cg.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        cg.current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(7, 0),
    )
    monkeypatch.setattr(cg.torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(
        cg,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    compilation = CompilationConfig(
        cudagraph_capture_sizes=list(CAPTURE_SIZES), max_cudagraph_capture_size=128
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(max_model_len=BUILTIN_MAX_CONTEXT),
        compilation_config=compilation,
        parallel_config=SimpleNamespace(
            data_parallel_size=1, tensor_parallel_size=4, pipeline_parallel_size=1
        ),
        speculative_config=SimpleNamespace(
            method=method, num_speculative_tokens=7, ngram_assist=False
        ),
    )
    manager = ModelCudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_AND_PIECEWISE, 8
    )
    manager._graphs_captured = True
    return manager


def _full(manager):
    return manager._capture_descs[CUDAGraphMode.FULL]


@pytest.mark.parametrize("num_reqs", [2, 3, 4, 5, 6, 7, 8])
def test_hot_switch_q1_batch_dispatches_full(monkeypatch, num_reqs):
    manager = _manager(monkeypatch, q1_batch=True, max_num_seqs=8)
    desc = manager.dispatch(num_reqs, num_reqs, 1)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == 1
    assert desc.num_reqs == desc.num_tokens == num_reqs
    assert desc in _full(manager)


def test_q1_batch_pads_to_next_captured_size(monkeypatch):
    manager = _manager(monkeypatch, q1_batch=True, max_num_seqs=16)
    # B=9..16 replay the B=16 graph with padded requests.
    for num_reqs in range(9, 17):
        desc = manager.dispatch(num_reqs, num_reqs, 1)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert (desc.num_tokens, desc.num_reqs, desc.uniform_token_count) == (
            16,
            16,
            1,
        )
    # Above max_num_seqs there is no q1 descriptor (24 reqs > 16).
    assert manager.dispatch(17, 17, 1).cg_mode == CUDAGraphMode.PIECEWISE


def test_q1_single_request_keeps_the_existing_tail(monkeypatch):
    on = _manager(monkeypatch, q1_batch=True, max_num_seqs=8)
    off = _manager(monkeypatch, q1_batch=False, max_num_seqs=8)
    assert on.dispatch(1, 1, 1) == off.dispatch(1, 1, 1)
    assert on.dispatch(1, 1, 1).num_reqs == 1


def test_q8_batches_and_tails_are_unchanged(monkeypatch):
    on = _manager(monkeypatch, q1_batch=True, max_num_seqs=16)
    off = _manager(monkeypatch, q1_batch=False, max_num_seqs=16)
    for q in range(2, 9):
        assert on.dispatch(1, q, q) == off.dispatch(1, q, q)
    for num_reqs in range(1, 17):
        assert on.dispatch(num_reqs, num_reqs * 8, 8) == off.dispatch(
            num_reqs, num_reqs * 8, 8
        )
    # q2..q7 request-major batches stay un-captured (only q1 is added).
    for q in range(2, 8):
        assert on.dispatch(2, 2 * q, q).cg_mode != CUDAGraphMode.FULL
    # Non-uniform decode still falls to PIECEWISE.
    assert on.dispatch(3, 3, None).cg_mode == CUDAGraphMode.PIECEWISE


def test_only_q1_descriptors_are_added(monkeypatch):
    on = _manager(monkeypatch, q1_batch=True, max_num_seqs=16)
    off = _manager(monkeypatch, q1_batch=False, max_num_seqs=16)
    added = set(_full(on)) - set(_full(off))
    assert set(_full(off)) <= set(_full(on))
    assert {d.uniform_token_count for d in added} == {1}
    assert {d.num_reqs for d in added} == {2, 3, 4, 5, 6, 7, 8, 16}
    assert all(d.attention_context_bucket is None for d in added)
    assert all(d.num_tokens == d.num_reqs <= 16 for d in added)


@pytest.mark.parametrize("q1_batch", [None, False])
def test_env_off_matches_base(monkeypatch, q1_batch):
    manager = _manager(monkeypatch, q1_batch=q1_batch, max_num_seqs=8)
    base = _manager(monkeypatch, q1_batch=False, max_num_seqs=8)
    assert set(_full(manager)) == set(_full(base))
    for num_reqs in range(2, 9):
        assert manager.dispatch(num_reqs, num_reqs, 1).cg_mode == (
            CUDAGraphMode.PIECEWISE
        )


def test_q1_batch_requires_tail_graphs(monkeypatch):
    manager = _manager(monkeypatch, q1_batch=True, max_num_seqs=8, tail=False)
    assert not manager._sm70_dflash2_q1_batch_graphs
    assert manager.dispatch(4, 4, 1).cg_mode != CUDAGraphMode.FULL


def test_non_dflash_process_is_untouched(monkeypatch):
    manager = _manager(monkeypatch, q1_batch=True, max_num_seqs=8, method="mtp")
    assert not manager._sm70_dflash2_q1_batch_graphs
