# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch/output coverage; run with a serving venv and CUDA hidden."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.compilation.cuda_graph import CUDAGraphLogging, CUDAGraphStat
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class _CPUAsyncOutput:
    def __init__(self, *, model_runner_output, **kwargs):
        self.output = model_runner_output

    def get_output(self):
        return self.output


def _make_runner(monkeypatch, mode, *, enabled=True, async_output=False):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    # Two requests with one real token each. Mock the *final* dispatch (after
    # DP synchronization), including a padding size larger than the local batch.
    padded = 2 if mode == CUDAGraphMode.NONE else 4
    descriptor = BatchExecutionDescriptor(mode, padded, 2, 1)
    schedule = SimpleNamespace(
        num_scheduled_tokens={"a": 1, "b": 1},
        total_num_scheduled_tokens=2,
        new_block_ids_to_zero=[],
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
    )
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        num_reqs=2,
        num_tokens=2,
        num_tokens_after_padding=padded,
        num_draft_tokens=0,
        input_ids=torch.tensor([3, 5]),
        positions=torch.tensor([10, 20]),
        idx_mapping=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 1, 2]),
        seq_lens_cpu_upper_bound=np.array([11, 21]),
    )
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    runner.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(cudagraph_metrics=enabled)
    )
    runner.mixed_prefill_timer = None
    runner.execute_model_state = None
    runner.is_encoder_decoder = False
    runner.is_first_pp_rank = runner.is_last_pp_rank = True
    runner.supports_mm_inputs = runner.use_aux_hidden_state_outputs = False
    runner.use_async_scheduling = async_output
    runner.lora_config = runner.speculator = runner.pp_handler = None
    runner._ple_offload_connector = None
    runner.main_stream = runner.output_copy_stream = None
    runner.dp_size, runner.dp_rank = 2, 0
    runner.kv_cache_config = runner.input_buffers = object()
    runner.attn_groups = []
    runner.req_states = Mock()
    runner.block_tables = Mock()
    runner.kv_connector = Mock()
    runner.kv_connector.post_forward.return_value = None
    runner.eplb = Mock()
    runner.model = Mock(return_value=hidden)
    runner.model_state = Mock()
    runner.model_state.prepare_inputs.return_value = {}
    runner.cudagraph_manager = Mock()
    runner.cudagraph_manager.select_attention_graph.side_effect = lambda d, _: d
    runner.cudagraph_manager.run_fullgraph.return_value = hidden
    runner.prepare_inputs = Mock(return_value=batch)
    runner.prepare_attn = Mock(return_value=({}, {}))
    runner.prepare_dummy_attn = Mock(return_value=({}, {}))
    runner._get_uniform_decode_token_count = Mock(return_value=1)
    runner.prompt_logprobs_worker = Mock()
    runner.prompt_logprobs_worker.compute_prompt_logprobs.return_value = {}
    runner.sample = Mock(
        return_value=(
            SimpleNamespace(sampled_token_ids=torch.tensor([[7], [9]])),
            torch.ones(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )
    )
    runner.pooling_runner = Mock()
    runner.pooling_runner.pool.return_value = (hidden, torch.ones(2, dtype=torch.bool))
    for name in (
        "update_pp_decode_requests",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
        "postprocess_sampled",
        "postprocess_num_computed_tokens",
        "_sm70_v2_mtp_profile_start",
        "_sm70_v2_mtp_profile_finish",
        "_sm70_v2_mtp_profile_report",
    ):
        setattr(runner, name, Mock(return_value=None))
    dispatch = Mock(return_value=(descriptor, torch.tensor([padded, padded])))
    monkeypatch.setattr(mrv2, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(mrv2, "build_slot_mappings_by_layer", lambda *_: {})
    monkeypatch.setattr(mrv2, "set_forward_context", lambda *_, **__: nullcontext())
    monkeypatch.setattr(mrv2.InputBatch, "make_dummy", lambda *_: batch)
    monkeypatch.setattr(mrv2, "AsyncOutput", _CPUAsyncOutput)
    monkeypatch.setattr(mrv2, "AsyncPoolingOutput", _CPUAsyncOutput)
    return runner, schedule, descriptor, dispatch, hidden


@pytest.mark.parametrize("mode", list(CUDAGraphMode.valid_runtime_modes()))
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_output", [False, True])
def test_dispatch_stats_reach_sample_output(monkeypatch, mode, enabled, async_output):
    runner, schedule, descriptor, dispatch, hidden = _make_runner(
        monkeypatch, mode, enabled=enabled, async_output=async_output
    )
    stat_factory = Mock(wraps=CUDAGraphStat)
    monkeypatch.setattr(mrv2, "CUDAGraphStat", stat_factory)

    assert runner.execute_model(schedule) is None
    stat = runner.execute_model_state.cudagraph_stats
    assert runner.execute_model_state.hidden_states is hidden
    dispatch.assert_called_once_with(
        runner.cudagraph_manager, 2, 2, 1, 2, 0, need_eager=False
    )
    output = runner.sample_tokens(None)
    if async_output:
        output = output.get_output()
    assert isinstance(output, ModelRunnerOutput)
    assert output.cudagraph_stats is stat
    assert output.req_ids == ["a", "b"]
    runner.sample.assert_called_once_with(
        hidden, runner.prepare_inputs.return_value, None
    )
    assert runner.execute_model_state is None
    # A second sampling call cannot leak the previous step's statistics.
    assert runner.sample_tokens(None) is None
    if enabled:
        stat_factory.assert_called_once()
        assert stat == CUDAGraphStat(
            num_unpadded_tokens=2,
            num_padded_tokens=descriptor.num_tokens,
            num_paddings=descriptor.num_tokens - 2,
            runtime_mode=str(mode),
        )
    else:
        stat_factory.assert_not_called()
        assert stat is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_output", [False, True])
def test_dispatch_stats_reach_pool_output(monkeypatch, enabled, async_output):
    runner, schedule, _, _, _ = _make_runner(
        monkeypatch, CUDAGraphMode.FULL, enabled=enabled, async_output=async_output
    )
    runner.execute_model(schedule)
    stat = runner.execute_model_state.cudagraph_stats
    output = runner.pool()
    if async_output:
        output = output.get_output()
    assert isinstance(output, ModelRunnerOutput)
    assert output.cudagraph_stats is stat
    assert (stat is not None) == enabled
    assert runner.execute_model_state is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dummy_run": True},
        {"is_profile": True},
        {"dummy_run": True, "is_profile": True},
    ],
)
def test_dummy_and_profile_do_not_emit_stats(monkeypatch, kwargs):
    runner, schedule, _, dispatch, _ = _make_runner(monkeypatch, CUDAGraphMode.NONE)
    stat_factory = Mock(side_effect=AssertionError("not a serving dispatch"))
    monkeypatch.setattr(mrv2, "CUDAGraphStat", stat_factory)
    runner.execute_model(schedule, **kwargs)
    stat_factory.assert_not_called()
    assert runner.execute_model_state.cudagraph_stats is None
    assert dispatch.call_args.kwargs["need_eager"] == kwargs.get("is_profile", False)


@pytest.mark.parametrize("empty_after_dispatch", [False, True])
def test_empty_step_does_not_emit_stats(monkeypatch, empty_after_dispatch):
    runner, schedule, _, dispatch, _ = _make_runner(monkeypatch, CUDAGraphMode.NONE)
    if empty_after_dispatch:
        dispatch.return_value = (
            BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0),
            None,
        )
    else:
        schedule.total_num_scheduled_tokens = 0
        schedule.num_scheduled_tokens = {}
    stat_factory = Mock(side_effect=AssertionError("no model execution"))
    monkeypatch.setattr(mrv2, "CUDAGraphStat", stat_factory)
    assert runner.execute_model(schedule) is runner.kv_connector.no_forward.return_value
    stat_factory.assert_not_called()
    assert runner.execute_model_state is None
    assert dispatch.call_count == int(empty_after_dispatch)


@pytest.mark.parametrize("async_output", [False, True])
def test_twenty_dispatches_produce_twenty_table_observations(monkeypatch, async_output):
    runner, schedule, _, dispatch, _ = _make_runner(
        monkeypatch, CUDAGraphMode.FULL, async_output=async_output
    )
    graph_logger = CUDAGraphLogging(CUDAGraphMode.FULL_AND_PIECEWISE, [2, 4])
    for step in range(20):
        mode = CUDAGraphMode.FULL if step % 2 else CUDAGraphMode.PIECEWISE
        dispatch.return_value = (
            BatchExecutionDescriptor(mode, 4, 2, 1),
            torch.tensor([4, 4]),
        )
        runner.execute_model(schedule)
        output = runner.sample_tokens(None)
        if async_output:
            output = output.get_output()
        assert output.cudagraph_stats is not None
        graph_logger.observe(output.cudagraph_stats)

    table = graph_logger.generate_metric_table()
    rows = [
        line.split("|")[1:-1] for line in table.splitlines() if line.startswith("| ")
    ]
    data_rows = rows[1:]  # Skip column headers; separator is not a data row.
    assert len(data_rows) == 2
    assert {row[3].strip() for row in data_rows} == {
        str(CUDAGraphMode.FULL),
        str(CUDAGraphMode.PIECEWISE),
    }
    assert sum(int(row[-1].strip()) for row in data_rows) == 20
    logged = Mock()
    graph_logger.log(logged)
    logged.assert_called_once_with(table)
    assert graph_logger.stats == []
