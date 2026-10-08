# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual V2 execute/sample entrypoints with CPU dispatch stubs."""

from unittest.mock import Mock

import pytest

from vllm.config.compilation import CUDAGraphMode

from .test_gpu_model_runner_v2_cudagraph_metrics import _make_runner


@pytest.mark.parametrize("mode", list(CUDAGraphMode.valid_runtime_modes()))
@pytest.mark.parametrize("async_output", [False, True])
def test_timer_covers_execute_through_sampling_without_changing_output(
    monkeypatch, mode, async_output
):
    runner, schedule, descriptor, _, _ = _make_runner(
        monkeypatch, mode, async_output=async_output
    )
    runner.cadence_step_timer = Mock()
    assert runner.execute_model(schedule) is None
    runner.cadence_step_timer.begin.assert_called_once_with(schedule)
    runner.cadence_step_timer.route.assert_called_once_with(
        str(mode), descriptor.num_tokens
    )
    runner.cadence_step_timer.finish.assert_not_called()
    result = runner.sample_tokens(None)
    runner.cadence_step_timer.finish.assert_called_once()
    if async_output:
        result = result.get_output()
    assert result.req_ids == ["a", "b"]
    runner.sample_tokens(None)
    runner.cadence_step_timer.finish.assert_called_once()


def test_capture_dummy_does_not_begin_events(monkeypatch):
    runner, schedule, _, _, _ = _make_runner(monkeypatch, CUDAGraphMode.NONE)
    runner.cadence_step_timer = Mock()
    runner.execute_model(schedule, dummy_run=True)
    runner.cadence_step_timer.begin.assert_not_called()
    runner.cadence_step_timer.route.assert_not_called()
