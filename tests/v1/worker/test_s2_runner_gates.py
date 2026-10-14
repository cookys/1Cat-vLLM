# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1-fix-3: walk the real GPUModelRunner branches that carry the
drafter_resident gate (sample_tokens propose / _dummy_run dummy_propose) and
the S1 skip_draft_propose branch, with a fake speculator; CPU-only."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.worker.gpu import model_runner as mr
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _FakeSpeculator:
    supports_mm_inputs = False
    requires_host_token_state = False

    def __init__(self, resident):
        self.drafter_resident = resident
        self.propose_calls = 0

    def propose(self, *a, **k):
        self.propose_calls += 1
        return torch.zeros((1, 1), dtype=torch.int64)


def _runner(resident):
    r = MagicMock()
    spec = _FakeSpeculator(resident)
    r.speculator = spec
    r.pp_handler = None
    r.device = torch.device("cpu")
    r.is_last_pp_rank = True
    r.use_async_scheduling = False
    r._sm70_v2_mtp_profile_pending = None
    r.mixed_prefill_timer = None
    r._require_drafter_resident = lambda entry: GPUModelRunner._require_drafter_resident(r, entry)
    r.sample.return_value = (MagicMock(), MagicMock(), MagicMock())
    r.req_states.draft_tokens = torch.zeros((4, 1), dtype=torch.int64)
    ib = MagicMock()
    ib.req_ids = ["a"]
    ib.num_reqs = 1
    ib.idx_mapping = torch.zeros(1, dtype=torch.int64)
    return r, spec, ib


def _state(ib, skip):
    return ExecuteModelState(
        input_batch=ib,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=torch.zeros(1, 1),
        aux_hidden_states=None,
        finished_req_ids=set(),
        skip_draft_propose=skip,
    )


@pytest.fixture(autouse=True)
def _patch_outputs(monkeypatch):
    monkeypatch.setattr(mr, "AsyncOutput", MagicMock())
    monkeypatch.setattr(mr, "ModelRunnerOutput", MagicMock())
    # the DFlash2 isinstance branch must not fire for the fake speculator
    monkeypatch.setattr(mr, "DFlash2Speculator", type("_NoMatch", (), {}))


def test_sample_tokens_propose_gate_raises_when_released():
    r, spec, ib = _runner(resident=False)
    r.execute_model_state = _state(ib, skip=False)
    with pytest.raises(RuntimeError, match="propose"):
        GPUModelRunner.sample_tokens(r, None)
    assert spec.propose_calls == 0


def test_sample_tokens_propose_runs_when_resident():
    r, spec, ib = _runner(resident=True)
    r.execute_model_state = _state(ib, skip=False)
    GPUModelRunner.sample_tokens(r, None)
    assert spec.propose_calls == 1


def test_skip_draft_propose_does_not_trip_gate_when_released():
    # S1 skip branch: an all-K=0 step never enters the drafter, so a released
    # drafter must not turn a normal nospec step into an error.
    r, spec, ib = _runner(resident=False)
    r.execute_model_state = _state(ib, skip=True)
    GPUModelRunner.sample_tokens(r, None)
    assert spec.propose_calls == 0
    r.draft_tokens_handler.set_draft_tokens.assert_called_once()


def test_dummy_run_dummy_propose_gate_raises_when_released():
    r, spec, ib = _runner(resident=False)
    r.max_num_reqs = 4
    r.decode_query_len = 1
    r.is_first_pp_rank = True
    r.execute_model_state = _state(ib, skip=False)
    # execute_model(dummy_run=True) is a mock; it leaves execute_model_state set
    with pytest.raises(RuntimeError, match="dummy_propose"):
        GPUModelRunner._dummy_run(r, 1)
    assert spec.propose_calls == 0


def test_dummy_run_dummy_propose_runs_when_resident():
    r, spec, ib = _runner(resident=True)
    r.max_num_reqs = 4
    r.decode_query_len = 1
    r.is_first_pp_rank = True
    r.execute_model_state = _state(ib, skip=False)
    GPUModelRunner._dummy_run(r, 1)
    assert spec.propose_calls == 1
