# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1 (P2-5): every drafter GPU entry point fails loud (RuntimeError)
while the drafter weights are released; CPU-only (gate is the first statement
or sits right before the graph replay, so no GPU state is needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _spec(cls, resident):
    s = object.__new__(cls)
    s.set_drafter_resident(resident)
    return s


class _Graph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def _calls(spec):
    ib = SimpleNamespace(
        num_reqs=1,
        num_tokens=8,
        num_draft_tokens=7,
        is_prefilling_np=np.array([False]),
        num_scheduled_tokens=np.array([8]),
        positions=None,
    )
    spec.num_query_per_req = 8
    graph = _Graph()
    spec._context_compute_graphs = {8: graph}
    return graph, {
        "propose": lambda: spec.propose(ib, {}, {}, None, None, None, None, None, None, None, None),
        "_run_model": lambda: spec._run_model(8, None, None, None),
        "_precompute_context_kv": lambda: spec._precompute_context_kv(None, None, None),
        "capture": lambda: spec.capture(),
        "prepare_target_context": lambda: spec.prepare_target_context(ib, None, None),
    }


@pytest.mark.parametrize("cls", [DFlashSpeculator, DFlash2Speculator])
def test_gate_closed_every_entry_raises(cls):
    spec = _spec(cls, False)
    graph, calls = _calls(spec)
    if cls is DFlashSpeculator:
        calls.pop("prepare_target_context")  # dflash2-only entry
    for name, fn in calls.items():
        with pytest.raises(RuntimeError, match=name):
            fn()
    assert graph.replays == 0


def test_prepare_target_context_gate_before_graph_replay():
    spec = _spec(DFlash2Speculator, False)
    graph, calls = _calls(spec)
    with pytest.raises(RuntimeError, match="prepare_target_context"):
        calls["prepare_target_context"]()
    assert graph.replays == 0


def test_prepare_target_context_early_return_does_not_trip_gate():
    # nospec batch (no K=7 rows): the entry returns before touching the
    # drafter, so a closed gate must not turn a normal nospec step into an error.
    spec = _spec(DFlash2Speculator, False)
    graph, _ = _calls(spec)
    ib = SimpleNamespace(
        num_reqs=1, num_tokens=8, num_draft_tokens=0,
        is_prefilling_np=np.array([False]), num_scheduled_tokens=np.array([8]),
    )
    spec._prepared_context_batch = object()
    spec.prepare_target_context(ib, None, None)
    assert graph.replays == 0 and spec._prepared_context_batch is None


def test_gate_reopens():
    spec = _spec(DFlash2Speculator, False)
    spec.set_drafter_resident(True)
    spec._require_drafter_resident("x")  # no raise


def test_runner_gate_and_setter():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    spec = _spec(DFlash2Speculator, True)
    runner = SimpleNamespace(speculator=spec)
    GPUModelRunner.set_drafter_resident(runner, False)
    assert spec.drafter_resident is False
    with pytest.raises(RuntimeError, match="propose"):
        GPUModelRunner._require_drafter_resident(runner, "propose")
    GPUModelRunner.set_drafter_resident(runner, True)
    GPUModelRunner._require_drafter_resident(runner, "propose")


def test_runner_setter_without_releasable_drafter_raises():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    for sp in (None, object()):
        with pytest.raises(RuntimeError):
            GPUModelRunner.set_drafter_resident(SimpleNamespace(speculator=sp), False)
