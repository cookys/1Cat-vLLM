# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU real-scheduler tests; no models, CUDA calls, or network required."""

import json
from unittest.mock import patch

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.prefill_cadence import PrefillCadence
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.cadence_step_timer import CadenceStepTimer

from . import utils
from .test_mixed_prefill_budget import complete, factory, resident  # noqa: F401

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def configure(make_scheduler, k, log=True, **kw):
    s = make_scheduler(target=0, max_num_batched_tokens=8192, **kw)
    s.scheduler_config.prefill_cadence_decode_steps = k
    s.scheduler_config.prefill_cadence_step_log = log
    s.prefill_cadence = PrefillCadence(k)
    # Exercise the production 4096 aligned splitting with the real allocator.
    s.need_mamba_block_aligned_split = True
    s.mamba_state_block_size = 4096
    return s


def add(s, name, length):
    r = utils.create_requests(1, num_tokens=length, req_ids=[name], max_tokens=256)[0]
    s.add_request(r)
    return r


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("spec", [0, 7])
def test_real_schedule_spacing_and_bounded_prefill_progress(
    factory,  # noqa: F811
    k,
    async_scheduling,
    spec,  # noqa: F811
):  # noqa: F811
    s = configure(
        factory,
        k,
        async_scheduling=async_scheduling,
        num_speculative_tokens=spec or None,
    )
    d = resident(s)
    # No leftover credit from the decoder's own initial prefill.
    s.prefill_cadence.remaining = 0
    p = add(s, "long", 3 * 4096 + 31)
    prefill_steps, chunk_ends = [], []
    for i in range(4 * (k + 1) + 1):
        if spec and not async_scheduling:
            d.spec_token_ids = list(range(spec))
        step = s.schedule()
        assert step.num_scheduled_tokens["decode"] == spec + 1
        if "long" in step.num_scheduled_tokens:
            prefill_steps.append(i)
            chunk_ends.append(p.num_computed_tokens)
        else:
            assert step.cadence_step["prefill_rows"] == 0
            assert step.cadence_step["hold_prefill"]
        complete(s, step)
        if p.num_output_tokens:
            break
    assert prefill_steps == [j * (k + 1) for j in range(4)]
    assert chunk_ends == [4096, 8192, 12288, 12319]
    assert p.num_output_tokens == 1  # Eventually reaches first output.


@pytest.mark.parametrize("k", [1, 3])
def test_no_decoder_never_delays_cold_prefill(factory, k):  # noqa: F811
    s = configure(factory, k)
    p = add(s, "cold", 9000)
    ends = []
    for _ in range(3):
        step = s.schedule()
        ends.append(p.num_computed_tokens)
        assert not step.cadence_step["hold_prefill"]
        complete(s, step)
    assert ends == [4096, 8192, 9000]


def test_holds_running_and_waiting_in_original_queue_order(factory):  # noqa: F811
    s = configure(factory, 3)
    resident(s)
    s.prefill_cadence.remaining = 0
    p = add(s, "running", 12000)
    complete(s, s.schedule())
    add(s, "waiting1", 9000)
    add(s, "waiting2", 9000)
    s.running.reverse()  # Partial prefill before the decoder.
    before = ([r.request_id for r in s.running], [r.request_id for r in s.waiting])
    for _ in range(3):
        step = s.schedule()
        assert list(step.num_scheduled_tokens) == ["decode"]
        assert p.num_computed_tokens == 4096
        complete(s, step)
        assert (
            [r.request_id for r in s.running],
            [r.request_id for r in s.waiting],
        ) == before
    step = s.schedule()
    assert step.num_scheduled_tokens["running"] == 4096


@pytest.mark.parametrize("termination", ["abort", "ineligible", "output_limit"])
def test_loss_of_eligible_decoder_releases_prefill(factory, termination):  # noqa: F811
    s = configure(factory, 3)
    d = resident(s)
    s.prefill_cadence.remaining = 3
    add(s, "prefill", 9000)
    if termination == "abort":
        s.finish_requests(["decode"], RequestStatus.FINISHED_ABORTED)
    elif termination == "ineligible":
        d.next_decode_eligible_step = s.current_step + 10
    else:
        d.num_output_placeholders = 1
        d.num_computed_tokens = d.num_prompt_tokens + d.max_tokens - 1
    step = s.schedule()
    assert step.num_scheduled_tokens["prefill"] == 4096


def test_empty_allocation_failure_does_not_spin(factory):  # noqa: F811
    s = configure(factory, 3)
    resident(s)
    add(s, "prefill", 9000)
    s.prefill_cadence.remaining = 3
    with patch.object(s.kv_cache_manager, "allocate_slots", return_value=None):
        step = s.schedule()
    assert step.total_num_scheduled_tokens == 0
    assert s.prefill_cadence.remaining == 0
    # The decoder was preempted by the original allocator; no gate deadlock.
    step = s.schedule()
    assert step.total_num_scheduled_tokens > 0


def test_async_inflight_steps_count_fifo_dispatch_not_feedback_twice(factory):  # noqa: F811
    s = configure(factory, 3, async_scheduling=True, num_speculative_tokens=7)
    resident(s)
    s.prefill_cadence.remaining = 0
    add(s, "prefill", 9000)
    mixed = s.schedule()
    assert mixed.cadence_step["prefill_rows"] == 4096
    # Submit next batch before consuming the prior result, as async engine does.
    dec = s.schedule()
    assert dec.cadence_step["prefill_rows"] == 0
    assert s.prefill_cadence.remaining == 2
    complete(s, mixed)
    complete(s, dec)
    assert s.prefill_cadence.remaining == 2  # Feedback doesn't grant extra credit.
    for _ in range(2):
        step = s.schedule()
        assert step.cadence_step["prefill_rows"] == 0
        complete(s, step)
    assert s.schedule().cadence_step["prefill_rows"] == 4096


def test_off_does_not_scan_requests_or_emit_step_metadata(factory):  # noqa: F811
    s = configure(factory, 0, log=False)
    resident(s)
    add(s, "prefill", 9000)
    with patch.object(s, "_mixed_prefill_residents", side_effect=AssertionError("OFF")):
        step = s.schedule()
    assert step.num_scheduled_tokens == {"decode": 1, "prefill": 4096}
    assert step.cadence_step is None


@pytest.mark.parametrize("k", [1, 3])
def test_mutual_exclusion_including_static_p8(k):
    args = dict(
        max_model_len=16384, is_encoder_decoder=False, prefill_cadence_decode_steps=k
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SchedulerConfig(
            **args,
            mixed_prefill_step_latency_ms=1,
            mixed_prefill_min_tokens=512,
            mixed_prefill_max_tokens=512,
        )
    with pytest.raises(ValueError, match="chunked"):
        SchedulerConfig(**args, enable_chunked_prefill=False)
    assert (
        SchedulerConfig(**args).compute_hash()
        == SchedulerConfig(max_model_len=16384, is_encoder_decoder=False).compute_hash()
    )


def test_step_timer_preserves_long_intervals_no_sync_or_request_ids(caplog):
    events = []

    class Event:
        def __init__(self, **kw):
            self.ready = False
            events.append(self)

        def record(self):
            pass

        def query(self):
            return self.ready

        def elapsed_time(self, end):
            assert end.ready
            return 4099.065  # Must not saturate at one second.

        def synchronize(self):
            pytest.fail("must never synchronize")

    out = SchedulerOutput.make_empty()
    out.total_num_scheduled_tokens = 4104
    out.cadence_step = dict(
        step_id=17,
        prefill_rows=4096,
        decode_rows=8,
        decode_req_ids=("private-request-id",),
    )
    timer = CadenceStepTimer(rank=2, max_pending=1)
    with (
        patch("torch.cuda.Event", Event),
        patch("vllm.v1.worker.cadence_step_timer.logger.info") as emit,
        patch("vllm.v1.worker.cadence_step_timer.logger.warning") as warn,
    ):
        timer.begin(out)
        timer.route("NONE", 4104)
        timer.finish()
        assert emit.call_count == 0
        timer.begin(out)  # Bounded queue: explicit gap, not silent drop.
        assert warn.call_count == 1 and len(events) == 2
        events[-1].ready = True
        timer.begin(SchedulerOutput.make_empty())
        row = json.loads(emit.call_args.args[1])
        assert row["stream_ms"] == 4099.065
        assert row["step_id"] == 17 and row["rank"] == 2
        assert row["route"] == "NONE" and row["padded_rows"] == 4104
        assert "private-request-id" not in emit.call_args.args[1]
        assert not timer.pending


def test_completion_counts_match_original_step_even_if_output_reordered(factory):  # noqa: F811
    s = configure(factory, 1)
    resident(s)
    s.prefill_cadence.remaining = 0
    add(s, "prefill", 9000)
    step = s.schedule()
    output = ModelRunnerOutput(
        req_ids=["prefill", "decode"],
        req_id_to_index={"prefill": 0, "decode": 1},
        sampled_token_ids=[[], [42]],
    )
    with patch("vllm.v1.core.sched.scheduler.logger.info") as emit:
        s.update_from_output(step, output)
    message = next(
        c for c in emit.call_args_list if "PREFILL_CADENCE_COMPLETE" in c.args[0]
    )
    assert message.args[1:4] == (step.cadence_step["step_id"], 1, 1)


@pytest.mark.parametrize("k", [1, 3])
def test_abort_discards_raw_sampled_tokens_from_completion_counts(factory, k):  # noqa: F811
    s = configure(factory, k)
    resident(s)
    step = s.schedule()
    s.finish_requests(["decode"], RequestStatus.FINISHED_ABORTED)
    out = ModelRunnerOutput(
        req_ids=["decode"], req_id_to_index={"decode": 0}, sampled_token_ids=[[42]]
    )
    with patch("vllm.v1.core.sched.scheduler.logger.info") as emit:
        s.update_from_output(step, out)
    msg = next(
        c for c in emit.call_args_list if "PREFILL_CADENCE_COMPLETE" in c.args[0]
    )
    assert msg.args[2] == 0


def test_cli_knobs_parse_and_defaults_are_off():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    off = EngineArgs.from_cli_args(parser.parse_args([]))
    on = EngineArgs.from_cli_args(
        parser.parse_args(
            ["--prefill-cadence-decode-steps", "3", "--prefill-cadence-step-log"]
        )
    )
    assert off.prefill_cadence_decode_steps == 0
    assert not off.prefill_cadence_step_log
    assert on.prefill_cadence_decode_steps == 3 and on.prefill_cadence_step_log
    import torch

    assert not torch.cuda.is_initialized()


def test_real_mamba_spec_4096_chunks(factory):  # noqa: F811
    import torch

    from vllm.v1.kv_cache_interface import MambaSpec

    spec = MambaSpec(
        block_size=4096,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    s = factory(target=0, max_num_batched_tokens=8192, kv_cache_spec=spec)
    s.prefill_cadence = PrefillCadence(1)
    s.scheduler_config.prefill_cadence_decode_steps = 1
    s.scheduler_config.prefill_cadence_step_log = True
    # CacheConfig's global mode must match the real Mamba group used above.
    s.need_mamba_block_aligned_split = True
    assert s.mamba_state_block_size == 4096
    resident(s)
    s.prefill_cadence.remaining = 0
    p = add(s, "p", 9000)
    seen = []
    for _ in range(5):
        step = s.schedule()
        seen.append(step.num_scheduled_tokens.get("p", 0))
        complete(s, step)
    assert seen == [4096, 0, 4096, 0, 808]
    assert p.num_output_tokens == 1


def test_readout_long_steps_and_cadence_decode_are_interfered():
    from benchmarks.prefill_cadence_readout import analyze

    lines = []
    for sid, prefill, pending, ms in [
        (1, 4096, 1, 2900),
        (2, 0, 1, 170),
        (3, 0, 0, 170),
    ]:
        for rank in range(4):
            row = dict(
                step_id=sid,
                rank=rank,
                prefill_rows=prefill,
                decode_rows=80,
                decode_reqs=10,
                pending_prefill_reqs=pending,
                stream_ms=ms,
                route="NONE" if prefill else "FULL",
                stream_gap_ms=0,
            )
            lines.append("PREFILL_CADENCE_STEP " + json.dumps(row))
        lines.append(
            f"PREFILL_CADENCE_COMPLETE step_id={sid} decode_accepted_tokens=40"
        )
    data = analyze("\n".join(lines))
    assert data["status"] == "INCONCLUSIVE"  # Three steps cannot pass coverage.
    assert data["regimes"]["interfered"]["steps"] == 2
    assert data["phases"]["mixed"]["stream_ms_p50"] == 2900
    assert data["regimes"]["interfered"]["per_decoder_tok_s"] == pytest.approx(
        80 / 30.7
    )
    assert analyze("\n".join(lines[:-1]))["status"] == "INCONCLUSIVE"
    assert analyze("\n".join(lines + [lines[0]]))["status"] == "INCONCLUSIVE"


def readout_fixture(mixed=20, decode=20, cold_count=10):
    lines = []
    for sid in range(mixed + decode):
        prefill = sid < mixed
        for rank in range(4):
            lines.append(
                "PREFILL_CADENCE_STEP "
                + json.dumps(
                    dict(
                        step_id=sid,
                        rank=rank,
                        prefill_rows=4096 if prefill else 0,
                        decode_rows=80,
                        decode_reqs=10,
                        pending_prefill_reqs=int(prefill),
                        stream_ms=2000 if prefill else 170,
                        stream_gap_ms=5,
                        route="NONE" if prefill else "FULL",
                    )
                )
            )
        lines.append(
            f"PREFILL_CADENCE_COMPLETE step_id={sid} decode_accepted_tokens=40"
        )
    cold = {
        "cohorts": {
            name: [
                dict(ok=True, ttft_s=3 + i, cached_tokens=0) for i in range(cold_count)
            ]
            for name in ("128K", "200K")
        }
    }
    return "\n".join(lines), cold


@pytest.mark.parametrize("mixed,decode", [(19, 20), (20, 19)])
def test_readout_nineteen_phase_and_regime_steps_are_inconclusive(mixed, decode):
    from benchmarks.prefill_cadence_readout import analyze

    log, cold = readout_fixture(mixed, decode)
    result = analyze(log, cold_evidence=cold)
    assert result["status"] == "INCONCLUSIVE"
    assert any(p.startswith("insufficient_phases:") for p in result["problems"])
    assert any(p.startswith("insufficient_regimes:") for p in result["problems"])


def test_readout_twenty_steps_ten_cold_complete_without_pure_prefill():
    from benchmarks.prefill_cadence_readout import analyze

    log, cold = readout_fixture()
    result = analyze(log, cold_evidence=cold)
    assert result["status"] == "COMPLETE"
    assert result["phases"]["prefill"]["steps"] == 0
    assert result["comparison_fields"] == {
        r: f"regimes.{r}.per_decoder_tok_s_with_gap" for r in ("steady", "interfered")
    }
    assert result["cold"]["cohorts"]["128K"]["ttft_s_p90"] == pytest.approx(11.1)


@pytest.mark.parametrize(
    "case", ["nine", "missing", "unknown_cache", "warm", "nan", "outcome"]
)
@pytest.mark.parametrize("cohort", ["128K", "200K"])
def test_readout_cold_evidence_required_in_status(case, cohort):
    from benchmarks.prefill_cadence_readout import analyze

    log, cold = readout_fixture()
    row = cold["cohorts"][cohort][0]
    if case == "nine":
        cold["cohorts"][cohort].pop()
    elif case == "missing":
        cold = None
    elif case == "unknown_cache":
        row["cached_tokens"] = None
    elif case == "warm":
        row["cached_tokens"] = 4097
    elif case == "nan":
        row["ttft_s"] = float("nan")
    else:
        del row["ok"]
    result = analyze(log, cold_evidence=cold)
    assert result["status"] == "INCONCLUSIVE"
    assert result["cold"]["cohorts"][cohort]["ttft_s_p90"] is None


@pytest.mark.parametrize("missing", ["phase", "regime"])
def test_readout_phase_and_regime_gates_are_independent(missing):
    from benchmarks.prefill_cadence_readout import analyze

    log, cold = readout_fixture()
    lines = []
    for line in log.splitlines():
        if "PREFILL_CADENCE_STEP " in line:
            row = json.loads(line.split(" ", 1)[1])
            if missing == "phase" and 1 <= row["step_id"] < 20:
                row["prefill_rows"] = 0
            if missing == "regime" and row["step_id"] >= 20:
                row["pending_prefill_reqs"] = 1
            line = "PREFILL_CADENCE_STEP " + json.dumps(row)
        lines.append(line)
    result = analyze("\n".join(lines), cold_evidence=cold)
    assert result["status"] == "INCONCLUSIVE"
    expected = "phases" if missing == "phase" else "regimes"
    other = "regimes" if missing == "phase" else "phases"
    assert any(p.startswith(f"insufficient_{expected}:") for p in result["problems"])
    assert not any(p.startswith(f"insufficient_{other}:") for p in result["problems"])


def test_readout_failed_cold_request_is_not_removed_from_tail():
    from benchmarks.prefill_cadence_readout import analyze

    log, cold = readout_fixture()
    cold["cohorts"]["128K"][0].update(ok=False, error="timeout", ttft_s=None)
    result = analyze(log, cold_evidence=cold)
    assert result["status"] == "FAIL"
    assert result["cold"]["cohorts"]["128K"]["requests"] == 10
    assert result["cold"]["cohorts"]["128K"]["ttft_s_p90"] is None
