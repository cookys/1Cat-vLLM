# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only scheduler, feedback, and disabled-path regressions for P8."""

import json
from unittest.mock import patch

import pytest

from vllm.config import SchedulerConfig, SpeculativeConfig
from vllm.v1.core.sched.mixed_prefill import MixedPrefillBudget
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import MixedPrefillTiming, ModelRunnerOutput
from vllm.v1.worker.mixed_prefill import MixedPrefillTimer

from . import utils

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def factory(tmp_path):
    # Local config only: neither model weights nor network access is needed.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 262144,
                "vocab_size": 128,
                "bos_token_id": 2,
                "eos_token_id": 2,
                "pad_token_id": 1,
            }
        )
    )

    def make(target=250, cap=512, floor=128, **kwargs):
        config = lambda **kw: SchedulerConfig(
            mixed_prefill_step_latency_ms=target,
            mixed_prefill_max_tokens=cap,
            mixed_prefill_min_tokens=floor,
            **kw,
        )
        options = dict(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_num_batched_tokens=1024,
            max_model_len=16384,
        )
        options.update(kwargs)

        # ngram_gpu uses the same scheduler speculation contract and is valid
        # under async scheduling; no proposer is constructed in these tests.
        def spec_config(**kw):
            return SpeculativeConfig(
                **{**kw, "model": "ngram_gpu", "method": "ngram_gpu"}
            )

        with (
            patch.object(utils, "SchedulerConfig", side_effect=config),
            patch.object(utils, "SpeculativeConfig", side_effect=spec_config),
        ):
            scheduler = utils.create_scheduler(**options)
        scheduler.mixed_prefill_enabled = True
        return scheduler

    return make


def complete(scheduler, step):
    ids = list(step.num_scheduled_tokens)
    tokens = [[] if scheduler.requests[rid].is_prefill_chunk else [42] for rid in ids]
    output = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=tokens,
    )
    return scheduler.update_from_output(step, output)


def resident(scheduler, name="decode"):
    req = utils.create_requests(1, num_tokens=32, req_ids=[name], max_tokens=128)[0]
    scheduler.add_request(req)
    complete(scheduler, scheduler.schedule())
    return req


def add_prefill(scheduler, name="long", length=1536):
    req = utils.create_requests(1, num_tokens=length, req_ids=[name])[0]
    scheduler.add_request(req)
    return req


def test_budget_adapts_but_cannot_exceed_hard_cap():
    budget = MixedPrefillBudget(1024, 250)
    assert budget.tokens == 512
    budget.update(MixedPrefillTiming(512, 80, 512, 400))
    assert budget.tokens == 320
    budget.update(MixedPrefillTiming(8, 80, 320, 150))
    assert budget.tokens == 320
    for _ in range(30):
        n = budget.tokens
        budget.update(MixedPrefillTiming(n, 80, n, n * 0.01))
    assert budget.tokens == 1024
    budget.update(MixedPrefillTiming(1024, 80, 1024, 1e9))
    assert budget.tokens == 128  # Progress floor, not a hard latency guarantee.


def test_decode_floor_above_target_cannot_shrink_allowance_below_minimum():
    budget = MixedPrefillBudget(512, 250, 128)
    for _ in range(100):
        n = budget.tokens
        budget.update(MixedPrefillTiming(n, 80, n, 400 + n * 0.1))
    assert budget.tokens == 128
    fixed = MixedPrefillBudget(1024, 250, 1024)
    fixed.update(MixedPrefillTiming(1024, 80, 1024, 900))
    assert fixed.tokens == 1024 and fixed.ms_per_token is None


def test_invalid_progress_floor_fails_before_loading():
    with pytest.raises(ValueError, match="mixed_prefill_min_tokens"):
        SchedulerConfig(
            max_model_len=4096,
            is_encoder_decoder=False,
            mixed_prefill_step_latency_ms=250,
            mixed_prefill_max_tokens=64,
        )
    with pytest.raises(ValueError, match="effective maximum"):
        MixedPrefillBudget(64, 250, 128)


def test_static_cap_ignores_all_feedback(factory):
    scheduler = factory(cap=512, floor=512)
    resident(scheduler)
    add_prefill(scheduler, length=10000)
    for elapsed in [1e-9, 1e9, 100, 900, float("nan")]:
        scheduler.mixed_prefill_budget.update(MixedPrefillTiming(512, 8, 512, elapsed))
        step = scheduler.schedule()
        assert step.num_scheduled_tokens["long"] == 512
        complete(scheduler, step)
    assert scheduler.mixed_prefill_budget.ms_per_token is None


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_is_ignored(elapsed):
    budget = MixedPrefillBudget(512, 250)
    budget.update(MixedPrefillTiming(512, 8, 512, elapsed))
    assert budget.tokens == 512


def test_zero_target_ignores_feedback_and_preserves_graph_hash():
    budget = MixedPrefillBudget(512, 0)
    budget.update(MixedPrefillTiming(512, 8, 512, 1000))
    assert budget.tokens == 512
    args = dict(max_model_len=4096, is_encoder_decoder=False)
    off = SchedulerConfig(**args)
    on = SchedulerConfig(
        **args, mixed_prefill_step_latency_ms=250, mixed_prefill_max_tokens=1024
    )
    assert off.mixed_prefill_step_latency_ms == 0
    assert off.compute_hash() == on.compute_hash()


def test_timer_never_waits_and_keeps_delayed_sample_counts():
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.ready = False
            events.append(self)

        def record(self):
            pass

        def query(self):
            return self.ready

        def elapsed_time(self, end):
            assert end.ready
            return 300.0

        def synchronize(self):
            pytest.fail("feedback must not synchronize")

    timer = MixedPrefillTimer()
    mixed = SchedulerOutput.make_empty()
    mixed.mixed_prefill_tokens = mixed.mixed_prefill_budget = 512
    mixed.mixed_decode_tokens = 80
    with patch("torch.cuda.Event", Event):
        timer.begin(SchedulerOutput.make_empty())
        assert timer.finish() is None and not events
        timer.begin(mixed)
        assert timer.finish() is None
        events[-1].ready = True
        timer.begin(SchedulerOutput.make_empty())
        assert timer.finish() == MixedPrefillTiming(512, 80, 512, 300)
        assert len(events) == 2
        # An uncompleted execute is abandoned, not associated with another step.
        timer.begin(mixed)
        timer.begin(SchedulerOutput.make_empty())
        assert timer.finish() is None


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("spec", [0, 4, 7])
def test_prefill_before_decoder_preserves_verify_rows(factory, async_scheduling, spec):
    scheduler = factory(
        async_scheduling=async_scheduling, num_speculative_tokens=spec or None
    )
    decode = resident(scheduler)
    decode.spec_token_ids = list(range(spec))
    add_prefill(scheduler)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"decode": 1 + spec, "long": 512}
    complete(scheduler, first)
    decode.spec_token_ids = list(range(spec))
    scheduler.running.reverse()
    step = scheduler.schedule()
    assert step.num_scheduled_tokens["decode"] == 1 + spec
    assert step.num_scheduled_tokens["long"] == 512
    assert step.mixed_prefill_tokens == 512
    assert step.mixed_decode_tokens == 1 + spec


def test_reserve_limits_prefill_when_global_budget_is_tight(factory):
    scheduler = factory(cap=1024, floor=16, max_num_batched_tokens=64)
    decode = resident(scheduler)
    decode.spec_token_ids = list(range(7))
    add_prefill(scheduler)
    complete(scheduler, scheduler.schedule())
    decode.spec_token_ids = list(range(7))
    scheduler.running.reverse()
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"long": 56, "decode": 8}


def test_running_and_waiting_prefills_share_one_cap(factory):
    scheduler = factory()
    resident(scheduler)
    add_prefill(scheduler, "tail", 600)
    complete(scheduler, scheduler.schedule())
    add_prefill(scheduler, "waiting", 2000)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"decode": 1, "tail": 88, "waiting": 424}
    assert step.mixed_prefill_tokens == 512


def test_pure_prefill_and_off_keep_normal_budget(factory):
    scheduler = factory()
    add_prefill(scheduler)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"long": 1024}
    assert step.mixed_prefill_tokens == 0
    scheduler = factory(target=0)
    resident(scheduler)
    add_prefill(scheduler)
    with patch.object(
        scheduler,
        "_mixed_prefill_residents",
        side_effect=AssertionError("OFF must not scan"),
    ):
        step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"decode": 1, "long": 1023}
    assert step.mixed_prefill_tokens == step.mixed_decode_tokens == 0


def test_ineligible_async_decoder_does_not_reserve_or_limit_prefill(factory):
    scheduler = factory()
    decode = resident(scheduler)
    add_prefill(scheduler)
    decode.next_decode_eligible_step = scheduler.current_step + 2
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"long": 1024}
    assert step.mixed_prefill_tokens == 0


def test_async_at_output_limit_and_max_length_have_no_reserve(factory):
    scheduler = factory()
    decode = resident(scheduler)
    decode.num_output_placeholders = 1
    decode.num_computed_tokens = decode.num_prompt_tokens + decode.max_tokens - 1
    assert scheduler._mixed_prefill_residents()[1] == {}
    decode.num_output_placeholders = 0
    decode.num_computed_tokens = scheduler.max_model_len - 1
    assert scheduler._mixed_prefill_residents()[1] == {}


def test_failed_prefill_allocation_does_not_preempt_decoder(factory):
    scheduler = factory()
    resident(scheduler)
    add_prefill(scheduler)
    complete(scheduler, scheduler.schedule())
    scheduler.running.reverse()
    allocate = scheduler.kv_cache_manager.allocate_slots

    def fail_prefill(req, *a, **kw):
        return None if req.request_id == "long" else allocate(req, *a, **kw)

    with patch.object(scheduler.kv_cache_manager, "allocate_slots", fail_prefill):
        step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"decode": 1}
    assert not step.preempted_req_ids
    assert [r.request_id for r in scheduler.running] == ["long", "decode"]


def test_completed_mixed_stats_are_drained_once(factory):
    scheduler = factory()
    resident(scheduler)
    add_prefill(scheduler)
    step = scheduler.schedule()
    outputs = complete(scheduler, step)
    stats = [out.scheduler_stats for out in outputs.values() if out.scheduler_stats]
    assert len(stats) == 1
    assert stats[0].mixed_prefill_steps == 1
    assert stats[0].mixed_prefill_tokens == 512
    assert scheduler.make_stats().mixed_prefill_steps == 0
    assert scheduler.make_stats().mixed_prefill_tokens == 0


def test_feedback_is_applied_to_next_budget_not_delayed_sample_counts(factory):
    scheduler = factory()
    resident(scheduler)
    add_prefill(scheduler, length=4000)
    step = scheduler.schedule()
    output = ModelRunnerOutput(
        req_ids=["decode", "long"],
        req_id_to_index={"decode": 0, "long": 1},
        sampled_token_ids=[[43], []],
        mixed_prefill_timing=MixedPrefillTiming(512, 8, 512, 400),
    )
    scheduler.update_from_output(step, output)
    assert scheduler.schedule().mixed_prefill_tokens == 320


def test_async_inflight_verify_is_reserved_before_earlier_prefill(factory):
    scheduler = factory(async_scheduling=True, num_speculative_tokens=7)
    resident(scheduler)
    inflight = scheduler.schedule()
    add_prefill(scheduler)
    mixed = scheduler.schedule()
    assert mixed.num_scheduled_tokens == {"decode": 8, "long": 512}
    complete(scheduler, inflight)
    complete(scheduler, mixed)
    scheduler.running.reverse()
    next_step = scheduler.schedule()
    assert next_step.num_scheduled_tokens == {"long": 512, "decode": 8}


def test_mixed_cap_keeps_full_isl_admission_and_lookahead(factory):
    scheduler = factory(num_speculative_tokens=7)
    scheduler.scheduler_reserve_full_isl = True
    resident(scheduler)
    add_prefill(scheduler, length=8000)
    allocate = scheduler.kv_cache_manager.allocate_slots
    with patch.object(
        scheduler.kv_cache_manager, "allocate_slots", wraps=allocate
    ) as spy:
        step = scheduler.schedule()
    call = next(c for c in spy.call_args_list if c.args[0].request_id == "long")
    assert call.kwargs["full_sequence_must_fit"] is True
    assert call.args[1] == 512
    assert step.num_scheduled_tokens["long"] == 512
    call = next(c for c in spy.call_args_list if c.args[0].request_id == "decode")
    assert call.kwargs["num_lookahead_tokens"] == scheduler.num_lookahead_tokens


def test_prometheus_completed_mixed_counters(factory):
    from prometheus_client import REGISTRY

    from vllm.v1.metrics.loggers import PrometheusStatLogger, unregister_vllm_metrics

    scheduler = factory()
    resident(scheduler)
    add_prefill(scheduler)
    stats = next(
        out.scheduler_stats
        for out in complete(scheduler, scheduler.schedule()).values()
        if out.scheduler_stats
    )
    logger = PrometheusStatLogger(scheduler.vllm_config)
    try:
        logger.record(stats, None, engine_idx=0)
        logger.record(scheduler.make_stats(), None, engine_idx=0)
        labels = dict(zip(("model_name", "engine"), logger.per_engine_labelvalues[0]))
        assert REGISTRY.get_sample_value("vllm:mixed_prefill_steps_total", labels) == 1
        assert (
            REGISTRY.get_sample_value("vllm:mixed_prefill_tokens_total", labels) == 512
        )
    finally:
        unregister_vllm_metrics()


def test_cli_import_and_roundtrip():
    from vllm import envs
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert (
        EngineArgs.from_cli_args(parser.parse_args([])).mixed_prefill_step_latency_ms
        == 0
    )
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--mixed-prefill-step-latency-ms",
                "250",
                "--mixed-prefill-max-tokens",
                "1024",
            ]
        )
    )
    assert args.mixed_prefill_step_latency_ms == 250
    assert args.mixed_prefill_max_tokens == 1024
    assert envs.VLLM_USE_V2_MODEL_RUNNER in (None, False, True)


@pytest.mark.parametrize("blocks", [80, 7000])
def test_ten_verifiers_with_100k_prefill_and_kv_pressure(factory, blocks):
    scheduler = factory(num_speculative_tokens=7, num_blocks=blocks)
    scheduler.scheduler_reserve_full_isl = False
    decoders = [resident(scheduler, f"d{i}") for i in range(10)]
    for request in decoders:
        request.max_tokens = 1000
    add_prefill(scheduler, length=100000)
    progress = []
    for _ in range(50):
        for request in decoders:
            request.spec_token_ids = list(range(7))
        step = scheduler.schedule()
        assert all(step.num_scheduled_tokens.get(r.request_id) == 8 for r in decoders)
        assert step.num_scheduled_tokens.get("long", 0) <= 512
        progress.append(step.num_scheduled_tokens.get("long", 0))
        complete(scheduler, step)
        n = scheduler.mixed_prefill_budget.tokens
        scheduler.mixed_prefill_budget.update(MixedPrefillTiming(n, 80, n, 450))
    assert scheduler.mixed_prefill_budget.tokens == 128
    assert sum(progress) > 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() >= 0
    if blocks == 7000:
        assert progress[-20:] == [128] * 20
