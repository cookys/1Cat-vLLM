# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU scheduler/KV allocator; connector I/O is mocked, no CUDA."""

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from vllm.v1.request import RequestStatus

from . import utils
from .test_mixed_prefill_budget import complete, factory  # noqa: F401


def attach(s, enabled, diagnostics=False):
    connector = s.connector
    connector.connector_scheduler = NS(host_parking=True)
    s.vllm_config.kv_transfer_config.kv_connector_extra_config.update(
        host_parking_async_admission=enabled, parking_diagnostics=diagnostics
    )
    s._configure_host_parking_admission()
    connector.update_state_after_alloc = Mock()
    return connector


def request(s, name):
    r = utils.create_requests(1, num_tokens=8192, req_ids=[name], max_tokens=8)[0]
    s.add_request(r)
    return r


def make(make_scheduler, enabled, **kw):
    s = make_scheduler(
        target=0,
        max_num_batched_tokens=4096,
        use_kv_connector=utils.mock_kv(4096, True),
        **kw,
    )
    attach(s, enabled)
    return s


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_completed_receive_tail_does_not_block_new_load(
    factory, enabled, async_scheduling  # noqa: F811
):
    s = make(factory, enabled, async_scheduling=async_scheduling)
    old = request(s, "completed-load")
    initial = s.schedule()
    assert initial.total_num_scheduled_tokens == 0
    assert old.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    s.finished_recving_kv_req_ids.add(old.request_id)
    new = request(s, "new-load")
    s.connector.update_state_after_alloc.reset_mock()
    step = s.schedule()
    assert step.num_scheduled_tokens == {old.request_id: 4096}
    assert len(s.running) == 1 and new not in s.running
    calls = s.connector.update_state_after_alloc.call_args_list
    submitted = [c.args[0].request_id for c in calls if c.args[2] > 0]
    assert submitted == ([new.request_id] if enabled else [])
    assert new.status == (
        RequestStatus.WAITING_FOR_REMOTE_KVS if enabled else RequestStatus.WAITING
    )
    if enabled:
        assert new.num_computed_tokens == 4096
        blocks = s.kv_cache_manager.get_block_ids(new.request_id)[0]
        assert len(blocks) == 4096 // 16  # Destination pages, no 4096 tail allocation.


@pytest.mark.parametrize("enabled", [False, True])
def test_running_tail_does_not_block_new_load(factory, enabled):  # noqa: F811
    s = make(factory, enabled)
    s.connector.get_num_new_matched_tokens = Mock(return_value=(0, False))
    old = request(s, "running")
    first = s.schedule()
    complete(s, first)
    assert old in s.running and old.num_computed_tokens == 4096
    s.connector.get_num_new_matched_tokens.return_value = (4096, True)
    new = request(s, "new-load")
    s.connector.update_state_after_alloc.reset_mock()
    step = s.schedule()
    assert step.num_scheduled_tokens == {old.request_id: 4096}
    assert s.connector.update_state_after_alloc.call_count == int(enabled)
    assert new not in s.running


def test_zero_budget_skips_miss_and_pending_without_promoting_or_looping(factory):  # noqa: F811
    s = make(factory, True)
    old = request(s, "ready")
    pending = request(s, "still-loading")
    s.schedule()
    s.finished_recving_kv_req_ids.add(old.request_id)
    miss = request(s, "cold-miss")
    hit = request(s, "new-hit")
    s.connector.get_num_new_matched_tokens = Mock(
        side_effect=lambda r, n: (0, False) if r is miss else (4096, True)
    )
    s.connector.update_state_after_alloc.reset_mock()
    step = s.schedule()
    assert step.num_scheduled_tokens == {old.request_id: 4096}
    assert miss.num_computed_tokens == 0 and miss not in s.running
    assert pending.status == hit.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert [
        c.args[0] for c in s.connector.get_num_new_matched_tokens.call_args_list
    ] == [miss, hit]
    assert [c.args[0] for c in s.connector.update_state_after_alloc.call_args_list] == [
        old,
        hit,
    ]
    assert len(s.waiting) + len(s.skipped_waiting) == 3


@pytest.mark.parametrize("failure", ["capacity", "seq_limit", "paused"])
def test_async_admission_preserves_resource_and_pause_gates(factory, failure):  # noqa: F811
    from vllm.v1.core.sched.interface import PauseState

    s = make(factory, True, max_num_seqs=1 if failure == "seq_limit" else 16)
    old = request(s, "ready")
    s.schedule()
    s.finished_recving_kv_req_ids.add(old.request_id)
    new = request(s, "new")
    original = s.kv_cache_manager.allocate_slots
    s.connector.update_state_after_alloc.reset_mock()
    if failure == "paused":
        s._pause_state = PauseState.PAUSED_ALL
    with patch.object(
        s.kv_cache_manager,
        "allocate_slots",
        side_effect=lambda r, *a, **kw: None
        if failure == "capacity" and r is new
        else original(r, *a, **kw),
    ):
        s.schedule()
    assert new.status == RequestStatus.WAITING
    assert all(
        c.args[0] is not new
        for c in s.connector.update_state_after_alloc.call_args_list
    )


def test_diagnostics_timestamps_and_default_off(factory):  # noqa: F811
    s = make(factory, False)
    assert not s.host_parking_async_admission and not s.host_parking_diagnostics
    attach(s, True, diagnostics=True)
    with patch("vllm.v1.core.sched.scheduler.logger.info") as log:
        r = request(s, "timed")
        s.schedule()
    assert r._parking_engine_received_ns > 0
    assert any("engine_received" in c.args[0] for c in log.call_args_list)
    assert any("waiting_scan" in c.args[0] for c in log.call_args_list)


@pytest.mark.parametrize("value", [1, "1", "true", None])
def test_admission_flag_rejects_non_boolean(factory, value):  # noqa: F811
    s = make(factory, False)
    s.vllm_config.kv_transfer_config.kv_connector_extra_config[
        "host_parking_async_admission"
    ] = value
    with pytest.raises(ValueError, match="JSON boolean"):
        s._configure_host_parking_admission()


def test_admission_flag_rejects_other_connectors(factory):  # noqa: F811
    s = make(factory, False)
    s.connector.connector_scheduler.host_parking = False
    s.vllm_config.kv_transfer_config.kv_connector_extra_config[
        "host_parking_async_admission"
    ] = True
    with pytest.raises(ValueError, match="HostParkingSpec"):
        s._configure_host_parking_admission()


@pytest.mark.parametrize("failure", ["kv_capacity", "connector_pending"])
def test_failed_reservation_retries_once_next_step(factory, failure):  # noqa: F811
    s = make(factory, True)
    old = request(s, "ready")
    s.schedule()
    s.finished_recving_kv_req_ids.add(old.request_id)
    new = request(s, "new")
    allocate = s.kv_cache_manager.allocate_slots
    lookup = s.connector.get_num_new_matched_tokens
    s.connector.update_state_after_alloc.reset_mock()
    with (
        patch.object(
            s.kv_cache_manager,
            "allocate_slots",
            side_effect=lambda r, *a, **kw: None
            if r is new and failure == "kv_capacity"
            else allocate(r, *a, **kw),
        ) as alloc,
        patch.object(
            s.connector,
            "get_num_new_matched_tokens",
            side_effect=lambda r, n: (None, False)
            if r is new and failure == "connector_pending"
            else lookup(r, n),
        ) as get,
    ):
        step = s.schedule()
        assert sum(c.args[0] is new for c in get.call_args_list) == 1
        assert sum(c.args[0] is new for c in alloc.call_args_list) <= 1
    assert new.status == RequestStatus.WAITING
    assert not any(
        c.args[0] is new for c in s.connector.update_state_after_alloc.call_args_list
    )
    complete(s, step)
    s.schedule()
    assert new.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert (
        sum(
            c.args[0] is new
            for c in s.connector.update_state_after_alloc.call_args_list
        )
        == 1
    )


def off_replay(make_scheduler):
    """Full output/state ledger; fixture generated from 7982f2cfc scheduler."""
    s = make_scheduler(
        target=0,
        max_num_batched_tokens=4096,
        use_kv_connector=utils.mock_kv(4096, True),
    )
    request(s, "first")
    traces = []
    for index in range(6):
        if index == 1:
            request(s, "second")
            request(s, "third")
        for r in s.skipped_waiting:
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                s.finished_recving_kv_req_ids.add(r.request_id)
        out = s.schedule()
        output = dataclasses.asdict(out)
        output["kv_connector_metadata"] = out.kv_connector_metadata.__dict__
        for new in output["scheduled_new_reqs"]:
            new.pop("sampling_params")
        complete(s, out)
        pool = s.kv_cache_manager.block_pool
        traces.append(
            dict(
                output=output,
                running=[r.request_id for r in s.running],
                waiting=[r.request_id for r in s.waiting],
                skipped=[r.request_id for r in s.skipped_waiting],
                requests={
                    rid: [
                        int(r.status),
                        r.num_computed_tokens,
                        r.num_output_placeholders,
                        list(r.output_token_ids),
                    ]
                    for rid, r in s.requests.items()
                },
                refcounts=[b.ref_cnt for b in pool.blocks],
                free=pool.get_num_free_blocks(),
            )
        )
    return json.dumps(
        traces, sort_keys=True, default=lambda x: sorted(x), separators=(",", ":")
    )


def test_flag_off_matches_pre_fix_native_connector_fixture(factory):  # noqa: F811
    fixture = json.loads(
        Path(__file__).with_name("host_parking_admission_off.json").read_text()
    )
    assert fixture["base"] == "7982f2cfc"
    assert hashlib.sha256(off_replay(factory).encode()).hexdigest() == fixture["sha256"]
