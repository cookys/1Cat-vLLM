# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1 (P2-4): ``spec_inflight`` counts K>0 requests in running +
waiting + skipped_waiting (a preempted K=7 request still counts); CPU-only."""

import pytest

from tests.v1.core import utils
from tests.v1.core.test_spec_hot_switch import complete, sync_scheduler
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]



def _add(s, rid, spec_on, n=30):
    s.set_spec_enabled(spec_on)
    (r,) = utils.create_requests(1, num_tokens=n, req_ids=[rid], max_tokens=64)
    s.add_request(r)
    return r


def test_inflight_counts_all_three_queues():
    s = sync_scheduler()
    a = _add(s, "run7", True)
    complete(s, s.schedule())  # a -> running
    assert a in s.running
    w = _add(s, "wait7", True)  # waiting K=7
    sk = _add(s, "skip7", True)
    # park one request in skipped_waiting
    s.waiting.remove_request(sk)
    s.skipped_waiting.add_request(sk)
    _add(s, "wait0", False)  # waiting K=0 must not count
    assert w in s.waiting and sk in s.skipped_waiting
    info = s.get_spec_mode_info()
    assert info["spec_inflight"] == 3
    # S1's running_spec only sees the running queue
    assert info["running_spec"] == 1
    assert s.count_spec_inflight() == 3


def test_preempted_k7_request_still_counts():
    s = sync_scheduler()
    a = _add(s, "a", True)
    complete(s, s.schedule())
    assert a in s.running and not a.spec_off
    s.running.remove(a)
    s._preempt_request(a, 0.0)
    assert a.status == RequestStatus.PREEMPTED and a in s.waiting
    assert a.spec_off is False  # preempt does not reset K
    assert s.get_spec_mode_info()["spec_inflight"] == 1
    assert s.get_spec_mode_info()["running_spec"] == 0


def test_finished_request_leaves_the_count():
    s = sync_scheduler()
    _add(s, "a", True)
    complete(s, s.schedule())
    assert s.count_spec_inflight() == 1
    s.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert s.count_spec_inflight() == 0


def test_inflight_zero_when_only_k0():
    s = sync_scheduler()
    _add(s, "x", False)
    complete(s, s.schedule())
    _add(s, "y", False)
    info = s.get_spec_mode_info()
    assert info["spec_inflight"] == 0 and info["running_nospec"] == 1


def test_state_and_drafter_resident_defaults_and_setter():
    s = sync_scheduler()
    info = s.get_spec_mode_info()
    assert info["drafter_resident"] is True and info["state"] == "DFLASH2"
    _add(s, "a", True)
    s.set_spec_enabled(False)
    assert s.get_spec_mode_info()["state"] == "NOSPEC_DRAINING"  # K=7 still waiting
    s.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert s.get_spec_mode_info()["state"] == "NOSPEC"
    s.set_drafter_resident(False)
    assert s.get_spec_mode_info()["drafter_resident"] is False
    s.spec_state = "NOSPEC_RELEASED"  # engine-core state machine override
    assert s.get_spec_mode_info()["state"] == "NOSPEC_RELEASED"


def test_inflight_is_zero_on_plain_engine():
    s = utils.create_scheduler(max_num_batched_tokens=2048)  # no speculative tokens
    (r,) = utils.create_requests(1, num_tokens=30, req_ids=["p"], max_tokens=8)
    s.add_request(r)
    assert s.count_spec_inflight() == 0
