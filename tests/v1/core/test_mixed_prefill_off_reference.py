# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OFF schedule/state trace against the unmodified P7 + retention-fix base.

The fixture hashes were generated using scheduler.py and async_scheduler.py
from 4a470cc3a (104c33e55 plus f6707f55b), not the implementation under test.
They cover schedule output, KV allocation/refcounts, request states and queue
order, excluding timestamps/statistics and the new zero-valued timing fields.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from vllm.v1.request import RequestStatus

from . import utils
from .test_mixed_prefill_budget import complete, factory  # noqa: F401

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def replay(make_scheduler, async_scheduling, spec, blocks, target=0):
    scheduler = make_scheduler(
        target=target,
        cap=32,
        async_scheduling=async_scheduling,
        num_speculative_tokens=spec or None,
        num_blocks=blocks,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    traces = []
    for index in range(20):
        if index in (0, 1, 4, 8):
            request = utils.create_requests(
                1,
                num_tokens=32 if index == 0 else 160,
                req_ids=[f"r{index}"],
                max_tokens=24,
            )[0]
            scheduler.add_request(request)
        if index == 12:
            scheduler.finish_requests(["r4"], RequestStatus.FINISHED_ABORTED)
        if not async_scheduling and spec:
            for request in scheduler.running:
                if request.num_output_tokens:
                    request.spec_token_ids = list(range(spec))
        step = scheduler.schedule()
        output = dataclasses.asdict(step)
        for name in (
            "mixed_prefill_tokens",
            "mixed_decode_tokens",
            "mixed_prefill_budget",
            "cadence_step",  # New diagnostic field is None by default.
        ):
            output.pop(name, None)
        for new in output["scheduled_new_reqs"]:
            new.pop("sampling_params")  # Identical input object; not scheduler output.
        # Reject all drafts; deterministic one-token feedback.
        complete(scheduler, step)
        pool = scheduler.kv_cache_manager.block_pool
        traces.append(
            {
                "output": output,
                "running": [r.request_id for r in scheduler.running],
                "waiting": [r.request_id for r in scheduler.waiting],
                "requests": {
                    rid: [
                        int(r.status),
                        r.num_computed_tokens,
                        r.num_output_placeholders,
                        list(r.output_token_ids),
                        list(r.spec_token_ids),
                        r.num_preemptions,
                    ]
                    for rid, r in scheduler.requests.items()
                },
                "refcounts": [b.ref_cnt for b in pool.blocks],
                "free": pool.get_num_free_blocks(),
            }
        )
    return json.dumps(
        traces, default=lambda x: sorted(x), sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("spec", [0, 7])
@pytest.mark.parametrize("blocks", [20, 64])
def test_off_matches_base_trace(factory, async_scheduling, spec, blocks):  # noqa: F811
    reference = json.loads(
        Path(__file__).with_name("mixed_prefill_off_reference.json").read_text()
    )
    key = f"async={int(async_scheduling)},spec={spec},blocks={blocks}"
    actual = hashlib.sha256(replay(factory, async_scheduling, spec, blocks).encode())
    assert actual.hexdigest() == reference["sha256"][key]
