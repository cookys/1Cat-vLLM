# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-step stream intervals; never synchronize or bin by wall seconds."""

import json
import time
from collections import deque

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class CadenceStepTimer:
    def __init__(self, rank: int, max_pending: int = 128) -> None:
        self.rank = rank
        self.max_pending = max_pending
        self.pending = deque()
        self.active = None
        self.previous_end = None

    def poll(self) -> None:
        while self.pending and self.pending[0][1].query():
            start, end, previous, meta = self.pending.popleft()
            meta["stream_ms"] = start.elapsed_time(end)
            meta["stream_gap_ms"] = (
                previous.elapsed_time(start) if previous is not None else None
            )
            logger.info("PREFILL_CADENCE_STEP %s", json.dumps(meta, sort_keys=True))

    def begin(self, output) -> None:
        self.poll()
        if self.active is not None:
            logger.warning(
                "PREFILL_CADENCE_TIMING_GAP rank=%d reason=unfinished", self.rank
            )
            self.active = None
        if output.cadence_step is None or output.total_num_scheduled_tokens == 0:
            return
        if len(self.pending) >= self.max_pending:
            logger.warning(
                "PREFILL_CADENCE_TIMING_GAP rank=%d reason=backlog", self.rank
            )
            return
        # Copy only scalar diagnostic metadata. Never serialize request IDs.
        meta = {k: v for k, v in output.cadence_step.items() if k != "decode_req_ids"}
        meta["rank"] = self.rank
        meta["host_begin_ns"] = time.monotonic_ns()
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.active = start, meta

    def route(self, mode: str, padded_rows: int) -> None:
        if self.active is not None:
            self.active[1].update(route=mode, padded_rows=padded_rows)

    def finish(self) -> None:
        if self.active is not None:
            start, meta = self.active
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            meta["host_enqueue_ms"] = (
                time.monotonic_ns() - meta["host_begin_ns"]
            ) / 1e6
            self.pending.append((start, end, self.previous_end, meta))
            self.previous_end = end
            self.active = None
        self.poll()
