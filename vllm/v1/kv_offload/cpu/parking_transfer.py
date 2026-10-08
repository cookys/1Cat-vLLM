# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult

logger = init_logger(__name__)


class RecoveringHandler(OffloadingHandler):
    """Drain partial copies before reporting recoverable failure.

    Synchronization happens only on an error. A healthy context permits cold
    recomputation. If even device synchronization fails, memory ownership is
    unknowable: raise and let engine teardown reclaim all rank-local storage.
    """

    def __init__(self, inner, pool=None):
        self.inner = inner
        self.pool = pool
        self.pending = set()
        self.failed = []
        self.failure_count = 0
        self.fatal_count = 0

    def _recover(self, error):
        self.failure_count += 1
        try:
            torch.cuda.synchronize()
        except Exception as fatal:
            self.fatal_count += 1
            logger.error(
                "HOST_PARKING fatal_copy_context total=%d; cannot reclaim "
                "rank buffers safely: %s",
                self.fatal_count,
                fatal,
            )
            raise RuntimeError("HOST_PARKING unrecoverable rank copy state") from fatal
        logger.warning(
            "HOST_PARKING copy_failure total=%d drained_jobs=%d; invalidate "
            "snapshot and recompute: %s",
            self.failure_count,
            len(self.pending),
            error,
        )
        # All streams are quiescent. Do not release the fixed pinned pool;
        # only remove native in-flight bookkeeping. Returned jobs are failed.
        self.inner._transfers.clear()
        self.inner._transfer_events.clear()
        self.failed.extend(
            TransferResult(job_id=j, success=False) for j in sorted(self.pending)
        )
        self.pending.clear()

    def transfer_async(self, job_id, spec):
        if job_id in self.pending:
            raise ValueError("duplicate parking transfer job")
        self.pending.add(job_id)
        try:
            if not self.inner.transfer_async(job_id, spec):
                raise RuntimeError("native transfer submission refused")
        except Exception as error:
            self._recover(error)
        # A recoverable refusal is a submitted *failed* job, reported on poll.
        return True

    def get_finished(self):
        try:
            results = self.inner.get_finished()
        except Exception as error:
            self._recover(error)
            results = []
        for result in results:
            self.pending.discard(result.job_id)
        results.extend(self.failed)
        self.failed = []
        return results

    def wait(self, job_ids):
        try:
            self.inner.wait(job_ids)
        except Exception as error:
            self._recover(error)

    def shutdown(self):
        # Native shutdown waits before dropping tensor owners. A broken
        # context is intentionally not disguised as successful reclamation.
        self.inner.shutdown()
        self.pending.clear()
        self.failed.clear()
        if self.pool is not None:
            self.pool.release_direction()
            self.pool = None


class ValidationFaultHandler:
    """Opt-in test wrapper *inside* RecoveringHandler, rank zero only.

    completed_copy changes a result only after the native event has completed.
    It simulates a reported copy failure without corrupting a CUDA context.
    All native queues/events remain authoritative and are delegated unchanged.
    """

    def __init__(self, inner, mode):
        if mode not in ("pre_submit", "completed_copy"):
            raise ValueError("unknown parking validation fault mode")
        self.inner = inner
        self.mode = mode
        self.armed = True

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def transfer_async(self, job_id, spec):
        if self.armed and self.mode == "pre_submit":
            self.armed = False
            logger.warning(
                "HOST_PARKING injected_failure mode=pre_submit job=%d", job_id
            )
            return False
        return self.inner.transfer_async(job_id, spec)

    def get_finished(self):
        results = self.inner.get_finished()
        if self.armed and self.mode == "completed_copy" and results:
            self.armed = False
            first = results[0]
            results[0] = TransferResult(job_id=first.job_id, success=False)
            logger.warning(
                "HOST_PARKING injected_failure mode=completed_copy job=%d",
                first.job_id,
            )
        return results
