# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic dump of the hidden states fed to ``EagleSpeculator._sample_draft``.

Enabled by ``VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR``; the speculator holds no
dumper (and ``_sample_draft`` pays one attribute check) when it is unset.  The
files are the real decode inputs of the draft lm_head, which is what
``benchmarks/sm70_draft_nvfp4_lm_head_check.py --hidden-dump <dir>`` needs to
measure the draft-token disagreement of the NVFP4+rerank head against the FP16
path.

Limits worth knowing:

* ``_sample_draft`` is recorded inside the FULL CUDA graphs of the draft, and a
  replay does not execute python.  Run the dump pass with CUDA graphs disabled
  for the drafter (eager, or ``cudagraph_mode=NONE``).  Under graph capture the
  hook never dumps (a device-to-host copy is illegal there).
* The device-to-host copy synchronizes the host.  This is the only place that
  does; it is a diagnostic, not a production path.
* Warmup and profiling batches are all zeros and are skipped.
"""

from __future__ import annotations

import os

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # process groups not initialized (unit tests, tools)
        return 0


class DraftHiddenDumper:
    """Save the first ``max_batches`` real hidden-state batches as ``.pt`` files.

    Each file is ``{dump_dir}/rank{tp_rank}_step{n:05d}.pt`` holding a dict with
    ``hidden`` (CPU tensor [rows, hidden]), ``n``, ``rank``, ``rows`` and
    ``current_draft_step`` (the value of the speculator's step register when
    the batch was sampled).  The hidden states are replicated across TP ranks,
    so one rank's files are enough for the comparison script.
    """

    def __init__(self, dump_dir: str, max_batches: int) -> None:
        self.dump_dir = dump_dir
        self.max_batches = max_batches
        self.saved = 0
        self.skipped_zero = 0
        os.makedirs(dump_dir, exist_ok=True)
        logger.info(
            "SM70 MTP draft hidden-state dump enabled: dir=%s steps=%d "
            "(eager draft only; host-synchronizing diagnostic)",
            dump_dir,
            max_batches,
        )

    def maybe_dump(
        self, hidden_states: torch.Tensor, current_draft_step: torch.Tensor | int
    ) -> None:
        if self.saved >= self.max_batches:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        hidden = hidden_states.detach().to("cpu").contiguous()  # the only sync
        if not bool(hidden.any()):
            self.skipped_zero += 1
            return
        step = (
            int(current_draft_step.item())
            if isinstance(current_draft_step, torch.Tensor)
            else int(current_draft_step)
        )
        rank = _tp_rank()
        path = os.path.join(self.dump_dir, f"rank{rank}_step{self.saved:05d}.pt")
        torch.save(
            {
                "hidden": hidden,
                "n": self.saved,
                "rank": rank,
                "rows": int(hidden.shape[0]),
                "current_draft_step": step,
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)  # a reader globbing *.pt never sees half a file
        self.saved += 1
        if self.saved == self.max_batches:
            logger.info(
                "SM70 MTP draft hidden-state dump complete: %d batches in %s "
                "(%d all-zero warmup batches skipped)",
                self.saved,
                self.dump_dir,
                self.skipped_zero,
            )


def maybe_create_draft_hidden_dumper() -> DraftHiddenDumper | None:
    dump_dir = envs.VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR
    steps = envs.VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_STEPS
    if not dump_dir or steps <= 0:
        return None
    return DraftHiddenDumper(dump_dir, steps)
