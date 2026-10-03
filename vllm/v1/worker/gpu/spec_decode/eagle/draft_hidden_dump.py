# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic dump of the hidden states the draft lm_head really consumes.

Enabled by ``VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR``; the speculator holds no
dumper (and pays one attribute check) when it is unset.  Each file is one draft
step of one ``propose`` round: the lm_head input ``hidden`` [rows, K] plus the
global token ids ``draft_tokens`` [rows] the step produced.  They are what
``benchmarks/sm70_draft_nvfp4_lm_head_check.py --hidden-dump <dir>`` needs to
measure the disagreement of the NVFP4+rerank head against the FP16 path and to
validate the dump itself (``dump_consistency``).  Take the dump with
``VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD`` unset, so ``draft_tokens`` are the FP16
head's answers.

How it works with CUDA graphs ON
--------------------------------
``EagleSpeculator._sample_draft`` is recorded inside the FULL CUDA graphs of the
draft prefill and decode steps, and a replay does not run python.  So the dump
is split in two:

* ``stage`` runs inside ``_sample_draft`` (captured with the graph, replayed
  with it): one device-to-device copy of the exact lm_head input into the
  persistent ``_stage`` buffer.  No host work, nothing to synchronize, legal
  under capture.
* ``dump_step`` runs at the EAGER call sites that follow each completed draft
  step (after the prefill step in ``propose``, after every decode step in
  ``multi_step_decode``): it copies the stage buffer and the step's draft
  tokens to the host and writes the file.  The device-to-host copy
  synchronizes the host (once per dumped step).  This is the only place that
  does; it is a diagnostic, not a production path.

Why a staging buffer and not ``EagleSpeculator.hidden_states``: that buffer is
the draft's *feedback* hidden state.  For the Qwen3.8 MTP it is the 4-stream
``[rows, hc_count * H]`` tensor (``multi_hidden``), while the lm_head reads the
single-stream ``[rows, H]`` ``sample_hidden_states`` (the first element of the
model's return tuple), and the final draft step does not update it at all.

Warmup and profiling batches (and any step whose staged rows are all zero) are
skipped.
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
    """Save the first ``max_batches`` real draft-step batches as ``.pt`` files.

    Each file is ``{dump_dir}/rank{tp_rank}_step{n:05d}.pt`` (``n`` counts the
    saved files) holding a dict with ``hidden`` (CPU tensor [rows, K], the
    lm_head input), ``draft_tokens`` (CPU int64 [rows], the global ids that
    step produced), ``step`` (the draft step, 0 = the draft prefill), ``n``,
    ``rank`` and ``rows``.  The hidden states and the tokens are replicated
    across TP ranks, so one rank's files are enough for the comparison script.
    """

    def __init__(
        self,
        dump_dir: str,
        max_batches: int,
        *,
        max_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.dump_dir = dump_dir
        self.max_batches = max_batches
        self.saved = 0
        self.skipped_zero = 0
        # Allocated here, before any graph capture: the captured copy in
        # ``stage`` writes to this exact storage on every replay.
        self._stage = torch.zeros(max_rows, hidden_size, dtype=dtype, device=device)
        self._stage_usable = True
        os.makedirs(dump_dir, exist_ok=True)
        logger.info(
            "SM70 MTP draft hidden-state dump enabled: dir=%s files=%d rows<=%d "
            "K=%d (works with CUDA graphs; one host sync per dumped draft step)",
            dump_dir,
            max_batches,
            max_rows,
            hidden_size,
        )

    def stage(self, hidden_states: torch.Tensor) -> None:
        """Copy the lm_head input into the staging buffer (device only).

        Runs inside ``_sample_draft``, i.e. inside the captured graphs: no host
        synchronization, no allocation, no data-dependent python.
        """
        rows, width = hidden_states.shape
        if rows > self._stage.shape[0] or width != self._stage.shape[1]:
            self._stage_usable = False
            logger.warning_once(
                "SM70 MTP draft hidden-state dump disabled: lm_head input shape "
                "does not fit the staging buffer (the buffer is sized for "
                "max_num_reqs rows of the draft hidden size)."
            )
            return
        self._stage[:rows].copy_(hidden_states)

    def dump_step(self, step: int, num_reqs: int, draft_tokens: torch.Tensor) -> None:
        """Write the batch of draft step ``step``; eager call sites only.

        ``draft_tokens`` is the speculator's ``[max_num_reqs, steps]`` buffer;
        column ``step`` holds the tokens this step just produced.  Host-
        synchronizing (device-to-host copies).
        """
        if self.saved >= self.max_batches or not self._stage_usable:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        rows = min(num_reqs, self._stage.shape[0])
        if rows <= 0:
            return
        # copy=True: on a CPU buffer (unit tests) ``to`` would alias the stage,
        # which is zeroed below.  On CUDA this is the one host synchronization.
        hidden = self._stage[:rows].detach().to("cpu", copy=True).contiguous()
        # Consumed: a later step that stages nothing must not re-dump this one.
        self._stage[:rows].zero_()
        if not bool(hidden.any()):
            self.skipped_zero += 1
            return
        # A compact copy: torch.save would otherwise serialize the whole strided
        # storage of the speculator's [max_num_reqs, steps] buffer.
        tokens = (
            draft_tokens[:rows, step]
            .detach()
            .to("cpu", copy=True)
            .to(torch.int64)
            .contiguous()
        )
        rank = _tp_rank()
        path = os.path.join(self.dump_dir, f"rank{rank}_step{self.saved:05d}.pt")
        torch.save(
            {
                "hidden": hidden,
                "draft_tokens": tokens,
                "step": int(step),
                "n": self.saved,
                "rank": rank,
                "rows": int(hidden.shape[0]),
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)  # a reader globbing *.pt never sees half a file
        self.saved += 1
        if self.saved == self.max_batches:
            logger.info(
                "SM70 MTP draft hidden-state dump complete: %d files in %s "
                "(%d all-zero warmup/unstaged batches skipped)",
                self.saved,
                self.dump_dir,
                self.skipped_zero,
            )


def maybe_create_draft_hidden_dumper(
    *,
    max_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> DraftHiddenDumper | None:
    """The dumper when the dump env is set, else None (nothing is allocated)."""
    dump_dir = envs.VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR
    steps = envs.VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_STEPS
    if not dump_dir or steps <= 0:
        return None
    return DraftHiddenDumper(
        dump_dir,
        steps,
        max_rows=max_rows,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
