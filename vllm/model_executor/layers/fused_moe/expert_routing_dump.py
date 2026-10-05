# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic dump of the MoE router's top-k choices, step by step.

Enabled by ``VLLM_SM70_EXPERT_ROUTING_DUMP_DIR``.  For every model step (decode
step, speculative verification step or prefill chunk) it records, per MoE layer
and per token row, the router's top-k expert ids and weights, plus the step's
metadata (anonymous request uid and position of every row, phase).  The MTP
draft layer is recorded for every draft step of the round on its own axis.
Nothing is allocated, no op is added and no hook runs when the env is unset
(``ACTIVE`` stays ``None``).

Where the ids come from
-----------------------
``MoERunner._apply_quant_method`` calls ``router.select_experts`` and hands the
materialised ``topk_weights`` [M, k] float32 / ``topk_ids`` [M, k] int32 to
``quant_method.apply``.  The SM70 NVFP4 and FP8 MoE methods are not monolithic,
so the router never hides inside a fused kernel.  ``BaseRouter.select_experts``
calls ``dump_fn(topk_weights, topk_ids)`` on its final outputs; ``bind_routers``
sets that callback, one closure per layer, on every FusedMoE of the target and
the draft model once they are loaded.

Why the hook lives in the router and not in ``moe_runner.py``: the runner's
``forward`` is inlined by torch.compile, so every byte of ``moe_runner.py`` is
part of the compile-cache key (``code_hash`` of the traced files).  The router
is only called inside the opaque MoE op, hence untraced: with the dump env unset
the cache key of the compiled graph is unchanged, and the dump env vars are on
``ignored_factors`` so that setting them does not move it either.

How it works with CUDA graphs ON
--------------------------------
``moe_forward`` is an opaque custom op whose python body runs during graph
capture (and eagerly in non-graph steps) but not on a replay.  Therefore the
dump is split in three, mirroring ``draft_hidden_dump.py``:

* ``stage`` runs inside ``select_experts`` (via the bound callback): two
  device-to-device ``copy_`` calls (int32 -> int16 ids, float32 -> float16
  weights) into the static buffer of that layer.  The destination slot is fixed
  per bound layer, the extent depends only on the capture-time row count:
  nothing is allocated, nothing synchronizes, nothing reads device data in
  python.  It is recorded into the CUDA graph and replayed with it.  The
  buffers are only written, so no computed value changes.
* ``record_target_step`` / ``record_draft_step`` run at the EAGER call sites
  after the target forward and after every completed draft step (a graph replay
  ran no python).  They enqueue asynchronous device-to-host copies of the valid
  rows into one slot of a pinned host ring, on the same stream as the forward,
  then clear the staged ids, so a layer that does not write in a later step can
  never leak an old step.  No host synchronization.
* ``_hand_over`` runs when a ring is full (and when traffic stops): one event
  synchronization that waits only for the copies already enqueued, then the
  ring goes to a writer thread (validation, npz, sha256, index) while the
  second ring takes the next steps.  If the writer is more than one file behind,
  steps are DROPPED (counted, visible as a gap in ``step_idx``) rather than
  stalling serving.

Row layout
----------
Row ``i`` of a target layer is token ``i`` of the batch: requests are ordered as
``input_batch.req_ids`` and each owns ``num_scheduled_tokens`` consecutive rows
(one for a plain decode, ``1 + drafts`` for a verification step, a chunk for a
prefill).  ``num_tokens`` is the real row count, never the CUDA-graph padded
one; the padded count is in ``padded_num_tokens`` because the MoE kernels
process (and move expert weights for) the padded rows too.  Draft step 0 (the
draft prefill) runs on the same rows; draft steps >= 1 run one row per request,
in batch order.  ``draft_num_rows`` records the valid row count of every draft
step.

A step is recorded whole or not at all: a step with more real rows than
``max_rows``, a step whose ring was not available, and a step in which any valid
row of any target layer lacks ten distinct in-range expert ids are dropped as a
step (``dropped_steps``, its real rows in ``dropped_rows``).  Nothing partial is
ever written, so ``num_tokens[s]`` rows of every target array are always
complete.

Shared expert
-------------
The shared expert of this model runs for every token of every layer; only its
output is scaled by a per-row sigmoid gate (the gate value is not recorded).
``shared_expert_used`` is therefore constant True over valid rows.

Privacy
-------
meta.json carries a fixed whitelist of parameters.  It never contains argv or
the environment.  Request ids are anonymised to per-process integers
(``request_uid``); the uid -> id text map exists only with
``VLLM_SM70_EXPERT_ROUTING_DUMP_IDS=1``.  Directories are 0o700, files 0o600.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
from typing import Any

import numpy as np
import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# The dumper of this process, or None when the dump is off.  The worker call
# sites read this one attribute and nothing else.
ACTIVE: ExpertRoutingDumper | None = None

_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_RING_BYTES = 256 << 20  # host ring budget per ring (two rings per rank)
_IDLE_FLUSH_S = 3.0  # hand a partial ring to the writer after this much silence
_META_EXTRA_KEYS = ("model", "tp", "synthetic", "num_speculative_tokens")

ROUND_DEFINITION = (
    "one step = one target forward; "
    "U(s,l)=|unique(target_topk[s,:num_tokens[s],l,:])|"
)
ROW_LAYOUT = (
    "target rows = batch tokens (requests in batch order, tokens_per_request "
    "rows each); num_tokens = real rows (never the padded count); draft step 0 "
    "uses the same rows, draft steps >= 1 use one row per request; "
    "draft_num_rows[s, d] = valid rows of draft step d; "
    "draft_topk[s, d, row, l, :] = draft step d, draft MoE layer l; padding -1"
)
SHARED_EXPERT_NOTE = (
    "unconditional: computed for every token of every layer, output scaled by "
    "a per-row sigmoid gate (gate value not recorded); shared_expert_used is "
    "constant True over valid rows"
)
DROPPED_ROWS_NOTE = (
    "real token rows of dropped steps (ring overrun, more rows than max_rows, "
    "failed validation) plus draft rows whose ids were not written; per-file "
    "dropped_steps/dropped_rows in index.jsonl count what was dropped since the "
    "previous file was handed over, meta.json holds the run totals"
)

# Public schema, kept in one place so the tests and the fixture can assert it.
NPZ_ARRAYS = (
    "step_idx",
    "step_wall_ns",
    "num_tokens",
    "padded_num_tokens",
    "step_phase",
    "active_requests",
    "request_uid",
    "position",
    "token_is_prefill",
    "target_topk",
    "target_topk_weight",
    "draft_topk",
    "draft_num_rows",
    "shared_expert_used",
)
INDEX_KEYS = (
    "file",
    "sha256",
    "valid_steps",
    "first_step",
    "last_step",
    "dropped_steps",
    "dropped_rows",
)
META_KEYS = (
    "run_id",
    "concurrency",
    "cell_label",
    "synthetic",
    "round_definition",
    "padding_traffic",
    "tp_rank",
    "tp",
    "model",
    "num_experts",
    "top_k",
    "target_layers",
    "draft_layers",
    "num_draft_steps",
    "num_speculative_tokens",
    "layers",
    "steps_per_file",
    "steps_per_file_requested",
    "max_rows",
    "with_weights",
    "dump_ids",
    "dump_dir",
    "kv_pool_blocks",
    "block_size",
    "git_tip",
    "row_layout",
    "shared_expert",
    "dropped_rows_definition",
    "valid_steps",
    "dropped_steps",
    "dropped_rows",
    "files_written",
    "write_errors",
    "created_unix_ns",
    "updated_unix_ns",
)


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # process groups not initialized (unit tests, tools)
        return 0


def _git_tip() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    tip = out.stdout.strip()
    return tip if out.returncode == 0 and tip else None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_makedirs(path: str) -> None:
    """Create ``path`` (and parents) with mode 0o700 on the directories we create."""
    missing = []
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        missing.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:  # another rank created the shared run dir
            continue
        os.chmod(directory, 0o700)  # the umask must not widen it


def _write_private(path: str, writer: Any, mode: str = "wb") -> None:
    """Write ``path`` through a 0o600 file, atomically (tmp + rename)."""
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, mode) as f:
        writer(f)
    os.replace(tmp, path)


def _append_private(path: str, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(text)


class _Run:
    """One output run: a directory with its own counters (see rotate.json)."""

    def __init__(
        self,
        run_dir: str,
        rank: int,
        concurrency: int | str | None,
        cell_label: str | None,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = os.path.basename(os.path.normpath(run_dir))
        self.rank_dir = os.path.join(run_dir, f"rank{rank}")
        self.concurrency = concurrency
        self.cell_label = cell_label
        self.created_ns = time.time_ns()
        self.step_counter = 0
        self.valid_steps = 0
        self.dropped_steps = 0
        self.dropped_rows = 0
        self.files_written = 0
        self.write_errors = 0
        self.pending_steps = 0  # drops not yet attributed to an index line
        self.pending_rows = 0
        self.new_uids: list[tuple[int, str]] = []
        _private_makedirs(self.rank_dir)


class _Ring:
    """Pinned host buffers of one output file (one slot per step).

    Row-major with the layer axis inside, so the valid rows of a step are one
    contiguous prefix of its slot and the readback is a single DMA copy.
    """

    def __init__(
        self,
        *,
        steps: int,
        num_target_layers: int,
        num_draft_layers: int,
        num_draft_steps: int,
        max_rows: int,
        top_k: int,
        with_weights: bool,
        pin: bool,
    ) -> None:
        def host(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, pin_memory=pin)

        self.t_ids = host(steps, max_rows, num_target_layers, top_k, dtype=torch.int16)
        self.t_w = (
            host(steps, max_rows, num_target_layers, top_k, dtype=torch.float16)
            if with_weights
            else None
        )
        self.pos = host(steps, max_rows, dtype=torch.int64)
        self.d_ids = host(
            steps,
            max(num_draft_steps, 1),
            max_rows,
            max(num_draft_layers, 1),
            top_k,
            dtype=torch.int16,
        )
        self.d_rows = np.zeros((steps, max(num_draft_steps, 1)), dtype=np.int32)
        self.recs: list[dict[str, Any]] = []
        self.run: _Run | None = None

    def reset(self) -> None:
        self.recs = []
        self.run = None


def ring_bytes_per_step(
    *,
    num_target_layers: int,
    num_draft_layers: int,
    num_draft_steps: int,
    max_rows: int,
    top_k: int,
    with_weights: bool,
) -> int:
    target = max_rows * num_target_layers * top_k * (4 if with_weights else 2)
    draft = max(num_draft_steps, 1) * max_rows * max(num_draft_layers, 1) * top_k * 2
    return target + draft + max_rows * 8


class ExpertRoutingDumper:
    """Stage, read back and write the router's top-k choices (see module doc)."""

    def __init__(
        self,
        dump_dir: str,
        *,
        rank: int,
        num_target_layers: int,
        num_draft_layers: int,
        num_draft_steps: int,
        top_k: int,
        num_experts: int,
        steps_per_file: int,
        max_rows: int,
        with_weights: bool,
        device: torch.device | str,
        dump_ids: bool = False,
        concurrency: int | str | None = None,
        cell_label: str | None = None,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        self.rank = rank
        self.num_target_layers = num_target_layers
        self.num_draft_layers = num_draft_layers
        self.num_draft_steps = num_draft_steps if num_draft_layers > 0 else 0
        self.top_k = top_k
        self.num_experts = num_experts
        self.steps_per_file_requested = steps_per_file
        self.max_rows = max_rows
        self.with_weights = with_weights
        self.dump_ids = dump_ids
        extra = {k: v for k, v in (meta_extra or {}).items() if k in _META_EXTRA_KEYS}
        self._extra = extra
        device = torch.device(device)
        self._cuda = device.type == "cuda"

        per_step = ring_bytes_per_step(
            num_target_layers=num_target_layers,
            num_draft_layers=num_draft_layers,
            num_draft_steps=self.num_draft_steps,
            max_rows=max_rows,
            top_k=top_k,
            with_weights=with_weights,
        )
        self.steps_per_file = max(1, min(steps_per_file, _RING_BYTES // per_step))
        if self.steps_per_file < steps_per_file:
            logger.warning(
                "SM70 expert routing dump: %d steps x %d rows would need %.0f MiB "
                "of pinned memory per ring, files hold %d steps instead.",
                steps_per_file,
                max_rows,
                steps_per_file * per_step / (1 << 20),
                self.steps_per_file,
            )

        # Static staging buffers.  Allocated here, before any graph capture:
        # the captured copies in ``stage`` write to exactly this storage on
        # every replay.  Row-major, layer axis inside: [rows, layers, k].
        self._t_ids = torch.full(
            (max_rows, num_target_layers, top_k), -1, dtype=torch.int16, device=device
        )
        self._t_w = (
            torch.zeros(
                max_rows, num_target_layers, top_k, dtype=torch.float16, device=device
            )
            if with_weights
            else None
        )
        self._d_ids = torch.full(
            (max_rows, max(num_draft_layers, 1), top_k),
            -1,
            dtype=torch.int16,
            device=device,
        )

        # Two pinned host rings: a full one goes to the writer thread while the
        # other takes the next steps.
        ring_args = dict(
            steps=self.steps_per_file,
            num_target_layers=num_target_layers,
            num_draft_layers=num_draft_layers,
            num_draft_steps=self.num_draft_steps,
            max_rows=max_rows,
            top_k=top_k,
            with_weights=with_weights,
            pin=self._cuda,
        )
        self._free_rings: queue.Queue[_Ring] = queue.Queue()
        self._free_rings.put(_Ring(**ring_args))
        self._ring: _Ring | None = _Ring(**ring_args)
        self._jobs: queue.Queue[_Ring | None] = queue.Queue()

        # Slot of a layer name: ("t"|"d", index), or None for a layer we ignore.
        self._layer_slots: dict[str, tuple[str, int] | None] = {}
        self._layer_names: dict[str, dict[str, str]] = {"target": {}, "draft": {}}
        self._shape_ok = True

        self._lock = threading.RLock()
        self._filled = 0
        self._open_slot: int | None = None
        self._open_tokens = 0
        self._event = torch.cuda.Event() if self._cuda else None
        self._uids: dict[str, int] = {}
        self._rotate_mtime = 0
        self._last_record = time.monotonic()
        self._kv_pool_blocks: int | None = None
        self._block_size: int | None = None
        self._rotate_seq = 0
        self._git_tip = _git_tip()
        self._base_dir = os.path.abspath(dump_dir)
        self._closed = False

        self._run = _Run(self._base_dir, rank, concurrency, cell_label)
        self._stop = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop, name="expert-routing-dump-writer", daemon=True
        )
        self._writer.start()
        self._idle = threading.Thread(
            target=self._idle_loop, name="expert-routing-dump-idle", daemon=True
        )
        self._idle.start()
        atexit.register(self._close_quietly)
        logger.info(
            "SM70 expert routing dump enabled: dir=%s steps=%d max_rows=%d "
            "layers=%d+%d (rank %d; CUDA graph safe, one event sync per file)",
            dump_dir,
            self.steps_per_file,
            max_rows,
            num_target_layers,
            self.num_draft_steps * num_draft_layers,
            rank,
        )

    # ------------------------------------------------------------------ stage
    def slot_for(self, layer_name: str) -> tuple[str, int] | None:
        """Staging slot of a MoE layer: ("t", target index), ("d", draft index)."""
        slot = self._layer_slots.get(layer_name, False)
        if slot is not False:
            return slot  # type: ignore[return-value]
        match = _LAYER_RE.search(layer_name)
        slot = None
        if match is not None:
            idx = int(match.group(1))
            lt = self.num_target_layers
            if layer_name.startswith("mtp.") or ".mtp." in layer_name:
                # The draft layers continue the target numbering (48 for the
                # first MTP layer); accept a draft-local index as well, so a
                # renumbering can never overwrite a target layer.
                didx = idx - lt if idx >= lt else idx
                if didx < self.num_draft_layers:
                    slot = ("d", didx)
            elif idx < lt:
                slot = ("t", idx)
            elif idx < lt + self.num_draft_layers:
                slot = ("d", idx - lt)
            if slot is not None:
                key = "target" if slot[0] == "t" else "draft"
                self._layer_names[key][str(slot[1])] = layer_name
        if slot is None:
            logger.warning_once(
                "SM70 expert routing dump: ignoring MoE layer %s (not one of the "
                "%d target or %d draft layers).",
                layer_name,
                self.num_target_layers,
                self.num_draft_layers,
            )
        self._layer_slots[layer_name] = slot
        return slot

    def bind_routers(self, models: list[Any]) -> int:
        """Set the dump callback on every FusedMoE router of ``models``.

        Call once after the target and draft models are loaded and before any
        CUDA graph is captured.  Returns the number of bound layers.
        """
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

        bound = 0
        for model in models:
            for module in model.modules():
                if not isinstance(module, FusedMoE):
                    continue
                if not isinstance(module.router, BaseRouter):
                    logger.warning(
                        "SM70 expert routing dump: router of %s is a %s, not a "
                        "BaseRouter; layer not dumped.",
                        module.layer_name,
                        type(module.router).__name__,
                    )
                    continue
                slot = self.slot_for(module.layer_name)
                if slot is None:
                    continue

                def dump_fn(topk_weights, topk_ids, _slot=slot):
                    self.stage(_slot, topk_ids, topk_weights)

                module.router.set_dump_fn(dump_fn)
                bound += 1
        expected = self.num_target_layers + self.num_draft_layers
        if bound != expected:
            logger.warning(
                "SM70 expert routing dump: bound %d MoE routers, expected %d "
                "(%d target + %d draft); steps with a missing target layer are "
                "dropped.",
                bound,
                expected,
                self.num_target_layers,
                self.num_draft_layers,
            )
        else:
            logger.info("SM70 expert routing dump: bound %d MoE routers.", bound)
        return bound

    def stage(
        self,
        slot: tuple[str, int],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Copy one layer's router output into its staging buffer (device only).

        Runs inside ``BaseRouter.select_experts``, i.e. inside the captured
        graphs: two ``copy_`` calls into static storage, no allocation, no host
        synchronization, no data-dependent python.
        """
        if not self._shape_ok:
            return
        if topk_ids.dim() != 2 or topk_ids.shape[1] != self.top_k:
            self._shape_ok = False
            logger.warning_once(
                "SM70 expert routing dump disabled: router ids have shape %s, "
                "expected [rows, %d].",
                tuple(topk_ids.shape),
                self.top_k,
            )
            return
        rows = min(topk_ids.shape[0], self.max_rows)
        kind, idx = slot
        if kind == "t":
            self._t_ids[:rows, idx].copy_(topk_ids[:rows])
            if self._t_w is not None:
                self._t_w[:rows, idx].copy_(topk_weights[:rows])
        else:
            self._d_ids[:rows, idx].copy_(topk_ids[:rows])

    def set_kv_info(self, num_blocks: int | None, block_size: int | None) -> None:
        """Engine KV pool facts for meta.json (None when unknown)."""
        self._kv_pool_blocks = None if num_blocks is None else int(num_blocks)
        self._block_size = None if block_size is None else int(block_size)

    # ------------------------------------------------------------- eager hooks
    def _capturing(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

    def _mark(self) -> None:
        if self._event is not None:
            self._event.record()

    def _uid(self, run: _Run, request_id: str) -> int:
        uid = self._uids.get(request_id)
        if uid is None:
            uid = len(self._uids)
            self._uids[request_id] = uid
            if self.dump_ids:
                run.new_uids.append((uid, request_id))
        return uid

    def _drop(self, run: _Run, steps: int, rows: int, why: str) -> None:
        run.dropped_steps += steps
        run.dropped_rows += rows
        run.pending_steps += steps
        run.pending_rows += rows
        logger.warning_once(
            "SM70 expert routing dump: a step was dropped (%s); dropped steps "
            "show as gaps in step_idx and are counted in meta.json / "
            "index.jsonl.",
            why,
        )

    def _try_acquire(self) -> bool:
        if self._ring is not None:
            return True
        try:
            self._ring = self._free_rings.get_nowait()
        except queue.Empty:
            return False
        self._filled = 0
        return True

    def record_target_step(self, input_batch: Any) -> None:
        """After the target forward of a real step: queue its readback.

        Eager call site (``GPUModelRunner.execute_model``): asynchronous
        device-to-host copies only; the host synchronizes once per file, in
        ``_hand_over``.
        """
        if self._closed or not self._shape_ok or self._capturing():
            return
        with self._lock:
            self._last_record = time.monotonic()
            self._maybe_rotate()
            run = self._run
            step_idx = run.step_counter
            run.step_counter += 1
            self._open_slot = None
            n_real = int(input_batch.num_tokens)
            padded = int(getattr(input_batch, "num_tokens_after_padding", 0) or n_real)
            clear_rows = min(max(padded, n_real), self.max_rows)
            if n_real > self.max_rows:
                self._drop(run, 1, n_real, "more rows than max_rows")
                self._t_ids[:clear_rows].fill_(-1)
                return
            if self._ring is not None and self._filled == self.steps_per_file:
                self._hand_over()
            if not self._try_acquire():
                self._drop(run, 1, n_real, "writer is behind, no free ring")
                self._t_ids[:clear_rows].fill_(-1)
                return
            ring = self._ring
            assert ring is not None
            if self._filled == 0:
                ring.run = run
            s = self._filled
            m = n_real
            ring.t_ids[s, :m].copy_(self._t_ids[:m], non_blocking=True)
            if ring.t_w is not None and self._t_w is not None:
                ring.t_w[s, :m].copy_(self._t_w[:m], non_blocking=True)
            ring.pos[s, :m].copy_(input_batch.positions[:m], non_blocking=True)
            self._t_ids[:clear_rows].fill_(-1)
            ring.d_rows[s, :] = 0
            prefilling = np.asarray(input_batch.is_prefilling_np, dtype=bool)
            if prefilling.all():
                phase = 1
            elif prefilling.any():
                phase = 2
            else:
                phase = 0
            req_ids = tuple(input_batch.req_ids)
            ring.recs.append(
                {
                    "step_idx": step_idx,
                    "wall_ns": time.time_ns(),
                    "num_tokens": n_real,
                    "padded": padded,
                    "uids": tuple(self._uid(run, r) for r in req_ids),
                    "tokens_per_req": np.array(
                        input_batch.num_scheduled_tokens, copy=True
                    ),
                    "req_prefilling": prefilling.copy(),
                    "phase": phase,
                    "active": len(req_ids),
                }
            )
            self._filled += 1
            self._open_slot = s
            self._open_tokens = n_real
            self._mark()

    def record_draft_step(self, draft_step: int, num_rows: int) -> None:
        """After one completed draft step of the open round: queue its readback.

        Eager call site (``EagleSpeculator._dump_draft_step``).  ``num_rows`` is
        the request count; draft step 0 runs on every target token row instead.
        """
        if self._closed or self.num_draft_layers == 0 or self._capturing():
            return
        with self._lock:
            s = self._open_slot
            ring = self._ring
            if s is None or ring is None:
                self._d_ids.fill_(-1)
                return
            if not 0 <= draft_step < self.num_draft_steps:
                logger.warning_once(
                    "SM70 expert routing dump: draft step %d outside the %d "
                    "configured draft steps, ignored.",
                    draft_step,
                    self.num_draft_steps,
                )
                self._d_ids.fill_(-1)
                return
            rows = self._open_tokens if draft_step == 0 else int(num_rows)
            m = min(rows, self.max_rows)
            ring.d_ids[s, draft_step, :m].copy_(self._d_ids[:m], non_blocking=True)
            ring.d_rows[s, draft_step] = m
            self._d_ids.fill_(-1)
            self._mark()

    # ------------------------------------------------------- flush and writer
    def _hand_over(self) -> None:
        """Send the filled ring to the writer thread and try to take a free one.

        The only host synchronization of the dump: an event recorded after the
        last queued copy, so it never waits for the forward already enqueued.
        """
        ring = self._ring
        if ring is None or self._filled == 0:
            return
        if self._event is not None:
            self._event.synchronize()
        self._jobs.put(ring)
        self._ring = None
        self._filled = 0
        self._open_slot = None
        self._try_acquire()

    def _idle_loop(self) -> None:
        """Flush a partial ring once traffic stops (also covers SIGTERM tails)."""
        while not self._stop.wait(1.0):
            with self._lock:
                if (
                    self._filled > 0
                    and time.monotonic() - self._last_record > _IDLE_FLUSH_S
                ):
                    try:
                        self._hand_over()
                    except Exception:
                        logger.warning(
                            "SM70 expert routing dump: idle flush failed",
                            exc_info=True,
                        )

    def _maybe_rotate(self) -> None:
        """Start a new run directory when rotate.json asks for it (lock held)."""
        path = os.path.join(self._base_dir, "rotate.json")
        try:  # one stat per step; the file is parsed only when it changed
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            return
        if mtime == self._rotate_mtime:
            return
        self._rotate_mtime = mtime
        try:
            with open(path) as f:
                request = json.load(f)
            seq = int(request["seq"])
            subdir = str(request["subdir"])
        except (OSError, ValueError, KeyError, TypeError):
            self._rotate_mtime = 0  # half-written: look again next step
            return
        if seq <= self._rotate_seq:
            return
        self._rotate_seq = seq
        if not subdir or os.sep in subdir or subdir in (".", ".."):
            logger.warning("SM70 expert routing dump: rotate.json subdir invalid")
            return
        self._hand_over()  # the old run keeps every step it recorded
        concurrency = request.get("concurrency")
        run = _Run(
            os.path.join(os.path.dirname(self._base_dir), subdir),
            self.rank,
            _parse_concurrency(None if concurrency is None else str(concurrency)),
            request.get("cell_label"),
        )
        self._run = run
        logger.info("SM70 expert routing dump: new run %s", run.run_dir)

    def _writer_loop(self) -> None:
        while True:
            ring = self._jobs.get()
            try:
                if ring is None:
                    return
                self._write_file(ring)
            except Exception:
                if ring is not None and ring.run is not None:
                    ring.run.write_errors += 1
                logger.exception("SM70 expert routing dump: writing a file failed")
            finally:
                if ring is not None:
                    ring.reset()
                    self._free_rings.put(ring)
                self._jobs.task_done()

    def _valid_step(self, ids: np.ndarray) -> bool:
        """ids [rows, layers, k]: every id in range and ten distinct per row."""
        if ids.size == 0 or (ids < 0).any() or (ids >= self.num_experts).any():
            return False
        ordered = np.sort(ids, axis=-1)
        return bool((ordered[..., 1:] != ordered[..., :-1]).all())

    def _write_file(self, ring: _Ring) -> None:
        run = ring.run
        assert run is not None
        k = self.top_k
        lt = self.num_target_layers
        ld = self.num_draft_layers
        kd = self.num_draft_steps
        t_ids_host = ring.t_ids.numpy()
        t_w_host = ring.t_w.numpy() if ring.t_w is not None else None
        d_ids_host = ring.d_ids.numpy()
        pos_host = ring.pos.numpy()

        keep: list[int] = []
        bad_steps = 0
        bad_rows = 0
        for s, rec in enumerate(ring.recs):
            if self._valid_step(t_ids_host[s, : rec["num_tokens"]]):
                keep.append(s)
            else:
                bad_steps += 1
                bad_rows += rec["num_tokens"]
        with self._lock:
            run.dropped_steps += bad_steps
            run.dropped_rows += bad_rows
            pending_steps = run.pending_steps + bad_steps
            pending_rows = run.pending_rows + bad_rows
            run.pending_steps = run.pending_rows = 0
        if bad_steps:
            logger.warning_once(
                "SM70 expert routing dump: a step failed validation (a target "
                "layer missing, an id out of range or repeated); dropped."
            )
        count = len(keep)
        if count == 0:
            with self._lock:  # nothing to write: the counts wait for a file
                run.pending_steps += pending_steps
                run.pending_rows += pending_rows
            self._write_meta(run)
            return

        recs = [ring.recs[s] for s in keep]
        m_file = max(r["num_tokens"] for r in recs)
        target = np.full((count, m_file, lt, k), -1, dtype=np.int16)
        target_w = (
            np.full((count, m_file, lt, k), -1.0, dtype=np.float16)
            if t_w_host is not None
            else None
        )
        draft = np.full((count, kd, m_file, ld, k), -1, dtype=np.int16)
        draft_rows = np.zeros((count, kd), dtype=np.int32)
        position = np.full((count, m_file), -1, dtype=np.int32)
        uid = np.full((count, m_file), -1, dtype=np.int32)
        token_prefill = np.zeros((count, m_file), dtype=bool)
        shared = np.zeros((count, m_file, lt), dtype=bool)
        missing_draft_rows = 0
        for i, (s, rec) in enumerate(zip(keep, recs)):
            m = rec["num_tokens"]
            target[i, :m] = t_ids_host[s, :m]
            if target_w is not None and t_w_host is not None:
                target_w[i, :m] = t_w_host[s, :m]
            position[i, :m] = pos_host[s, :m]
            shared[i, :m] = True
            owner = np.repeat(np.arange(len(rec["uids"])), rec["tokens_per_req"])[:m]
            uid[i, :m] = np.asarray(rec["uids"], dtype=np.int32)[owner]
            token_prefill[i, :m] = rec["req_prefilling"][owner]
            for d in range(kd):
                r = int(ring.d_rows[s, d])
                draft_rows[i, d] = r
                if r > 0:
                    block = d_ids_host[s, d, :r, :ld]
                    draft[i, d, :r] = block
                    missing_draft_rows += int((block < 0).any(axis=(1, 2)).sum())
        if missing_draft_rows:
            with self._lock:
                run.dropped_rows += missing_draft_rows
            pending_rows += missing_draft_rows

        step_idx = np.array([r["step_idx"] for r in recs], dtype=np.int32)
        arrays: dict[str, np.ndarray] = {
            "step_idx": step_idx,
            "step_wall_ns": np.array([r["wall_ns"] for r in recs], dtype=np.int64),
            "num_tokens": np.array([r["num_tokens"] for r in recs], dtype=np.int32),
            "padded_num_tokens": np.array([r["padded"] for r in recs], dtype=np.int32),
            "step_phase": np.array([r["phase"] for r in recs], dtype=np.int8),
            "active_requests": np.array([r["active"] for r in recs], dtype=np.int32),
            "request_uid": uid,
            "position": position,
            "token_is_prefill": token_prefill,
            "target_topk": target,
            "draft_topk": draft,
            "draft_num_rows": draft_rows,
            "shared_expert_used": shared,
        }
        if target_w is not None:
            arrays["target_topk_weight"] = target_w

        first, last = int(step_idx.min()), int(step_idx.max())
        name = f"routing_{first}_{last}.npz"
        path = os.path.join(run.rank_dir, name)
        _write_private(path, lambda f: np.savez(f, **arrays))
        entry = {
            "file": name,
            "sha256": _sha256(path),
            "valid_steps": count,
            "first_step": first,
            "last_step": last,
            "dropped_steps": pending_steps,
            "dropped_rows": pending_rows,
        }
        _append_private(
            os.path.join(run.rank_dir, "index.jsonl"), json.dumps(entry) + "\n"
        )
        with self._lock:
            run.valid_steps += count
            run.files_written += 1
            new_uids, run.new_uids = run.new_uids, []
        if new_uids:
            _append_private(
                os.path.join(run.rank_dir, "request_map.jsonl"),
                "".join(
                    json.dumps({"uid": u, "request_id": r}) + "\n" for u, r in new_uids
                ),
            )
        self._write_meta(run)
        logger.info(
            "SM70 expert routing dump: wrote %s (steps %d-%d, %d valid, "
            "%d dropped steps, %d dropped rows)",
            path,
            first,
            last,
            count,
            pending_steps,
            pending_rows,
        )

    def _meta(self, run: _Run) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": run.run_id,
                "concurrency": run.concurrency,
                "cell_label": run.cell_label,
                "synthetic": bool(self._extra.get("synthetic", False)),
                "round_definition": ROUND_DEFINITION,
                "padding_traffic": "counted",
                "tp_rank": self.rank,
                "tp": self._extra.get("tp"),
                "model": self._extra.get("model"),
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "target_layers": self.num_target_layers,
                "draft_layers": self.num_draft_layers,
                "num_draft_steps": self.num_draft_steps,
                "num_speculative_tokens": self._extra.get(
                    "num_speculative_tokens", self.num_draft_steps
                ),
                "layers": {k: dict(v) for k, v in self._layer_names.items()},
                "steps_per_file": self.steps_per_file,
                "steps_per_file_requested": self.steps_per_file_requested,
                "max_rows": self.max_rows,
                "with_weights": self.with_weights,
                "dump_ids": self.dump_ids,
                "dump_dir": run.run_dir,
                "kv_pool_blocks": self._kv_pool_blocks,
                "block_size": self._block_size,
                "git_tip": self._git_tip,
                "row_layout": ROW_LAYOUT,
                "shared_expert": SHARED_EXPERT_NOTE,
                "dropped_rows_definition": DROPPED_ROWS_NOTE,
                "valid_steps": run.valid_steps,
                "dropped_steps": run.dropped_steps,
                "dropped_rows": run.dropped_rows,
                "files_written": run.files_written,
                "write_errors": run.write_errors,
                "created_unix_ns": run.created_ns,
                "updated_unix_ns": time.time_ns(),
            }

    def _write_meta(self, run: _Run) -> None:
        meta = self._meta(run)
        _write_private(
            os.path.join(run.rank_dir, "meta.json"),
            lambda f: json.dump(meta, f, indent=2, sort_keys=True),
            mode="w",
        )

    def _close_quietly(self) -> None:
        try:
            self.close()
        except Exception:  # interpreter shutdown: the dump dir may be gone
            logger.warning(
                "SM70 expert routing dump: final flush failed", exc_info=True
            )

    def close(self) -> None:
        """Write the partial last file, wait for the writer, stop (idempotent)."""
        global ACTIVE
        if self._closed:
            return
        atexit.unregister(self._close_quietly)
        try:
            self._stop.set()
            with self._lock:
                self._hand_over()
                self._closed = True
            self._jobs.put(None)
            self._writer.join(timeout=120)
            self._write_meta(self._run)
        finally:
            self._closed = True
            if ACTIVE is self:
                ACTIVE = None


def _parse_ranks(raw: str) -> set[int] | None:
    raw = (raw or "all").strip().lower()
    if raw in ("", "all", "*"):
        return None
    return {int(part) for part in raw.split(",") if part.strip()}


def _parse_concurrency(raw: str | None) -> int | str | None:
    """The nominal client concurrency of the run, as given (int when numeric)."""
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def resolve_max_rows(configured: int, max_num_seqs: int, num_draft_steps: int) -> int:
    """Row capacity: the env value, or max(128, max_num_seqs * (K + 1))."""
    if configured > 0:
        return configured
    return max(128, int(max_num_seqs) * (int(num_draft_steps) + 1))


def maybe_enable(
    *,
    num_target_layers: int,
    num_draft_layers: int,
    num_draft_steps: int,
    top_k: int,
    num_experts: int,
    device: torch.device | str,
    max_num_seqs: int = 1,
    meta_extra: dict[str, Any] | None = None,
) -> ExpertRoutingDumper | None:
    """Create the process-wide dumper when the env asks for one, else None.

    Must run before the model's CUDA graphs are captured (the staging buffers
    are allocated here and ``ACTIVE`` is what the worker call sites check).
    """
    global ACTIVE
    dump_dir = envs.VLLM_SM70_EXPERT_ROUTING_DUMP_DIR
    steps = envs.VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS
    if not dump_dir or steps <= 0:
        return None
    if top_k <= 0 or num_target_layers <= 0:
        logger.warning(
            "SM70 expert routing dump requested but the model has no MoE "
            "layers (top_k=%d); dump disabled.",
            top_k,
        )
        return None
    rank = _tp_rank()
    ranks = _parse_ranks(envs.VLLM_SM70_EXPERT_ROUTING_DUMP_RANKS)
    if ranks is not None and rank not in ranks:
        return None
    max_rows = resolve_max_rows(
        envs.VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS, max_num_seqs, num_draft_steps
    )
    ACTIVE = ExpertRoutingDumper(
        dump_dir,
        rank=rank,
        num_target_layers=num_target_layers,
        num_draft_layers=num_draft_layers,
        num_draft_steps=num_draft_steps,
        top_k=top_k,
        num_experts=num_experts,
        steps_per_file=steps,
        max_rows=max_rows,
        with_weights=envs.VLLM_SM70_EXPERT_ROUTING_DUMP_WEIGHTS,
        device=device,
        dump_ids=envs.VLLM_SM70_EXPERT_ROUTING_DUMP_IDS,
        concurrency=_parse_concurrency(envs.VLLM_SM70_EXPERT_ROUTING_DUMP_CONCURRENCY),
        cell_label=envs.VLLM_SM70_EXPERT_ROUTING_DUMP_LABEL,
        meta_extra=meta_extra,
    )
    return ACTIVE


def shutdown() -> None:
    """Flush and drop the process-wide dumper (worker shutdown)."""
    if ACTIVE is not None:
        ACTIVE.close()
