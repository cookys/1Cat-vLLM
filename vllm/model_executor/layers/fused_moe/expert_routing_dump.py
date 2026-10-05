# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic dump of the MoE router's top-k choices, step by step.

Enabled by ``VLLM_SM70_EXPERT_ROUTING_DUMP_DIR``.  For every model step (decode
step, speculative verification step or prefill chunk) it records, per MoE layer
and per token row, the router's top-k expert ids and weights, plus the step's
metadata (request id and position of every row, prefill flag).  The MTP draft
layer is recorded for every draft step of the round.  Nothing is allocated, no
op is added and no hook runs when the env is unset (``ACTIVE`` stays ``None``).

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
  ran no python).  They enqueue asynchronous device-to-host copies of the static
  buffers into one slot of a pinned host ring, on the same stream as the
  forward, so the copy always sees that step's data and the next forward cannot
  overwrite it first.  No host synchronization.
* ``_hand_over`` runs every ``steps_per_file`` steps (and at close): one event
  synchronization that waits only for the copies already enqueued, then the full
  pinned ring goes to a writer thread (assembly, npz, sha256, index) while the
  second ring takes the next steps, so the worker's step loop never waits for
  disk or for the numpy reshuffling.

Row layout
----------
Row ``i`` of a target layer is token ``i`` of the batch: requests are ordered as
``input_batch.req_ids`` and each owns ``num_scheduled_tokens`` consecutive rows
(one for a plain decode, ``1 + drafts`` for a verification step, a chunk for a
prefill).  Draft step 0 (the draft prefill) runs on the same rows; draft steps
>= 1 run one row per request, in batch order.  ``draft_num_rows`` records the
valid row count of every draft step.

Shared expert
-------------
The shared expert of this model runs for every token of every layer; only its
output is scaled by a per-row sigmoid gate (the gate value is not recorded).
``shared_expert_used`` is therefore constant True over valid rows.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# The dumper of this process, or None when the dump is off.  The MoE runner and
# the model runner / speculator read this one attribute and nothing else.
ACTIVE: ExpertRoutingDumper | None = None

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


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


def _env_snapshot() -> dict[str, str]:
    prefixes = ("VLLM_", "C4140_", "CUDA_", "NCCL_", "TRITON_")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _Ring:
    """Pinned host buffers of one output file (one slot per step)."""

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

        dsteps = max(num_draft_steps, 1)
        self.t_ids = host(steps, num_target_layers, max_rows, top_k, dtype=torch.int16)
        self.t_w = (
            host(steps, num_target_layers, max_rows, top_k, dtype=torch.float16)
            if with_weights
            else None
        )
        self.pos = host(steps, max_rows, dtype=torch.int64)
        self.d_ids = host(
            steps,
            dsteps,
            max(num_draft_layers, 1),
            max_rows,
            top_k,
            dtype=torch.int16,
        )
        self.d_rows = np.zeros((steps, dsteps), dtype=np.int32)
        self.recs: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.recs = []


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
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        self.rank = rank
        self.rank_dir = os.path.join(dump_dir, f"rank{rank}")
        self.num_target_layers = num_target_layers
        self.num_draft_layers = num_draft_layers
        self.num_draft_steps = num_draft_steps if num_draft_layers > 0 else 0
        self.top_k = top_k
        self.num_experts = num_experts
        self.steps_per_file = steps_per_file
        self.max_rows = max_rows
        self.with_weights = with_weights
        self.meta_extra = dict(meta_extra or {})
        device = torch.device(device)
        self._cuda = device.type == "cuda"

        # Static staging buffers.  Allocated here, before any graph capture:
        # the captured copies in ``stage`` write to exactly this storage on
        # every replay.
        self._t_ids = torch.full(
            (num_target_layers, max_rows, top_k), -1, dtype=torch.int16, device=device
        )
        self._t_w = (
            torch.zeros(
                num_target_layers, max_rows, top_k, dtype=torch.float16, device=device
            )
            if with_weights
            else None
        )
        self._d_ids = torch.full(
            (max(num_draft_layers, 1), max_rows, top_k),
            -1,
            dtype=torch.int16,
            device=device,
        )

        # Two pinned host rings (one slot per step of a file), same dtypes as
        # the staging buffers so every readback is a plain DMA copy.  A full ring
        # goes to the writer thread while the other one takes the next steps.
        ring_args = dict(
            steps=steps_per_file,
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
        self._ring = _Ring(**ring_args)
        self._jobs: queue.Queue[_Ring | None] = queue.Queue()
        self._writer = threading.Thread(
            target=self._writer_loop, name="expert-routing-dump-writer", daemon=True
        )
        self._writer.start()
        self.write_errors = 0

        # Slot of a layer name: ("t"|"d", index), or None for a layer we ignore.
        self._layer_slots: dict[str, tuple[str, int] | None] = {}
        self._layer_names: dict[str, dict[str, str]] = {"target": {}, "draft": {}}
        self._shape_ok = True

        self._filled = 0
        self._open_slot: int | None = None
        self._open_tokens = 0
        self._event = torch.cuda.Event() if self._cuda else None

        self.step_counter = 0
        self.files_written = 0
        self.rows_dropped_total = 0
        self._closed = False
        self._meta_static = {
            "num_experts": num_experts,
            "top_k": top_k,
            "tp_rank": rank,
            "num_target_layers": num_target_layers,
            "num_draft_layers": num_draft_layers,
            "num_draft_steps": self.num_draft_steps,
            "steps_per_file": steps_per_file,
            "max_rows": max_rows,
            "with_weights": with_weights,
            "argv": list(sys.argv),
            "env": _env_snapshot(),
            "git_tip": _git_tip(),
            "created_unix_ns": time.time_ns(),
            "shared_expert": (
                "unconditional: computed for every token of every layer, output "
                "scaled by a per-row sigmoid gate (gate value not recorded); "
                "shared_expert_used is constant True over valid rows"
            ),
            "row_layout": (
                "target rows = batch tokens (requests in batch order, "
                "num_scheduled_tokens rows each); draft step 0 uses the same "
                "rows, draft steps >= 1 use one row per request; "
                "draft_num_rows[s, d] = valid rows of draft step d; "
                "draft_topk[..., d * num_draft_layers + l, :] = draft step d, "
                "draft MoE layer l; padding is -1"
            ),
            **self.meta_extra,
        }
        os.makedirs(self.rank_dir, exist_ok=True)
        atexit.register(self._close_quietly)
        logger.info(
            "SM70 expert routing dump enabled: dir=%s steps=%d max_rows=%d "
            "layers=%d+%d (rank %d; CUDA graph safe, one event sync per file)",
            dump_dir,
            steps_per_file,
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
                "(%d target + %d draft); the missing layers stay -1 in the dump.",
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
            self._t_ids[idx, :rows].copy_(topk_ids[:rows])
            if self._t_w is not None:
                self._t_w[idx, :rows].copy_(topk_weights[:rows])
        else:
            self._d_ids[idx, :rows].copy_(topk_ids[:rows])

    # ------------------------------------------------------------- eager hooks
    def _capturing(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

    def _mark(self) -> None:
        if self._event is not None:
            self._event.record()

    def _note_dropped(self, step_rec: dict[str, Any], dropped: int) -> None:
        if dropped <= 0:
            return
        step_rec["rows_dropped"] += dropped
        self.rows_dropped_total += dropped
        logger.warning_once(
            "SM70 expert routing dump: rows beyond max_rows=%d are dropped "
            "(counted in meta.json / index.jsonl); raise "
            "VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS to keep them.",
            self.max_rows,
        )

    def record_target_step(self, input_batch: Any) -> None:
        """After the target forward of a real step: queue its readback.

        Eager call site (``GPUModelRunner.execute_model``): asynchronous
        device-to-host copies only; the host synchronizes once per file, in
        ``_hand_over``.
        """
        if self._closed or not self._shape_ok or self._capturing():
            return
        if self._filled == self.steps_per_file:
            self._hand_over()
        ring = self._ring
        s = self._filled
        n_real = int(input_batch.num_tokens)
        m = min(n_real, self.max_rows)
        ring.t_ids[s].copy_(self._t_ids, non_blocking=True)
        if ring.t_w is not None and self._t_w is not None:
            ring.t_w[s].copy_(self._t_w, non_blocking=True)
        if m > 0:
            ring.pos[s, :m].copy_(input_batch.positions[:m], non_blocking=True)
        ring.d_rows[s, :] = 0
        prefilling = np.asarray(input_batch.is_prefilling_np, dtype=bool)
        rec = {
            "step_idx": self.step_counter,
            "wall_ns": time.time_ns(),
            "num_tokens": n_real,
            "rows": m,
            "req_ids": tuple(input_batch.req_ids),
            "tokens_per_req": np.array(input_batch.num_scheduled_tokens, copy=True),
            "req_prefilling": prefilling.copy(),
            "is_prefill": bool(prefilling.any()),
            "num_draft_tokens": int(getattr(input_batch, "num_draft_tokens", 0) or 0),
            "rows_dropped": 0,
        }
        self._note_dropped(rec, n_real - m)
        ring.recs.append(rec)
        self._filled += 1
        self.step_counter += 1
        self._open_slot = s
        self._open_tokens = n_real
        self._mark()

    def record_draft_step(self, draft_step: int, num_rows: int) -> None:
        """After one completed draft step of the open round: queue its readback.

        Eager call site (``EagleSpeculator._dump_draft_step``).  ``num_rows`` is
        the request count; draft step 0 runs on every target token row instead.
        """
        if (
            self._closed
            or self._open_slot is None
            or self.num_draft_layers == 0
            or self._capturing()
        ):
            return
        if not 0 <= draft_step < self.num_draft_steps:
            logger.warning_once(
                "SM70 expert routing dump: draft step %d outside the %d "
                "configured draft steps, ignored.",
                draft_step,
                self.num_draft_steps,
            )
            return
        ring = self._ring
        s = self._open_slot
        rows = self._open_tokens if draft_step == 0 else int(num_rows)
        m = min(rows, self.max_rows)
        ring.d_ids[s, draft_step].copy_(self._d_ids, non_blocking=True)
        ring.d_rows[s, draft_step] = m
        self._note_dropped(ring.recs[-1], rows - m)
        self._mark()

    # ------------------------------------------------------- flush and writer
    def _hand_over(self) -> None:
        """Send the filled ring to the writer thread and take a free one.

        The only host synchronization of the dump: an event recorded after the
        last queued copy, so it never waits for the forward already enqueued.
        Blocks only if the writer still holds the other ring, i.e. is more than
        one file behind.
        """
        if self._filled == 0:
            return
        if self._event is not None:
            self._event.synchronize()
        done = self._ring
        self._jobs.put(done)
        self._ring = self._free_rings.get()
        self._filled = 0
        self._open_slot = None

    def _writer_loop(self) -> None:
        while True:
            ring = self._jobs.get()
            try:
                if ring is None:
                    return
                self._write_file(ring)
            except Exception:
                self.write_errors += 1
                logger.exception("SM70 expert routing dump: writing a file failed")
            finally:
                if ring is not None:
                    ring.reset()
                    self._free_rings.put(ring)
                self._jobs.task_done()

    def _write_file(self, ring: _Ring) -> None:
        recs = ring.recs
        count = len(recs)
        if count == 0:
            return
        k = self.top_k
        rows = np.array([r["rows"] for r in recs], dtype=np.int32)
        m_file = max(int(rows.max()), 1)
        lt = self.num_target_layers
        ld = self.num_draft_layers
        dsteps = self.num_draft_steps

        t_ids_host = ring.t_ids.numpy()
        t_w_host = ring.t_w.numpy() if ring.t_w is not None else None
        d_ids_host = ring.d_ids.numpy()
        pos_host = ring.pos.numpy()

        target = np.full((count, m_file, lt, k), -1, dtype=np.int16)
        target_w = (
            np.full((count, m_file, lt, k), -1.0, dtype=np.float16)
            if t_w_host is not None
            else None
        )
        draft = np.full((count, m_file, max(dsteps * ld, 1), k), -1, dtype=np.int16)
        position = np.full((count, m_file), -1, dtype=np.int32)
        shared = np.zeros((count, m_file, lt), dtype=bool)
        row_prefill = np.zeros((count, m_file), dtype=bool)
        request_ids: list[list[str]] = []
        draft_rows = np.zeros((count, max(dsteps, 1)), dtype=np.int32)
        for s, rec in enumerate(recs):
            m = rec["rows"]
            if m > 0:
                target[s, :m] = t_ids_host[s, :, :m].transpose(1, 0, 2)
                if target_w is not None and t_w_host is not None:
                    target_w[s, :m] = t_w_host[s, :, :m].transpose(1, 0, 2)
                position[s, :m] = pos_host[s, :m]
                shared[s, :m] = True
            owner = np.repeat(np.arange(len(rec["req_ids"])), rec["tokens_per_req"])[:m]
            ids = [rec["req_ids"][i] for i in owner]
            request_ids.append(ids + [""] * (m_file - len(ids)))
            if m > 0:
                row_prefill[s, :m] = rec["req_prefilling"][owner]
            for d in range(dsteps):
                r = int(ring.d_rows[s, d])
                draft_rows[s, d] = r
                if r > 0:
                    block = d_ids_host[s, d, :, :r].transpose(1, 0, 2)  # [r, ld, k]
                    draft[s, :r, d * ld : (d + 1) * ld] = block

        step_idx = np.array([r["step_idx"] for r in recs], dtype=np.int32)
        arrays: dict[str, np.ndarray] = {
            "step_idx": step_idx,
            "step_wall_ns": np.array([r["wall_ns"] for r in recs], dtype=np.int64),
            "num_tokens": np.array([r["num_tokens"] for r in recs], dtype=np.int32),
            "request_ids": np.array(request_ids, dtype=np.str_).reshape(count, m_file),
            "position": position,
            "is_prefill": np.array([r["is_prefill"] for r in recs], dtype=bool),
            "row_is_prefill": row_prefill,
            "num_reqs": np.array([len(r["req_ids"]) for r in recs], dtype=np.int32),
            "num_draft_tokens": np.array(
                [r["num_draft_tokens"] for r in recs], dtype=np.int32
            ),
            "rows_dropped": np.array([r["rows_dropped"] for r in recs], dtype=np.int32),
            "target_topk": target,
            "draft_topk": draft,
            "draft_num_rows": draft_rows,
            "shared_expert_used": shared,
        }
        if target_w is not None:
            arrays["target_topk_weight"] = target_w

        first, last = int(step_idx[0]), int(step_idx[-1])
        name = f"routing_{first}_{last}.npz"
        path = os.path.join(self.rank_dir, name)
        np.savez(path + ".tmp.npz", **arrays)
        os.replace(path + ".tmp.npz", path)  # a reader never sees half a file
        total_tokens = int(arrays["num_tokens"].sum())
        dropped = int(arrays["rows_dropped"].sum())
        entry = {
            "file": name,
            "first_step": first,
            "last_step": last,
            "steps": count,
            "total_tokens": total_tokens,
            "rows": m_file,
            "rows_dropped": dropped,
            "bytes": os.path.getsize(path),
            "sha256": _sha256(path),
        }
        with open(os.path.join(self.rank_dir, "index.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.files_written += 1
        self._write_meta()
        logger.info(
            "SM70 expert routing dump: wrote %s (steps %d-%d, %d tokens, "
            "%d rows dropped, %d bytes)",
            path,
            first,
            last,
            total_tokens,
            dropped,
            entry["bytes"],
        )

    def _write_meta(self) -> None:
        meta = {
            **self._meta_static,
            "layers": self._layer_names,
            "files_written": self.files_written,
            "steps_recorded": self.step_counter,
            "rows_dropped_total": self.rows_dropped_total,
            "write_errors": self.write_errors,
            "updated_unix_ns": time.time_ns(),
        }
        path = os.path.join(self.rank_dir, "meta.json")
        with open(path + ".tmp", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(path + ".tmp", path)

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
            self._hand_over()
            self._jobs.put(None)
            self._writer.join(timeout=120)
            if self.files_written == 0:
                self._write_meta()
        finally:
            self._closed = True
            if ACTIVE is self:
                ACTIVE = None


def _parse_ranks(raw: str) -> set[int] | None:
    raw = (raw or "all").strip().lower()
    if raw in ("", "all", "*"):
        return None
    return {int(part) for part in raw.split(",") if part.strip()}


def maybe_enable(
    *,
    num_target_layers: int,
    num_draft_layers: int,
    num_draft_steps: int,
    top_k: int,
    num_experts: int,
    device: torch.device | str,
    meta_extra: dict[str, Any] | None = None,
) -> ExpertRoutingDumper | None:
    """Create the process-wide dumper when the env asks for one, else None.

    Must run before the model's CUDA graphs are captured (the staging buffers
    are allocated here and ``ACTIVE`` is what the MoE runner checks).
    """
    global ACTIVE
    dump_dir = envs.VLLM_SM70_EXPERT_ROUTING_DUMP_DIR
    steps = envs.VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS
    max_rows = envs.VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS
    if not dump_dir or steps <= 0 or max_rows <= 0:
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
        meta_extra=meta_extra,
    )
    return ACTIVE


def shutdown() -> None:
    """Flush and drop the process-wide dumper (worker shutdown)."""
    if ACTIVE is not None:
        ACTIVE.close()
