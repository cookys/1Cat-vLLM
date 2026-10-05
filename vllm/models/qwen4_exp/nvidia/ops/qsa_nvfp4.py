# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA sparse attention over an NVFP4 main K/V cache: gather, decode, FP16 kernel.

The route has two steps per group of query rows.

1. ``gather_dequant_nvfp4_sides_triton`` reads each row's selected tokens from the
   packed cache and decodes them to FP16 without the layer scale. The decode is
   exact: an e2m1 value times an E4M3 block scale has at most five significant
   bits, a magnitude between 2**-10 and 2688.
2. The existing split-K FP16 kernel runs on that tensor, with the same masks,
   online softmax and merge. The gathered ``[rows, topk, heads, head_dim]`` K and V
   are used as a paged cache of ``rows`` pages of ``topk`` tokens, so nothing is
   copied: row ``r`` is request ``r``, its block table is the single page ``r``, and
   entry ``c`` of its selection is token ``c``. An entry the kernel must mask (the
   rule of ``nvfp4_entry_validity``) becomes ``-1``.

The layer scales are folded as for E4M3: ``k_scale`` multiplies the QK scores and
``v_scale`` the normalized output, once, in FP32 (``FOLD_SCALES`` of the kernels).

The gathered tensors are 1 KiB per selected token at head size 256, so the rows
are processed in groups that keep them under ``NVFP4_GATHER_SCRATCH_BYTES``. A
decode step (up to 31 rows at topk 2051) is one group. The fused reader
(``VLLM_SM70_QSA_NVFP4_FUSED_READER``, ``KV_NVFP4`` of the split-K kernel) never
materializes them and is selected in ``qsa_sparse_paged_attention``; see the design
note, step C.

Plan 071 option B' adds a third route for prefill chunks
(``VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH``): the written prefix of each request is
decoded once per layer and chunk into a resident FP16 scratch (``Nvfp4PrefillScratch``,
the layout of an FP16 paged cache), and the grouped page4 CUDA route that serves an
E4M3 or FP16 cache runs on it. ``qsa_sparse_attention_nvfp4_prefill`` is the entry
point; ``qsa_sparse_paged_attention`` calls it before the fused reader and falls
through to the reader whenever it returns ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv import (
    NVFP4_KV_CACHE_DTYPE,
    nvfp4_entry_validity,
    nvfp4_side_views,
)
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton import (
    dequant_nvfp4_prefix_triton,
    gather_dequant_nvfp4_sides_triton,
)

logger = init_logger(__name__)

# K and V together, per group of rows. 64 MiB is 31 rows at topk 2051, head 256.
NVFP4_GATHER_SCRATCH_BYTES = 64 * 1024 * 1024

__all__ = [
    "NVFP4_GATHER_SCRATCH_BYTES",
    "Nvfp4PrefillScratch",
    "Nvfp4ScratchWatermark",
    "ensure_nvfp4_prefill_scratch",
    "get_nvfp4_prefill_scratch",
    "install_nvfp4_prefill_scratch",
    "log_nvfp4_route_knobs",
    "nvfp4_fused_reader_enabled",
    "nvfp4_gather_rows_per_group",
    "nvfp4_prefill_min_rows",
    "nvfp4_prefill_scratch_bytes",
    "nvfp4_prefill_scratch_capacity_tokens",
    "nvfp4_prefill_scratch_enabled",
    "nvfp4_prefill_scratch_pages",
    "nvfp4_prefix_scratch_table",
    "qsa_sparse_attention_nvfp4",
    "qsa_sparse_attention_nvfp4_prefill",
    "reserve_nvfp4_prefill_scratch",
    "reset_nvfp4_prefill_scratch",
]


def nvfp4_fused_reader_enabled(override: bool | None = None) -> bool:
    """Whether an NVFP4 cache is read inside the split-K kernel (no FP16 gather).

    ``override`` (the ``nvfp4_fused_reader`` argument of the ops entry point) wins;
    otherwise ``VLLM_SM70_QSA_NVFP4_FUSED_READER``, which is off by default: the
    gather route is the one that has been measured.
    """
    if override is not None:
        return override
    return envs.VLLM_SM70_QSA_NVFP4_FUSED_READER


def nvfp4_gather_rows_per_group(
    topk: int,
    heads: int,
    head_size: int,
    scratch_bytes: int | None = None,
) -> int:
    """Rows whose gathered FP16 K and V fit in ``scratch_bytes`` (at least one).

    ``None`` means ``NVFP4_GATHER_SCRATCH_BYTES``, read at call time.
    """
    if scratch_bytes is None:
        scratch_bytes = NVFP4_GATHER_SCRATCH_BYTES
    per_row = 2 * topk * heads * head_size * 2
    return max(1, scratch_bytes // per_row)


def qsa_sparse_attention_nvfp4(
    q: torch.Tensor,
    k_side: torch.Tensor,
    v_side: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    output_gate: torch.Tensor | None,
    lse: torch.Tensor | None,
    k_scale: float,
    v_scale: float,
    *,
    scratch_bytes: int | None = None,
) -> torch.Tensor:
    """Called by ``qsa_sparse_paged_attention`` for ``kv_cache_dtype == "nvfp4"``.

    ``k_side``/``v_side`` are the 4-D uint8 halves of the cache; ``output_gate`` is
    the gate already viewed as ``q``; the arguments have been validated.
    """
    from .qsa import QSA_NVFP4_GATHERED_DTYPE, qsa_sparse_paged_attention

    rows, topk = logical_indices.shape
    num_blocks, block_size, heads, _ = k_side.shape
    head_size = q.shape[2]
    # Same checks as the gather, before any scratch is allocated.
    nvfp4_side_views(k_side)
    nvfp4_side_views(v_side)
    group_rows = nvfp4_gather_rows_per_group(topk, heads, head_size, scratch_bytes)
    columns = torch.arange(topk, dtype=torch.int32, device=q.device)
    for start in range(0, rows, group_rows):
        stop = min(rows, start + group_rows)
        count = stop - start
        indices = logical_indices[start:stop]
        requests = token_to_req[start:stop]
        # The layer scales are not applied here: the kernel folds them in FP32.
        keys, values = gather_dequant_nvfp4_sides_triton(
            k_side,
            v_side,
            block_table,
            requests,
            indices,
            k_scale=1.0,
            v_scale=1.0,
            out_dtype=q.dtype,
        )
        valid = nvfp4_entry_validity(
            indices, requests, block_table, num_blocks, block_size
        )
        selection = torch.where(valid, columns, columns.new_full((), -1))
        ids = torch.arange(count, dtype=torch.int32, device=q.device)
        qsa_sparse_paged_attention(
            q[start:stop],
            keys,
            values,
            selection,
            ids.view(count, 1),
            ids,
            out=out[start:stop],
            output_gate=None if output_gate is None else output_gate[start:stop],
            kv_cache_dtype=QSA_NVFP4_GATHERED_DTYPE,
            k_scale=k_scale,
            v_scale=v_scale,
            lse=None if lse is None else lse[start:stop],
        )
    return out


# ------------------------------------------------------------------------------
# Option B': prefill through a decoded FP16 scratch and the grouped page4 route.
# ------------------------------------------------------------------------------


def nvfp4_prefill_scratch_enabled() -> bool:
    """``VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH``, off by default (unmeasured on a GPU)."""
    return bool(envs.VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH)


def nvfp4_prefill_min_rows() -> int:
    """Rows from which a chunk takes the scratch route (fewer keep the reader)."""
    return int(envs.VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS)


def nvfp4_prefill_scratch_capacity_tokens(max_model_len: int) -> int:
    """Token capacity of the scratch: ``VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH_TOKENS``
    when set, else the model's maximum length."""
    override = envs.VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH_TOKENS
    if override:
        return max(1, int(override))
    return max(1, int(max_model_len))


def nvfp4_prefill_scratch_pages(capacity_tokens: int, block_size: int) -> int:
    """Pages of the scratch: one per cache block of the longest supported context."""
    return -(-int(capacity_tokens) // int(block_size))


def nvfp4_prefill_scratch_bytes(
    capacity_tokens: int, block_size: int, num_kv_heads: int, head_size: int
) -> int:
    """K and V in FP16, whole pages: 1 KiB per token at one head of 256."""
    pages = nvfp4_prefill_scratch_pages(capacity_tokens, block_size)
    return pages * block_size * num_kv_heads * head_size * 2 * 2


@dataclass(eq=False)
class Nvfp4PrefillScratch:
    """Two contiguous FP16 tensors ``[pages, block_size, heads, head_size]``.

    The shape and strides are those of an FP16 paged cache with K and V kept
    separate (``stride(0) == block_size * head_size`` per head 1), which is what
    ``_qsa_xqa_page4_shape_supported`` and the grouped page4 kernel accept. One scratch
    serves every layer of a rank: layers run one after the other on one stream and each
    decodes what it reads, so it is a per-device singleton (see
    ``ensure_nvfp4_prefill_scratch``).
    """

    key: torch.Tensor
    value: torch.Tensor
    block_size: int
    heads: int
    head_size: int
    watermark: "Nvfp4ScratchWatermark"

    @property
    def pages(self) -> int:
        return int(self.key.shape[0])

    @property
    def capacity_tokens(self) -> int:
        return self.pages * self.block_size

    @property
    def nbytes(self) -> int:
        return 2 * self.key.numel() * self.key.element_size()


_PREFILL_SCRATCH: dict[tuple[str, int], Nvfp4PrefillScratch] = {}


def _device_key(device: torch.device | str) -> tuple[str, int]:
    device = torch.device(device)
    index = device.index
    if index is None:
        index = torch.cuda.current_device() if device.type == "cuda" else 0
    return device.type, index


def get_nvfp4_prefill_scratch(device: torch.device | str) -> Nvfp4PrefillScratch | None:
    """The reserved scratch of ``device``; never allocates."""
    return _PREFILL_SCRATCH.get(_device_key(device))


def install_nvfp4_prefill_scratch(
    device: torch.device | str, scratch: Nvfp4PrefillScratch
) -> None:
    """Make ``scratch`` the one of ``device`` (a benchmark holding several)."""
    _PREFILL_SCRATCH[_device_key(device)] = scratch


def reset_nvfp4_prefill_scratch() -> None:
    """Drop every scratch and re-arm the one-line route log (tests)."""
    global _ROUTE_KNOBS_LOGGED
    _PREFILL_SCRATCH.clear()
    _ROUTE_KNOBS_LOGGED = False


def ensure_nvfp4_prefill_scratch(
    device: torch.device | str,
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    capacity_tokens: int,
) -> Nvfp4PrefillScratch:
    """Reserve the scratch of ``device`` (idempotent; grows, never shrinks).

    Call it where the KV pool has not been sized yet: the profile run of the worker,
    reached through the early return of ``Qwen4ExpQSAAttention._run_qsa``. The bytes
    then count as torch memory in use when the pool is sized. The route itself never
    allocates, so a scratch created later could not be accounted for.
    """
    key = _device_key(device)
    pages = nvfp4_prefill_scratch_pages(capacity_tokens, block_size)
    existing = _PREFILL_SCRATCH.get(key)
    if (
        existing is not None
        and existing.block_size == block_size
        and existing.heads == num_kv_heads
        and existing.head_size == head_size
        and existing.pages >= pages
    ):
        return existing
    shape = (pages, block_size, num_kv_heads, head_size)
    # Zeros, not empty: the grouped planner may point a padding microblock at page 0
    # of a scratch that no layer has decoded yet, and 0 * garbage must stay finite.
    scratch = Nvfp4PrefillScratch(
        key=torch.zeros(shape, dtype=torch.float16, device=device),
        value=torch.zeros(shape, dtype=torch.float16, device=device),
        block_size=block_size,
        heads=num_kv_heads,
        head_size=head_size,
        watermark=Nvfp4ScratchWatermark(),
    )
    _PREFILL_SCRATCH[key] = scratch
    logger.info(
        "QSA NVFP4 prefill scratch reserved: %d pages x %d tokens (%d tokens, "
        "%.1f MiB per rank) on %s; reserved before the KV pool is sized.",
        pages,
        block_size,
        scratch.capacity_tokens,
        scratch.nbytes / 2**20,
        device,
    )
    return scratch


def reserve_nvfp4_prefill_scratch(
    *,
    device: torch.device | str,
    kv_cache_dtype: str,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    max_model_len: int,
) -> Nvfp4PrefillScratch | None:
    """``ensure_nvfp4_prefill_scratch`` when the knob is on and the cache is NVFP4."""
    if kv_cache_dtype != NVFP4_KV_CACHE_DTYPE or not nvfp4_prefill_scratch_enabled():
        return None
    return ensure_nvfp4_prefill_scratch(
        device,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        capacity_tokens=nvfp4_prefill_scratch_capacity_tokens(max_model_len),
    )


def nvfp4_prefix_scratch_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    scratch_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact block table of the scratch and the first scratch page of each request.

    Request ``r`` owns ``ceil(seq_len_r / block_size)`` consecutive scratch pages
    (clamped to the block table's width) starting at ``offsets[r]``, the exclusive
    cumulative sum of the page counts. ``compact[r, j] = offsets[r] + j`` for the pages
    a request owns, ``-1`` for the rest and for pages that would not fit in
    ``scratch_pages``. Both are int32 and built on the device without a
    synchronization. The grouped planner reads ``compact`` as an ordinary block table
    (``physical_page_stride = block_size / 4``, ``num_cache_blocks = scratch_pages``).
    """
    width = block_table.shape[1]
    pages = torch.clamp(
        (seq_lens.clamp(min=0) + (block_size - 1)) // block_size, max=width
    ).to(torch.int32)
    offsets = (torch.cumsum(pages, 0, dtype=torch.int32) - pages).to(torch.int32)
    columns = torch.arange(width, dtype=torch.int32, device=block_table.device)
    compact = offsets[:, None] + columns[None, :]
    live = (columns[None, :] < pages[:, None]) & (compact < scratch_pages)
    compact = torch.where(live, compact, torch.full_like(compact, -1))
    return compact.contiguous(), offsets.contiguous()


class Nvfp4ScratchWatermark:
    """Host-side record of what the scratch holds, to skip tokens already decoded.

    The scratch is shared by all layers and overwritten by each, so the tokens a layer
    decoded for the previous chunk are still there only if nothing else wrote since. The
    rule, deliberately narrow (correctness first; any doubt is a full re-decode):

    * the last writer is the same ``owner`` (layer), so no other layer or request
      overwrote the pages;
    * the batch is the same list of request ids, in the same order; unknown ids
      (``None``) are never trusted, because a cache hit, a preemption or a reused
      physical page cannot be told from a continuation without an identity;
    * the chunk starts exactly where the recorded decode ended
      (``query_start == recorded seq_len``) for that request;
    * the request's scratch pages did not move: its first page offset is the same as
      then (an earlier request in the batch gaining a page shifts the later ones).

    With one scratch shared by all layers the first condition is never met by a layer
    that is not alone on the scratch, so production decodes the whole prefix each time;
    the cost is the 0.1-0.3 us per token of the design estimate. A per-layer scratch
    (12 x the memory) would make this pay.
    """

    def __init__(self) -> None:
        self._owner: object | None = None
        self._ids: tuple | None = None
        self._seq_lens: list[int] = []

    def record(
        self, owner: object, request_ids: tuple | None, seq_lens: list[int]
    ) -> None:
        """Note that ``owner`` has decoded ``seq_lens`` tokens of each request."""
        self._owner = owner
        self._ids = None if request_ids is None else tuple(request_ids)
        self._seq_lens = list(seq_lens)

    def start_tokens(
        self,
        owner: object,
        request_ids: tuple | None,
        query_starts: list[int],
        seq_lens: list[int],
        block_size: int | None = None,
    ) -> list[int]:
        """First token to decode for each request (``0`` = decode the whole prefix)."""
        count = len(query_starts)
        none = [0] * count
        if (
            request_ids is None
            or self._ids is None
            or self._owner is not owner
            or tuple(request_ids) != self._ids
            or len(self._seq_lens) != count
        ):
            return none
        if count > 1 and block_size is None:
            return none
        starts = []
        old_offset = new_offset = 0
        for index in range(count):
            moved = False
            if block_size is not None:
                moved = old_offset != new_offset
                old_offset += -(-self._seq_lens[index] // block_size)
                new_offset += -(-seq_lens[index] // block_size)
            continuous = query_starts[index] == self._seq_lens[index]
            starts.append(self._seq_lens[index] if continuous and not moved else 0)
        return starts


def _log_prefill_fallback(reason: str) -> None:
    logger.info_once(
        "QSA NVFP4 prefill scratch route declined (%s); using the NVFP4 reader.",
        reason,
    )


_ROUTE_KNOBS_LOGGED = False


def log_nvfp4_route_knobs(fused_reader: bool) -> None:
    """One line per process: which route each kind of batch takes, and the knobs.

    The first call decides; later calls return at once, so the decode hot path does
    not re-read the environment for a log line.
    """
    global _ROUTE_KNOBS_LOGGED
    if _ROUTE_KNOBS_LOGGED:
        return
    _ROUTE_KNOBS_LOGGED = True
    scratch = nvfp4_prefill_scratch_enabled()
    min_rows = nvfp4_prefill_min_rows()
    logger.info_once(
        "QSA NVFP4 KV read routes: decode and small batches use the %s "
        "(VLLM_SM70_QSA_NVFP4_FUSED_READER=%d); chunks of >=%d rows use the %s "
        "(VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH=%d).",
        "fused Triton reader" if fused_reader else "FP16 gather",
        int(fused_reader),
        min_rows,
        "prefill scratch route (decode once to FP16 + grouped page4)"
        if scratch
        else ("fused Triton reader" if fused_reader else "FP16 gather"),
        int(scratch),
    )


def qsa_sparse_attention_nvfp4_prefill(
    q: torch.Tensor,
    k_side: torch.Tensor,
    v_side: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    output_gate: torch.Tensor | None,
    k_scale: float,
    v_scale: float,
    *,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
    max_sequence_length: int | None,
) -> torch.Tensor | None:
    """Decode the prefix once, then run the FP16 grouped page4 route (option B').

    Returns ``out`` on success and ``None`` when the batch is not eligible, in which
    case nothing was written and the caller runs the reader it would have used anyway.

    1. ``nvfp4_prefix_scratch_table`` and ``dequant_nvfp4_prefix_triton`` write K and V
       of every cached token of every request into the scratch exactly once, as the
       exact FP16 values of the gather (no layer scale), zeros past each sequence
       length.
    2. The grouped planner and kernel run on the scratch as an FP16 cache
       (``kv_cache_dtype="auto"``, which ignores ``k_scale``/``v_scale``), on the
       compact table. ``k_scale`` is folded into the softmax scale, once, on the host;
       a remainder of ``rows % 8`` goes through the XQA page4 route with the same
       scale.
    3. ``v_scale`` and the output gate are applied by one elementwise kernel.

    The attention is the CUDA grouped kernel's own arithmetic, so it differs from the
    split-K Triton routes at FP16 rounding level, not bit for bit.
    """
    from . import qsa as ops

    rows = q.shape[0]
    if rows < nvfp4_prefill_min_rows():
        return None
    if query_positions is None or sequence_lengths is None:
        _log_prefill_fallback("no query positions or sequence lengths")
        return None
    scratch = get_nvfp4_prefill_scratch(q.device)
    if scratch is None:
        logger.warning_once(
            "QSA NVFP4 prefill scratch was not reserved during the profile run, "
            "so it is not allocated now (the KV pool is already sized); using the "
            "NVFP4 reader."
        )
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _log_prefill_fallback("the stream is being captured")
        return None
    num_blocks, block_size, heads, _ = k_side.shape
    if (
        block_size != scratch.block_size
        or heads != scratch.heads
        or q.shape[2] != scratch.head_size
    ):
        _log_prefill_fallback("cache geometry differs from the scratch")
        return None
    if max_sequence_length is None:
        _log_prefill_fallback("the longest sequence is unknown")
        return None
    needed_pages = block_table.shape[0] * -(-int(max_sequence_length) // block_size)
    if needed_pages > scratch.pages:
        _log_prefill_fallback(
            f"{needed_pages} pages needed, scratch holds {scratch.pages}"
        )
        return None
    reason = ops._qsa_nvfp4_prefill_extension_reason()
    if reason is not None:
        _log_prefill_fallback(reason)
        return None

    compact, offsets = nvfp4_prefix_scratch_table(
        block_table, sequence_lengths, block_size, scratch.pages
    )
    if not ops._qsa_xqa_page4_shape_supported(
        q,
        scratch.key,
        scratch.value,
        logical_indices,
        compact,
        token_to_req,
        query_positions,
        sequence_lengths,
    ):
        _log_prefill_fallback("shape or dtype outside the page4 contract")
        return None

    dequant_nvfp4_prefix_triton(
        k_side,
        v_side,
        block_table,
        sequence_lengths,
        offsets,
        scratch.key,
        scratch.value,
        max_pages=min(
            block_table.shape[1], -(-int(max_sequence_length) // block_size)
        ),
    )
    result = ops._qsa_sparse_paged_attention_sm70_xqa_page4(
        q,
        scratch.key,
        scratch.value,
        logical_indices,
        compact,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
        "auto",
        1.0,
        1.0,
        softmax_scale=q.shape[2] ** -0.5 * k_scale,
    )
    if result is None:
        _log_prefill_fallback("the page4 route returned no result")
        return None
    ops._qsa_output_scale_gate(out, output_gate, v_scale)
    logger.info_once(
        "QSA NVFP4 prefill scratch route active: prefix decoded once to FP16 "
        "(%d pages x %d tokens, %.1f MiB) + grouped page4 (first chunk: %d rows).",
        scratch.pages,
        scratch.block_size,
        scratch.nbytes / 2**20,
        rows,
    )
    return out
