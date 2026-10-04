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
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv import (
    nvfp4_entry_validity,
    nvfp4_side_views,
)
from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton import (
    gather_dequant_nvfp4_sides_triton,
)

# K and V together, per group of rows. 64 MiB is 31 rows at topk 2051, head 256.
NVFP4_GATHER_SCRATCH_BYTES = 64 * 1024 * 1024

__all__ = [
    "NVFP4_GATHER_SCRATCH_BYTES",
    "nvfp4_fused_reader_enabled",
    "nvfp4_gather_rows_per_group",
    "qsa_sparse_attention_nvfp4",
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
