# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offsets of the NVFP4 KV Triton kernels are int64 before they meet a stride.

Background (2026-10-05, q15-w1b2): ``_gather_dequant_nvfp4_kernel`` computed
``row * stride_out_row`` with ``row = tl.program_id(0)``, an int32. At head size 256
and topk 2051 a row is 525,056 elements, so row 4096 and beyond wrap past 2**31 into a
negative offset and the kernel faulted (an illegal memory access at M=5568).

What this file can and cannot show on a machine without a GPU:

* The small-case test (rows 3, topk 4, head size 16) checks, bit for bit, that the
  ``.to(tl.int64)`` casts did not change what the kernel writes or where.
* The static test reads the kernel sources and requires every program-id derived
  index that is multiplied by a ``stride_*`` to be int64 first. It would have caught
  the original line.
* Nothing here reproduces the 2**31 overflow itself: the interpreter would have to
  allocate a >2 GiB output. That check is a GPU job: ``--rows 5 64 5568`` as a single
  case under ``CUDA_LAUNCH_BLOCKING=1`` in the kernel-check script.

Run with ``TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python -m pytest
--noconftest tests/models/qwen4_exp/test_nvfp4_kv_int64_offsets.py``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

KERNEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "models"
    / "qwen4_exp"
    / "nvidia"
    / "ops"
    / "nvfp4_kv_triton.py"
)
KERNELS = (
    "_gather_dequant_nvfp4_kernel",
    "_dequant_nvfp4_prefix_kernel",
    "_overlay_fp16_chunk_kernel",
    "_store_nvfp4_kernel",
)


# ---------------------------------------------------------------- static check
def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _uses_program_id(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "program_id"
        for n in ast.walk(node)
    )


def _has_int64(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "int64" for n in ast.walk(node)
    )


def narrow_stride_products(source: str, kernel: str) -> list[str]:
    """Program-id derived names that meet a ``stride_*`` argument while still int32.

    A name is program-id derived when it is assigned from ``tl.program_id`` or from
    an expression of such names and constants only (no loads). It is wide when its
    assignment casts to int64. Names read back from memory (a block table entry, a
    slot) are checked by the explicit cases in ``test_loaded_indices_are_int64``.
    """
    tree = ast.parse(source)
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == kernel
    )
    assigns: dict[str, ast.AST] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigns[target.id] = node.value
    derived: set[str] = {n for n, v in assigns.items() if _uses_program_id(v)}
    changed = True
    while changed:
        changed = False
        for name, value in assigns.items():
            if name in derived or any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"load", "arange"}
                for n in ast.walk(value)
            ):
                continue
            refs = {r for r in _names(value) if r not in {"tl"} and not r.isupper()}
            if refs and refs <= derived:
                derived.add(name)
                changed = True
    wide = {n for n in derived if _has_int64(assigns[n])}
    bad: list[str] = []
    for node in ast.walk(func):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            continue
        for stride_side, other in ((node.left, node.right), (node.right, node.left)):
            if not any(n.startswith("stride_") for n in _names(stride_side)):
                continue
            if _has_int64(other):
                continue
            bad += [
                f"{kernel}: {n} * stride (line {node.lineno})"
                for n in sorted(_names(other) & (derived - wide))
            ]
    return bad


@pytest.mark.parametrize("kernel", KERNELS)
def test_program_id_indices_are_int64_before_they_meet_a_stride(kernel) -> None:
    assert narrow_stride_products(KERNEL_FILE.read_text(), kernel) == []


def test_the_static_check_flags_the_original_gather_line() -> None:
    source = KERNEL_FILE.read_text()
    narrowed = source.replace(
        "row = tl.program_id(0).to(tl.int64)", "row = tl.program_id(0)"
    )
    assert narrowed != source
    bad = narrow_stride_products(narrowed, "_gather_dequant_nvfp4_kernel")
    assert any("row * stride" in line for line in bad), bad


@pytest.mark.parametrize(
    "kernel, required",
    [
        ("_gather_dequant_nvfp4_kernel", "block = tl.maximum(physical, 0).to(tl.int64)"),
        ("_dequant_nvfp4_prefix_kernel", "block = tl.maximum(physical, 0).to(tl.int64)"),
        (
            "_dequant_nvfp4_prefix_kernel",
            "scratch_page.to(tl.int64) * stride_out_page",
        ),
        ("_overlay_fp16_chunk_kernel", "scratch_page.to(tl.int64) * stride_out_page"),
        ("_store_nvfp4_kernel", "block = (safe_slot // PAGE_SIZE).to(tl.int64)"),
    ],
)
def test_loaded_indices_are_int64(kernel, required) -> None:
    tree = ast.parse(KERNEL_FILE.read_text())
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == kernel
    )
    text = ast.get_source_segment(KERNEL_FILE.read_text(), func)
    assert text is not None
    assert required in text


# --------------------------------------------------------- bit-level small case
pytestmark_run = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs TRITON_INTERPRET=1 (CPU interpreter) or a CUDA GPU",
)

BLOCK = 16
HEAD_SIZE = 16


@pytestmark_run
def test_the_gather_writes_every_row_col_head_where_the_reference_puts_it() -> None:
    from vllm.v1.attention.backends import fa_utils

    if not torch.cuda.is_available():
        fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv
    from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv_triton import (
        gather_dequant_nvfp4_kv_triton,
    )

    device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    rows, topk, heads, num_blocks, tokens = 3, 4, 2, 3, 40
    generator = torch.Generator().manual_seed(7)
    cache = torch.zeros(
        (num_blocks, 2, BLOCK, heads, nv.nvfp4_kv_row_bytes(HEAD_SIZE)),
        dtype=torch.uint8,
    )
    slots = torch.randperm(num_blocks * BLOCK, generator=generator)[:tokens]
    key = torch.randn((tokens, heads, HEAD_SIZE), generator=generator).half()
    value = torch.randn((tokens, heads, HEAD_SIZE), generator=generator).half()
    nv.reshape_and_cache_nvfp4_reference(key, value, cache, slots)
    table = torch.arange(num_blocks, dtype=torch.int32).view(1, -1)
    requests = torch.zeros(rows, dtype=torch.int32)
    indices = torch.randint(
        0, num_blocks * BLOCK, (rows, topk), generator=generator
    ).int()
    indices[1, 2] = -1  # one illegal entry: it must stay +0.0 in its own slot

    got_k, got_v = (
        x.cpu()
        for x in gather_dequant_nvfp4_kv_triton(
            cache.to(device),
            table.to(device),
            requests.to(device),
            indices.to(device),
        )
    )

    # Reference one: the whole batch through the torch reference.
    want_k, want_v = nv.gather_dequant_nvfp4_kv(cache, table, requests, indices)
    assert torch.equal(got_k.view(torch.int16), want_k.view(torch.int16))
    assert torch.equal(got_v.view(torch.int16), want_v.view(torch.int16))

    # Reference two, placed by numpy: every (row, col) decoded alone, then written at
    # its own position. A wrong offset moves an entry, which the batch compare alone
    # could in principle share with the reference only by sharing the same mistake.
    expect_k = np.zeros((rows, topk, heads, HEAD_SIZE), dtype=np.float16)
    expect_v = np.zeros_like(expect_k)
    for r in range(rows):
        for c in range(topk):
            one_k, one_v = nv.gather_dequant_nvfp4_kv(
                cache, table, requests[r : r + 1], indices[r : r + 1, c : c + 1]
            )
            expect_k[r, c] = one_k[0, 0].numpy()
            expect_v[r, c] = one_v[0, 0].numpy()
    assert np.array_equal(got_k.numpy().view(np.int16), expect_k.view(np.int16))
    assert np.array_equal(got_v.numpy().view(np.int16), expect_v.view(np.int16))
    assert not got_k[1, 2].any() and not got_v[1, 2].any()
