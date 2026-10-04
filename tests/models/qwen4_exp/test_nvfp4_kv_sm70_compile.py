# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only gate: the NVFP4 Triton kernels lower to SM70 cubins.

No GPU is needed or used. Triton's bundled ptxas builds each kernel for a Volta
target and the checks read the PTX and the resource usage of the cubin. This
shows the kernels use nothing SM70 lacks (no bf16 or fp8 conversion, no mma
shape of newer architectures, no async copy) and stay small; it does not show
that they run or how fast. Run it without ``TRITON_INTERPRET`` (the interpreter
replaces the compiler).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import regex as re

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") == "1",
    reason="compile-only checks need the real Triton compiler",
)

pytest.importorskip("triton")
from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv_triton as kernels  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

# The compiler entry points are not re-exported by vllm.triton_utils.
GPUTarget = importlib.import_module("triton.backends.compiler").GPUTarget
ASTSource = importlib.import_module("triton.compiler").ASTSource

VOLTA = GPUTarget("cuda", 70, 32)
CUOBJDUMP = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
# Instruction families that do not exist on SM70 (or that would mean the
# software decode silently became a hardware one).
BANNED = (
    "cvt.f32.bf16",
    "cvt.rn.bf16",
    "e4m3",
    "e2m1",
    "wgmma",
    "cp.async",
    "mma.sync.aligned.m16n8",
)

# Measured with Triton 3.6.0's ptxas: the split-K kernel on the gathered FP16 rows
# has no stack frame at 2 or 4 warps (the E4M3 kernel has 304 bytes at 2 warps).
GATHERED_MAX_STACK = 0
# The merge kernel keeps a [splits, head_dim] tile in registers; its 72 byte stack is
# the same with and without scale folding.
MERGE_MAX_STACK = 128

STORE = {
    "x_ptr": "*fp16",
    "slots_ptr": "*i64",
    "data_ptr": "*u8",
    "scale_ptr": "*u8",
    "layer_scale_ptr": "*fp32",
    **{
        name: "i32"
        for name in (
            "stride_x_token",
            "stride_x_head",
            "stride_data_block",
            "stride_data_token",
            "stride_data_head",
            "stride_scale_block",
            "stride_scale_token",
            "stride_scale_head",
            "capacity",
        )
    },
    "PAGE_SIZE": "constexpr",
    "GROUPS": "constexpr",
}
GATHER = {
    "k_data_ptr": "*u8",
    "k_scale_ptr": "*u8",
    "v_data_ptr": "*u8",
    "v_scale_ptr": "*u8",
    "block_table_ptr": "*i32",
    "token_to_req_ptr": "*i32",
    "indices_ptr": "*i32",
    "k_out_ptr": "*fp16",
    "v_out_ptr": "*fp16",
    "k_layer_scale": "fp32",
    "v_layer_scale": "fp32",
    **{
        name: "i32"
        for name in (
            "stride_data_block",
            "stride_data_token",
            "stride_data_head",
            "stride_scale_block",
            "stride_scale_token",
            "stride_scale_head",
            "stride_table_req",
            "stride_indices_row",
            "stride_out_row",
            "stride_out_col",
            "stride_out_head",
            "num_blocks",
            "num_requests",
        )
    },
    "PAGE_SIZE": "constexpr",
    "TABLE_WIDTH": "constexpr",
    "HALF": "constexpr",
}


def _compile(kernel, signature, constexprs, num_warps):
    return triton.compile(
        ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
        target=VOLTA,
        options={"num_warps": num_warps},
    )


def _usage(compiled) -> tuple[int, int, int] | None:
    """(registers, stack bytes, local bytes) of the cubin, or None without cuobjdump."""
    if not CUOBJDUMP.is_file():
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin:
        cubin.write(compiled.asm["cubin"])
        cubin.flush()
        report = subprocess.run(
            [str(CUOBJDUMP), "--dump-resource-usage", cubin.name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:\d+ LOCAL:(\d+)", report)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _check(compiled, max_registers=64, max_stack=16) -> None:
    ptx = compiled.asm["ptx"]
    assert re.search(r"\.target sm_70\b", ptx)
    assert len(compiled.asm["cubin"]) > 0
    for banned in BANNED:
        assert banned not in ptx, banned
    usage = _usage(compiled)
    if usage is not None:
        registers, stack, local = usage
        assert registers <= max_registers, registers
        assert local == 0, "the kernel uses local memory"
        assert stack <= max_stack, f"stack frame of {stack} bytes"


@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("input_type", ["*fp16", "*fp32", "*bf16"])
def test_the_store_lowers_to_sm70(head_size, input_type) -> None:
    signature = {**STORE, "x_ptr": input_type}
    constexprs = {"PAGE_SIZE": 2784, "GROUPS": head_size // 16}
    _check(_compile(kernels._store_nvfp4_kernel, signature, constexprs, 1))


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_the_store_lowers_for_every_launch_width(num_warps) -> None:
    constexprs = {"PAGE_SIZE": 1616, "GROUPS": 16}
    _check(_compile(kernels._store_nvfp4_kernel, STORE, constexprs, num_warps))


@pytest.mark.parametrize("output_type", ["*fp16", "*bf16"])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_the_gather_lowers_to_sm70(output_type, num_warps) -> None:
    signature = {**GATHER, "k_out_ptr": output_type, "v_out_ptr": output_type}
    constexprs = {"PAGE_SIZE": 2784, "TABLE_WIDTH": 96, "HALF": 128}
    _check(
        _compile(kernels._gather_dequant_nvfp4_kernel, signature, constexprs, num_warps)
    )


def test_the_page_size_is_a_compile_time_constant_of_any_value() -> None:
    # Block sizes 1568, 1616, 2784, 2864 and the packed 1392 are not powers of two.
    for page in (1392, 1568, 1616, 2784, 2864):
        constexprs = {"PAGE_SIZE": page, "TABLE_WIDTH": 64, "HALF": 128}
        _check(_compile(kernels._gather_dequant_nvfp4_kernel, GATHER, constexprs, 2))


def test_the_banned_list_would_notice_an_fp8_conversion() -> None:
    # A kernel that asks Triton for a native fp8 conversion must trip the gate
    # (or fail to compile on SM70), so a clean result above means something.
    @triton.jit
    def fp8_kernel(x_ptr, y_ptr):
        index = triton.language.arange(0, 32)
        x = triton.language.load(x_ptr + index)
        triton.language.store(y_ptr + index, x.to(triton.language.float8e4nv))

    signature = {"x_ptr": "*fp32", "y_ptr": "*fp8e4nv"}
    try:
        compiled = _compile(fp8_kernel, signature, {}, 1)
    except Exception:
        return  # it does not even lower for Volta: the gate is not needed
    assert any(banned in compiled.asm["ptx"] for banned in BANNED)


def test_the_store_uses_no_atomic_so_it_is_deterministic() -> None:
    compiled = _compile(
        kernels._store_nvfp4_kernel, STORE, {"PAGE_SIZE": 2784, "GROUPS": 16}, 1
    )
    assert not re.search(r"^\s*(atom|red)\.", compiled.asm["ptx"], re.MULTILINE)


# --------------------------------------------------------------------------- #
# The attention kernels of the NVFP4 route (plan 071 step D)
# --------------------------------------------------------------------------- #
_SPLITK_RUNTIME = {
    "q_ptr": "*fp16",
    "k_cache_ptr": "*fp16",
    "v_cache_ptr": "*fp16",
    "indices_ptr": "*i32",
    "block_table_ptr": "*i32",
    "token_to_req_ptr": "*i32",
    "partial_output_ptr": "*fp32",
    "partial_lse_ptr": "*fp32",
    "output_ptr": "*fp16",
    "final_lse_ptr": "*fp32",
    "output_gate_ptr": "*fp16",
    "k_scale": "fp32",
    "v_scale": "fp32",
    **{
        name: "i32"
        for name in (
            "stride_q_row",
            "stride_q_head",
            "stride_k_block",
            "stride_k_token",
            "stride_k_head",
            "stride_v_block",
            "stride_v_token",
            "stride_v_head",
            "stride_indices_row",
            "stride_table_req",
            "stride_output_row",
            "stride_output_head",
            "stride_output_gate_row",
            "stride_output_gate_head",
            "num_rows",
            "num_cache_blocks",
            "num_requests",
        )
    },
}
_MERGE_RUNTIME = {
    "partial_output_ptr": "*fp32",
    "partial_lse_ptr": "*fp32",
    "output_ptr": "*fp16",
    "final_lse_ptr": "*fp32",
    "output_gate_ptr": "*fp16",
    "v_scale": "fp32",
    **{
        name: "i32"
        for name in (
            "stride_output_row",
            "stride_output_head",
            "stride_output_gate_row",
            "stride_output_gate_head",
            "num_rows",
        )
    },
}


def _with_constexprs(kernel, runtime):
    # Every parameter that is not a runtime value is a constexpr, so an added
    # constexpr shows up as a compile error here instead of a silent skip.
    return {name: runtime.get(name, "constexpr") for name in kernel.arg_names}


def _gathered_splitk(num_warps, *, topk=2051, block_n=16, splits=64):
    """The split-K kernel as the NVFP4 route launches it: FP16 caches, the gathered
    rows as ``rows`` pages of ``topk`` tokens, one page per request, scales folded."""
    kernel = qsa_ops._qsa_sparse_paged_gqa_splitk_kernel
    constexprs = {
        "TOPK": topk,
        "PAGE_SIZE": topk,
        "PAGE_TABLE_WIDTH": 1,
        "GROUP_SIZE": 6,
        "HEAD_DIM": 256,
        "NUM_QUERY_HEADS": 6,
        "NUM_SPLITS": splits,
        "NUM_TILES": -(-topk // block_n),
        "BLOCK_M": 8,
        "BLOCK_N": block_n,
        "KV_E4M3": False,
        "FOLD_SCALES": True,
        "RESOLVED_INDICES": False,
    }
    return triton.compile(
        ASTSource(
            fn=kernel,
            signature=_with_constexprs(kernel, _SPLITK_RUNTIME),
            constexprs=constexprs,
        ),
        target=VOLTA,
        options={"num_warps": num_warps, "num_stages": 2},
    )


@pytest.mark.parametrize("num_warps", [2, 4])
def test_the_split_k_kernel_lowers_to_sm70_for_the_gathered_nvfp4_rows(num_warps):
    # 2 warps is the exact-TP4 decode launch (the two-warp partial), 4 the other
    # pre-Ampere profile. The kernel has more registers than the store and gather.
    _check(_gathered_splitk(num_warps), max_registers=255, max_stack=GATHERED_MAX_STACK)


def test_the_gathered_kernel_is_not_larger_than_the_e4m3_kernel_it_replaces():
    # Same tile and warps: the FP16 inputs skip the software decode, so the gathered
    # route cannot need more registers than the E4M3 route it sits next to.
    kernel = qsa_ops._qsa_sparse_paged_gqa_splitk_kernel
    runtime = {**_SPLITK_RUNTIME, "k_cache_ptr": "*u8", "v_cache_ptr": "*u8"}
    e4m3 = triton.compile(
        ASTSource(
            fn=kernel,
            signature=_with_constexprs(kernel, runtime),
            constexprs={
                "TOPK": 2051,
                "PAGE_SIZE": 1616,
                "PAGE_TABLE_WIDTH": 163,
                "GROUP_SIZE": 6,
                "HEAD_DIM": 256,
                "NUM_QUERY_HEADS": 6,
                "NUM_SPLITS": 64,
                "NUM_TILES": 129,
                "BLOCK_M": 8,
                "BLOCK_N": 16,
                "KV_E4M3": True,
                "FOLD_SCALES": True,
                "RESOLVED_INDICES": False,
            },
        ),
        target=VOLTA,
        options={"num_warps": 4, "num_stages": 2},
    )
    usage_e4m3, usage_gathered = _usage(e4m3), _usage(_gathered_splitk(4))
    if usage_e4m3 is not None and usage_gathered is not None:
        assert usage_gathered[0] <= usage_e4m3[0]


@pytest.mark.parametrize("fold_scales", [False, True])
def test_the_merge_kernel_lowers_to_sm70_with_and_without_scale_folding(fold_scales):
    kernel = qsa_ops._qsa_merge_splitk_kernel
    compiled = triton.compile(
        ASTSource(
            fn=kernel,
            signature=_with_constexprs(kernel, _MERGE_RUNTIME),
            constexprs={
                "HEAD_DIM": 256,
                "NUM_QUERY_HEADS": 6,
                "NUM_SPLITS": 64,
                "BLOCK_SPLITS": 64,
                "FOLD_SCALES": fold_scales,
            },
        ),
        target=VOLTA,
        options={"num_warps": 2, "num_stages": 1},
    )
    _check(compiled, max_registers=255, max_stack=MERGE_MAX_STACK)


def test_scale_folding_changes_only_the_kernels_that_fold():
    # FOLD_SCALES=False must compile to the same code whatever else is asked of the
    # kernel: the PTX without folding has no multiply by the scale arguments that the
    # folding one has. Compare the two PTX bodies, stripped of line info.
    def body(compiled):
        lines = []
        for line in compiled.asm["ptx"].splitlines():
            stripped = line.strip()
            if stripped.startswith((".loc", ".file", "//")) or not stripped:
                continue
            lines.append(re.sub(r"\s*//.*$", "", line))
        return "\n".join(lines).split(".section\t.debug_abbrev")[0]

    kernel = qsa_ops._qsa_sparse_paged_gqa_splitk_kernel
    constexprs = {
        "TOPK": 64,
        "PAGE_SIZE": 64,
        "PAGE_TABLE_WIDTH": 1,
        "GROUP_SIZE": 6,
        "HEAD_DIM": 256,
        "NUM_QUERY_HEADS": 6,
        "NUM_SPLITS": 1,
        "NUM_TILES": 4,
        "BLOCK_M": 8,
        "BLOCK_N": 16,
        "KV_E4M3": False,
        "RESOLVED_INDICES": False,
    }

    def build(fold):
        return triton.compile(
            ASTSource(
                fn=kernel,
                signature=_with_constexprs(kernel, _SPLITK_RUNTIME),
                constexprs={**constexprs, "FOLD_SCALES": fold},
            ),
            target=VOLTA,
            options={"num_warps": 4, "num_stages": 2},
        )

    plain, folded = body(build(False)), body(build(True))
    assert plain != folded
    assert plain.count("mul.f32") < folded.count("mul.f32")
