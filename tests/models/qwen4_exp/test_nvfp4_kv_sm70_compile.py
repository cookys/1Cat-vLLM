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


def _usage(compiled) -> tuple[int, int] | None:
    """(registers, local bytes) of the cubin, or None without cuobjdump."""
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
    match = re.search(r"REG:(\d+) STACK:\d+ SHARED:\d+ LOCAL:(\d+)", report)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _check(compiled, max_registers=64) -> None:
    ptx = compiled.asm["ptx"]
    assert re.search(r"\.target sm_70\b", ptx)
    assert len(compiled.asm["cubin"]) > 0
    for banned in BANNED:
        assert banned not in ptx, banned
    usage = _usage(compiled)
    if usage is not None:
        registers, local = usage
        assert registers <= max_registers, registers
        assert local == 0, "the kernel spills to local memory"


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
