# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU policy test: env parsing of the Q1.23 P1.2 NVFP4 smallq knobs.

The C++ side reads the variables with getenv at dispatch time:
  VLLM_FLASH_V100_NVFP4_SMALLQ_192  default ON, only a leading '0' disables
  VLLM_FLASH_V100_NVFP4_HALF_P      default OFF, only a leading '1' enables
This test pins that contract against the source text so a silent default flip
fails on CPU, without needing a GPU or a rebuild.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

KERNEL = (
    Path(__file__).resolve().parents[3]
    / "flash-attention-v100"
    / "kernel"
    / "flash_decode_paged.cu"
)


def _function_body(name: str) -> str:
    if not KERNEL.exists():
        pytest.skip("flash-attention-v100 sources not present")
    text = KERNEL.read_text()
    match = re.search(r"bool\s+" + name + r"\(\)\s*\{(.*?)\n\}", text, re.S)
    assert match, f"{name} not found"
    return match.group(1)


def _eval_policy(body: str, value: str | None) -> bool:
    """Evaluate the one-line `return` of an env policy for a given value."""
    ret = re.search(r"return\s+(.*?);", body, re.S).group(1)
    present = value is not None
    first = value[0] if value else "\0"
    expr = (
        ret.replace("value == nullptr", str(not present))
        .replace("value != nullptr", str(present))
        .replace("value[0] != '0'", str(first != "0"))
        .replace("value[0] == '1'", str(first == "1"))
        .replace("&&", " and ")
        .replace("||", " or ")
    )
    return bool(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307


@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("1", True), ("", True), ("0", False), ("0x", False), ("2", True)],
)
def test_smallq_192_is_opt_out(value, expected):
    body = _function_body("nvfp4_smallq_192_enabled")
    assert "VLLM_FLASH_V100_NVFP4_SMALLQ_192" in body
    assert _eval_policy(body, value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("0", False), ("", False), ("1", True), ("2", False)],
)
def test_half_p_is_opt_in(value, expected):
    body = _function_body("nvfp4_half_p_enabled")
    assert "VLLM_FLASH_V100_NVFP4_HALF_P" in body
    assert _eval_policy(body, value) is expected
