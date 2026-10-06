# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only check of the NVFP4 32-dim loader's e2m1 half-precision table.

The kernel header (fp8_kv_utils.cuh) keeps a 16-entry table of fp16 bit
patterns for the e2m1 codes and derives its PRMT lookup constants from it.
This test parses that table, compares it with ``nvfp4_kv.E2M1_GRID`` and
re-runs the device bit manipulation (PRMT lookup, sign move, half multiply)
in numpy for every 4-code word, so a wrong constant is caught without a GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

HEADER = (
    Path(__file__).resolve().parents[3]
    / "flash-attention-v100"
    / "kernel"
    / "fp8_kv_utils.cuh"
)


def _parse_table() -> list[int]:
    if not HEADER.exists():
        pytest.skip("flash-attention-v100 sources are not in this checkout")
    text = HEADER.read_text()
    match = re.search(
        r"kNvfp4E2m1HalfBits\[16\]\s*=\s*\{([^}]*)\}", text, flags=re.S
    )
    assert match, "kNvfp4E2m1HalfBits table not found in fp8_kv_utils.cuh"
    values = [int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    assert len(values) == 16
    return values


def _grid() -> tuple[float, ...]:
    nvfp4_kv = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv")
    return tuple(nvfp4_kv.E2M1_GRID)


def _half_from_bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def test_e2m1_half_table_matches_reference_grid():
    table = _parse_table()
    grid = _grid()
    assert len(grid) == 8
    for code in range(16):
        expected = grid[code & 7] * (-1.0 if code & 8 else 1.0)
        assert _half_from_bits(table[code]) == expected, f"code {code}"
        # bit-exact sign of zero as well
        assert (table[code] >> 15) == (code >> 3)


def _prmt(x: int, y: int, selector: int) -> int:
    """PTX/CUDA __byte_perm for selectors that never use the sign mode."""
    src = [(x >> (8 * i)) & 0xFF for i in range(4)] + [
        (y >> (8 * i)) & 0xFF for i in range(4)
    ]
    out = 0
    for i in range(4):
        nib = (selector >> (4 * i)) & 0xF
        assert nib < 8, "sign-replicate mode must not be reachable"
        out |= src[nib] << (8 * i)
    return out


def _decode_four(table: list[int], w16: int) -> tuple[int, int]:
    """Python mirror of nvfp4_data2_to_half2x2 (header)."""
    lo = hi = 0
    for i in range(4):
        lo |= (table[i] >> 8) << (8 * i)
        hi |= (table[4 + i] >> 8) << (8 * i)
    mags = _prmt(lo, hi, w16 & 0x7777)
    s = w16 & 0x8888
    h01 = _prmt(mags, 0, 0x1404) | ((s << 12) & 0x00008000) | ((s << 24) & 0x80000000)
    h23 = _prmt(mags, 0, 0x3424) | ((s << 4) & 0x00008000) | ((s << 16) & 0x80000000)
    return h01 & 0xFFFFFFFF, h23 & 0xFFFFFFFF


def test_prmt_lookup_matches_table_for_all_four_code_words():
    table = _parse_table()
    for w16 in range(1 << 16):
        h01, h23 = _decode_four(table, w16)
        codes = [(w16 >> (4 * i)) & 0xF for i in range(4)]
        got = [h01 & 0xFFFF, h01 >> 16, h23 & 0xFFFF, h23 >> 16]
        assert got == [table[c] for c in codes], f"w16={w16:#06x}"


def test_half_multiply_is_exact_for_every_code_and_e4m3_scale():
    table = _parse_table()
    vals = np.array([_half_from_bits(b) for b in table], dtype=np.float16)
    for raw in range(256):
        if (raw & 0x7F) == 0x7F:  # E4M3FN NaN
            continue
        sign = -1.0 if raw & 0x80 else 1.0
        exp, mant = (raw >> 3) & 0xF, raw & 7
        scale = (
            mant * 2.0**-9 if exp == 0 else (1 + mant / 8) * 2.0 ** (exp - 7)
        ) * sign
        scale_h = np.float16(scale)
        assert float(scale_h) == scale  # E4M3 -> fp16 is exact
        half_product = vals * scale_h
        exact = vals.astype(np.float64) * scale
        assert np.array_equal(half_product.astype(np.float64), exact), f"raw={raw}"
