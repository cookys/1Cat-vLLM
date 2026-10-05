# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NVFP4 decode routes are bit for bit what they were before option B' (plan 071).

Option B' (the prefill scratch route) must not touch a decode step: a batch of fewer
than ``VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS`` rows reads the cache through the FP16
gather or the fused reader exactly as at the base commit ``112830b6f``, whatever the
scratch knob says. The golden outputs in ``data/nvfp4_kv_decode_route_golden.pt`` were
captured at that commit, with this file's own batch builder::

    cd <worktree at 112830b6f> && TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= \\
        PYTHONPATH=$PWD python <this file> --capture <this dir>/data/\\
nvfp4_kv_decode_route_golden.pt

They are stored, not regenerated at test time, because the base worktree is a
developer's checkout and a CI machine has none. A route change that is intended must
re-capture the golden at the commit that defines the new decode contract and say so in
the commit message. The batch has illegal lanes (a negative token, one far past the
table, a row that belongs to no request), whose outputs must stay exact zeros.

With ``TRITON_INTERPRET=1`` on a machine without a GPU::

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \\
        python -m pytest --noconftest \\
        tests/models/qwen4_exp/test_nvfp4_kv_decode_route_golden.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="the golden is the CPU interpreter's output: needs TRITON_INTERPRET=1",
)

if not torch.cuda.is_available():
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2

from tests.models.qwen4_exp.test_nvfp4_kv_decode import (  # noqa: E402
    Env,
    _TorchMerge,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "nvfp4_kv_decode_route_golden.pt"
K_SCALE, V_SCALE = 0.0213623046875, 0.0404924675822258
FUSED_ENV = "VLLM_SM70_QSA_NVFP4_FUSED_READER"
SCRATCH_ENV = "VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH"
ROWS, TOPK = 5, 48


def build_batch() -> Env:
    """Five decode rows at topk 48 over two requests, with illegal lanes."""
    token_to_req = torch.tensor([0, 1, 0, 1, -1], dtype=torch.int32)
    env = Env(rows=ROWS, topk=TOPK, seed=7105, token_to_req=token_to_req)
    env.indices[0, 0] = -1  # a negative token
    env.indices[0, 1] = 2**30  # far past the block table
    env.indices[1, 5] = -7
    return env


def run(env: Env, fused: bool | None, scaled: bool) -> torch.Tensor:
    kwargs = {"k_scale": K_SCALE, "v_scale": V_SCALE} if scaled else {}
    return env.nvfp4(nvfp4_fused_reader=fused, **kwargs).cpu()


def _capture() -> dict[str, torch.Tensor]:
    env = build_batch()
    golden = {}
    for name, fused in (("gather", False), ("fused", True)):
        for scales, scaled in (("unit", False), ("scaled", True)):
            golden[f"{name}_{scales}"] = run(env, fused, scaled).clone()
    return golden


@pytest.fixture(autouse=True)
def _interpreter_merge(monkeypatch):
    monkeypatch.setattr(qsa_ops, "_qsa_merge_splitk_kernel", _TorchMerge())


@pytest.fixture(scope="module")
def golden() -> dict[str, torch.Tensor]:
    return torch.load(GOLDEN, weights_only=True)


@pytest.fixture(scope="module")
def env() -> Env:
    return build_batch()


def _bit_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    different = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
    assert different == 0, f"{different} of {actual.numel()} elements differ"


def test_the_golden_has_live_and_empty_lanes(golden):
    for name, out in golden.items():
        assert out.shape[0] == ROWS, name
        assert out[:4].abs().sum() > 0, name
        assert not out[4].any(), f"{name}: the row of no request is not exact zero"


@pytest.mark.parametrize("scratch", ["0", "1"])
@pytest.mark.parametrize("scaled", [False, True], ids=["unit", "scaled"])
def test_the_gather_route_is_the_base_commits_with_the_scratch_knob_off_and_on(
    monkeypatch, env, golden, scratch, scaled
):
    monkeypatch.setenv(SCRATCH_ENV, scratch)
    out = run(env, False, scaled)
    _bit_equal(out, golden["gather_scaled" if scaled else "gather_unit"])


@pytest.mark.parametrize("scratch", ["0", "1"])
def test_the_fused_reader_is_the_base_commits_with_the_scratch_knob_off_and_on(
    monkeypatch, env, golden, scratch
):
    monkeypatch.setenv(SCRATCH_ENV, scratch)
    _bit_equal(run(env, True, True), golden["fused_scaled"])
    _bit_equal(run(env, True, False), golden["fused_unit"])


@pytest.mark.parametrize("scratch", ["0", "1"])
@pytest.mark.parametrize("fused_env", ["0", "1"])
def test_the_knobs_select_the_route_and_the_scratch_knob_never_moves_a_decode_step(
    monkeypatch, env, golden, fused_env, scratch
):
    """No explicit argument: ``VLLM_SM70_QSA_NVFP4_FUSED_READER`` picks the route."""
    monkeypatch.setenv(FUSED_ENV, fused_env)
    monkeypatch.setenv(SCRATCH_ENV, scratch)
    route = "fused" if fused_env == "1" else "gather"
    _bit_equal(run(env, None, False), golden[f"{route}_unit"])
    _bit_equal(run(env, None, True), golden[f"{route}_scaled"])


def test_the_two_routes_differ_so_the_pin_above_can_tell_them_apart(golden):
    """Why the kernel check pins the knobs: the fused reader is not the gather.

    At unit layer scales a few outputs differ at FP16 rounding (the two dots add the
    same products in another order); at the small realistic scales they happen to round
    alike on this batch, so only the unit pair tells the routes apart.
    """
    assert not torch.equal(golden["gather_unit"], golden["fused_unit"])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--capture":
        raise SystemExit(f"usage: {sys.argv[0]} --capture <out.pt>")
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("needs TRITON_INTERPRET=1")
    qsa_ops._qsa_merge_splitk_kernel = _TorchMerge()
    for knob in (FUSED_ENV, SCRATCH_ENV):
        os.environ.pop(knob, None)
    with torch.inference_mode():
        out = _capture()
    torch.save(out, sys.argv[2])
    print(f"wrote {sys.argv[2]} from {qsa_ops.__file__}")
    print({k: tuple(v.shape) for k, v in out.items()})
