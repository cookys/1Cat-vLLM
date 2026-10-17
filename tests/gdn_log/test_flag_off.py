# SPDX-License-Identifier: Apache-2.0
"""Flag off must be stage-FIX: the G-LOG edits add constexpr-gated branches
only. Compared bit for bit against the stage-FIX file itself (CPU, interpreter)."""

import importlib.util
import os

import pytest
import torch

from tests.gdn_log import glog_fixtures as fx

STAGE_FIX = os.environ.get("GLOG_STAGE_FIX", "/data/bench/s2-impl/c1fix/stage-FIX")
FSG = "vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py"


def _load_stage_fix():
    path = os.path.join(STAGE_FIX, FSG)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    spec = importlib.util.spec_from_file_location(
        "vllm.model_executor.layers.fla.ops.fsg_stage_fix", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.exp = lambda x: mod.tl.exp(x)
    return mod


@pytest.mark.parametrize("accepted", [1, 3, 8])
def test_classic_kernel_matches_stage_fix(accepted):
    old = _load_stage_fix()
    inputs = fx.make_round_inputs(2, seed=3)
    outs, states = [], []
    for runner in ("new", "old"):
        pages = fx.Pages(2 * fx.WIDTH)
        gen = torch.Generator().manual_seed(1)
        pages.ssm[:] = torch.randn(pages.ssm.shape, generator=gen) * 0.05
        idx = torch.arange(2 * fx.WIDTH, dtype=torch.int32).view(2, fx.WIDTH)
        sel = torch.full((2,), accepted, dtype=torch.int32)
        if runner == "new":
            out = fx.run_verify(pages.ssm, idx, sel, inputs, gdn_log=None)
        else:
            mixed, g, beta, cu = inputs
            out = torch.zeros(mixed.shape[0], 1, fx.HV, fx.V, dtype=torch.float16)
            old.fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
                A_log=torch.zeros(fx.HV), a=g, b=beta, dt_bias=torch.zeros(fx.HV),
                mixed_qkv=mixed, num_q_heads=fx.H, num_v_heads=fx.HV,
                head_k_dim=fx.K, head_v_dim=fx.V, scale=fx.K**-0.5,
                initial_state=pages.ssm, out=out, cu_seqlens=cu,
                ssm_state_indices=idx, num_accepted_tokens=sel,
                use_qk_l2norm_in_kernel=True, precomputed_g=g,
                precomputed_beta=beta, quantize_state_each_step=False,
                match_recurrent_schedule=True, match_recurrent_numerics=True,
            )
        outs.append(out)
        states.append(pages.raw.clone())
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(states[0], states[1])  # incl. every byte outside the ssm view


def test_flag_off_leaves_spec_untouched(monkeypatch):
    from vllm.v1.gdn_log import gdn_log_active

    monkeypatch.delenv("VLLM_DFLASH2_GDN_LOG", raising=False)

    class Cfg:  # no speculative_config: nospec engine
        speculative_config = None

    assert gdn_log_active(Cfg()) is False
    monkeypatch.setenv("VLLM_DFLASH2_GDN_LOG", "1")
    assert gdn_log_active(Cfg()) is False  # nospec stays byte-identical even with flag
