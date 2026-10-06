# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 paged KV -> FP16 bridge of Flash-V100 (Q1.23 P3).

GPU-only (SM70). The cache is built with the plan-071 reference store and the
bridge output is compared with the reference dequantizer.

Bridge contract (same as the FP8 bridges): fp16 output
``[batch * out_blocks, out_page, H, 256]`` with an identity block table per
batch row, the layer scale folded into the output, tokens in
``[seq_len, round_up(seq_len, 16))`` exact zeros, nothing written beyond.
"""

from __future__ import annotations

import pytest
import torch

HEAD_DIM = 256
SENTINEL = 7.0


def _require_nvfp4_bridge():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Flash-V100 NVFP4 bridge is SM70-only")
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    if not hasattr(flash_attn_v100, "flash_attn_nvfp4_kv_available"):
        pytest.skip("Flash-V100 Python package lacks the NVFP4 probe")
    if not flash_attn_v100.flash_attn_nvfp4_kv_available(min_version=2):
        pytest.skip("Flash-V100 extension lacks the NVFP4 KV bridge")
    if not hasattr(flash_attn_v100, "nvfp4_paged_kv_to_fp16"):
        pytest.skip("Flash-V100 Python package lacks nvfp4_paged_kv_to_fp16")
    return flash_attn_v100


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in fp16 steps on the ordered bit pattern (sign-magnitude aware)."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        bits = x.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _run_case(page, heads, seq_lens, nb, k_scale, v_scale, seed):
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv

    flash_attn_v100 = _require_nvfp4_bridge()
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(seq_lens)
    blocks = batch * nb + 3
    physical = torch.randperm(blocks, dtype=torch.int32, device=device)[: batch * nb]
    block_table = physical.view(batch, nb).contiguous()

    cache = torch.zeros(
        blocks,
        2,
        page,
        heads,
        nvfp4_kv.nvfp4_kv_row_bytes(HEAD_DIM),
        dtype=torch.uint8,
        device=device,
    )
    for b, n in enumerate(seq_lens):
        keys = torch.randn(n, heads, HEAD_DIM, device=device).mul_(0.5).half()
        values = torch.randn(n, heads, HEAD_DIM, device=device).mul_(0.5).half()
        token = torch.arange(n, device=device)
        slots = block_table[b].long()[token // page] * page + token % page
        nvfp4_kv.reshape_and_cache_nvfp4_reference(
            keys,
            values,
            cache,
            slots.to(torch.int32),
            k_scale=k_scale,
            v_scale=v_scale,
        )

    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv.nvfp4_kv_split_views(cache)
    sides = {}
    for name, data, scales, ls in (
        ("k", k_data, k_scales, k_scale),
        ("v", v_data, v_scales, v_scale),
    ):
        # Reference of the brief: dequantize with the layer scale, cast to fp16.
        ref = nvfp4_kv.dequantize_kv_nvfp4(
            data, scales, layer_scale=ls, out_dtype=torch.float16
        )
        # Bridge arithmetic model: exact fp16 product first, then one fp32
        # multiply by the layer scale and one rounding to fp16.
        unit = nvfp4_kv.dequantize_kv_nvfp4(
            data, scales, layer_scale=1.0, out_dtype=torch.float16
        )
        model = (unit.float() * ls).half()
        sides[name] = (ref, model)

    out_blocks = nb
    key_out = torch.full(
        (batch * out_blocks, page, heads, HEAD_DIM),
        SENTINEL,
        dtype=torch.float16,
        device=device,
    )
    value_out = key_out.clone()
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    flash_attn_v100.nvfp4_paged_kv_to_fp16(
        cache[:, 0],
        cache[:, 1],
        block_table,
        sl,
        key_out,
        value_out,
        k_scale,
        v_scale,
    )
    torch.cuda.synchronize()

    max_abs = 0.0
    for name, out in (("k", key_out), ("v", value_out)):
        ref, model = sides[name]
        for b, n in enumerate(seq_lens):
            got = out[b * out_blocks : (b + 1) * out_blocks].reshape(
                nb * page, heads, HEAD_DIM
            )
            padded = min(nb * page, (n + 15) // 16 * 16)
            exp_ref = ref[block_table[b].long()].reshape(nb * page, heads, HEAD_DIM)
            exp_model = model[block_table[b].long()].reshape(nb * page, heads, HEAD_DIM)
            live = got[:n]
            # Live rows: exact against the arithmetic model, within 1 ulp of
            # the dequantizer reference (bit-exact when the scale is 1.0).
            assert torch.equal(live, exp_model[:n]), f"{name} row {b}: model mismatch"
            ulp = _ulp_distance(live, exp_ref[:n])
            limit = 0 if (k_scale == 1.0 and v_scale == 1.0) else 1
            assert int(ulp.max()) <= limit, (
                f"{name} row {b}: {int(ulp.max())} ulp from the dequantizer"
            )
            max_abs = max(
                max_abs, float((live.float() - exp_ref[:n].float()).abs().max())
            )
            # Padding rows are exact zeros.
            assert not got[n:padded].view(torch.int16).any(), (
                f"{name} row {b}: padding rows are not zero"
            )
            # Rows beyond the padded length are untouched.
            assert torch.all(got[padded:] == SENTINEL), (
                f"{name} row {b}: rows beyond the padding were written"
            )
    return max_abs


def test_nvfp4_bridge_probe_version_2():
    flash_attn_v100 = _require_nvfp4_bridge()
    assert flash_attn_v100.flash_attn_nvfp4_kv_available(min_version=2)
    assert not flash_attn_v100.flash_attn_nvfp4_kv_available(min_version=10**6)


@pytest.mark.parametrize(
    "page,heads,seq_lens,nb",
    [
        (32, 1, [50], 2),  # two pages, H=1, 50 = 3*16+2, not a page multiple
        (32, 2, [33, 64], 2),  # H=2, batch of 2, one row exactly two pages
        (16, 1, [17], 3),  # tail inside the second 16-row tile
        (16, 2, [16, 1], 2),  # full tile and a single-token row
        (2048, 1, [2100], 2),  # production page size, crosses a page
    ],
)
def test_nvfp4_bridge_unit_scale_is_bit_exact(page, heads, seq_lens, nb):
    _run_case(page, heads, seq_lens, nb, 1.0, 1.0, seed=7 + page + heads)


@pytest.mark.parametrize(
    "page,heads,seq_lens,nb",
    [
        (32, 1, [50], 2),
        (32, 2, [33, 64], 2),
        (2048, 1, [2100], 2),
    ],
)
def test_nvfp4_bridge_folds_layer_scales(page, heads, seq_lens, nb):
    max_abs = _run_case(page, heads, seq_lens, nb, 0.75, 1.25, seed=99 + page)
    print(f"max|d| vs dequantizer (layer scales 0.75/1.25): {max_abs:.3e}")
