# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the draft-only NVFP4 lm_head (VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD).

The native QPN2 ops need an SM70 GPU, so the GEMM is replaced by a CPU matmul
against the dequantized weights; everything else (quantizer, candidate
selection, FP16 rerank, TP reduction, eligibility gate) is the production code.
"""

import inspect
from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm import envs
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
    cast_to_fp4,
    dequantize_to_dtype,
    ref_nvfp4_quant,
    ref_nvfp4_quant_dequant,
)
from vllm.v1.worker.gpu.spec_decode.eagle import draft_nvfp4_lm_head as mod
from vllm.v1.worker.gpu.spec_decode.eagle.draft_nvfp4_lm_head import (
    DraftNvfp4LMHead,
    build_draft_nvfp4_lm_head,
    quantize_fp16_to_nvfp4_packed,
    ref_global_scale,
    unpack_nvfp4_packed,
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MTP_GREEDY = SimpleNamespace(method="mtp", draft_sample_method="greedy")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _lm_head(weight, start=0, pad=0, bias=None, org_vocab_size=None):
    return SimpleNamespace(
        weight=weight,
        bias=bias,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=pad
        ),
        org_vocab_size=org_vocab_size,
    )


@pytest.fixture
def cpu_ops(monkeypatch):
    """Identity layout 'prepare' and a CPU dequant-matmul in place of QPN2."""
    calls = []

    def prepare(packed, scales):
        return packed.clone(), scales.clone()

    def gemm(out, x, codes, scales, global_scale, split_k, chains):
        assert out.dtype == torch.float16 and x.dtype == torch.float16
        assert out.is_contiguous() and x.is_contiguous()
        assert 1 <= x.shape[0] <= 64
        weight = unpack_nvfp4_packed(codes, scales, global_scale, kernel_exact=True)
        out.copy_((x.float() @ weight.T).half())
        calls.append((x.shape[0], split_k, chains))

    monkeypatch.setattr(mod, "_prepare_qpn2", prepare)
    monkeypatch.setattr(mod, "_qpn2_gemm_out", gemm)
    # No process group in a unit test: default to TP=1 (tests re-patch it).
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    return calls


def _random_head(n=512, k=256, rerank_k=64, start=0, pad=0, seed=0, scale=0.5):
    gen = torch.Generator().manual_seed(seed)
    weight = (torch.randn(n, k, generator=gen) * scale / k**0.5).half()
    return weight, DraftNvfp4LMHead(
        _lm_head(weight, start=start, pad=pad), rerank_k=rerank_k
    )


def _hidden(m, k=256, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=gen).half()


# --------------------------------------------------------------------------
# (a) quantizer vs the repo reference
# --------------------------------------------------------------------------
def _test_matrix(dtype):
    gen = torch.Generator().manual_seed(7)
    w = torch.randn(256, 2560, generator=gen) * 0.05
    w[0] = 0.0  # all-zero row
    w[1] = 1e-7  # tiny non-zero row, far below the shard maximum
    w[2, 0] = -1.0  # the shard maximum, negative
    w[3, :16] = torch.tensor([1.0, -1.0] * 8)  # +-amax inside one block
    return w.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quantizer_matches_repo_reference(dtype):
    w = _test_matrix(dtype)
    packed, scales, g = quantize_fp16_to_nvfp4_packed(w)
    # g is the multiplier, the repo reference takes gq = 1 / g.
    gq = ref_global_scale(g, "cpu")
    ref_fp4, ref_scale = ref_nvfp4_quant(w, gq, 16)

    values = break_fp4_bytes(packed, torch.float32)
    assert torch.equal(values, ref_fp4)
    mine = scales.float()
    # The reference leaves blocks whose scale rounds to 0 at scale 0; the
    # quantizer clamps them to the smallest positive E4M3 value instead.
    zero = ref_scale == 0
    assert zero.any()
    assert torch.equal(mine[~zero], ref_scale[~zero])
    assert (mine[zero] == 2.0**-9).all()

    dequant = unpack_nvfp4_packed(packed, scales, g)
    assert torch.equal(
        dequant,
        dequantize_to_dtype(
            packed, scales, torch.tensor(g), torch.float32, 16, swizzle=False
        ),
    )
    ref_dequant = ref_nvfp4_quant_dequant(w.float(), gq, 16)
    torch.testing.assert_close(dequant, ref_dequant, rtol=1e-5, atol=1e-9)
    rel = (dequant - w.float()).norm() / w.float().norm()
    assert rel < 0.2


def test_global_scale_and_edge_blocks():
    w = _test_matrix(torch.float16)
    packed, scales, g = quantize_fp16_to_nvfp4_packed(w)
    assert g == pytest.approx(1.0 / (6 * 448))
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(w.shape)
    # all-zero row and the sub-E4M3 row: smallest scale, all-zero codes
    for row in (0, 1):
        assert (scales[row].float() == 2.0**-9).all()
        assert (codes[row] == 0).all()
    # the shard maximum maps to -6 with block scale 448
    assert scales[2, 0].float() == 448.0
    assert codes[2, 0] == 0b1111
    # +-amax inside one block: codes 7 / 15 alternate
    assert scales[3, 0].float() == 448.0
    assert (codes[3, 0:16:2] == 7).all() and (codes[3, 1:16:2] == 15).all()


def test_fp4_codes_match_cast_to_fp4_including_ties():
    ties = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.2500001, 0.7499999, 6.0, 7.0]
    )
    random = torch.randn(4096) * 2.5
    x = torch.cat([ties, -ties, random, torch.zeros(1)]).clamp(-6.0, 6.0)
    codes = mod._fp4_codes(x)
    decoded = E2M1[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    expected = cast_to_fp4(x.clone())
    assert torch.equal(decoded, expected)
    # a negative value that rounds to zero is code 0, never the code of -0
    assert mod._fp4_codes(torch.tensor([-0.1]))[0] == 0


# --------------------------------------------------------------------------
# (b) shape / dtype contract
# --------------------------------------------------------------------------
def test_packing_contract_and_chunk_invariance():
    w = _test_matrix(torch.float16)
    packed, scales, g = quantize_fp16_to_nvfp4_packed(w)
    assert packed.dtype == torch.uint8 and packed.shape == (256, 1280)
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (256, 160)
    assert isinstance(g, float) and g > 0
    assert torch.tensor(g, dtype=torch.float32).item() == g  # float32 exact
    # low nibble = even K index, high nibble = odd K index
    unpacked = break_fp4_bytes(packed, torch.float32)
    low = packed.long() & 0xF
    assert torch.equal(
        unpacked[:, 0::2], E2M1[low & 7] * (1 - 2 * ((low >> 3) & 1)).float()
    )
    high = packed.long() >> 4
    assert torch.equal(
        unpacked[:, 1::2], E2M1[high & 7] * (1 - 2 * ((high >> 3) & 1)).float()
    )
    # chunking (including a ragged last chunk) must not change a single bit
    for chunk_rows in (1, 100, 255, 256, 4096):
        p2, s2, g2 = quantize_fp16_to_nvfp4_packed(w, chunk_rows=chunk_rows)
        assert torch.equal(p2, packed)
        assert torch.equal(s2.view(torch.uint8), scales.view(torch.uint8))
        assert g2 == g


def test_all_zero_shard_gets_unit_multiplier():
    packed, scales, g = quantize_fp16_to_nvfp4_packed(torch.zeros(32, 128).half())
    assert g == 1.0
    assert (packed == 0).all() and (scales.float() == 2.0**-9).all()


@pytest.mark.parametrize(
    "weight",
    [
        torch.zeros(4, 128, dtype=torch.int32),
        torch.zeros(4, 120).half(),
        torch.zeros(128).half(),
        torch.full((4, 128), float("nan")).half(),
        torch.full((4, 128), float("inf")).half(),
        torch.full((4, 128), 1e-20).float(),
    ],
)
def test_quantizer_rejects_bad_input(weight):
    with pytest.raises(ValueError):
        quantize_fp16_to_nvfp4_packed(weight)


def test_kernel_exact_dequant_is_fp16_rounded():
    packed, scales, g = quantize_fp16_to_nvfp4_packed(_test_matrix(torch.float16))
    plain = unpack_nvfp4_packed(packed, scales, g)
    exact = unpack_nvfp4_packed(packed, scales, g, kernel_exact=True)
    assert torch.equal(exact, exact.half().float())
    torch.testing.assert_close(exact, plain, rtol=2.0**-9, atol=1e-7)


# --------------------------------------------------------------------------
# (c) top_tokens logic with the GEMM replaced by a CPU matmul
# --------------------------------------------------------------------------
def _check_against_oracle(head, weight, hidden, tokens, rerank_k, start=0):
    """tokens must be the exact argmax over the NVFP4 top-R candidates."""
    exact = hidden.float() @ weight.float().T
    q = head.out[: hidden.shape[0]].float()
    hits = 0
    for row in range(hidden.shape[0]):
        threshold = q[row].topk(rerank_k).values.min()
        local = int(tokens[row]) - start
        assert q[row, local] >= threshold  # chosen token is a candidate
        true_best = int(exact[row].argmax())
        if q[row, true_best] > threshold:  # strictly inside the top-R
            assert local == true_best
        hits += local == true_best
    return hits


@pytest.mark.parametrize("rerank_k", [8, 32, 64, 128])
def test_rerank_returns_fp16_argmax_whenever_it_is_in_the_top_r(cpu_ops, rerank_k):
    weight, head = _random_head(n=512, rerank_k=rerank_k, start=1000)
    hidden = _hidden(4)
    tokens = head.top_tokens(hidden)
    assert tokens.dtype == torch.int64 and tokens.shape == (4,)
    hits = _check_against_oracle(head, weight, hidden, tokens, rerank_k, start=1000)
    assert cpu_ops == [(4, head.split_k, head.accumulator_chains)]
    if rerank_k >= 64:
        assert hits >= 3


def test_rerank_many_random_rows(cpu_ops):
    weight, head = _random_head(n=512, rerank_k=64, seed=3)
    hits = total = 0
    for seed in range(8):
        hidden = _hidden(8, seed=seed)
        tokens = head.top_tokens(hidden)
        hits += _check_against_oracle(head, weight, hidden, tokens, 64)
        total += 8
    assert hits >= 0.9 * total


def test_rerank_zero_returns_raw_nvfp4_argmax(cpu_ops):
    weight, head = _random_head(n=512, rerank_k=0, start=96)
    hidden = _hidden(3)
    tokens = head.top_tokens(hidden)
    q = head.out[:3].float()
    assert torch.equal(tokens, q.argmax(dim=-1) + 96)


def test_rerank_fixes_raw_nvfp4_misrankings(cpu_ops):
    weight, raw = _random_head(n=512, rerank_k=0, seed=11)
    _, reranked = _random_head(n=512, rerank_k=64, seed=11)
    hidden = _hidden(64, seed=12)
    truth = (hidden.float() @ weight.float().T).argmax(-1)
    raw_tokens = raw.top_tokens(hidden)
    tokens = reranked.top_tokens(hidden)
    q = reranked.out[:64].float()
    in_top_r = q.gather(1, truth[:, None]).squeeze(1) > q.topk(64).values[:, -1]
    raw_wrong = raw_tokens != truth
    # 4-bit logits do flip the argmax on a good share of rows ...
    assert raw_wrong.sum() >= 4
    # ... and the exact rerank restores every row whose winner is in the top-R.
    assert (tokens[in_top_r] == truth[in_top_r]).all()
    assert (tokens != truth).sum() < raw_wrong.sum()


# --------------------------------------------------------------------------
# tie-breaking: the smallest id wins, like torch.max / get_top_tokens
# --------------------------------------------------------------------------
def test_rerank_tie_picks_the_smallest_id(cpu_ops):
    # Every logit is 0, so FP16 argmax (and LogitsProcessor.get_top_tokens)
    # returns id 0.  The candidates come out of topk(sorted=False) in an
    # arbitrary order; the rerank must not depend on it (it once returned 64).
    weight = torch.zeros(128, 128, dtype=torch.float16)
    x = torch.ones(1, 128, dtype=torch.float16)
    assert int((x.float() @ weight.float().T).argmax()) == 0
    for start in (0, 1000):
        head = DraftNvfp4LMHead(_lm_head(weight, start=start), rerank_k=128)
        assert head.top_tokens(x).tolist() == [start]
        value, local_id = head.local_top_tokens(x)
        assert value.tolist() == [0.0] and local_id.tolist() == [start]


def _tie_weights():
    """[256, 128] noise rows plus exact duplicate rows that tie for the win."""
    gen = torch.Generator().manual_seed(31)
    weight = (torch.randn(256, 128, generator=gen) * 0.01).half()
    lo = torch.cat([torch.ones(64), torch.zeros(64)]).half()
    hi = torch.cat([torch.zeros(64), torch.ones(64)]).half()
    weight[[40, 200]] = lo  # batch row 0: two-way tie, smallest id 40
    weight[[130, 7, 250]] = hi  # batch row 1: three-way tie, smallest id 7
    weight[[10, 11]] = lo * 0.5  # a lower tie group inside the top-R: loses
    return weight, torch.stack([lo, hi])


@pytest.mark.parametrize("rerank_k", [8, 64])
def test_rerank_ties_in_several_rows_pick_the_smallest_id(
    cpu_ops, monkeypatch, rerank_k
):
    weight, x = _tie_weights()
    head = DraftNvfp4LMHead(_lm_head(weight, start=500), rerank_k=rerank_k)
    # What the FP16 path (get_top_tokens) answers: the first maximum.
    reference = (x.float() @ weight.float().T).half().argmax(-1)
    assert reference.tolist() == [40, 7]

    real_topk = torch.topk

    def candidates_in_descending_id_order(q, r, dim=-1, sorted=False, out=None):
        values, ids = real_topk(q, r, dim=dim, sorted=True)
        order = ids.argsort(dim=-1, descending=True)
        values, ids = values.gather(-1, order), ids.gather(-1, order)
        out[0].copy_(values)
        out[1].copy_(ids)
        return values, ids

    # The natural order and the adversarial one (largest tied id first) agree.
    assert head.top_tokens(x).tolist() == [540, 507]
    monkeypatch.setattr(torch, "topk", candidates_in_descending_id_order)
    assert head.top_tokens(x).tolist() == [540, 507]


def test_rerank_unique_winner_beats_lower_tie_groups(cpu_ops):
    # The exact winner is unique; the tie groups among the lower candidates
    # (rows 10/11 and the duplicated rows of the other batch row) must not
    # change the answer.
    weight, x = _tie_weights()
    weight[77] = x[0] * 2.0  # unique winner for batch row 0
    head = DraftNvfp4LMHead(_lm_head(weight), rerank_k=8)
    assert head.top_tokens(x[:1]).tolist() == [77]


@pytest.mark.parametrize("rerank_k", [8, 64])
def test_rerank_nan_row_never_leaks_the_tie_break_sentinel(cpu_ops, rerank_k):
    _, head = _random_head(n=256, k=256, rerank_k=rerank_k, start=500)
    hidden = _hidden(3)
    hidden[1, 5] = float("nan")
    values, ids = head.local_top_tokens(hidden)
    assert values[1].isnan() and not values[[0, 2]].isnan().any()
    assert ((ids >= 500) & (ids < 500 + 256)).all()


def test_rerank_zero_tie_picks_the_smallest_id(cpu_ops):
    weight = torch.zeros(128, 128, dtype=torch.float16)
    head = DraftNvfp4LMHead(_lm_head(weight, start=32), rerank_k=0)
    assert head.top_tokens(torch.ones(2, 128, dtype=torch.float16)).tolist() == [32, 32]


def test_fp32_rerank_resolves_fp16_near_ties_differently(cpu_ops):
    """Documented behaviour: the draft can differ from the FP16 path (E2).

    Both logits are one FP16 value (1.0) in the FP16 GEMM, so the FP16 argmax is
    the first id.  The FP32 rerank sees 1 + 2**-12 > 1 and picks id 1.  The
    target distribution is unchanged: the draft is only a proposal.
    """
    weight = torch.zeros(128, 128, dtype=torch.float16)
    weight[0, 0] = 1.0
    weight[1, 0] = 1.0
    weight[1, 1] = 2.0**-12
    x = torch.ones(1, 128, dtype=torch.float16)
    fp16_logits = (x.float() @ weight.float().T).half()
    assert fp16_logits[0, 0] == fp16_logits[0, 1] == 1.0
    assert int(fp16_logits.argmax()) == 0
    head = DraftNvfp4LMHead(_lm_head(weight), rerank_k=128)
    assert head.top_tokens(x).tolist() == [1]


def test_padding_rows_are_masked(cpu_ops):
    k = 256
    gen = torch.Generator().manual_seed(5)
    weight = (torch.randn(256, k, generator=gen) * 0.05).half()
    hidden = _hidden(2, k=k)
    # the 5 padding rows would win by a mile if they were not masked
    weight[-5:] = (hidden[0] * 4).half()
    for rerank_k in (0, 8, 64):
        head = DraftNvfp4LMHead(_lm_head(weight, pad=5), rerank_k=rerank_k)
        tokens = head.top_tokens(hidden)
        assert (tokens < 256 - 5).all()
        unmasked = DraftNvfp4LMHead(_lm_head(weight, pad=0), rerank_k=rerank_k)
        assert int(unmasked.top_tokens(hidden)[0]) >= 256 - 5


def test_noncontiguous_hidden_is_accepted(cpu_ops):
    weight, head = _random_head(n=512, rerank_k=64)
    wide = _hidden(4, k=512)
    view = wide[:, ::2]
    assert not view.is_contiguous()
    assert torch.equal(head.top_tokens(view), head.top_tokens(view.contiguous()))


@pytest.mark.parametrize("bad", [0, 65])
def test_row_count_bounds(cpu_ops, bad):
    _, head = _random_head()
    with pytest.raises(ValueError):
        head.top_tokens(torch.zeros(bad, 256).half())


def test_dtype_and_width_are_checked(cpu_ops):
    _, head = _random_head()
    with pytest.raises(ValueError):
        head.top_tokens(torch.zeros(2, 256, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        head.top_tokens(torch.zeros(2, 128).half())


def test_resident_bytes_and_buffers(cpu_ops):
    weight, head = _random_head(n=512, k=256, rerank_k=64)
    codes, scales = 512 * 128, 512 * 16
    logits = 64 * 512 * 2  # FP16 [64, N]
    candidates = 64 * 64 * (2 + 8 + 4)  # topk values f16, ids i64, exact f32
    pair = 64 * 2 * 4
    assert head.resident_bytes() == codes + scales + logits + candidates + pair
    assert head.out.shape == (64, 512) and head.out.dtype == torch.float16
    assert head.weight.data_ptr() == weight.data_ptr()  # shares the FP16 shard
    head.rerank_k = 0
    assert head.resident_bytes() == codes + scales + logits + pair


def test_per_call_buffers_are_preallocated_and_stable(cpu_ops):
    _, head = _random_head(n=512, rerank_k=32)
    names = ("out", "_pair", "_topk_values", "_topk_ids", "_exact")
    before = {n: getattr(head, n).data_ptr() for n in names}
    for m in (1, 2, 4, 17, 64):
        head.top_tokens(_hidden(m))
    assert {n: getattr(head, n).data_ptr() for n in names} == before
    # contiguous [64, R] rows, so topk(out=) and the Triton kernel can use slices
    for n in ("_topk_values", "_topk_ids", "_exact"):
        assert getattr(head, n).shape == (64, 32) and getattr(head, n).is_contiguous()
    assert head._topk_ids.dtype == torch.int64 and head._exact.dtype == torch.float32
    assert head._topk_values.dtype == torch.float16


def test_result_does_not_alias_the_static_buffers(cpu_ops):
    _, head = _random_head(n=512, rerank_k=32)
    first = head.top_tokens(_hidden(4, seed=1))
    snapshot = first.clone()
    head.top_tokens(_hidden(4, seed=2))
    assert torch.equal(first, snapshot)


def test_rerank_k_setter_reallocates_and_validates(cpu_ops):
    weight, head = _random_head(n=512, rerank_k=64)
    hidden = _hidden(4)
    head.rerank_k = 8
    assert head.rerank_k == 8 and head._topk_ids.shape == (64, 8)
    tokens8 = head.top_tokens(hidden)
    head.rerank_k = 0
    assert head._topk_ids is None and head._exact is None
    raw = head.top_tokens(hidden)
    assert torch.equal(raw, head.out[:4].float().argmax(-1))
    head.rerank_k = 128
    assert head._exact.shape == (64, 128)
    assert head.top_tokens(hidden).shape == tokens8.shape
    for bad in (7, 129, -1):
        with pytest.raises(ValueError):
            head.rerank_k = bad
    assert head.rerank_k == 128


def test_activation_log_reports_cost_and_kernel_config(cpu_ops, caplog):
    with caplog.at_level("INFO"):
        _, head = _random_head(n=512, k=256, rerank_k=64)
    line = next(r.getMessage() for r in caplog.records if "active" in r.getMessage())
    assert caplog.text.count("draft NVFP4 lm_head active") == 1
    for fragment in (
        "rows=512",
        "K=256",
        "rerank_k=64",
        f"split_k={head.split_k}",
        f"chains={head.accumulator_chains}",
        "quantize=",
        "peak_extra=",
        "resident=",
    ):
        assert fragment in line, (fragment, line)
    assert head.quantize_seconds > 0
    assert head.peak_extra_bytes == 0  # no peak accounting off-CUDA


def test_exact_rerank_uses_indexed_fp32_logits_on_cuda(cpu_ops, monkeypatch):
    """The CUDA branch calls the Triton kernel with exactly its input contract."""
    import vllm.model_executor.layers.sm70_fp32_lm_head as kernel

    seen = []

    def fake_kernel(x, weight, ids, out):
        m, r = ids.shape
        assert x.dtype == torch.float16 and x.is_contiguous() and x.dim() == 2
        assert weight.dtype == torch.float16 and weight.is_contiguous()
        assert weight.shape[1] == x.shape[1]
        assert ids.dtype == torch.int64 and ids.is_contiguous()
        assert out.dtype == torch.float32 and out.is_contiguous()
        assert out.shape == (m, r) and x.shape[0] == m
        rows = weight[ids.reshape(-1)].view(m, r, -1).float()
        out.copy_(torch.einsum("mk,mrk->mr", x.float(), rows))
        seen.append((m, r))

    weight, head = _random_head(n=512, rerank_k=32)
    hidden = _hidden(5)
    expected = head.top_tokens(hidden)  # torch fallback
    monkeypatch.setattr(kernel, "indexed_fp32_logits", fake_kernel)
    monkeypatch.setattr(mod, "_use_triton_rerank", lambda x: True)
    assert torch.equal(head.top_tokens(hidden), expected)
    assert seen == [(5, 32)]


def test_constructor_validates_arguments(cpu_ops):
    weight = torch.randn(64, 256).half()
    with pytest.raises(ValueError):
        DraftNvfp4LMHead(_lm_head(weight), rerank_k=7)
    with pytest.raises(ValueError):
        DraftNvfp4LMHead(_lm_head(weight), rerank_k=8, split_k=64)
    with pytest.raises(ValueError):
        DraftNvfp4LMHead(_lm_head(weight), rerank_k=8, accumulator_chains=3)


# ---- TP reduction -------------------------------------------------------
def _fake_tp(monkeypatch, peers):
    """Make this process rank 0 of a 4-rank group; ``peers`` is [M, 3, 2]."""
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 4)

    def all_gather(pair, dim=-1):
        assert dim == -1 and pair.dtype == torch.float32 and pair.shape[-1] == 2
        full = torch.cat([pair.unsqueeze(1), peers], dim=1)  # [M, 4, 2]
        return full.reshape(pair.shape[0], 8)

    monkeypatch.setattr(mod, "tensor_model_parallel_all_gather", all_gather)


def test_tp_reduction_picks_the_global_winner(cpu_ops, monkeypatch):
    _, head = _random_head(n=512, rerank_k=64, start=0)
    hidden = _hidden(3)
    local_value, local_id = head.local_top_tokens(hidden)
    peers = torch.tensor(
        [
            # row 0: a peer wins outright
            [[1e9, 777.0], [-1.0, 5.0], [0.0, 6.0]],
            # row 1: this rank wins
            [[-1e9, 1.0], [-1e9, 2.0], [-1e9, 3.0]],
            # row 2: exact tie with rank 2 -> first rank (this one) wins
            [[-1e9, 1.0], [0.0, 2.0], [0.0, 3.0]],
        ]
    )
    peers[2, 1, 0] = local_value[2] - 1.0
    peers[2, 2, 0] = local_value[2]
    _fake_tp(monkeypatch, peers)
    tokens = head.top_tokens(hidden)
    assert tokens.dtype == torch.int64
    assert tokens[0] == 777
    assert tokens[1] == local_id[1]
    assert tokens[2] == local_id[2]


def test_tp_reduction_resolves_ties_to_the_lowest_rank(cpu_ops, monkeypatch):
    # Equal values on several ranks: the lowest rank, i.e. the smallest global
    # id, wins - the same answer as torch.max over the full vocabulary.
    _, head = _random_head(n=512, rerank_k=64, start=0)
    hidden = _hidden(2)
    local_value, _ = head.local_top_tokens(hidden)
    peers = torch.tensor(
        [
            # row 0: ranks 1 and 2 tie above this rank
            [[0.0, 600.0], [0.0, 700.0], [-1.0, 800.0]],
            # row 1: ranks 1, 2 and 3 tie with this rank
            [[0.0, 600.0], [0.0, 700.0], [0.0, 800.0]],
        ]
    )
    peers[0, :2, 0] = local_value[0] + 1.0
    peers[1, :, 0] = local_value[1]
    _fake_tp(monkeypatch, peers)
    tokens = head.top_tokens(hidden)
    assert tokens[0] == 600
    assert tokens[1] == head.local_top_tokens(hidden)[1][1]


def test_tp_size_one_is_a_short_circuit(cpu_ops, monkeypatch):
    _, head = _random_head()

    def boom(*args, **kwargs):
        raise AssertionError("no collective with tp_size == 1")

    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mod, "tensor_model_parallel_all_gather", boom)
    hidden = _hidden(2)
    assert torch.equal(head.top_tokens(hidden), head.local_top_tokens(hidden)[1])


def test_tp_shards_reconstruct_the_single_shard_answer(cpu_ops, monkeypatch):
    """The reduction returns the best of the four shard winners (global ids)."""
    k = 256
    gen = torch.Generator().manual_seed(21)
    full = (torch.randn(4 * 128, k, generator=gen) * 0.05).half()
    hidden = _hidden(4, k=k, seed=22)
    shards = [
        DraftNvfp4LMHead(
            _lm_head(full[r * 128 : (r + 1) * 128].contiguous(), start=r * 128),
            rerank_k=64,
        )
        for r in range(4)
    ]
    pairs = [s.local_top_tokens(hidden) for s in shards]
    stacked = torch.stack(
        [torch.stack([v.float(), i.float()], dim=-1) for v, i in pairs], dim=1
    )  # [M, 4, 2]
    _fake_tp(monkeypatch, stacked[:, 1:, :])
    tokens = shards[0].top_tokens(hidden)
    expected = stacked[:, :, 0].argmax(-1)
    assert torch.equal(tokens, stacked[torch.arange(4), expected, 1].long())
    # shard-local ids were offset by each shard's vocabulary start
    assert (tokens // 128 == expected).all()
    exact = (hidden.float() @ full.float().T).argmax(-1)
    assert (tokens == exact).sum() >= 3


def test_hot_path_has_no_host_sync():
    for fn in (DraftNvfp4LMHead.local_top_tokens, DraftNvfp4LMHead.top_tokens):
        source = inspect.getsource(fn)
        for banned in (
            r"\.item\(",
            r"\.tolist\(",
            r"\.cpu\(",
            r"\bbool\(",
            r"\.numpy\(",
            r"synchronize",
            r"nonzero",
            r"\.any\(",
        ):
            assert not re.search(banned, source), (fn.__name__, banned)


# --------------------------------------------------------------------------
# eligibility gate
# --------------------------------------------------------------------------
@pytest.fixture
def sm70_gate(monkeypatch):
    monkeypatch.setattr(mod, "_sm70_device_reason", lambda device: None)
    monkeypatch.setattr(mod, "_missing_ops", lambda: [])


def _meta_head(n=62080, k=2560, dtype=torch.float16, **kwargs):
    return _lm_head(torch.empty(n, k, dtype=dtype, device="meta"), **kwargs)


def _eligible(lm_head, config=MTP_GREEDY, device="cuda:0"):
    return DraftNvfp4LMHead.eligible(lm_head, config, torch.device(device))


def test_gate_accepts_the_production_shard(sm70_gate):
    assert _eligible(_meta_head(org_vocab_size=248320)) == (True, "ok")
    assert mod.default_split_config(2560, 62080) == (16, 2)


@pytest.mark.parametrize(
    "config, fragment",
    [
        (SimpleNamespace(method="eagle3", draft_sample_method="greedy"), "mtp/eagle"),
        (SimpleNamespace(method="ngram", draft_sample_method="greedy"), "mtp/eagle"),
        (SimpleNamespace(method="mtp", draft_sample_method="probabilistic"), "greedy"),
    ],
)
def test_gate_rejects_other_speculators(sm70_gate, config, fragment):
    ok, reason = _eligible(_meta_head(), config)
    assert not ok and fragment in reason


def test_gate_accepts_plain_eagle(sm70_gate):
    config = SimpleNamespace(method="eagle", draft_sample_method="greedy")
    assert _eligible(_meta_head(), config)[0]


def test_gate_device_and_ops(monkeypatch):
    ok, reason = _eligible(_meta_head(), device="cpu")
    assert not ok and "not CUDA" in reason
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    ok, reason = _eligible(_meta_head(), device="cuda:0")
    assert not ok and "(7, 0)" in reason
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    monkeypatch.setattr(mod, "_missing_ops", lambda: ["nvfp4_qpn2_gemm_sm70_out"])
    ok, reason = _eligible(_meta_head(), device="cuda:0")
    assert not ok and "missing native QPN2 operators" in reason


@pytest.mark.parametrize(
    "lm_head, fragment",
    [
        (_meta_head(dtype=torch.bfloat16), "float16"),
        (_meta_head(dtype=torch.uint8), "float16"),
        (_meta_head(k=2624), "multiple of 128"),  # 2624 % 128 == 64
        (_meta_head(k=2560 + 64), "multiple of 128"),
        (_meta_head(n=62081), "multiple of 32"),
        (_meta_head(n=96), "at least"),
        (_meta_head(pad=3), "padding"),
        (
            _lm_head(torch.empty(2560, 62080, dtype=torch.float16, device="meta").t()),
            "contiguous",
        ),
        (_meta_head(bias=torch.zeros(1)), "bias"),
        (_meta_head(org_vocab_size=1 << 24), "fp32"),
        (SimpleNamespace(weight=torch.empty(8, device="meta")), "2-D"),
        (SimpleNamespace(), "2-D"),
    ],
)
def test_gate_rejections(sm70_gate, lm_head, fragment):
    ok, reason = _eligible(lm_head)
    assert not ok and fragment in reason


def test_gate_rejects_split_k_that_does_not_divide(sm70_gate, monkeypatch):
    monkeypatch.setattr(mod, "default_split_config", lambda k, n: (32, 2))
    ok, reason = _eligible(_meta_head(k=2688))  # K/16 = 168, 168 % 32 != 0
    assert not ok and "split_k=32" in reason


# --------------------------------------------------------------------------
# builder (what EagleSpeculator.load_model calls)
# --------------------------------------------------------------------------
def _model(weight, scale=1.0):
    return SimpleNamespace(
        lm_head=_lm_head(weight), logits_processor=SimpleNamespace(scale=scale)
    )


def _build(model, dtype=torch.float16):
    return build_draft_nvfp4_lm_head(
        model, MTP_GREEDY, torch.device("cpu"), dtype, rerank_k=64
    )


def test_builder_returns_a_head_when_everything_is_green(
    cpu_ops, sm70_gate, monkeypatch
):
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    weight = (torch.randn(512, 256) * 0.05).half()
    # shape gate: K=256 is a multiple of 128 and N=512 of 32
    head = _build(_model(weight))
    assert isinstance(head, DraftNvfp4LMHead) and head.rerank_k == 64


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda m: setattr(m, "lm_head", None), "no lm_head"),
        (lambda m: setattr(m.logits_processor, "scale", -1.0), "not positive"),
        (
            lambda m: setattr(m.lm_head, "weight", m.lm_head.weight.bfloat16()),
            "float16",
        ),
    ],
)
def test_builder_falls_back_with_a_reason(
    cpu_ops, sm70_gate, monkeypatch, caplog, mutate, fragment
):
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    model = _model((torch.randn(512, 256) * 0.05).half())
    mutate(model)
    with caplog.at_level("INFO"):
        assert _build(model) is None
    assert (
        any(
            "disabled" in r.getMessage() and fragment in r.getMessage()
            for r in caplog.records
        )
        or fragment in caplog.text
    )


def test_builder_rejects_non_fp16_model_dtype(cpu_ops, sm70_gate, monkeypatch):
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    assert _build(_model((torch.randn(512, 256) * 0.05).half()), torch.bfloat16) is None


def test_builder_swallows_construction_errors(sm70_gate, monkeypatch):
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)

    def boom(packed, scales):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(mod, "_prepare_qpn2", boom)
    assert _build(_model((torch.randn(512, 256) * 0.05).half())) is None


# ---- load-time TP consensus on the CPU group ------------------------------
def _fake_cpu_consensus(monkeypatch, tp_size, rank, peers):
    """Fake a TP group; ``peers`` maps the other ranks to their ``_STATE_*``.

    The all-reduce stand-in checks that the collective only ever sees a CPU
    int32 vector and that the device group is never used.
    """
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: rank)
    calls = []

    def all_reduce(table):
        assert table.device.type == "cpu" and table.dtype == torch.int32
        calls.append(table.tolist())
        full = table.clone()
        for peer, state in peers.items():
            full[peer] = state
        return full

    def no_device_collective(*args, **kwargs):
        raise AssertionError("the consensus must not use the device group")

    monkeypatch.setattr(mod, "_tp_cpu_all_reduce", all_reduce)
    monkeypatch.setattr(mod, "tensor_model_parallel_all_gather", no_device_collective)
    return calls


OK, FALLBACK, FATAL = mod._STATE_OK, mod._STATE_FALLBACK, mod._STATE_FATAL


def test_builder_consensus_all_ranks_ok(cpu_ops, sm70_gate, monkeypatch):
    calls = _fake_cpu_consensus(monkeypatch, 4, 1, {0: OK, 2: OK, 3: OK})
    head = _build(_model((torch.randn(512, 256) * 0.05).half()))
    assert isinstance(head, DraftNvfp4LMHead)
    assert calls == [[0, OK, 0, 0]]  # this rank only fills its own slot


def test_builder_consensus_one_recoverable_peer_disables_everywhere(
    cpu_ops, sm70_gate, monkeypatch, caplog
):
    calls = _fake_cpu_consensus(monkeypatch, 4, 0, {1: OK, 2: FALLBACK, 3: OK})
    with caplog.at_level("INFO"):
        head = _build(_model((torch.randn(512, 256) * 0.05).half()))
    assert head is None
    assert calls == [[OK, 0, 0, 0]]
    assert "TP rank 2 could not build the head" in caplog.text
    assert caplog.text.count("draft NVFP4 lm_head disabled") == 1


def test_builder_consensus_local_recoverable_failure_still_joins(
    sm70_gate, monkeypatch, caplog
):
    # An ineligible rank must still take part in the all-reduce, otherwise the
    # peers that did build the head would hang.
    calls = _fake_cpu_consensus(monkeypatch, 4, 3, {0: OK, 1: OK, 2: OK})
    with caplog.at_level("INFO"):
        assert _build(SimpleNamespace(lm_head=None)) is None
    assert calls == [[0, 0, 0, FALLBACK]]
    assert "no lm_head" in caplog.text


@pytest.mark.parametrize(
    "error",
    [ValueError("bad shard"), RuntimeError("out of memory"), KeyError("op")],
)
def test_builder_construction_error_is_recoverable(
    cpu_ops, sm70_gate, monkeypatch, error
):
    calls = _fake_cpu_consensus(monkeypatch, 2, 0, {1: OK})

    def boom(packed, scales):
        raise error

    monkeypatch.setattr(mod, "_prepare_qpn2", boom)
    assert _build(_model((torch.randn(512, 256) * 0.05).half())) is None
    assert calls == [[FALLBACK, 0]]


def test_builder_consensus_fatal_peer_aborts_every_rank(
    cpu_ops, sm70_gate, monkeypatch
):
    calls = _fake_cpu_consensus(monkeypatch, 4, 0, {1: OK, 2: FATAL, 3: FALLBACK})
    with pytest.raises(
        RuntimeError,
        match=r"draft NVFP4 lm_head: fatal CUDA error on rank 2; aborting startup",
    ) as info:
        _build(_model((torch.randn(512, 256) * 0.05).half()))
    assert info.value.__cause__ is None  # the peer's exception is not ours
    assert calls == [[OK, 0, 0, 0]]


@pytest.mark.parametrize(
    "error",
    [
        torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 89 MiB"),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        RuntimeError("device-side assert triggered"),
        RuntimeError("unspecified launch failure"),
        RuntimeError("an illegal instruction was encountered"),
    ],
)
def test_builder_local_fatal_error_aborts_with_the_cause(
    cpu_ops, sm70_gate, monkeypatch, error
):
    # The failed rank reports FATAL (a CPU vector, no device work) and raises.
    calls = _fake_cpu_consensus(monkeypatch, 4, 3, {0: OK, 1: OK, 2: OK})

    def boom(packed, scales):
        raise error

    monkeypatch.setattr(mod, "_prepare_qpn2", boom)
    with pytest.raises(RuntimeError, match=r"fatal CUDA error on rank 3") as info:
        _build(_model((torch.randn(512, 256) * 0.05).half()))
    assert info.value.__cause__ is error
    assert calls == [[0, 0, 0, FATAL]]


def test_builder_tp_one_has_no_collective(cpu_ops, sm70_gate, monkeypatch):
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)

    def boom(*args, **kwargs):
        raise AssertionError("no collective with tp_size == 1")

    monkeypatch.setattr(mod, "_tp_cpu_all_reduce", boom)
    assert isinstance(
        _build(_model((torch.randn(512, 256) * 0.05).half())), DraftNvfp4LMHead
    )
    assert _build(SimpleNamespace(lm_head=None)) is None

    def fatal(packed, scales):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(mod, "_prepare_qpn2", fatal)
    with pytest.raises(RuntimeError, match=r"fatal CUDA error on rank 0"):
        _build(_model((torch.randn(512, 256) * 0.05).half()))


@pytest.mark.parametrize("default_device", ["meta"])
def test_consensus_tensors_are_pinned_to_cpu(
    sm70_gate, monkeypatch, caplog, default_device
):
    # An outer default-device context (meta here, cuda in a worker) must not
    # move the consensus vector off the CPU: the collective runs on the gloo
    # group and a failed rank must not touch the GPU.
    calls = _fake_cpu_consensus(monkeypatch, 4, 3, {0: OK, 1: FALLBACK, 2: OK})
    with torch.device(default_device):
        assert mod._tp_consensus(OK) == [OK, FALLBACK, OK, OK]
        with caplog.at_level("INFO"):
            # ineligible rank -> same decision table as without the context
            assert _build(SimpleNamespace(lm_head=None)) is None
    assert calls == [[0, 0, 0, OK], [0, 0, 0, FALLBACK]]
    assert "no lm_head" in caplog.text

    _fake_cpu_consensus(monkeypatch, 4, 0, {1: OK, 2: FATAL, 3: OK})
    with (
        torch.device(default_device),
        pytest.raises(RuntimeError, match=r"fatal CUDA error on rank 2"),
    ):
        _build(SimpleNamespace(lm_head=None))


def test_consensus_uses_the_tp_cpu_group(monkeypatch):
    group = object()
    monkeypatch.setattr(mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=group))
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 2)
    seen = {}

    def all_reduce(tensor, op=None, group=None):
        seen.update(op=op, group=group, tensor=tensor.clone())
        tensor += torch.tensor([OK, FALLBACK, 0], dtype=torch.int32)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert mod._tp_consensus(OK) == [OK, FALLBACK, OK]
    assert seen["group"] is group
    assert seen["op"] == torch.distributed.ReduceOp.SUM
    assert seen["tensor"].device.type == "cpu"
    assert seen["tensor"].tolist() == [0, 0, OK]


def test_fatal_cuda_error_classification():
    fatal = [
        torch.cuda.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        RuntimeError("CUDA error: unspecified launch failure"),
        RuntimeError("CUDA error: device-side assert triggered"),
    ]
    recoverable = [
        ValueError("CUDA error in the message but not a RuntimeError"),
        RuntimeError("out of memory"),
        RuntimeError("Triton compile failure: ptxas fatal"),
        RuntimeError("nvfp4_qpn2_prepare_sm70 is not registered"),
        AttributeError("_C has no op"),
    ]
    assert all(mod.is_fatal_cuda_error(e) for e in fatal)
    assert not any(mod.is_fatal_cuda_error(e) for e in recoverable)


# --------------------------------------------------------------------------
# env registration and speculator routing
# --------------------------------------------------------------------------
def test_env_defaults_and_validation(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD", raising=False)
    monkeypatch.delenv("VLLM_SM70_MTP_DRAFT_NVFP4_RERANK_K", raising=False)
    assert envs.VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD is False
    assert envs.VLLM_SM70_MTP_DRAFT_NVFP4_RERANK_K == 64
    monkeypatch.setenv("VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD", "1")
    assert envs.VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD is True
    for choice in (0, 8, 16, 32, 64, 128):
        monkeypatch.setenv("VLLM_SM70_MTP_DRAFT_NVFP4_RERANK_K", str(choice))
        assert choice == envs.VLLM_SM70_MTP_DRAFT_NVFP4_RERANK_K
        assert choice in mod.RERANK_K_CHOICES
    monkeypatch.setenv("VLLM_SM70_MTP_DRAFT_NVFP4_RERANK_K", "7")
    with pytest.raises(ValueError):
        _ = envs.VLLM_SM70_MTP_DRAFT_NVFP4_RERANK_K


def _speculator_stub(head):
    sentinel = {
        "local": torch.tensor([1]),
        "full": torch.tensor([2]),
        "argmax": torch.tensor([3]),
    }
    model = SimpleNamespace(
        get_top_tokens=lambda h: sentinel["local"],
        compute_logits=lambda h: torch.tensor([[0.0, 1.0, 0.0]]),
    )
    return sentinel, SimpleNamespace(
        _draft_nvfp4_head=head,
        _draft_hidden_dump=None,
        use_local_argmax_reduction=False,
        model=model,
        temperature=None,
        seeds=None,
        use_fp64_gumbel=False,
    )


def test_speculator_routes_through_the_draft_head(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as spec

    taken = []
    head = SimpleNamespace(
        top_tokens=lambda h: taken.append(h.shape[0]) or torch.tensor([42])
    )
    _, stub = _speculator_stub(head)
    sample = spec.EagleSpeculator._sample_draft

    hidden = torch.zeros(4, 8)
    assert sample(stub, hidden, None, None, None, None).tolist() == [42]
    assert sample(stub, torch.zeros(64, 8), None, None, None, None).tolist() == [42]
    assert taken == [4, 64]

    # an empty batch never reaches the kernel (M must be in [1, 64])
    assert sample(stub, torch.zeros(0, 8), None, None, None, None).tolist() == [1]
    assert taken == [4, 64]
    # more than 64 rows -> existing path (argmax of compute_logits)
    assert sample(stub, torch.zeros(65, 8), None, None, None, None).tolist() == [1]
    # ... or the local-argmax reduction when configured
    stub.use_local_argmax_reduction = True
    assert sample(stub, torch.zeros(65, 8), None, None, None, None).tolist() == [1]
    # probabilistic drafts (draft_logits is not None) never use the head
    monkeypatch.setattr(spec, "gumbel_sample", lambda *a, **k: torch.tensor([7]))
    assert sample(
        stub, hidden, None, torch.zeros(1), None, torch.zeros(1)
    ).tolist() == [7]
    assert taken == [4, 64]
    # no head -> untouched behaviour
    stub._draft_nvfp4_head = None
    stub.use_local_argmax_reduction = False
    assert sample(stub, hidden, None, None, None, None).tolist() == [1]


def test_speculator_warns_once_when_a_batch_exceeds_the_kernel_rows(caplog):
    from vllm.logger import _print_warning_once
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as spec

    _print_warning_once.cache_clear()
    taken = []
    head = SimpleNamespace(
        top_tokens=lambda h: taken.append(h.shape[0]) or torch.tensor([42])
    )
    _, stub = _speculator_stub(head)
    sample = spec.EagleSpeculator._sample_draft
    marker = "more than 64 rows"

    with caplog.at_level("WARNING"):
        sample(stub, torch.zeros(64, 8), None, None, None, None)
        sample(stub, torch.zeros(0, 8), None, None, None, None)
        assert marker not in caplog.text  # in range / empty: silent
        for rows in (65, 80, 65, 300):  # new batch sizes do not log again
            assert sample(
                stub, torch.zeros(rows, 8), None, None, None, None
            ).tolist() == [1]
    assert caplog.text.count(marker) == 1
    assert "FP16 lm_head path" in caplog.text
    assert taken == [64]

    # no head, or a probabilistic draft: nothing to warn about
    caplog.clear()
    _print_warning_once.cache_clear()
    stub._draft_nvfp4_head = None
    with caplog.at_level("WARNING"):
        sample(stub, torch.zeros(65, 8), None, None, None, None)
    assert marker not in caplog.text


def test_rerank_zero_logs_the_diagnostic_mode_once(cpu_ops, caplog):
    from vllm.logger import _print_warning_once

    _print_warning_once.cache_clear()
    with caplog.at_level("WARNING"):
        _random_head(rerank_k=0)
        _random_head(rerank_k=0, seed=1)
        _random_head(rerank_k=64)
    assert caplog.text.count("rerank_k=0 is active") == 1
    assert "diagnostic" in caplog.text
    caplog.clear()
    _print_warning_once.cache_clear()
    with caplog.at_level("WARNING"):
        _random_head(rerank_k=64)
    assert "rerank_k=0" not in caplog.text


# --------------------------------------------------------------------------
# real hidden-state dump (VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR)
# --------------------------------------------------------------------------
def _dumper(tmp_path, steps=3):
    from vllm.v1.worker.gpu.spec_decode.eagle.draft_hidden_dump import DraftHiddenDumper

    return DraftHiddenDumper(str(tmp_path), steps)


def test_dumper_saves_the_first_n_nonzero_batches(tmp_path):
    dumper = _dumper(tmp_path, steps=3)
    batches = [torch.zeros(2, 8).half()] + [
        _hidden(m, k=8, seed=m) for m in (1, 4, 2, 3)
    ]
    for i, batch in enumerate(batches):
        dumper.maybe_dump(batch, torch.tensor(i % 4))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f"rank0_step{n:05d}.pt" for n in range(3)]  # no .tmp left
    assert dumper.saved == 3 and dumper.skipped_zero == 1
    first = torch.load(tmp_path / files[0])
    assert torch.equal(first["hidden"], batches[1])
    assert (
        first["hidden"].dtype == torch.float16 and first["hidden"].device.type == "cpu"
    )
    assert (first["n"], first["rank"], first["rows"]) == (0, 0, 1)
    assert first["current_draft_step"] == 1
    assert torch.equal(torch.load(tmp_path / files[2])["hidden"], batches[3])


def test_dumper_names_files_by_tp_rank(tmp_path, monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle import draft_hidden_dump as dump

    monkeypatch.setattr(dump, "_tp_rank", lambda: 3)
    _dumper(tmp_path).maybe_dump(_hidden(2, k=8), 2)
    assert [p.name for p in tmp_path.iterdir()] == ["rank3_step00000.pt"]


def test_dumper_does_nothing_during_graph_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    dumper = _dumper(tmp_path)
    dumper.maybe_dump(_hidden(2, k=8), 0)
    assert dumper.saved == 0 and list(tmp_path.iterdir()) == []


def test_dumper_env_gating(tmp_path, monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle.draft_hidden_dump import (
        maybe_create_draft_hidden_dumper,
    )

    envs.disable_envs_cache()
    for name in (
        "VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR",
        "VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_STEPS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR is None
    assert envs.VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_STEPS == 256
    assert maybe_create_draft_hidden_dumper() is None
    target = tmp_path / "dump"
    monkeypatch.setenv("VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR", str(target))
    monkeypatch.setenv("VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_STEPS", "5")
    dumper = maybe_create_draft_hidden_dumper()
    assert dumper is not None and dumper.max_batches == 5 and target.is_dir()
    monkeypatch.setenv("VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_STEPS", "0")
    assert maybe_create_draft_hidden_dumper() is None


def test_speculator_calls_the_dumper_first_and_only_when_set():
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as spec

    order = []
    dumper = SimpleNamespace(
        maybe_dump=lambda h, step: order.append(("dump", tuple(h.shape), step))
    )
    head = SimpleNamespace(
        top_tokens=lambda h: order.append("head") or torch.tensor([9])
    )
    _, stub = _speculator_stub(head)
    stub._draft_hidden_dump = dumper
    hidden, step = torch.zeros(3, 8), torch.tensor(2)
    assert spec.EagleSpeculator._sample_draft(
        stub, hidden, None, None, step, None
    ).tolist() == [9]
    assert order == [("dump", (3, 8), step), "head"]
    # the dump also sees batches the NVFP4 head does not serve (> 64 rows)
    order.clear()
    spec.EagleSpeculator._sample_draft(stub, torch.zeros(65, 8), None, None, step, None)
    assert order == [("dump", (65, 8), step)]
    # unset -> no dumper is consulted
    stub._draft_hidden_dump = None
    order.clear()
    spec.EagleSpeculator._sample_draft(stub, hidden, None, None, step, None)
    assert order == ["head"]
