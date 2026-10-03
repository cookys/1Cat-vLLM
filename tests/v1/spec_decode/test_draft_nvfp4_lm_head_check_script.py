# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the pure-python parts of benchmarks/sm70_draft_nvfp4_lm_head_check.py.

The QPN2 GEMM is replaced by a dequantized CPU matmul; the loaders, the global
TP-style reduction and the disagreement bookkeeping are the script's own code.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.eagle import draft_nvfp4_lm_head as dh

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "sm70_draft_nvfp4_lm_head_check.py"
)
K = 2560


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("sm70_draft_head_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_ops(monkeypatch):
    def gemm(out, x, codes, scales, global_scale, split_k, chains):
        weight = dh.unpack_nvfp4_packed(codes, scales, global_scale, kernel_exact=True)
        out.copy_((x.float() @ weight.T).half())

    monkeypatch.setattr(dh, "_prepare_qpn2", lambda p, s: (p.clone(), s.clone()))
    monkeypatch.setattr(dh, "_qpn2_gemm_out", gemm)
    monkeypatch.setattr(dh, "get_tensor_model_parallel_world_size", lambda: 1)


def _shards(num_shards, rows=256, seed=0, pad=0, edit=None):
    gen = torch.Generator().manual_seed(seed)
    weights, heads = [], []
    for shard in range(num_shards):
        weight = (torch.randn(rows, K, generator=gen) * 0.02).half()
        if edit is not None:
            edit(weight)
        lm_head = SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=shard * rows, num_org_vocab_padding=pad
            ),
        )
        weights.append(weight)
        heads.append(dh.DraftNvfp4LMHead(lm_head, rerank_k=64))
    return weights, heads


def _hidden(rows, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, K, generator=gen).half()


def test_load_hidden_dump_accepts_directory_file_and_plain_tensor(script, tmp_path):
    a, b, other_rank = _hidden(3, 1), _hidden(5, 2), _hidden(2, 3)
    torch.save({"hidden": a, "n": 0}, tmp_path / "rank0_step00000.pt")
    torch.save({"hidden": b, "n": 1}, tmp_path / "rank0_step00001.pt")
    torch.save({"hidden": other_rank}, tmp_path / "rank1_step00000.pt")
    hidden, files = script.load_hidden_dump(tmp_path, "rank0_step*.pt")
    assert files == 2 and torch.equal(hidden, torch.cat([a, b]))
    hidden, files = script.load_hidden_dump(tmp_path, "rank*_step*.pt")
    assert files == 3 and hidden.shape[0] == 10
    hidden, files = script.load_hidden_dump(tmp_path / "rank0_step00001.pt", "*")
    assert files == 1 and torch.equal(hidden, b)
    torch.save(a, tmp_path / "plain.pt")
    hidden, _ = script.load_hidden_dump(tmp_path / "plain.pt", "*")
    assert torch.equal(hidden, a)
    with pytest.raises(SystemExit):
        script.load_hidden_dump(tmp_path, "nothing*.pt")


def test_load_hidden_dump_round_trips_the_real_dumper(script, tmp_path):
    from vllm.v1.worker.gpu.spec_decode.eagle.draft_hidden_dump import DraftHiddenDumper

    dumper = DraftHiddenDumper(str(tmp_path), 4)
    batches = [_hidden(m, seed=m) for m in (1, 2, 4)]
    for i, batch in enumerate(batches):
        dumper.maybe_dump(batch, i)
    hidden, files = script.load_hidden_dump(tmp_path, "rank0_step*.pt")
    assert files == 3 and torch.equal(hidden, torch.cat(batches))


def test_global_disagreement_with_the_tp_style_reduction(script, cpu_ops):
    weights, heads = _shards(2)
    hidden = _hidden(130)
    ranks = [0, 8, 64, 128]
    result = script.evaluate(heads, weights, hidden, ranks)
    assert result["scope"] == "global" and result["shards"] == 2
    assert result["rows"] == 130
    rate = result["disagreement_rate"]
    rows = result["disagreement_rows"]
    assert set(rate) == {"0", "8", "64", "128"}
    assert all(rows[r] == pytest.approx(rate[r] * 130) for r in rate)
    # the raw 4-bit argmax disagrees with FP16 on a visible share of rows ...
    assert rate["0"] > 0.02
    # ... and the exact rerank over enough candidates removes all of it.
    assert rate["128"] == 0.0 and rate["64"] <= rate["8"] <= rate["0"]
    assert sum(result["rank_hist"].values()) == 130
    assert 0.0 < result["logits_rel_l2"] < 0.2

    # independent global argmax: FP16 over the concatenated vocabulary
    full = torch.cat([hidden @ w.t() for w in weights], dim=1)
    truth = full.argmax(-1)
    for head in heads:
        head.rerank_k = 128
    values = torch.stack([head.local_top_tokens(hidden[:64])[0] for head in heads])
    ids = torch.stack([head.local_top_tokens(hidden[:64])[1] for head in heads])
    token = ids.gather(0, values.argmax(0)[None])[0]
    assert torch.equal(token, truth[:64])


def test_single_shard_scope_and_rank_histogram(script, cpu_ops):
    weights, heads = _shards(1, rows=512)
    hidden = _hidden(70, seed=4)
    result = script.evaluate(heads, weights, hidden, [0, 64])
    assert result["scope"] == "shard" and result["shards"] == 1
    # rank 1 <=> the FP16 argmax is also the raw NVFP4 argmax (up to f16 ties)
    wrong_raw = result["disagreement_rows"]["0"]
    assert abs(result["rank_hist"]["1"] - (70 - wrong_raw)) <= 2
    assert result["disagreement_rows"]["64"] <= wrong_raw
    assert result["fp16_top1_top2_gap_p50"] >= result["fp16_top1_top2_gap_p10"] >= 0


def test_tp_rank_picks_the_default_dump_files(script, tmp_path):
    torch.save({"hidden": _hidden(4, 1)}, tmp_path / "rank0_step00000.pt")
    torch.save({"hidden": _hidden(5, 2)}, tmp_path / "rank2_step00000.pt")
    args = SimpleNamespace(
        hidden_dump=tmp_path,
        hidden_glob=None,
        tp_rank="2",
        max_hidden_rows=0,
        num_hidden=0,
        sigma=1.0,
        seed=0,
    )
    hidden, source, files = script.make_hidden(args, torch.device("cpu"))
    assert hidden.shape[0] == 5 and files == 1
    args.tp_rank = "all"
    hidden, _, _ = script.make_hidden(args, torch.device("cpu"))
    assert hidden.shape[0] == 4
    args.tp_rank, args.hidden_glob = "2", "rank*_step*.pt"
    hidden, _, files = script.make_hidden(args, torch.device("cpu"))
    assert hidden.shape[0] == 9 and files == 2
    args.hidden_glob, args.tp_rank = None, "1"
    with pytest.raises(SystemExit, match="hidden-glob"):
        script.make_hidden(args, torch.device("cpu"))


def test_f16_ulp_matches_numpy_nextafter(script):
    np = pytest.importorskip("numpy")
    values = np.array(
        [0.0, 1e-7, 3e-5, 6.2e-5, 0.1, 0.5, 1.0, 1.5, 1.9990234, 2.0, 7.3, 31.9, 100.0],
        dtype=np.float16,
    )
    expected = np.nextafter(values, np.float16(np.inf)).astype(np.float64) - values
    got = script.f16_ulp(torch.from_numpy(values.astype(np.float32)))
    assert np.allclose(got.double().numpy(), expected, rtol=0, atol=0)
    assert script.f16_ulp(torch.tensor(-2.0)) == script.f16_ulp(torch.tensor(2.0))


def test_near_tie_rate_separates_fp16_ties_from_4bit_effects(script, cpu_ops):
    def tie_rows(weight):
        weight[6] = weight[5]  # identical rows -> identical FP16 logits

    weights, heads = _shards(1, rows=256, edit=tie_rows)
    hidden = _hidden(40, seed=6)
    hidden[:4] = (weights[0][5].float() * 40).half()  # rows 5 and 6 tie on top
    result = script.evaluate(heads, weights, hidden, [0, 64])
    assert result["fp16_near_tie_rows"] >= 4
    assert result["fp16_near_tie_rate"] == pytest.approx(
        result["fp16_near_tie_rows"] / 40
    )
    assert result["fp16_exact_top1_ties"] >= 4
    for r in ("0", "64"):
        assert (
            result["disagreement_rows_not_near_tie"][r]
            <= result["disagreement_rows"][r]
        )
    rate = result["disagreement_rate_not_near_tie"]["64"]
    assert rate == result["disagreement_rows_not_near_tie"]["64"] / (
        40 - result["fp16_near_tie_rows"]
    )


def test_padding_rows_are_masked_in_both_paths(script, cpu_ops):
    def plant(weight):
        weight[-3:] = 5 * weight[0]  # would win everywhere if not masked

    weights, heads = _shards(1, rows=256, pad=3, edit=plant)
    hidden = _hidden(20, seed=8)
    hidden[:] = (weights[0][0].float() * 30).half()
    result = script.evaluate(heads, weights, hidden, [0, 64])
    assert result["disagreement_rate"]["64"] == 0.0
    assert 0.0 < result["logits_rel_l2"] < 0.2  # padding excluded, no inf/nan
