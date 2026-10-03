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

    dumper = DraftHiddenDumper(
        str(tmp_path),
        4,
        max_rows=4,
        hidden_size=K,
        dtype=torch.float16,
        device="cpu",
    )
    tokens = torch.arange(16, dtype=torch.int64).view(4, 4) + 500
    batches = [_hidden(m, seed=m) for m in (1, 2, 4)]
    for step, batch in enumerate(batches):
        dumper.stage(batch)
        dumper.dump_step(step, batch.shape[0], tokens)
    hidden, files = script.load_hidden_dump(tmp_path, "rank0_step*.pt")
    assert files == 3 and torch.equal(hidden, torch.cat(batches))
    hidden, loaded_tokens, files = script.load_dump(tmp_path, "rank0_step*.pt")
    assert files == 3 and torch.equal(hidden, torch.cat(batches))
    expected = torch.cat([tokens[:m, step] for step, m in enumerate((1, 2, 4))])
    assert torch.equal(loaded_tokens, expected)


def test_load_dump_returns_tokens_only_when_every_file_has_them(
    script, tmp_path, capsys
):
    torch.save(
        {"hidden": _hidden(2, 1), "draft_tokens": torch.tensor([7, 8])},
        tmp_path / "rank0_step00000.pt",
    )
    _, tokens, _ = script.load_dump(tmp_path, "rank0_step*.pt")
    assert tokens.tolist() == [7, 8]
    torch.save(
        {"hidden": _hidden(3, 2)}, tmp_path / "rank0_step00001.pt"
    )  # an old dump
    _, tokens, files = script.load_dump(tmp_path, "rank0_step*.pt")
    assert tokens is None and files == 2
    assert "no draft_tokens" in capsys.readouterr().err
    with pytest.raises(script.NoDumpFiles):
        script.load_dump(tmp_path, "nothing*.pt")


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


def _args(tmp_path, **overrides):
    values = {
        "hidden_dump": tmp_path,
        "hidden_glob": None,
        "tp_rank": "2",
        "max_hidden_rows": 0,
        "num_hidden": 16,
        "sigma": 1.0,
        "seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tp_rank_picks_the_default_dump_files(script, tmp_path):
    torch.save({"hidden": _hidden(4, 1)}, tmp_path / "rank0_step00000.pt")
    torch.save({"hidden": _hidden(5, 2)}, tmp_path / "rank2_step00000.pt")
    args = _args(tmp_path)
    rows = script.make_hidden(args, torch.device("cpu"))
    assert rows.hidden.shape[0] == 5 and rows.files == 1 and not rows.synthetic
    assert rows.draft_tokens is None and rows.source.startswith("dump:")
    args.tp_rank = "all"
    assert script.make_hidden(args, torch.device("cpu")).hidden.shape[0] == 4
    args.tp_rank, args.hidden_glob = "2", "rank*_step*.pt"
    rows = script.make_hidden(args, torch.device("cpu"))
    assert rows.hidden.shape[0] == 9 and rows.files == 2


def test_hidden_dump_without_files_falls_back_to_synthetic(script, tmp_path, capsys):
    # --hidden-dump given but nothing matches: no SystemExit, a loud warning
    rows = script.make_hidden(_args(tmp_path, tp_rank="1"), torch.device("cpu"))
    assert rows.synthetic and rows.source == "synthetic" and rows.files == 0
    assert rows.hidden.shape == (16, K) and rows.hidden.dtype == torch.float16
    assert rows.draft_tokens is None
    assert "hidden-glob" in rows.note
    assert "WARNING" in capsys.readouterr().err
    # a missing file path behaves the same way
    rows = script.make_hidden(_args(tmp_path / "missing.pt"), torch.device("cpu"))
    assert rows.synthetic
    # without --hidden-dump nothing changes: the labelled random rows
    rows = script.make_hidden(_args(None), torch.device("cpu"))
    assert not rows.synthetic and rows.source == "random_normal_sigma=1.0"


def test_synthetic_rows_report_no_accuracy_and_a_null_disagreement(
    script, cpu_ops, tmp_path, capsys
):
    weights, heads = _shards(1, rows=256)
    rows = script.make_hidden(_args(tmp_path), torch.device("cpu"))
    assert script.consistency_section(heads, weights, rows) == {
        "dump_consistency": None,
        "dump_consistency_detail": None,
        "dump_consistency_ok": None,
    }
    section = script.accuracy_section(heads, weights, rows, [0, 64])
    assert section == {"accuracy": None, "disagreement_rate": None}
    assert "not measured" in capsys.readouterr().err
    # the same rows through the real-dump path would have produced numbers
    real = rows._replace(synthetic=False)
    section = script.accuracy_section(heads, weights, real, [0, 64])
    assert set(section["disagreement_rate"]) == {"0", "64"}
    assert section["disagreement_rate"] == section["accuracy"]["disagreement_rate"]
    assert all(head.rerank_k == 64 for head in heads)  # left as production


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


def _fp16_tokens(weights, hidden):
    """Independent global FP16 argmax over the concatenated shards."""
    return torch.cat([hidden @ w.t() for w in weights], dim=1).argmax(-1)


def test_dump_consistency_is_one_when_the_dump_matches_the_fp16_head(script, cpu_ops):
    weights, heads = _shards(2)  # shard r owns ids [r * 256, (r + 1) * 256)
    hidden = _hidden(70, seed=9)
    detail = script.dump_consistency(
        heads, weights, hidden, _fp16_tokens(weights, hidden)
    )
    assert detail["scope"] == "global" and detail["value"] == 1.0
    assert (detail["rows_total"], detail["rows_checked"], detail["rows_matching"]) == (
        70,
        70,
        70,
    )
    assert detail["mismatch_near_tie_rows"] == 0


def test_dump_consistency_counts_the_rows_that_differ(script, cpu_ops, capsys):
    weights, heads = _shards(2)
    hidden = _hidden(40, seed=10)
    tokens = _fp16_tokens(weights, hidden)
    wrong = tokens.clone()
    wrong[:4] = (wrong[:4] + 1) % 512  # an NVFP4-style (or wrong-tensor) token
    detail = script.dump_consistency(heads, weights, hidden, wrong)
    assert detail["rows_matching"] == 36 and detail["value"] == pytest.approx(0.9)
    assert script.report_dump_consistency(detail) is False
    captured = capsys.readouterr()
    assert captured.out.startswith("dump_consistency=0.9000")
    assert "WARNING" in captured.err and "NVFP4_LM_HEAD" in captured.err
    ok = script.dump_consistency(heads, weights, hidden, tokens)
    assert script.report_dump_consistency(ok) is True
    assert capsys.readouterr().err == ""


def test_dump_consistency_with_one_shard_checks_only_its_own_rows(script, cpu_ops):
    weights, heads = _shards(2)
    hidden = _hidden(60, seed=11)
    tokens = _fp16_tokens(weights, hidden)  # global winners over both shards
    in_shard0 = tokens < 256
    assert 0 < int(in_shard0.sum()) < 60
    detail = script.dump_consistency(heads[:1], weights[:1], hidden, tokens)
    assert detail["scope"] == "shard"
    assert detail["rows_checked"] == int(in_shard0.sum())
    assert detail["value"] == 1.0
    # shard 1 has start 256: its rows are the other ones
    detail = script.dump_consistency(heads[1:], weights[1:], hidden, tokens)
    assert (
        detail["rows_checked"] == 60 - int(in_shard0.sum()) and detail["value"] == 1.0
    )
    # nothing dumped lies in the loaded shard -> not measurable, never a warning
    outside = torch.full((60,), 400, dtype=torch.int64)
    detail = script.dump_consistency(heads[:1], weights[:1], hidden, outside)
    assert detail["value"] is None and detail["rows_checked"] == 0
    assert script.report_dump_consistency(detail) is True


def test_dump_consistency_separates_fp16_near_ties(script, cpu_ops):
    def tie_rows(weight):
        weight[6] = weight[5]  # identical rows -> identical FP16 logits

    weights, heads = _shards(1, rows=256, edit=tie_rows)
    hidden = _hidden(20, seed=12)
    hidden[:4] = (weights[0][5].float() * 40).half()  # rows 5 and 6 tie on top
    tokens = _fp16_tokens(weights, hidden)
    assert (tokens[:4] == 5).all()  # the first maximum
    tokens[:4] = 6  # the server's kernel broke the tie the other way
    detail = script.dump_consistency(heads, weights, hidden, tokens)
    assert detail["rows_matching"] == 16 and detail["mismatch_near_tie_rows"] == 4


def test_consistency_section_prints_first_and_flags_a_bad_dump(script, cpu_ops, capsys):
    weights, heads = _shards(2)
    hidden = _hidden(32, seed=13)
    tokens = _fp16_tokens(weights, hidden)
    rows = script.HiddenRows(hidden, "dump:x (1 file(s))", 1, tokens, False, None)
    section = script.consistency_section(heads, weights, rows)
    assert section["dump_consistency"] == 1.0 and section["dump_consistency_ok"]
    assert section["dump_consistency_detail"]["rows_checked"] == 32
    assert capsys.readouterr().out.startswith("dump_consistency=1.0000")
    bad = rows._replace(draft_tokens=(tokens + 1) % 512)
    section = script.consistency_section(heads, weights, bad)
    assert section["dump_consistency_ok"] is False
    assert section["dump_consistency"] < script.DUMP_CONSISTENCY_MIN
    assert "WARNING" in capsys.readouterr().err
    # an old dump without draft_tokens: no metric, said so on the first line
    old = rows._replace(draft_tokens=None)
    assert script.consistency_section(heads, weights, old)["dump_consistency"] is None
    assert "no draft_tokens" in capsys.readouterr().out
