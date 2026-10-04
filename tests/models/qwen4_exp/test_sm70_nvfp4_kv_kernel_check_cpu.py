# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for benchmarks/sm70_nvfp4_kv_kernel_check.py.

The script needs a V100 to answer its question. What runs without one is everything
around the measurement: argument parsing, the geometry it reads from the model
config, the case and cache planning, the seeded K/V and selection builders (including
the illegal entries, whose expected positions are checked against the library's own
validity rule), the byte comparison, the error and timing statistics, the ABBA order,
the verdict and exit-status logic, and the fail-fast paths of ``main``. With
``TRITON_INTERPRET=1`` one small case also runs end to end on CPU tensors through the
Triton interpreter (store, gather, decode, E4M3 control, timing), which shows the
script's wiring is right even though the numbers are not GPU numbers.

    cd <worktree> && CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> \\
        uv run --no-project --python <venv>/bin/python --with pytest -- \\
        python -m pytest --noconftest \\
        tests/models/qwen4_exp/test_sm70_nvfp4_kv_kernel_check_cpu.py -q
    # add TRITON_INTERPRET=1 for the end-to-end case
"""

import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "sm70_nvfp4_kv_kernel_check.py"
)
REAL_CONFIG = Path("/data/models/Qwen3.8-Flash-Next-NVFP4/config.json")
INTERPRETER = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture(scope="module")
def chk():
    spec = importlib.util.spec_from_file_location("sm70_nvfp4_kv_kernel_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


# ------------------------------------------------------------ argument parsing
def test_defaults_describe_the_production_run(chk):
    args = chk.parse_args([])
    assert args.nvfp4_blocks == [2784, 2864]
    assert args.e4m3_block == 1616
    assert args.rows == [1, 5]
    assert args.tp == 4 and args.context == 8192
    assert not args.allow_busy and not args.no_timing and args.out is None
    assert args.max_used_mib == 1024


@pytest.mark.parametrize(
    "argv",
    [
        ["--iters", "0"],
        ["--rounds", "0"],
        ["--warmup", "-1"],
        ["--context", "0"],
        ["--k-scale", "0"],
        ["--v-scale", "-1"],
    ],
)
def test_unusable_numbers_are_refused(chk, argv):
    with pytest.raises(SystemExit):
        chk.parse_args(argv)


# ------------------------------------------------------------------- geometry
@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="model config not on this host")
def test_geometry_from_the_real_config_matches_production(chk):
    tp4 = chk.read_geometry_from_config(REAL_CONFIG, 4)
    assert tp4 == {"head_dim": 256, "q_heads": 6, "kv_heads": 1, "topk": 2051}
    tp1 = chk.read_geometry_from_config(REAL_CONFIG, 1)
    assert tp1 is not None and (tp1["q_heads"], tp1["kv_heads"]) == (24, 2)
    tp2 = chk.read_geometry_from_config(REAL_CONFIG, 2)
    assert tp2 is not None and (tp2["q_heads"], tp2["kv_heads"]) == (12, 1)


def test_unreadable_or_inconsistent_configs_give_none(chk, tmp_path):
    assert chk.read_geometry_from_config(tmp_path / "missing.json", 4) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert chk.read_geometry_from_config(bad, 4) is None
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"head_dim": 256}))
    assert chk.read_geometry_from_config(partial, 4) is None
    config = {
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"text_config": config}))
    assert chk.read_geometry_from_config(path, 4)["topk"] == 2051
    assert chk.read_geometry_from_config(path, 5) is None  # 24 % 5
    assert chk.read_geometry_from_config(path, 0) is None
    config["head_dim"] = 250  # not a multiple of the 16-value group
    path.write_text(json.dumps(config))
    assert chk.read_geometry_from_config(path, 4) is None


def test_resolve_geometry_falls_back_and_applies_overrides(chk, tmp_path):
    args = chk.parse_args(["--config", str(tmp_path / "missing.json")])
    geometry = chk.resolve_geometry(args)
    assert (geometry.head_dim, geometry.q_heads, geometry.kv_heads, geometry.topk) == (
        256,
        6,
        1,
        2051,
    )
    assert geometry.source == "production defaults" and geometry.group == 6
    args = chk.parse_args(
        [
            "--config",
            str(tmp_path / "x"),
            "--topk",
            "64",
            "--q-heads",
            "12",
            "--kv-heads",
            "2",
        ]
    )
    geometry = chk.resolve_geometry(args)
    assert (geometry.topk, geometry.group) == (64, 6) and "override" in geometry.source
    with pytest.raises(ValueError, match="multiple"):
        chk.resolve_geometry(
            chk.parse_args(
                ["--config", str(tmp_path / "x"), "--q-heads", "5", "--kv-heads", "2"]
            )
        )


# ------------------------------------------------------------------- planning
def test_plan_cases_crosses_blocks_rows_and_adds_the_control(chk):
    cases = chk.plan_cases([2784, 2864], 1616, [1, 5])
    assert [c.label for c in cases] == [
        "nvfp4-B2784-M1",
        "nvfp4-B2864-M1",
        "nvfp4-B2784-M5",
        "nvfp4-B2864-M5",
        "e4m3-B1616-M1",
        "e4m3-B1616-M5",
    ]
    assert [c.label for c in chk.plan_cases([2784], None, [5])] == ["nvfp4-B2784-M5"]


@pytest.mark.parametrize(
    ("blocks", "e4m3", "rows"),
    [
        ([2784], 1616, []),
        ([2784], 1616, [0]),
        ([6], None, [1]),
        ([2784], 7, [1]),
        ([], None, [1]),
        ([0], None, [1]),
    ],
)
def test_plan_cases_refuses_unusable_input(chk, blocks, e4m3, rows):
    with pytest.raises(ValueError):
        chk.plan_cases(blocks, e4m3, rows)


def test_layout_covers_the_context_and_keeps_two_spare_blocks(chk):
    layout = chk.plan_layout(2784, rows=5, topk=2051, context=8192)
    assert (layout.pages, layout.requests, layout.num_blocks) == (3, 2, 8)
    assert layout.tokens_per_request == 8192
    one = chk.plan_layout(2784, rows=1, topk=2051, context=8192)
    assert (one.requests, one.num_blocks) == (1, 5)
    # A context shorter than the selection is stretched so top-k tokens exist.
    short = chk.plan_layout(32, rows=2, topk=100, context=10)
    assert short.tokens_per_request == 101 and short.pages == 4


def test_block_table_and_slots_are_a_bijection_onto_distinct_cells(chk):
    layout = chk.plan_layout(32, rows=4, topk=16, context=100)
    table = chk.make_block_table(layout, seed=3)
    assert table.dtype == torch.int32 and table.shape == (2, layout.pages)
    assert torch.unique(table).numel() == table.numel()
    assert int(table.min()) >= 0 and int(table.max()) < layout.num_blocks
    assert torch.equal(table, chk.make_block_table(layout, seed=3))
    assert not torch.equal(table, chk.make_block_table(layout, seed=4))
    slots = chk.slot_mapping(table, layout)
    assert slots.dtype == torch.int64 and slots.numel() == 2 * 100
    assert torch.unique(slots).numel() == slots.numel()
    assert int(slots.max()) < layout.num_blocks * layout.block_size
    # Token 37 of request 1: page 1, offset 5.
    assert int(slots[100 + 37]) == int(table[1, 1]) * 32 + 5


def test_kv_is_seeded_finite_clamped_and_spread_over_magnitudes(chk):
    key, value = chk.make_kv(512, 1, 256, seed=1)
    assert key.dtype == value.dtype == torch.float16 and key.shape == (512, 1, 256)
    again, _ = chk.make_kv(512, 1, 256, seed=1)
    assert torch.equal(key, again) and not torch.equal(key, value)
    assert torch.isfinite(key.float()).all()
    assert float(key.abs().max()) <= chk.MAX_ABS_VALUE
    per_token = key.float().abs().amax(dim=(1, 2))
    assert float(per_token.max() / per_token.min()) > 10  # decades of spread


# ------------------------------------------------------------------ selections
def _selection(chk, rows, topk=24, illegal=False, seed=5):
    layout = chk.plan_layout(32, rows=rows, topk=topk, context=100)
    table = chk.make_block_table(layout, seed=1)
    return layout, chk.build_selection(
        layout, table, rows, topk, seed, inject_illegal=illegal
    )


def test_a_clean_selection_has_distinct_legal_tokens(chk):
    layout, sel = _selection(chk, rows=5)
    assert sel.indices.dtype == torch.int32 and sel.indices.shape == (5, 24)
    assert not sel.illegal.any() and sel.empty_rows == ()
    for row in range(5):
        assert torch.unique(sel.indices[row]).numel() == 24
        assert int(sel.indices[row].min()) >= 0
        assert int(sel.indices[row].max()) < layout.tokens_per_request
    assert torch.equal(
        sel.token_to_req, torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
    )


@pytest.mark.parametrize("rows", [1, 2, 3, 5])
def test_injected_illegal_entries_are_where_the_script_says_and_the_rule_agrees(
    chk, rows
):
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv

    layout, sel = _selection(chk, rows=rows, illegal=True)
    assert sel.illegal.any()
    assert bool(sel.illegal[0, :3].all())  # negative, past the table, huge
    assert (len(sel.empty_rows) == 1) == (rows >= 3)
    for row in sel.empty_rows:
        assert bool(sel.illegal[row].all())
    # The independent expectation equals the library's validity rule, entry by entry.
    valid = nv.nvfp4_entry_validity(
        sel.indices, sel.token_to_req, sel.table, layout.num_blocks, layout.block_size
    )
    assert torch.equal(~valid, sel.illegal)
    if layout.requests > 1:
        assert int(sel.table[1, 0]) == layout.num_blocks + 5
    assert int(sel.table[0, 1]) == -1


def test_illegal_injection_does_not_change_the_clean_table(chk):
    layout = chk.plan_layout(32, rows=3, topk=24, context=100)
    table = chk.make_block_table(layout, seed=1)
    before = table.clone()
    chk.build_selection(layout, table, 3, 24, 5, inject_illegal=True)
    assert torch.equal(table, before)
    with pytest.raises(ValueError, match="width of 8"):
        chk.build_selection(layout, table, 3, 4, 5, inject_illegal=True)


# ------------------------------------------------------------------ statistics
def test_compare_bytes_counts_and_locates_differences(chk):
    a = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
    assert chk.compare_bytes(a, a.clone()) == {
        "bytes": 24,
        "different": 0,
        "first_index": None,
    }
    b = a.clone()
    b[1, 2, 3] ^= 1
    b[1, 0, 0] ^= 1
    result = chk.compare_bytes(a, b)
    assert result == {"bytes": 24, "different": 2, "first_index": [1, 0, 0]}
    with pytest.raises(ValueError, match="shape"):
        chk.compare_bytes(a, a[:1])


def test_region_comparisons_merge_k_then_v(chk):
    zeros = torch.zeros(4, dtype=torch.uint8)
    one = zeros.clone()
    one[2] = 1
    k = chk.compare_cache_regions((zeros, zeros), (zeros, zeros))
    v = chk.compare_cache_regions((zeros, zeros), (one, one))
    merged = chk.merge_region_comparisons([k, v])
    assert merged["data"] == {"bytes": 8, "different": 1, "first_index": [2]}
    assert merged["scale"]["different"] == 1 and merged["scale"]["bytes"] == 8
    clean = chk.merge_region_comparisons([k, k])
    assert clean["data"]["first_index"] is None


def test_error_stats_known_values(chk):
    reference = torch.tensor([3.0, 4.0])
    actual = torch.tensor([3.0, 4.5])
    stats = chk.error_stats(actual, reference)
    assert stats["max_abs"] == 0.5 and stats["mean_abs"] == 0.25
    assert stats["rel_l2"] == pytest.approx(0.5 / 5.0)
    assert stats["reference_norm"] == pytest.approx(5.0)
    zero = chk.error_stats(torch.ones(2), torch.zeros(2))
    assert zero["rel_l2"] != zero["rel_l2"]  # NaN: no reference energy
    same = chk.error_stats(reference, reference)
    assert same["max_abs"] == 0.0 and same["rel_l2"] == 0.0


def test_timing_summary(chk):
    summary = chk.summarize_timing([[10.0, 12.0, 11.0], [20.0, 22.0, 21.0]])
    assert summary["median"] == pytest.approx(16.0)
    assert summary["p10"] == pytest.approx(10.5) and summary["p90"] == pytest.approx(
        21.5
    )
    assert summary["samples"] == 6
    # medians 11 and 21 around a center of 16: a spread of 10/16.
    assert summary["round_spread_pct"] == pytest.approx(62.5)
    with pytest.raises(ValueError):
        chk.summarize_timing([])


def test_abba_order_alternates_and_balances_the_arms(chk):
    order = chk.abba_order(["a", "b", "c"], 4)
    assert order == list("abc") + list("cba") + list("abc") + list("cba")
    for arm in "abc":
        assert order.count(arm) == 4
    assert chk.abba_order(["a"], 3) == ["a", "a", "a"]


# --------------------------------------------------------------------- verdict
def _ok_result(label="nvfp4-B2784-M1", rows=1, rel=0.1):
    return {
        "case": label,
        "kind": "nvfp4",
        "rows": rows,
        "store": {
            "data": {"bytes": 8, "different": 0, "first_index": None},
            "scale": {"bytes": 2, "different": 0, "first_index": None},
        },
        "gather": {"different": 0, "elements": 8, "max_abs": 0.0},
        "zero_fill": {"ok": True},
        "decode_path": {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0},
        "decode_quality": {"max_abs": 0.2, "mean_abs": 0.01, "rel_l2": rel},
    }


def _e4m3_result(rows=1, rel=0.05):
    return {
        "case": f"e4m3-B1616-M{rows}",
        "kind": "e4m3",
        "rows": rows,
        "decode_quality": {"max_abs": 0.1, "mean_abs": 0.005, "rel_l2": rel},
    }


def test_a_clean_run_says_yes_everywhere_and_ratios_the_errors(chk):
    results = [_ok_result(), _ok_result("nvfp4-B2864-M1", rel=0.2), _e4m3_result()]
    verdict = chk.compute_verdict(results)
    for key in (
        "STORE_MATCHES_REFERENCE",
        "GATHER_MATCHES_REFERENCE",
        "ZERO_FILL_OK",
        "DECODE_PATH_IDENTICAL",
    ):
        assert verdict[key] == "YES"
    assert verdict["NVFP4_VS_E4M3_REL_L2"] == {"M=1": pytest.approx(4.0)}
    assert verdict["errors"] == {}
    lines = chk.format_verdict(verdict, results)
    assert lines[:4] == [
        f"{k}: YES"
        for k in (
            "STORE_MATCHES_REFERENCE",
            "GATHER_MATCHES_REFERENCE",
            "ZERO_FILL_OK",
            "DECODE_PATH_IDENTICAL",
        )
    ]
    assert any(line.startswith("DECODE_VS_FP16 nvfp4-B2784-M1:") for line in lines)
    assert any(line.startswith("E4M3_CONTROL e4m3-B1616-M1:") for line in lines)
    assert "NVFP4_VS_E4M3_REL_L2 M=1: 4.00" in lines
    assert chk.exit_code_for(results) == 0


def test_each_broken_link_turns_only_its_own_line_to_no_with_evidence(chk):
    broken = _ok_result()
    broken["store"]["scale"] = {"bytes": 2, "different": 1, "first_index": [0, 1, 0, 0]}
    verdict = chk.compute_verdict([broken, _e4m3_result()])
    assert verdict["STORE_MATCHES_REFERENCE"] == "NO"
    assert verdict["GATHER_MATCHES_REFERENCE"] == "YES"
    assert any(
        line.startswith("STORE_DIFF") and "[0, 1, 0, 0]" in line
        for line in chk.format_verdict(verdict, [broken])
    )

    gather = _ok_result()
    gather["gather"] = {"different": 3, "elements": 8, "max_abs": 0.5}
    assert chk.compute_verdict([gather])["GATHER_MATCHES_REFERENCE"] == "NO"
    zero = _ok_result()
    zero["zero_fill"] = {"ok": False}
    assert chk.compute_verdict([zero])["ZERO_FILL_OK"] == "NO"
    path = _ok_result()
    path["decode_path"] = {"max_abs": 1e-3, "mean_abs": 0.0, "rel_l2": 1e-4}
    verdict = chk.compute_verdict([path])
    assert verdict["DECODE_PATH_IDENTICAL"] == "NO"
    assert any(
        line.startswith("DECODE_PATH_DIFF")
        for line in chk.format_verdict(verdict, [path])
    )


def test_unknown_when_a_case_never_got_there_and_no_beats_unknown(chk):
    failed = {
        "case": "nvfp4-B2784-M1",
        "kind": "nvfp4",
        "rows": 1,
        "error": "RuntimeError: boom",
    }
    verdict = chk.compute_verdict([failed])
    assert verdict["STORE_MATCHES_REFERENCE"] == "UNKNOWN"
    assert verdict["errors"] == {"nvfp4-B2784-M1": "RuntimeError: boom"}
    assert chk.exit_code_for([failed]) == 1
    assert any(
        line.startswith("CASE_ERROR") for line in chk.format_verdict(verdict, [failed])
    )
    # One case verified and fine, one never got there: UNKNOWN, not YES.
    assert (
        chk.compute_verdict([_ok_result(), failed])["STORE_MATCHES_REFERENCE"]
        == "UNKNOWN"
    )
    # One case that failed its comparison, one never got there: NO.
    bad = _ok_result()
    bad["gather"] = {"different": 1, "elements": 2, "max_abs": 1.0}
    assert chk.compute_verdict([bad, failed])["GATHER_MATCHES_REFERENCE"] == "NO"
    assert chk.compute_verdict([_e4m3_result()])["STORE_MATCHES_REFERENCE"] == "UNKNOWN"


def test_the_error_ratio_needs_both_formats_at_the_same_row_count(chk):
    verdict = chk.compute_verdict(
        [_ok_result(rows=5, label="nvfp4-B2784-M5"), _e4m3_result(1)]
    )
    assert verdict["NVFP4_VS_E4M3_REL_L2"] == {}


def test_roundtrip_lines_compare_every_nvfp4_arm_with_the_e4m3_decode(chk):
    timings = {
        "M=1": {
            "decode_nvfp4_B2784": {"median": 300.0},
            "decode_nvfp4_B2864": {"median": 330.0},
            "decode_fp16_dequantized_B2784": {"median": 100.0},
            "decode_e4m3_B1616": {"median": 150.0},
        },
        "M=5": {"decode_nvfp4_B2784": {"median": 400.0}},
    }
    lines = chk.decode_roundtrip_lines(timings)
    assert lines == [
        "DECODE_ROUNDTRIP_US M=1: nvfp4 300.0/330.0 e4m3 150.0 ratio 2.00/2.20"
    ]


# ------------------------------------------------------------------- fail fast
def test_main_refuses_without_cuda(chk, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert chk.main([]) == 2
    assert "needs a CUDA device" in capsys.readouterr().out


def test_main_refuses_a_gpu_that_is_already_in_use(chk, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 << 30, 32 << 30))
    assert chk.main([]) == 2
    output = capsys.readouterr().out
    assert "already holds 30720 MiB" in output and "--allow-busy" in output
    assert chk.refuse_busy_gpu(1024) is not None
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: ((32 << 30) - (300 << 20), 32 << 30)
    )
    assert chk.refuse_busy_gpu(1024) is None  # a context's worth of memory is fine


def test_main_refuses_unusable_geometry_before_loading_any_kernel(
    chk, monkeypatch, capsys
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(chk, "load_kernels", lambda: pytest.fail("loaded kernels"))
    assert chk.main(["--allow-busy", "--nvfp4-blocks", "6"]) == 2
    assert "unusable arguments" in capsys.readouterr().out


# ------------------------------------------------------- end to end, interpreter
@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_one_small_case_runs_end_to_end_on_cpu_tensors(chk, monkeypatch):
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    args = Namespace(
        seed=0,
        context=100,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        no_timing=False,
        rounds=2,
        iters=1,
        warmup=1,
    )
    geometry = chk.Geometry(head_dim=256, q_heads=6, kv_heads=1, topk=16, source="test")
    cases = chk.plan_cases([32], 32, [3])
    logs: list[str] = []
    with torch.inference_mode():
        results, timings = chk.run_all(
            cases, geometry, chk.load_kernels(), args, logs.append
        )
    errors = {r["case"]: r.get("error") for r in results}
    assert not any(errors.values()), errors
    verdict = chk.compute_verdict(results)
    assert verdict["STORE_MATCHES_REFERENCE"] == "YES", chk.format_verdict(
        verdict, results
    )
    assert verdict["GATHER_MATCHES_REFERENCE"] == "YES"
    assert verdict["ZERO_FILL_OK"] == "YES"
    assert verdict["DECODE_PATH_IDENTICAL"] == "YES"
    nvfp4 = next(r for r in results if r["kind"] == "nvfp4")
    e4m3 = next(r for r in results if r["kind"] == "e4m3")
    assert 0.0 < nvfp4["decode_quality"]["rel_l2"] < 0.5
    assert 0.0 < e4m3["decode_quality"]["rel_l2"] < 0.5
    assert (
        "NVFP4_VS_E4M3_REL_L2" in verdict and "M=3" in verdict["NVFP4_VS_E4M3_REL_L2"]
    )
    assert set(timings) == {"M=3"}
    assert {
        "store_nvfp4_B32",
        "gather_nvfp4_B32",
        "decode_nvfp4_B32",
        "decode_fp16_dequantized_B32",
        "store_e4m3_B32",
        "decode_e4m3_B32",
    } <= set(timings["M=3"])
    assert all(t["median"] > 0 for t in timings["M=3"].values())
    roundtrip = chk.decode_roundtrip_lines(timings)
    assert any("DECODE_ROUNDTRIP_US" in line for line in roundtrip)
