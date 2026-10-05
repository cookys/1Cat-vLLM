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


# ------------------------------------------------------------------- self-check
def _views():
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv

    return nv, nv.nvfp4_kv_split_views


def _random_cache(nv, blocks=3, block_size=8, heads=1, head_dim=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cache = torch.zeros(
        (blocks, 2, block_size, heads, nv.nvfp4_kv_row_bytes(head_dim)),
        dtype=torch.uint8,
    )
    key = torch.randn(blocks * block_size, heads, head_dim, generator=generator).half()
    value = torch.randn(
        blocks * block_size, heads, head_dim, generator=generator
    ).half()
    nv.reshape_and_cache_nvfp4_reference(
        key, value, cache, torch.arange(blocks * block_size)
    )
    return cache


def test_swapping_nibbles_exchanges_the_halves_of_every_byte(chk):
    data = torch.tensor([0x00, 0xAB, 0x1F, 0xF0, 0x77], dtype=torch.uint8)
    swapped = chk.swap_nibbles(data)
    assert swapped.tolist() == [0x00, 0xBA, 0xF1, 0x0F, 0x77]
    assert swapped.dtype == torch.uint8
    assert torch.equal(chk.swap_nibbles(swapped), data)


def test_the_sm100_swizzle_is_a_bijection_that_matches_the_kernel_formula(chk):
    source, destination = chk.sm100_scale_swizzle(8, 16)
    assert torch.equal(source.sort().values, torch.arange(8 * 16))
    assert torch.equal(destination.sort().values, torch.arange(8 * 16))
    # nvfp4_kv_cache_kernels.cu swizzle_scale_offset(t=5, s=7, scale_dim=16):
    # group 4, swizzled_t = 4 + 1, swizzled_s = 3 * 4 + 1.
    position = int((source == 5 * 16 + 7).nonzero())
    assert int(destination[position]) == 5 * 16 + 13
    assert not torch.equal(source, destination)
    for block, dim in ((6, 16), (8, 6)):
        with pytest.raises(ValueError, match="divisible by 4"):
            chk.sm100_scale_swizzle(block, dim)


def test_swizzled_scales_keep_the_bytes_and_move_them(chk):
    scales = torch.arange(2 * 8 * 3 * 16, dtype=torch.int64).remainder(251)
    scales = scales.reshape(2, 8, 3, 16).to(torch.uint8)
    moved = chk.swizzle_scales(scales)
    assert moved.shape == scales.shape
    assert torch.equal(
        moved.reshape(-1).sort().values, scales.reshape(-1).sort().values
    )
    assert not torch.equal(moved, scales)
    # Byte (t=5, s=7) of block 1, head 2 lands at (t=5, s=13) of that block and head.
    assert torch.equal(moved[1, 5, 2, 13], scales[1, 5, 2, 7])


def test_each_cache_mutation_changes_only_its_own_region(chk):
    nv, views = _views()
    cache = _random_cache(nv)
    before = cache.clone()
    (k_data, v_data), (k_scale, v_scale) = views(cache)
    original = {
        "data": (k_data.clone(), v_data.clone()),
        "scale": (k_scale.clone(), v_scale.clone()),
    }
    for mutation, changed, kept in (
        ("nibble_order_swapped", "data", "scale"),
        ("scale_placement_wrong", "scale", "data"),
    ):
        mutated = chk.mutate_cache(cache, mutation, views)
        assert torch.equal(cache, before)  # the input is only copied
        (m_k_data, m_v_data), (m_k_scale, m_v_scale) = views(mutated)
        now = {"data": (m_k_data, m_v_data), "scale": (m_k_scale, m_v_scale)}
        assert any(
            not torch.equal(a, b) for a, b in zip(original[changed], now[changed])
        )
        assert all(torch.equal(a, b) for a, b in zip(original[kept], now[kept]))
    with pytest.raises(ValueError, match="unknown mutation"):
        chk.mutate_cache(cache, "nope", views)


def test_half_up_encoders_differ_from_round_to_even_exactly_on_ties(chk):
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv

    ties = torch.tensor(chk.E2M1_TIE_VALUES)
    values = torch.cat([ties, -ties, torch.linspace(-7, 7, 2001)])
    rne, up = nv.e2m1_encode(values), chk.e2m1_encode_half_up(values)
    differs = rne != up
    assert differs.any()
    # Every difference is a tie that round-to-even sends down in magnitude and
    # half-up sends up.
    assert bool(
        torch.isin(values[differs].abs(), torch.tensor([0.25, 1.25, 2.5, 5.0])).all()
    )
    assert bool((up[differs] > rne[differs]).all())  # a larger magnitude code
    assert torch.equal(rne[~differs], up[~differs])

    codes = torch.arange(0, 0x7E, dtype=torch.uint8)
    grid = nv.e4m3_decode(codes)
    midpoints = (grid[:-1] + grid[1:]) / 2
    samples = torch.cat([midpoints, grid, torch.linspace(0, 448, 3001)])
    rne = nv.e4m3_encode(samples)
    up = chk.e4m3_encode_half_up(samples, nv.e4m3_encode, nv.e4m3_decode)
    differs = rne != up
    assert differs.any() and bool((up[differs] == rne[differs] + 1).all())
    tie_down = nv.e4m3_decode(rne[differs]) < samples[differs]
    assert bool(tie_down.all())  # only where round-to-even went down onto a tie
    assert torch.equal(rne[~differs], up[~differs])


def test_the_tie_data_sits_on_ties_and_separates_both_mutants(chk):
    nv, views = _views()
    key = chk.make_tie_kv(8, 1, 64)
    value = chk.make_tie_kv(8, 1, 64, shift=1)
    assert key.dtype == torch.float16 and key.shape == (8, 1, 64)
    assert not torch.equal(key, value)
    assert float(key[0, 0, :16].abs().max()) == 6.0  # an E2M1 tie group
    assert float(key[1, 0, :16].abs().max()) in chk.E4M3_TIE_AMAX  # an E4M3 tie group
    shape = (1, 2, 8, 1, nv.nvfp4_kv_row_bytes(64))

    def store() -> torch.Tensor:
        cache = torch.zeros(shape, dtype=torch.uint8)
        nv.reshape_and_cache_nvfp4_reference(key, value, cache, torch.arange(8))
        return cache

    correct = store()
    original = (nv.e2m1_encode, nv.e4m3_encode)
    with chk.patched(nv, e2m1_encode=chk.e2m1_encode_half_up):
        e2m1_mutant = store()
    rne_encode, decode = nv.e4m3_encode, nv.e4m3_decode
    with chk.patched(
        nv, e4m3_encode=lambda x: chk.e4m3_encode_half_up(x, rne_encode, decode)
    ):
        e4m3_mutant = store()
    assert (nv.e2m1_encode, nv.e4m3_encode) == original  # restored
    (c_data, _), (c_scale, _) = views(correct)
    (m2_data, _), (m2_scale, _) = views(e2m1_mutant)
    (m4_data, _), (m4_scale, _) = views(e4m3_mutant)
    assert not torch.equal(c_data, m2_data)  # the nibble mutant moves nibbles
    assert torch.equal(c_scale, m2_scale)  # and leaves the scale bytes alone
    assert not torch.equal(c_scale, m4_scale)  # the scale mutant moves scale bytes
    assert m4_data.shape == c_data.shape


def test_patched_restores_even_when_the_block_raises(chk):
    class Holder:
        value = 1

    with pytest.raises(RuntimeError), chk.patched(Holder, value=2):
        assert Holder.value == 2
        raise RuntimeError
    assert Holder.value == 1


def test_model_attention_matches_a_hand_computation(chk):
    layout = chk.plan_layout(8, rows=2, topk=2, context=4)
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]).half()
    keys = torch.zeros(layout.requests * layout.tokens_per_request, 1, 2)
    values = torch.zeros_like(keys)
    keys[0], keys[1] = torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 0.0]])
    values[0], values[1] = torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 20.0]])
    keys[layout.tokens_per_request] = torch.tensor([[0.0, 3.0]])
    values[layout.tokens_per_request] = torch.tensor([[7.0, 7.0]])
    selection = chk.Selection(
        indices=torch.tensor([[0, 1], [0, 0]], dtype=torch.int32),
        table=torch.zeros((layout.requests, layout.pages), dtype=torch.int32),
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        illegal=torch.zeros((2, 2), dtype=torch.bool),
        empty_rows=(),
    )
    out = chk.model_attention(q, keys, values, selection, layout.tokens_per_request)
    scale = 2**-0.5  # head_dim 2
    weights = torch.softmax(torch.tensor([2.0, 0.0]) * scale, dim=0)
    expected_row0 = weights[0] * torch.tensor([10.0, 0.0]) + weights[1] * torch.tensor(
        [0.0, 20.0]
    )
    assert torch.allclose(out[0, 0], expected_row0, atol=1e-6)
    # Row 1 reads request 1's token 0 twice: any weights give that value.
    assert torch.allclose(out[1, 0], torch.tensor([7.0, 7.0]), atol=1e-6)


def test_the_error_band_is_a_ratio_to_the_model(chk):
    assert chk.error_in_band(1.0, 1.0)
    assert chk.error_in_band(0.7, 1.0) and chk.error_in_band(1.4, 1.0)
    assert not chk.error_in_band(0.69, 1.0) and not chk.error_in_band(1.41, 1.0)
    assert chk.error_in_band(0.014, 0.01)
    for measured, model in (
        (0.0, 1.0),
        (1.0, 0.0),
        (float("nan"), 1.0),
        (1.0, float("nan")),
    ):
        assert not chk.error_in_band(measured, model)


def _region(different=0, bytes_=8):
    return {"bytes": bytes_, "different": different, "first_index": None}


def _comparison(data=0, scale=0):
    return {"data": _region(data), "scale": _region(scale)}


def _passing_nvfp4(rows=1, label=None):
    flagged = {
        "nibble_order_swapped": {
            "store": _comparison(data=5),
            "gather": {"different": 7},
            "decode": {"max_abs": 0.5},
        },
        "scale_placement_wrong": {
            "store": _comparison(scale=5),
            "gather": {"different": 7},
            "decode": {"max_abs": 0.5},
        },
    }
    result = _ok_result(label or f"nvfp4-B32-M{rows}", rows=rows)
    result["controls"] = flagged
    result["tie_control"] = {
        "correct": _comparison(),
        "e2m1_mutant": _comparison(data=3),
        "e4m3_mutant": _comparison(scale=2, data=1),
    }
    result["model_check"] = {"model_rel_l2": 0.1, "measured_rel_l2": 0.11}
    return result


def _passing_e4m3(rows=1):
    result = _e4m3_result(rows)
    result["model_check"] = {"model_rel_l2": 0.05, "measured_rel_l2": 0.049}
    return result


def _passing_run():
    results = [_passing_nvfp4(1), _passing_e4m3(1)]
    timings = {"M=1": {"decode_nvfp4_B32": {"median": 10.0}}}
    return results, timings


def test_a_fully_passing_run_is_a_passing_self_check(chk):
    results, timings = _passing_run()
    check = chk.compute_self_check(results, timings)
    assert check["verdict"] == "PASS" and check["reasons"] == []
    assert set(check["positive"].values()) == {"PASS"}
    assert set(check["negative"].values()) == {"FLAGGED"}
    assert set(check["stages"].values()) == {"RAN"}
    assert set(check["positive"]) == set(chk.POSITIVE_CONTROLS)
    assert set(check["stages"]) == set(chk.STAGES)
    assert len(check["negative"]) == len(chk.MUTATIONS) * len(chk.LINKS) + len(
        chk.TIE_MUTATIONS
    )
    lines = chk.format_self_check(check, results)
    assert lines[-1] == "SELF_CHECK: PASS"
    assert any(line.startswith("E4M3_BAND e4m3-B1616-M1:") for line in lines)
    assert any(line.startswith("NVFP4_BAND nvfp4-B32-M1:") for line in lines)
    assert "NEGATIVE_CONTROL nibble_order_swapped store: FLAGGED" in lines
    assert "NEGATIVE_CONTROL tie_rule_removed_e4m3 tie_store: FLAGGED" in lines
    assert "STAGE timing: RAN" in lines
    assert chk.exit_code_for_self_check(check) == 0


def _fail_reasons(chk, results, timings, **kwargs):
    check = chk.compute_self_check(results, timings, **kwargs)
    assert check["verdict"] == "FAIL"
    assert chk.exit_code_for_self_check(check) == 1
    assert chk.format_self_check(check, results)[-1].startswith("SELF_CHECK: FAIL (")
    return check


def test_a_positive_control_that_fails_fails_the_self_check(chk):
    results, timings = _passing_run()
    results[0]["store"] = {"data": _region(1), "scale": _region(0)}
    check = _fail_reasons(chk, results, timings)
    assert check["positive"]["STORE_MATCHES_REFERENCE"] == "FAIL"
    results, timings = _passing_run()
    results[0]["tie_control"]["correct"] = _comparison(scale=1)
    assert (
        _fail_reasons(chk, results, timings)["positive"]["TIE_STORE_MATCHES_REFERENCE"]
        == "FAIL"
    )


def test_an_error_outside_the_band_fails_for_either_format(chk):
    for index, key in ((0, "NVFP4_ERROR_IN_BAND"), (1, "E4M3_ERROR_IN_BAND")):
        results, timings = _passing_run()
        results[index]["model_check"]["measured_rel_l2"] *= 3
        assert _fail_reasons(chk, results, timings)["positive"][key] == "FAIL"
        results, timings = _passing_run()
        results[index]["model_check"]["measured_rel_l2"] /= 3
        assert _fail_reasons(chk, results, timings)["positive"][key] == "FAIL"


def test_a_mutation_the_comparison_cannot_see_is_missed(chk):
    results, timings = _passing_run()
    results[0]["controls"]["nibble_order_swapped"]["gather"] = {"different": 0}
    check = _fail_reasons(chk, results, timings)
    assert check["negative"]["nibble_order_swapped gather"] == "MISSED"
    assert check["negative"]["nibble_order_swapped store"] == "FLAGGED"
    results, timings = _passing_run()
    # The scale mutation must show in the scale region, not only in the data.
    results[0]["controls"]["scale_placement_wrong"]["store"] = _comparison(data=9)
    assert (
        _fail_reasons(chk, results, timings)["negative"]["scale_placement_wrong store"]
        == "MISSED"
    )
    results, timings = _passing_run()
    results[0]["tie_control"]["e2m1_mutant"] = _comparison()
    assert (
        _fail_reasons(chk, results, timings)["negative"][
            "tie_rule_removed_e2m1 tie_store"
        ]
        == "MISSED"
    )
    results, timings = _passing_run()
    results[0]["controls"]["scale_placement_wrong"]["decode"] = {"max_abs": 0.0}
    assert (
        _fail_reasons(chk, results, timings)["negative"]["scale_placement_wrong decode"]
        == "MISSED"
    )


def test_a_skipped_stage_fails_the_self_check(chk):
    results, timings = _passing_run()
    check = _fail_reasons(chk, results, timings, no_timing=True)
    assert check["stages"]["timing"] == "SKIPPED"
    assert any("stage timing SKIPPED" in r for r in check["reasons"])
    _fail_reasons(chk, results, {})  # timing requested but nothing was timed
    results, timings = _passing_run()
    check = _fail_reasons(chk, [results[0]], timings)  # no E4M3 control
    assert check["stages"]["e4m3_control"] == "SKIPPED"
    assert check["positive"]["E4M3_ERROR_IN_BAND"] == "SKIPPED"
    for stage_key, stage in (
        ("tie_control", "tie_control"),
        ("controls", "negative_controls"),
        ("model_check", "e4m3_control"),
    ):
        results, timings = _passing_run()
        target = results[1] if stage_key == "model_check" else results[0]
        del target[stage_key]
        assert _fail_reasons(chk, results, timings)["stages"][stage] == "SKIPPED"


def test_a_case_that_raised_fails_the_self_check(chk):
    results, timings = _passing_run()
    results.append(
        {"case": "nvfp4-B64-M5", "kind": "nvfp4", "rows": 5, "error": "RuntimeError: x"}
    )
    check = _fail_reasons(chk, results, timings)
    assert "nvfp4-B64-M5 raised" in check["reasons"]
    assert chk.exit_code_for(results) == 1


def test_a_zero_timing_is_not_a_run_timing(chk):
    results, _ = _passing_run()
    check = _fail_reasons(chk, results, {"M=1": {"decode_nvfp4_B32": {"median": 0.0}}})
    assert check["stages"]["timing"] == "SKIPPED"


def test_main_explains_a_vllm_without_the_nvfp4_modules(chk, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def missing():
        raise ImportError("No module named 'vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv'")

    monkeypatch.setattr(chk, "load_kernels", missing)
    assert chk.main(["--allow-busy"]) == 2
    output = capsys.readouterr().out
    assert (
        "cannot import the NVFP4 kernels" in output
        and "PYTHONPATH=<worktree>" in output
    )


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
    check = chk.compute_self_check(results, timings)
    assert check["verdict"] == "PASS", chk.format_self_check(check, results)
    lines = chk.format_self_check(check, results)
    assert lines[-1] == "SELF_CHECK: PASS"
    assert any(line.startswith("E4M3_BAND") for line in lines)
    ratio = results[0]["model_check"]
    assert chk.error_in_band(ratio["measured_rel_l2"], ratio["model_rel_l2"])


# ------------------------------------------------------------- the fused arm
def _fused_block(rel=3e-4, illegal=3e-4, zero_ok=True):
    stats = {"max_abs": 1e-3, "mean_abs": 1e-4, "rel_l2": rel, "reference_norm": 1.0}
    return {
        "vs_gather": stats,
        "vs_gather_illegal": {**stats, "rel_l2": illegal},
        "quality": {"max_abs": 0.2, "mean_abs": 0.01, "rel_l2": 0.1},
        "zero_fill_ok": zero_ok,
        "controls": {
            "nibble_order_swapped": {"decode": {"max_abs": 0.5}},
            "scale_placement_wrong": {"decode": {"max_abs": 0.5}},
        },
    }


def _passing_fused_run():
    results, timings = _passing_run()
    results[0]["fused"] = _fused_block()
    timings["M=1"]["decode_fused_nvfp4_B32"] = {"median": 8.0}
    return results, timings


def test_the_arms_default_to_gather_and_fused_and_gather_alone_is_accepted(chk):
    assert chk.parse_args([]).arms == ["gather", "fused"]
    assert chk.parse_args(["--arms", "gather"]).arms == ["gather"]
    with pytest.raises(SystemExit):
        chk.parse_args(["--arms", "bogus"])


def test_a_run_without_the_fused_arm_has_no_fused_lines(chk):
    results, timings = _passing_run()
    verdict = chk.compute_verdict(results)
    assert "FUSED_MATCHES_GATHER" not in verdict
    assert not any(line.startswith("FUSED") for line in chk.format_verdict(verdict, results))
    check = chk.compute_self_check(results, timings, fused_arm=False)
    assert check["verdict"] == "PASS" and "decode_fused" not in check["stages"]


def test_a_fused_arm_inside_the_tolerance_passes_and_prints_its_lines(chk):
    results, timings = _passing_fused_run()
    verdict = chk.compute_verdict(results)
    assert verdict["FUSED_MATCHES_GATHER"] == "YES"
    lines = chk.format_verdict(verdict, results)
    assert "FUSED_MATCHES_GATHER: YES" in lines
    assert any(line.startswith("FUSED_VS_GATHER nvfp4-B32-M1: max|d|") for line in lines)
    assert any(line.startswith("FUSED_VS_FP16 nvfp4-B32-M1:") for line in lines)
    check = chk.compute_self_check(results, timings, fused_arm=True)
    assert check["verdict"] == "PASS", check["reasons"]
    assert check["positive"]["FUSED_MATCHES_GATHER"] == "PASS"
    assert check["negative"]["nibble_order_swapped fused_decode"] == "FLAGGED"
    assert check["negative"]["scale_placement_wrong fused_decode"] == "FLAGGED"
    assert check["stages"]["decode_fused"] == "RAN"


@pytest.mark.parametrize(
    "block",
    [
        {"rel": 5e-3},
        {"illegal": 5e-3},
        {"zero_ok": False},
    ],
    ids=["clean_selection", "illegal_entries", "nonzero_empty_row"],
)
def test_a_fused_arm_outside_the_tolerance_fails(chk, block):
    results, timings = _passing_fused_run()
    results[0]["fused"] = _fused_block(**block)
    assert chk.compute_verdict(results)["FUSED_MATCHES_GATHER"] == "NO"
    check = _fail_reasons(chk, results, timings, fused_arm=True)
    assert check["positive"]["FUSED_MATCHES_GATHER"] == "FAIL"


def test_a_fused_mutation_the_comparison_cannot_see_is_missed(chk):
    results, timings = _passing_fused_run()
    results[0]["fused"]["controls"]["scale_placement_wrong"] = {"decode": {"max_abs": 0.0}}
    check = _fail_reasons(chk, results, timings, fused_arm=True)
    assert check["negative"]["scale_placement_wrong fused_decode"] == "MISSED"


def test_a_requested_fused_arm_that_never_ran_fails_even_if_every_case_raised(chk):
    results, timings = _passing_run()  # no "fused" key anywhere
    check = _fail_reasons(chk, results, timings, fused_arm=True)
    assert check["stages"]["decode_fused"] == "SKIPPED"
    assert check["positive"]["FUSED_MATCHES_GATHER"] == "SKIPPED"


def test_the_fused_arm_does_not_pollute_the_gather_roundtrip_line(chk):
    timings = {
        "M=1": {
            "decode_nvfp4_B2784": {"median": 300.0},
            "decode_fused_nvfp4_B2784": {"median": 240.0},
            "decode_e4m3_B1616": {"median": 150.0},
        }
    }
    assert chk.decode_roundtrip_lines(timings) == [
        "DECODE_ROUNDTRIP_US M=1: nvfp4 300.0 e4m3 150.0 ratio 2.00",
        "FUSED_ROUNDTRIP_US M=1: fused 240.0 gather 300.0 ratio 0.80",
    ]


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_one_small_case_runs_the_fused_arm_end_to_end_on_cpu_tensors(chk, monkeypatch):
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
        arms=["gather", "fused"],
    )
    geometry = chk.Geometry(head_dim=256, q_heads=6, kv_heads=1, topk=16, source="test")
    cases = chk.plan_cases([32], 32, [3])
    with torch.inference_mode():
        results, timings = chk.run_all(cases, geometry, chk.load_kernels(), args, lambda m: None)
    assert not any(r.get("error") for r in results), results
    nvfp4 = next(r for r in results if r["kind"] == "nvfp4")
    fused = nvfp4["fused"]
    assert fused["vs_gather"]["rel_l2"] < chk.FUSED_REL_L2_TOLERANCE
    assert fused["vs_gather_illegal"]["rel_l2"] < chk.FUSED_REL_L2_TOLERANCE
    assert fused["zero_fill_ok"]
    assert 0.0 < fused["quality"]["rel_l2"] < 0.5
    assert all(c["decode"]["max_abs"] > 0 for c in fused["controls"].values())
    assert "decode_fused_nvfp4_B32" in timings["M=3"]
    verdict = chk.compute_verdict(results)
    assert verdict["FUSED_MATCHES_GATHER"] == "YES"
    check = chk.compute_self_check(results, timings, fused_arm=True)
    assert check["verdict"] == "PASS", chk.format_self_check(check, results)


# ------------------------------------------------------ the prefill scratch arm
def _scratch_block(different=0, tail=True, route=None):
    block = {
        "decode": {"elements": 100, "different": different, "tail_zero_ok": tail},
        "controls": {
            "nibble_order_swapped": {"decode": {"different": 40}},
            "scale_placement_wrong": {"decode": {"different": 30}},
        },
    }
    if route is not None:
        block["route"] = route
    return block


def _route_block(fused=3e-4, gather=3e-4):
    stats = {"max_abs": 1e-3, "mean_abs": 1e-4, "reference_norm": 1.0}
    return {
        "vs_fused": {**stats, "rel_l2": fused},
        "vs_gather": {**stats, "rel_l2": gather},
    }


def _passing_scratch_run(route=None):
    results, timings = _passing_run()
    results[0]["prefill_scratch"] = _scratch_block(route=route)
    timings["M=1"]["prefill_scratch_decode_nvfp4_B32"] = {"median": 5.0}
    return results, timings


def test_the_prefill_scratch_arm_is_opt_in_and_the_default_arms_are_unchanged(chk):
    assert chk.parse_args([]).arms == ["gather", "fused"]
    assert chk.parse_args(["--arms", "prefill_scratch"]).arms == ["prefill_scratch"]
    assert "prefill_scratch" in chk.ARMS and "prefill_scratch" not in chk.DEFAULT_ARMS


def test_a_run_without_the_scratch_arm_has_no_scratch_lines(chk):
    results, timings = _passing_run()
    verdict = chk.compute_verdict(results)
    assert "SCRATCH_MATCHES_GATHER" not in verdict
    assert not any(
        line.startswith("SCRATCH") for line in chk.format_verdict(verdict, results)
    )
    check = chk.compute_self_check(results, timings, prefill_arm=False)
    assert check["verdict"] == "PASS" and "prefill_scratch" not in check["stages"]


def test_a_matching_scratch_decode_passes_and_prints_its_lines(chk):
    results, timings = _passing_scratch_run()
    verdict = chk.compute_verdict(results)
    assert verdict["SCRATCH_MATCHES_GATHER"] == "YES"
    assert verdict["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "UNKNOWN"  # no route off a GPU
    lines = chk.format_verdict(verdict, results)
    assert "SCRATCH_MATCHES_GATHER: YES" in lines
    assert any(
        line.startswith("SCRATCH_DECODE nvfp4-B32-M1: 0 of 100") for line in lines
    )
    check = chk.compute_self_check(results, timings, prefill_arm=True)
    assert check["verdict"] == "PASS", check["reasons"]
    assert check["positive"]["SCRATCH_MATCHES_GATHER"] == "PASS"
    assert "SCRATCH_ROUTE_WITHIN_TOLERANCE" not in check["positive"]  # not on a GPU
    assert check["negative"]["nibble_order_swapped scratch_decode"] == "FLAGGED"
    assert check["negative"]["scale_placement_wrong scratch_decode"] == "FLAGGED"
    assert check["stages"]["prefill_scratch"] == "RAN"


@pytest.mark.parametrize(
    "block",
    [{"different": 1}, {"tail": False}],
    ids=["a_value_differs_from_the_gather", "tail_not_zero"],
)
def test_a_scratch_decode_that_differs_from_the_gather_fails(chk, block):
    results, timings = _passing_scratch_run()
    results[0]["prefill_scratch"] = _scratch_block(**block)
    assert chk.compute_verdict(results)["SCRATCH_MATCHES_GATHER"] == "NO"
    check = _fail_reasons(chk, results, timings, prefill_arm=True)
    assert check["positive"]["SCRATCH_MATCHES_GATHER"] == "FAIL"


def test_a_scratch_mutation_the_comparison_cannot_see_is_missed(chk):
    results, timings = _passing_scratch_run()
    results[0]["prefill_scratch"]["controls"]["scale_placement_wrong"] = {
        "decode": {"different": 0}
    }
    check = _fail_reasons(chk, results, timings, prefill_arm=True)
    assert check["negative"]["scale_placement_wrong scratch_decode"] == "MISSED"


def test_a_requested_scratch_arm_that_never_ran_fails(chk):
    results, timings = _passing_run()  # no "prefill_scratch" anywhere
    check = _fail_reasons(chk, results, timings, prefill_arm=True)
    assert check["stages"]["prefill_scratch"] == "SKIPPED"
    assert check["positive"]["SCRATCH_MATCHES_GATHER"] == "SKIPPED"


def test_the_route_is_required_on_a_gpu_and_judged_by_both_tolerances(chk):
    results, timings = _passing_scratch_run(route=_route_block())
    assert chk.compute_verdict(results)["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "YES"
    lines = chk.format_verdict(chk.compute_verdict(results), results)
    for name in ("SCRATCH_VS_FUSED", "SCRATCH_VS_GATHER"):
        assert any(
            line.startswith(f"{name} nvfp4-B32-M1: max|d|") for line in lines
        )
    check = chk.compute_self_check(
        results, timings, prefill_arm=True, prefill_route_required=True
    )
    assert check["verdict"] == "PASS", check["reasons"]
    assert check["positive"]["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "PASS"
    for bad in (_route_block(fused=5e-3), _route_block(gather=5e-3)):
        results[0]["prefill_scratch"]["route"] = bad
        assert chk.compute_verdict(results)["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "NO"
        failed = chk.compute_self_check(
            results, timings, prefill_arm=True, prefill_route_required=True
        )
        assert failed["positive"]["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "FAIL"


def test_a_route_that_was_skipped_fails_a_gpu_run_but_not_a_cpu_run(chk):
    results, timings = _passing_scratch_run(route={"skipped": "no extension"})
    lines = chk.format_verdict(chk.compute_verdict(results), results)
    assert "SCRATCH_ROUTE nvfp4-B32-M1: skipped (no extension)" in lines
    cpu = chk.compute_self_check(results, timings, prefill_arm=True)
    assert cpu["verdict"] == "PASS"
    gpu = chk.compute_self_check(
        results, timings, prefill_arm=True, prefill_route_required=True
    )
    assert gpu["verdict"] == "FAIL"
    assert gpu["positive"]["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "SKIPPED"


def test_the_prefill_roundtrip_line_ratios_the_routes_to_e4m3(chk):
    timings = {
        "M=5568": {
            "prefill_scratch_route_nvfp4_B2784": {"median": 900.0},
            "prefill_fused_nvfp4_B2784": {"median": 1500.0},
            "prefill_e4m3_B1616": {"median": 1000.0},
            "decode_nvfp4_B2784": {"median": 300.0},
        }
    }
    lines = chk.decode_roundtrip_lines(timings)
    assert lines == [
        "PREFILL_ROUNDTRIP_US M=5568: e4m3 1000.0 scratch route 900.0 fused 1500.0 "
        "ratio to e4m3 (scratch route, fused) 0.90, 1.50"
    ]


def test_the_prefill_selection_is_causal_grouped_and_ends_in_the_tail(chk):
    layout = chk.plan_layout(32, 9, 16, 100)
    table = chk.make_block_table(layout, 0)
    selection = chk.build_prefill_selection(layout, table, 9, 16, 3)
    seq_len = layout.tokens_per_request
    assert selection.positions.tolist() == list(range(seq_len - 9, seq_len))
    assert selection.seq_lens.tolist() == [seq_len]
    assert selection.table.shape == (1, layout.pages)
    assert not selection.token_to_req.any()
    for row, position in enumerate(selection.positions.tolist()):
        entries = selection.indices[row].tolist()
        live = [e for e in entries if e >= 0]
        assert all(e <= position for e in live), "an entry past the row's position"
        visible = position + 1
        full, tail = visible // 4, visible % 4
        groups = (len(live) - tail) // 4
        assert groups == min(full, (16 - 3) // 4)
        for g in range(groups):
            first = entries[4 * g]
            assert first % 4 == 0 and entries[4 * g : 4 * g + 4] == list(
                range(first, first + 4)
            )
        assert entries[: 4 * groups] == sorted(entries[: 4 * groups])
        if tail:
            # The slot after the groups holds the partial group, from its first token.
            want = [full * 4 + t for t in range(tail)]
            assert entries[4 * groups : 4 * groups + tail] == want
        assert all(e == -1 for e in entries[4 * groups + tail :])


def test_a_chunk_longer_than_the_context_is_refused(chk):
    layout = chk.plan_layout(32, 9, 16, 100)
    with pytest.raises(ValueError, match="--context"):
        chk.build_prefill_selection(
            layout,
            chk.make_block_table(layout, 0),
            layout.tokens_per_request + 1,
            16,
            0,
        )


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_one_small_case_runs_the_scratch_arm_end_to_end_on_cpu(chk, monkeypatch):
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
        arms=["gather", "fused", "prefill_scratch"],
    )
    geometry = chk.Geometry(head_dim=256, q_heads=6, kv_heads=1, topk=16, source="test")
    cases = chk.plan_cases([32], 32, [9])
    with torch.inference_mode():
        results, timings = chk.run_all(
            cases, geometry, chk.load_kernels(), args, lambda m: None
        )
    assert not any(r.get("error") for r in results), results
    nvfp4 = next(r for r in results if r["kind"] == "nvfp4")
    scratch = nvfp4["prefill_scratch"]
    assert scratch["decode"]["different"] == 0 and scratch["decode"]["tail_zero_ok"]
    assert all(c["decode"]["different"] > 0 for c in scratch["controls"].values())
    assert "skipped" in scratch["route"]  # the grouped CUDA route needs a GPU
    assert "prefill_scratch_decode_nvfp4_B32" in timings["M=9"]
    assert "prefill_fused_nvfp4_B32" in timings["M=9"]
    assert not any("prefill_scratch_route" in name for name in timings["M=9"])
    verdict = chk.compute_verdict(results)
    assert verdict["SCRATCH_MATCHES_GATHER"] == "YES"
    check = chk.compute_self_check(results, timings, fused_arm=True, prefill_arm=True)
    assert check["verdict"] == "PASS", chk.format_self_check(check, results)


def test_a_chunk_too_short_for_the_cuda_route_is_not_held_against_the_route(chk):
    # --rows 1 5 512: the 1- and 5-row cases only prove the decode; the route is asked
    # of the 512-row case alone.
    results, timings = _passing_run()
    short = results[0]
    short["prefill_scratch"] = _scratch_block(
        route={"skipped": "1 rows (the route starts at 64)", "not_applicable": True}
    )
    long = _passing_nvfp4(rows=512, label="nvfp4-B32-M512")
    long["prefill_scratch"] = _scratch_block(route=_route_block())
    results.append(long)
    timings["M=512"] = {"prefill_scratch_route_nvfp4_B32": {"median": 9.0}}
    check = chk.compute_self_check(
        results, timings, prefill_arm=True, prefill_route_required=True
    )
    assert check["verdict"] == "PASS", check["reasons"]
    assert check["positive"]["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "PASS"
    # Only short cases: the route was never established, which a GPU run must say.
    only_short = chk.compute_self_check(
        results[:2],
        {"M=1": timings["M=1"]},
        prefill_arm=True,
        prefill_route_required=True,
    )
    assert only_short["positive"]["SCRATCH_ROUTE_WITHIN_TOLERANCE"] == "SKIPPED"
    assert only_short["verdict"] == "FAIL"


# ------------------------------------------------- the route knobs are pinned
def test_the_route_knobs_are_pinned_inside_the_context_and_restored_after(
    chk, monkeypatch
):
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_FUSED_READER", "1")
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS", "8")
    monkeypatch.delenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", raising=False)
    with chk.pinned_route_knobs():
        for name, value in chk.ROUTE_KNOBS.items():
            assert os.environ[name] == value
        assert chk.ROUTE_KNOBS["VLLM_SM70_QSA_NVFP4_FUSED_READER"] == "0"
        assert chk.ROUTE_KNOBS["VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH"] == "0"
        assert chk.ROUTE_KNOBS["VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS"] == "64"
    assert os.environ["VLLM_SM70_QSA_NVFP4_FUSED_READER"] == "1"
    assert os.environ["VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS"] == "8"
    assert "VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH" not in os.environ
    assert chk.ambient_route_knobs() == {
        "VLLM_SM70_QSA_NVFP4_FUSED_READER": "1",
        "VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH": None,
        "VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS": "8",
    }


def test_the_pin_is_restored_when_the_run_raises(chk, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", "1")
    with pytest.raises(RuntimeError):
        with chk.pinned_route_knobs():
            raise RuntimeError("boom")
    assert os.environ["VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH"] == "1"


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_an_exported_serving_environment_does_not_move_the_decode_controls(
    chk, monkeypatch
):
    """W1 of plan 071 exported FUSED_READER=1 and PREFILL_SCRATCH=1 (the chain's
    environment): the decode-path controls then ran the fused reader and read NO, and
    the timed gather arm was the fused reader. The run pins the knobs, so it reads the
    same as with a clean environment."""
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_FUSED_READER", "1")
    monkeypatch.setenv("VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH", "1")
    args = Namespace(
        seed=0,
        context=100,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        no_timing=True,
        rounds=1,
        iters=1,
        warmup=0,
        arms=["gather", "fused", "prefill_scratch"],
    )
    geometry = chk.Geometry(head_dim=256, q_heads=6, kv_heads=1, topk=16, source="test")
    cases = chk.plan_cases([32], 32, [3])
    logs: list[str] = []
    with torch.inference_mode():
        results, _ = chk.run_all(
            cases, geometry, chk.load_kernels(), args, logs.append
        )
    assert not any(r.get("error") for r in results), results
    verdict = chk.compute_verdict(results)
    assert verdict["ZERO_FILL_OK"] == "YES", chk.format_verdict(verdict, results)
    assert verdict["DECODE_PATH_IDENTICAL"] == "YES"
    assert verdict["FUSED_MATCHES_GATHER"] == "YES"
    nvfp4 = next(r for r in results if r["kind"] == "nvfp4")
    assert nvfp4["decode_path"]["max_abs"] == 0.0
    assert any(line.startswith("ROUTE_KNOBS ambient") for line in logs), logs
    assert os.environ["VLLM_SM70_QSA_NVFP4_FUSED_READER"] == "1"
    assert os.environ["VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH"] == "1"


# ------------------------------------------------- host memory (W1b, 2026-10-05)
# W1b ran ``--rows 5 64 5568`` in a 48G fence and was OOM-killed building the 5568-row
# case: the CPU reference of the gather (``gather_dequant_nvfp4_kv``) costs ~18.6 bytes
# per output element, ~19 KiB per row at topk 2051 x head_dim 256 for K alone, so
# 5568 rows needed ~51 GiB. The references now run on ``--ref-rows`` rows, a chunk at a
# time; the kernels and the timings still run on every row.
GIB = float(1 << 30)


def _geometry(chk, topk=2051):
    return chk.Geometry(head_dim=256, q_heads=6, kv_heads=1, topk=topk, source="test")


def _fa_patch():
    from vllm.v1.attention.backends import fa_utils

    fa_utils.get_flash_attn_version = lambda *args, **kwargs: 2


def _case_run(chk, monkeypatch, rows, *, topk=2051, block=2784, context=8192, **kwargs):
    """A CaseRun and the reference cache bytes of its K/V, built on the CPU."""
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    _fa_patch()
    kernels = chk.load_kernels()
    run = chk.CaseRun(
        chk.Case("nvfp4", block, rows),
        _geometry(chk, topk),
        kernels,
        seed=0,
        context=context,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        **kwargs,
    )
    nv = kernels["nv"]
    shape = (
        run.layout.num_blocks,
        2,
        run.layout.block_size,
        1,
        nv.nvfp4_kv_row_bytes(256),
    )
    reference = torch.zeros(shape, dtype=torch.uint8)
    nv.reshape_and_cache_nvfp4_reference(
        run.key,
        run.value,
        reference,
        run.slots,
        k_scale=run.k_scale,
        v_scale=run.v_scale,
    )
    return run, nv, reference


def _legacy_gather_check(nv, reference, sel, dev_k, dev_v, ks, vs):
    """The gather comparison of the kernel check at 017cbeb88, verbatim: the reference
    over the whole batch, then K, V and their FP32 copies, all in host RAM."""
    ref_keys, ref_values = nv.gather_dequant_nvfp4_kv(
        reference, sel.table, sel.token_to_req, sel.indices, k_scale=ks, v_scale=vs
    )
    keys, values = dev_k.clone(), dev_v.clone()  # ``.cpu()`` of the device output
    different = int((keys != ref_keys).sum() + (values != ref_values).sum())
    elements = keys.numel() + values.numel()
    max_abs = float(
        max(
            (keys.float() - ref_keys.float()).abs().max(),
            (values.float() - ref_values.float()).abs().max(),
        )
    )
    illegal = sel.illegal[..., None, None].expand_as(keys)
    zeros_ok = bool((keys[illegal] == 0).all() and (values[illegal] == 0).all())
    return {"different": different, "elements": elements, "max_abs": max_abs}, zeros_ok


def _peak_delta(fn):
    """Peak host RSS above the RSS at the start of ``fn``, or None without /proc."""
    import gc

    gc.collect()
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")  # reset VmHWM to the current RSS
    except OSError:
        return None
    start = _rss("VmRSS")
    fn()
    return _rss("VmHWM") - start


def _rss(field):
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise AssertionError(field)


def test_the_new_options_parse_with_safe_defaults(chk):
    args = chk.parse_args([])
    assert args.ref_rows == chk.DEFAULT_REF_ROWS == 256
    assert args.ref_chunk == chk.REF_CHUNK_ROWS == 64
    assert args.max_host_gb == chk.DEFAULT_MAX_HOST_GB == 24.0
    args = chk.parse_args(["--ref-rows", "0", "--max-host-gb", "0", "--ref-chunk", "8"])
    assert (args.ref_rows, args.max_host_gb, args.ref_chunk) == (0, 0.0, 8)
    for bad in (["--ref-rows", "-1"], ["--max-host-gb", "-1"], ["--ref-chunk", "0"]):
        with pytest.raises(SystemExit):
            chk.parse_args(bad)


def test_reference_rows_are_every_row_up_to_the_limit_then_a_fixed_subset(chk):
    assert chk.reference_rows(5, 256) == [0, 1, 2, 3, 4]
    assert chk.reference_rows(256, 256) == list(range(256))
    assert chk.reference_rows(5568, 0) == list(range(5568))  # 0: no limit
    for rows, limit in ((257, 256), (1024, 256), (5568, 256), (5568, 64), (100, 10)):
        chosen = chk.reference_rows(rows, limit)
        assert chosen == sorted(set(chosen)) and len(chosen) <= limit
        # the rows the builder treats specially, and both ends of both requests
        assert {0, 1, rows - 2, rows - 1} <= set(chosen)
        assert chosen == chk.reference_rows(rows, limit)  # deterministic
        # spread over the whole batch, not a prefix
        assert max(b - a for a, b in zip(chosen, chosen[1:], strict=False)) <= 2 * (
            rows // max(1, limit - 4)
        ) + 2
    assert len(chk.reference_rows(5568, 256)) >= 250


@pytest.mark.parametrize("inject", [False, True])
@pytest.mark.parametrize("rows,block,context", [(1, 4, 40), (3, 4, 40), (7, 8, 90)])
def test_the_vectorized_illegal_mask_equals_the_per_entry_loop(
    chk, rows, block, context, inject
):
    layout = chk.plan_layout(block, rows, 16, context)
    table = chk.make_block_table(layout, 0)
    sel = chk.build_selection(layout, table, rows, 16, 5, inject_illegal=inject)
    # the loop 017cbeb88 used, as the oracle
    expected = torch.zeros(sel.indices.shape, dtype=torch.bool)
    for row in range(rows):
        request = int(sel.token_to_req[row])
        for col in range(16):
            token = int(sel.indices[row, col])
            bad = request < 0 or request >= layout.requests or token < 0
            if not bad:
                page = token // layout.block_size
                bad = page >= layout.pages
                if not bad:
                    block_id = int(sel.table[request, page])
                    bad = block_id < 0 or block_id >= layout.num_blocks
            expected[row, col] = bad
    assert torch.equal(sel.illegal, expected)
    assert bool(sel.illegal.any()) == inject


def test_the_per_row_byte_model_explains_the_w1b_oom(chk):
    g = _geometry(chk)
    n = 5568 * g.topk
    legacy_reference = chk.ref_gather_peak_bytes(n, 1, 256)
    # one 5568-row reference gather alone, ~51 GiB: past the fence's 48 GiB
    assert 48 * GIB < legacy_reference < 54 * GIB
    # and the K side alone holds 11.4M entries x 256 x 2 B = 5.4 GiB of FP16
    assert abs(n * 256 * 2 / GIB - 5.45) < 0.05
    new = chk.host_mem_model("nvfp4", 5568, g, 2784, 8192, prefill=True)
    assert new["peak"] < 2.5 * GIB  # the whole case, not 51 GiB
    # linear in rows without the cap, flat with it
    for rows in (1024, 5568):
        capped = chk.host_mem_model("nvfp4", rows, g, 2784, 8192)["peak"]
        assert capped < 2.5 * GIB
    uncapped = [
        chk.host_mem_model("nvfp4", rows, g, 2784, 8192, ref_rows=0)["peak"]
        for rows in (1024, 2048)
    ]
    assert 1.7 < uncapped[1] / uncapped[0] < 2.1
    # the E4M3 control never builds a gather reference
    assert chk.host_mem_model("e4m3", 5568, g, 1616, 8192)["peak"] < 1.5 * GIB


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
@pytest.mark.parametrize("rows", [64, 256])
def test_the_model_matches_the_measured_peak_rss(chk, monkeypatch, rows):
    """Measured (VmHWM) against the model within 30 %, for the code that died at 5568
    (the whole-batch reference and its comparison) and for the code that replaces it."""
    run, nv, reference = _case_run(chk, monkeypatch, rows, prefill_scratch=True)
    g, sel = run.geometry, run.dirty
    ks, vs = run.k_scale, run.v_scale
    entries = rows * g.topk
    elements = entries * 256
    # What the GPU holds in the real run: zero host bytes (a stride-0 view here), and
    # for the old code the ``.cpu()`` copies it made.
    dev = torch.zeros((1, g.topk, 1, 256), dtype=torch.float16).expand(rows, -1, -1, -1)
    legacy_dev = torch.zeros((rows, g.topk, 1, 256), dtype=torch.float16)

    legacy = _peak_delta(
        lambda: _legacy_gather_check(nv, reference, sel, legacy_dev, legacy_dev, ks, vs)
    )
    new = _peak_delta(
        lambda: (
            run._gather_diff(dev, dev, *run._ref_gather(nv, reference, sel)),
            run._zero_fill_ok(dev, dev, sel.illegal),
        )
    )
    if legacy is None or new is None:
        pytest.skip("no /proc/self/clear_refs")
    legacy_model = max(
        chk.ref_gather_peak_bytes(entries, 1, 256),
        20 * elements,  # keys, values, ref x2 in FP16, plus three FP32 temporaries
    )
    parts = chk.host_mem_model("nvfp4", rows, g, 2784, 8192, prefill=True)
    assert 0.7 < legacy_model / legacy < 1.3, (legacy_model / GIB, legacy / GIB)
    assert 0.7 < parts["reference_phase"] / new < 1.3, (
        parts["reference_phase"] / GIB,
        new / GIB,
    )
    # the point of the change: at 256 rows the old path is already ~2.5x the new one
    if rows == 256:
        assert legacy > 2 * new


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_the_chunked_reference_equals_the_whole_batch_reference_bitwise(chk, monkeypatch):
    # M <= 64 at the production geometry: every row, several chunks
    run, nv, reference = _case_run(chk, monkeypatch, 64, ref_chunk=24)
    sel = run.dirty
    assert run.checked == list(range(64))
    got_k, got_v = run._ref_gather(nv, reference, sel)
    want_k, want_v = nv.gather_dequant_nvfp4_kv(
        reference,
        sel.table,
        sel.token_to_req,
        sel.indices,
        k_scale=run.k_scale,
        v_scale=run.v_scale,
    )
    assert torch.equal(got_k.view(torch.int16), want_k.view(torch.int16))
    assert torch.equal(got_v.view(torch.int16), want_v.view(torch.int16))
    # the comparison reports what the old formulas report, on a perturbed output
    dev_k, dev_v = want_k.clone(), want_v.clone()
    dev_k[3, 5, 0, 7] += 1.0
    dev_v[63, 2000, 0, 255] += 0.5  # the last row, the last chunk
    dev_k[0, 0, 0, 0] = 9.0  # row 0 entry 0 is illegal: must also trip zero-fill
    old, old_zero = _legacy_gather_check(
        nv, reference, sel, dev_k, dev_v, run.k_scale, run.v_scale
    )
    assert old["different"] == 3 and not old_zero
    assert run._gather_diff(dev_k, dev_v, got_k, got_v) == old
    assert run._zero_fill_ok(dev_k, dev_v, sel.illegal) == old_zero
    assert run._zero_fill_ok(want_k, want_v, sel.illegal)
    # a nonzero illegal entry in the last chunk is seen too (row 63 is the empty row)
    bad = want_k.clone()
    bad[63, int(sel.illegal[63].nonzero()[0]), 0, 0] = 1.0
    assert not run._zero_fill_ok(bad, want_v, sel.illegal)


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_a_subset_reference_equals_the_whole_batch_reference_on_those_rows(
    chk, monkeypatch
):
    rows, limit = 130, 20
    run, nv, reference = _case_run(chk, monkeypatch, rows, topk=512, ref_rows=limit, ref_chunk=7)
    sel = run.dirty
    assert run.checked == chk.reference_rows(rows, limit) and len(run.checked) <= limit
    got_k, got_v = run._ref_gather(nv, reference, sel)
    want_k, want_v = nv.gather_dequant_nvfp4_kv(
        reference,
        sel.table,
        sel.token_to_req,
        sel.indices,
        k_scale=run.k_scale,
        v_scale=run.v_scale,
    )
    pick = torch.tensor(run.checked)
    assert got_k.shape[0] == len(run.checked)
    assert torch.equal(got_k.view(torch.int16), want_k[pick].view(torch.int16))
    assert torch.equal(got_v.view(torch.int16), want_v[pick].view(torch.int16))
    # a difference in a checked row is counted, one in an unchecked row is not (that is
    # the cost of the subset; the report says how many rows were checked)
    unchecked = next(r for r in range(rows) if r not in run.checked)
    dev_k = want_k.clone()
    dev_k[run.checked[5], 100, 0, 3] += 1.0
    dev_k[unchecked, 100, 0, 3] += 1.0
    result = run._gather_diff(dev_k, want_v, got_k, got_v)
    assert result["different"] == 1
    assert result["elements"] == 2 * len(run.checked) * 512 * 256
    # but the zero-fill check sees every row
    illegal = sel.illegal.nonzero()
    row, col = int(illegal[-1][0]), int(illegal[-1][1])  # the empty last row
    poisoned = want_k.clone()
    poisoned[row, col, 0, 0] = 1.0
    assert not run._zero_fill_ok(poisoned, want_v, sel.illegal)


def test_a_case_over_the_host_limit_is_skipped_not_run(chk, monkeypatch):
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    args = Namespace(
        seed=0,
        context=8192,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        no_timing=False,
        rounds=1,
        iters=1,
        warmup=0,
        arms=["gather", "fused", "prefill_scratch"],
        ref_rows=chk.DEFAULT_REF_ROWS,
        max_host_gb=0.001,  # nothing fits
    )
    cases = chk.plan_cases([2784], 1616, [5, 5568])
    logs: list[str] = []
    # no kernels needed: a skipped case never builds a CaseRun
    results, timings = chk.run_all(cases, _geometry(chk), {}, args, logs.append)
    assert [r["case"] for r in results] == [c.label for c in cases]
    assert all(r.get("skipped") and not r.get("error") for r in results)
    assert timings == {}
    estimates = [line for line in logs if line.startswith("HOST_MEM_EST: rows=")]
    assert len(estimates) == len(cases)
    assert estimates[0].startswith("HOST_MEM_EST: rows=5 est_peak_GB=")
    assert any(line.startswith("HOST_MEM_EST: rows=5568 est_peak_GB=") for line in logs)
    assert sum(line.startswith("SKIPPED ") for line in logs) == len(cases)
    est = results[0]["host_mem_est"]
    assert est["rows"] == 5 and est["est_peak_gib"] > 0 and est["max_host_gb"] == 0.001
    assert est["model_gib"]["peak"] > 0
    verdict = chk.compute_verdict(results)
    assert verdict["skipped"].keys() == {r["case"] for r in results}
    assert verdict["GATHER_MATCHES_REFERENCE"] == "UNKNOWN"
    lines = chk.format_verdict(verdict, results)
    assert any(line.startswith("CASE_SKIPPED nvfp4-B2784-M5568:") for line in lines)
    check = chk.compute_self_check(results, timings)
    assert check["verdict"] == "FAIL"
    assert any("skipped by the host-memory guard" in r for r in check["reasons"])
    json.dumps(results)  # the estimate goes into the JSON report


def test_the_default_limit_lets_the_5568_row_case_through(chk, monkeypatch):
    """With the defaults the 5568-row case is estimated well under the 24 GiB limit
    (the estimate adds this process's own RSS to the model)."""
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    args = Namespace(
        seed=0,
        context=8192,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        no_timing=True,
        rounds=1,
        iters=1,
        warmup=0,
        arms=["gather", "fused", "prefill_scratch"],
    )
    cases = chk.plan_cases([2784, 2864], 1616, [5568])
    logs: list[str] = []
    # kernels={} makes the CaseRun raise (reported per case) after the guard passed
    results, _ = chk.run_all(cases, _geometry(chk), {}, args, logs.append)
    assert not any(line.startswith("SKIPPED ") for line in logs), logs
    for r in results:
        assert r["host_mem_est"]["est_peak_gib"] < chk.DEFAULT_MAX_HOST_GB
        assert r["ref_rows"]["total"] == 5568 and 250 <= r["ref_rows"]["checked"] <= 256


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_a_batch_over_the_reference_limit_runs_end_to_end_and_says_how_many_rows(
    chk, monkeypatch
):
    _fa_patch()
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    args = Namespace(
        seed=0,
        context=100,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        no_timing=True,
        rounds=1,
        iters=1,
        warmup=0,
        arms=["gather"],
        ref_rows=5,
        ref_chunk=2,
    )
    geometry = _geometry(chk, topk=16)
    cases = chk.plan_cases([32], 32, [9])
    logs: list[str] = []
    with torch.inference_mode():
        results, _ = chk.run_all(cases, geometry, chk.load_kernels(), args, logs.append)
    assert not any(r.get("error") for r in results), results
    nvfp4 = next(r for r in results if r["kind"] == "nvfp4")
    checked = chk.reference_rows(9, 5)
    assert nvfp4["ref_rows"] == {"checked": len(checked), "total": 9}
    assert len(checked) < 9
    assert nvfp4["gather"]["elements"] == 2 * len(checked) * 16 * 256
    assert nvfp4["gather"]["different"] == 0
    assert nvfp4["zero_fill"]["ok"]
    verdict = chk.compute_verdict(results)
    assert verdict["GATHER_MATCHES_REFERENCE"] == "YES"
    assert verdict["ZERO_FILL_OK"] == "YES"
    assert verdict["REFERENCE_ROWS"]["nvfp4-B32-M9"] == f"{len(checked)}/9"
    lines = chk.format_verdict(verdict, results)
    assert any(
        line.startswith(f"REFERENCE_ROWS nvfp4-B32-M9: CPU reference and model on "
                        f"{len(checked)} of 9 rows (a deterministic subset)")
        for line in lines
    )
    for r in results:  # the model check measures on the same rows it models
        assert chk.error_in_band(
            r["model_check"]["measured_rel_l2"], r["model_check"]["model_rel_l2"]
        )
    assert "measured_vmhwm_gib" in nvfp4["host_mem_est"]


# ------------------------------------ the 5- and 64-row controls did not move
GOLDEN = Path(__file__).parent / "data" / "kernel_check_017cbeb88_golden.json"
NEW_KEYS = {"ref_rows", "host_mem_est"}


def _same(path, got, want):
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        assert set(got) - NEW_KEYS == set(want), (path, set(got) ^ set(want))
        for key, value in want.items():
            _same(f"{path}.{key}", got[key], value)
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _same(f"{path}[{i}]", g, w)
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-12) or (
            want != want and got != got
        ), (path, got, want)
    else:
        assert got == want, (path, got, want)


def _run_against_golden(chk, monkeypatch, rows):
    _fa_patch()
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    golden = json.loads(GOLDEN.read_text())
    args = Namespace(
        seed=0,
        context=100,
        k_scale=chk.DEFAULT_K_SCALE,
        v_scale=chk.DEFAULT_V_SCALE,
        no_timing=True,
        rounds=1,
        iters=1,
        warmup=0,
        arms=["gather", "fused"],
    )
    cases = chk.plan_cases([32], 32, [rows])
    with torch.inference_mode():
        results, _ = chk.run_all(
            cases, _geometry(chk, topk=16), chk.load_kernels(), args, lambda m: None
        )
    want = [r for r in golden["results"] if r["rows"] == rows]
    assert [r["case"] for r in results] == [r["case"] for r in want]
    _same("results", results, want)
    verdict = chk.compute_verdict(results)
    for key in ("STORE_MATCHES_REFERENCE", "GATHER_MATCHES_REFERENCE", "ZERO_FILL_OK"):
        assert verdict[key] == golden["verdict"][key] == "YES"
    assert verdict["FUSED_MATCHES_GATHER"] == golden["verdict"]["FUSED_MATCHES_GATHER"]


@pytest.mark.skipif(not INTERPRETER, reason="needs TRITON_INTERPRET=1")
def test_the_5_row_controls_equal_those_of_commit_017cbeb88(chk, monkeypatch):
    """Store, gather, zero-fill, decode path, the mutated-cache controls, the tie
    control, the quantization error and the model check of an M = 5 case: the numbers
    the kernel check printed before the reference was chunked (golden captured with
    the script of that commit)."""
    _run_against_golden(chk, monkeypatch, 5)


@pytest.mark.skipif(
    not INTERPRETER or os.environ.get("KERNEL_CHECK_GOLDEN_M64") != "1",
    reason="~3 min in the interpreter: set TRITON_INTERPRET=1 KERNEL_CHECK_GOLDEN_M64=1",
)
def test_the_64_row_controls_equal_those_of_commit_017cbeb88(chk, monkeypatch):
    _run_against_golden(chk, monkeypatch, 64)


# ------------------------------------------------------- a poisoned CUDA context
class AcceleratorError(RuntimeError):
    """Stands in for ``torch.AcceleratorError`` (matched by its class name)."""


def test_only_faults_that_kill_the_context_count_as_poisoning(chk):
    assert chk.is_context_poisoning(AcceleratorError("CUDA error: whatever"))
    assert chk.is_context_poisoning(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    )
    assert chk.is_context_poisoning(RuntimeError("CUDA error: misaligned address"))
    assert not chk.is_context_poisoning(ValueError("shape mismatch"))
    assert not chk.is_context_poisoning(
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )


def _fake_case_run(fail_label, exc):
    class FakeCaseRun:
        def __init__(self, case, *args, **kwargs):
            self.case = case

        def run_nvfp4(self):
            if self.case.label == fail_label:
                raise exc
            return {"gather": {"different": 0, "elements": 1, "max_abs": 0.0}}

        def run_e4m3(self):
            return {}

        def timing_arms(self):
            return {}

    return FakeCaseRun


def _run_poison_args(**extra):
    return Namespace(
        seed=0,
        context=100,
        k_scale=1.0,
        v_scale=1.0,
        no_timing=False,
        rounds=1,
        iters=1,
        warmup=0,
        **extra,
    )


def _poison_run(chk, monkeypatch, **extra):
    geometry = chk.Geometry(head_dim=16, q_heads=1, kv_heads=1, topk=4, source="test")
    cases = chk.plan_cases([32, 48, 64], None, [3])
    exc = AcceleratorError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(chk, "CaseRun", _fake_case_run(cases[1].label, exc))
    logs: list[str] = []
    results, timings = chk.run_all(
        cases, geometry, {}, _run_poison_args(**extra), logs.append
    )
    return cases, results, timings, logs


def test_a_poisoned_context_stops_the_run_and_the_later_cases_are_not_run(
    chk, monkeypatch
):
    cases, results, timings, logs = _poison_run(chk, monkeypatch)
    assert [r["case"] for r in results] == [c.label for c in cases]
    first, poisoned, later = results
    assert "error" not in first and not first.get("context_poisoned")
    assert poisoned["context_poisoned"] and "AcceleratorError" in poisoned["error"]
    assert later["context_poison_skip"] and "error" not in later
    assert later["skipped"].startswith("not run: CONTEXT_POISONED")
    assert timings == {}
    assert "TIMING SKIPPED: CONTEXT_POISONED" in logs
    assert any(line.startswith("CONTEXT_POISONED at") for line in logs)
    assert not any(line.startswith("FAILED " + cases[2].label) for line in logs)

    verdict = chk.compute_verdict(results)
    assert verdict["CONTEXT_POISONED"] == "YES"
    assert verdict["context_poisoned_at"] == cases[1].label
    lines = chk.format_verdict(verdict, results)
    assert "CONTEXT_POISONED: YES" in lines
    assert any(
        line.startswith(f"CONTEXT_POISONED_AT {cases[1].label}:")
        and cases[2].label in line
        for line in lines
    )
    assert chk.exit_code_for(results) == 1
    check = chk.compute_self_check(results, timings)
    assert check["verdict"] == "FAIL"
    assert f"{cases[1].label} CONTEXT_POISONED" in check["reasons"]
    assert f"{cases[2].label} not run after CONTEXT_POISONED" in check["reasons"]


def test_the_continue_knob_keeps_the_old_behavior(chk, monkeypatch):
    cases, results, _, logs = _poison_run(
        chk, monkeypatch, continue_on_cuda_error=True
    )
    assert not any(r.get("context_poisoned") or r.get("context_poison_skip") for r in results)
    assert results[1]["error"] and "error" not in results[2]
    assert not any("CONTEXT_POISONED" in line for line in logs)
    assert chk.compute_verdict(results)["CONTEXT_POISONED"] == "NO"


def test_an_ordinary_error_does_not_stop_the_run(chk, monkeypatch):
    geometry = chk.Geometry(head_dim=16, q_heads=1, kv_heads=1, topk=4, source="test")
    cases = chk.plan_cases([32, 48, 64], None, [3])
    monkeypatch.setattr(
        chk, "CaseRun", _fake_case_run(cases[1].label, ValueError("bad shape"))
    )
    args = _run_poison_args()
    args.no_timing = True
    results, _ = chk.run_all(cases, geometry, {}, args, lambda m: None)
    assert results[1]["error"] == "ValueError: bad shape"
    assert not any(r.get("context_poison_skip") for r in results)
    assert chk.compute_verdict(results)["CONTEXT_POISONED"] == "NO"


def test_a_clean_run_prints_context_poisoned_no(chk):
    results = [_ok_result(), _e4m3_result()]
    assert "CONTEXT_POISONED: NO" in chk.format_verdict(
        chk.compute_verdict(results), results
    )


def test_the_continue_flag_is_parsed_and_off_by_default(chk):
    assert chk.parse_args([]).continue_on_cuda_error is False
    assert chk.parse_args(["--continue-on-cuda-error"]).continue_on_cuda_error is True
