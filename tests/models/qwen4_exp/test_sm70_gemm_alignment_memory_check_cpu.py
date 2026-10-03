# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for benchmarks/sm70_gemm_alignment_memory_check.py.

The script needs a GPU to answer its question. What can run without one is
everything around the measurement: argument parsing, the shapes derived from the
model config, the settings and the worker environment, the byte-offset views, the
alignment and free-memory scenario planning against a fake ``mem_get_info``, the
hashing and reference helpers, the verdict logic on synthetic results, the worker
command lines, both workers on CPU tensors with a fake memory probe (including a
fake GEMM that depends on alignment or on the scenario, to prove the workers
report a dependence instead of hiding it), and the driver with a fake worker.

    cd <worktree> && CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> \
        uv run --no-project --python <venv>/bin/python --with pytest -- \
        python -m pytest --noconftest \
        tests/models/qwen4_exp/test_sm70_gemm_alignment_memory_check_cpu.py -q
"""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "sm70_gemm_alignment_memory_check.py"
)
REAL_CONFIG = Path("/data/models/Qwen3.8-Flash-Next-NVFP4-e4m3kv/config.json")
MIB = 1 << 20
GIB = 1 << 30
TINY_DIMS = {
    "hidden_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 8,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 8,
    "num_experts": 16,
}


@pytest.fixture(scope="module")
def chk():
    spec = importlib.util.spec_from_file_location("sm70_gemm_alignment_check", SCRIPT)
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


def test_defaults_match_the_task(chk):
    args = chk.parse_args([])
    assert args.m == [17, 1616, 6464, 8080]
    assert args.offsets == [0, 16, 32, 64, 128, 256, 512, 1024]
    assert args.b_offsets == [16]
    assert args.repeats == 3 and args.seed == 1234 and args.tp == 4
    assert args.free_mib == [16384, 8192, 4096, 2048, 1024, 512]
    assert args.config == "/data/models/Qwen3.8-Flash-Next-NVFP4-e4m3kv/config.json"
    assert args.settings == "all" and args.groups == "align,memory"
    assert args.memory_mode == "both"
    assert (args.shapes, args.n, args.worker, args.out) == (None, None, None, None)
    assert not (args.no_profile or args.cpu_reference or args.print_all_algo)


def test_overrides_and_lists_are_parsed(chk):
    args = chk.parse_args(
        [
            "--m", "5", "9", "--n", "32", "48", "--tp", "2", "--seed", "7",
            "--repeats", "2", "--offsets", "0,2,4", "8", "--b-offsets", "2,4",
            "--free-mib", "100", "--settings", "default,wscap16", "--groups", "align",
            "--memory-mode", "fresh", "--no-profile", "--cpu-reference",
            "--print-all-algo", "--workdir", "/tmp", "--out", "/tmp/x.json",
        ]
    )  # fmt: skip
    assert (args.m, args.n, args.tp, args.seed, args.repeats) == (
        [5, 9],
        [32, 48],
        2,
        7,
        2,
    )
    assert args.offsets == [0, 2, 4, 8] and args.b_offsets == [2, 4]
    assert args.free_mib == [100] and args.memory_mode == "fresh"
    assert args.no_profile and args.cpu_reference and args.print_all_algo
    assert args.out == "/tmp/x.json" and args.workdir == "/tmp"


def test_shape_triples_accept_commas_or_x(chk):
    assert chk.parse_shape_triple("17,2560,1792") == (17, 2560, 1792)
    assert chk.parse_shape_triple("17x2560x1792") == (17, 2560, 1792)
    args = chk.parse_args(["--shapes", "1,2,3", "4,5,6"])
    assert args.shapes == [(1, 2, 3), (4, 5, 6)]


@pytest.mark.parametrize(
    "argv",
    [
        ["--repeats", "0"],
        ["--tp", "0"],
        ["--m", "0"],
        ["--n", "-4"],
        ["--free-mib", "0"],
        ["--offsets", "3"],  # not a multiple of the 2-byte element
        ["--offsets", "-2"],
        ["--b-offsets", "1"],
        ["--offsets", "x"],
        ["--shapes", "1,2"],
        ["--shapes", "1,2,0"],
        ["--shapes", "1,2,3", "--m", "4"],
        ["--shapes", "1,2,3", "--n", "4"],
        ["--memory-mode", "sometimes"],
        ["--worker", "align", "--setting", "default"],  # a worker needs shapes
        ["--worker", "align", "--shapes", "1,2,3", "--setting", "nonsense"],
    ],
)
def test_bad_arguments_exit_2(chk, argv):
    with pytest.raises(SystemExit) as exc:
        chk.parse_args(argv)
    assert exc.value.code == 2


def test_name_selection(chk):
    available = ["default", "wscap16", "no_lt"]
    assert chk._select_names("all", available) == available
    assert chk._select_names("default, no_lt", available) == ["default", "no_lt"]
    with pytest.raises(SystemExit):
        chk._select_names("nonsense", available)


def test_help_works_without_a_gpu():
    completed = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--help"],
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout and "--memory-mode" in completed.stdout


# ---------------------------------------------------------------------- shapes


@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="model config not on this host")
def test_default_shapes_from_the_real_model_config(chk):
    dims = chk.read_model_dims(REAL_CONFIG)
    assert dims is not None
    projections = {label: (k, n) for label, k, n in chk.default_projections(dims, 4)}
    assert projections == {
        "qkv_gated": (2560, 3584),  # engine: gate packed next to Q, KV replicated
        "qkv": (2560, 1792),  # the formula of the task brief
        "o_proj": (1536, 2560),
        "gdn_qkvz": (2560, 4096),
        "router": (2560, 512),
    }
    shapes, source = chk.resolve_shapes(chk.parse_args(["--config", str(REAL_CONFIG)]))
    assert len(shapes) == 20 and str(REAL_CONFIG) in source
    assert {s.m for s in shapes} == {17, 1616, 6464, 8080}
    assert shapes[0].key == "17x2560x3584" and shapes[0].label == "qkv_gated"


def test_projection_formulas_on_a_tiny_config(chk):
    projections = {
        label: (k, n) for label, k, n in chk.default_projections(TINY_DIMS, 2)
    }
    assert projections == {
        "qkv_gated": (64, 160),  # (2*8/2 + 2*1) * 16
        "qkv": (64, 96),  # (8 + 2*2) * 16 / 2
        "o_proj": (64, 64),  # K = 8 * 16 / 2
        "gdn_qkvz": (64, 96),  # (2*4*8 + 2*8*8) / 2
        "router": (64, 16),
    }


@pytest.mark.parametrize(
    ("kv", "tp", "expected"),
    [
        (2, 4, (12 + 2) * 256),  # kv < tp: one replicated KV head per rank
        (8, 4, (12 + 4) * 256),  # kv >= tp: split over the ranks
        (4, 4, (12 + 2) * 256),
    ],
)
def test_qkv_gated_width_handles_replicated_kv_heads(chk, kv, tp, expected):
    dims = {**chk.PRODUCTION_MODEL, "num_key_value_heads": kv}
    assert chk.qkv_gated_width(dims, tp) == expected


def test_widths_that_do_not_divide_by_tp_are_an_error(chk):
    with pytest.raises(ValueError, match="not divisible"):
        chk.default_projections({**chk.PRODUCTION_MODEL, "num_attention_heads": 25}, 4)
    with pytest.raises(ValueError, match="multiple of num_key_value_heads"):
        chk.qkv_gated_width(chk.PRODUCTION_MODEL, 3)


def test_model_dims_in_text_config_or_top_level(chk, tmp_path):
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"text_config": TINY_DIMS, "vision_config": {}}))
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(TINY_DIMS))
    assert chk.read_model_dims(nested) == TINY_DIMS
    assert chk.read_model_dims(flat) == TINY_DIMS


def test_missing_or_incomplete_config_falls_back_to_production(chk, tmp_path):
    assert chk.read_model_dims(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert chk.read_model_dims(broken) is None
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"text_config": {"hidden_size": 8}}))
    assert chk.read_model_dims(partial) is None
    args = chk.parse_args(["--config", str(partial), "--m", "17"])
    shapes, source = chk.resolve_shapes(args)
    assert "production defaults" in source
    assert (shapes[0].k, shapes[0].n) == (2560, 3584)


def test_n_override_replaces_the_projections_and_keeps_hidden_as_k(chk):
    shapes = chk.build_shapes(TINY_DIMS, 2, [3, 5], n_override=[10, 20])
    assert [s.key for s in shapes] == ["3x64x10", "5x64x10", "3x64x20", "5x64x20"]
    assert shapes[0].label == "n10"


def test_explicit_shapes_win_and_are_deduplicated(chk):
    shapes = chk.build_shapes({}, 4, [1], explicit=[(1, 2, 3), (4, 5, 6), (1, 2, 3)])
    assert [s.key for s in shapes] == ["1x2x3", "4x5x6"]
    assert {s.label for s in shapes} == {"explicit"}
    args = chk.parse_args(["--shapes", "9,8,7"])
    resolved, source = chk.resolve_shapes(args)
    assert [s.key for s in resolved] == ["9x8x7"] and "explicit" in source


# ------------------------------------------------------------ settings and env


def test_settings_match_the_task(chk):
    settings = chk.build_settings()
    assert list(settings) == [
        "default",
        "wscap4096",
        "wscap16",
        "no_reduced_precision",
        "deterministic",
        "no_lt",
    ]
    assert settings["default"] == {
        "env": {},
        "allow_fp16_reduced_precision_reduction": True,
        "deterministic_algorithms": False,
    }
    assert settings["wscap4096"]["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert settings["wscap16"]["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    assert (
        settings["no_reduced_precision"]["allow_fp16_reduced_precision_reduction"]
        is False
    )
    # torch needs a workspace config before it allows deterministic algorithms.
    assert settings["deterministic"]["deterministic_algorithms"] is True
    assert settings["deterministic"]["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert settings["no_lt"]["env"] == {"DISABLE_ADDMM_CUDA_LT": "1"}


def test_worker_environment_clears_the_managed_variables(chk, monkeypatch):
    monkeypatch.setenv("KEEP_ME", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":1:1")
    monkeypatch.setenv("DISABLE_ADDMM_CUDA_LT", "1")
    env = chk.build_worker_env({})
    assert env["KEEP_ME"] == "1"
    assert "CUBLAS_WORKSPACE_CONFIG" not in env
    assert "DISABLE_ADDMM_CUDA_LT" not in env
    env = chk.build_worker_env({"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert "DISABLE_ADDMM_CUDA_LT" not in env


def test_torch_side_flags_of_a_setting_are_applied_and_reported(chk, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    saved = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )
    settings = chk.build_settings()
    try:
        flags = chk._apply_setting(settings["no_reduced_precision"])
        assert flags["allow_fp16_reduced_precision_reduction"] is False
        assert flags["deterministic_algorithms"] is False
        assert flags["env"] == {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "DISABLE_ADDMM_CUDA_LT": None,
        }
        flags = chk._apply_setting(settings["deterministic"])
        assert flags["allow_fp16_reduced_precision_reduction"] is True
        assert flags["deterministic_algorithms"] is True
        flags = chk._apply_setting(settings["default"])
        assert flags["allow_fp16_reduced_precision_reduction"] is True
        assert flags["deterministic_algorithms"] is False
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = saved[0]
        torch.use_deterministic_algorithms(saved[1])


# ----------------------------------------------------------------- inputs


def test_inputs_are_seeded_scaled_and_shared(chk):
    a1 = chk.make_activations(1234, 64, 256)
    assert a1.dtype == torch.float16 and a1.shape == (64, 256)
    assert torch.equal(a1, chk.make_activations(1234, 64, 256))
    assert not torch.equal(a1, chk.make_activations(1235, 64, 256))
    assert abs(float(a1.float().std()) - 4.0) < 0.3
    w = chk.make_weights(1234, 256, 32)
    assert w.dtype == torch.float16 and w.shape == (32, 256)  # [N, K]
    assert abs(float(w.float().std()) - 256**-0.5) < 0.01
    assert chk.derive_seed(1, "A", 2, 3) == chk.derive_seed(1, "A", 2, 3)
    assert chk.derive_seed(1, "A", 2, 3) != chk.derive_seed(1, "W", 2, 3)
    assert chk.derive_seed(1, "A", 2, 3) != chk.derive_seed(1, "A", 3, 2)
    cache = chk.InputCache(5, torch.device("cpu"))
    assert cache.activations(4, 8) is cache.activations(4, 8)
    assert cache.weights(8, 6) is cache.weights(8, 6)
    assert cache.activations(4, 8).shape == (4, 8)


# ------------------------------------------------------------- byte offsets


@pytest.mark.parametrize("offset", [0, 2, 16, 32, 64, 128, 256, 512, 1024])
def test_the_view_starts_the_requested_bytes_into_an_aligned_buffer(chk, offset):
    values = torch.arange(35, dtype=torch.float16).view(5, 7)
    base = chk.alignment_base([0, 2, 16, 1024])
    placed = chk.place_with_offset(values, offset, base)
    ptr = placed.view.data_ptr()
    assert (ptr - offset) % base == 0  # offset bytes above a base-aligned address
    assert ptr % 256 == offset % 256
    assert placed.offset_bytes == offset


def test_the_view_is_contiguous_keeps_values_and_stays_inside_its_storage(chk):
    values = torch.randn(6, 5).half()
    placed = chk.place_with_offset(values, 1024, 2048)
    assert placed.view.is_contiguous() and placed.view.shape == (6, 5)
    assert placed.view.dtype == torch.float16
    assert torch.equal(placed.view, values)
    start = placed.storage.data_ptr()
    end = start + placed.storage.numel() * placed.storage.element_size()
    view_end = placed.view.data_ptr() + placed.view.numel() * 2
    assert start <= placed.view.data_ptr() and view_end <= end
    # The copy is independent of the source.
    values.zero_()
    assert not torch.equal(placed.view, values)


def test_placement_accepts_non_contiguous_sources_and_other_ranks(chk):
    values = torch.arange(12, dtype=torch.float16).view(3, 4).t()  # not contiguous
    placed = chk.place_with_offset(values, 16, 512)
    assert placed.view.shape == (4, 3) and torch.equal(placed.view, values)
    flat = chk.place_with_offset(torch.arange(5, dtype=torch.float16), 2, 512)
    assert flat.view.shape == (5,) and flat.view.data_ptr() % 256 == 2


def test_the_buffer_around_the_view_is_zeroed(chk):
    placed = chk.place_with_offset(torch.ones(8, dtype=torch.float16), 64, 512)
    assert float(placed.storage.sum()) == 8.0


@pytest.mark.parametrize("offset", [-2, 1, 3, 15])
def test_offsets_must_be_non_negative_multiples_of_the_element_size(chk, offset):
    with pytest.raises(ValueError, match="multiple of the element size"):
        chk.place_with_offset(torch.zeros(4, dtype=torch.float16), offset)


def test_base_must_be_a_multiple_of_the_element_size(chk):
    with pytest.raises(ValueError, match="base"):
        chk.place_with_offset(torch.zeros(4, dtype=torch.float16), 0, 3)


def test_the_alignment_base_is_a_power_of_two_above_twice_the_largest_offset(chk):
    assert chk.alignment_base([0, 16, 1024], [16]) == 2048
    assert chk.alignment_base([0, 2, 4]) == 512
    assert chk.alignment_base([300]) == 1024
    assert chk.alignment_base([]) == 512


# ------------------------------------------------------ alignment scenarios


def test_alignment_scenarios_follow_the_default_offsets(chk):
    scenarios = chk.plan_alignment(chk.DEFAULT_OFFSETS, chk.DEFAULT_B_OFFSETS)
    assert [s.name for s in scenarios] == [
        "A0", "A16", "A32", "A64", "A128", "A256", "A512", "A1024", "B16",
    ]  # fmt: skip
    assert scenarios[0] == chk.AlignScenario("A0", 0, 0)
    assert scenarios[-1] == chk.AlignScenario("B16", 0, 16)
    assert all(s.b_offset == 0 for s in scenarios[:-1])


def test_the_baseline_is_always_first_and_duplicates_collapse(chk):
    scenarios = chk.plan_alignment([64, 16, 16, 2], [0, 32, 32, 16])
    assert [s.name for s in scenarios] == ["A0", "A2", "A16", "A64", "B16", "B32"]
    assert chk.plan_alignment([], [])[0].name == chk.ALIGN_BASELINE


# --------------------------------------------------------- memory scenarios


def test_scenarios_from_a_fake_mem_get_info_with_plenty_of_memory(chk):
    free = 31 * GIB
    plans = chk.plan_scenarios(lambda: (free, 32 * GIB))
    assert [p.name for p in plans] == [
        "whole", "free16GiB", "free8GiB", "free4GiB", "free2GiB", "free1GiB",
        "free512MiB",
    ]  # fmt: skip
    assert all(p.reachable for p in plans)
    assert plans[0].dummy_bytes == 0 and plans[0].target_free_bytes is None
    for plan in plans[1:]:
        assert plan.dummy_bytes == free - plan.target_free_bytes


def test_scenarios_that_cannot_be_reached_are_skipped(chk):
    plans = chk.plan_scenarios(lambda: (6 * GIB, 32 * GIB))
    reachable = {p.name: p.reachable for p in plans}
    assert reachable == {
        "whole": True,
        "free16GiB": False,
        "free8GiB": False,
        "free4GiB": True,
        "free2GiB": True,
        "free1GiB": True,
        "free512MiB": True,
    }
    skipped = next(p for p in plans if p.name == "free8GiB")
    assert skipped.dummy_bytes == 0 and "nothing to take away" in skipped.reason


def test_a_target_within_the_margin_of_the_free_memory_is_the_whole_scenario(chk):
    plan = chk.plan_scenario(16 * GIB + chk.MIN_DUMMY_BYTES - 1, 16 * GIB)
    assert not plan.reachable
    plan = chk.plan_scenario(16 * GIB + chk.MIN_DUMMY_BYTES, 16 * GIB)
    assert plan.reachable and plan.dummy_bytes == chk.MIN_DUMMY_BYTES


def test_scenario_names_and_targets(chk):
    assert chk.scenario_name(None) == "whole"
    assert chk.scenario_name(2 * GIB) == "free2GiB"
    assert chk.scenario_name(512 * MIB) == "free512MiB"
    assert chk.scenario_name(1536 * MIB) == "free1536MiB"
    assert chk.memory_targets([16384, 512]) == {
        "whole": None,
        "free16GiB": 16 * GIB,
        "free512MiB": 512 * MIB,
    }


# ----------------------------------------------------------- hashing and diffs


def test_sha256_is_over_the_raw_bytes(chk):
    t = torch.arange(10, dtype=torch.float16)
    assert chk.sha256_tensor(t) == hashlib.sha256(t.numpy().tobytes()).hexdigest()
    other = t.clone()
    other[3] += 1
    assert chk.sha256_tensor(other) != chk.sha256_tensor(t)
    assert chk.sha256_tensor(t.float()) != chk.sha256_tensor(t)
    m = torch.arange(12, dtype=torch.float32).view(3, 4)
    assert chk.sha256_tensor(m.t()) == chk.sha256_tensor(m.t().contiguous())


def test_max_abs_diff_across_chunks_and_dtypes(chk):
    a = torch.zeros(5, 7, dtype=torch.float16)
    b = a.clone()
    b[4, 6] = 0.5
    b[0, 0] = -0.25
    assert chk.max_abs_diff(a, b) == 0.5
    assert chk.max_abs_diff(a, b, chunk=4) == 0.5
    assert chk.max_abs_diff(a, a.clone()) == 0.0
    assert chk.max_abs_diff(a, b.float()) == 0.5
    with pytest.raises(ValueError, match="shape"):
        chk.max_abs_diff(a, torch.zeros(7, 5))


def test_a_nan_output_is_not_hidden_by_the_diff_helpers(chk):
    a = torch.zeros(4, 6, dtype=torch.float16)
    b = a.clone()
    b[3, 5] = float("nan")
    for chunk in (1 << 24, 5):  # the NaN sits in a later chunk too
        assert math.isnan(chk.max_abs_diff(a, b, chunk=chunk))
        assert math.isnan(chk.max_abs_diff(b, a, chunk=chunk))
    x = torch.ones(4, 3, dtype=torch.float16)
    w = torch.ones(2, 3, dtype=torch.float16)
    bad = torch.full((4, 2), 3.0, dtype=torch.float16)
    bad[3, 1] = float("nan")
    assert math.isnan(chk.reference_max_abs_diff(x, w, bad, chunk_rows=2))
    assert chk._worse(0.5, 0.25) == 0.5 and math.isnan(chk._worse(float("nan"), 1.0))
    assert math.isnan(chk._worse(1.0, float("nan")))


def test_the_shape_summary_keeps_a_nan_diff(chk):
    per = worker_result("align", {S1: ALIGN_SAME})
    per["shapes"][S1]["scenarios"][1]["max_abs_diff_vs_baseline"] = float("nan")
    summary = chk.analyze_setting(entry(align=per), groups=("align",))
    assert math.isnan(summary["shapes"][S1]["max_abs_diff_vs_baseline"])


def test_the_reference_diff_is_exact_on_summable_inputs_and_chunk_independent(chk):
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(-8, 9, (37, 12), generator=generator).half()
    w = torch.randint(-4, 5, (9, 12), generator=generator).half()
    exact = F.linear(x.float(), w.float()).half()  # integers: exactly representable
    assert chk.reference_max_abs_diff(x, w, exact) == 0.0
    off = exact.clone()
    off[36, 8] += 2.0  # in the last chunk
    for rows in (1, 5, 1024):
        assert chk.reference_max_abs_diff(x, w, off, chunk_rows=rows) == 2.0
    with pytest.raises(ValueError, match="does not match"):
        chk.reference_max_abs_diff(x, w, exact[:, :4])


def test_the_reference_diff_of_the_real_fp16_gemm_is_small(chk):
    x = chk.make_activations(1, 33, 64)
    w = chk.make_weights(1, 64, 16)
    out = chk._gemm(x, w)
    assert out.dtype == torch.float16 and out.shape == (33, 16)
    assert 0.0 <= chk.reference_max_abs_diff(x, w, out) < 0.05


def test_the_gemm_is_the_engine_call(chk):
    x = torch.randn(3, 8).half().float().half()
    w = torch.randn(5, 8).half()
    assert torch.equal(chk._gemm(x, w), F.linear(x, w, None))


def test_baseline_store_in_memory_and_on_disk(chk, tmp_path):
    t = torch.arange(6, dtype=torch.float16).view(2, 3)
    memory = chk.BaselineStore(None)
    assert memory.get("k") is None
    memory.put("k", t)
    assert torch.equal(memory.get("k"), t)
    disk = chk.BaselineStore(tmp_path)
    assert disk.get("k") is None
    disk.put("k", t)
    assert (tmp_path / "k.pt").exists()
    # A second process sees it.
    again = chk.BaselineStore(tmp_path).get("k")
    assert torch.equal(again, t) and again.dtype == torch.float16
    (tmp_path / "bad.pt").write_bytes(b"not a tensor")
    assert chk.BaselineStore(tmp_path).get("bad") is None


# ---------------------------------------------------------------- kernel names


def test_kernel_classification_and_description(chk):
    flags = chk.classify_kernels(
        ["volta_h884gemm_fp16_128x128_ldg8_nn", "cutlass_80_splitKreduce_kernel"]
    )
    assert flags == {
        "cutlass": True,
        "splitk": True,
        "gemv": False,
        "sm70": False,
        "volta": True,
    }
    assert chk.classify_kernels(["sm70_xmma_gemv_f16"])["gemv"] is True
    assert chk.classify_kernels(["sm70_xmma_gemv_f16"])["sm70"] is True
    assert not any(chk.classify_kernels([]).values())
    assert chk.describe_kernels(["a", "b"]) == "a | b"
    assert "unavailable" in chk.describe_kernels(None)
    assert chk.describe_kernels([]) == "none seen"


def test_kernel_names_run_the_call_even_when_the_profiler_cannot_start(
    chk, monkeypatch
):
    import torch.profiler

    def broken(*args, **kwargs):
        raise RuntimeError("no CUPTI")

    monkeypatch.setattr(torch.profiler, "profile", broken)
    calls = []
    assert chk._kernel_names(lambda: calls.append(1)) is None
    assert calls == [1]


@pytest.mark.filterwarnings("ignore::UserWarning")  # profiler on a CPU-only host
def test_kernel_names_propagate_an_error_of_the_call(chk):
    def boom():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        chk._kernel_names(boom)


# --------------------------------------------------------------------- verdict


def scen(name, digest, group, repeats=True, status="ok", kernels=None, **extra):
    if status != "ok":
        return {"name": name, "group": group, "status": status}
    record = {
        "name": name,
        "group": group,
        "status": "ok",
        "sha256": [digest, digest, digest if repeats else digest + "x"],
        "repeats_identical": repeats,
        "max_abs_diff_vs_baseline": 0.0 if digest == "aa" else 0.5,
        "max_abs_diff_vs_reference_fp32": 0.01,
        **extra,
    }
    if kernels is not None:
        record["kernels"] = kernels
    return record


def worker_result(group, per_shape, **kwargs):
    """``per_shape`` is ``{shape_key: {scenario name: digest}}``."""
    return {
        "shapes": {
            key: {"scenarios": [scen(n, d, group, **kwargs) for n, d in names.items()]}
            for key, names in per_shape.items()
        }
    }


def entry(align=None, inprocess=None, fresh=None):
    return {
        "align": align,
        "memory_inprocess": inprocess,
        "memory_fresh": fresh or {},
    }


S1, S2 = "17x64x32", "64x64x32"
ALIGN_SAME = {"A0": "aa", "A16": "aa", "B16": "aa"}
ALIGN_DIFF = {"A0": "aa", "A16": "bb", "B16": "aa"}
MEM_SAME = {"whole": "aa", "free8GiB": "aa", "free512MiB": "aa"}
MEM_DIFF = {"whole": "aa", "free8GiB": "aa", "free512MiB": "cc"}


def clean_entry():
    return entry(
        worker_result("align", {S1: ALIGN_SAME, S2: ALIGN_SAME}),
        worker_result("memory_inprocess", {S1: MEM_SAME, S2: MEM_SAME}),
        {
            "whole": worker_result(
                "memory_fresh", {S1: {"whole": "aa"}, S2: {"whole": "aa"}}
            ),
            "free8GiB": worker_result(
                "memory_fresh", {S1: {"free8GiB": "aa"}, S2: {"free8GiB": "aa"}}
            ),
        },
    )


def test_tri_state_helpers(chk):
    assert chk.tri_any([False, True, None]) is True
    assert chk.tri_any([False, None]) is None
    assert chk.tri_any([False, False]) is False
    assert chk.tri_any([]) is None
    assert chk.tri_all([True, False, None]) is False
    assert chk.tri_all([True, None]) is None
    assert chk.tri_all([True, True]) is True
    assert chk.tri_all([]) is None


def test_verdict_when_nothing_depends_on_alignment_or_memory(chk):
    results = {name: clean_entry() for name in chk.build_settings()}
    verdict = chk.compute_verdict(results)
    assert verdict["ALIGN_DEPENDENT"] == "NO"
    assert verdict["MEMORY_DEPENDENT"] == "NO"
    assert verdict["REPEATS_IDENTICAL"] == "YES"
    assert verdict["WSCAP_FIXES"] == {"wscap4096": "N/A", "wscap16": "N/A"}
    assert verdict["NO_REDUCED_PRECISION_FIXES"] == "N/A"
    assert verdict["DETERMINISTIC_FIXES"] == "N/A"
    assert verdict["NO_LT_FIXES"] == "N/A"
    assert set(verdict["SAME_BYTES_AS_DEFAULT"].values()) == {"YES"}
    assert "default" not in verdict["SAME_BYTES_AS_DEFAULT"]
    assert verdict["CROSS_PROCESS_REPRODUCIBLE"] == "YES"
    assert verdict["DEPENDENT_SHAPES"] == []
    s = verdict["settings"]["default"]["shapes"][S1]
    assert s["distinct_output_hashes"] == 1 and s["scenarios_run"] == 8


def test_alignment_dependence_and_which_settings_fix_it(chk):
    def aligned(diff_digest):
        align = {S1: ALIGN_DIFF if diff_digest else ALIGN_SAME, S2: ALIGN_SAME}
        return entry(worker_result("align", align))

    results = {
        "default": aligned(True),
        "wscap4096": aligned(True),  # still dependent
        "wscap16": aligned(False),
        "no_reduced_precision": aligned(False),
        "deterministic": aligned(False),
        "no_lt": aligned(True),
    }
    verdict = chk.compute_verdict(results, groups=("align",))
    assert verdict["ALIGN_DEPENDENT"] == "YES"
    assert verdict["MEMORY_DEPENDENT"] == "NOT_RUN"
    assert verdict["WSCAP_FIXES"] == {"wscap4096": "NO", "wscap16": "YES"}
    assert verdict["NO_REDUCED_PRECISION_FIXES"] == "YES"
    assert verdict["DETERMINISTIC_FIXES"] == "YES"
    assert verdict["NO_LT_FIXES"] == "NO"
    assert verdict["DEPENDENT_SHAPES"] == [S1]  # only the shape that differs
    default = verdict["settings"]["default"]["shapes"]
    assert default[S1]["distinct_output_hashes"] == 2
    assert default[S1]["max_abs_diff_vs_baseline"] == 0.5
    assert default[S2]["align_dependent"] is False


def test_a_b_offset_that_differs_counts_as_alignment_dependence(chk):
    per = {S1: {"A0": "aa", "A16": "aa", "B16": "dd"}}
    verdict = chk.compute_verdict({"default": entry(worker_result("align", per))})
    assert verdict["ALIGN_DEPENDENT"] == "YES"


def test_memory_dependence_in_the_fresh_group_only(chk):
    results = {
        "default": entry(
            worker_result("align", {S1: ALIGN_SAME}),
            worker_result("memory_inprocess", {S1: MEM_SAME}),
            {
                "whole": worker_result("memory_fresh", {S1: {"whole": "aa"}}),
                "free512MiB": worker_result("memory_fresh", {S1: {"free512MiB": "cc"}}),
            },
        ),
        "wscap16": clean_entry(),
    }
    verdict = chk.compute_verdict(results)
    assert verdict["ALIGN_DEPENDENT"] == "NO"
    assert verdict["MEMORY_DEPENDENT"] == "YES"
    assert verdict["WSCAP_FIXES"]["wscap16"] == "YES"
    assert verdict["WSCAP_FIXES"]["wscap4096"] == "NOT_RUN"
    assert verdict["DEPENDENT_SHAPES"] == [S1]
    shape = verdict["settings"]["default"]["shapes"][S1]
    assert shape["memory_dependent"] is True and shape["align_dependent"] is False


def test_memory_dependence_in_the_inprocess_group(chk):
    results = {
        "default": entry(
            worker_result("align", {S1: ALIGN_SAME}),
            worker_result("memory_inprocess", {S1: MEM_DIFF}),
        )
    }
    assert chk.compute_verdict(results)["MEMORY_DEPENDENT"] == "YES"


def test_a_setting_that_changes_the_baseline_bytes_is_reported(chk):
    changed = entry(
        worker_result("align", {S1: {"A0": "ee", "A16": "ee", "B16": "ee"}}),
        worker_result("memory_inprocess", {S1: {"whole": "ee", "free8GiB": "ee"}}),
    )
    results = {"default": clean_entry(), "wscap16": changed, "no_lt": clean_entry()}
    verdict = chk.compute_verdict(results)
    assert verdict["SAME_BYTES_AS_DEFAULT"] == {"wscap16": "NO", "no_lt": "YES"}
    # Consistent inside itself, so it still "fixes" nothing here: N/A.
    assert verdict["WSCAP_FIXES"]["wscap16"] == "N/A"


def test_cross_process_baselines_that_disagree_are_flagged(chk):
    results = {
        "default": entry(
            worker_result("align", {S1: ALIGN_SAME}),
            worker_result("memory_inprocess", {S1: MEM_SAME}),
            {
                "whole": worker_result("memory_fresh", {S1: {"whole": "zz"}}),
                "free8GiB": worker_result("memory_fresh", {S1: {"free8GiB": "zz"}}),
            },
        )
    }
    verdict = chk.compute_verdict(results)
    assert verdict["CROSS_PROCESS_REPRODUCIBLE"] == "NO"
    # Each group is consistent inside itself, so neither dependence is claimed.
    assert verdict["ALIGN_DEPENDENT"] == "NO" and verdict["MEMORY_DEPENDENT"] == "NO"
    only_one = {"default": entry(worker_result("align", {S1: ALIGN_SAME}))}
    assert chk.compute_verdict(only_one)["CROSS_PROCESS_REPRODUCIBLE"] == "UNKNOWN"


def test_repeats_that_disagree_are_reported_separately(chk):
    per = worker_result("align", {S1: ALIGN_SAME}, repeats=False)
    verdict = chk.compute_verdict({"default": entry(per)}, groups=("align",))
    assert verdict["REPEATS_IDENTICAL"] == "NO"
    # The first repeat of every scenario still agrees across scenarios.
    assert verdict["ALIGN_DEPENDENT"] == "NO"


def test_skipped_oom_and_error_scenarios_do_not_count(chk):
    memory = {
        "shapes": {
            S1: {
                "scenarios": [
                    scen("whole", "aa", "memory_inprocess"),
                    scen("free512MiB", "", "memory_inprocess", status="oom"),
                    scen("free16GiB", "", "memory_inprocess", status="skipped"),
                    scen("free1GiB", "", "memory_inprocess", status="error"),
                ]
            }
        }
    }
    verdict = chk.compute_verdict({"default": entry(inprocess=memory)})
    shape = verdict["settings"]["default"]["shapes"][S1]
    assert shape["scenarios_run"] == 1
    assert shape["oom_scenarios"] == ["free512MiB"]
    assert shape["skipped_scenarios"] == ["free16GiB"]
    assert shape["error_scenarios"] == ["free1GiB"]
    # One scenario cannot show a dependence either way.
    assert verdict["MEMORY_DEPENDENT"] == "UNKNOWN"


def test_a_missing_baseline_is_unknown_unless_the_others_already_differ(chk):
    no_base = {"A16": "aa", "A32": "aa"}
    result = worker_result("align", {S1: no_base})
    assert chk.compute_verdict({"default": entry(result)})["ALIGN_DEPENDENT"] == (
        "UNKNOWN"
    )
    differing = worker_result("align", {S1: {"A16": "aa", "A32": "bb"}})
    assert chk.compute_verdict({"default": entry(differing)})["ALIGN_DEPENDENT"] == (
        "YES"
    )


def test_verdict_without_the_default_setting(chk):
    verdict = chk.compute_verdict({"wscap16": clean_entry()})
    assert verdict["ALIGN_DEPENDENT"] == "UNKNOWN"
    assert verdict["MEMORY_DEPENDENT"] == "UNKNOWN"
    assert verdict["WSCAP_FIXES"] == {"wscap4096": "NOT_RUN", "wscap16": "UNKNOWN"}
    assert verdict["SAME_BYTES_AS_DEFAULT"] == {}
    assert verdict["DEPENDENT_SHAPES"] == []


def test_a_failed_worker_leaves_the_verdict_unknown(chk):
    results = {"default": entry(None, None, {}), "wscap16": clean_entry()}
    verdict = chk.compute_verdict(results)
    assert verdict["ALIGN_DEPENDENT"] == "UNKNOWN"
    assert verdict["MEMORY_DEPENDENT"] == "UNKNOWN"
    assert verdict["WSCAP_FIXES"]["wscap16"] == "UNKNOWN"


def test_a_memory_group_that_cannot_compare_makes_the_answer_unknown(chk):
    """Only ``whole`` ran in the fresh group (the others failed or were skipped):
    that group neither shows nor excludes a dependence, so the line must not say NO.
    """
    results = {
        "default": entry(
            worker_result("align", {S1: ALIGN_SAME}),
            worker_result("memory_inprocess", {S1: MEM_SAME}),
            {"whole": worker_result("memory_fresh", {S1: {"whole": "aa"}})},
        )
    }
    verdict = chk.compute_verdict(results)
    assert verdict["MEMORY_DEPENDENT"] == "UNKNOWN"
    assert verdict["ALIGN_DEPENDENT"] == "NO"
    # A difference found anywhere still counts.
    results["default"]["memory_inprocess"] = worker_result(
        "memory_inprocess", {S1: MEM_DIFF}
    )
    assert chk.compute_verdict(results)["MEMORY_DEPENDENT"] == "YES"


def test_records_of_all_workers_of_a_setting_are_merged_per_shape(chk):
    merged = chk.collect_records(
        entry(
            worker_result("align", {S1: ALIGN_SAME, S2: ALIGN_SAME}),
            worker_result("memory_inprocess", {S1: MEM_SAME}),
            {
                "whole": worker_result("memory_fresh", {S1: {"whole": "aa"}}),
                "free8GiB": worker_result("memory_fresh", {S1: {"free8GiB": "aa"}}),
            },
        )
    )
    assert set(merged) == {S1, S2}
    assert [r["name"] for r in merged[S1]["memory_fresh"]] == ["whole", "free8GiB"]
    assert set(merged[S2]) == {"align"}
    assert chk.display_name("memory_fresh", "whole") == "fresh-whole"
    assert chk.display_name("align", "A0") == "A0"


def test_algo_lines_show_the_baseline_and_only_the_scenarios_that_changed(chk):
    base = ["volta_h884gemm_fp16"]
    other = ["volta_h884gemm_fp16_splitK", "splitKreduce_kernel"]
    per = {
        "shapes": {
            S1: {
                "scenarios": [
                    scen("A0", "aa", "align", kernels=base),
                    scen("A16", "aa", "align", kernels=base),
                    scen("A32", "bb", "align", kernels=other),
                ]
            }
        }
    }
    lines = chk.algo_lines({"default": entry(per)})
    assert lines == [
        f"ALGO: default {S1} A0: volta_h884gemm_fp16",
        (
            f"ALGO: default {S1} A32: volta_h884gemm_fp16_splitK | splitKreduce_kernel "
            "(differs from baseline)"
        ),
        f"ALGO_SUMMARY: default {S1}: 2 of 3 scenarios run the baseline kernels",
    ]
    everything = chk.algo_lines({"default": entry(per)}, print_all=True)
    assert len(everything) == 4 and f"ALGO: default {S1} A16: " in everything[1]
    # No profiler data, no lines.
    bare = {"shapes": {S1: {"scenarios": [scen("A0", "aa", "align")]}}}
    assert chk.algo_lines({"default": entry(bare)}) == []


# ----------------------------------------------------------------- subprocess


def test_worker_command_carries_the_shapes_and_round_trips(chk, tmp_path):
    args = chk.parse_args(
        ["--repeats", "2", "--free-mib", "1024", "512", "--offsets", "0,2,16",
         "--b-offsets", "8", "--no-profile", "--cpu-reference"]
    )  # fmt: skip
    shapes = [chk.Shape(17, 2560, 3584, "qkv_gated"), chk.Shape(1616, 1536, 2560)]
    command = chk.build_worker_command(
        "/x/script.py", "memory", "wscap16", shapes, args, "free512MiB", tmp_path
    )
    assert command[1:3] == ["/x/script.py", "--worker"]
    parsed = chk.parse_args(command[2:])
    assert parsed.worker == "memory" and parsed.setting == "wscap16"
    assert parsed.shapes == [(17, 2560, 3584), (1616, 1536, 2560)]
    assert parsed.repeats == 2 and parsed.free_mib == [1024, 512]
    assert parsed.offsets == [0, 2, 16] and parsed.b_offsets == [8]
    assert parsed.no_profile and parsed.cpu_reference
    assert parsed.worker_scenario == "free512MiB"
    assert parsed.worker_baseline_dir == str(tmp_path)
    plain = chk.build_worker_command("/x/s.py", "align", "default", shapes, args)
    assert "--worker-scenario" not in plain and "--worker-baseline-dir" not in plain


def test_result_extraction(chk):
    out = 'warning noise\nRESULT_JSON:{"a": 1}\nRESULT_JSON:{"a": 2}\ntrailing\n'
    assert chk.extract_result(out) == {"a": 2}
    with pytest.raises(ValueError, match="no result"):
        chk.extract_result("nothing here\n")


def test_run_worker_reads_the_result_of_a_real_subprocess(chk):
    code = (
        "import sys; print('noise'); "
        "print('RESULT_JSON:{\"ok\": 7}'); sys.stderr.write('w')"
    )
    assert chk.run_worker([sys.executable, "-c", code], {}) == {"ok": 7}


def test_run_worker_raises_with_the_stderr_tail_on_failure(chk):
    code = "import sys; sys.stderr.write('Traceback: boom\\n'); sys.exit(3)"
    command = [sys.executable, "-c", code, "x", "y", "z", "u"]
    with pytest.raises(RuntimeError, match="worker exited 3") as exc:
        chk.run_worker(command, {})
    assert "boom" in str(exc.value)


# ------------------------------------------------- full workers on CPU tensors


class FakeMemory:
    """mem_get_info for a device with ``free`` bytes and no real allocation."""

    def __init__(self, free: int):
        self.free = free

    def __call__(self, device):
        return self.free, self.free * 2


@pytest.fixture
def cpu_worker(chk, monkeypatch):
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    monkeypatch.setattr(chk, "MIN_DUMMY_BYTES", 1 * MIB)
    memory = FakeMemory(64 * MIB)
    monkeypatch.setattr(chk, "_mem_get_info", memory)
    return memory


SHAPES = "--shapes", "4,8,6", "17,16,12"


def worker_args(chk, kind, *extra, setting="default"):
    return chk.parse_args(
        ["--worker", kind, "--setting", setting, *SHAPES, "--no-profile", *extra]
    )


def shapes_of(chk, args):
    return [chk.Shape(m, k, n) for m, k, n in args.shapes]


def run_align(chk, *extra, setting="default"):
    args = worker_args(chk, "align", *extra, setting=setting)
    return chk.worker_align(args, shapes_of(chk, args))


def run_memory(chk, *extra, setting="default"):
    args = worker_args(chk, "memory", *extra, setting=setting)
    return chk.worker_memory(args, shapes_of(chk, args))


def test_the_alignment_worker_on_cpu(chk, cpu_worker):
    result = run_align(chk, "--offsets", "0,2,16,1024", "--b-offsets", "16,32")
    assert result["worker"] == "align" and result["setting"] == "default"
    assert result["device"] == {"name": "cpu", "capability": None}
    assert list(result["shapes"]) == ["4x8x6", "17x16x12"]
    for shape in result["shapes"].values():
        scenarios = shape["scenarios"]
        assert [s["name"] for s in scenarios] == [
            "A0", "A2", "A16", "A1024", "B16", "B32",
        ]  # fmt: skip
        for s in scenarios:
            assert s["status"] == "ok" and s["group"] == "align"
            assert len(s["sha256"]) == 3 and s["repeats_identical"] is True
            # The CPU GEMM does not read the alignment: every scenario matches A0.
            assert s["max_abs_diff_vs_baseline"] == 0.0
            assert s["max_abs_diff_vs_reference_fp32"] < 5e-2
            assert s["c_data_ptr_mod_256"] in range(256)
            assert s["c_data_ptr_mod_512"] % 256 == s["c_data_ptr_mod_256"]
        assert len({s["sha256"][0] for s in scenarios}) == 1
        by_name = {s["name"]: s for s in scenarios}
        assert by_name["A2"]["a_data_ptr_mod_256"] == 2
        assert by_name["A16"]["a_data_ptr_mod_256"] == 16
        assert by_name["A1024"]["a_data_ptr_mod_256"] == 0
        assert by_name["B32"]["b_data_ptr_mod_256"] == 32
        assert by_name["B32"]["a_offset_bytes"] == 0
        assert by_name["B32"]["b_offset_bytes"] == 32
        assert by_name["A0"]["a_data_ptr_mod_256"] == 0
        assert by_name["A0"]["b_data_ptr_mod_256"] == 0
    verdict = chk.compute_verdict({"default": entry(align=result)}, groups=("align",))
    assert verdict["ALIGN_DEPENDENT"] == "NO"


def test_an_alignment_dependent_gemm_is_reported_not_hidden(
    chk, cpu_worker, monkeypatch
):
    real = chk._gemm

    def depends_on_alignment(x, w):
        out = real(x, w)
        if x.data_ptr() % 32 == 16:  # the 16-byte-aligned-only class
            out = out + torch.tensor(2.0**-9, dtype=out.dtype)
        return out

    monkeypatch.setattr(chk, "_gemm", depends_on_alignment)
    result = run_align(chk, "--offsets", "0,16,32")
    for shape in result["shapes"].values():
        by_name = {s["name"]: s for s in shape["scenarios"]}
        assert by_name["A0"]["max_abs_diff_vs_baseline"] == 0.0
        assert by_name["A16"]["max_abs_diff_vs_baseline"] > 0.0
        assert by_name["A16"]["sha256"][0] != by_name["A0"]["sha256"][0]
        assert by_name["A32"]["sha256"][0] == by_name["A0"]["sha256"][0]
        assert by_name["A16"]["repeats_identical"] is True
    verdict = chk.compute_verdict({"default": entry(align=result)})
    assert verdict["ALIGN_DEPENDENT"] == "YES"
    assert verdict["DEPENDENT_SHAPES"] == ["4x8x6", "17x16x12"]


def test_the_inprocess_memory_worker_on_cpu(chk, cpu_worker):
    result = run_memory(chk, "--free-mib", "32", "16", "100")
    assert result["mode"] == "inprocess"
    for shape in result["shapes"].values():
        by_name = {s["name"]: s for s in shape["scenarios"]}
        assert list(by_name) == ["whole", "free32MiB", "free16MiB", "free100MiB"]
        # 100 MiB is more than the 64 MiB the fake device has free.
        assert by_name["free100MiB"]["status"] == "skipped"
        assert "nothing to take away" in by_name["free100MiB"]["reason"]
        assert by_name["whole"]["dummy_mib"] == 0
        assert by_name["free32MiB"]["dummy_mib"] == 32
        assert by_name["free16MiB"]["dummy_mib"] == 48
        ok = [s for s in shape["scenarios"] if s["status"] == "ok"]
        assert len(ok) == 3
        assert all(s["group"] == "memory_inprocess" for s in ok)
        assert all(s["max_abs_diff_vs_baseline"] == 0.0 for s in ok)
        assert all(s["a_data_ptr_mod_256"] == 0 for s in ok)
        assert len({s["sha256"][0] for s in ok}) == 1
    verdict = chk.compute_verdict({"default": entry(inprocess=result)})
    assert verdict["MEMORY_DEPENDENT"] == "NO"


def test_a_scenario_dependent_gemm_is_reported_by_the_memory_worker(
    chk, cpu_worker, monkeypatch
):
    real = chk._gemm
    calls = {"n": 0}

    def memory_dependent(x, w):
        calls["n"] += 1
        out = real(x, w)
        # 2 shapes x 3 repeats = the whole scenario; every call after is "tight".
        if calls["n"] > 6:
            out = out + torch.tensor(2.0**-9, dtype=out.dtype)
        return out

    monkeypatch.setattr(chk, "_gemm", memory_dependent)
    result = run_memory(chk, "--free-mib", "32")
    for shape in result["shapes"].values():
        whole, tight = shape["scenarios"]
        assert whole["max_abs_diff_vs_baseline"] == 0.0
        assert tight["max_abs_diff_vs_baseline"] > 0.0
        assert tight["sha256"][0] != whole["sha256"][0]
    verdict = chk.compute_verdict({"default": entry(inprocess=result)})
    assert verdict["MEMORY_DEPENDENT"] == "YES"


def test_the_fresh_worker_runs_one_scenario_and_shares_its_baseline(
    chk, cpu_worker, tmp_path, monkeypatch
):
    whole = run_memory(
        chk, "--free-mib", "32", "--worker-scenario", "whole",
        "--worker-baseline-dir", str(tmp_path),
    )  # fmt: skip
    assert whole["mode"] == "fresh"
    for shape in whole["shapes"].values():
        assert [s["name"] for s in shape["scenarios"]] == ["whole"]
        assert shape["scenarios"][0]["group"] == "memory_fresh"
    assert {p.name for p in tmp_path.iterdir()} == {"4x8x6.pt", "17x16x12.pt"}

    tight = run_memory(
        chk, "--free-mib", "32", "--worker-scenario", "free32MiB",
        "--worker-baseline-dir", str(tmp_path),
    )  # fmt: skip
    for key, shape in tight["shapes"].items():
        (record,) = shape["scenarios"]
        assert record["name"] == "free32MiB" and record["dummy_mib"] == 32
        assert record["max_abs_diff_vs_baseline"] == 0.0  # read from the file
        assert record["sha256"] == whole["shapes"][key]["scenarios"][0]["sha256"]

    # Without a baseline file the diff is unknown, not zero.
    empty = tmp_path / "empty"
    empty.mkdir()
    lonely = run_memory(
        chk, "--free-mib", "32", "--worker-scenario", "free32MiB",
        "--worker-baseline-dir", str(empty),
    )  # fmt: skip
    record = lonely["shapes"]["4x8x6"]["scenarios"][0]
    assert record["max_abs_diff_vs_baseline"] is None and record["status"] == "ok"

    # A GEMM that changed between the two processes shows up in the diff.
    real = chk._gemm
    monkeypatch.setattr(
        chk,
        "_gemm",
        lambda x, w: real(x, w) + torch.tensor(2.0**-9, dtype=torch.float16),
    )
    changed = run_memory(
        chk, "--free-mib", "32", "--worker-scenario", "free32MiB",
        "--worker-baseline-dir", str(tmp_path),
    )  # fmt: skip
    record = changed["shapes"]["4x8x6"]["scenarios"][0]
    assert record["max_abs_diff_vs_baseline"] > 0.0
    assert record["sha256"] != whole["shapes"]["4x8x6"]["scenarios"][0]["sha256"]


def test_fresh_workers_run_through_main_under_inference_mode(
    chk, cpu_worker, tmp_path, capsys
):
    """main() wraps the worker in torch.inference_mode, as the engine runs."""
    results = {}
    for scenario in ("whole", "free32MiB"):
        code = chk.main(
            ["--worker", "memory", "--setting", "default", "--shapes", "4,8,6",
             "--free-mib", "32", "--worker-scenario", scenario, "--no-profile",
             "--worker-baseline-dir", str(tmp_path)]
        )  # fmt: skip
        assert code == 0
        (line,) = [
            ln
            for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("RESULT_JSON:")
        ]
        results[scenario] = json.loads(line[len("RESULT_JSON:") :])
    (record,) = results["free32MiB"]["shapes"]["4x8x6"]["scenarios"]
    assert record["status"] == "ok" and record["max_abs_diff_vs_baseline"] == 0.0
    assert (tmp_path / "4x8x6.pt").exists()


def test_the_fresh_worker_dummy_is_taken_before_the_first_gemm(
    chk, cpu_worker, monkeypatch
):
    """cuBLAS must meet the reduced memory when it creates its handle."""
    order: list[str] = []
    real_take = chk._take_free_memory
    real_gemm = chk._gemm

    def take(device, target):
        order.append("dummy")
        return real_take(device, target)

    def gemm(x, w):
        order.append("gemm")
        return real_gemm(x, w)

    monkeypatch.setattr(chk, "_take_free_memory", take)
    monkeypatch.setattr(chk, "_gemm", gemm)
    run_memory(chk, "--free-mib", "32", "--worker-scenario", "free32MiB")
    assert order[0] == "dummy" and order.count("dummy") == 1 and "gemm" in order


def test_the_fresh_worker_rejects_an_unknown_scenario(chk, cpu_worker):
    with pytest.raises(SystemExit, match="unknown scenario"):
        run_memory(chk, "--free-mib", "32", "--worker-scenario", "free7MiB")


def test_an_out_of_memory_gemm_is_recorded_not_raised(chk, cpu_worker, monkeypatch):
    real = chk._gemm
    calls = {"n": 0}

    def flaky(x, w):
        calls["n"] += 1
        if calls["n"] > 6:  # the whole scenario (2 shapes x 3 repeats) succeeds
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate")
        return real(x, w)

    monkeypatch.setattr(chk, "_gemm", flaky)
    result = run_memory(chk, "--free-mib", "32")
    whole, small = result["shapes"]["4x8x6"]["scenarios"]
    assert whole["status"] == "ok" and small["status"] == "oom"
    assert "out of memory" in small["reason"].lower() and "sha256" not in small


def test_a_cublas_error_is_recorded_as_an_error_scenario(chk, cpu_worker, monkeypatch):
    real = chk._gemm
    calls = {"n": 0}

    def broken(x, w):
        calls["n"] += 1
        if calls["n"] > 6:
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED when calling cublasGemmEx")
        return real(x, w)

    monkeypatch.setattr(chk, "_gemm", broken)
    result = run_memory(chk, "--free-mib", "32")
    records = result["shapes"]["17x16x12"]["scenarios"]
    assert [s["status"] for s in records] == ["ok", "error"]
    assert "CUBLAS_STATUS_NOT_SUPPORTED" in records[1]["reason"]


def test_a_failing_reference_does_not_lose_the_hashes(chk, cpu_worker, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("reference ran out of memory")

    monkeypatch.setattr(chk, "reference_max_abs_diff", broken)
    result = run_align(chk, "--offsets", "0")
    record = result["shapes"]["4x8x6"]["scenarios"][0]
    assert record["status"] == "ok" and len(record["sha256"]) == 3
    assert "ran out of memory" in record["reference_error"]
    assert "max_abs_diff_vs_reference_fp32" not in record


def test_repeats_follow_the_argument(chk, cpu_worker):
    result = run_align(chk, "--offsets", "0", "--repeats", "5")
    assert len(result["shapes"]["4x8x6"]["scenarios"][0]["sha256"]) == 5


def test_a_nondeterministic_gemm_shows_in_repeats_identical(
    chk, cpu_worker, monkeypatch
):
    real = chk._gemm
    calls = {"n": 0}

    def noisy(x, w):
        calls["n"] += 1
        out = real(x, w)
        return (
            out + torch.tensor(2.0**-9, dtype=out.dtype) if calls["n"] % 3 == 0 else out
        )

    monkeypatch.setattr(chk, "_gemm", noisy)
    result = run_align(chk, "--offsets", "0")
    record = result["shapes"]["4x8x6"]["scenarios"][0]
    assert record["repeats_identical"] is False
    verdict = chk.compute_verdict({"default": entry(align=result)}, groups=("align",))
    assert verdict["REPEATS_IDENTICAL"] == "NO"


def test_profiled_kernels_are_recorded_and_flagged(chk, cpu_worker, monkeypatch):
    names = ["volta_h884gemm_fp16_splitK_kernel", "splitKreduce_kernel"]
    monkeypatch.setattr(
        chk, "_kernel_names", lambda run: (run(), names)[1]
    )  # run() really executes
    args = chk.parse_args(
        ["--worker", "align", "--setting", "default", *SHAPES, "--offsets", "0"]
    )
    result = chk.worker_align(args, shapes_of(chk, args))
    record = result["shapes"]["4x8x6"]["scenarios"][0]
    assert record["kernels"] == names
    assert record["kernel_flags"]["splitk"] and record["kernel_flags"]["volta"]
    assert not record["kernel_flags"]["gemv"]
    assert record["profiled_matches_repeat1"] is True


def test_no_profile_leaves_the_kernel_fields_out(chk, cpu_worker):
    record = run_align(chk, "--offsets", "0")["shapes"]["4x8x6"]["scenarios"][0]
    assert "kernels" not in record and "profiled_matches_repeat1" not in record


def test_the_cpu_reference_is_added_for_the_baseline_only(chk, cpu_worker):
    result = run_align(chk, "--offsets", "0,16", "--cpu-reference")
    a0, a16, *_ = result["shapes"]["4x8x6"]["scenarios"]
    assert 0.0 <= a0["max_abs_diff_vs_reference_cpu_fp32"] < 5e-2
    assert "max_abs_diff_vs_reference_cpu_fp32" not in a16
    plain = run_align(chk, "--offsets", "0")["shapes"]["4x8x6"]["scenarios"][0]
    assert "max_abs_diff_vs_reference_cpu_fp32" not in plain


def test_the_worker_applies_the_flags_of_its_setting(chk, cpu_worker):
    saved = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        result = run_align(chk, "--offsets", "0", setting="no_reduced_precision")
        assert result["torch_flags"]["allow_fp16_reduced_precision_reduction"] is False
        result = run_align(chk, "--offsets", "0", setting="deterministic")
        assert result["torch_flags"]["deterministic_algorithms"] is True
        assert result["setting"] == "deterministic"
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = saved[0]
        torch.use_deterministic_algorithms(saved[1])


def test_a_worker_invocation_prints_one_json_line(chk, cpu_worker, capsys):
    code = chk.main(
        ["--worker", "align", "--setting", "default", "--shapes", "4,8,6",
         "--offsets", "0,16", "--no-profile"]
    )  # fmt: skip
    assert code == 0
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("RESULT_JSON:")
    ]
    assert len(lines) == 1
    result = json.loads(lines[0][len("RESULT_JSON:") :])
    names = [s["name"] for s in result["shapes"]["4x8x6"]["scenarios"]]
    assert names == ["A0", "A16", "B16"]


# --------------------------------------------------------------- the driver


def option(command: list[str], flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def shape_keys_of(command: list[str]) -> list[str]:
    start = command.index("--shapes") + 1
    keys = []
    for token in command[start:]:
        if token.startswith("--"):
            break
        keys.append(token.replace(",", "x"))
    return keys


def fake_worker_factory(calls, fail: tuple | None = None):
    """Worker results for a world where the default setting is both alignment and
    memory dependent on the first shape, wscap4096 only fixes the memory part, and
    ``deterministic`` moves the baseline bits."""

    def run_worker(command: list[str], env: dict[str, str]) -> dict[str, Any]:
        calls.append((command, env))
        worker = option(command, "--worker")
        setting = option(command, "--setting")
        scenario = option(command, "--worker-scenario")
        if fail == (setting, worker, scenario):
            raise RuntimeError("worker exited 1: boom")
        keys = shape_keys_of(command)
        base = "det" if setting == "deterministic" else "ref"

        def digest(key: str, group: str, name: str) -> str:
            d = f"{base}-{key}"
            first = key == keys[0]
            if setting in ("default", "wscap4096") and first and name == "A16":
                d += "-x"
            if setting == "default" and first and group == "memory_fresh":
                d += "-y" if name == "free512MiB" else ""
            return d

        if worker == "align":
            group, names = "align", ["A0", "A16", "B16"]
        elif scenario is None:
            group, names = "memory_inprocess", ["whole", "free16GiB", "free512MiB"]
        else:
            group, names = "memory_fresh", [scenario]
        return {
            "worker": worker,
            "setting": setting,
            "device": {"name": "Fake V100", "capability": [7, 0]},
            "shapes": {
                key: {
                    "scenarios": [
                        {
                            "name": n,
                            "group": group,
                            "status": "ok",
                            "sha256": [digest(key, group, n)] * 3,
                            "repeats_identical": True,
                            "max_abs_diff_vs_baseline": 0.0,
                            "max_abs_diff_vs_reference_fp32": 0.01,
                            "kernels": ["volta_h884gemm_fp16"],
                        }
                        for n in names
                    ]
                }
                for key in keys
            },
        }

    return run_worker


@pytest.fixture
def fake_cuda(chk, monkeypatch):
    """CUDA looks available, but the driver must never open a context itself.

    A context in the driver process would take memory from the GPU the workers
    measure, so anything that initializes CUDA fails the test.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("the driver process must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("get_device_name", "get_device_capability", "mem_get_info", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


DRIVER_ARGS = [
    "--shapes", "17,16,12", "64,16,8", "--free-mib", "16384", "512",
]  # fmt: skip


def test_the_driver_orchestrates_workers_and_prints_the_verdict(
    chk, fake_cuda, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":1:1")  # must not leak in
    calls: list = []
    monkeypatch.setattr(chk, "run_worker", fake_worker_factory(calls))
    out = tmp_path / "report.json"
    code = chk.main([*DRIVER_ARGS, "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "device {'name': 'Fake V100', 'capability': [7, 0]}" in text
    assert "ALIGN_DEPENDENT: YES" in text
    assert "MEMORY_DEPENDENT: YES" in text
    assert "REPEATS_IDENTICAL: YES" in text
    assert "WSCAP_FIXES (wscap4096): NO" in text  # alignment still differs
    assert "WSCAP_FIXES (wscap16): YES" in text
    assert "NO_REDUCED_PRECISION_FIXES: YES" in text
    assert "DETERMINISTIC_FIXES: YES" in text
    assert "NO_LT_FIXES: YES" in text
    assert "SAME_BYTES_AS_DEFAULT (wscap16): YES" in text
    assert "SAME_BYTES_AS_DEFAULT (deterministic): NO" in text
    assert "CROSS_PROCESS_REPRODUCIBLE: YES" in text
    assert "DEPENDENT_SHAPES: 17x16x12 ()" not in text  # labels come from --shapes
    assert "DEPENDENT_SHAPES: 17x16x12 (explicit)" in text
    assert "ALGO: default 17x16x12 A0: volta_h884gemm_fp16" in text
    assert "ALGO_SUMMARY: default 17x16x12: 9 of 9 scenarios" in text
    assert "NOTE: C alignment is engine-controlled" in text
    assert "default 17x16x12 (explicit): 9 scenarios, 3 distinct" in text

    # Per setting: 1 align + 1 in-process + 3 fresh (whole, 16 GiB, 512 MiB).
    assert len(calls) == 6 * 5
    kinds = [(option(c, "--worker"), option(c, "--worker-scenario")) for c, _ in calls]
    assert kinds.count(("align", None)) == 6
    assert kinds.count(("memory", None)) == 6
    assert kinds.count(("memory", "whole")) == 6
    assert kinds.count(("memory", "free512MiB")) == 6
    for command, env in calls:
        setting = option(command, "--setting")
        want = chk.build_settings()[setting]["env"].get("CUBLAS_WORKSPACE_CONFIG")
        assert env.get("CUBLAS_WORKSPACE_CONFIG") == want
        if option(command, "--worker-scenario") is not None:
            assert option(command, "--worker-baseline-dir") is not None
        else:
            assert option(command, "--worker-baseline-dir") is None
    # One baseline directory per setting, shared by that setting's fresh workers.
    dirs = {
        option(c, "--setting"): {
            option(c2, "--worker-baseline-dir")
            for c2, _ in calls
            if option(c2, "--setting") == option(c, "--setting")
            and option(c2, "--worker-baseline-dir")
        }
        for c, _ in calls
    }
    assert all(len(d) == 1 for d in dirs.values())
    assert len({next(iter(d)) for d in dirs.values()}) == 6

    report = json.loads(out.read_text())
    assert [s["key"] for s in report["shapes"]] == ["17x16x12", "64x16x8"]
    assert report["shapes"][0] == {
        "key": "17x16x12", "label": "explicit", "m": 17, "k": 16, "n": 12,
    }  # fmt: skip
    assert report["verdict"]["ALIGN_DEPENDENT"] == "YES"
    assert report["errors"] == [] and report["memory_mode"] == "both"
    assert report["device"] == {"name": "Fake V100", "capability": [7, 0]}
    assert list(report["settings"]) == list(chk.build_settings())
    assert report["offsets"] == [0, 16, 32, 64, 128, 256, 512, 1024]
    assert report["notes"] == chk.NOTES and report["groups"] == ["align", "memory"]
    assert set(report["results"]["default"]) == {
        "align", "memory_inprocess", "memory_fresh",
    }  # fmt: skip
    assert set(report["results"]["default"]["memory_fresh"]) == {
        "whole", "free16GiB", "free512MiB",
    }  # fmt: skip


def test_the_driver_honours_settings_groups_and_memory_mode(
    chk, fake_cuda, monkeypatch, capsys
):
    calls: list = []
    monkeypatch.setattr(chk, "run_worker", fake_worker_factory(calls))
    code = chk.main(
        [*DRIVER_ARGS, "--settings", "default,wscap16", "--groups", "memory",
         "--memory-mode", "inprocess"]
    )  # fmt: skip
    assert code == 0
    assert [(option(c, "--setting"), option(c, "--worker")) for c, _ in calls] == [
        ("default", "memory"),
        ("wscap16", "memory"),
    ]
    text = capsys.readouterr().out
    assert "ALIGN_DEPENDENT: NOT_RUN" in text
    assert "MEMORY_DEPENDENT: NO" in text  # the fake only differs in the fresh group
    assert "WSCAP_FIXES (wscap4096): NOT_RUN" in text
    assert "NO_LT_FIXES: NOT_RUN" in text
    calls.clear()
    chk.main([*DRIVER_ARGS, "--settings", "default", "--memory-mode", "fresh"])
    kinds = [(option(c, "--worker"), option(c, "--worker-scenario")) for c, _ in calls]
    assert kinds == [
        ("align", None), ("memory", "whole"), ("memory", "free16GiB"),
        ("memory", "free512MiB"),
    ]  # fmt: skip


def test_the_driver_prints_every_algo_line_on_request(
    chk, fake_cuda, monkeypatch, capsys
):
    monkeypatch.setattr(chk, "run_worker", fake_worker_factory([]))
    chk.main([*DRIVER_ARGS, "--settings", "default", "--print-all-algo"])
    text = capsys.readouterr().out
    assert "ALGO: default 17x16x12 A16: " in text
    assert "ALGO: default 17x16x12 fresh-free512MiB: " in text


def test_the_driver_exits_1_when_a_worker_fails_but_keeps_going(
    chk, fake_cuda, monkeypatch, tmp_path, capsys
):
    calls: list = []
    monkeypatch.setattr(
        chk,
        "run_worker",
        fake_worker_factory(calls, fail=("wscap16", "memory", "free512MiB")),
    )
    out = tmp_path / "report.json"
    code = chk.main([*DRIVER_ARGS, "--out", str(out)])
    assert code == 1
    assert "ERROR: wscap16 memory fresh free512MiB" in capsys.readouterr().out
    assert len(calls) == 6 * 5  # the run went on after the failure
    report = json.loads(out.read_text())
    assert len(report["errors"]) == 1
    assert "free512MiB" not in report["results"]["wscap16"]["memory_fresh"]
    assert report["verdict"]["ALIGN_DEPENDENT"] == "YES"


def test_the_driver_exits_2_without_cuda_and_starts_no_worker(chk, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("no worker without a GPU")

    monkeypatch.setattr(chk, "run_worker", forbidden)
    assert chk.main([]) == 2
    err = capsys.readouterr().err
    assert "needs a CUDA device" in err and "CUDA_VISIBLE_DEVICES" in err


def test_the_driver_exits_2_when_the_shapes_cannot_be_derived(
    chk, fake_cuda, monkeypatch, tmp_path, capsys
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": TINY_DIMS}))
    monkeypatch.setattr(chk, "run_worker", fake_worker_factory([]))
    assert chk.main(["--config", str(config), "--tp", "3"]) == 2
    assert "not divisible" in capsys.readouterr().err


def test_the_driver_builds_the_default_shapes_from_a_config(
    chk, fake_cuda, monkeypatch, tmp_path, capsys
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": TINY_DIMS}))
    calls: list = []
    monkeypatch.setattr(chk, "run_worker", fake_worker_factory(calls))
    code = chk.main(
        ["--config", str(config), "--tp", "2", "--m", "3", "5", "--settings", "default",
         "--groups", "align"]
    )  # fmt: skip
    assert code == 0
    ((command, _env),) = calls
    assert shape_keys_of(command) == [
        "3x64x160", "5x64x160", "3x64x96", "5x64x96", "3x64x64", "5x64x64",
        "3x64x16", "5x64x16",
    ]  # fmt: skip
    # qkv (64, 96) and gdn_qkvz (64, 96) are the same shape here: listed once.
    assert "8 shapes" in capsys.readouterr().out
