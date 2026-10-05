# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for benchmarks/sm70_moe_unique_experts_bench.py.

The script needs a V100 to measure anything. What runs here: the routing generator
(exactly U unique experts, top-k distinct ids per token), the per-expert byte model
derived from the config shapes (768,000 B at TP4 with FP16 scales, 691,200 B with the
checkpoint's E4M3 scales), the ncu command lines, the ncu CSV parser on fixtures in
both page layouts, the implied-U interpolation, the e2m1 dequantizer, and the
fail-fast exit 2 of ``main`` when no CUDA device is visible.

    cd <worktree> && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= \\
        /data/venvs/1cat-p070/bin/python -m pytest \\
        tests/models/qwen4_exp/test_sm70_moe_unique_experts_bench_cpu.py -q
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks" / "sm70_moe_unique_experts_bench.py"
REAL_CONFIG = Path("/data/models/Qwen3.8-Flash-Next-NVFP4/config.json")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("sm70_moe_unique_experts_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def stub_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"text_config": {"hidden_size": 2560, "moe_intermediate_size": 640,'
        ' "num_experts": 512, "num_experts_per_tok": 10}}'
    )
    return path


# ------------------------------------------------------------------ routing
@pytest.mark.parametrize("rows", [1, 2, 5, 8, 16])
def test_routing_yields_exactly_u_unique_experts(bench, rows):
    for u in (10, 20, 30, 40, 50, 80, 160):
        if not bench.feasible(rows, 10, 512, u):
            with pytest.raises(ValueError):
                bench.make_routing(rows, u, 10, 512, seed=0)
            continue
        ids = bench.make_routing(rows, u, 10, 512, seed=3)
        assert ids.shape == (rows, 10) and ids.dtype == torch.int32
        assert bench.unique_count(ids) == u
        assert int(ids.min()) >= 0 and int(ids.max()) < 512
        for row in ids.tolist():  # no expert twice inside one token
            assert len(set(row)) == 10


def test_default_sweep_feasibility(bench):
    assert [u for u in bench.DEFAULT_U if bench.feasible(5, 10, 512, u)] == [10, 20, 30, 40, 50]
    assert [u for u in bench.DEFAULT_U if bench.feasible(1, 10, 512, u)] == [10]


def test_routing_is_seeded(bench):
    a = bench.make_routing(5, 30, 10, 512, seed=7)
    assert torch.equal(a, bench.make_routing(5, 30, 10, 512, seed=7))
    assert not torch.equal(a, bench.make_routing(5, 30, 10, 512, seed=8))


# ------------------------------------------------------------------ bytes
def test_expert_bytes_from_stub_config(bench, stub_config):
    shapes = bench.read_shapes(stub_config, tp=4)
    assert (shapes.hidden, shapes.inter, shapes.experts, shapes.top_k) == (2560, 160, 512, 10)
    fp16 = shapes.expert_bytes(2)
    assert fp16 == {"w13": 512_000, "w2": 256_000, "total": 768_000}
    assert shapes.expert_bytes(1)["total"] == 691_200  # the checkpoint's E4M3 scales


@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="model config not on this host")
def test_expert_bytes_from_real_config(bench):
    shapes = bench.read_shapes(REAL_CONFIG, tp=4)
    assert "config" in shapes.source
    assert shapes.expert_bytes(2)["total"] == 768_000


def test_unreadable_config_falls_back_to_production(bench, tmp_path):
    shapes = bench.read_shapes(tmp_path / "missing.json", tp=4)
    assert shapes.source == "production defaults"
    assert shapes.expert_bytes(2)["total"] == 768_000


# ------------------------------------------------------------------ ncu
RAW_CSV = """==PROF== Connected to process 1 (python)
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.sum"
"","","","","","","","","","","","byte","byte","second"
"0","1","python","h","w13_kernel","1","7","(256, 1, 1)","(5, 20, 1)","0","7.0","10,240,000","2,048","4.0e-05"
"1","1","python","h","w2_batch_reduce_kernel","1","7","(1024, 1, 1)","(80, 1, 1)","0","7.0","5,120,000","1,024","2.0e-05"
"2","1","python","h","w13_kernel","1","7","(256, 1, 1)","(5, 20, 1)","0","7.0","10,240,000","2,048","4.2e-05"
"3","1","python","h","w2_batch_reduce_kernel","1","7","(1024, 1, 1)","(80, 1, 1)","0","7.0","5,376,000","1,024","1.8e-05"
"""

DETAILS_CSV = """"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","python","h","w13_kernel","1","7","(256, 1, 1)","(5, 20, 1)","0","7.0","Command line profiler metrics","dram__bytes_read.sum","Mbyte","10.24"
"0","1","python","h","w13_kernel","1","7","(256, 1, 1)","(5, 20, 1)","0","7.0","Command line profiler metrics","gpu__time_duration.sum","usecond","40"
"1","1","python","h","w2_kernel","1","7","(1024, 1, 1)","(80, 1, 1)","0","7.0","Command line profiler metrics","dram__bytes_read.sum","Mbyte","5.12"
"1","1","python","h","w2_kernel","1","7","(1024, 1, 1)","(80, 1, 1)","0","7.0","Command line profiler metrics","gpu__time_duration.sum","usecond","20"
"""


def test_parse_raw_page_and_summary(bench):
    launches = bench.parse_ncu_csv(RAW_CSV)
    assert [x["name"] for x in launches] == ["w13_kernel", "w2_batch_reduce_kernel"] * 2
    assert launches[0]["bytes_read"] == 10_240_000 and launches[1]["seconds"] == pytest.approx(2.0e-5)
    row = bench.summarize_ncu(launches, u=20, model_bytes=768_000)
    assert row["calls"] == 2
    assert row["bytes_read"] == pytest.approx(15_488_000)  # median of 15,360,000 and 15,616,000
    assert row["bytes_per_unique_expert"] == pytest.approx(774_400)
    assert row["vs_model_768000"] == pytest.approx(774_400 / 768_000)
    assert row["gbps_ncu"] == pytest.approx(15_488_000 / 6.0e-5 / 1e9, rel=0.02)
    assert "774400" in "\n".join(bench.format_ncu_table([row]))


# Shape of the real ``--csv --page raw --print-units base`` output (Q1.2, 2026-10-05): two ==PROF==
# lines, a header with hundreds of columns (device attributes whose values are text such as
# No-CC / CachePreferNone, some with "__" in the name), a unit row, then one row per launch.
REAL_SHAPE_CSV = """==PROF== Connected to process 1758033 (/usr/bin/python3.12)
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","c2clink__enabled_mask","device__attribute_confidential_computing_mode","launch__func_cache_config","dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.sum"
"","","","","","","","","","","","","","","byte","byte","ns"
"0","1758033","python3.12","127.0.0.1","void <unnamed>::w13_kernel<4, 1>(const __half *, const int *)","1","7","(256, 1, 1)","(5, 20, 1)","0","7.0","0x3","No-CC","CachePreferNone","5,156,128","5,760","28,448"
"1","1758033","python3.12","127.0.0.1","<unnamed>::w2_batch_reduce_kernel(const __half *, const float *, int)","1","7","(1024, 1, 1)","(80, 1, 1)","0","7.0","0x3","No-CC","CachePreferNone","2,591,712","0","13,312"
==PROF== Disconnected from process 1758033
"""


def test_parse_real_shape_raw_page_with_attribute_columns(bench):
    launches, skipped = bench.parse_ncu_csv_counted(REAL_SHAPE_CSV)
    assert skipped == 0 and [x["id"] for x in launches] == [0, 1]
    assert launches[0]["bytes_read"] == 5_156_128 and launches[0]["bytes_write"] == 5_760
    assert launches[0]["seconds"] == pytest.approx(28_448e-9)  # ns from the unit row
    row = bench.summarize_ncu(launches, u=10, model_bytes=768_000, rows=5, skipped=skipped)
    assert row["total_read"] == 5_156_128 + 2_591_712 and row["kernel_rows"] == 2
    assert row["total_write"] == 5_760 and row["total_seconds"] == pytest.approx(41_760e-9)
    assert row["bytes_per_unique_expert"] == pytest.approx(774_784)  # 7,747,840 / 10
    assert row["vs_model_768000"] == pytest.approx(774_784 / 768_000)
    assert set(row["per_kernel"]) == {"w13_kernel", "w2_batch_reduce_kernel"}
    assert row["per_kernel"]["w13_kernel"]["bytes_per_unique_expert"] == pytest.approx(515_612.8)
    assert row["per_kernel"]["w2_batch_reduce_kernel"]["gbps"] == pytest.approx(2_591_712 / 13_312e-9 / 1e9)
    table = "\n".join(bench.format_ncu_table([row], {"w13": 512_000, "w2": 256_000}))
    assert "NCU_PER_KERNEL" in table and "w2_batch_reduce_kernel" in table


def test_parse_counts_na_rows_and_rejects_unknown_unit(bench):
    na = REAL_SHAPE_CSV.replace('"2,591,712"', '"n/a"')
    launches, skipped = bench.parse_ncu_csv_counted(na)
    assert skipped == 1 and [x["id"] for x in launches] == [0]
    with pytest.raises(ValueError, match="unrecognised ncu unit"):
        bench.parse_ncu_csv(REAL_SHAPE_CSV.replace('"byte","byte","ns"', '"byte","byte","furlong"'))


def test_m_from_name(bench):
    assert bench.m_from_name("/d/ncu_M5_U30.csv") == 5 and bench.m_from_name("/d/ncu_M1_U10.csv") == 1
    assert bench.m_from_name("/d/other.csv") is None


def test_parse_details_page_with_units(bench):
    launches = bench.parse_ncu_csv(DETAILS_CSV)
    assert len(launches) == 2
    assert launches[0]["bytes_read"] == pytest.approx(10.24e6)
    assert launches[0]["seconds"] == pytest.approx(40e-6)
    row = bench.summarize_ncu(launches, u=20, model_bytes=768_000)
    assert row["bytes_read"] == pytest.approx(15.36e6)
    assert row["vs_model_768000"] == pytest.approx(1.0)


def test_parse_rejects_text_without_header(bench):
    with pytest.raises(ValueError):
        bench.parse_ncu_csv("==PROF== nothing here\n")


def test_ncu_command_shape(bench):
    cmd = bench.ncu_command("/x/bench.py", 5, 30, "/data/bench", warmup=3, calls=4, seed=0)
    assert "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum" in cmd
    assert "regex:w13_kernel|w2_batch_reduce_kernel" in cmd
    assert "--launch-skip 6 --launch-count 8" in cmd
    assert "ncu_M5_U30.csv" in cmd and "--ncu-target --rows 5 --u-list 30" in cmd
    assert "w2_kernel" in bench.ncu_command("/x/bench.py", 8, 30, "/d", 3, 4, 0)


def test_u_from_name(bench):
    assert bench.u_from_name("/d/ncu_M5_U30.csv") == 30
    assert bench.u_from_name("/d/other.csv") is None


# ------------------------------------------------------------------ statistics
def test_implied_u_interpolates(bench):
    pts = [(10, 30.0), (20, 50.0), (30, 70.0)]
    assert bench.implied_u(pts, 60.0) == pytest.approx(25.0)
    assert bench.implied_u(pts, 20.0) == "<10"
    assert bench.implied_u(pts, 90.0) == ">30"


def test_dequant_e2m1(bench):
    nib = torch.tensor([[0], [1], [7], [8], [9], [15]] + [[2]] * 10, dtype=torch.uint8)
    scales = torch.tensor([[0.5]], dtype=torch.float16)
    got = bench.dequant(nib, scales)[:6, 0].tolist()
    assert got == [0.0, 0.25, 3.0, -0.0, -0.25, -3.0]


# ------------------------------------------------------------------ main
def test_main_without_cuda_exits_2(bench, capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bench.main(["--rows", "5", "--u-list", "10", "20"]) == 2
    out = capsys.readouterr().out
    assert "EXPERT_BYTES" in out and "needs a CUDA device" in out


def test_main_ncu_prints_commands_without_cuda(bench, capsys):
    assert bench.main(["--ncu", "--rows", "5", "1", "--u-list", "10", "20"]) == 0
    out = capsys.readouterr().out
    assert out.count("ncu --metrics") == 3  # M=5 U=10,20 and M=1 U=10 (U=20 infeasible)
    assert "infeasible, skipped [20]" in out


def test_main_parse_ncu(bench, capsys, tmp_path):
    path = tmp_path / "ncu_M5_U20.csv"
    path.write_text(RAW_CSV)
    assert bench.main(["--parse-ncu", str(path)]) == 0
    out = capsys.readouterr().out
    assert "NCU_TABLE" in out and "NCU_PER_KERNEL" in out
