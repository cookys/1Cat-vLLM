# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for benchmarks/sm70_hbm_bandwidth_probe.py.

The script needs a V100 to measure anything. What runs here: size parsing, the GB/s
arithmetic and the read / copy / write / lm_head byte conventions, the timing statistics
and ABBA order (with a fake clock), the verdict and JSON row schema, the lm_head shape
derived from the real config + index (62,080 x 2560 fp16 = 317,849,600 B), the argument
rules, the ncu command list, the fail-fast exit 2 without a CUDA device, and the Triton
read kernel and every arm's self-check run under TRITON_INTERPRET=1 on the CPU.

    cd <worktree> && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= \\
        /data/venvs/1cat-p070/bin/python -m pytest \\
        tests/models/qwen4_exp/test_sm70_hbm_bandwidth_probe_cpu.py -q
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks" / "sm70_hbm_bandwidth_probe.py"
REAL_CONFIG = Path("/data/models/Qwen3.8-Flash-Next-NVFP4/config.json")


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("sm70_hbm_bandwidth_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


# ------------------------------------------------------------------ sizes and arithmetic
def test_parse_size_binary_units(probe):
    assert probe.parse_size("256M") == 256 * 1024**2
    assert probe.parse_size("1G") == 1024**3
    assert probe.parse_size("512MiB") == 512 * 1024**2
    assert probe.parse_size("16777216") == 16 * 1024**2
    assert probe.size_label(1024**3) == "1G" and probe.size_label(256 * 1024**2) == "256M"


@pytest.mark.parametrize("bad", ["", "abc", "1.5G", "4M", "1000001"])
def test_parse_size_rejects(probe, bad):
    with pytest.raises(ValueError):
        probe.parse_size(bad)


def test_gbps_is_decimal(probe):
    assert probe.gbps(1e9, 1000.0) == pytest.approx(1000.0)  # 1 GB in 1 ms
    assert probe.gbps(317_849_600, 473.0) == pytest.approx(671.99, abs=0.1)  # appendix A: 672 GB/s
    assert probe.pct(672.0, probe.SPEC_GBPS) == pytest.approx(74.8, abs=0.05)


def _timing(median, lo=None, hi=None):
    return {"median": median, "p10": lo or median * 0.98, "p90": hi or median * 1.02, "round_spread_pct": 0.5, "samples": 80}


def test_copy_row_counts_oneway_and_traffic(probe):
    size = 1 << 30
    row = probe.make_row("copy", size, size, _timing(3000.0))
    assert row["kind"] == "copy" and row["traffic_factor"] == 2
    assert row["gbps"] == pytest.approx(size / 3e-3 / 1e9)
    assert row["traffic_gbps"] == pytest.approx(2 * row["gbps"])
    read = probe.make_row("read_triton", size, size, _timing(1500.0))
    assert read["traffic_factor"] == 1 and read["traffic_gbps"] == pytest.approx(read["gbps"])
    assert probe.make_row("lmhead_m5", 317_849_600, 317_849_600, _timing(473.0))["kind"] == "lmhead"


ROW_KEYS = {"arm", "kind", "size_bytes", "bytes_counted", "traffic_factor", "median_us", "p10_us", "p90_us", "round_spread_pct",
            "samples", "gbps", "gbps_lo", "gbps_hi", "traffic_gbps", "note"}


def _rows(probe):
    size = 1 << 30
    return [
        probe.make_row("read_torch", size, size, _timing(1700.0)),
        probe.make_row("read_triton", size, size, _timing(1600.0)),
        probe.make_row("copy", size, size, _timing(3000.0)),
        probe.make_row("clone", size, size, _timing(3100.0)),
        probe.make_row("fill", size, size, _timing(1700.0)),
        probe.make_row("lmhead_m1", 317_849_600, 317_849_600, _timing(473.0)),
    ]


def test_row_schema_and_json_roundtrip(probe):
    rows = _rows(probe)
    assert all(set(r) == ROW_KEYS for r in rows)
    assert json.loads(json.dumps(rows))[0]["arm"] == "read_torch"


def test_ceilings_take_best_traffic_per_kind(probe):
    c = probe.ceilings(_rows(probe))
    assert c["read"]["arm"] == "read_triton"
    assert c["copy"]["arm"] == "copy" and c["copy"]["oneway_gbps"] == pytest.approx(c["copy"]["traffic_gbps"] / 2)
    assert c["lmhead"]["traffic_gbps"] == pytest.approx(671.99, abs=0.1)
    assert c["read"]["pct_of_spec"] == pytest.approx(c["read"]["traffic_gbps"] / 898.048 * 100)
    json.dumps(c)


def test_verdict_lines_format(probe):
    lines = probe.verdict_lines(probe.ceilings(_rows(probe)))
    first = lines[0]
    assert re.match(r"^HBM_CEILING_GBPS read=\d+\.\d copy=\d+\.\d write=\d+\.\d lmhead=\d+\.\d", first)
    assert any(line.startswith("PCT_OF_SPEC read=") for line in lines)
    assert any(line.startswith("COPY_ONEWAY_GBPS") for line in lines)
    assert "n/a" in probe.verdict_lines({})[0]


# ------------------------------------------------------------------ statistics
def test_summarize_timing(probe):
    s = probe.summarize_timing([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    assert s["median"] == pytest.approx(16.0) and s["samples"] == 6
    assert s["round_spread_pct"] == pytest.approx(10.0 / 16.0 * 100)
    with pytest.raises(ValueError):
        probe.summarize_timing([])


def test_abba_order(probe):
    assert probe.abba_order(["a", "b"], 2) == ["a", "b", "b", "a"]
    assert probe.abba_order(["a", "b", "c"], 1) == ["a", "b", "c"]


def test_time_arms_divides_by_calls_and_interleaves(probe):
    order = []
    fake = iter(range(1, 10_000))

    def clock(run):
        run()
        return float(next(fake))

    arms = {"a": (lambda: order.append("a"), 4), "b": (lambda: order.append("b"), 1)}
    out = probe.time_arms(arms, rounds=2, iters=3, warmup=1, clock=clock)
    assert out["a"]["samples"] == 6 and out["b"]["samples"] == 6
    assert out["a"]["median"] < out["b"]["median"] * 2  # per-call: 'a' divided by 4
    timed = order[2:]  # after one warmup run of each
    assert timed == ["a"] * 3 + ["b"] * 3 + ["b"] * 3 + ["a"] * 3


# ------------------------------------------------------------------ lm_head shape
@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="checkpoint config not on this host")
def test_lmhead_shape_from_real_config_and_index(probe):
    shape = probe.read_lmhead_shape(REAL_CONFIG, 4)
    assert (shape["vocab"], shape["hidden"], shape["rows"]) == (248320, 2560, 62080)
    assert shape["bytes"] == probe.LMHEAD_BYTES_EXPECTED == 317_849_600
    assert shape["quant_ignore_lm_head"] is True
    assert shape["index_entry"] == {"lm_head.weight": "model-bf16-00012.safetensors"}


def test_lmhead_shape_stub_and_defaults(probe, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"text_config": {"vocab_size": 1000, "hidden_size": 64}, "quantization_config": {"ignore": ["x"]}}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "f.safetensors", "a": "b"}}))
    shape = probe.read_lmhead_shape(cfg, 4)
    assert shape["rows"] == 250 and shape["bytes"] == 250 * 64 * 2
    assert shape["index_entry"] == {"lm_head.weight": "f.safetensors"} and shape["quant_ignore_lm_head"] is False
    assert probe.read_lmhead_shape(tmp_path / "nope.json", 4)["bytes"] == 317_849_600
    with pytest.raises(ValueError):
        probe.read_lmhead_shape(cfg, 3)


# ------------------------------------------------------------------ arguments
def test_parse_args_defaults(probe):
    a = probe.parse_args([])
    assert a.size_bytes == [256 * 1024**2, 512 * 1024**2, 1024**3]
    assert a.arm_list == ["read_torch", "read_triton", "copy", "clone", "fill", "zero"] and a.do_lmhead
    assert a.iters >= 20 and a.rounds >= 2 and a.sets >= 2 and a.lmhead_rows == [1, 5]


def test_parse_args_arm_selection(probe):
    a = probe.parse_args(["--arms", "read_triton", "clone", "--sizes", "64M"])
    assert a.arm_list == ["read_triton", "clone"] and not a.do_lmhead and a.size_bytes == [64 * 1024**2]
    assert probe.parse_args(["--arms", "lmhead"]).arm_list == []


@pytest.mark.parametrize("argv", [["--sizes", "1M"], ["--sizes", "x"], ["--arms", "bogus"], ["--iters", "5"], ["--sets", "1"],
                                  ["--triton-block", "3000"], ["--rounds", "0"]])
def test_parse_args_rejects(probe, argv):
    with pytest.raises(SystemExit) as exc:
        probe.parse_args(argv)
    assert exc.value.code == 2


def test_ncu_commands_skip_memcpy_arms(probe):
    a = probe.parse_args(["--ncu", "--ncu-dir", "/tmp/o"])
    lines = probe.ncu_commands("/x/probe.py", a.arm_list, a)
    joined = "\n".join(lines)
    assert "ncu_hbm_copy" not in joined and "ncu_hbm_clone" not in joined
    assert "regex:hbm_read_kernel" in joined and "--log-file /tmp/o/ncu_hbm_read_triton.csv" in joined
    assert "--lmhead-rows 5" in joined and all("--ncu-target" in line for line in lines)


# ------------------------------------------------------------------ main without CUDA
def test_main_without_cuda_exits_2(probe, capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert probe.main(["--sizes", "256M"]) == 2
    out = capsys.readouterr().out
    assert "LMHEAD_BYTES" in out and "needs a CUDA device" in out


def test_main_ncu_prints_commands_without_cuda(probe, capsys):
    assert probe.main(["--ncu", "--arms", "read", "--sizes", "1G"]) == 0
    out = capsys.readouterr().out
    assert out.count("ncu --metrics") == 2 and "cudaMemcpy" in out


def test_cli_exit_code_2_without_cuda():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--sizes", "256M"], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "needs a CUDA device" in proc.stdout


# ------------------------------------------------------------------ the arms themselves, on the CPU (Triton interpreter)
INTERP = r"""
import importlib.util, json, sys, torch
spec = importlib.util.spec_from_file_location("p", sys.argv[1]); m = importlib.util.module_from_spec(spec)
sys.modules["p"] = m; spec.loader.exec_module(m)
a = m.parse_args(["--sizes", "8M", "--triton-block", "256", "--triton-iters", "2"])
sw = m.Sweep(8 * 1024 * 1024, 2, "cpu", a)
chk = sw.self_check()
for arm in m.KIND:
    sw.call(arm)
shape = {"rows": 96, "hidden": 64, "bytes": 96 * 64 * 2}
lm = m.LmHead(shape, [1, 5], 2, "cpu")
rel = {mm: lm.self_check(mm, cols=96) for mm in (1, 5)}
print(json.dumps({"chk": chk, "rel": rel}))
"""


def test_arms_and_self_checks_under_triton_interpreter():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"}
    proc = subprocess.run([sys.executable, "-c", INTERP, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    chk = got["chk"]
    assert chk["read_torch_rel"] < 1e-3 and chk["read_triton_rel"] < 1e-3
    assert chk["read_control_moved"] and chk["copy_equal"] and chk["clone_equal"] and chk["fill_ok"] and chk["zero_ok"]
    assert all(v < 1e-2 for v in got["rel"].values())
