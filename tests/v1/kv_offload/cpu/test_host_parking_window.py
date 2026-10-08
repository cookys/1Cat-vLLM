# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for seven-gate reporting and safe, bounded window commands."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

BENCH = Path(__file__).resolve().parents[4] / "benchmarks"
# Standalone scripts are deliberately importable without importing the engine.
sys.path.insert(0, str(BENCH))
import host_parking_judge as J  # noqa: E402
import host_parking_window as W  # noqa: E402


def args():
    return NS(python=Path("/venv/bin/python"), port=18037, gpu_blocks=1800)


def cases(delta=0.0, token=1):
    return [
        dict(label=str(i), ok=True, ids=[token], logprobs=[-0.4 + delta])
        for i in range(24)
    ]


def test_plan_arms_only_differ_connector_config():
    a = args()
    off = W.command(a, "off1")
    on = W.command(a, "on1")
    assert on[: len(off)] == off
    assert on[len(off)] == "--kv-transfer-config"
    assert "8001" not in off and "8021" not in off
    assert "--mixed-prefill-step-latency-ms" not in off
    cfg = __import__("json").loads(on[-1])["kv_connector_extra_config"]
    assert cfg["cpu_bytes_to_use"] == 16 << 30
    assert cfg["mamba_state_slots_reference_tokens"] == 32768


def test_fault_command_rank1_second_job():
    import json

    cmd = W.command(args(), "fault-pre_submit", "pre_submit")
    cfg = json.loads(cmd[-1])["kv_connector_extra_config"]
    assert cfg["parking_test_fail_rank"] == 1 and cfg["parking_test_fail_nth"] == 2


def test_missing_evidence_is_not_pass(tmp_path):
    r = J.judge(tmp_path)
    assert len(r["gates"]) == 7 and r["status"] == "INCONCLUSIVE"
    assert not r["production_go"]
    assert all(g["status"] == "INCONCLUSIVE" for g in r["gates"].values())


def test_parity_floor_and_nonzero_floor_do_not_allow_token_mismatch():
    all_cases = {k: cases() for k in J.ORDER}
    assert J.parity_gate(all_cases)["status"] == "PASS"
    all_cases["off2"] = cases(0.01)
    all_cases["on1"] = cases(0.02)
    assert J.parity_gate(all_cases)["status"] == "FAIL"
    all_cases["on1"] = cases(0.005, token=2)
    assert J.parity_gate(all_cases)["status"] == "FAIL"


def returns(n=100, latency=1.0):
    return [
        dict(
            return_concurrency=c,
            ok=True,
            prompt_tokens=32768,
            restored_tokens=28672,
            arrival_to_h2d_observed_s=latency,
        )
        for c in (1, 10)
        for _ in range(n)
    ]


def test_latency_insufficient_miss_and_numerical_thresholds():
    assert J.latency_gate(returns())["status"] == "PASS"
    assert J.latency_gate(returns(99))["status"] == "INCONCLUSIVE"
    assert J.latency_gate(returns(latency=2.1))["status"] == "FAIL"
    r = returns()
    r[0]["restored_tokens"] = 1
    assert J.latency_gate(r)["status"] == "FAIL"


def test_load_join_uses_matching_request_and_all_rank_observation():
    r = [dict(request_id="aa", arrival_ns=100)]
    events = [
        dict(
            request_id="cmpl-aa-0",
            success=True,
            external_tokens=8192,
            all_rank_events_observed_ns=1_000_000_100,
        )
    ]
    assert W.join_loads(r, events)[0]["arrival_to_h2d_observed_s"] == 1
    assert r[0]["restored_tokens"] == 8192


def test_memory_same_pid_identity_and_thresholds():
    assert (
        W.ram_safe([{"las": {"1:10": 0}}, {"las": {"1:10": 128 * 1024 + 1}}]) is False
    )
    assert W.ram_safe([{"las": {"1:10": 0}}, {"las": {"1:11": 128 * 1024 + 1}}]) is True
    assert W.ram_safe([{"las": {}}, {"las": {"1:11": 2 * 1024 * 1024}}]) is False
    assert W.ram_safe([]) is None


def test_percentiles_interpolate():
    assert J.quantile([1, 2, 3, 4], 0.5) == W.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert J.quantile([], 0.99) is None
