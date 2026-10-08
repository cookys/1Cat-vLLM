# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for seven-gate reporting and safe, bounded window commands."""

import importlib.machinery
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

BENCH = Path(__file__).resolve().parents[4] / "benchmarks"
# Standalone scripts are deliberately importable without importing the engine.
sys.path.insert(0, str(BENCH))
import host_parking_import_check as I  # noqa: E402
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


def test_staging_exposes_only_vllm_and_environment_isolated(tmp_path, monkeypatch):
    wt, out = tmp_path / "worktree", tmp_path / "out"
    for name in ("vllm", "flash_qla", "flashinfer"):
        (wt / name).mkdir(parents=True)
    out.mkdir()
    a = NS(worktree=wt, out=out)
    a.staging = W.make_staging(a)
    assert list(a.staging.iterdir()) == [a.staging / "vllm"]
    assert (a.staging / "vllm").resolve() == wt / "vllm"
    monkeypatch.setenv("PYTHONPATH", str(wt))
    monkeypatch.setenv("PYTHONHOME", "/wrong-python")
    monkeypatch.setenv("VLLM_HOST_PARKING_FAULT_INJECT", "1")
    monkeypatch.setenv("VLLM_SM70_P8_ANYTHING", "1")
    env = W.engine_env(a)
    assert env["PYTHONPATH"] == str(a.staging)
    assert env["PYTHONSAFEPATH"] == env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONHOME" not in env and "TRITON_INTERPRET" not in env
    assert "VLLM_HOST_PARKING_FAULT_INJECT" not in env
    assert "VLLM_SM70_P8_ANYTHING" not in env
    assert W.engine_env(a, True)["VLLM_HOST_PARKING_FAULT_INJECT"] == "1"
    with pytest.raises(FileExistsError):
        W.make_staging(a)


def fake_imports(tmp_path, monkeypatch):
    wt, venv = tmp_path / "wt", tmp_path / "venv"
    origins = {
        n: str((wt if n == "vllm" else venv / "site-packages") / n / "__init__.py")
        for n in I.PACKAGES
    }
    origins[I.PREBUILT] = str(
        venv / ("prebuilt" + importlib.machinery.EXTENSION_SUFFIXES[0])
    )
    imported = []

    def load(name):
        imported.append(name)
        return NS(__file__=origins[name])

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(I.sys, "prefix", str(venv))
    monkeypatch.setattr(
        I.importlib.util, "find_spec", lambda n: NS(origin=origins.get(n))
    )
    monkeypatch.setattr(I.importlib, "import_module", load)
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_initialized=lambda: False)))
    return wt, venv, origins, imported


def test_import_check_accepts_only_worktree_vllm_and_venv_dependencies(
    tmp_path, monkeypatch
):
    wt, venv, origins, imported = fake_imports(tmp_path, monkeypatch)
    result = I.check(wt, venv)
    assert result["status"] == "PASS" and not result["cuda_initialized"]
    assert imported == list(I.PACKAGES) + [I.PREBUILT]
    assert all(result["packages"][n]["__file__"] == p for n, p in origins.items())


@pytest.mark.parametrize("package", ["vllm", "flash_qla", "flashinfer", "torch"])
def test_shadowing_rejected_before_any_import(tmp_path, monkeypatch, package):
    wt, venv, origins, imported = fake_imports(tmp_path, monkeypatch)
    origins[package] = str(tmp_path / "wrong" / package / "__init__.py")
    result = I.check(wt, venv)
    assert result["status"] == "FAIL" and package in result["error"]
    assert imported == []  # Especially: no FlashQLA JIT fallback.


@pytest.mark.parametrize("origin", [None, "/outside/prebuilt.so", "not_native.py"])
def test_missing_or_wrong_prebuilt_refuses_without_loading_it(
    tmp_path, monkeypatch, origin
):
    wt, venv, origins, imported = fake_imports(tmp_path, monkeypatch)
    origins[I.PREBUILT] = str(venv / origin) if origin is not None else None
    result = I.check(wt, venv)
    assert result["status"] == "FAIL" and I.PREBUILT not in imported


def test_symlink_escape_is_rejected(tmp_path):
    venv, outside = tmp_path / "venv", tmp_path / "outside"
    venv.mkdir()
    outside.mkdir()
    (venv / "flash_qla").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside"):
        I.checked_origin(
            "flash_qla", str(venv / "flash_qla/__init__.py"), tmp_path, venv
        )


def test_preflight_uses_cpu_staging_and_serving_interpreter(tmp_path, monkeypatch):
    a = NS(
        out=tmp_path,
        worktree=tmp_path / "wt",
        python=Path("/venv/bin/python"),
        staging=tmp_path / "stage",
    )
    calls = []
    monkeypatch.setattr(
        W, "timed_command", lambda *args, **kw: calls.append((args, kw))
    )
    W.import_preflight(a, "off1")
    (cmd, env, logfile, seconds), kwargs = calls[0]
    assert cmd[:2] == [str(a.python), "-P"]
    assert cmd[cmd.index("--venv") + 1] == "/venv"
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTHONPATH"] == str(a.staging)
    assert kwargs["cwd"] == a.staging and seconds == 120
    assert logfile == tmp_path / "off1.imports.log"


def test_server_never_starts_after_failed_preflight(tmp_path, monkeypatch):
    a = NS(port=18037)
    sock = NS(connect_ex=lambda _: 1)
    monkeypatch.setattr(W.socket, "socket", lambda: W.contextlib.nullcontext(sock))

    def refuse(args, label):
        assert label == "off1"
        raise RuntimeError("bad package origin")

    monkeypatch.setattr(W, "import_preflight", refuse)
    monkeypatch.setattr(
        W.subprocess, "Popen", lambda *a, **kw: pytest.fail("server started")
    )
    with pytest.raises(RuntimeError, match="bad package origin"), W.server(a, "off1"):
        pytest.fail("server yielded")


def test_raw_probe_never_starts_after_failed_preflight(tmp_path, monkeypatch):
    a = NS(out=tmp_path / "new-run", worktree=tmp_path / "wt", gpu_blocks=1800)
    (a.worktree / "vllm").mkdir(parents=True)
    monkeypatch.setattr(W.subprocess, "check_output", lambda *a, **kw: "tip")

    def refuse(args, label):
        assert label == "raw"
        assert list(args.staging.iterdir()) == [args.staging / "vllm"]
        raise RuntimeError("bad package origin")

    monkeypatch.setattr(W, "import_preflight", refuse)
    monkeypatch.setattr(
        W, "timed_command", lambda *a, **kw: pytest.fail("probe started")
    )
    with pytest.raises(RuntimeError, match="bad package origin"):
        W.run(a)


@pytest.mark.parametrize("cli", [[], ["--out", "cli-out"], ["--out=cli-out"]])
def test_wrapper_output_plan_and_existing_directory_guard(tmp_path, cli):
    env = dict(
        os.environ,
        M37_PYTHON=sys.executable,
        M37_WORKTREE=str(BENCH.parent),
        M37_OUT=str(tmp_path / "env-out"),
    )
    script = str(BENCH / "host_parking_window.sh")
    result = subprocess.run(
        ["bash", script, *cli],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    path = tmp_path / ("cli-out" if cli else "env-out")
    assert Path(json.loads(result.stdout)["out"]) == path
    assert not path.exists()  # A plan must not reserve the GPU result directory.
    path.mkdir()
    env["M37_GO"] = "1"
    result = subprocess.run(
        ["bash", script, "--execute", *cli],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and "Refuse to overwrite" in result.stderr
    # Direct entrypoint must also refuse before error reporting can overwrite
    # artifacts from an earlier run (without relying on the shell wrapper).
    sentinel = path / "window-error.json"
    sentinel.write_text("old evidence")
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH / "host_parking_window.py"),
            "--execute",
            "--out",
            str(path),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and "refuse to overwrite" in result.stderr
    assert sentinel.read_text() == "old evidence"
