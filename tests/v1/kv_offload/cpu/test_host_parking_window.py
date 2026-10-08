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
    (wt / "vllm").mkdir(parents=True)
    W.make_staging(NS(out=tmp_path, worktree=wt))
    origins = {
        n: str((wt if n == "vllm" else venv / "site-packages") / n / "__init__.py")
        for n in I.PACKAGES
    }
    for name in I.PREBUILTS:
        origins[name] = str(venv / (name + importlib.machinery.EXTENSION_SUFFIXES[0]))
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
    result = I.check(wt, venv, tmp_path / "python-staging")
    assert result["status"] == "PASS" and not result["cuda_initialized"]
    assert imported == list(I.PACKAGES) + list(I.PREBUILTS)
    assert result["staging_contents"] == ["vllm"]
    assert all(result["packages"][n]["__file__"] == p for n, p in origins.items())


@pytest.mark.parametrize(
    "package", ["vllm", "flash_qla", "flashinfer", "torch", "flash_attn_v100"]
)
def test_shadowing_rejected_before_any_import(tmp_path, monkeypatch, package):
    wt, venv, origins, imported = fake_imports(tmp_path, monkeypatch)
    origins[package] = str(tmp_path / "wrong" / package / "__init__.py")
    result = I.check(wt, venv, tmp_path / "python-staging")
    assert result["status"] == "FAIL" and package in result["error"]
    assert imported == []  # Especially: no FlashQLA JIT fallback.


@pytest.mark.parametrize("origin", [None, "/outside/prebuilt.so", "not_native.py"])
@pytest.mark.parametrize("package", I.PREBUILTS)
def test_missing_or_wrong_prebuilt_refuses_without_loading_it(
    tmp_path, monkeypatch, origin, package
):
    wt, venv, origins, imported = fake_imports(tmp_path, monkeypatch)
    origins[package] = str(venv / origin) if origin is not None else None
    result = I.check(wt, venv, tmp_path / "python-staging")
    assert result["status"] == "FAIL" and package not in imported


@pytest.mark.parametrize("extra", ["flash_qla", ".hidden", "__pycache__"])
def test_extra_staging_entry_refused_before_import(tmp_path, monkeypatch, extra):
    wt, venv, _, imported = fake_imports(tmp_path, monkeypatch)
    stage = tmp_path / "python-staging"
    (stage / extra).mkdir()
    result = I.check(wt, venv, stage)
    assert result["status"] == "FAIL" and "staging contents" in result["error"]
    assert imported == []


@pytest.mark.parametrize("replacement", ["missing", "directory", "wrong-target"])
def test_staging_vllm_must_point_to_worktree(tmp_path, monkeypatch, replacement):
    wt, venv, _, imported = fake_imports(tmp_path, monkeypatch)
    stage = tmp_path / "python-staging"
    link = stage / "vllm"
    link.unlink()
    if replacement == "directory":
        link.mkdir()
    elif replacement == "wrong-target":
        link.symlink_to(venv, target_is_directory=True)
    result = I.check(wt, venv, stage)
    assert result["status"] == "FAIL" and "staging" in result["error"]
    assert imported == []


def test_whole_worktree_pythonpath_rejected_in_real_child(tmp_path):
    # Run the actual checker with the OLD launch environment; do not mock
    # find_spec/import_module. Rejection must happen before any package import.
    wt = BENCH.parent
    stage = W.make_staging(NS(out=tmp_path, worktree=wt))
    report = tmp_path / "old-path.json"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(wt),
        PYTHONNOUSERSITE="1",
        PYTHONSAFEPATH="1",
    )
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            str(BENCH / "host_parking_import_check.py"),
            "--worktree",
            str(wt),
            "--venv",
            sys.prefix,
            "--staging",
            str(stage),
            "--out",
            str(report),
        ],
        cwd=stage,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "FAIL" and data["error"].startswith("RuntimeError(")
    assert "flash_qla" in data["error"] and "outside" in data["error"]
    assert data["staging_contents"] == ["vllm"]
    assert Path(data["packages"]["flash_qla"]["spec_origin"]).is_relative_to(wt)
    assert not any("__file__" in p for p in data["packages"].values())


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
    assert cmd[cmd.index("--staging") + 1] == str(a.staging)
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTHONPATH"] == str(a.staging)
    assert kwargs["cwd"] == a.staging and seconds == 120
    assert logfile == tmp_path / "off1.imports.log"


@pytest.mark.parametrize(
    "failure", ["correct", "unexpected-pass", "wrong-error", "already-imported"]
)
def test_preflight_negative_control_requires_precise_failure(
    tmp_path, monkeypatch, failure
):
    a = NS(
        out=tmp_path,
        worktree=tmp_path / "wt",
        python=Path("/venv/bin/python"),
        staging=tmp_path / "stage",
    )
    calls = []
    report = {
        "status": "FAIL",
        "error_type": "RuntimeError",
        "error": "RuntimeError('flash_qla origin is outside venv')",
        "packages": {},
    }
    if failure == "unexpected-pass":
        report["status"] = "PASS"
    elif failure == "wrong-error":
        report["error"] = "RuntimeError('staging contents wrong')"
    elif failure == "already-imported":
        report["packages"] = {"flash_qla": {"__file__": "bad.py"}}

    def command(cmd, env, path, seconds, **kw):
        calls.append((env, kw))
        if kw.get("expected_rc") == 1:
            Path(cmd[-1]).write_text(json.dumps(report))

    monkeypatch.setattr(W, "timed_command", command)
    if failure == "correct":
        W.import_preflight(a, "raw", negative_control=True)
    else:
        with pytest.raises(RuntimeError, match="old PYTHONPATH"):
            W.import_preflight(a, "raw", negative_control=True)
    assert len(calls) == 2
    assert calls[0][0]["PYTHONPATH"] == str(a.staging)
    assert calls[1][0]["PYTHONPATH"] == str(a.worktree)
    assert calls[1][0]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[1][1] == {"cwd": a.staging, "expected_rc": 1}


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

    def refuse(args, label, *, negative_control):
        assert label == "raw"
        assert negative_control is True
        assert list(args.staging.iterdir()) == [args.staging / "vllm"]
        raise RuntimeError("bad package origin")

    monkeypatch.setattr(W, "import_preflight", refuse)
    monkeypatch.setattr(
        W, "timed_command", lambda *a, **kw: pytest.fail("probe started")
    )
    with pytest.raises(RuntimeError, match="bad package origin"):
        W.run(a)


def test_arm_budgets_and_outer_cap_cover_selected_inner_deadlines():
    a = NS(arms=["on1", *W.FAULT_ARMS], skip_raw=True)
    assert W.arm_timeout(a, "on1") == 5400
    assert W.arm_timeout(a, "on2") == 2700
    assert W.arm_timeout(a, W.FAULT_ARMS[0]) == 1500
    assert W.window_budget(a) == 12360
    assert W.window_budget(a) > sum(W.arm_timeout(a, x) + 1140 for x in a.arms)
    a.on1_timeout_s = 6000
    assert W.window_budget(a) == 12960


def test_window_deadline_is_not_swallowed_as_request_failure(monkeypatch):
    def alarm(*a, **kw):
        raise W.WindowInterrupted("deadline")

    monkeypatch.setattr(W.urllib.request, "urlopen", alarm)
    with pytest.raises(W.WindowInterrupted, match="deadline"):
        W.complete(args(), [1], "deadline")


def test_phase_and_parity_checkpoint_survive_interruption(tmp_path, monkeypatch):
    a = NS(out=tmp_path)
    monkeypatch.setattr(W, "reset", lambda *a: None)
    monkeypatch.setattr(W, "prompt", lambda *a: [1])
    count = 0

    def complete(*a, **kw):
        nonlocal count
        count += 1
        if count == 3:
            raise W.WindowInterrupted("deadline")
        return dict(label=str(count), ok=True)

    monkeypatch.setattr(W, "complete", complete)
    with pytest.raises(W.WindowInterrupted), W.phase(a, "on1", "parity"):
        W.suite(a, lambda rows: W.write(tmp_path / "cases.json", rows))
    assert len(json.loads((tmp_path / "cases.json").read_text())) == 2
    assert not (tmp_path / "cases.json.tmp").exists()
    events = [
        json.loads(x) for x in (tmp_path / "phases.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in events] == ["begin", "interrupted"]
    assert all(r["arm"] == "on1" and r["phase"] == "parity" for r in events)


def test_return_checkpoint_survives_next_burst_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "reset", lambda *a: None)
    monkeypatch.setattr(W, "prompt", lambda *a: [1])

    def complete(a, tokens, label, **kw):
        if label == "return-c1-1-0":
            raise W.WindowInterrupted("burst deadline")
        return dict(label=label, ok=True)

    monkeypatch.setattr(W, "complete", complete)
    path = tmp_path / "returns.json"
    with pytest.raises(W.WindowInterrupted, match="burst deadline"):
        W.returns(args(), "on1", True, lambda rows: W.write(path, rows))
    rows = json.loads(path.read_text())
    assert len(rows) == 1 and rows[0]["label"] == "return-c1-0-0"


def test_reused_evidence_is_not_relabelled_or_overwritten(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for name in ("off1.cases.json", "on1.cases.json", "on1.serve.log"):
        (old / name).write_text("[]")
    for r in range(4):
        (old / f"raw-rank{r}.json").write_text('{"status":"PASS"}')
    (old / "off1.evil.py").write_text("do not copy")
    a = NS(reuse_from=old, out=new, arms=["on1", *W.FAULT_ARMS], skip_raw=True)
    W.reuse_evidence(a)
    assert (new / "off1.cases.json").read_bytes() == (
        old / "off1.cases.json"
    ).read_bytes()
    assert not (new / "on1.cases.json").exists()
    assert not (new / "off2.cases.json").exists()
    assert not (new / "off1.evil.py").exists()
    manifest = json.loads((new / "reused-evidence.json").read_text())
    assert len(manifest) == 5 and all(len(r["sha256"]) == 64 for r in manifest)
    assert J.judge(new)["status"] != "PASS"
    with pytest.raises(RuntimeError, match="collision"):
        W.reuse_evidence(a)


def test_skip_raw_requires_all_four_prior_results(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "raw-rank0.json").write_text("{}")
    with pytest.raises(RuntimeError, match="all four"):
        W.reuse_evidence(NS(reuse_from=old, out=new, arms=["on1"], skip_raw=True))


def test_diagnostics_do_not_pass_uninstrumented_interference_gate(
    tmp_path, monkeypatch
):
    W.write(tmp_path / "prereg.json", dict(parking_diagnostics=True))
    for arm in J.ORDER:
        W.write(tmp_path / (arm + ".traffic.json"), {})
        W.write(tmp_path / (arm + ".copy.json"), {})
    monkeypatch.setattr(J, "interference_gate", lambda *a: J.gate("PASS", "test"))
    gate = J.judge(tmp_path)["gates"]["4 interference"]
    assert gate["status"] == "INCONCLUSIVE"
    assert gate["diagnostic_result"]["status"] == "PASS"


def test_partial_fault_checkpoint_does_not_pass(tmp_path):
    W.write(
        tmp_path / "fault-pre_submit.json", {"cases": [dict(label="producer", ok=True)]}
    )
    assert J.failure_gate(tmp_path)["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_fault_evidence_uses_labels_not_record_order(tmp_path, reverse):
    for mode in ("pre_submit", "completed_copy"):
        arm = "fault-" + mode
        labels = [
            "producer",
            "first-good-return",
            "injected-return",
            "failed-key-retry",
            "unrelated",
            "after-reset",
            "after-aborts",
        ]
        rows = [
            dict(
                label=x,
                ok=True,
                ids=[1],
                loads=[dict(success=False)] if x == "injected-return" else [],
            )
            for x in labels
        ]
        W.write(tmp_path / (arm + ".json"), {"cases": rows[::-1] if reverse else rows})
        quota = dict(
            in_use_slots=0, in_use_bytes_per_rank=0, read_write_refs=0, invalid_keys=0
        )
        (tmp_path / (arm + ".serve.log")).write_text(
            f"injected_failure mode={mode}\nfailed_ranks=1\nsnapshot_discard\n"
            "HOST_PARKING invalid_lookup_miss\nHOST_PARKING reset generation=1\n"
            "HOST_PARKING abort req=x pending_loads=1 pending_stores=0\n"
            "HOST_PARKING abort req=y pending_loads=0 pending_stores=1\n"
            + "HOST_PARKING pool_released\n" * 4
            + "HOST_PARKING quota "
            + json.dumps(quota)
            + "\n"
        )
    assert J.failure_gate(tmp_path)["status"] == "PASS"


def test_selected_arm_plan_runs_no_preflight(tmp_path):
    path = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH / "host_parking_window.py"),
            "--out",
            str(path),
            "--arms",
            "on1",
            "fault-pre_submit",
            "--parking-diagnostics",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "PLAN_ONLY" and not path.exists()
    assert data["order"] == ["on1", "fault-pre_submit"]
    cfg = json.loads(data["argv_on"][-1])["kv_connector_extra_config"]
    assert cfg["parking_diagnostics"] is True


@pytest.mark.parametrize(
    "extra",
    [
        ["--arms", "on1", "on1"],
        ["--arms", "fault-pre_submit", "on1"],
        ["--on1-timeout-s", "0"],
        ["--skip-raw"],
    ],
)
def test_invalid_rerun_selection_rejected_before_any_launch(extra):
    result = subprocess.run(
        [sys.executable, str(BENCH / "host_parking_window.py"), *extra],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2


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
