# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No processes/signals/GPU: model API exit before descendant cleanup."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
import host_parking_window as W  # noqa: E402


def process(pid, pgid=10, start=123):
    return dict(pid=pid, pgid=pgid, session_id=10, start_ticks=start, name="worker")


def setup(monkeypatch, tmp_path, *, exit_at=0.2, workers_until=0.7):
    clock = NS(now=0.0, killed=False)
    events = []
    monkeypatch.setattr(W, "GRACEFUL_SHUTDOWN_S", 2)
    monkeypatch.setattr(W, "FORCED_SHUTDOWN_S", 1)
    monkeypatch.setattr(W.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(W.time, "sleep", lambda s: setattr(clock, "now", clock.now + s))
    log, report = tmp_path / "serve.log", tmp_path / "shutdown.json"
    log.write_text("")

    class Child:
        pid = 10

        def poll(self):
            return 0 if clock.now >= exit_at or clock.killed else None

        def terminate(self):
            events.append(("parent-term", clock.now))

        def wait(self, timeout):
            assert self.poll() is not None
            events.append(("reap", clock.now))
            return 0

    def scan(_sid):
        assert _sid == 10
        active = {}
        if clock.now < exit_at and not clock.killed:
            active[10] = process(10)
        if clock.now < workers_until and not clock.killed:
            active.update(
                {i: process(i, pgid=11 if i == 14 else 10) for i in range(11, 15)}
            )
        elif not clock.killed:
            log.write_text(
                "".join(
                    f"HOST_PARKING pool_released pid={i} bytes=0 slots=0\n"
                    for i in range(11, 15)
                )
            )
        events.append(("scan", clock.now, sorted(active)))
        return active

    def kill(pgid, sig):
        assert sig == W.signal.SIGKILL
        assert clock.now >= 2  # No group signal while cleanup still has budget.
        events.append(("group-kill", clock.now, pgid))
        clock.killed = True

    monkeypatch.setattr(W, "session_processes", scan)
    monkeypatch.setattr(W.os, "killpg", kill)
    return Child(), clock, events, log, report


def test_api_exit_does_not_end_wait_for_workers(monkeypatch, tmp_path):
    child, clock, events, log, report = setup(monkeypatch, tmp_path)
    W.graceful_shutdown(child, log, report, parking=True)
    data = json.loads(report.read_text())
    assert data["status"] == "graceful"
    assert data["all_rank_pool_release"] is True
    assert data["pool_released_pids"] == [11, 12, 13, 14]
    assert 0.7 <= clock.now < 2
    assert events[1] == ("parent-term", 0.0)
    assert not any(e[0] == "group-kill" for e in events)
    assert len(data["observed_processes"]) == 5


def test_dead_api_still_waits_for_its_workers(monkeypatch, tmp_path):
    child, clock, events, log, report = setup(monkeypatch, tmp_path, exit_at=0)
    W.graceful_shutdown(child, log, report, parking=True)
    assert clock.now >= 0.7
    assert not any(e[0] in ("parent-term", "group-kill") for e in events)
    assert json.loads(report.read_text())["all_rank_pool_release"]


def test_timeout_forces_only_owned_groups_and_keeps_missing_release(
    monkeypatch, tmp_path
):
    child, clock, events, log, report = setup(monkeypatch, tmp_path, workers_until=100)
    W.graceful_shutdown(child, log, report, parking=True)
    data = json.loads(report.read_text())
    assert data["status"] == "forced" and not data["graceful"]
    assert data["all_rank_pool_release"] is False
    assert data["pool_released_pids"] == []
    assert {e[2] for e in events if e[0] == "group-kill"} == {10, 11}
    assert 2 <= clock.now < 3


def test_pid_reuse_at_escalation_is_not_signalled(monkeypatch, tmp_path):
    child, _, events, log, report = setup(monkeypatch, tmp_path, exit_at=0)
    count = 0

    def scan(sid):
        nonlocal count
        count += 1
        # One old identity until escalation, then reused PID; it must not be killed.
        return {11: process(11, start=123 if count <= 21 else 999)}

    monkeypatch.setattr(W, "session_processes", scan)
    with pytest.raises(RuntimeError, match="survived forced shutdown"):
        W.graceful_shutdown(child, log, report, parking=True)
    assert not any(e[0] == "group-kill" for e in events)
    assert "error" in json.loads(report.read_text())


def test_missing_owned_parent_refuses_signal(monkeypatch, tmp_path):
    child, _, events, log, report = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(W, "session_processes", lambda _: {})
    with pytest.raises(RuntimeError, match="not in its owned session"):
        W.graceful_shutdown(child, log, report, parking=True)
    assert not any(e[0] in ("parent-term", "group-kill") for e in events)


def test_proc_scan_excludes_foreign_sessions_zombies_and_gone(tmp_path):
    def stat(pid, sid=10, state="S"):
        d = tmp_path / str(pid)
        d.mkdir()
        fields = [state, "1", "10", str(sid)] + ["0"] * 15 + ["456"]
        d.joinpath("stat").write_text(f"{pid} (worker (name)) " + " ".join(fields))

    stat(11)
    stat(12, sid=99)
    stat(13, state="Z")
    (tmp_path / "14").mkdir()
    found = W.session_processes(10, tmp_path)
    assert list(found) == [11]
    assert found[11]["name"] == "worker (name)"
    assert found[11]["start_ticks"] == 456


def test_command_gives_engine_real_shutdown_budget():
    a = NS(python=Path("/venv/bin/python"), port=18037, gpu_blocks=1800)
    for arm in ("off2", "on2", "fault-pre_submit"):
        cmd = W.command(a, arm, "pre_submit" if arm.startswith("fault") else None)
        assert cmd[cmd.index("--shutdown-timeout") + 1] == "90"


def test_server_finally_disarms_alarm_and_calls_graceful_on_interruption(
    monkeypatch, tmp_path
):
    calls = []
    child = NS(pid=10, poll=lambda: None)
    a = NS(out=tmp_path, staging=tmp_path, port=18037, gpu_blocks=1800)

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 1  # No listener; no real socket used.

    monkeypatch.setattr(W.socket, "socket", Socket)
    monkeypatch.setattr(W, "import_preflight", lambda *a: None)
    monkeypatch.setattr(W, "engine_env", lambda *a: dict(W.FLAGS))
    monkeypatch.setattr(W, "command", lambda *a: ["mock-server"])

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(W.subprocess, "Popen", popen)
    monkeypatch.setattr(W, "http", lambda *a, **k: {})
    monkeypatch.setattr(W, "numa_pools", lambda *a: [])
    monkeypatch.setattr(W.signal, "alarm", lambda s: calls.append(("alarm", s)))

    def shutdown(process, log, report, *, parking):
        assert process is child and log.is_file() and parking
        assert report.name == "on2.shutdown.json"
        calls.append(("graceful",))

    monkeypatch.setattr(W, "graceful_shutdown", shutdown)
    with pytest.raises(W.WindowInterrupted), W.server(a, "on2"):
        raise W.WindowInterrupted("test arm deadline")
    assert calls == [("alarm", 2700), ("alarm", 0), ("graceful",)]
