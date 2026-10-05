# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of benchmarks/sm70_routing_dump_observe.py (concurrency observer)."""

import http.server
import importlib.util
import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
spec = importlib.util.spec_from_file_location(
    "routing_observe", os.path.join(ROOT, "benchmarks/sm70_routing_dump_observe.py")
)
observe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observe)

BODY = (
    "# HELP vllm:num_requests_running x\n"
    "# TYPE vllm:num_requests_running gauge\n"
    'vllm:num_requests_running{engine="0",model_name="m"} {n}.0\n'
    'vllm:num_requests_waiting{engine="0",model_name="m"} 9.0\n'
)


def test_parse_running_ignores_waiting_and_takes_the_largest_sample():
    assert observe.parse_running(BODY.replace("{n}", "3")) == 3.0
    two = BODY.replace("{n}", "2") + 'vllm:num_requests_running{engine="1"} 5.0\n'
    assert observe.parse_running(two) == 5.0
    assert observe.parse_running("nothing here") is None


@pytest.fixture()
def metrics_server():
    state = {"n": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = BODY.replace("{n}", str(state["n"])).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield state, f"http://127.0.0.1:{srv.server_address[1]}/metrics"
    srv.shutdown()


def run(url, nominal, out, code):
    return observe.main(
        [
            "--metrics-url",
            url,
            "--nominal",
            str(nominal),
            "--out",
            str(out),
            "--interval",
            "0.05",
            "--",
            sys.executable,
            "-c",
            code,
        ]
    )


def test_reached_concurrency_is_recorded_not_marked_missing(metrics_server, tmp_path):
    state, url = metrics_server
    state["n"] = 4
    out = tmp_path / "run" / "summary.json"
    rc = run(url, 4, out, "import time; time.sleep(0.4)")
    summary = json.loads(out.read_text())
    assert rc == 0
    assert summary["observed_max_active"] == 4
    assert summary["data_missing_for_c"] is None and summary["samples"] >= 2
    assert oct(out.stat().st_mode & 0o777) == "0o600"
    assert "cmd" not in summary  # never store the client argv


def test_unreached_concurrency_is_marked_data_missing(metrics_server, tmp_path):
    state, url = metrics_server
    state["n"] = 3  # 16 clients queued, only 3 ever ran
    out = tmp_path / "summary.json"
    run(url, 16, out, "import time; time.sleep(0.3)")
    summary = json.loads(out.read_text())
    assert summary["observed_max_active"] == 3
    assert summary["data_missing_for_c"] == 16


def test_unreachable_metrics_never_claim_concurrency(tmp_path):
    out = tmp_path / "summary.json"
    rc = run("http://127.0.0.1:1/metrics", 2, out, "raise SystemExit(7)")
    summary = json.loads(out.read_text())
    assert rc == 7 and summary["command_exit_code"] == 7
    assert summary["observed_max_active"] is None
    assert summary["data_missing_for_c"] == 2 and summary["poll_errors"] >= 1
