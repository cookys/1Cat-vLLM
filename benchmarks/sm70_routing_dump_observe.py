# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record the concurrency a bench run really reached next to its routing dump.

Runs a client command and, while it runs, polls the server's Prometheus
``vllm:num_requests_running`` (requests inside a step's batch, not queued
clients).  Writes ``summary.json`` with ``observed_max_active`` and, when the
nominal concurrency was not reached, ``data_missing_for_c``.

    python benchmarks/sm70_routing_dump_observe.py \
        --metrics-url http://127.0.0.1:8001/metrics --nominal 16 \
        --out /data/bench/routing-<stamp>/nospec-c16/summary.json \
        -- python3 scripts/eval/concurrency-curve.py ...

The command's exit code is returned.  Only the exit code of the command is
stored, never its argv (it may carry credentials).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

_METRIC = re.compile(
    r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M
)


def parse_running(text: str) -> float | None:
    """Largest ``vllm:num_requests_running`` sample in a /metrics body."""
    values = [float(v) for v in _METRIC.findall(text)]
    return max(values) if values else None


def poll_loop(url: str, interval: float, stop: threading.Event, state: dict) -> None:
    while not stop.is_set():
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                value = parse_running(resp.read().decode("utf-8", "replace"))
        except Exception:
            value = None
        if value is None:
            state["errors"] += 1
        else:
            state["samples"] += 1
            state["max"] = max(state["max"], int(value))
        stop.wait(interval)


def summarize(state: dict, nominal: int, exit_code: int) -> dict:
    observed = state["max"] if state["samples"] else None
    missing = None
    if observed is None or observed < nominal:
        missing = nominal
    return {
        "nominal_concurrency": nominal,
        "observed_max_active": observed,
        "data_missing_for_c": missing,
        "samples": state["samples"],
        "poll_errors": state["errors"],
        "command_exit_code": exit_code,
        "started_unix": state["start"],
        "ended_unix": time.time(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics")
    parser.add_argument("--nominal", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("a command is required after --")
    state = {"max": 0, "samples": 0, "errors": 0, "start": time.time()}
    stop = threading.Event()
    poller = threading.Thread(
        target=poll_loop, args=(args.metrics_url, args.interval, stop, state)
    )
    poller.start()
    try:
        code = subprocess.call(cmd)
    finally:
        stop.set()
        poller.join(timeout=10)
    summary = summarize(state, args.nominal, code)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fd = os.open(args.out + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(args.out + ".tmp", args.out)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
