# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record host pressure and CADO swap across an owner-controlled vLLM startup.

No CUDA import, sysctl, cache drop, process kill or automatic service stop.
An optional command after '--' is launched once, with the caller's environment.
"""

import argparse
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def key_values(path, *, kib_only=False):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, value, *unit = line.replace(":", "").split()
            if value.isdecimal() and (not kib_only or unit == ["kB"]):
                result[key] = int(value)
    except (OSError, ValueError):
        pass
    return result


def las_processes(names):
    records = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            comm = (path / "comm").read_text().strip()
            if comm not in names:
                continue
            # Field 22 is the start time. Use it to distinguish PID reuse.
            start_ticks = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
            status = key_values(path / "status")
            records[f"{path.name}:{start_ticks}"] = {
                "pid": int(path.name),
                "comm": comm,
                "VmSwap_kib": status.get("VmSwap", 0),
                "VmRSS_kib": status.get("VmRSS", 0),
            }
        except (OSError, IndexError):
            continue
    return records


def snapshot(names, started):
    vmstat = key_values("/proc/vmstat")
    processes = las_processes(names)
    return {
        "kind": "sample",
        "elapsed_s": time.monotonic() - started,
        "unix_s": time.time(),
        "meminfo_kib": key_values("/proc/meminfo", kib_only=True),
        "vmstat": {
            k: v
            for k, v in vmstat.items()
            if k.startswith(
                ("pswp", "pgscan_direct", "pgsteal_direct", "allocstall", "pgmajfault")
            )
        },
        "las": processes,
        "las_swap_kib": sum(p["VmSwap_kib"] for p in processes.values()),
    }


def summarize(samples, ready_s, command_exit):
    first, last = samples[0], samples[-1]
    common = first["las"].keys() & last["las"].keys()
    memkeys = {k for sample in samples for k in sample["meminfo_kib"]}
    return {
        "kind": "summary",
        "duration_s": last["elapsed_s"],
        "ready_s": ready_s,
        "command_exit": command_exit,
        "samples": len(samples),
        "meminfo_min_kib": {
            k: min(s["meminfo_kib"][k] for s in samples if k in s["meminfo_kib"])
            for k in sorted(memkeys)
        },
        "meminfo_max_kib": {
            k: max(s["meminfo_kib"][k] for s in samples if k in s["meminfo_kib"])
            for k in sorted(memkeys)
        },
        "las_swap_before_kib": first["las_swap_kib"],
        "las_swap_after_kib": last["las_swap_kib"],
        "las_swap_peak_kib": max(s["las_swap_kib"] for s in samples),
        "surviving_initial_las_swap_delta_kib": sum(
            last["las"][p]["VmSwap_kib"] - first["las"][p]["VmSwap_kib"] for p in common
        ),
        "initial_las_missing_at_end": sorted(first["las"].keys() - last["las"].keys()),
        "new_las_at_end": sorted(last["las"].keys() - first["las"].keys()),
        "vmstat_delta": {
            k: last["vmstat"][k] - value
            for k, value in first["vmstat"].items()
            if k in last["vmstat"]
        },
        "notes": (
            "Sampled extrema, not a continuous peak; VmSwap excludes "
            "shared-memory swap. Inspect process churn and independent host activity."
        ),
    }


def url_ready(url):
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--ready-url")
    parser.add_argument("--settle-seconds", type=float, default=60)
    parser.add_argument("--process-name", action="append", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        not all(
            math.isfinite(x)
            for x in (args.interval, args.duration, args.settle_seconds)
        )
        or args.interval <= 0
        or args.duration <= 0
        or args.settle_seconds < 0
    ):
        parser.error("interval/duration must be positive; settle must be non-negative")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command and args.ready_url and url_ready(args.ready_url):
        parser.error(
            "Endpoint is already ready; stop the intended test service yourself first"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    names = args.process_name or ["las"]
    started = time.monotonic()
    samples = [snapshot(names, started)]
    ready_s = None
    process = None
    next_probe = 0.0
    with args.out.open("x") as output:
        output.write(
            json.dumps(
                {
                    "kind": "config",
                    "command": command,
                    "interval_s": args.interval,
                    "ready_url": args.ready_url,
                    "process_names": names,
                    "pin_env": {
                        k: v
                        for k, v in os.environ.items()
                        if k.startswith("VLLM_QWEN4EXP_PLE_")
                    },
                }
            )
            + "\n"
        )
        output.write(json.dumps(samples[0]) + "\n")
        output.flush()
        if command:
            process = subprocess.Popen(command)
        try:
            while time.monotonic() - started < args.duration:
                now = time.monotonic() - started
                if process is not None and process.poll() not in (None, 0):
                    break
                if args.ready_url and ready_s is None and now >= next_probe:
                    if url_ready(args.ready_url):
                        ready_s = time.monotonic() - started
                    next_probe = now + 1.0
                if ready_s is not None and now >= ready_s + args.settle_seconds:
                    break
                time.sleep(args.interval)
                sample = snapshot(names, started)
                samples.append(sample)
                output.write(json.dumps(sample) + "\n")
                output.flush()
        except KeyboardInterrupt:
            pass
        finally:
            # A launched command is intentionally not terminated on timeout/Ctrl-C.
            # The owner retains service lifecycle control.
            sample = snapshot(names, started)
            samples.append(sample)
            output.write(json.dumps(sample) + "\n")
            summary = summarize(samples, ready_s, process.poll() if process else None)
            output.write(json.dumps(summary) + "\n")
            print(json.dumps(summary, indent=2))
    return int(
        bool(args.ready_url and ready_s is None)
        or bool(process and process.poll() not in (None, 0))
    )


if __name__ == "__main__":
    raise SystemExit(main())
