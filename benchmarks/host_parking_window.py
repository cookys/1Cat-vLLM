# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated lead-run window. Default is a CPU-only plan. Never uses port 8001.

No GPU/library imports here. Server and raw-byte probe run only with --execute
and M37_GO=1. The external lead wrapper owns production stop/restore and fence.
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client as http_client
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

MODEL = "Qwen3.8-27B-QUASAR-NVFP4"
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")
FAULT_ARMS = ("fault-pre_submit", "fault-completed_copy")
LEGACY_ARM = "legacy-c10"
CELL_ARMS = ORDER + (LEGACY_ARM,)
ALL_ARMS = CELL_ARMS + FAULT_ARMS
DEFAULT_ARMS = ORDER + FAULT_ARMS


class WindowInterrupted(BaseException):
    """Do not let a request's broad Exception handler swallow the deadline."""

    def __init__(self, message, signum=None):
        super().__init__(message)
        self.signum = signum


def arm_timeout(args, arm):
    if arm == LEGACY_ARM:
        return 900  # Total arm envelope, including startup and cleanup reserves.
    if arm in FAULT_ARMS:
        return getattr(args, "fault_timeout_s", 1500)
    if arm == "on1":
        return getattr(args, "on1_timeout_s", 5400)
    return getattr(args, "cell_timeout_s", 2700)


def window_budget(args):
    # Per-server: import 120 + startup 900 + cleanup 120. Raw preflight includes
    # the negative provenance control. Keep the outer cap above the inner sum.
    arms = getattr(args, "arms", DEFAULT_ARMS)
    return (
        240
        + (0 if getattr(args, "skip_raw", False) else 600)
        + sum(
            (0 if arm == LEGACY_ARM else 1140) + arm_timeout(args, arm) for arm in arms
        )
        + 300
    )


@contextlib.contextmanager
def phase(args, arm, name):
    def emit(status):
        with (args.out / "phases.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        arm=arm,
                        phase=name,
                        status=status,
                        epoch_ns=time.time_ns(),
                        monotonic_ns=time.monotonic_ns(),
                    )
                )
                + "\n"
            )

    emit("begin")
    try:
        yield
    except BaseException:
        emit("interrupted")
        raise
    else:
        emit("complete")


SPEC = {
    "method": "dflash",
    "model": "incoai/Qwen3.8-27B-DFlash2",
    "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
    "num_speculative_tokens": 7,
    "kv_cache_dtype": "auto",
    "attention_backend": "FLASH_ATTN_V100",
    "draft_sample_method": "probabilistic",
    "enforce_eager": False,
}
FLAGS = {
    "VLLM_KV_CACHE_LAYOUT": "NHD",
    "VLLM_SERVER_DEV_MODE": "1",
    "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY": "1",
    "VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_EXTRA_PAGES": "2048,4096",
    "VLLM_FLASH_V100_NVFP4_FIRST_CHUNK_FP16": "1",
    "VLLM_FLASH_V100_NVFP4_PREFIX_DECODE_ROWS": "1",
    "VLLM_FLASH_V100_GROUPED_VERIFY_PER_REQUEST_FALLBACK": "1",
    "VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY": "1",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
    "NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/g++-14",
    "TMPDIR": "/data/tmp",
    "HF_HOME": "/data/hf",
}


def write(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def make_staging(args):
    """Expose only the patched engine; leave other packages in the serving venv."""
    stage = args.out / "python-staging"
    stage.mkdir(exist_ok=False)
    source = args.worktree.resolve() / "vllm"
    if not source.is_dir():
        raise RuntimeError(f"missing engine package {source}")
    (stage / "vllm").symlink_to(source, target_is_directory=True)
    return stage.resolve()


def engine_env(args, fault=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "C4140_"))}
    env.update(FLAGS)
    if fault:
        env["VLLM_HOST_PARKING_FAULT_INJECT"] = "1"
    env.pop("TRITON_INTERPRET", None)
    env.pop("PYTHONHOME", None)
    # Do not inherit a developer's worktree, pytest deps or user site-packages.
    env.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        PYTHONPATH=str(args.staging),
        PYTHONNOUSERSITE="1",
        PYTHONSAFEPATH="1",
    )
    return env


def import_preflight(args, label, *, negative_control=False):
    env = engine_env(args)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        str(args.python),
        "-P",
        str(args.worktree / "benchmarks/host_parking_import_check.py"),
        "--worktree",
        str(args.worktree),
        "--venv",
        str(args.python.parent.parent),
        "--staging",
        str(args.staging),
        "--out",
        str(args.out / (label + ".imports.json")),
    ]
    timed_command(cmd, env, args.out / (label + ".imports.log"), 120, cwd=args.staging)
    if negative_control:
        # Run once per window, before the raw GPU probe. Reproduce the old
        # whole-worktree launch and require a provenance failure, not any error.
        negative_out = args.out / "old-path-rejected.json"
        negative_cmd = cmd[:-1] + [str(negative_out)]
        negative_env = dict(env, PYTHONPATH=str(args.worktree))
        timed_command(
            negative_cmd,
            negative_env,
            args.out / "old-path-rejected.log",
            120,
            cwd=args.staging,
            expected_rc=1,
        )
        report = json.loads(negative_out.read_text())
        if (
            report.get("status") != "FAIL"
            or report.get("error_type") != "RuntimeError"
            or "flash_qla origin" not in report.get("error", "")
            or "outside" not in report.get("error", "")
            or any("__file__" in p for p in report.get("packages", {}).values())
        ):
            raise RuntimeError("old PYTHONPATH did not reject FlashQLA before import")


def percentile(xs, q):
    if not xs:
        return None
    a = sorted(xs)
    x = (len(a) - 1) * q
    i = int(x)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (x - i)


def config(diagnostics=False, quota="native", async_admission=False):
    result = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "spec_name": "HostParkingSpec",
            "spec_module_path": "vllm.v1.kv_offload.cpu.parking_spec",
            "host_parking": True,
            "cpu_bytes_to_use": 16 << 30,
            "parking_numa_node": 0,
            "parking_mode": "completed_boundary",
            "offload_prompt_only": True,
            "store_threshold": 0,
            "eviction_policy": "lru",
            "mamba_state_slots_reference_tokens": 32768,
        },
    }
    if diagnostics:
        result["kv_connector_extra_config"]["parking_diagnostics"] = True
    if async_admission:
        result["kv_connector_extra_config"]["host_parking_async_admission"] = True
    if quota == "balanced-32k":
        result["kv_connector_extra_config"]["parking_group_slot_ratios"] = {
            str(g): 2 if g < 6 else 8 if g < 8 else 16 for g in range(9)
        }
    elif quota != "native":
        raise ValueError("unknown parking quota recipe")
    return result


def command(args, arm, fault=None):
    cmd = [
        "taskset",
        "-c",
        "28-54:2,84-110:2",
        "numactl",
        "--preferred=0",
        str(args.python.parent / "vllm"),
        "serve",
        "/data/models/" + MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        MODEL,
        "--trust-remote-code",
        "--dtype",
        "half",
        "--tensor-parallel-size",
        "4",
        "--attention-backend",
        "FLASH_ATTN_V100",
        "--kv-cache-dtype",
        "nvfp4",
        "--max-model-len",
        "262144",
        "--gpu-memory-utilization",
        "0.80",
        "--num-gpu-blocks-override",
        str(args.gpu_blocks),
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        "16",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--block-size",
        "4096",
        "--mamba-block-size",
        "4096",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--seed",
        "0",
        "--speculative-config",
        json.dumps(SPEC),
        "--enable-prompt-tokens-details",
    ]
    if arm.startswith("on") or fault or arm == LEGACY_ARM:
        cfg = config(
            getattr(args, "parking_diagnostics", False) or arm == LEGACY_ARM,
            "native" if arm == LEGACY_ARM else getattr(args, "parking_quota", "native"),
            getattr(args, "async_admission", False),
        )
        if fault:
            cfg["kv_connector_extra_config"].update(
                parking_validation=True,
                parking_test_fail_direction="CPU_to_GPU",
                parking_test_fail_mode=fault,
                parking_test_fail_rank=1,
                parking_test_fail_nth=2,
            )
        cmd += ["--kv-transfer-config", json.dumps(cfg)]
    return cmd


def http(args, path, data=None, timeout=120):
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}{path}",
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def cgroup_oom():
    group = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    path = Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.events"
    data = dict(line.split() for line in path.read_text().splitlines())
    return int(data.get("oom", 0)) + int(data.get("oom_kill", 0))


def numa_pools(log):
    rows = []
    text = Path(log).read_text(errors="replace")
    physical = [
        json.loads(line.split("HOST_PARKING pool_usage ", 1)[1])
        for line in text.splitlines()
        if "HOST_PARKING pool_usage " in line
    ]
    for pid, base, size in set(
        re.findall(
            r"HOST_PARKING registered pid=(\d+) base=([0-9a-f]+) bytes=(\d+)", text
        )
    ):
        address = int(base, 16)
        target = None
        maps = Path(f"/proc/{pid}/maps").read_text()
        for line in maps.splitlines():
            begin, end = [int(x, 16) for x in line.split()[0].split("-")]
            if begin <= address and address + int(size) <= end:
                target = begin
                break
        lines = Path(f"/proc/{pid}/numa_maps").read_text().splitlines()
        line = next((line for line in lines if int(line.split()[0], 16) == target), "")
        nodes = {int(n): int(p) for n, p in re.findall(r"N(\d+)=(\d+)", line)}
        bound = re.search(r"bind:(\d+)", line)
        page = re.search(r"kernelpagesize_kB=(\d+)", line)
        rows.append(
            dict(
                pid=int(pid),
                base=base,
                bytes=int(size),
                physical_pool=next((p for p in physical if p["pid"] == int(pid)), None),
                numa_map=line,
                bound_node=int(bound.group(1)) if bound else None,
                other_node_pages=sum(v for k, v in nodes.items() if k != 0),
                resident_bytes=sum(nodes.values())
                * (int(page.group(1)) * 1024 if page else 4096),
            )
        )
    return rows


def metrics(args):
    raw = http(args, "/metrics", timeout=10)
    counters = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        if "kv_offload_total_" in name and "_created" not in name:
            for direction in ("GPU_to_CPU", "CPU_to_GPU"):
                if direction in name:
                    unit = "bytes" if "total_bytes" in name else "seconds"
                    k = direction + "_" + unit
                    counters[k] = counters.get(k, 0) + float(value)
        elif "num_requests_running" in name or "num_requests_waiting" in name:
            k = "running" if "running" in name else "waiting"
            counters[k] = counters.get(k, 0) + float(value)
    return raw, counters


def settle(args):
    previous = None
    stable_since = time.monotonic()
    limit = time.monotonic() + 120
    while time.monotonic() < limit:
        _, now = metrics(args)
        if now != previous or now.get("running", 0) or now.get("waiting", 0):
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= 6:
            return
        previous = now
        time.sleep(0.5)
    raise TimeoutError("parking transfers or requests did not settle")


def reset(args, external=False):
    settle(args)
    # This API returns 200 even when reset is refused; verify logs/hit telemetry.
    http(
        args,
        "/reset_prefix_cache?reset_running_requests=false&reset_external="
        + str(external).lower(),
        {},
        120,
    )


def prompt(args, length, salt):
    # Only benign synthetic text is tokenized. Reuse token IDs, never real traffic.
    text = f"Parking test document {salt}. The garden has green trees and clear water. "
    tokens = json.loads(http(args, "/tokenize", {"model": MODEL, "prompt": text}))[
        "tokens"
    ]
    return (tokens * math.ceil(length / len(tokens)))[:length]


def complete(args, tokens, label, count=128, temperature=0.0, seed=0, timeout_s=300):
    rid = "m37-" + uuid.uuid4().hex
    payload = {
        "model": MODEL,
        "prompt": tokens,
        "max_tokens": count,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
        "seed": seed,
        "ignore_eos": True,
        "logprobs": 1,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": rid},
    )
    rec = {
        "label": label,
        "request_id": rid,
        "arrival_ns": time.time_ns(),
        "prompt_tokens": len(tokens),
        "ids": [],
        "logprobs": [],
        "usage": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                if data.get("usage"):
                    rec["usage"] = data["usage"]
                for choice in data.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if (ids or choice.get("text")) and "first_ns" not in rec:
                        rec["first_ns"] = time.time_ns()
                    rec["ids"].extend(ids)
                    rec["logprobs"].extend(
                        (choice.get("logprobs") or {}).get("token_logprobs", [])
                    )
        rec["ok"] = len(rec["ids"]) == count and len(rec["logprobs"]) == count
    except Exception as e:
        rec.update(ok=False, error=repr(e))
    rec["end_ns"] = time.time_ns()
    rec["ttft_s"] = (
        (rec["first_ns"] - rec["arrival_ns"]) / 1e9 if "first_ns" in rec else None
    )
    return rec


def suite(args, checkpoint=None):
    rows = []
    for n in (1024, 16384):
        for temp in (0.0, 1.0):
            for seed in range(3):
                reset(args, True)
                rows.append(
                    complete(
                        args,
                        prompt(args, n, f"parity-{n}-{seed}"),
                        f"parity-{n}-{temp}-{seed}",
                        temperature=temp,
                        seed=seed,
                    )
                )
                if checkpoint:
                    checkpoint(rows)
    for chain in range(3):
        reset(args, True)
        base = prompt(args, 32767, f"chain-{chain}")
        for turn in range(4):
            reset(args)  # force a host return, not a GPU-local hit
            tokens = base + prompt(args, turn * 1024 + 1, f"append-{chain}")
            rows.append(complete(args, tokens, f"chain-{chain}-{turn}", seed=chain))
            if checkpoint:
                checkpoint(rows)
    return rows


def returns(args, arm, extensive, checkpoint=None):
    rows = []
    for concurrency, n, repetitions in (
        (1, 262000, 100 if extensive else 1),
        (10, 32768, 10 if extensive else 1),
    ):
        reset(args, True)
        prompts = [prompt(args, n, f"restore-{n}-{i}") for i in range(concurrency)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda x: complete(args, x, "producer", count=1), prompts))
            for trial in range(repetitions):
                reset(args)
                batch = list(
                    pool.map(
                        lambda item, concurrency=concurrency, trial=trial: complete(
                            args,
                            item[1],
                            f"return-c{concurrency}-{trial}-{item[0]}",
                            count=1,
                        ),
                        enumerate(prompts),
                    )
                )
                for row in batch:
                    row["return_concurrency"] = concurrency
                rows.extend(batch)
                if checkpoint:
                    checkpoint(rows)
    return rows


def abort_attempt(args, tokens, logpath, direction):
    # Cancellation is triggered only after log evidence of an in-flight load
    # (or during a long producer for D2H). Gate judges actual pending_* logs.
    rid = "m37-abort-" + uuid.uuid4().hex
    data = json.dumps(
        {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": 2048,
            "stream": True,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode()
    connection = http_client.HTTPConnection("127.0.0.1", args.port, timeout=120)
    start = logpath.stat().st_size
    connection.request(
        "POST",
        "/v1/completions",
        data,
        {"Content-Type": "application/json", "X-Request-Id": rid},
    )
    limit = time.monotonic() + 90
    observed = False
    while time.monotonic() < limit:
        with logpath.open() as f:
            f.seek(start)
            text = f.read()
        tag = "load_submitted" if direction == "H2D" else "store_submitted"
        if any(rid in line and tag in line for line in text.splitlines()):
            observed = True
            break
        time.sleep(0.002)
    connection.close()
    settle(args)
    return {"request_id": rid, "direction": direction, "submit_observed": observed}


def sample_memory():
    sample = {"epoch_ns": time.time_ns(), "las": {}, "meminfo_kib": {}}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.split(":")[0] in ("MemAvailable", "MemFree", "Mlocked", "Unevictable"):
            sample["meminfo_kib"][line.split(":")[0]] = int(line.split()[1])
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.joinpath("comm").read_text().strip() != "las":
                continue
            status = d.joinpath("status").read_text()
            swap = int(re.search(r"VmSwap:\s+(\d+)", status).group(1))
            start = d.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
            sample["las"][d.name + ":" + start] = swap
        except (OSError, AttributeError, IndexError):
            pass
    return sample


def ram_safe(samples):
    if not samples:
        return None
    baseline = samples[0]["las"]
    for s in samples:
        if sum(s["las"].values()) >= 2 * 1024 * 1024:
            return False
        if (
            sum(max(0, value - baseline.get(k, value)) for k, value in s["las"].items())
            > 128 * 1024
        ):
            return False
    return True


class Monitor:
    def __init__(self, path):
        self.path, self.samples, self.stop = path, [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open("w") as f:
            while not self.stop.is_set():
                sample = sample_memory()
                self.samples.append(sample)
                f.write(json.dumps(sample) + "\n")
                f.flush()
                if ram_safe(self.samples) is False:
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
                self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


@contextlib.contextmanager
def server(args, arm, fault=None):
    arm_started = time.monotonic()
    # Refuse any pre-existing listener. Never attach to or stop somebody else's.
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("dedicated test port is already occupied")
    import_preflight(args, arm)
    env = engine_env(args, bool(fault))
    args.parking = arm.startswith("on") or bool(fault) or arm == LEGACY_ARM
    cmd = command(args, arm, fault)
    write(
        args.out / (arm + ".argv.json"),
        {
            "argv": cmd,
            "env": {
                k: env[k]
                for k in set(FLAGS)
                | ({"VLLM_HOST_PARKING_FAULT_INJECT"} if fault else set())
            },
            "pythonpath": str(args.staging),
            "cwd": str(args.staging),
            "gpu_blocks": args.gpu_blocks,
        },
    )
    logpath = args.out / (arm + ".serve.log")
    with logpath.open("w") as log:
        child = subprocess.Popen(
            cmd,
            env=env,
            cwd=args.staging,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            end = time.monotonic() + 900
            if arm == LEGACY_ARM:
                end = min(end, arm_started + 660)
            while time.monotonic() < end:
                if child.poll() is not None:
                    raise RuntimeError(
                        f"{arm} exited during startup: {child.returncode}"
                    )
                try:
                    http(args, "/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError(f"{arm} startup deadline exceeded")
            write(args.out / (arm + ".numa.json"), numa_pools(logpath))
            seconds = arm_timeout(args, arm)
            if arm == LEGACY_ARM:
                # Reserve at most 120 s for outstanding short HTTP work and
                # 120 s for owned-child cleanup within the 15-minute envelope.
                seconds = max(1, int(900 - (time.monotonic() - arm_started) - 240))
            write(
                args.out / (arm + ".deadline.json"),
                dict(seconds=seconds, armed_epoch_ns=time.time_ns()),
            )
            signal.alarm(seconds)
            yield logpath
        finally:
            signal.alarm(0)
            # Only the session created above, and only when lead executes GO.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)


def timed_command(cmd, env, path, seconds, cwd=None, expected_rc=0):
    write(
        str(path) + ".argv.json",
        {
            "argv": cmd,
            "timeout_seconds": seconds,
            "expected_returncode": expected_rc,
            "cwd": None if cwd is None else str(cwd),
            "pythonpath": env.get("PYTHONPATH"),
        },
    )
    with Path(path).open("w") as log:
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=seconds)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
    if rc != expected_rc:
        raise RuntimeError(f"command rc={rc}, expected {expected_rc}; {path}")


def traffic(args, arm):
    path = args.out / (arm + ".traffic.json")
    cmd = [
        str(args.python),
        str(args.repo / "scripts/c4140-ab/team_traffic_workload.py"),
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        MODEL,
        "--users",
        "10",
        "--plan-file",
        str(args.plan),
        "--out",
        str(path),
        "--label",
        "m37-" + arm,
        "--metrics-interval",
        "1",
        "--server-prompt-tokens-details",
        "--timeout-s",
        "1200",
        "--greedy",
    ]
    timed_command(cmd, os.environ.copy(), args.out / (arm + ".traffic.out"), 1800)
    return json.loads(path.read_text())


def event_rows(log):
    rows = []
    for line in Path(log).read_text(errors="replace").splitlines():
        if "HOST_PARKING load_timing " in line:
            rows.append(json.loads(line.split("HOST_PARKING load_timing ", 1)[1]))
    return rows


def diagnostic_misses(log, start_line=0):
    counts = {}
    peak_used = {}
    capacities = {}
    for line in Path(log).read_text(errors="replace").splitlines()[start_line:]:
        if "HOST_PARKING step_stats " not in line:
            continue
        sample = json.loads(line.split("HOST_PARKING step_stats ", 1)[1])
        capacities.update(sample["group_capacity_slots"])
        for g, used in sample["group_slots"].items():
            peak_used[g] = max(used, peak_used.get(g, 0))
        for miss in sample["lookup_misses"]:
            if miss["reason"] == "absent":
                g = str(miss["group"])
                counts[g] = counts.get(g, 0) + miss["count"]
    return dict(
        absent_lookup_counts=counts,
        peak_used_slots=peak_used,
        group_capacity_slots=capacities,
        sw_miss_observed=counts.get("8", 0) > 0 and capacities.get("8") == 91,
    )


def run_legacy(args):
    """Short native-quota control. Never substitutes for a full normal arm."""
    arm = LEGACY_ARM
    rows = []
    start_line = None
    log = args.out / (arm + ".serve.log")
    try:
        with server(args, arm), phase(args, arm, "legacy-c10"):
            reset(args, True)
            prompts = [prompt(args, 32768, f"restore-32768-{i}") for i in range(10)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                producers = list(
                    pool.map(
                        lambda tokens: complete(
                            args, tokens, "legacy-producer", count=1, timeout_s=120
                        ),
                        prompts,
                    )
                )
                write(args.out / (arm + ".producers.json"), producers)
                settle(args)
                start_line = len(log.read_text(errors="replace").splitlines())
                for trial in range(2):
                    reset(args)
                    batch = list(
                        pool.map(
                            lambda item, trial=trial: complete(
                                args,
                                item[1],
                                f"legacy-return-{trial}-{item[0]}",
                                count=1,
                                timeout_s=120,
                            ),
                            enumerate(prompts),
                        )
                    )
                    for row in batch:
                        row["return_concurrency"] = 10
                    rows.extend(batch)
                    join_loads(rows, event_rows(log))
                    write(args.out / (arm + ".returns.json"), rows)
    except WindowInterrupted as e:
        if e.signum != signal.SIGALRM:
            raise
        write(args.out / (arm + ".timeout.json"), dict(error=str(e)))
    result = (
        diagnostic_misses(log, start_line)
        if start_line is not None
        else {"sw_miss_observed": None, "reason": "no return phase reached"}
    )
    result.update(
        attempts=len(rows),
        genuine_restores=sum(
            bool(r.get("ok") and r.get("restored_tokens", 0) >= 28672) for r in rows
        ),
    )
    write(args.out / (arm + ".diagnostic.json"), result)


def join_loads(rows, events):
    for row in rows:
        found = [e for e in events if row["request_id"] in e["request_id"]]
        row["loads"] = found
        successful = [e for e in found if e["success"]]
        row["restored_tokens"] = sum(e["external_tokens"] for e in successful)
        if successful:
            row["arrival_to_h2d_observed_s"] = (
                max(e["all_rank_events_observed_ns"] for e in successful)
                - row["arrival_ns"]
            ) / 1e9
    return rows


def reuse_evidence(args):
    """Copy immutable evidence, never relabel one run as multiple samples."""
    source = getattr(args, "reuse_from", None)
    if source is None:
        return
    selected = set(getattr(args, "arms", DEFAULT_ARMS))
    manifest = []
    for path in sorted(source.iterdir()):
        raw = re.fullmatch(r"raw-rank[0-3]\.json", path.name)
        arm = next((a for a in ALL_ARMS if path.name.startswith(a + ".")), None)
        if not path.is_file() or not (
            (raw and getattr(args, "skip_raw", False))
            or (arm is not None and arm not in selected)
        ):
            continue
        # Results/logs only; never copy executable code or Python overlay links.
        if path.suffix not in (".json", ".log", ".out") and not path.name.endswith(
            (".metrics-before", ".metrics-after")
        ):
            continue
        dest = args.out / path.name
        if dest.exists():
            raise RuntimeError(f"evidence collision: {dest}")
        data = path.read_bytes()
        shutil.copy2(path, dest)
        manifest.append(
            dict(
                source=str(path.resolve()),
                file=path.name,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    write(args.out / "reused-evidence.json", manifest)
    source_prereg = source / "prereg.json"
    write(
        args.out / "reused-run.json",
        dict(
            source=str(source.resolve()),
            source_prereg=json.loads(source_prereg.read_text())
            if source_prereg.exists()
            else None,
            source_prereg_sha256=hashlib.sha256(source_prereg.read_bytes()).hexdigest()
            if source_prereg.exists()
            else None,
            timing_comparable=False,
            note="Cross-window control; driver/venv equivalence and G4 not asserted",
        ),
    )
    if getattr(args, "skip_raw", False) and not all(
        (args.out / f"raw-rank{rank}.json").is_file() for rank in range(4)
    ):
        raise RuntimeError("--skip-raw requires all four prior raw result files")


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    args.staging = make_staging(args)
    write(
        args.out / "prereg.json",
        {
            "order": list(getattr(args, "arms", DEFAULT_ARMS)),
            "full_gate_order": list(DEFAULT_ARMS),
            "arm_timeout_s": {
                a: arm_timeout(args, a) for a in getattr(args, "arms", DEFAULT_ARMS)
            },
            "outer_timeout_s": window_budget(args),
            "parking_diagnostics": getattr(args, "parking_diagnostics", False),
            "parking_quota": getattr(args, "parking_quota", "native"),
            "async_admission": getattr(args, "async_admission", False),
            "quota_external_hit_gate": {
                "c10_attempts": 100,
                "minimum_full_hits": 95,
                "minimum_external_tokens": 28672,
            },
            "restore_count_c1": 100,
            "restore_count_c10": 100,
            "latency_limits_s": {"p50": 2.0, "p99": 4.0},
            "fixed_cases": 12,
            "chains": 3,
            "turns_per_chain": 4,
            "max_output_tokens": 128,
            "host_config": config(
                getattr(args, "parking_diagnostics", False),
                getattr(args, "parking_quota", "native"),
                getattr(args, "async_admission", False),
            ),
            "gpu_blocks": args.gpu_blocks,
            "git_tip": subprocess.check_output(
                ["git", "-C", str(args.worktree), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    reuse_evidence(args)
    import_preflight(args, "raw", negative_control=True)
    env = engine_env(args)
    with Monitor(args.out / "memory.jsonl") as monitor:
        raw = [
            str(args.python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(args.worktree / "benchmarks/host_parking_roundtrip.py"),
            "--execute-gpu",
            "--output",
            str(args.out / "raw-rank{rank}.json"),
        ]
        if not getattr(args, "skip_raw", False):
            timed_command(raw, env, args.out / "raw.out", 600, cwd=args.staging)
        for arm in (a for a in getattr(args, "arms", DEFAULT_ARMS) if a in CELL_ARMS):
            if ram_safe(monitor.samples) is False:
                raise RuntimeError("RAM/las stop threshold exceeded")
            if arm == LEGACY_ARM:
                run_legacy(args)
                continue
            with server(args, arm) as log:
                started = time.time()

                def checkpoint(kind, rows, arm=arm, log=log):
                    join_loads(rows, event_rows(log))
                    write(args.out / (arm + "." + kind + ".json"), rows)

                with phase(args, arm, "parity"):
                    suite(args, lambda rows: checkpoint("cases", rows))
                with phase(args, arm, "traffic"):
                    reset(args, True)
                    before, counters_before = metrics(args)
                    (args.out / (arm + ".metrics-before")).write_text(before)
                    t0 = time.time()
                    traffic(args, arm)
                    settle(args)
                    after, counters_after = metrics(args)
                    (args.out / (arm + ".metrics-after")).write_text(after)
                    write(
                        args.out / (arm + ".copy.json"),
                        {
                            "wall_s": time.time() - t0,
                            "delta": {
                                k: counters_after.get(k, 0) - counters_before.get(k, 0)
                                for k in set(counters_before) | set(counters_after)
                            },
                            "elapsed_cell_s_at_checkpoint": time.time() - started,
                        },
                    )
                with phase(args, arm, "returns"):
                    restored = returns(
                        args,
                        arm,
                        arm == "on1",
                        lambda rows: checkpoint("returns", rows),
                    )
                    settle(args)
                    checkpoint("returns", restored)
                write(
                    args.out / (arm + ".cell.json"),
                    dict(status="complete", cell_s=time.time() - started),
                )
        for arm in (a for a in getattr(args, "arms", DEFAULT_ARMS) if a in FAULT_ARMS):
            mode = arm.removeprefix("fault-")
            with server(args, arm, mode) as log, phase(args, arm, "recovery"):
                cases, aborts = [], []

                def save_fault(row=None, cases=cases, aborts=aborts, arm=arm, log=log):
                    if row is not None:
                        cases.append(row)
                    join_loads(cases, event_rows(log))
                    write(
                        args.out / (arm + ".json"), {"cases": cases, "aborts": aborts}
                    )

                tokens = prompt(args, 32769, arm)
                producer = complete(args, tokens, "producer", count=1)
                save_fault(producer)
                reset(args)
                first_return = complete(args, tokens, "first-good-return", count=1)
                save_fault(first_return)
                reset(args)
                failed_return = complete(args, tokens, "injected-return", count=1)
                save_fault(failed_return)
                reset(args)
                miss_return = complete(args, tokens, "failed-key-retry", count=1)
                save_fault(miss_return)
                # Do not assume a 200 is proof: require native snapshot_discard.
                followup = complete(
                    args, prompt(args, 4095, arm + "-unrelated"), "unrelated", count=1
                )
                save_fault(followup)
                reset(args, True)
                after_reset = complete(args, tokens, "after-reset", count=1)
                save_fault(after_reset)
                reset(args)
                aborts.append(abort_attempt(args, tokens, log, "H2D"))
                save_fault()
                reset(args, True)
                aborts.append(
                    abort_attempt(
                        args, prompt(args, 131073, arm + "-abort"), log, "D2H"
                    )
                )
                save_fault()
                final = complete(
                    args, prompt(args, 4097, arm + "-final"), "after-aborts", count=1
                )
                save_fault(final)
                settle(args)
                save_fault()
    return analyze(args.out)


def gate(status, reason, **data):
    return {"status": status, "reason": reason, **data}


def analyze(out):
    # Imported late so --plan does not need any engine libraries or a GPU.
    from host_parking_judge import judge

    result = judge(out)
    write(out / "gates.json", result)
    text = "| Gate | Result | Reason |\n|---|---|---|\n"
    for key, value in result["gates"].items():
        text += f"| {key} | {value['status']} | {value['reason']} |\n"
    if value := result.get("quota_external_hit"):
        text += f"| Additional quota hit | {value['status']} | {value['reason']} |\n"
    (out / "gates.md").write_text(text)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--budget-only", action="store_true")
    p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(DEFAULT_ARMS))
    p.add_argument("--reuse-from", type=Path)
    p.add_argument("--skip-raw", action="store_true")
    p.add_argument("--parking-diagnostics", action="store_true")
    p.add_argument(
        "--async-admission",
        action="store_true",
        help="Opt in to zero-token parking loads beyond the compute budget",
    )
    p.add_argument(
        "--parking-quota", choices=("native", "balanced-32k"), default="native"
    )
    p.add_argument("--on1-timeout-s", type=int, default=5400)
    p.add_argument("--cell-timeout-s", type=int, default=2700)
    p.add_argument("--fault-timeout-s", type=int, default=1500)
    p.add_argument("--out", type=Path, default=Path("/data/bench/m37"))
    p.add_argument("--port", type=int, default=18037)
    p.add_argument("--gpu-blocks", type=int, default=1800)
    p.add_argument(
        "--python",
        type=Path,
        default=Path("/data/venvs/1cat-main-integp7pr-p8/bin/python"),
    )
    p.add_argument("--worktree", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--repo", type=Path, default=Path("/home/cookys/projects/llm-playground")
    )
    p.add_argument("--plan", type=Path, default=Path("/data/bench/m19/Ap_c10fit.json"))
    a = p.parse_args()
    if len(set(a.arms)) != len(a.arms):
        p.error("each arm must be unique; repeated evidence is not replication")
    if a.arms != [x for x in a.arms if x in CELL_ARMS] + [
        x for x in a.arms if x in FAULT_ARMS
    ]:
        p.error("normal arms must precede fault arms")
    if min(a.on1_timeout_s, a.cell_timeout_s, a.fault_timeout_s) <= 0:
        p.error("timeouts must be positive")
    if a.skip_raw and a.reuse_from is None:
        p.error("--skip-raw requires --reuse-from")
    if a.reuse_from is not None:
        a.reuse_from = a.reuse_from.absolute()
        if not a.reuse_from.is_dir():
            p.error("--reuse-from must be an existing evidence directory")
    # Child processes run from staging, not the caller's (possibly engine) repo.
    # Keep the venv interpreter path lexical: resolving its symlink loses venv.
    for name in ("out", "python", "worktree", "repo", "plan"):
        setattr(a, name, getattr(a, name).absolute())
    if a.port in (8001, 8021) or not 1024 <= a.port <= 65535:
        p.error("dedicated unprivileged test port required; 8001/8021 forbidden")
    if a.budget_only:
        print(window_budget(a))
        return
    if a.analyze:
        print(json.dumps(analyze(a.out), indent=2))
        return
    if not a.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "order": a.arms,
                    "out": str(a.out),
                    "argv_off": command(a, "off1"),
                    "argv_on": command(a, "on1"),
                    "arm_timeout_s": {arm: arm_timeout(a, arm) for arm in a.arms},
                    "outer_timeout_s": window_budget(a),
                    "note": (
                        "Hard timeout sum, not an expected duration; "
                        "subset cannot close missing full-matrix gates"
                    ),
                },
                indent=2,
            )
        )
        return
    if os.environ.get("M37_GO") != "1":
        p.error("lead window requires explicit M37_GO=1")
    if a.out.exists() or a.out.is_symlink():
        p.error(f"refuse to overwrite existing output {a.out}")

    def interrupted(signum, frame):
        raise WindowInterrupted(
            f"window interruption {signum}; cleaning owned children", signum
        )

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, interrupted)
    baseline_oom = cgroup_oom()
    try:
        result = run(a)
    except BaseException as e:
        if a.out.is_dir():
            write(a.out / "window-error.json", {"error": repr(e)})
            write(
                a.out / "cgroup-events.json",
                {
                    "before": baseline_oom,
                    "after": cgroup_oom(),
                    "oom_delta": cgroup_oom() - baseline_oom,
                },
            )
            analyze(a.out)
        raise
    write(
        a.out / "cgroup-events.json",
        {
            "before": baseline_oom,
            "after": cgroup_oom(),
            "oom_delta": cgroup_oom() - baseline_oom,
        },
    )
    result = analyze(a.out)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
