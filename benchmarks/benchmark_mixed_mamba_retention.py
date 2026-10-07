#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short serving probe for mixed-page sparse Mamba retention.

Run only in an assigned GPU window, against an already running idle server.
For the 27B TP4 DFlash2 FP8 layout use --num-gpu-blocks-override 192 on
the SERVER, retaining block_size=mamba_block_size=2048. Startup is outside
this client's 270-second budget. Repeat on baseline and patched servers.
No cache reset, server mutation, or prompt/response content is logged.
"""

import argparse
import json
import random
import secrets
import threading
import time
import urllib.request
from pathlib import Path


def metric(text, alternatives):
    for name in alternatives:
        values = []
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                values.append(float(line.rsplit(" ", 1)[1]))
        if values:
            return sum(values)
    return None


def run(args):
    started = time.monotonic()
    deadline = started + args.budget_s
    result = {
        "server": args.base_url,
        "case": args.label,
        "budget_s": args.budget_s,
        "requests": [],
        "kv_usage_max_fraction": None,
        "metrics_errors": 0,
        "note": "Single-client metrics deltas; no inference from banner tokens.",
    }
    stop = threading.Event()

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=3) as r:
            text = r.read().decode()
        return {
            "hits": metric(
                text, ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"]
            ),
            "queries": metric(
                text, ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"]
            ),
            "preempt": metric(
                text, ["vllm:num_preemptions_total", "vllm:num_preemptions"]
            ),
            "kv": metric(text, ["vllm:kv_cache_usage_perc"]),
        }

    def poll():
        while not stop.is_set():
            try:
                value = metrics()["kv"]
                if value is not None:
                    old = result["kv_usage_max_fraction"]
                    result["kv_usage_max_fraction"] = max(old or 0, value)
            except Exception:
                result["metrics_errors"] += 1
            stop.wait(0.25)

    def save():
        result["elapsed_s"] = time.monotonic() - started
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    # Unique first hash blocks avoid contamination by previous probes. Numeric
    # IDs bypass tokenizer length differences; these IDs fit the 27B vocabulary.
    seed = secrets.randbits(64)
    rng = random.Random(seed)
    a = [rng.randrange(1000, 12000) for _ in range(32769)]
    b = [rng.randrange(12000, 24000) for _ in range(65537)]
    extension = a + [rng.randrange(1000, 12000) for _ in range(512)]
    cases = [
        ("cold_a", a),
        ("immediate_a", a),
        ("pressure_b", b),
        ("after_pressure_a", a),
        ("append_a", extension),
    ]
    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()
    try:
        for name, prompt in cases:
            if time.monotonic() >= deadline - 5:
                result["incomplete"] = "budget exhausted"
                break
            before = metrics()
            request = urllib.request.Request(
                args.base_url + "/v1/completions",
                data=json.dumps(
                    {
                        "model": args.model,
                        "prompt": prompt,
                        "max_tokens": 1,
                        "temperature": 0,
                        "ignore_eos": True,
                        "seed": 17,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            begin = time.monotonic()
            with urllib.request.urlopen(
                request, timeout=max(1, deadline - begin - 4)
            ) as r:
                response = json.load(r)
            elapsed = time.monotonic() - begin
            # Allow periodic server metrics to catch up; record raw counters so
            # missing/delayed metrics remain visible rather than implying zero.
            after = metrics()
            while (
                before["queries"] is not None
                and after["queries"] == before["queries"]
                and time.monotonic() < min(begin + elapsed + 6, deadline - 4)
            ):
                time.sleep(0.1)
                after = metrics()
            delta = {
                key: (
                    after[key] - before[key]
                    if after[key] is not None and before[key] is not None
                    else None
                )
                for key in ("hits", "queries", "preempt")
            }
            row = {
                "case": name,
                "prompt_tokens": len(prompt),
                "elapsed_s": elapsed,
                "usage": response.get("usage"),
                "metrics_before": before,
                "metrics_after": after,
                "delta": delta,
            }
            row["prefix_hit_fraction"] = (
                delta["hits"] / delta["queries"]
                if delta["hits"] is not None and delta["queries"]
                else None
            )
            row["metrics_cover_one_request"] = (
                delta["queries"] is not None
                and len(prompt) - 2 <= delta["queries"] <= len(prompt) + 2
                and delta["hits"] is not None
                and 0 <= delta["hits"] <= delta["queries"]
            )
            result["requests"].append(row)
            save()
            print(json.dumps(row), flush=True)
    except Exception as error:
        result["incomplete"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        monitor.join(timeout=3.5)
        save()
    return 1 if "incomplete" in result else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.8-27B-QUASAR-NVFP4")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-s", type=float, default=270)
    args = parser.parse_args()
    if not 10 <= args.budget_s <= 280:
        parser.error("budget must be between 10 and 280 seconds")
    args.base_url = args.base_url.rstrip("/")
    raise SystemExit(run(args))
