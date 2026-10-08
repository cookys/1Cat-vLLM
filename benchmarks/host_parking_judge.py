# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only m37 seven-gate judge; missing evidence never becomes PASS."""

import json
import re
import statistics
from pathlib import Path

ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")


def load(path):
    return json.loads(Path(path).read_text())


def gate(status, reason, **evidence):
    return dict(status=status, reason=reason, **evidence)


def quantile(values, q):
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * q
    i = int(x)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (x - i)


def compare(left, right):
    a = {r["label"]: r for r in left}
    b = {r["label"]: r for r in right}
    if a.keys() != b.keys() or len(a) != 24:
        return dict(valid=False, reason="need 12 fixed cases + 12 chain turns")
    diffs = []
    ids_equal = True
    for k in a:
        x, y = a[k], b[k]
        if (
            not x.get("ok")
            or not y.get("ok")
            or len(x["logprobs"]) != len(y["logprobs"])
        ):
            return dict(valid=False, reason="failed or incomplete response")
        ids_equal &= x["ids"] == y["ids"]
        for p, q in zip(x["logprobs"], y["logprobs"]):
            if p is None or q is None:
                return dict(valid=False, reason="missing numeric logprob")
            diffs.append(abs(p - q))
    return dict(
        valid=bool(diffs),
        ids_equal=ids_equal,
        maximum=max(diffs, default=0),
        mean=statistics.mean(diffs) if diffs else 0,
    )


def parity_gate(cases):
    off = [cases[k] for k in ("off1", "off2", "off3")]
    floor = [compare(off[i], off[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    comparisons = [
        compare(cases[a], cases[b])
        for a, b in (("off1", "on1"), ("off2", "on2"), ("off3", "on3"))
    ]
    if not all(x["valid"] for x in floor + comparisons):
        return gate(
            "INCONCLUSIVE",
            "missing parity IDs/logprobs",
            comparisons=comparisons,
            floor=floor,
        )
    max_floor = max(1e-6, *(x["maximum"] for x in floor))
    mean_floor = max(1e-7, *(x["mean"] for x in floor))
    passed = all(x["ids_equal"] for x in floor + comparisons) and all(
        x["maximum"] <= max_floor and x["mean"] <= mean_floor for x in comparisons
    )
    return gate(
        "PASS" if passed else "FAIL",
        "IDs exact; raw logprob max/mean within OFF floor",
        comparisons=comparisons,
        floor=floor,
        threshold_max=max_floor,
        threshold_mean=mean_floor,
    )


def latency_gate(rows):
    report = {}
    passed = True
    for n in (1, 10):
        selected = [r for r in rows if r.get("return_concurrency") == n]
        # A tiny surviving prefix is not a restored image. One 4096-token tail
        # can be recomputed, consistent with committed-boundary/last-token rule.
        valid = [
            r
            for r in selected
            if r.get("ok")
            and r.get("restored_tokens", 0) >= r["prompt_tokens"] - 4096
            and r.get("arrival_to_h2d_observed_s") is not None
        ]
        values = [r["arrival_to_h2d_observed_s"] for r in valid]
        report[str(n)] = {
            "attempts": len(selected),
            "restores": len(valid),
            "p50_s": quantile(values, 0.5),
            "p99_s": quantile(values, 0.99),
        }
        if len(selected) < 100:
            return gate(
                "INCONCLUSIVE", "need 100 attempted returns per regime", regimes=report
            )
        passed &= (
            len(valid) >= 100
            and quantile(values, 0.5) <= 2
            and quantile(values, 0.99) <= 4
        )
    return gate(
        "PASS" if passed else "FAIL",
        "100 genuine restores/regime; p50<=2s p99<=4s",
        regimes=report,
    )


def interference_gate(traffic, copies):
    fields = ("steady_tok_s_per_user_median", "interfered_tok_s_per_user_median")
    detail = {}
    passed = True
    for field in fields:
        off = [
            traffic[a]["summary"]["windows"].get(field)
            for a in ("off1", "off2", "off3")
        ]
        on = [
            traffic[a]["summary"]["windows"].get(field) for a in ("on1", "on2", "on3")
        ]
        if any(x is None or x <= 0 for x in off + on):
            return gate("INCONCLUSIVE", "missing S steady/interfered samples")
        floor = (max(off) - min(off)) / min(off)
        degradation = max(0, 1 - statistics.median(on) / statistics.median(off))
        passed &= degradation <= floor
        detail[field] = dict(off=off, on=on, noise_floor=floor, degradation=degradation)
    for a in ORDER:
        j = traffic[a]["summary"]
        agg = j["aggregate"]
        if (
            agg.get("errors", 0)
            or j["coresident"].get("samples_running_eq_users_and_waiting_0", 0) == 0
        ):
            return gate("INCONCLUSIVE", "errors or no verified c10 co-residence", arm=a)
    copy_report = {}
    for a in ("on1", "on2", "on3"):
        c = copies[a]
        d = c["delta"]
        wall = c["wall_s"]
        if d.get("GPU_to_CPU_bytes", 0) <= 0 or d.get("GPU_to_CPU_seconds", 0) <= 0:
            return gate("INCONCLUSIVE", "no actual immediate D2H activity", arm=a)
        # Each native direction serializes its transfers. Across TP4, this
        # is mean per-rank union-of-job-event intervals, NOT cross-rank union.
        copy_report[a] = dict(
            d2h_gbs=d["GPU_to_CPU_bytes"] / wall / 1e9,
            summed_d2h_event_s=d["GPU_to_CPU_seconds"],
            mean_rank_d2h_job_busy_fraction=d["GPU_to_CPU_seconds"] / 4 / wall,
            h2d_gbs=d.get("CPU_to_GPU_bytes", 0) / wall / 1e9,
        )
    detail["R_annotation"] = {
        a: traffic[a]["summary"]["aggregate"].get("per_stream_tok_s") for a in ORDER
    }
    detail["interfered_fraction"] = {
        a: traffic[a]["summary"]["windows"].get("pct_time_interfered_overall")
        for a in ORDER
    }
    return gate(
        "PASS" if passed else "FAIL",
        "S degradation <= three-OFF noise; R annotation only",
        metrics=detail,
        copy=copy_report,
    )


def utility_gate(returns):
    pairs = []
    for off, on in (("off1", "on1"), ("off2", "on2"), ("off3", "on3")):
        a = {r["label"]: r for r in returns[off]}
        b = {r["label"]: r for r in returns[on]}
        common = a.keys() & b.keys()
        gains = []
        proxy = []
        for label in common:
            x, y = a[label], b[label]
            if not x.get("ok") or not y.get("ok") or not y.get("loads"):
                continue
            if y.get("restored_tokens", 0) < y["prompt_tokens"] - 4096:
                continue
            gains.append(x["ttft_s"] - y["ttft_s"])
            proxy.append(x["ttft_s"] - y["arrival_to_h2d_observed_s"])
        if len(gains) != 11:
            return gate(
                "FAIL",
                "all 11 paired returns must actually restore the host image",
                pair=[off, on],
                count=len(gains),
            )
        pairs.append(
            dict(
                arms=[off, on],
                median_ttft_s_saved=statistics.median(gains),
                avoided_prefill_upper_proxy_s=sum(proxy),
                total_ttft_s_saved=sum(gains),
            )
        )
    ok = all(
        p["median_ttft_s_saved"] > 0 and p["total_ttft_s_saved"] > 0 for p in pairs
    )
    return gate(
        "PASS" if ok else "FAIL",
        "positive paired net TTFT savings (GPU reset pressure surrogate)",
        pairs=pairs,
        limitation=(
            "Does not prove natural-LRU capacity benefit; "
            "prefill seconds proxy includes first decode"
        ),
    )


def failure_gate(out):
    reports = []
    for mode in ("pre_submit", "completed_copy"):
        arm = "fault-" + mode
        cases = load(out / (arm + ".json"))["cases"]
        text = (out / (arm + ".serve.log")).read_text(errors="replace")
        quotas = [
            json.loads(line.split("HOST_PARKING quota ", 1)[1])
            for line in text.splitlines()
            if "HOST_PARKING quota " in line
        ]
        checks = {
            "all_requests_ok": all(r.get("ok") for r in cases),
            "injected": f"injected_failure mode={mode}" in text,
            "one_rank_failed": "failed_ranks=1" in text,
            "discarded": "snapshot_discard" in text,
            "affected_recomputed": any(
                not e["success"] for e in cases[1].get("loads", [])
            )
            and cases[1]["ids"] == cases[0]["ids"],
            "failed_host_key_missed": "HOST_PARKING invalid_lookup_miss" in text,
            "quota_reset": bool(quotas)
            and all(
                q["in_use_slots"] == 0
                and q["in_use_bytes_per_rank"] == 0
                and q["read_write_refs"] == 0
                and q["invalid_keys"] == 0
                for q in quotas
            ),
            "all_rank_pool_release": text.count("HOST_PARKING pool_released") >= 4,
            "reset_generation": bool(
                re.search(r"HOST_PARKING reset generation=[1-9]", text)
            ),
            "reset_miss": not cases[3].get("loads"),
            "aborted_during_load": bool(
                re.search(r"HOST_PARKING abort .*pending_loads=[1-9]", text)
            ),
            "aborted_during_store": bool(
                re.search(
                    r"HOST_PARKING abort .*pending_loads=\d+ pending_stores=[1-9]", text
                )
            ),
            "no_fatal": "fatal_copy_context" not in text
            and "EngineDeadError" not in text,
        }
        reports.append(dict(mode=mode, checks=checks))
    if not all(v for r in reports for v in r["checks"].values()):
        failed = any(
            not r["checks"]["all_requests_ok"] or not r["checks"]["no_fatal"]
            for r in reports
        )
        return gate(
            "FAIL" if failed else "INCONCLUSIVE",
            "injection/reset/abort evidence incomplete; inspect individual checks",
            reports=reports,
        )
    return gate(
        "PASS",
        "single-rank recoverable fault, all-rank drain and unrelated followup",
        reports=reports,
        limitation=(
            "No actual device-loss injection; fatal context is CPU-only teardown test"
        ),
    )


def judge(out):
    out = Path(out)
    gates = {}

    def guarded(name, fn):
        try:
            gates[name] = fn()
        except (OSError, KeyError, TypeError, ValueError, IndexError) as e:
            gates[name] = gate("INCONCLUSIVE", "missing/invalid evidence: " + str(e))

    def raw():
        data = [load(out / f"raw-rank{r}.json") for r in range(4)]
        ok = all(
            d["status"] == "PASS"
            and len(d["transfers"]) == 54
            and all(t["byte_equal"] and t["padding_untouched"] for t in d["transfers"])
            for d in data
        )
        return gate(
            "PASS" if ok else "FAIL",
            "4 ranks x 9 groups x single/shuffled multi-block x 3 cycles",
        )

    guarded("1 raw bytes", raw)
    guarded(
        "2 parity",
        lambda: parity_gate({a: load(out / (a + ".cases.json")) for a in ORDER}),
    )
    guarded("3 latency", lambda: latency_gate(load(out / "on1.returns.json")))
    guarded(
        "4 interference",
        lambda: interference_gate(
            {a: load(out / (a + ".traffic.json")) for a in ORDER},
            {a: load(out / (a + ".copy.json")) for a in ORDER},
        ),
    )
    guarded(
        "5 useful work",
        lambda: utility_gate({a: load(out / (a + ".returns.json")) for a in ORDER}),
    )

    def ram():
        from host_parking_window import ram_safe

        samples = [
            json.loads(x) for x in (out / "memory.jsonl").read_text().splitlines()
        ]
        if not samples:
            return gate("INCONCLUSIVE", "no RAM samples")
        placements = [load(out / (a + ".numa.json")) for a in ("on1", "on2", "on3")]
        placement_ok = all(
            len(p) == 4
            and all(
                r["other_node_pages"] == 0
                and r["resident_bytes"] >= r["bytes"]
                and r["physical_pool"]["in_use_bytes"] == r["bytes"]
                and r["physical_pool"]["in_use_slots"] > 0
                and r["physical_pool"]["pinned"]
                for r in p
            )
            for p in placements
        )
        quota_ok = all(sum(r["bytes"] for r in p) <= 16 << 30 for p in placements)
        # Endpoint snapshots supplement per-second monitor; actual cgroup OOMs fail.
        oom = load(out / "cgroup-events.json")
        ok = ram_safe(samples) and placement_ok and quota_ok and oom["oom_delta"] == 0
        return gate(
            "PASS" if ok else "FAIL",
            "16 GiB quota, NUMA0, no OOM/las regression",
            min_memavailable_kib=min(s["meminfo_kib"]["MemAvailable"] for s in samples),
            placements=placements,
            oom=oom,
        )

    guarded("6 RAM/las", ram)
    guarded("7 recovery", lambda: failure_gate(out))
    status = (
        "FAIL"
        if any(g["status"] == "FAIL" for g in gates.values())
        else "PASS"
        if all(g["status"] == "PASS" for g in gates.values())
        else "INCONCLUSIVE"
    )
    return dict(
        status=status,
        production_go=False,
        gates=gates,
        note=(
            "Technical window only; natural-LRU capacity and "
            "isolated fatal-rank recovery remain separate gates"
        ),
    )
