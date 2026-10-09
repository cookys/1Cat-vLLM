#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read V2 per-step logs offline. No CUDA, HTTP, or prompt data access."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

MIN_STEPS = 20
REQUIRED_PHASES = ("mixed", "decode")
REQUIRED_REGIMES = ("steady", "interfered")
COMPARISON_FIELD = "per_decoder_tok_s_with_gap"
MIN_COLD_REQUESTS = 10
COLD_COHORTS = ("128K", "200K")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * fraction
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def cold_readout(evidence):
    """Validate raw fixed-arrival cohorts; never silently drop failed requests."""
    problems, failures, cohorts = [], [], {}
    source = evidence.get("cohorts", {}) if isinstance(evidence, dict) else {}
    for name in COLD_COHORTS:
        rows = source.get(name, [])
        if not isinstance(rows, list):
            rows = []
            problems.append(f"invalid_cold_cohort:{name}")
        if len(rows) < MIN_COLD_REQUESTS:
            problems.append(f"insufficient_cold:{name}:{len(rows)}<{MIN_COLD_REQUESTS}")
        values = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                problems.append(f"invalid_cold_row:{name}:{index}")
                continue
            if row.get("ok") is False or row.get("error"):
                failures.append(f"cold_request_failed:{name}:{index}")
            elif row.get("ok") is not True:
                problems.append(f"missing_cold_outcome:{name}:{index}")
            cached = row.get("cached_tokens")
            if type(cached) is not int or not 0 <= cached <= 4096:
                problems.append(f"unverified_cold_cache:{name}:{index}")
            ttft = row.get("ttft_s")
            if type(ttft) not in (float, int) or not math.isfinite(ttft) or ttft <= 0:
                problems.append(f"invalid_cold_ttft:{name}:{index}")
            else:
                values.append(ttft)
        valid = len(values) == len(rows) and not any(
            f":{name}:" in p for p in problems + failures
        )
        cohorts[name] = dict(
            requests=len(rows),
            ttft_s_p90=percentile(values, 0.9) if valid else None,
        )
    return dict(cohorts=cohorts, min_requests=MIN_COLD_REQUESTS), problems, failures


def analyze(text, ranks=4, first_step=0, last_step=None, cold_evidence=None):
    timings, complete = defaultdict(dict), {}
    problems = []
    for line in text.splitlines():
        if "PREFILL_CADENCE_TIMING_GAP" in line:
            problems.append("explicit_timing_gap")
        if "PREFILL_CADENCE_STEP " in line:
            row = json.loads(line.split("PREFILL_CADENCE_STEP ", 1)[1])
            sid, rank = row["step_id"], row["rank"]
            if rank in timings[sid]:
                problems.append("duplicate_rank_step")
            timings[sid][rank] = row
        if "PREFILL_CADENCE_COMPLETE " in line:
            row = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
            sid = row["step_id"]
            if sid in complete:
                problems.append("duplicate_completion")
            complete[sid] = row
    selected = sorted(
        s
        for s in timings.keys() | complete.keys()
        if s >= first_step and (last_step is None or s <= last_step)
    )
    steps = []
    for sid in selected:
        by_rank = timings.get(sid, {})
        if (
            len(by_rank) != ranks
            or set(by_rank) != set(range(ranks))
            or sid not in complete
        ):
            problems.append(f"incomplete_step:{sid}")
            continue
        row = dict(by_rank[0])
        for other in by_rank.values():
            for field in (
                "prefill_rows",
                "decode_rows",
                "decode_reqs",
                "pending_prefill_reqs",
            ):
                if row[field] != other[field]:
                    problems.append(f"metadata_mismatch:{sid}")
            if not math.isfinite(other["stream_ms"]) or other["stream_ms"] < 0:
                problems.append(f"invalid_timing:{sid}")
        row["stream_ms"] = max(v["stream_ms"] for v in by_rank.values())
        # Both endpoints are on each rank's main stream. Preserve gaps as a
        # separate ledger: adding the max gap to the max duration is conservative.
        row["stream_gap_ms"] = max(
            v.get("stream_gap_ms") or 0 for v in by_rank.values()
        )
        row["decode_accepted_tokens"] = complete[sid]["decode_accepted_tokens"]
        row["phase"] = (
            ("mixed" if row["decode_reqs"] else "prefill")
            if row["prefill_rows"]
            else "decode"
        )
        row["regime"] = "interfered" if row["pending_prefill_reqs"] else "steady"
        steps.append(row)

    def summarize(rows):
        ms = [r["stream_ms"] for r in rows]
        person_ms = sum(r["decode_reqs"] * r["stream_ms"] for r in rows)
        person_with_gap = sum(
            r["decode_reqs"] * (r["stream_ms"] + r["stream_gap_ms"]) for r in rows
        )
        tokens = sum(r["decode_accepted_tokens"] for r in rows)
        return dict(
            steps=len(rows),
            stream_ms_p50=statistics.median(ms) if ms else None,
            stream_ms_p90=percentile(ms, 0.9),
            stream_ms_p99=percentile(ms, 0.99),
            decoder_tokens=tokens,
            decoder_person_ms=person_ms,
            per_decoder_tok_s=1000 * tokens / person_ms if person_ms else None,
            per_decoder_tok_s_with_gap=1000 * tokens / person_with_gap
            if person_with_gap
            else None,
            full_steps=sum("FULL" in r.get("route", "") for r in rows),
        )

    regimes = {
        regime: summarize(
            [r for r in steps if r["decode_reqs"] and r["regime"] == regime]
        )
        for regime in ("steady", "interfered")
    }
    phases = {
        phase: summarize([r for r in steps if r["phase"] == phase])
        for phase in ("mixed", "decode", "prefill")
    }
    # Cadence may insert pure decode while prefill remains pending. Require
    # both physical phases AND both exposure regimes; they are not synonyms.
    # Pure prefill has no decoder denominator and is diagnostic only.
    for group, required in (("phases", REQUIRED_PHASES), ("regimes", REQUIRED_REGIMES)):
        table = phases if group == "phases" else regimes
        for name in required:
            if table[name]["steps"] < MIN_STEPS:
                problems.append(
                    f"insufficient_{group}:{name}:{table[name]['steps']}<{MIN_STEPS}"
                )
    cold, cold_problems, failures = cold_readout(cold_evidence)
    problems.extend(cold_problems)
    return dict(
        status=(
            "FAIL"
            if failures
            else "INCONCLUSIVE"
            if problems or not steps
            else "COMPLETE"
        ),
        problems=sorted(set(problems)),
        failures=failures,
        cold=cold,
        comparison_fields={
            regime: f"regimes.{regime}.{COMPARISON_FIELD}"
            for regime in REQUIRED_REGIMES
        },
        coverage_requirements=dict(
            min_steps=MIN_STEPS,
            phases=REQUIRED_PHASES,
            regimes=REQUIRED_REGIMES,
            diagnostic_only_phases=["prefill"],
        ),
        regimes=regimes,
        phases=phases,
        steps=steps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--first-step", type=int, default=0)
    parser.add_argument("--last-step", type=int)
    parser.add_argument(
        "--cold-json",
        type=Path,
        help="Raw {cohorts: {128K: [rows], 200K: [rows]}}; missing => INCONCLUSIVE",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.log.read_text(),
        args.ranks,
        args.first_step,
        args.last_step,
        json.loads(args.cold_json.read_text()) if args.cold_json else None,
    )
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
