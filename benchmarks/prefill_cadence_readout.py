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


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * fraction
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def analyze(text, ranks=4, first_step=0, last_step=None):
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

    return dict(
        status="INCONCLUSIVE" if problems or not steps else "COMPLETE",
        problems=sorted(set(problems)),
        regimes={
            regime: summarize(
                [r for r in steps if r["decode_reqs"] and r["regime"] == regime]
            )
            for regime in ("steady", "interfered")
        },
        phases={
            phase: summarize([r for r in steps if r["phase"] == phase])
            for phase in ("mixed", "decode", "prefill")
        },
        steps=steps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--first-step", type=int, default=0)
    parser.add_argument("--last-step", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.log.read_text(), args.ranks, args.first_step, args.last_step)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
