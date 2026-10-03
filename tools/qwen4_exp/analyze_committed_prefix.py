# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline length proxy and actual committed-prefix scheduler log accounting.

No vLLM import, server request or GPU access. SWE lengths cannot establish a
cache hit: exact rendered tokens, MTP lookahead and block lifetimes are absent.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def swe_proxy(paths, alignment):
    counts = collections.Counter()
    for path in paths:
        data = json.loads(path.read_text())
        calls = [
            row for row in data.get("transcript", []) if row.get("role") == "assistant"
        ]
        if not calls:
            continue
        counts["tasks"] += 1
        counts["calls"] += len(calls)
        counts["first_calls"] += 1
        for previous, current in zip(calls, calls[1:]):
            counts["followups"] += 1
            p, c = previous.get("prompt_tokens"), previous.get("completion_tokens")
            n = current.get("prompt_tokens")
            if any(not isinstance(x, int) for x in (p, c, n)):
                counts["missing_lengths"] += 1
                continue
            if current.get("trimmed") or n < p:
                counts["trimmed_or_shrinking"] += 1
                continue
            counts["nonshrinking_untrimmed"] += 1
            boundary = (p - 1) // alignment * alignment
            if boundary <= 0:
                counts["previous_prompt_too_short"] += 1
                continue
            counts["earlier_prefill_boundary_proxy"] += 1
            if boundary >= 2 * alignment:
                counts["previous_prefill_boundary_at_least_two_blocks"] += 1
            shared = min(n - 1, p + c - 1)
            latest = shared // alignment * alignment
            key = (
                "latest_boundary_in_previous_prefill_proxy"
                if latest < p
                else "latest_in_decode_but_earlier_prefill_proxy"
            )
            counts[key] += 1
    result = dict(counts)
    total = counts["calls"]
    for key in (
        "earlier_prefill_boundary_proxy",
        "latest_boundary_in_previous_prefill_proxy",
    ):
        result[key + "_fraction_of_all_calls"] = counts[key] / total if total else None
    result["qualification"] = (
        "Length proxies only, not eligible hits. Assumes rendered history keeps "
        "the same token prefix and token[B]; ignores eviction, producer completion, "
        "preemption and chunk placement. Never multiply all calls by one page."
    )
    return result


def log_accounting(paths):
    lookups, finishes, admissions = [], [], []
    scheduled = collections.Counter()
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            if "MTP_COMMITTED_PREFIX " not in line:
                continue
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if "MTP_COMMITTED_PREFIX lookup " in line:
                for key in (
                    "prompt_tokens",
                    "baseline_tokens",
                    "hit_tokens",
                    "saved_tokens",
                    "exclusive_blocks",
                    "evicted_blocks",
                    "free_blocks",
                ):
                    fields[key] = int(fields[key])
                lookups.append(fields)
            elif "MTP_COMMITTED_PREFIX finish " in line:
                finishes.append(fields)
            elif "MTP_COMMITTED_PREFIX admit " in line:
                admissions.append(fields)
            elif "MTP_COMMITTED_PREFIX scheduled " in line:
                scheduled[fields["request"]] += int(fields["prefill_tokens"])
    # Scheduling retries are separate observations, not necessarily executed
    # prefill. Dedup by request uses its LAST lookup, still only admission intent.
    last = {row["request"]: row for row in lookups}
    rows = list(last.values())
    hits = sum(row["saved_tokens"] > 0 for row in rows)
    completed = {row["request"] for row in finishes if row["publish"] == "True"}
    return {
        "lookup_attempts": len(lookups),
        # Unlike lookup intent, admission records require successful allocation.
        # A preempted/resumed request can have more than one admission.
        "admissions": len(admissions),
        "warm_producer_admissions": sum(
            row["warm_producer"] == "True" for row in admissions
        ),
        "non_warm_admissions": sum(
            row["warm_producer"] == "False" for row in admissions
        ),
        "admission_cut_reasons": dict(
            collections.Counter(
                row.get("cut_reason", "legacy_unspecified") for row in admissions
            )
        ),
        "unique_requests": len(rows),
        "last_lookup_hit_requests": hits,
        "last_lookup_hit_fraction": hits / len(rows) if rows else None,
        "last_lookup_saved_tokens": sum(row["saved_tokens"] for row in rows),
        "last_lookup_baseline_replay_tokens": sum(
            row["prompt_tokens"] - row["baseline_tokens"] for row in rows
        ),
        "last_lookup_candidate_replay_tokens": sum(
            row["prompt_tokens"] - row["hit_tokens"] for row in rows
        ),
        "max_exclusive_blocks": max(
            (row["exclusive_blocks"] for row in lookups), default=0
        ),
        "max_cumulative_evicted_blocks": max(
            (row["evicted_blocks"] for row in lookups), default=0
        ),
        "min_free_blocks": min((row["free_blocks"] for row in lookups), default=None),
        "published_certificates": sum(
            int(row["certificates"]) for row in finishes if row["publish"] == "True"
        ),
        "scheduled_prefill_tokens": sum(scheduled.values()),
        "normal_finished_requests": len(completed),
        "normal_finished_scheduled_prefill_tokens": sum(
            value for request, value in scheduled.items() if request in completed
        ),
        "per_request_scheduled_prefill_tokens": dict(scheduled),
        "qualification": (
            "One server run per input; concatenate only non-overlapping log slices. "
            "Last lookup per request is scheduler admission intent, not proof the "
            "forward executed. Scheduled counters cover actual scheduler outputs; "
            "the normal-finished subset excludes aborts (but includes recomputation "
            "after any preemption). Logs must cover each entire request. "
            "Warm is admission eligibility, not proof an extra cut was taken. "
            "chain_warm_v2b requires a candidate certificate or a positive "
            "ordinary hit within one page of B. Older warm_v2 used any positive "
            "ordinary hit. Admission counters are absent in v1 logs; resumed "
            "admissions count again. "
            "Exclusive blocks count extra Mamba blocks, not all certificate blocks."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swe-results", type=Path)
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--alignment", type=int, default=1616)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.swe_results and not args.log:
        parser.error("provide --swe-results and/or --log")
    if args.alignment <= 0:
        parser.error("--alignment must be positive")
    result = {"alignment": args.alignment}
    if args.swe_results:
        result["swe_length_proxy"] = swe_proxy(
            sorted(args.swe_results.glob("*.json")), args.alignment
        )
    if args.log:
        result["scheduler_lookup"] = log_accounting(args.log)
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
