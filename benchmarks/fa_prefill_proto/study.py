# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only statistics and reporting for the five-arm GPU window."""

import json
import math
import random
import statistics
from pathlib import Path

ARMS = ("0", "1", "2", "1_free", "2_free")
LABELS = {"0": "P0", "1": "P1-r2", "2": "P2",
          "1_free": "P1-free", "2_free": "P2-free"}
BLOCKS = 6
SAMPLES = 8
WARMUP = 5
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 8417
POINT_THRESHOLDS = {"96": 1.0, "4032": 0.9}
UPPER95_THRESHOLDS = {"96": 1.03, "4032": 0.9}


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = math.floor(index)
    return values[left] + (values[math.ceil(index)] - values[left]) * (index - left)


def summarize(blocks):
    if len(blocks) < 2:
        raise ValueError("need at least two ABBA blocks")
    ratios, aa_ratios, all_a, all_b = [], [], [], []
    for block in blocks:
        if [p["arm"] for p in block] != ["A", "B", "B", "A"]:
            raise ValueError("not ABBA")
        for phase in block:
            samples = phase["event_ms"]
            if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
                raise ValueError("event times must be finite and positive")
            (all_a if phase["arm"] == "A" else all_b).extend(samples)
        a1, b1, b2, a2 = [statistics.median(p["event_ms"]) for p in block]
        ratios.append(math.sqrt(b1 * b2 / (a1 * a2)))
        aa_ratios.append(a2 / a1)
    rng = random.Random(BOOTSTRAP_SEED)
    resamples = [statistics.median(rng.choices(ratios, k=len(ratios)))
                 for _ in range(BOOTSTRAPS)]
    ratio = statistics.median(ratios)
    return {
        "baseline_median_ms": statistics.median(all_a),
        "candidate_median_ms": statistics.median(all_b),
        "paired_ratio_median": ratio,
        "relative_p0_percent": 100 * (ratio - 1),
        "raw_medians_percent": 100 * (statistics.median(all_b)
                                      / statistics.median(all_a) - 1),
        "block_ratios": ratios,
        "noise_ratio_min_max": [min(ratios), max(ratios)],
        "bootstrap_lower95_one_sided": percentile(resamples, 0.05),
        "bootstrap_upper95_one_sided": percentile(resamples, 0.95),
        "aa_phase_ratios": aa_ratios,
        "aa_noise_ratio_min_max": [min(aa_ratios), max(aa_ratios)],
        "aa_max_abs_drift_percent": 100 * max(abs(r - 1) for r in aa_ratios),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "blocks": len(blocks),
    }


def classify(stats, bitwise, finite, baseline_match):
    if not finite:
        return "INVALID_NONFINITE"
    if not baseline_match:
        return "BASELINE_MISMATCH_NO_PRODUCTION_INFERENCE"
    if any(stats[m]["paired_ratio_median"] > cap
           for m, cap in POINT_THRESHOLDS.items()):
        return "SPEED_FAIL" if bitwise else "SPEED_FAIL_E2_PENDING"
    if any(stats[m]["bootstrap_upper95_one_sided"] > cap
           for m, cap in UPPER95_THRESHOLDS.items()):
        return "INCONCLUSIVE" if bitwise else "INCONCLUSIVE_E2_PENDING"
    return "E1_MICROBENCH_CANDIDATE" if bitwise else "E2_PENDING"


def numerical_metrics(reference, actual, ref_bytes, actual_bytes):
    """CPU arrays (numpy at runtime); report byte equality including signed zero."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).ravel()
    val = np.asarray(actual, dtype=np.float64).ravel()
    if ref.shape != val.shape:
        raise ValueError("output shape mismatch")
    finite = bool(np.isfinite(ref).all() and np.isfinite(val).all())
    result = {"bitwise": ref_bytes == actual_bytes, "finite": finite,
              "max_abs_diff": None, "max_rel_diff": None, "relative_l2": None}
    if finite:
        diff = np.abs(val - ref)
        result.update(
            max_abs_diff=float(diff.max(initial=0)),
            max_rel_diff=float((diff / np.maximum(np.abs(ref), 1e-7)).max(initial=0)),
            relative_l2=float(np.linalg.norm(diff)
                              / max(float(np.linalg.norm(ref)), 1e-7)),
        )
    return result


def markdown(result):
    lines = ["# Prefill FA five-arm GPU study", "",
             f"Status: **{result['status']}**. No profiler; CUDA-event samples.",
             "No serving adoption or distribution-equivalence claim.", "",
             "Speed gates (point / one-sided 95% upper): "
             f"M96 <= {POINT_THRESHOLDS['96']:.2f} / "
             f"{UPPER95_THRESHOLDS['96']:.2f}; "
             f"X126 <= {POINT_THRESHOLDS['4032']:.2f} / "
             f"{UPPER95_THRESHOLDS['4032']:.2f}.", "",
             "| M | Arm | kernel median ms | paired P0 Δ% | ABBA ratio min–max "
             "| 95% upper | output/LSE bits | max abs O/LSE | REG | smem B "
             "| spill store/load B | CTA/SM (runtime) |",
             "|---:|---|---:|---:|---|---:|---|---|---:|---:|---|---:|"]
    resources = result.get("runtime_resources", {})
    compiled = result.get("manifest", {}).get("variants", {})
    checks = {c["case"]: c for c in result.get("correctness", [])}
    for m, timings in result.get("timings", {}).items():
        check = checks.get(f"M{m}-N200704-identity", {})
        for arm in ARMS:
            row = timings.get(arm)
            if row is None:
                continue
            r = resources.get(arm, {})
            c = compiled.get(arm, {})
            parity = check.get("arms", {}).get(arm, {})
            o, lse = parity.get("output", {}), parity.get("lse", {})
            bits = f"{o.get('bitwise')}/{lse.get('bitwise')}"
            if not check.get("repeat_sha_match", {}).get(arm, True):
                bits += " (repeat changed; see JSON)"
            noise = row["noise_ratio_min_max"]
            upper = row.get("bootstrap_upper95_one_sided")
            upper_text = "—" if upper is None else f"{upper:.5f}"
            lines.append(
                f"| {m} | {LABELS[arm]} | {row['candidate_median_ms']:.4f} "
                f"| {row['relative_p0_percent']:+.2f} "
                f"| {noise[0]:.5f}–{noise[1]:.5f} | {upper_text} "
                f"| {bits} "
                f"| {o.get('max_abs_diff')}/{lse.get('max_abs_diff')} "
                f"| {r.get('reg')} | {r.get('shared')} "
                f"| {c.get('spill_store_bytes')}/{c.get('spill_load_bytes')} "
                f"| {r.get('ctas_per_sm')} |"
            )
    lines += ["", "P0 median pools the A samples of the four comparisons; its "
              "noise band is A2/A1. Each candidate uses its own paired P0 denominator.",
              "The JSON contains all event samples, paired denominators, A/A drift, "
              "absolute/relative differences, resource reports and artifact hashes.",
              "", "## Decisions", ""]
    for arm, verdict in result.get("decisions", {}).items():
        lines.append(f"- {LABELS[arm]}: {verdict}")
    lines += ["", "E2_PENDING means numerical/quality review is required; it is "
              "not a proof of unchanged sampling distribution."]
    if result.get("error"):
        lines += ["", "## Execution error", "", "```", result["error"], "```"]
    return "\n".join(lines) + "\n"


def save(result, output_dir):
    output_dir = Path(output_dir)
    for name, text in (("results.json", json.dumps(result, indent=2, allow_nan=False)
                       + "\n"), ("results.md", markdown(result))):
        temporary = output_dir / (name + ".tmp")
        temporary.write_text(text)
        temporary.replace(output_dir / name)
