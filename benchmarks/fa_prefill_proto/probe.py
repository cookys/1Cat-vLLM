# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GPU study. Default --describe validates artifacts entirely on CPU."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
import traceback
from pathlib import Path

import study
from resources import digest

INSTALLED_SHA = "011a6dcb07b2bf85de55494721c833e4f6194712c13b5d6e3b9c0ba9eb72cb8f"
MAIN_CASES = ((96, (200704,), False), (4032, (200704,), False))
CHECK_CASES = (
    (96, (200704,), True), (4032, (200704,), True),
    (33, (127,), False), (95, (785,), True), (129, (1583,), True),
    (33, (801, 799), True),
)


def checked_manifest(build):
    manifest = json.loads((build / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("five-arm manifest schema 2 required")
    if set(manifest["variants"]) != set(study.ARMS):
        raise ValueError("missing or unexpected arms")
    for arm, item in manifest["variants"].items():
        required = {"cu", "so", "resources.txt", "build.log", "sass"}
        if set(item["files_sha256"]) != required:
            raise ValueError("missing or unexpected artifact types")
        for suffix, expected in item["files_sha256"].items():
            path = build / f"proto{arm}.{suffix}"
            if digest(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        if (build / f"proto{arm}.so").resolve() != Path(item["library"]).resolve():
            raise ValueError("manifest library path mismatch")
    return manifest


def describe(build):
    return {
        "cpu_only": True, "manifest": checked_manifest(build),
        "arms": study.LABELS, "blocks": study.BLOCKS,
        "samples_per_phase": study.SAMPLES, "warmup": study.WARMUP,
        "main_cases": MAIN_CASES, "correctness_cases": CHECK_CASES,
        "D": 256, "Hq": 6, "Hkv": 1, "page_size": 784,
        "seeds_K_V_Q": [7101, 7102, 7103],
        "thresholds": {"4032": 0.9, "96": 1.0},
        "bootstrap_seed": study.BOOTSTRAP_SEED,
        "bootstrap_resamples": study.BOOTSTRAPS,
        "resource_gate_is_admission": False,
        "expected_installed_sha256": INSTALLED_SHA,
        "estimated_minutes": [5, 8],
        "scripts_sha256": {p.name: digest(p) for p in Path(__file__).parent.iterdir()
                           if p.suffix in (".py", ".sh", ".cuh")},
    }


def check_space():
    free = {p: shutil.disk_usage(p).free for p in ("/tmp", "/data/tmp")}
    if min(free.values()) < 4 * 1024**3:
        raise RuntimeError(f"need 4 GiB free in /tmp and /data/tmp: {free}")
    return free


class Library:
    """This class is instantiated only after the explicit GPU opt-in checks."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.fa_launch.argtypes = (
            [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5
            + [ctypes.c_int64] * 6 + [ctypes.c_float, ctypes.c_uint64]
        )
        self.lib.fa_launch.restype = ctypes.c_int
        self.lib.fa_resources.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
        self.lib.fa_resources.restype = ctypes.c_int

    def resources(self):
        values = [ctypes.c_int() for _ in range(4)]
        code = self.lib.fa_resources(*(ctypes.byref(v) for v in values))
        if code:
            raise RuntimeError(f"fa_resources CUDA error {code}")
        return dict(zip(("reg", "shared", "local", "ctas_per_sm"),
                        (v.value for v in values)))

    def launcher(self, tensors, output, lse, stream):
        q, k, v, table, lengths = tensors
        b, h, m, _ = q.shape
        arguments = (
            *(t.data_ptr() for t in (q, k, v, output, lse, table, lengths)),
            b, h, m, table.shape[1], k.shape[2],
            *k.stride()[:3], *v.stride()[:3], 0.0625, stream.cuda_stream,
        )

        def launch():
            code = self.lib.fa_launch(*arguments)
            if code:
                raise RuntimeError(f"fa_launch CUDA error {code}")

        return launch


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def compare(reference, actual):
    a, b = (t.detach().cpu().numpy() for t in (reference, actual))
    metrics = study.numerical_metrics(a, b, a.tobytes(), b.tobytes())
    metrics["reference_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
    metrics["actual_sha256"] = hashlib.sha256(b.tobytes()).hexdigest()
    return metrics


def make_inputs(torch, m, ns, shuffled):
    device = torch.device("cuda:0")
    b = len(ns)
    pages = math.ceil(max(ns) / 784)

    def rand(shape, seed):
        gen = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, generator=gen, dtype=torch.float16, device=device)

    k = rand((b * pages, 784, 1, 256), 7101)
    v = rand(k.shape, 7102)
    q_full = rand((b, 6, 4032, 256), 7103)
    q = q_full[:, :, -m:, :].contiguous()
    indices = list(range(b * pages))
    if shuffled:
        random.Random(8418).shuffle(indices)
    rows = [indices[i * pages:(i + 1) * pages] for i in range(b)]
    # Initialize all potentially fetched 16-token tail lanes to positive zero.
    for i, n in enumerate(ns):
        for slot, physical in enumerate(rows[i]):
            valid = max(0, min(784, n - slot * 784))
            k[physical, valid:].zero_()
            v[physical, valid:].zero_()
    table = torch.tensor(rows, dtype=torch.int32, device=device)
    lengths = torch.tensor(ns, dtype=torch.int32, device=device)
    return q, k, v, table, lengths


def case_name(m, ns, shuffled):
    n = str(ns[0]) if len(ns) == 1 else "-".join(map(str, ns))
    return f"M{m}-N{n}-{'shuffled' if shuffled else 'identity'}"


def correctness(torch, interface, libraries, tensors, stream, name):
    q, k, v, table, lengths = tensors
    outputs = {arm: (torch.full_like(q, float("nan")),
                     torch.full(q.shape[:3], float("nan"), dtype=torch.float32,
                                device=q.device)) for arm in study.ARMS}
    launchers = {arm: lib.launcher(tensors, *outputs[arm], stream)
                 for arm, lib in libraries.items()}
    installed_out = torch.full_like(q, float("nan"))
    installed_lse = torch.full(q.shape[:3], float("nan"), dtype=torch.float32,
                               device=q.device)
    returned = interface.flash_attn_prefill_paged_d256_bm32_allp_pair_scratch(
        q, k, v, table, lengths, softmax_scale=0.0625,
        out=installed_out, softmax_lse=installed_lse,
    )
    if [t.data_ptr() for t in returned] != [installed_out.data_ptr(),
                                          installed_lse.data_ptr()]:
        raise RuntimeError("installed baseline replaced preallocated output")
    for launch in launchers.values():
        launch()
    stream.synchronize()
    record = {"case": name, "arms": {}, "baseline_vs_installed": {}}
    for key, ref, actual in zip(("output", "lse"),
                                (installed_out, installed_lse), outputs["0"]):
        record["baseline_vs_installed"][key] = compare(ref, actual)
    for arm in study.ARMS:
        record["arms"][arm] = {
            key: compare(ref, val) for key, ref, val in
            zip(("output", "lse"), outputs["0"], outputs[arm])
        }
    return record, outputs, launchers


def run_gpu(args, result):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "0" or os.environ.get("FA_RUN_GPU") != "1":
        raise RuntimeError("requires FA_RUN_GPU=1 and CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("TMPDIR") != "/data/tmp":
        raise RuntimeError("TMPDIR must be /data/tmp")
    result["disk_free_bytes"] = check_space()
    # No torch, driver call, or library loading on the CPU describe path.
    import torch
    import flash_attn_v100.flash_attn_interface as interface

    extension = Path(interface.flash_attn_v100_cuda.__file__).resolve()
    result["installed_extension"] = str(extension)
    result["installed_sha256"] = digest(extension)
    if result["installed_sha256"] != INSTALLED_SHA:
        raise RuntimeError("installed baseline .so changed; review before profiling")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("exactly one visible SM70 GPU required")
    result.update(torch_version=str(torch.__version__),
                  device_name=torch.cuda.get_device_name(0),
                  cpu_only=False, started_unix_s=time.time())
    libraries = {arm: Library(args.build / f"proto{arm}.so") for arm in study.ARMS}
    for arm, lib in libraries.items():
        info = lib.resources()
        result["runtime_resources"][arm] = info
        compiled = result["manifest"]["variants"][arm]
        # Static LOCAL/STACK and runtime localSizeBytes may use different
        # accounting. Preserve both; resources are observations, not admission.
        info["compile_runtime_differences"] = {
            field: {"compiled": compiled[field], "runtime": info[field]}
            for field in ("reg", "shared", "local")
            if info[field] != compiled[field]
        }
        if info["ctas_per_sm"] < 1:
            raise RuntimeError(f"unlaunchable kernel {arm}")
    study.save(result, args.out)

    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for m, ns, shuffled in MAIN_CASES + CHECK_CASES:
            name = case_name(m, ns, shuffled)
            tensors = make_inputs(torch, m, ns, shuffled)
            before = {key: tensor_digest(t) for key, t in
                      zip(("Q", "K", "V", "table", "seq_lens"), tensors)}
            record, outputs, launchers = correctness(
                torch, interface, libraries, tensors, stream, name)
            record["inputs_sha256"] = before
            result["correctness"].append(record)
            study.save(result, args.out)
            timed = not shuffled and ns == (200704,)
            if timed:
                mkey = str(m)
                result["raw_timing"][mkey] = {}
                result["timings"][mkey] = {}
                for launch in launchers.values():
                    for _ in range(study.WARMUP):
                        launch()
                stream.synchronize()
                for arm in study.ARMS[1:]:
                    blocks = []
                    result["raw_timing"][mkey][arm] = blocks
                    for _ in range(study.BLOCKS):
                        phases = []
                        for label, choice in (("A", "0"), ("B", arm),
                                              ("B", arm), ("A", "0")):
                            times = []
                            for _ in range(study.SAMPLES):
                                start.record(stream)
                                launchers[choice]()
                                end.record(stream)
                                end.synchronize()
                                times.append(start.elapsed_time(end))
                            phases.append({"arm": label, "variant": choice,
                                           "event_ms": times})
                        blocks.append(phases)
                        study.save(result, args.out)
                    result["timings"][mkey][arm] = study.summarize(blocks)
                    study.save(result, args.out)
                all_a = [t for blocks in result["raw_timing"][mkey].values()
                         for block in blocks for p in block if p["arm"] == "A"
                         for t in p["event_ms"]]
                aa = [r for row in result["timings"][mkey].values()
                      for r in row["aa_phase_ratios"]]
                result["timings"][mkey]["0"] = {
                    "candidate_median_ms": statistics.median(all_a),
                    "relative_p0_percent": 0.0,
                    "noise_ratio_min_max": [min(aa), max(aa)],
                    "aa_phase_ratios": aa,
                }
                # Detect result drift or errors introduced by repeated execution.
                record["after_timing"] = {
                    arm: {key: compare(ref, actual) for key, ref, actual in
                          zip(("output", "lse"), outputs["0"], outputs[arm])}
                    for arm in study.ARMS
                }
                record["repeat_sha_match"] = {
                    arm: all(record["after_timing"][arm][key]["actual_sha256"]
                             == record["arms"][arm][key]["actual_sha256"]
                             for key in ("output", "lse"))
                    for arm in study.ARMS
                }
            record["inputs_unchanged"] = all(
                tensor_digest(t) == before[key] for key, t in
                zip(("Q", "K", "V", "table", "seq_lens"), tensors)
            )
            if not record["inputs_unchanged"]:
                raise RuntimeError("kernel modified a read-only input")
            study.save(result, args.out)
            del tensors, outputs, launchers

    checks = result["correctness"]
    baseline_match = all(c["baseline_vs_installed"][key]["bitwise"]
                         and c["baseline_vs_installed"][key]["finite"]
                         and c.get("repeat_sha_match", {}).get("0", True)
                         for c in checks for key in ("output", "lse"))
    for arm in study.ARMS[1:]:
        parity = all(c["arms"][arm][key]["bitwise"]
                     and c.get("repeat_sha_match", {}).get(arm, True)
                     for c in checks for key in ("output", "lse"))
        finite = all(c["arms"][arm][key]["finite"]
                     and c.get("after_timing", {}).get(arm, {})
                     .get(key, {}).get("finite", True)
                     for c in checks for key in ("output", "lse"))
        result["decisions"][arm] = study.classify(
            {m: result["timings"][m][arm] for m in ("96", "4032")},
            parity, finite, baseline_match)
    result.update(status="COMPLETE", finished_unix_s=time.time(),
                  baseline_matches_installed=baseline_match)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--run-on-gpu", action="store_true")
    p.add_argument("--describe", action="store_true")
    p.add_argument("--check-space", action="store_true")
    args = p.parse_args()
    plan = describe(args.build.resolve())
    if args.check_space:
        plan["disk_free_bytes"] = check_space()
    if not args.run_on_gpu or args.describe:
        print(json.dumps(plan, indent=2))
        return
    if args.out is None or not args.out.is_dir():
        p.error("--out must be an existing fresh directory created by profile.sh")
    if (args.out / "results.json").exists():
        p.error("refusing to overwrite results.json")
    result = {**plan, "status": "RUNNING", "runtime_resources": {},
              "correctness": [], "raw_timing": {}, "timings": {}, "decisions": {}}
    study.save(result, args.out)
    try:
        run_gpu(args, result)
    except BaseException:
        result.update(status="FAILED", error=traceback.format_exc())
        study.save(result, args.out)
        raise
    study.save(result, args.out)
    print(args.out / "results.md")


if __name__ == "__main__":
    main()
