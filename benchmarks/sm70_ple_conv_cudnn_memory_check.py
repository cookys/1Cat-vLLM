# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does the Qwen4Exp PLE prefill short-conv depend on free GPU memory?

Hypothesis under test: ``_short_conv_dilated_prefill_batched`` in
``vllm/models/qwen4_exp/nvidia/ple_layer.py`` runs

    F.conv1d(history, conv_weights.unsqueeze(1).contiguous(),
             groups=history.size(1), dilation=self.short_conv_dilation)

eagerly inside an opaque custom op. 1Cat sets neither
``torch.backends.cudnn.benchmark`` nor ``deterministic``, so, if the call reaches
cuDNN, PyTorch takes the first heuristic engine whose workspace allocation
succeeds. The same input would then give bit-different outputs depending on how
much GPU memory is free when the conv runs, and a text-parity regression after a
memory-reduction patch would look exactly like that.

What this script measures, with torch only (no vLLM engine), one subprocess per
cuDNN setting so that no plan or benchmark cache leaks between settings:

1. The real geometry: ``hidden_size * hc_count`` channels, ``ple_conv_kernel_size``
   taps and ``ngram_size`` as the dilation, read from the model config
   (``text_config`` or the top level), i.e. the keys ``ple_layer.py`` uses. The
   production values are 10240 channels, 4 taps, dilation 3, so a conv state of
   9 positions.
2. A seeded input ``[num_prefills, channels, max_len + conv_state_len]`` and
   weights ``[channels, 1, kernel_size]`` (contiguous, as production packs the
   tokens), convolved exactly as production does.
3. For each free-memory scenario (whole free, then 16, 8, 4, 2, 1 GiB and 512 MiB
   free at the moment the conv is called): a dummy tensor takes the rest of the
   memory after the inputs are resident; the conv runs 3 times; the script records
   the sha256 of each output, whether the repeats are identical, the max abs
   difference to the first scenario and to a pure-PyTorch fp32 reference. Scenarios
   that cannot be reached are skipped, and a conv that cannot allocate at all is
   recorded as ``oom``.
4. Four cuDNN settings: default (``benchmark=False``, ``deterministic=False``),
   ``deterministic=True``, ``benchmark=True`` and a workspace cap through the
   ``CUDNN_CONV_WSCAP_DBG`` environment variable (256 and 64 MB), plus the
   reference conv, which is deterministic by construction.
5. Which backend and algorithm ran, per scenario, in a subprocess of its own:
   ``torch._C._select_conv_backend`` for the backend, the CUDA kernel names from
   ``torch.profiler``, and the cuDNN API and frontend logs
   (``CUDNN_LOGLEVEL_DBG=3``, ``CUDNN_LOGDEST_DBG``, ``CUDNN_FRONTEND_LOG_INFO``)
   parsed for ``cudnnConvolutionForward`` algorithms and execution-plan engine
   tags. If a cuDNN build emits nothing, the output says so.

Expectation, not yet verified on this host: PyTorch's ``use_cudnn_depthwise``
limits the cuDNN depthwise path to undilated convolutions, so a dilated depthwise
fp16 conv (dilation 3 here) would go to PyTorch's own ``conv_depthwise2d`` CUDA
kernel and never reach cuDNN. Check 5 settles it for the installed build and the
verdict states ``CUDNN_USED``. If the backend is not cuDNN, none of the cuDNN
settings can change the result and the memory-dependence hypothesis is false for
this call by construction; the hash comparison still decides it empirically.

Verdict printed on stdout (exit status 0 for a clean run whatever the verdict):

    CUDNN_USED: YES/NO/UNKNOWN     backend per ``torch._C._select_conv_backend``
    MEMORY_DEPENDENT: YES/NO       any scenario hash differs, default setting
    REPEATS_IDENTICAL: YES/NO      the 3 repeats within every scenario agree
    DETERMINISTIC_FLAG_FIXES: YES/NO/N/A
    BENCHMARK_FLAG_FIXES: YES/NO/N/A
    WSCAP_FIXES: YES/NO/N/A        per cap; N/A if the default is not dependent
    ALGO: <setting> <max_len> <scenario>: <algorithm or kernel>

Exit status: 0 a clean run; 2 no CUDA device; 1 errors (a worker that failed).
An unreadable config is not an error: the production geometry is used and the
output says so.

    CUDA_VISIBLE_DEVICES=<idle gpu> /data/venvs/1cat-m589/bin/python \
        benchmarks/sm70_ple_conv_cudnn_memory_check.py \
        --out /data/bench/ple_conv_cudnn_memory_check.json

Use one idle GPU: the scenarios take memory away from the process that runs the
conv, and any other process on the device would move the baseline. The run takes
minutes with the default ``--algo-settings default`` (10 scenario workers plus 14
one-conv workers, each a fresh Python process); ``--algo-settings all`` adds 56
more.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_CONFIG = "/data/models/Qwen3.8-Flash-Next-NVFP4/config.json"
# Production values, used when the config has no PLE conv keys and no override.
PRODUCTION_GEOMETRY = {"channels": 10240, "kernel_size": 4, "dilation": 3}
MIB = 1 << 20
GIB = 1 << 30
DEFAULT_FREE_MIB = (16384, 8192, 4096, 2048, 1024, 512)
# A scenario whose target is within this much of the current free memory is the
# whole-memory scenario again; skip it.
MIN_DUMMY_BYTES = 32 * MIB
RESULT_PREFIX = "RESULT_JSON:"
# The device the workers run on; the CPU tests point it at "cpu".
WORKER_DEVICE = "cuda:0"
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ------------------------------------------------------------------ geometry


@dataclass(frozen=True)
class Geometry:
    channels: int
    kernel_size: int
    dilation: int
    source: str

    @property
    def conv_state_len(self) -> int:
        return (self.kernel_size - 1) * self.dilation


def read_geometry_from_config(path: str | Path) -> dict[str, int] | None:
    """PLE short-conv geometry from a model config.json, or None if absent.

    ``ple_layer.py`` reads ``config.hidden_size`` and ``config.hc_count`` (the conv
    runs over ``hidden_size * hc_count`` channels), ``config.ple_conv_kernel_size``
    and ``config.ngram_size`` (the dilation).
    """
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    for section in (config.get("text_config"), config):
        if not isinstance(section, dict):
            continue
        keys = ("hidden_size", "hc_count", "ple_conv_kernel_size", "ngram_size")
        if all(isinstance(section.get(k), int) for k in keys):
            return {
                "channels": section["hidden_size"] * section["hc_count"],
                "kernel_size": section["ple_conv_kernel_size"],
                "dilation": section["ngram_size"],
            }
    return None


def resolve_geometry(args: argparse.Namespace) -> Geometry:
    """Overrides win; then the config; then the production values."""
    from_config = read_geometry_from_config(args.config)
    values = dict(PRODUCTION_GEOMETRY)
    source = "production defaults (config unreadable or without PLE conv keys)"
    if from_config is not None:
        values.update(from_config)
        source = f"config {args.config}"
    overridden = []
    for name in ("channels", "kernel_size", "dilation"):
        override = getattr(args, name)
        if override is not None:
            values[name] = override
            overridden.append(name)
    if overridden:
        source += f", overridden: {', '.join(overridden)}"
    return Geometry(source=source, **values)


# ------------------------------------------------------------------ settings


def flags_for(setting: str) -> tuple[bool, bool]:
    """(benchmark, deterministic) of a setting; the wscap settings use the defaults."""
    return {"benchmark": (True, False), "deterministic": (False, True)}.get(
        setting, (False, False)
    )


def build_settings(wscap_mb: Sequence[int]) -> dict[str, dict[str, Any]]:
    """The cuDNN settings to compare. ``env`` is applied to the worker process."""
    settings: dict[str, dict[str, Any]] = {
        name: {
            "benchmark": flags_for(name)[0],
            "deterministic": flags_for(name)[1],
            "env": {},
        }
        for name in ("default", "deterministic", "benchmark")
    }
    for cap in wscap_mb:
        settings[f"wscap{cap}"] = {
            "benchmark": False,
            "deterministic": False,
            "env": {"CUDNN_CONV_WSCAP_DBG": str(cap)},
        }
    return settings


# ----------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    # Free memory wanted at the moment the conv is called; None means "whole".
    target_free_bytes: int | None
    dummy_bytes: int
    reachable: bool
    reason: str = ""


def scenario_name(target_free_bytes: int | None) -> str:
    if target_free_bytes is None:
        return "whole"
    if target_free_bytes % GIB == 0:
        return f"free{target_free_bytes // GIB}GiB"
    return f"free{target_free_bytes // MIB}MiB"


def plan_scenario(free_bytes: int, target_free_bytes: int | None) -> ScenarioPlan:
    """Dummy size that leaves ``target_free_bytes`` free, given ``free_bytes`` now."""
    name = scenario_name(target_free_bytes)
    if target_free_bytes is None:
        return ScenarioPlan(name, None, 0, True)
    dummy = free_bytes - target_free_bytes
    if dummy < MIN_DUMMY_BYTES:
        return ScenarioPlan(
            name,
            target_free_bytes,
            0,
            False,
            f"free memory {free_bytes // MIB} MiB leaves nothing to take away "
            f"for a {target_free_bytes // MIB} MiB target",
        )
    return ScenarioPlan(name, target_free_bytes, dummy, True)


def plan_scenarios(
    mem_get_info: Callable[[], tuple[int, int]],
    free_mib: Sequence[int] = DEFAULT_FREE_MIB,
) -> list[ScenarioPlan]:
    """The scenario list for the free memory ``mem_get_info`` reports now.

    The runner re-plans each scenario against the free memory it sees at that
    moment, because the earlier scenarios free their tensors and the caching
    allocator keeps blocks; this list is for planning and for tests.
    """
    free_bytes, _ = mem_get_info()
    targets: list[int | None] = [None, *[m * MIB for m in free_mib]]
    return [plan_scenario(free_bytes, t) for t in targets]


# ------------------------------------------------------------ reference conv


def reference_depthwise_dilated_conv(
    x: torch.Tensor, weight: torch.Tensor, dilation: int
) -> torch.Tensor:
    """Depthwise dilated 1-D convolution, fp32 accumulation, fixed tap order.

    ``x`` is ``[N, C, L]``, ``weight`` is ``[C, 1, K]``; the result is fp32
    ``[N, C, L - dilation * (K - 1)]``. Each output is the sum over the taps
    ``k = 0..K-1`` of ``weight[c, 0, k] * x[n, c, t + k * dilation]``, added in
    that order, so the result does not depend on a backend's algorithm choice.
    """
    channels, _, taps = weight.shape
    out_len = x.shape[-1] - dilation * (taps - 1)
    if out_len <= 0:
        raise ValueError("input is shorter than the dilated kernel")
    xf = x.float()
    wf = weight.float().reshape(channels, taps)
    acc = torch.zeros(
        x.shape[0], channels, out_len, dtype=torch.float32, device=x.device
    )
    for k in range(taps):
        acc += xf[:, :, k * dilation : k * dilation + out_len] * wf[:, k].view(
            1, channels, 1
        )
    return acc


# ------------------------------------------------------------------ measures


def sha256_tensor(t: torch.Tensor) -> str:
    """sha256 of the raw bytes of a (CPU or CUDA) tensor."""
    cpu = t.detach().cpu().contiguous().reshape(-1)
    raw = cpu.view(torch.uint8) if cpu.element_size() > 1 else cpu.to(torch.uint8)
    return hashlib.sha256(raw.numpy()).hexdigest()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 24) -> float:
    """Max |a - b| in fp32, in chunks so the temporaries stay small."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    worst = 0.0
    for start in range(0, fa.numel(), chunk):
        diff = fa[start : start + chunk].float() - fb[start : start + chunk].float()
        worst = max(worst, float(diff.abs().max()))
    return worst


# ----------------------------------------------------------- cuDNN log parse

_LEGACY_ALGO = re.compile(r"CUDNN_CONVOLUTION_FWD_ALGO_[A-Z0-9_]+")
# cudnn_frontend execution-plan tag, e.g. "eng14_k2=2_k3=0_k6=1".
_ENGINE_TAG = re.compile(r"\beng\d+(?:_k\d+=\d+)*")
_ENGINE_GLOBAL_INDEX = re.compile(
    r"CUDNN_ATTR_ENGINE_GLOBAL_INDEX[^\n]*\n(?:[^\n]*\n){0,4}?"
    r"[^\n]*arrayOfElements[^\n]*val=\[?\s*(-?\d+)"
)
_CALL_FORWARD = re.compile(r"cudnnConvolutionForward\s*\(")
_CALL_BACKEND_EXECUTE = re.compile(r"cudnnBackendExecute\s*\(")


def _distinct(items: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def parse_cudnn_log(text: str) -> dict[str, Any]:
    """Convolution algorithm information from a cuDNN API or frontend log.

    Understands the legacy ``cudnnConvolutionForward`` algorithm enum, the
    cudnn_frontend execution-plan tags (``eng<N>_k...``) and the engine global
    index set through ``cudnnBackendSetAttribute``. ``has_algorithm_info`` is False
    when the log carries none of them (the build emitted nothing, or cuDNN was not
    used).
    """
    legacy = _distinct(_LEGACY_ALGO.findall(text))
    tags = _distinct(_ENGINE_TAG.findall(text))
    indices = _distinct(_ENGINE_GLOBAL_INDEX.findall(text))
    return {
        "legacy_algos": legacy,
        "engine_tags": tags,
        "engine_global_indices": indices,
        "convolution_forward_calls": len(_CALL_FORWARD.findall(text)),
        "backend_execute_calls": len(_CALL_BACKEND_EXECUTE.findall(text)),
        "has_algorithm_info": bool(legacy or tags or indices),
    }


def describe_algorithm(parsed: dict[str, Any] | None, kernels: list[str] | None) -> str:
    """One line naming the algorithm: cuDNN log first, then the CUDA kernel."""
    if parsed is not None and parsed.get("has_algorithm_info"):
        for key in ("legacy_algos", "engine_tags", "engine_global_indices"):
            if parsed[key]:
                return f"cudnn {key}={','.join(parsed[key])}"
    if kernels:
        return f"kernel {kernels[0]}"
    return "unknown (no cuDNN log entry and no kernel name)"


# --------------------------------------------------------------- verdict


def analyze_setting(per_max_len: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cross-scenario summary of one setting's worker results."""
    summary: dict[str, Any] = {"max_len": {}, "all_identical": True}
    summary["repeats_identical"] = True
    for max_len, result in per_max_len.items():
        ok = [s for s in result["scenarios"] if s["status"] == "ok"]
        first_hashes = {s["sha256"][0] for s in ok}
        repeats = all(s["repeats_identical"] for s in ok)
        identical = len(first_hashes) <= 1
        summary["max_len"][max_len] = {
            "scenarios_run": len(ok),
            "distinct_output_hashes": len(first_hashes),
            "identical_across_scenarios": identical,
            "repeats_identical": repeats,
            "oom_scenarios": [
                s["name"] for s in result["scenarios"] if s["status"] == "oom"
            ],
            "error_scenarios": [
                s["name"] for s in result["scenarios"] if s["status"] == "error"
            ],
        }
        summary["all_identical"] &= identical
        summary["repeats_identical"] &= repeats
    return summary


def _yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def compute_verdict(
    results: dict[str, dict[str, dict[str, Any]]],
    backends: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verdict lines from the per-setting, per-max_len worker results.

    ``results[setting][max_len]`` is a worker result. A setting "fixes" the
    dependence when the default setting is memory dependent and this one gives the
    same output in every reachable scenario; it is N/A when the default is not
    memory dependent.
    """
    summaries = {s: analyze_setting(per) for s, per in results.items()}
    default = summaries.get("default")
    verdict: dict[str, Any] = {"settings": summaries}

    cudnn_used = "UNKNOWN"
    if backends:
        used = [b.get("cudnn_used") for b in backends.values() if b is not None]
        if used and all(u is not None for u in used):
            cudnn_used = _yes_no(any(used))
    verdict["CUDNN_USED"] = cudnn_used

    if default is None:
        verdict["MEMORY_DEPENDENT"] = "UNKNOWN"
        verdict["REPEATS_IDENTICAL"] = "UNKNOWN"
        return verdict
    dependent = not default["all_identical"]
    verdict["MEMORY_DEPENDENT"] = _yes_no(dependent)
    verdict["REPEATS_IDENTICAL"] = _yes_no(
        all(s["repeats_identical"] for s in summaries.values())
    )

    def fixes(setting: str) -> str:
        if setting not in summaries:
            return "UNKNOWN"
        if not dependent:
            return "N/A"
        return _yes_no(summaries[setting]["all_identical"])

    verdict["DETERMINISTIC_FLAG_FIXES"] = fixes("deterministic")
    verdict["BENCHMARK_FLAG_FIXES"] = fixes("benchmark")
    verdict["WSCAP_FIXES"] = {s: fixes(s) for s in summaries if s.startswith("wscap")}
    # Whether each setting reproduces the default setting's whole-memory bytes.
    whole = {}
    for setting, per in results.items():
        hashes = []
        for result in per.values():
            for s in result["scenarios"]:
                if s["name"] == "whole" and s["status"] == "ok":
                    hashes.append(s["sha256"][0])
        whole[setting] = hashes
    verdict["SAME_BYTES_AS_DEFAULT_WHOLE"] = {
        s: (whole[s] == whole["default"]) if "default" in whole and whole[s] else None
        for s in whole
    }
    return verdict


# ------------------------------------------------------------- workers


def build_worker_command(
    script: str,
    mode: str,
    setting: str,
    max_len: int,
    args: argparse.Namespace,
    geometry: Geometry,
    scenario: str | None = None,
) -> list[str]:
    """Command line of one worker subprocess; the geometry is passed explicitly."""
    command = [
        sys.executable,
        script,
        "--worker",
        mode,
        "--setting",
        setting,
        "--worker-max-len",
        str(max_len),
        "--channels",
        str(geometry.channels),
        "--kernel-size",
        str(geometry.kernel_size),
        "--dilation",
        str(geometry.dilation),
        "--dtype",
        args.dtype,
        "--num-prefills",
        str(args.num_prefills),
        "--seed",
        str(args.seed),
        "--repeats",
        str(args.repeats),
        "--free-mib",
        *[str(m) for m in args.free_mib],
    ]
    if scenario is not None:
        command += ["--worker-scenario", scenario]
    return command


def build_worker_env(
    setting_env: dict[str, str], log_files: tuple[str, str] | None = None
) -> dict[str, str]:
    """Environment of a worker: the setting's variables and, optionally, cuDNN logs."""
    env = dict(os.environ)
    env.update(setting_env)
    if log_files is not None:
        api_log, frontend_log = log_files
        env["CUDNN_LOGLEVEL_DBG"] = "3"
        env["CUDNN_LOGINFO_DBG"] = "1"
        env["CUDNN_LOGDEST_DBG"] = api_log
        env["CUDNN_FRONTEND_LOG_INFO"] = "1"
        env["CUDNN_FRONTEND_LOG_FILE"] = frontend_log
    return env


def extract_result(stdout: str) -> dict[str, Any]:
    """The JSON a worker printed after ``RESULT_JSON:`` (the last such line)."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise ValueError("worker printed no result line")


def run_worker(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-12:]
        raise RuntimeError(
            f"worker exited {completed.returncode}: {' '.join(command[3:9])}\n"
            + "\n".join(tail)
        )
    return extract_result(completed.stdout)


def _select_backend(
    history: torch.Tensor, weights: torch.Tensor, geometry: Geometry
) -> dict[str, Any]:
    """Which backend PyTorch picks for the production conv call."""
    info: dict[str, Any] = {
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "selected": None,
        "cudnn_used": None,
    }
    # F.conv1d expands a 1-D conv to 2-D (view1d_as_2d) before it selects the
    # backend, so ask about the 4-D form; the 3-D answer is kept for comparison.
    forms = {
        "selected": (
            history.unsqueeze(2),
            weights.unsqueeze(2),
            [1, 1],
            [0, 0],
            [1, geometry.dilation],
            [0, 0],
        ),
        "selected_as_1d": (history, weights, [1], [0], [geometry.dilation], [0]),
    }
    for key, (inp, wgt, stride, padding, dilation, out_padding) in forms.items():
        try:
            backend = torch._C._select_conv_backend(
                inp,
                wgt,
                None,
                stride,
                padding,
                dilation,
                False,
                out_padding,
                geometry.channels,
            )
        except Exception as exc:  # noqa: BLE001 - introspection is best effort
            info[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
            continue
        info[key] = str(backend)
    if info["selected"] is not None:
        info["cudnn_used"] = "Cudnn" in info["selected"]
    return info


def _make_inputs(
    geometry: Geometry, dtype: torch.dtype, num_prefills: int, max_len: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded CPU inputs, so every process builds the same bytes."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    length = max_len + geometry.conv_state_len
    history = torch.randn(
        num_prefills, geometry.channels, length, generator=generator
    ).to(dtype)
    weights = (
        torch.randn(geometry.channels, 1, geometry.kernel_size, generator=generator)
        * 0.5
    ).to(dtype)
    return history.contiguous(), weights.contiguous()


def _conv(history: torch.Tensor, weights: torch.Tensor, dilation: int) -> torch.Tensor:
    # Exactly the production call.
    return F.conv1d(history, weights, groups=history.size(1), dilation=dilation)


def _apply_setting(setting: str) -> None:
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags_for(
        setting
    )


def _device_info(device: torch.device) -> dict[str, Any]:
    """Name and capability of the worker's device (the driver never opens CUDA)."""
    if device.type != "cuda":
        return {"name": str(device), "capability": None}
    return {
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def _failure_status(exc: RuntimeError) -> str:
    """oom for an allocator failure, error for any other conv failure."""
    return "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"


def _mem_get_info(device: torch.device) -> tuple[int, int]:
    return torch.cuda.mem_get_info(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _take_free_memory(device: torch.device, target: int | None):
    """Plan against the memory free now and allocate the dummy; (plan, dummy, free)."""
    torch.cuda.empty_cache()
    free, _ = _mem_get_info(device)
    plan = plan_scenario(free, target)
    dummy = None
    if plan.reachable and plan.dummy_bytes > 0:
        dummy = torch.empty(plan.dummy_bytes, dtype=torch.uint8, device=device)
    return plan, dummy, _mem_get_info(device)[0]


def worker_scenarios(args: argparse.Namespace, geometry: Geometry) -> dict[str, Any]:
    """All free-memory scenarios of one (setting, max_len), in this process."""
    device = torch.device(WORKER_DEVICE)
    dtype = DTYPES[args.dtype]
    _apply_setting(args.setting)
    max_len = args.worker_max_len
    history_cpu, weights_cpu = _make_inputs(
        geometry, dtype, args.num_prefills, max_len, args.seed
    )
    reference = reference_depthwise_dilated_conv(
        history_cpu, weights_cpu, geometry.dilation
    )
    history = history_cpu.to(device)
    weights = weights_cpu.to(device)
    del history_cpu, weights_cpu
    backend = _select_backend(history, weights, geometry)

    targets: list[int | None] = [None, *[m * MIB for m in args.free_mib]]
    first_output: torch.Tensor | None = None
    scenarios = []
    for target in targets:
        plan, dummy, free_before = _take_free_memory(device, target)
        record: dict[str, Any] = {
            "name": plan.name,
            "target_free_mib": None if target is None else target // MIB,
            "dummy_mib": plan.dummy_bytes // MIB,
            "free_mib_before_conv": free_before // MIB,
        }
        if not plan.reachable:
            record.update(status="skipped", reason=plan.reason)
            scenarios.append(record)
            continue
        hashes: list[str] = []
        status = "ok"
        scenario_first: torch.Tensor | None = None
        for _ in range(args.repeats):
            try:
                out = _conv(history, weights, geometry.dilation)
                _synchronize(device)
            except RuntimeError as exc:
                # An allocator failure, or a cuDNN error when no engine fits.
                status = _failure_status(exc)
                record["reason"] = str(exc).splitlines()[0]
                break
            cpu = out.cpu()
            del out
            hashes.append(sha256_tensor(cpu))
            if scenario_first is None:
                scenario_first = cpu
        record["status"] = status
        if status == "ok" and scenario_first is not None:
            if first_output is None:
                first_output = scenario_first
            record.update(
                sha256=hashes,
                repeats_identical=len(set(hashes)) == 1,
                max_abs_diff_vs_first_scenario=max_abs_diff(
                    scenario_first, first_output
                ),
                max_abs_diff_vs_reference_fp32=max_abs_diff(scenario_first, reference),
            )
        scenarios.append(record)
        del dummy
        torch.cuda.empty_cache()
    return {
        "setting": args.setting,
        "max_len": max_len,
        "device": _device_info(device),
        "backend": backend,
        "env": {k: v for k, v in os.environ.items() if k.startswith("CUDNN_")},
        "torch_cudnn": {
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
        },
        "scenarios": scenarios,
    }


def _kernel_names(run: Callable[[], Any]) -> list[str] | None:
    """Run ``run`` once and return the CUDA kernel names torch.profiler saw.

    ``run`` always executes, with or without a profiler: if the profiler cannot
    start (no CUPTI, a CPU build) the names are None and the conv still runs. An
    exception from ``run`` itself, such as an out-of-memory error, propagates.
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CUDA])
        prof = profiler.__enter__()
    except Exception:  # noqa: BLE001 - the profiler is best effort
        run()
        return None
    try:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        try:
            profiler.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            return None
    try:
        names = [
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and "Memcpy" not in event.name
            and "Memset" not in event.name
        ]
    except Exception:  # noqa: BLE001
        return None
    return _distinct([n[:160] for n in names])


def worker_algo(args: argparse.Namespace, geometry: Geometry) -> dict[str, Any]:
    """One scenario, two convs: the logged selection and a profiled repeat."""
    device = torch.device(WORKER_DEVICE)
    dtype = DTYPES[args.dtype]
    _apply_setting(args.setting)
    history_cpu, weights_cpu = _make_inputs(
        geometry, dtype, args.num_prefills, args.worker_max_len, args.seed
    )
    history = history_cpu.to(device)
    weights = weights_cpu.to(device)
    del history_cpu, weights_cpu
    backend = _select_backend(history, weights, geometry)
    by_name = {
        scenario_name(None if m is None else m * MIB): m for m in [None, *args.free_mib]
    }
    target_mib = by_name[args.worker_scenario]
    plan, dummy, free_before = _take_free_memory(
        device, None if target_mib is None else target_mib * MIB
    )
    record: dict[str, Any] = {
        "setting": args.setting,
        "max_len": args.worker_max_len,
        "scenario": args.worker_scenario,
        "device": _device_info(device),
        "backend": backend,
        "free_mib_before_conv": free_before // MIB,
        "status": "ok" if plan.reachable else "skipped",
    }
    if plan.reachable:
        try:
            holder: list[torch.Tensor] = []
            record["kernels_run1"] = _kernel_names(
                lambda: holder.append(_conv(history, weights, geometry.dilation))
            )
            record["kernels_run2"] = _kernel_names(
                lambda: holder.append(_conv(history, weights, geometry.dilation))
            )
            record["sha256"] = sha256_tensor(holder[0])
            record["run2_matches_run1"] = (
                len(holder) > 1 and sha256_tensor(holder[1]) == record["sha256"]
            )
        except RuntimeError as exc:
            record["status"] = _failure_status(exc)
            record["reason"] = str(exc).splitlines()[0]
    del dummy
    return record


# ---------------------------------------------------------------- driver


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--channels", type=int, help="override the config")
    parser.add_argument("--kernel-size", type=int, help="override the config")
    parser.add_argument("--dilation", type=int, help="override the config")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--num-prefills", type=int, default=2)
    parser.add_argument(
        "--max-len",
        type=int,
        nargs="+",
        default=[2048, 8192],
        help="packed prefill widths to run (history is max_len + conv_state_len)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--free-mib",
        type=int,
        nargs="+",
        default=list(DEFAULT_FREE_MIB),
        help="free memory left at the conv, in MiB, besides the whole-memory run",
    )
    parser.add_argument(
        "--wscap-mb",
        type=int,
        nargs="*",
        default=[256, 64],
        help="CUDNN_CONV_WSCAP_DBG values (MB) to try",
    )
    parser.add_argument(
        "--settings",
        default="all",
        help="comma list of default,deterministic,benchmark,wscap<N>, or all",
    )
    parser.add_argument(
        "--algo-settings",
        default="default",
        help="settings to identify the algorithm for, per scenario: a comma list, "
        "all, or none",
    )
    parser.add_argument("--out", help="write the JSON report here")
    # Internal: one worker subprocess.
    parser.add_argument(
        "--worker", choices=("scenarios", "algo"), help=argparse.SUPPRESS
    )
    parser.add_argument("--setting", help=argparse.SUPPRESS)
    parser.add_argument("--worker-max-len", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scenario", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.num_prefills < 1 or any(m < 1 for m in args.max_len):
        parser.error("--num-prefills and --max-len must be positive")
    return args


def _select_names(spec: str, available: Sequence[str]) -> list[str]:
    if spec == "none":
        return []
    if spec == "all":
        return list(available)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown setting(s) {unknown}; available: {list(available)}")
    return names


def run_driver(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("needs a CUDA device (set CUDA_VISIBLE_DEVICES to an idle GPU)")
        return 2
    geometry = resolve_geometry(args)
    settings = build_settings(args.wscap_mb)
    chosen = _select_names(args.settings, list(settings))
    algo_names = _select_names(args.algo_settings, chosen)
    script = os.path.abspath(__file__)
    # The driver never opens CUDA: its context would take memory from the GPU the
    # workers measure. The device is reported from the first worker's result.
    print(
        f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}; "
        f"geometry channels={geometry.channels} kernel={geometry.kernel_size} "
        f"dilation={geometry.dilation} conv_state_len={geometry.conv_state_len} "
        f"({geometry.source})"
    )
    device_info: dict[str, Any] | None = None

    results: dict[str, dict[str, dict[str, Any]]] = {}
    backends: dict[str, Any] = {}
    errors: list[str] = []
    for name in chosen:
        results[name] = {}
        for max_len in args.max_len:
            command = build_worker_command(
                script, "scenarios", name, max_len, args, geometry
            )
            print(f"scenario worker: setting={name} max_len={max_len}", flush=True)
            try:
                result = run_worker(command, build_worker_env(settings[name]["env"]))
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                errors.append(f"{name}/{max_len}: {exc}")
                print(f"  FAILED: {exc}")
                continue
            results[name][str(max_len)] = result
            backends[f"{name}/{max_len}"] = result["backend"]
            if device_info is None:
                device_info = result.get("device")
                print(f"device {device_info}")

    algo: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="ple_conv_cudnn_") as tmp:
        for name in algo_names:
            algo[name] = {}
            for max_len in args.max_len:
                base = results.get(name, {}).get(str(max_len))
                if base is None:
                    continue
                algo[name][str(max_len)] = {}
                for scenario in base["scenarios"]:
                    if scenario["status"] != "ok":
                        continue
                    api_log = os.path.join(
                        tmp, f"{name}_{max_len}_{scenario['name']}.api"
                    )
                    fe_log = os.path.join(
                        tmp, f"{name}_{max_len}_{scenario['name']}.fe"
                    )
                    command = build_worker_command(
                        script, "algo", name, max_len, args, geometry, scenario["name"]
                    )
                    try:
                        record = run_worker(
                            command,
                            build_worker_env(settings[name]["env"], (api_log, fe_log)),
                        )
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            f"algo {name}/{max_len}/{scenario['name']}: {exc}"
                        )
                        continue
                    text = ""
                    for path in (api_log, fe_log):
                        if os.path.exists(path):
                            text += Path(path).read_text(errors="replace")
                    record["cudnn_log_bytes"] = len(text)
                    record["cudnn_log"] = parse_cudnn_log(text)
                    record["algorithm"] = describe_algorithm(
                        record["cudnn_log"],
                        record.get("kernels_run2") or record.get("kernels_run1"),
                    )
                    algo[name][str(max_len)][scenario["name"]] = record

    verdict = compute_verdict(results, backends)
    report = {
        "device": device_info,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "geometry": {**asdict(geometry), "conv_state_len": geometry.conv_state_len},
        "dtype": args.dtype,
        "num_prefills": args.num_prefills,
        "max_len": args.max_len,
        "seed": args.seed,
        "repeats": args.repeats,
        "settings": {k: v for k, v in settings.items() if k in chosen},
        "results": results,
        "algo": algo,
        "errors": errors,
        "verdict": verdict,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print_verdict(verdict, algo, errors)
    return 1 if errors else 0


def print_verdict(
    verdict: dict[str, Any], algo: dict[str, Any], errors: list[str]
) -> None:
    print()
    for key in ("CUDNN_USED", "MEMORY_DEPENDENT", "REPEATS_IDENTICAL"):
        print(f"{key}: {verdict[key]}")
    for key in ("DETERMINISTIC_FLAG_FIXES", "BENCHMARK_FLAG_FIXES"):
        if key in verdict:
            print(f"{key}: {verdict[key]}")
    for setting, value in verdict.get("WSCAP_FIXES", {}).items():
        print(f"WSCAP_FIXES ({setting}): {value}")
    for setting, same in verdict.get("SAME_BYTES_AS_DEFAULT_WHOLE", {}).items():
        print(f"SAME_BYTES_AS_DEFAULT_WHOLE ({setting}): {same}")
    for setting, summary in verdict["settings"].items():
        for max_len, s in summary["max_len"].items():
            print(
                f"  {setting} max_len={max_len}: {s['scenarios_run']} scenarios, "
                f"{s['distinct_output_hashes']} distinct output hash(es), "
                f"repeats identical: {s['repeats_identical']}"
                + (f", oom: {s['oom_scenarios']}" if s["oom_scenarios"] else "")
                + (f", errors: {s['error_scenarios']}" if s["error_scenarios"] else "")
            )
    for setting, per_len in algo.items():
        for max_len, per_scenario in per_len.items():
            for scenario, record in per_scenario.items():
                print(
                    f"ALGO: {setting} {max_len} {scenario}: {record['algorithm']} "
                    f"(backend {record['backend'].get('selected')})"
                )
    if verdict["CUDNN_USED"] == "NO":
        print(
            "NOTE: PyTorch does not route this conv to cuDNN, so no cuDNN setting "
            "can change its result; the memory dependence hypothesis is false for "
            "this call whatever the hashes say about other causes."
        )
    for error in errors:
        print(f"ERROR: {error}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker is None:
        return run_driver(args)
    # A worker: print one JSON line and exit.
    geometry = Geometry(
        channels=args.channels,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        source="worker",
    )
    func = worker_scenarios if args.worker == "scenarios" else worker_algo
    with torch.inference_mode():
        result = func(args, geometry)
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
