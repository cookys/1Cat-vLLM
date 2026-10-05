# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How do the routed-expert NVFP4 grouped GEMMs scale with the unique experts U?

Question under test (plan 072 M-03, answering O-08; plan 071 appendix B 5-ii.1).
In the decode profile the two routed-expert kernels cost 3.01 ms per round
(``w13_kernel`` 2.12 + ``w2_batch_reduce_kernel`` 0.89 over 48 layers each, i.e.
about 62.7 us per layer). One expert is 768,000 B on the GPU (derived below from the
real tensor shapes), so the time is bandwidth-bound by U, which the profile does not
record: U = 50 (no overlap between the 5 MTP rows) would be about 613 GB/s, 91 % of
672 GB/s [estimate], U = 10 only about 123 GB/s. This script sweeps U, times the
production call, and (``--ncu``) prints the Nsight Compute commands that read the
bytes actually fetched from DRAM per unique expert.

Production call (not a re-implementation): ``vllm._sm70_ops.nvfp4_grouped_w13_sm70_out``
then ``nvfp4_grouped_w2_batch_reduce_sm70_out`` (``..._w2_sm70_out`` for M != 5), the
exact sequence at ``nvfp4_sm70_moe.py:1534-1566``: split 4 for M = 5 or 8, else 8;
interleaved gate/up as the default fused-SwiGLU prepare; buffers of the same shapes.
Weights come from the production prepare op (``nvfp4_sm70_prepare``) on random e2m1
nibbles and FP16 group-16 scales, a pool of ``--pool`` distinct experts tiled to 512
per layer set. ``--layers`` sets are cycled so consecutive calls do not hit the 6 MB
L2. No checkpoint is read; only shapes come from the config.

Bytes per expert at TP4 (checkpoint tensors ``mlp.experts.N.{gate,up,down}_proj``):
gate/up U8 [640,1280] and E4M3 scale [640,160], down U8 [2560,320] and scale
[2560,40] at TP1; per rank gate+up 2x[160,1280] + 2x[160,160], down [2560,80] +
[2560,10]. FP4 = 409,600 + 204,800; scales 76,800 B as E4M3 (691,200 total, the
checkpoint) or 153,600 B as FP16 (768,000 total, what the kernel reads).

Run (in the lead's GPU window, one idle GPU, never one a server holds):

    cd <worktree> && CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=$PWD \\
        /data/venvs/1cat-p070/bin/python benchmarks/sm70_moe_unique_experts_bench.py \\
        --out /data/bench/moe_unique_experts.json
    # ncu leg: print the commands, run them (service stopped), then parse
    ... sm70_moe_unique_experts_bench.py --ncu --ncu-dir /data/bench
    ... sm70_moe_unique_experts_bench.py --parse-ncu /data/bench/ncu_M5_U*.csv

Verdict lines: ``EXPERT_BYTES``, ``ROUTING_EXACT_U``, ``GROUPS_MATCH_U`` (the plan
kernel's group count equals U for M <= 8), ``W13_IN_BAND`` / ``W2_IN_BAND`` (kernel vs
a float32 torch reference on the dequantized weights, M=5 U=20), ``NEGATIVE_CONTROL``
(the same comparison against the wrong experts must differ), a timing table and
``SELF_CHECK: PASS`` / ``SELF_CHECK: FAIL (<what>)``. Exit 0 pass, 1 fail, 2 no CUDA
device, busy GPU, unusable arguments or a vllm without the grouped ops.

Caveats. ncu replay serialises kernels and flushes caches, so GB/s from ncu time is
not the serving-level number (plan 072 M-10 note). The GB/s column here is the model
bytes over graph-replayed time, an estimate that assumes every unique expert is read
once. Synthetic routing is uniform over 512 experts; production routing skew moves U
and is not modelled. M=1 is not on the grouped path in production (it uses the QPN
M1 kernel); here it is only the grouped kernels at one token. UNVERIFIED: everything
that runs on the GPU; written and tested on CPU only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import re
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_CONFIG = "/data/models/Qwen3.8-Flash-Next-NVFP4/config.json"
DEFAULT_U = (10, 20, 30, 40, 50)
DEFAULT_ROWS = (5, 1)
ROOFLINE_GBPS = 672.0  # plan 072 M-08 denominator, itself unverified
GROUP = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Plan 071 appendix A trace, M=5 production, per round over 48 layers.
TRACE_MS = {"w13": 2.12, "w2": 0.89, "pair": 3.01}
TRACE_LAYERS = 48
BAND = 2e-2  # relative L2 of kernel vs float32 reference
NCU_METRICS = "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"


def _load_sibling(name: str):
    """The kernel-check harness shares the timing and ABBA helpers."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ shapes


@dataclass(frozen=True)
class Shapes:
    hidden: int
    inter: int  # per rank
    experts: int
    top_k: int
    source: str

    def expert_bytes(self, scale_bytes: int = 2) -> dict[str, int]:
        w13_fp4 = self.hidden * 2 * self.inter // 2
        w2_fp4 = self.inter * self.hidden // 2
        w13_sc = self.hidden // GROUP * 2 * self.inter * scale_bytes
        w2_sc = self.inter // GROUP * self.hidden * scale_bytes
        return {
            "w13": w13_fp4 + w13_sc,
            "w2": w2_fp4 + w2_sc,
            "total": w13_fp4 + w2_fp4 + w13_sc + w2_sc,
        }


def read_shapes(path: str | Path, tp: int) -> Shapes:
    production = Shapes(2560, 640 // tp, 512, 10, "production defaults")
    try:
        text = json.loads(Path(path).read_text())
        text = text.get("text_config", text)
        inter = int(text["moe_intermediate_size"])
        if tp <= 0 or inter % (tp * GROUP):
            return production
        return Shapes(
            int(text["hidden_size"]),
            inter // tp,
            int(text["num_experts"]),
            int(text["num_experts_per_tok"]),
            f"config {path} tp={tp}",
        )
    except (OSError, KeyError, TypeError, ValueError):
        return production


# ------------------------------------------------------------------ routing


def feasible(rows: int, top_k: int, experts: int, u: int) -> bool:
    return top_k <= u <= min(rows * top_k, experts)


def make_routing(rows: int, u: int, top_k: int, experts: int, seed: int) -> torch.Tensor:
    """Top-k ids [rows, top_k] int32 whose union is exactly U experts.

    U ids are drawn without replacement; id i goes to row i % rows (so every id
    appears and no row exceeds top_k), then each row is filled to top_k with other
    ids of the pool. Uniform synthetic routing, not production's skew.
    """
    if not feasible(rows, top_k, experts, u):
        raise ValueError(f"U={u} needs {top_k} <= U <= min(rows*top_k, experts) at rows={rows}")
    gen = torch.Generator().manual_seed(seed * 100003 + rows * 1009 + u)
    pool = torch.randperm(experts, generator=gen)[:u].tolist()
    chosen: list[list[int]] = [[] for _ in range(rows)]
    for i, e in enumerate(pool):
        chosen[i % rows].append(e)
    for row in chosen:
        spare = [e for e in pool if e not in row]
        order = torch.randperm(len(spare), generator=gen).tolist()
        row.extend(spare[j] for j in order[: top_k - len(row)])
        perm = torch.randperm(top_k, generator=gen).tolist()
        row[:] = [row[j] for j in perm]
    return torch.tensor(chosen, dtype=torch.int32)


def unique_count(ids: torch.Tensor) -> int:
    return int(torch.unique(ids).numel())


# ------------------------------------------------------------------ ncu


def kernel_regex(rows: int) -> str:
    return "w13_kernel|w2_batch_reduce_kernel" if rows == 5 else "w13_kernel|w2_kernel"


def ncu_command(script: str, rows: int, u: int, out_dir: str, warmup: int, calls: int, seed: int) -> str:
    # two matched launches per call (w13, w2); launch-skip counts matched launches
    return (
        f"ncu --metrics {NCU_METRICS} --kernel-name 'regex:{kernel_regex(rows)}' "
        f"--launch-skip {2 * warmup} --launch-count {2 * calls} --csv --page raw "
        f"--print-units base --log-file {out_dir}/ncu_M{rows}_U{u}.csv "
        f"python {script} --ncu-target --rows {rows} --u-list {u} --seed {seed} "
        f"--ncu-warmup {warmup} --ncu-calls {calls}"
    )


_UNIT_BYTES = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}  # assumed decimal
_UNIT_SECONDS = {"second": 1.0, "s": 1.0, "msecond": 1e-3, "ms": 1e-3, "usecond": 1e-6, "us": 1e-6,
                 "nsecond": 1e-9, "ns": 1e-9}
_METRIC_COLUMNS = {
    "dram__bytes_read.sum": "bytes_read",
    "dram__bytes_write.sum": "bytes_write",
    "gpu__time_duration.sum": "seconds",
}
# ``--print-units base`` prints bytes and nanoseconds; used when a CSV has no unit row at all
_DEFAULT_UNITS = {"bytes_read": "byte", "bytes_write": "byte", "seconds": "ns"}


def _number(text: str) -> float:
    return float(text.replace(",", "").strip())


def _scale(field: str, unit: str) -> float:
    table = _UNIT_SECONDS if field == "seconds" else _UNIT_BYTES
    unit = unit.strip()
    if unit not in table:
        raise ValueError(f"unrecognised ncu unit {unit!r} for {field}; expected one of {sorted(table)}")
    return table[unit]


def parse_ncu_csv_counted(text: str) -> tuple[list[dict[str, Any]], int]:
    """(launches, skipped): one dict per kernel launch with name, bytes_read, bytes_write, seconds.

    Accepts ncu's ``--page raw`` (one row per launch, one column per metric, a unit row
    right under the header) and ``--page details`` (one row per metric) CSV. ``==PROF==``
    lines are skipped, the header is found by its ``ID`` and ``Kernel Name`` columns, and
    ONLY the ID / Kernel Name / dram__bytes_read.sum / dram__bytes_write.sum /
    gpu__time_duration.sum columns are read: the raw page also carries hundreds of device
    attribute columns (``CC`` = ``No-CC`` ...) that are never parsed. Values are scaled by
    the unit row (``byte``, ``ns``, ...). ``skipped`` counts launches whose required metric
    was ``n/a`` or empty; those launches are dropped.
    """
    rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].startswith("==")]
    start = next((i for i, r in enumerate(rows) if "Kernel Name" in r and "ID" in r), None)
    if start is None:
        raise ValueError("no ncu CSV header with ID and Kernel Name columns")
    header = rows[start]
    col = {name: i for i, name in enumerate(header)}
    details = "Metric Name" in col
    wanted = ("dram__bytes_read.sum", "gpu__time_duration.sum")
    if not details and any(m not in col for m in wanted):
        raise ValueError(f"raw-page CSV lacks metric column(s) {[m for m in wanted if m not in col]}")

    def cell(row: Sequence[str], name: str) -> str:
        i = col[name]
        return row[i].strip() if i < len(row) else ""

    units = dict(_DEFAULT_UNITS)
    launches: dict[str, dict[str, Any]] = {}
    broken: set[str] = set()
    for row in rows[start + 1:]:
        ident = cell(row, "ID")
        if not ident.isdigit():
            # the unit row of the raw page: empty ID, units under the metric columns
            if not ident and not details:
                for metric, field in _METRIC_COLUMNS.items():
                    if metric in col and cell(row, metric):
                        _scale(field, cell(row, metric))  # reject units we cannot convert
                        units[field] = cell(row, metric)
            continue
        launch = launches.setdefault(ident, {"id": int(ident), "name": cell(row, "Kernel Name")})
        if details:
            field = next((f for m, f in _METRIC_COLUMNS.items() if cell(row, "Metric Name").startswith(m)), None)
            if field is not None:
                _store(launch, field, cell(row, "Metric Value"), cell(row, "Metric Unit") or _DEFAULT_UNITS[field], broken)
        else:
            for metric, field in _METRIC_COLUMNS.items():
                if metric in col:
                    _store(launch, field, cell(row, metric), units[field], broken)
    out = [x for x in sorted(launches.values(), key=lambda d: d["id"]) if str(x["id"]) not in broken]
    out = [x for x in out if "bytes_read" in x and "seconds" in x]
    return out, len(launches) - len(out)


def parse_ncu_csv(text: str) -> list[dict[str, Any]]:
    return parse_ncu_csv_counted(text)[0]


def _store(launch: dict[str, Any], field: str, value: str, unit: str, broken: set[str]) -> None:
    try:
        v = _number(value)
    except ValueError:
        if value.strip().lower() in ("", "n/a"):
            if field != "bytes_write":  # a missing write counter is tolerated, a missing read/time is not
                broken.add(str(launch["id"]))
            return
        raise
    launch[field] = v * _scale(field, unit)


def kernel_label(name: str) -> str:
    m = re.search(r"(\w+_kernel)\b", name)
    return m.group(1) if m else name


def summarize_ncu(launches: Sequence[dict[str, Any]], u: int, model_bytes: int,
                  rows: int | None = None, skipped: int = 0) -> dict[str, Any]:
    w13 = [x for x in launches if "w13_kernel" in x["name"]]
    w2 = [x for x in launches if "w2_" in x["name"]]
    calls = min(len(w13), len(w2))
    if calls == 0:
        raise ValueError("need at least one w13 and one w2 launch")
    reads = [w13[i]["bytes_read"] + w2[i]["bytes_read"] for i in range(calls)]
    writes = [w13[i].get("bytes_write", 0.0) + w2[i].get("bytes_write", 0.0) for i in range(calls)]
    secs = [w13[i]["seconds"] + w2[i]["seconds"] for i in range(calls)]
    read = statistics.median(reads)
    sec = statistics.median(secs)
    per_kernel: dict[str, dict[str, Any]] = {}
    for x in launches:
        k = per_kernel.setdefault(kernel_label(x["name"]), {"n": 0, "read": 0.0, "write": 0.0, "seconds": 0.0})
        k["n"] += 1
        k["read"] += x["bytes_read"]
        k["write"] += x.get("bytes_write", 0.0)
        k["seconds"] += x["seconds"]
    for k in per_kernel.values():
        k["read_per_launch"] = k["read"] / k["n"]
        k["bytes_per_unique_expert"] = k["read_per_launch"] / u
        k["gbps"] = k["read"] / k["seconds"] / 1e9 if k["seconds"] > 0 else float("nan")
    return {
        "M": rows,
        "U": u,
        "calls": calls,
        "kernel_rows": len(launches),
        "skipped": skipped,
        "total_read": sum(x["bytes_read"] for x in launches),
        "total_write": sum(x.get("bytes_write", 0.0) for x in launches),
        "total_seconds": sum(x["seconds"] for x in launches),
        "per_kernel": per_kernel,
        "bytes_read": read,
        "bytes_write": statistics.median(writes),
        "bytes_per_unique_expert": read / u,
        "vs_model_768000": read / u / model_bytes,
        "vs_e4m3_691200_multiple": read / 691_200,
        "ncu_seconds": sec,
        "gbps_ncu": read / sec / 1e9 if sec > 0 else float("nan"),
        "pct_roofline": read / sec / 1e9 / ROOFLINE_GBPS * 100 if sec > 0 else float("nan"),
    }


def format_ncu_table(rows: Sequence[dict[str, Any]], expected: dict[str, int] | None = None) -> list[str]:
    """Per-file table (per-call medians) then per-kernel sub-totals; ``expected`` = {'w13': B, 'w2': B}."""
    out = [
        "NCU_TABLE (replay serialises kernels: GB/s here is ncu-time based, not serving-level; bytes/GB/s per call = median)",
        f"{'M':>2} {'U':>3} {'calls':>5} {'rows':>4} {'skip':>4} {'bytes_read':>12} {'B/unique':>10} {'vs768000':>8} "
        f"{'x691200':>8} {'GB/s':>7} {'%672':>6} {'sum_read':>12} {'sum_write':>10} {'sum_us':>9}",
    ]
    for r in rows:
        out.append(
            f"{r.get('M') or '-':>2} {r['U']:>3} {r['calls']:>5} {r['kernel_rows']:>4} {r['skipped']:>4} "
            f"{r['bytes_read']:>12.0f} {r['bytes_per_unique_expert']:>10.0f} "
            f"{r['vs_model_768000']:>8.3f} {r['vs_e4m3_691200_multiple']:>8.2f} "
            f"{r['gbps_ncu']:>7.1f} {r['pct_roofline']:>6.1f} "
            f"{r['total_read']:>12.0f} {r['total_write']:>10.0f} {r['total_seconds'] * 1e6:>9.1f}"
        )
    out += [
        "NCU_PER_KERNEL (read bytes per launch / U = per unique expert; expected = fp16-scale model split)",
        f"{'M':>2} {'U':>3} {'kernel':<24} {'n':>3} {'read/launch':>12} {'B/unique':>9} {'expected':>9} {'ratio':>6} "
        f"{'write/launch':>12} {'us/launch':>9} {'GB/s':>7}",
    ]
    for r in rows:
        for name, k in r["per_kernel"].items():
            exp = (expected or {}).get("w13" if name.startswith("w13") else "w2")
            out.append(
                f"{r.get('M') or '-':>2} {r['U']:>3} {name:<24} {k['n']:>3} {k['read_per_launch']:>12.0f} "
                f"{k['bytes_per_unique_expert']:>9.0f} {exp if exp else '-':>9} "
                f"{(k['bytes_per_unique_expert'] / exp if exp else float('nan')):>6.3f} "
                f"{k['write'] / k['n']:>12.0f} {k['seconds'] / k['n'] * 1e6:>9.2f} {k['gbps']:>7.1f}"
            )
    return out


def u_from_name(path: str) -> int | None:
    m = re.search(r"U(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def m_from_name(path: str) -> int | None:
    m = re.search(r"_M(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ statistics


def implied_u(points: Sequence[tuple[int, float]], target: float) -> str | float:
    """U at which the measured median crosses ``target`` (linear interpolation)."""
    pts = sorted(points)
    if len(pts) < 2:
        return "n/a"
    if target < pts[0][1]:
        return f"<{pts[0][0]}"
    if target > pts[-1][1]:
        return f">{pts[-1][0]}"
    for (u0, t0), (u1, t1) in zip(pts, pts[1:], strict=False):
        if t0 <= target <= t1 and t1 > t0:
            return u0 + (u1 - u0) * (target - t0) / (t1 - t0)
    return "n/a"


def rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a, r = actual.double(), reference.double()
    norm = float(r.norm())
    return float((a - r).norm()) / norm if norm > 0 else float("nan")


# ------------------------------------------------------------------ GPU side


def dequant(nib: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """e2m1 nibbles [K,N] and FP16 group scales [K/16,N] to float32 [K,N]."""
    table = torch.tensor(E2M1, dtype=torch.float32, device=nib.device)
    val = table[(nib & 7).long()] * torch.where((nib & 8) > 0, -1.0, 1.0)
    k, n = nib.shape
    return (val.view(k // GROUP, GROUP, n) * scales.float()[:, None, :]).view(k, n)


class Weights:
    """``layers`` tiled 512-expert weight sets plus the raw pool for the reference."""

    def __init__(self, shapes: Shapes, pool: int, layers: int, seed: int, device: str, ops: Any):
        if shapes.experts % pool:
            raise ValueError("--pool must divide the expert count")
        gen = torch.Generator().manual_seed(seed)
        self.pool, self.shapes = pool, shapes
        h, i = shapes.hidden, shapes.inter
        self.raw: list[tuple[torch.Tensor, ...]] = []
        w13, s13, w2, s2 = [], [], [], []
        for _ in range(pool):
            n13 = torch.randint(0, 16, (h, 2 * i), dtype=torch.uint8, generator=gen).to(device)
            c13 = (torch.rand(h // GROUP, 2 * i, generator=gen) * 0.008 + 0.002).half().to(device)
            n2 = torch.randint(0, 16, (i, h), dtype=torch.uint8, generator=gen).to(device)
            c2 = (torch.rand(i // GROUP, h, generator=gen) * 0.008 + 0.002).half().to(device)
            self.raw.append((n13, c13, n2, c2))
            a = ops.nvfp4_sm70_prepare(n13, c13, GROUP, True)
            b = ops.nvfp4_sm70_prepare(n2, c2, GROUP, False)
            w13.append(a[0]), s13.append(a[1]), w2.append(b[0]), s2.append(b[1])
        reps = shapes.experts // pool
        base = [torch.stack(t) for t in (w13, s13, w2, s2)]
        self.sets = [tuple(t.repeat(reps, *([1] * (t.dim() - 1))).contiguous() for t in base) for _ in range(layers)]

    def reference_w(self, expert: int, shift: int = 0):
        n13, c13, n2, c2 = self.raw[(expert + shift) % self.pool]
        return dequant(n13, c13), dequant(n2, c2)


class Call:
    """The production grouped-decode sequence on fixed buffers for ``rows`` tokens."""

    def __init__(self, ops: Any, shapes: Shapes, rows: int, ids: torch.Tensor, device: str, seed: int):
        self.ops, self.rows, self.shapes = ops, rows, shapes
        gen = torch.Generator().manual_seed(seed + rows)
        h, i, k = shapes.hidden, shapes.inter, shapes.top_k
        slots = rows * k
        self.x = torch.randn(rows, h, generator=gen).half().to(device)
        self.ids = ids.to(device).contiguous()
        self.tw = torch.softmax(torch.randn(rows, k, generator=gen), -1).to(device)
        self.mid = torch.empty(slots, i, dtype=torch.float16, device=device)
        self.routed = torch.empty(slots, h, dtype=torch.float16, device=device)
        self.out = torch.empty(rows, h, dtype=torch.float16, device=device)
        self.meta = (
            torch.empty(160, 8, dtype=torch.int32, device=device),
            torch.empty(160, dtype=torch.int32, device=device),
            torch.empty(160, dtype=torch.int32, device=device),
            torch.empty(1, dtype=torch.int32, device=device),
        )
        self.split = 4 if rows in (5, 8) else 8
        self.batch_reduce = rows == 5

    def w13(self, ws: tuple[torch.Tensor, ...]) -> None:
        self.ops.nvfp4_grouped_w13_sm70_out(
            self.mid, self.x, ws[0], ws[1], self.ids.view(-1), *self.meta, self.split, True
        )

    def w2(self, ws: tuple[torch.Tensor, ...]) -> None:
        op = self.ops.nvfp4_grouped_w2_batch_reduce_sm70_out if self.batch_reduce else self.ops.nvfp4_grouped_w2_sm70_out
        op(self.out, self.routed, self.mid, ws[2], ws[3], self.tw, *self.meta)

    def pair(self, ws: tuple[torch.Tensor, ...]) -> None:
        self.w13(ws)
        self.w2(ws)


def make_graphs(call: Call, sets: Sequence[tuple[torch.Tensor, ...]], kind: str):
    fn = call.pair if kind == "pair" else call.w13
    graphs = []
    for ws in sets:
        for _ in range(3):
            fn(ws)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(ws)
        graphs.append(g)
    return graphs


def self_check_case(ops: Any, shapes: Shapes, weights: Weights, device: str, seed: int) -> dict[str, Any]:
    rows, u = 5, 20
    ids = make_routing(rows, u, shapes.top_k, shapes.experts, seed)
    call = Call(ops, shapes, rows, ids, device, seed)
    ws = weights.sets[0]
    call.pair(ws)
    torch.cuda.synchronize()
    k = shapes.top_k
    flat = ids.view(-1).tolist()

    def reference(shift: int) -> tuple[torch.Tensor, torch.Tensor]:
        mids, routed = [], []
        for t, e in enumerate(flat):
            r13, r2 = weights.reference_w(e, shift)
            gu = call.x[t // k].float() @ r13
            gate, up = gu[: shapes.inter].half(), gu[shapes.inter :].half()
            act = (gate.float() * torch.sigmoid(gate.float())).half()
            mids.append(act * up)
            routed.append(((call.mid[t].float() @ r2)).half())
        out = (torch.stack(routed).float().view(rows, k, -1) * call.tw[:, :, None]).sum(1).half()
        return torch.stack(mids), out

    mid_ref, out_ref = reference(0)
    mid_bad, out_bad = reference(1)
    return {
        "rows": rows,
        "U": u,
        "w13_rel_l2": rel_l2(call.mid, mid_ref),
        "w2_rel_l2": rel_l2(call.out, out_ref),
        "wrong_expert_w13_rel_l2": rel_l2(call.mid, mid_bad),
        "wrong_expert_w2_rel_l2": rel_l2(call.out, out_bad),
        "groups": int(call.meta[3].item()),
    }


def time_sweep(chk: Any, ops: Any, shapes: Shapes, weights: Weights, rows: int, us: Sequence[int], args: argparse.Namespace, device: str):
    calls, arms, graphs, groups = {}, {}, {}, {}
    for u in us:
        ids = make_routing(rows, u, shapes.top_k, shapes.experts, args.seed)
        calls[u] = Call(ops, shapes, rows, ids, device, args.seed)
        calls[u].pair(weights.sets[0])
        torch.cuda.synchronize()
        groups[u] = int(calls[u].meta[3].item())
        for kind in ("pair", "w13"):
            counter = [0]
            if args.no_graph:
                fn = calls[u].pair if kind == "pair" else calls[u].w13

                def arm(fn=fn, counter=counter):
                    fn(weights.sets[counter[0] % len(weights.sets)])
                    counter[0] += 1
            else:
                graphs[(u, kind)] = make_graphs(calls[u], weights.sets, kind)

                def arm(g=graphs[(u, kind)], counter=counter):
                    g[counter[0] % len(g)].replay()
                    counter[0] += 1
            arms[f"U{u}:{kind}"] = arm
    timings = chk.time_arms(arms, rounds=args.rounds, iters=args.iters, warmup=args.warmup)
    return timings, groups


def format_timing(rows: int, us: Sequence[int], timings: dict[str, Any], groups: dict[int, int], bytes_total: int) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [
        f"TIMING M={rows} (graph-replay us per layer call; w2 = pair - w13 derived; GB/s = {bytes_total} B x U / pair "
        f"[estimate, each unique expert read once]; roofline {ROOFLINE_GBPS:.0f} GB/s unverified)",
        f"{'U':>3} {'grp':>4} {'pair med [p10,p90]':>26} {'w13':>8} {'w2*':>8} {'GB/s':>7} {'%672':>6} {'spread%':>8}",
    ]
    table = []
    for u in us:
        p, w = timings[f"U{u}:pair"], timings[f"U{u}:w13"]
        gbps = u * bytes_total / (p["median"] * 1e-6) / 1e9
        row = {
            "U": u, "rows": rows, "groups": groups[u], "pair_us": p["median"], "pair_p10": p["p10"],
            "pair_p90": p["p90"], "w13_us": w["median"], "w2_us_derived": p["median"] - w["median"],
            "gbps_estimate": gbps, "pct_roofline_estimate": gbps / ROOFLINE_GBPS * 100,
            "round_spread_pct": p["round_spread_pct"], "samples": p["samples"],
        }
        table.append(row)
        lines.append(
            f"{u:>3} {groups[u]:>4} {p['median']:>10.1f} [{p['p10']:.1f},{p['p90']:.1f}]".ljust(32)
            + f"{w['median']:>8.1f} {row['w2_us_derived']:>8.1f} {gbps:>7.1f} {row['pct_roofline_estimate']:>6.1f} "
            f"{p['round_spread_pct']:>8.1f}"
        )
    per_layer = {k: v / TRACE_LAYERS * 1000 for k, v in TRACE_MS.items()}
    lines.append(
        f"TRACE (plan 071 A, M=5 production, per layer): w13 {per_layer['w13']:.1f} us, w2 {per_layer['w2']:.1f} us, "
        f"pair {per_layer['pair']:.1f} us"
    )
    if rows == 5:
        est = implied_u([(r["U"], r["pair_us"]) for r in table], per_layer["pair"])
        lines.append(f"IMPLIED_U_FROM_TRACE_MEAN: {est if isinstance(est, str) else f'{est:.1f}'} "
                     "(estimate: linear interpolation of the synthetic sweep; production routing skew not modelled)")
    return lines, table


# ------------------------------------------------------------------ driver


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS), help="tokens M per call (5 = MTP4 target)")
    p.add_argument("--u-list", type=int, nargs="+", default=list(DEFAULT_U))
    p.add_argument("--pool", type=int, default=8, help="distinct experts prepared, tiled to the full count")
    p.add_argument("--layers", type=int, default=4, help="weight sets cycled to defeat L2 reuse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=50, help="calls per timing turn")
    p.add_argument("--rounds", type=int, default=4, help="ABBA timing rounds")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-graph", action="store_true", help="eager launches instead of CUDA-graph replay")
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--max-used-mib", type=int, default=1024)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--ncu", action="store_true", help="print the ncu command per U and exit (runs nothing)")
    p.add_argument("--ncu-dir", default="/data/bench")
    p.add_argument("--ncu-warmup", type=int, default=3)
    p.add_argument("--ncu-calls", type=int, default=4)
    p.add_argument("--ncu-target", action="store_true", help="the process ncu profiles: eager calls only")
    p.add_argument("--parse-ncu", nargs="+", metavar="CSV", help="summarise ncu CSVs (U from the file name U<n>)")
    a = p.parse_args(argv)
    if a.iters <= 0 or a.rounds <= 0 or a.warmup < 0 or a.layers <= 0 or a.pool <= 0:
        p.error("--iters, --rounds, --layers and --pool must be positive")
    return a


def git_tip() -> str:
    try:
        root = Path(__file__).resolve().parent
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shapes = read_shapes(args.config, args.tp)
    model = shapes.expert_bytes(2)
    e4m3 = shapes.expert_bytes(1)
    print(f"EXPERT_BYTES: FP16-scale {model['total']} (w13 {model['w13']} + w2 {model['w2']}), "
          f"E4M3-scale {e4m3['total']} ({shapes.source}; hidden {shapes.hidden}, inter/rank {shapes.inter}, "
          f"experts {shapes.experts}, top_k {shapes.top_k})")
    if model["total"] != 768_000:
        print(f"WARNING: per-expert bytes are {model['total']}, not the 768,000 B in plan 072 O-08; correct the plan")
    if args.parse_ncu:
        out = []
        for i, path in enumerate(args.parse_ncu):
            u = u_from_name(path)
            if u is None and len(args.u_list) == len(args.parse_ncu):
                u = args.u_list[i]
            if u is None:
                print(f"cannot tell U for {path}: name it ..._U<n>.csv or pass --u-list per file")
                return 2
            launches, skipped = parse_ncu_csv_counted(Path(path).read_text())
            out.append(summarize_ncu(launches, u, model["total"], rows=m_from_name(path), skipped=skipped))
        expected = {"w13": model["w13"], "w2": model["w2"]}
        print("\n".join(format_ncu_table(sorted(out, key=lambda r: (r["M"] or 0, r["U"])), expected)))
        return 0
    plan = {m: [u for u in args.u_list if feasible(m, shapes.top_k, shapes.experts, u)] for m in args.rows}
    for m, us in plan.items():
        skipped = [u for u in args.u_list if u not in us]
        print(f"M={m}: U sweep {us}" + (f" (infeasible, skipped {skipped}: need {shapes.top_k}<=U<=M*k)" if skipped else ""))
    if args.ncu:
        script = str(Path(__file__).resolve())
        print("# ncu serialises kernels: GB/s from ncu time is not the serving-level number (plan 072 M-10)")
        for m, us in plan.items():
            for u in us:
                print(ncu_command(script, m, u, args.ncu_dir, args.ncu_warmup, args.ncu_calls, args.seed))
        return 0
    if not torch.cuda.is_available():
        print("needs a CUDA device (set CUDA_VISIBLE_DEVICES to an idle GPU)")
        return 2
    chk = _load_sibling("sm70_nvfp4_kv_kernel_check")
    if not args.allow_busy:
        message = chk.refuse_busy_gpu(args.max_used_mib)
        if message:
            print(message)
            return 2
    try:
        from vllm import _sm70_ops as ops

        if not ops.has_nvfp4_grouped_decode_dispatch() or not ops.has_nvfp4_grouped_batch_reduce_dispatch():
            raise ImportError("grouped decode ops missing from torch.ops._C")
        print(f"SOURCE ops: {ops.__file__}")
    except ImportError as exc:
        print(f"cannot use the grouped NVFP4 ops ({exc}); run with PYTHONPATH=<worktree> and a built extension")
        return 2
    device = "cuda:0"
    info = chk.device_info() if hasattr(chk, "device_info") else {}
    print(f"device {info.get('name')} capability {info.get('capability')}, torch {torch.__version__}")
    if info.get("capability") != [7, 0]:
        print("NOTE: not an sm_70 device; the numbers are not V100 numbers")
    failures: list[str] = []
    report: dict[str, Any] = {"meta": {}, "rows": [], "self_check": {}}
    with torch.inference_mode():
        weights = Weights(shapes, args.pool, args.layers, args.seed, device, ops)
        if args.ncu_target:
            for m in args.rows:
                for u in plan[m]:
                    call = Call(ops, shapes, m, make_routing(m, u, shapes.top_k, shapes.experts, args.seed), device, args.seed)
                    for i in range(args.ncu_warmup + args.ncu_calls):
                        call.pair(weights.sets[i % len(weights.sets)])
            torch.cuda.synchronize()
            print("NCU_TARGET_DONE")
            return 0
        check = self_check_case(ops, shapes, weights, device, args.seed)
        report["self_check"] = check
        ok13, ok2 = check["w13_rel_l2"] < BAND, check["w2_rel_l2"] < BAND
        flagged = check["wrong_expert_w13_rel_l2"] > 10 * BAND and check["wrong_expert_w2_rel_l2"] > 10 * BAND
        print(f"W13_IN_BAND: {'PASS' if ok13 else 'FAIL'} (rel L2 {check['w13_rel_l2']:.2e}, band {BAND})")
        print(f"W2_IN_BAND: {'PASS' if ok2 else 'FAIL'} (rel L2 {check['w2_rel_l2']:.2e}, band {BAND})")
        print(f"NEGATIVE_CONTROL wrong_expert: {'FLAGGED' if flagged else 'MISSED'} "
              f"(w13 {check['wrong_expert_w13_rel_l2']:.2e}, w2 {check['wrong_expert_w2_rel_l2']:.2e})")
        failures += [n for n, good in (("w13 band", ok13), ("w2 band", ok2), ("negative control missed", flagged)) if not good]
        exact = True
        for m, us in plan.items():
            for u in us:
                exact &= unique_count(make_routing(m, u, shapes.top_k, shapes.experts, args.seed)) == u
        print(f"ROUTING_EXACT_U: {'YES' if exact else 'NO'}")
        if not exact:
            failures.append("routing U")
        if args.no_timing:
            failures.append("timing stage skipped")
        else:
            for m, us in plan.items():
                if not us:
                    continue
                timings, groups = time_sweep(chk, ops, shapes, weights, m, us, args, device)
                lines, table = format_timing(m, us, timings, groups, model["total"])
                print("\n".join(lines))
                report["rows"] += table
                if m <= 8:
                    match = all(groups[u] == u for u in us)
                    print(f"GROUPS_MATCH_U M={m}: {'YES' if match else 'NO'} (plan kernel groups {[groups[u] for u in us]})")
                    if not match:
                        failures.append(f"groups != U at M={m}")
    verdict = "PASS" if not failures else f"FAIL ({', '.join(failures)})"
    print(f"SELF_CHECK: {verdict}")
    report["meta"] = {
        "git_tip": git_tip(), "torch": torch.__version__, "triton": _triton_version(), "shapes": asdict(shapes),
        "expert_bytes_fp16_scale": model, "expert_bytes_e4m3_scale": e4m3, "rows": list(args.rows),
        "args": vars(args), "device": info, "estimates": "gbps/pct_roofline/implied U are estimates",
    }
    report["verdict"] = verdict
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0 if not failures else 1


def _triton_version() -> str:
    with contextlib.suppress(Exception):
        import triton

        return triton.__version__
    return "n/a"


if __name__ == "__main__":
    sys.exit(main())
