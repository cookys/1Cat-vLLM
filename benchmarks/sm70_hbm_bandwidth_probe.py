# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which number is the real HBM ceiling of a V100-SXM2-32GB: 672, ~701 or 898 GB/s?

Question under test (plan 072 M-08; plan 071 appendix B 5-ii.6). Every "% of HBM"
in plans 071/072 divides by one of three numbers: the 898.048 GB/s spec (4096 bit x
877 MHz x 2 / 8; nsys ``memoryBandwidth``), 672 GB/s (the lm_head kernel reading
317,849,600 B in 471-483 us, appendix A 2.4; ``HBM_MEAS_GBPS`` in
``scripts/c4140-ab/appendix_b_roofline.py``), or 701 GB/s (the row-gemv kernel). None of
them was measured as a ceiling: nothing ever streamed a buffer with no other work. This
probe does, with four independent arms, each swept over buffer sizes (default 256 MiB,
512 MiB, 1 GiB; every size is far above the 6 MB L2, and two buffers are cycled):

  read     pure streaming read of an fp16 buffer: ``read_torch`` = ``buf.sum(dtype=fp32)``,
           ``read_triton`` = a ``tl.load``-only Triton kernel with one fp32 partial per
           program written to a small tensor (so the loads cannot be elided).
  d2d      ``dst.copy_(src)`` and ``src.clone()``. Convention below.
  write    ``fill_(1.5)`` and ``zero_()``.
  lmhead   the appendix-A shape: out[M, rows] = x[M, hidden] @ W[rows, hidden]^T for
           M = 1 and 5, fp16 W of one TP rank (vocab / tp rows), random weights, CUDA
           graph replay, GB/s = weight bytes / time. This reproduces the "672" number.
           Production's kernel is a cutlass WMMA tn GEMM; this uses ``torch.mm`` (cuBLAS),
           so it reproduces the shape and the bytes, not the exact kernel.

Counting convention. Plan 071 appendix B counts the bytes a kernel READS from HBM
(``w_lo``/``w_hi``, ``act``; lm_head = 317,849,600 B read) and, for memcpy, the bytes
MOVED one way (nsys ``bytes``, e.g. "D2D 6.78 MB per round"), never 2x. This probe
therefore prints, for every arm, ``GB/s`` = bytes counted / time (one-way for copies:
the plan's convention) and ``traffic GB/s`` = HBM traffic (copy = read + write = 2x the
bytes moved; every other arm 1x). The ceiling verdict uses TRAFFIC, because the ceiling
is a property of the memory system, not of the plan's bookkeeping; the one-way copy
number is what to compare with a plan row that quotes a memcpy.

lm_head dims come from the checkpoint ``config.json`` (``vocab_size``, ``hidden_size``)
and ``model.safetensors.index.json`` (the index only; no weight file is opened):
``lm_head.weight`` is a plain tensor in ``model-bf16-*.safetensors``, ``lm_head`` is in
``quantization_config.ignore`` (not NVFP4) and serving converts BF16 to FP16, so the
per-rank weight is [vocab / tp, hidden] fp16 = 62,080 x 2560 x 2 = 317,849,600 B at TP4.

Timing: CUDA events around a CUDA-graph replay of ``--inner`` back-to-back calls that
cycle the buffer sets (``--no-graph``: eager calls, one event pair per call); per-call
time = replay time / inner. Arms of one size are interleaved A B ... B A over
``--rounds`` rounds, ``--iters`` replays per turn, median / p10 / p90 of all samples and
the spread of the per-round medians. The ceiling verdict takes, per kind, the highest
median over arms and sizes (the lowest-overhead, best-tuned kernel is the best estimate
of the ceiling; a Triton kernel can be under-tuned, hence ``--triton-block`` /
``--triton-iters``).

Run (in the lead's GPU window, one idle GPU, never one a server holds):

    cd <worktree> && CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=$PWD \\
        /data/venvs/1cat-p070/bin/python benchmarks/sm70_hbm_bandwidth_probe.py \\
        --out /data/bench/hbm-probe.json
    # Nsight Compute commands only (printed, nothing runs):
    ... sm70_hbm_bandwidth_probe.py --ncu --ncu-dir /data/bench

Verdict lines: ``LMHEAD_BYTES``, ``READ_CHECK`` / ``COPY_CHECK`` / ``WRITE_CHECK`` /
``LMHEAD_CHECK``, the timing table, ``HBM_CEILING_GBPS read=<x> copy=<y> write=<z>
lmhead=<w>`` (all HBM traffic, copy = read + write), ``COPY_ONEWAY_GBPS``,
``PCT_OF_SPEC`` (of 898.048), ``PCT_OF_672`` and ``SELF_CHECK: PASS`` /
``SELF_CHECK: FAIL (<what>)``. Exit 0 pass, 1 fail, 2 no CUDA device, busy GPU, not
enough free memory or unusable arguments.

Caveats. Clocks (SM / memory) are recorded from nvidia-smi before and after the sweep,
not under load. A 1 GiB sweep is a few ms per call; the 256 MiB size is about 0.4 ms, so
the p10-p90 spread tells how stable it is. UNVERIFIED: everything that runs on the GPU;
written and tested on CPU only (CPU tests cover arguments, sizes, arithmetic, statistics,
the JSON schema and exit 2).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

DEFAULT_CONFIG = "/data/models/Qwen3.8-Flash-Next-NVFP4/config.json"
DEFAULT_SIZES = ("256M", "512M", "1G")
SPEC_GBPS = 898.048  # 4096 bit x 877 MHz x 2 / 8; nsys memoryBandwidth
PLAN_GBPS = {"672": 672.0, "701": 701.0}  # appendix A lm_head / row-gemv, plan 071 appendix B
LMHEAD_BYTES_EXPECTED = 317_849_600  # plan 071 appendix A 2.4, per rank at TP4
L2_BYTES = 6 * 1024 * 1024  # V100 L2; every buffer must be a multiple of this to defeat reuse
MIN_SIZE = 8 * 1024 * 1024
ELEM_BYTES = 2  # fp16 buffers
READ_ARMS = ("read_torch", "read_triton")
D2D_ARMS = ("copy", "clone")
WRITE_ARMS = ("fill", "zero")
GROUPS = {"read": READ_ARMS, "d2d": D2D_ARMS, "write": WRITE_ARMS}
KIND = {**{a: "read" for a in READ_ARMS}, **{a: "copy" for a in D2D_ARMS}, **{a: "write" for a in WRITE_ARMS}}
TRAFFIC_FACTOR = {"copy": 2}  # read + write; every other kind is counted 1x
REL_TOL = {"read_torch": 1e-3, "read_triton": 1e-3, "lmhead": 1e-2}
NCU_METRICS = "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
NCU_KERNEL = {"read_triton": "hbm_read_kernel", "fill": "elementwise|fill", "zero": "elementwise|fill",
              "read_torch": "reduce_kernel", "lmhead_m1": "gemv|gemm|cutlass|sm70", "lmhead_m5": "gemv|gemm|cutlass|sm70"}


# ------------------------------------------------------------------ sizes, arithmetic, statistics
_SIZE_RE = re.compile(r"^(\d+)([KMGT]?)(i?B)?$", re.IGNORECASE)
_UNIT = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(text: str) -> int:
    """'256M' -> 268,435,456 (binary units); a bare number is bytes. Multiple of 2 (fp16), >= 8 MiB (L2 is 6 MB)."""
    m = _SIZE_RE.match(text.strip())
    if not m:
        raise ValueError(f"bad size {text!r}: use e.g. 256M, 1G or a byte count")
    value = int(m.group(1)) * _UNIT[m.group(2).upper()]
    if value % ELEM_BYTES:
        raise ValueError(f"size {text!r} is not a multiple of {ELEM_BYTES} B (fp16 elements)")
    if value < MIN_SIZE:
        raise ValueError(f"size {text!r} = {value} B is below {MIN_SIZE} B: the buffer must be well above the 6 MB L2")
    return value


def size_label(nbytes: int) -> str:
    for unit, div in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if nbytes % div == 0:
            return f"{nbytes // div}{unit}"
    return f"{nbytes}B"


def gbps(nbytes: float, us: float) -> float:
    """Decimal GB/s (1e9 B/s, the unit of 898.048 and 672) for ``nbytes`` moved in ``us`` microseconds."""
    return nbytes / (us * 1e-6) / 1e9 if us > 0 else float("nan")


def pct(value: float, reference: float) -> float:
    return value / reference * 100.0


def summarize_timing(rounds: Sequence[Sequence[float]]) -> dict[str, float]:
    """Median, p10, p90 of all samples and the spread of the per-round medians (same math as the kernel check)."""
    samples = sorted(x for r in rounds for x in r)
    if not samples:
        raise ValueError("no samples")

    def percentile(p: float) -> float:
        k = (len(samples) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return samples[lo] + (samples[hi] - samples[lo]) * (k - lo)

    medians = [statistics.median(r) for r in rounds if len(r)]
    center = statistics.median(medians)
    spread = (max(medians) - min(medians)) / center * 100 if center > 0 else 0.0
    return {"median": statistics.median(samples), "p10": percentile(0.1), "p90": percentile(0.9),
            "round_spread_pct": spread, "samples": len(samples)}


def abba_order(arms: Sequence[str], rounds: int) -> list[str]:
    """A B ... B A per round, reversed on every other round."""
    order: list[str] = []
    for index in range(rounds):
        order += list(arms) if index % 2 == 0 else list(reversed(arms))
    return order


def make_row(arm: str, size: int, nbytes: int, timing: dict[str, float], note: str = "") -> dict[str, Any]:
    """One result row. ``nbytes`` = bytes counted by the plan convention (copy: moved one way)."""
    kind = KIND.get(arm, "lmhead" if arm.startswith("lmhead") else "read")
    factor = TRAFFIC_FACTOR.get(kind, 1)
    med = timing["median"]
    return {
        "arm": arm, "kind": kind, "size_bytes": size, "bytes_counted": nbytes, "traffic_factor": factor,
        "median_us": med, "p10_us": timing["p10"], "p90_us": timing["p90"],
        "round_spread_pct": timing["round_spread_pct"], "samples": int(timing["samples"]),
        "gbps": gbps(nbytes, med), "gbps_lo": gbps(nbytes, timing["p90"]), "gbps_hi": gbps(nbytes, timing["p10"]),
        "traffic_gbps": gbps(nbytes * factor, med), "note": note,
    }


def ceilings(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Best median TRAFFIC GB/s per kind over all arms and sizes, with where it came from, and the oneway copy number."""
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = r["kind"]
        if k not in best or r["traffic_gbps"] > best[k]["traffic_gbps"]:
            best[k] = r
    out: dict[str, Any] = {k: {"traffic_gbps": r["traffic_gbps"], "arm": r["arm"], "size_bytes": r["size_bytes"],
                               "pct_of_spec": pct(r["traffic_gbps"], SPEC_GBPS),
                               "pct_of_672": pct(r["traffic_gbps"], PLAN_GBPS["672"]),
                               "pct_of_701": pct(r["traffic_gbps"], PLAN_GBPS["701"])} for k, r in best.items()}
    if "copy" in best:
        out["copy"]["oneway_gbps"] = best["copy"]["gbps"]
    return out


def verdict_lines(c: dict[str, Any]) -> list[str]:
    def f(kind: str, key: str = "traffic_gbps") -> str:
        return f"{c[kind][key]:.1f}" if kind in c else "n/a"

    lines = [f"HBM_CEILING_GBPS read={f('read')} copy={f('copy')} write={f('write')} lmhead={f('lmhead')}"
             "  (HBM traffic; copy = read + write; best median over arms and sizes)"]
    if "copy" in c:
        lines.append(f"COPY_ONEWAY_GBPS {c['copy']['oneway_gbps']:.1f} (bytes moved one way: the plan 071 appendix B memcpy convention)")
    for name, key, note in (("PCT_OF_SPEC", "pct_of_spec", "  (of 898.048 GB/s)"), ("PCT_OF_672", "pct_of_672", "")):
        parts = " ".join(f"{k}={c[k][key]:.1f}" for k in ("read", "copy", "write", "lmhead") if k in c)
        lines.append(f"{name} {parts}{note}")
    return lines


# ------------------------------------------------------------------ lm_head shape (config + index only)
def read_lmhead_shape(config: str | Path, tp: int) -> dict[str, Any]:
    """vocab / hidden from config.json; the lm_head.weight entry of the index (never a weight file); fp16 as served."""
    shape: dict[str, Any] = {"vocab": 248320, "hidden": 2560, "tp": tp, "dtype": "float16", "source": "production defaults",
                             "index_entry": None, "quant_ignore_lm_head": None, "checkpoint_dtype": None}
    path = Path(config)
    try:
        cfg = json.loads(path.read_text())
        text = cfg.get("text_config", cfg)
        shape.update(vocab=int(text["vocab_size"]), hidden=int(text["hidden_size"]), source=f"config {path} tp={tp}")
        shape["checkpoint_dtype"] = text.get("dtype") or text.get("torch_dtype") or cfg.get("dtype") or cfg.get("torch_dtype")
        shape["quant_ignore_lm_head"] = "lm_head" in (cfg.get("quantization_config") or {}).get("ignore", [])
    except (OSError, KeyError, TypeError, ValueError):
        pass
    with contextlib.suppress(OSError, ValueError, AttributeError):
        weight_map = json.loads((path.parent / "model.safetensors.index.json").read_text())["weight_map"]
        shape["index_entry"] = {k: v for k, v in weight_map.items() if k.endswith("lm_head.weight")}
    if tp <= 0 or shape["vocab"] % tp:
        raise ValueError(f"vocab {shape['vocab']} is not divisible by tp {tp}")
    shape["rows"] = shape["vocab"] // tp
    shape["bytes"] = shape["rows"] * shape["hidden"] * ELEM_BYTES
    return shape


# ------------------------------------------------------------------ Triton read kernel (built lazily: the CPU tests never need it)
_READ_KERNEL: Any = None


def read_kernel() -> Any:
    """``hbm_read_kernel(ptr, partials, n, BLOCK, ITERS)``: every program streams BLOCK*ITERS fp16 elements, accumulates in fp32
    registers and writes ONE partial; the store of the dependent partial is what keeps the loads alive."""
    global _READ_KERNEL
    if _READ_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def hbm_read_kernel(ptr, partials, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
            pid = tl.program_id(0).to(tl.int64)
            base = pid * BLOCK * ITERS
            offs = tl.arange(0, BLOCK)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for i in tl.static_range(ITERS):
                idx = base + i * BLOCK + offs
                acc += tl.load(ptr + idx, mask=idx < n, other=0.0).to(tl.float32)
            tl.store(partials + pid, tl.sum(acc, axis=0))

        _READ_KERNEL = hbm_read_kernel
    return _READ_KERNEL


def triton_read(buf: torch.Tensor, partials: torch.Tensor, block: int, iters: int, warps: int) -> None:
    n = buf.numel()
    read_kernel()[(math.ceil(n / (block * iters)),)](buf, partials, n, BLOCK=block, ITERS=iters, num_warps=warps)


def triton_partials(n: int, block: int, iters: int) -> int:
    return math.ceil(n / (block * iters))


def ref_sum(buf: torch.Tensor, chunk: int = 1 << 26) -> float:
    """float64 reference sum without a float64 copy of the whole buffer."""
    flat = buf.view(-1)
    return float(sum(flat[i:i + chunk].sum(dtype=torch.float64).item() for i in range(0, flat.numel(), chunk)))


# ------------------------------------------------------------------ the sweep
class Sweep:
    """Buffers and arms for one size. ``sets`` buffers of each role are cycled so consecutive calls never hit the L2."""

    def __init__(self, size: int, sets: int, device: str, args: argparse.Namespace):
        self.size, self.sets, self.device, self.args = size, sets, device, args
        n = size // ELEM_BYTES
        self.n = n
        self.src = [torch.empty(n, dtype=torch.float16, device=device).uniform_(0.0, 1.0) for _ in range(sets)]
        self.dst = [torch.empty(n, dtype=torch.float16, device=device) for _ in range(sets)]
        self.partials = torch.empty(triton_partials(n, args.triton_block, args.triton_iters), dtype=torch.float32, device=device)
        self.sink = torch.zeros((), dtype=torch.float32, device=device)
        self.sink_clone: torch.Tensor | None = None
        self.cursor = 0

    def next(self) -> int:
        i = self.cursor % self.sets
        self.cursor += 1
        return i

    def call(self, arm: str) -> None:
        i = self.next()
        if arm == "read_torch":
            self.sink = self.src[i].sum(dtype=torch.float32)
        elif arm == "read_triton":
            triton_read(self.src[i], self.partials, self.args.triton_block, self.args.triton_iters, self.args.triton_warps)
        elif arm == "copy":
            self.dst[i].copy_(self.src[i])
        elif arm == "clone":
            self.sink_clone = self.src[i].clone()
        elif arm == "fill":
            self.dst[i].fill_(1.5)
        elif arm == "zero":
            self.dst[i].zero_()
        else:
            raise ValueError(arm)

    def self_check(self) -> dict[str, Any]:
        """The read kernels' reduction equals the float64 reference; copy/clone equal the source; fill/zero hold their value."""
        res: dict[str, Any] = {}
        ref = ref_sum(self.src[0])
        res["read_torch_rel"] = abs(float(self.src[0].sum(dtype=torch.float32)) - ref) / ref
        triton_read(self.src[0], self.partials, self.args.triton_block, self.args.triton_iters, self.args.triton_warps)
        res["read_triton_rel"] = abs(float(self.partials.sum(dtype=torch.float64)) - ref) / ref
        # negative control: one element raised by 8 must move the Triton total by about 8 (the loads are live)
        before = float(self.partials.sum(dtype=torch.float64))
        old = self.src[0][self.n // 2].item()
        self.src[0][self.n // 2] = old + 8.0
        triton_read(self.src[0], self.partials, self.args.triton_block, self.args.triton_iters, self.args.triton_warps)
        res["read_control_delta"] = float(self.partials.sum(dtype=torch.float64)) - before
        self.src[0][self.n // 2] = old
        res["read_control_moved"] = abs(res["read_control_delta"] - 8.0) < 0.1
        self.dst[0].copy_(self.src[0])
        res["copy_equal"] = bool(torch.equal(self.dst[0], self.src[0]))
        res["clone_equal"] = bool(torch.equal(self.src[0].clone(), self.src[0]))
        self.dst[0].fill_(1.5)
        res["fill_ok"] = bool((self.dst[0] == 1.5).all().item())
        self.dst[0].zero_()
        res["zero_ok"] = not bool(self.dst[0].any().item())
        return res


class LmHead:
    """The appendix-A shape: out[M, rows] = x[M, hidden] @ W[rows, hidden]^T, fp16, several weight sets cycled."""

    def __init__(self, shape: dict[str, Any], rows_m: Sequence[int], sets: int, device: str):
        self.shape, self.sets = shape, sets
        rows, hidden = shape["rows"], shape["hidden"]
        self.w = [torch.empty(rows, hidden, dtype=torch.float16, device=device).uniform_(-1.0, 1.0) for _ in range(sets)]
        self.x = {m: torch.empty(m, hidden, dtype=torch.float16, device=device).uniform_(-1.0, 1.0) for m in rows_m}
        self.out = {m: torch.empty(m, rows, dtype=torch.float16, device=device) for m in rows_m}
        self.cursor = {m: 0 for m in rows_m}

    def call(self, m: int) -> None:
        i = self.cursor[m] % self.sets
        self.cursor[m] += 1
        torch.mm(self.x[m], self.w[i].t(), out=self.out[m])

    def self_check(self, m: int, cols: int = 2048) -> float:
        """Relative L2 of the kernel output vs a float32 matmul on the first ``cols`` weight rows."""
        torch.mm(self.x[m], self.w[0].t(), out=self.out[m])
        ref = self.x[m].float() @ self.w[0][:cols].float().t()
        got = self.out[m][:, :cols].float()
        return float((got - ref).norm() / ref.norm())

    def bytes(self) -> int:
        return self.shape["bytes"]


def make_timed(fn: Callable[[], None], inner: int, graph: bool) -> Callable[[], None]:
    """A callable that runs ``inner`` calls: captured once into a CUDA graph (default) or eager."""
    for _ in range(max(2, inner)):  # compile / allocate on the side stream before capture
        fn()
    torch.cuda.synchronize()
    if not graph:
        def eager() -> None:
            for _ in range(inner):
                fn()
        return eager
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    return g.replay


def time_arms(arms: dict[str, tuple[Callable[[], None], int]], *, rounds: int, iters: int, warmup: int,
              clock: Callable[[Callable[[], None]], float] | None = None) -> dict[str, dict[str, float]]:
    """``arms[name] = (run, calls_per_run)``; returns per-CALL microseconds. CUDA events by default; ``clock`` (tests) times a run."""
    def cuda_clock(fn: Callable[[], None]) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0

    clk = clock or cuda_clock
    for run, _ in arms.values():
        for _ in range(warmup):
            run()
    if clock is None:
        torch.cuda.synchronize()
    per_arm: dict[str, list[list[float]]] = {name: [] for name in arms}
    for name in abba_order(list(arms), rounds):
        run, calls = arms[name]
        per_arm[name].append([clk(run) / calls for _ in range(iters)])
    return {name: summarize_timing(r) for name, r in per_arm.items()}


# ------------------------------------------------------------------ cli
def expand_arms(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for a in GROUPS.get(n, (n,)):
            if a not in out:
                out.append(a)
    return out


def valid_arm(name: str) -> bool:
    return name in KIND or name in GROUPS or name == "lmhead"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES), help="buffer sizes, binary units (256M 512M 1G); >= 8 MiB")
    p.add_argument("--arms", nargs="+", default=["read", "d2d", "write", "lmhead"],
                   help="read d2d write lmhead, or single arms: " + " ".join(KIND))
    p.add_argument("--config", default=DEFAULT_CONFIG, help="checkpoint config.json (the index next to it is read too)")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--lmhead-rows", type=int, nargs="+", default=[1, 5], help="M of the lm_head GEMM (1 = draft, 5 = target)")
    p.add_argument("--sets", type=int, default=2, help="buffer sets cycled (>= 2); lm_head uses max(sets, 4)")
    p.add_argument("--inner", type=int, default=4, help="calls per CUDA-graph replay (lm_head: 10x this)")
    p.add_argument("--iters", type=int, default=20, help="replays per timing turn")
    p.add_argument("--rounds", type=int, default=4, help="ABBA timing rounds")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no-graph", action="store_true", help="eager launches, one event pair per run")
    p.add_argument("--triton-block", type=int, default=2048)
    p.add_argument("--triton-iters", type=int, default=8, help="tiles per program")
    p.add_argument("--triton-warps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-used-mib", type=int, default=1024)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--ncu", action="store_true", help="print the Nsight Compute commands and exit (runs nothing)")
    p.add_argument("--ncu-dir", default="/data/bench")
    p.add_argument("--ncu-target", action="store_true", help="the process ncu profiles: eager calls of the selected arms only")
    a = p.parse_args(argv)
    bad = [x for x in a.arms if not valid_arm(x)]
    if bad:
        p.error(f"unknown arm(s) {bad}")
    try:
        a.size_bytes = [parse_size(s) for s in a.sizes]
    except ValueError as exc:
        p.error(str(exc))
    if min(a.iters, a.rounds, a.inner, a.triton_block, a.triton_iters, a.triton_warps) <= 0 or a.warmup < 0 or a.sets < 2:
        p.error("--iters, --rounds, --inner and the --triton-* values must be positive, --sets >= 2")
    if a.iters < 20 and not (a.ncu or a.ncu_target):
        p.error("--iters must be >= 20 (plan 072 M-08 timing floor)")
    if a.triton_block & (a.triton_block - 1):
        p.error("--triton-block must be a power of two")
    a.arm_list = [x for x in expand_arms(a.arms) if x != "lmhead"]
    a.do_lmhead = "lmhead" in a.arms
    return a


def ncu_commands(script: str, arms: Sequence[str], args: argparse.Namespace) -> list[str]:
    big = size_label(max(args.size_bytes))
    names = list(arms) + ([f"lmhead_m{m}" for m in args.lmhead_rows] if args.do_lmhead else [])
    out = []
    for arm in names:
        if arm in ("copy", "clone"):
            continue  # D2D memcpy is not a kernel launch ncu can filter on; use nsys for it
        regex = NCU_KERNEL.get(arm, "gemv|gemm|cutlass")
        target = "lmhead" if arm.startswith("lmhead") else arm
        extra = f" --lmhead-rows {arm.removeprefix('lmhead_m')}" if arm.startswith("lmhead") else ""
        out.append(f"ncu --metrics {NCU_METRICS} --kernel-name 'regex:{regex}' --launch-skip 4 --launch-count 4 --csv --page raw "
                   f"--print-units base --log-file {args.ncu_dir}/ncu_hbm_{arm}.csv "
                   f"python {script} --ncu-target --arms {target} --sizes {big}{extra}")
    return out


def git_tip() -> str:
    try:
        root = Path(__file__).resolve().parent
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _version(name: str) -> str:
    with contextlib.suppress(Exception):
        module = __import__(name)
        return module.__version__
    return "n/a"


def smi_clocks() -> dict[str, Any]:
    """Current / max SM and memory clocks, power and temperature of the visible device (nvidia-smi; {} when unavailable)."""
    with contextlib.suppress(Exception):
        uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
        fields = "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
        text = subprocess.run(["nvidia-smi", "-i", uuid, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
        return dict(zip(fields.split(","), [v.strip() for v in text.split(",")], strict=False))
    return {}


def device_info() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
            "total_mib": props.total_memory // (1 << 20), "torch": torch.__version__}


def refuse_busy_gpu(max_used_mib: int) -> str | None:
    free, total = torch.cuda.mem_get_info()
    used_mib = (total - free) / (1 << 20)
    if used_mib > max_used_mib:
        return (f"the GPU already holds {used_mib:.0f} MiB (limit {max_used_mib}); run on an idle GPU, "
                "never one a server holds, or pass --allow-busy")
    return None


def table_lines(rows: Sequence[dict[str, Any]]) -> list[str]:
    head = (f"{'arm':<12} {'size':>5} {'med us':>9} {'p10':>8} {'p90':>8} {'spread%':>8} {'GB/s':>7} {'[lo,hi]':>15} "
            f"{'traffic':>8} {'%898':>6} {'%672':>6}")
    lines = ["TIMING (per call, graph replay unless --no-graph; GB/s = bytes counted / median; traffic = HBM read+write; "
             "bytes: read/write/lm_head = the buffer, copy = moved one way)", head]
    for r in rows:
        band = "[%.0f,%.0f]" % (r["gbps_lo"], r["gbps_hi"])
        lines.append(f"{r['arm']:<12} {size_label(r['size_bytes']):>5} {r['median_us']:>9.1f} {r['p10_us']:>8.1f} {r['p90_us']:>8.1f} "
                     f"{r['round_spread_pct']:>8.2f} {r['gbps']:>7.1f} {band:>15} "
                     f"{r['traffic_gbps']:>8.1f} {pct(r['traffic_gbps'], SPEC_GBPS):>6.1f} {pct(r['traffic_gbps'], PLAN_GBPS['672']):>6.1f}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        shape = read_lmhead_shape(args.config, args.tp)
    except ValueError as exc:
        print(f"unusable arguments: {exc}")
        return 2
    match = "MATCH" if shape["bytes"] == LMHEAD_BYTES_EXPECTED else f"DIFFERS from the {LMHEAD_BYTES_EXPECTED} B of plan 071 appendix A"
    print(f"LMHEAD_BYTES: {shape['bytes']} = {shape['rows']} rows x {shape['hidden']} x {ELEM_BYTES} B fp16 per rank "
          f"(vocab {shape['vocab']}, tp {shape['tp']}; {shape['source']}; checkpoint dtype {shape['checkpoint_dtype']}, "
          f"lm_head in quantization ignore list: {shape['quant_ignore_lm_head']}, index entry {shape['index_entry']}) {match}")
    if args.ncu:
        script = str(Path(__file__).resolve())
        print("# ncu serialises kernels and flushes caches: GB/s from ncu time is not the serving-level number (plan 072 M-10)")
        print("# D2D copy_/clone() are cudaMemcpy, not kernels: ncu cannot filter them (use nsys for those)")
        for line in ncu_commands(script, args.arm_list, args):
            print(line)
        return 0
    if not torch.cuda.is_available():
        print("needs a CUDA device (set CUDA_VISIBLE_DEVICES to an idle GPU)")
        return 2
    if not args.allow_busy:
        message = refuse_busy_gpu(args.max_used_mib)
        if message:
            print(message)
            return 2
    device = "cuda:0"
    info = device_info()
    print(f"device {info['name']} capability {info['capability']}, torch {torch.__version__}")
    if info["capability"] != [7, 0]:
        print("NOTE: not an sm_70 device; the numbers are not V100 numbers")
    free, _ = torch.cuda.mem_get_info()
    lm_sets = max(args.sets, 4)
    need = max(2 * args.sets * max(args.size_bytes) if args.arm_list else 0, lm_sets * shape["bytes"] if args.do_lmhead else 0) + (1 << 28)
    if need > free:
        print(f"needs about {need / (1 << 30):.1f} GiB free, the GPU has {free / (1 << 30):.1f} GiB: use smaller --sizes or fewer --sets")
        return 2
    torch.manual_seed(args.seed)
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    clocks_before = smi_clocks()
    with torch.inference_mode():
        if args.ncu_target:
            sw = Sweep(max(args.size_bytes), args.sets, device, args)
            for arm in args.arm_list:
                for _ in range(8):
                    sw.call(arm)
            if args.do_lmhead:
                lm = LmHead(shape, args.lmhead_rows, lm_sets, device)
                for m in args.lmhead_rows:
                    for _ in range(8):
                        lm.call(m)
            torch.cuda.synchronize()
            print("NCU_TARGET_DONE")
            return 0
        for size in args.size_bytes:
            sw = Sweep(size, args.sets, device, args)
            label = size_label(size)
            if any(a in READ_ARMS or a in D2D_ARMS or a in WRITE_ARMS for a in args.arm_list):
                chk = sw.self_check()
                checks[label] = chk
                tol = REL_TOL["read_torch"]
                read_ok = chk["read_torch_rel"] < tol and chk["read_triton_rel"] < tol and chk["read_control_moved"]
                print(f"READ_CHECK {label}: {'PASS' if read_ok else 'FAIL'} (rel vs float64: torch {chk['read_torch_rel']:.2e}, "
                      f"triton {chk['read_triton_rel']:.2e}, tol {tol}; control delta {chk['read_control_delta']:.2f})")
                print(f"COPY_CHECK {label}: {'PASS' if chk['copy_equal'] and chk['clone_equal'] else 'FAIL'} "
                      f"(copy_ equal {chk['copy_equal']}, clone equal {chk['clone_equal']})")
                print(f"WRITE_CHECK {label}: {'PASS' if chk['fill_ok'] and chk['zero_ok'] else 'FAIL'} "
                      f"(fill {chk['fill_ok']}, zero {chk['zero_ok']})")
                failures += [f"{n} {label}" for n, good in (("read", read_ok), ("copy", chk["copy_equal"] and chk["clone_equal"]),
                                                            ("write", chk["fill_ok"] and chk["zero_ok"])) if not good]
            arms = {a: (make_timed(lambda a=a, sw=sw: sw.call(a), args.inner, not args.no_graph), args.inner) for a in args.arm_list}
            if arms:
                timings = time_arms(arms, rounds=args.rounds, iters=args.iters, warmup=args.warmup)
                for a in args.arm_list:
                    rows.append(make_row(a, size, size, timings[a]))
            del sw
            torch.cuda.empty_cache()
        if args.do_lmhead:
            lm = LmHead(shape, args.lmhead_rows, lm_sets, device)
            inner = args.inner * 10
            bad = []
            for m in args.lmhead_rows:
                rel = lm.self_check(m)
                good = rel < REL_TOL["lmhead"]
                bad += [] if good else [f"lmhead M={m}"]
                print(f"LMHEAD_CHECK M={m}: {'PASS' if good else 'FAIL'} (rel L2 vs float32 on 2048 rows {rel:.2e}, tol {REL_TOL['lmhead']})")
            failures += bad
            arms = {f"lmhead_m{m}": (make_timed(lambda m=m: lm.call(m), inner, not args.no_graph), inner) for m in args.lmhead_rows}
            timings = time_arms(arms, rounds=args.rounds, iters=args.iters, warmup=args.warmup)
            for m in args.lmhead_rows:
                rows.append(make_row(f"lmhead_m{m}", lm.bytes(), lm.bytes(), timings[f"lmhead_m{m}"], note="torch.mm (cuBLAS), not the production cutlass kernel"))
    clocks_after = smi_clocks()
    print("\n".join(table_lines(rows)))
    ceil = ceilings(rows)
    for line in verdict_lines(ceil):
        print(line)
    print("COPY_CONVENTION: plan 071 appendix B counts nsys memcpy bytes one way and kernel READ bytes; HBM_CEILING_GBPS copy is "
          "read+write traffic = 2x COPY_ONEWAY_GBPS")
    if not rows:
        failures.append("no arm selected")
    verdict = "PASS" if not failures else f"FAIL ({', '.join(failures)})"
    print(f"SELF_CHECK: {verdict}")
    report = {
        "meta": {"git_tip": git_tip(), "torch": torch.__version__, "triton": _version("triton"), "device": info,
                 "clocks_before": clocks_before, "clocks_after": clocks_after, "lmhead": shape, "spec_gbps": SPEC_GBPS,
                 "plan_gbps": PLAN_GBPS, "args": {k: v for k, v in vars(args).items()},
                 "convention": "gbps = bytes counted / time (copy one way); traffic_gbps = HBM read+write (copy x2); "
                               "ceilings use traffic_gbps; decimal GB/s; sizes are binary (256M = 268,435,456 B)"},
        "rows": rows, "self_check": checks, "ceiling": ceil, "verdict": verdict,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
