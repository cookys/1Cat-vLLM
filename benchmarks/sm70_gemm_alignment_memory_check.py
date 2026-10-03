# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does the fp16 GEMM of the unquantized prefill projections depend on buffer
alignment or on free GPU memory?

Hypothesis under test. Text parity differs between two server instances that
differ only in memory footprint (``--gpu-memory-utilization`` 0.92 against 0.94, a
20 KB staging buffer that moves the KV pool by one block), while replaying the same
request inside one instance is bit-identical, and the differences first show in
cold prefill chunks of 6464 and 1616 tokens. cuDNN is already excluded. The
remaining candidate is cuBLAS algorithm selection for the fp16 projections the
quantization config leaves unquantized (``*.self_attn.*``, ``*.linear_attn.*``,
``*.mlp.gate*``). cuBLAS picks a kernel, a tile and a split-K factor from the
shape, the pointer alignment of A, B and C, the workspace it is given and the math
mode, and a different split-K factor changes the order in which partial sums are
added, so the output differs in the last bits.

The call under test is the one the engine makes. ``UnquantizedLinearMethod.apply``
(``vllm/model_executor/layers/linear.py:508``) calls
``dispatch_unquantized_gemm()(layer, x, layer.weight, bias)``, which on CUDA is
``default_unquantized_gemm`` (``vllm/model_executor/layers/utils.py:120``), and that
is ``torch.nn.functional.linear(x, weight, bias)``. Here: fp16, ``bias=None``, ``x``
is ``[M, K]``, ``weight`` is ``[N, K]``. vLLM is not imported, so the script runs
with torch alone and is importable without a GPU.

What it measures, one subprocess per cuBLAS setting so no handle, workspace or
heuristic state leaks between settings:

1. Shapes. ``--shapes M,K,N ...`` is taken as given. Otherwise the default list is
   built from ``--config`` (``text_config`` or the top level) for ``--tp`` ranks and
   crossed with ``--m`` (default 17, 1616, 6464, 8080, the chunk sizes in the logs):

   - ``qkv_gated``  K=hidden, N=(2*heads/tp + 2*max(1, kv/tp))*head_dim. This is
     what ``QKVParallelLinear`` builds in ``models/qwen4_exp/nvidia/qsa.py:301`` and
     ``:318``: the sigmoid output gate is packed next to Q and the KV heads are
     replicated when ``kv < tp``. 3584 at tp=4.
   - ``qkv``  K=hidden, N=(heads + 2*kv)*head_dim/tp. 1792 at tp=4. It is the
     formula in the task brief and NOT what the engine runs; it is kept as an
     extra shape because it costs little and marks where the two differ.
   - ``o_proj``  K=heads*head_dim/tp, N=hidden. (1536, 2560) at tp=4.
   - ``gdn_qkvz``  K=hidden, N=(2*lk_heads*lk_dim + 2*lv_heads*lv_dim)/tp. 4096.
   - ``router``  K=hidden, N=num_experts, replicated. 512.

   ``--n`` replaces the projection list by ``K=hidden, N=<each value>``. Other
   unquantized projections worth adding with ``--shapes``, widths from
   ``models/qwen4_exp/nvidia/low_latency_gemm.py`` at tp=4: GDN in_proj_ba
   (24, 2560), indexer q/k (640, 2560), shared-expert gate/up (320, 2560).
2. Inputs. Seeded on the CPU, so every process builds the same bytes: ``x`` is
   ``randn * 4`` (activation scale), ``weight`` is ``randn / sqrt(K)``.
3. Alignment scenarios, in one worker per setting. ``A<off>``: ``x`` is a
   contiguous view that starts ``off`` bytes into a larger buffer, for the offsets
   of ``--offsets`` (default 0 16 32 64 128 256 512 1024). The buffer is aligned so
   that the view address is exactly ``off`` bytes above a power-of-two boundary
   larger than twice the biggest offset, so ``0`` is the strongest alignment and
   every offset is a different alignment class. ``B<off>``: the weight is shifted
   instead, for ``--b-offsets`` (default 16), ``x`` at 0. Offsets must be multiples
   of 2 bytes. Offsets below 16 (try ``--offsets 0 2 4 8 16 ...``) force cuBLAS onto
   its unaligned kernels and make a positive control: if even those give the same
   bits, nothing in this call reads the pointer alignment.

   C (the output) cannot be placed: ``F.linear`` has no ``out=``, the output comes
   from the caching allocator and is 512-byte aligned in the engine as well. C
   alignment is engine-controlled, so it is not a scenario; the observed
   ``out.data_ptr()`` modulo 256 and 512 is recorded for every run.
4. Free-memory scenarios (``whole``, then 16, 8, 4, 2, 1 GiB and 512 MiB free at the
   moment the GEMM runs; a dummy tensor takes the rest, ``x`` and ``weight`` at 0).
   Two modes, ``--memory-mode``:

   - ``inprocess``: one worker per setting runs the scenarios one after another.
     cuBLAS keeps its handle and workspace for the process lifetime, so this is the
     steady state: does free memory matter once the handle exists.
   - ``fresh``: one worker per scenario. The dummy is taken after the inputs are
     resident and before the first cuBLAS call, so the handle and the workspace
     are created under the reduced memory, as an engine instance with a different
     footprint does at start-up. The ``whole`` worker saves its outputs in a
     temporary directory so later workers can report max|diff| against it.

   ``both`` (default) runs both and keeps them apart in the report.
5. Per scenario: sha256 of each of ``--repeats`` (3) outputs, whether they agree,
   max|diff| to the baseline scenario (``A0`` / ``whole``) and to an fp32 reference
   computed from the same inputs on the GPU (``--cpu-reference`` adds a CPU fp32
   reference for the baseline scenario), and the CUDA kernel names of one extra
   profiled call from ``torch.profiler`` (kernels with ``cutlass``, ``splitK``,
   ``gemv``, ``sm70`` or ``volta`` in the name are flagged; first 160 characters
   kept). Whether the profiled call reproduces the first repeat is recorded too.
6. Settings, each applied before the first CUDA call of its workers:

   - ``default``: nothing set. ``CUBLAS_WORKSPACE_CONFIG`` and
     ``DISABLE_ADDMM_CUDA_LT`` are removed from the worker environment.
   - ``wscap4096``: ``CUBLAS_WORKSPACE_CONFIG=:4096:8``.
   - ``wscap16``: ``CUBLAS_WORKSPACE_CONFIG=:16:8`` (the cap vLLM's batch-invariant
     mode uses to switch split-K off).
   - ``no_reduced_precision``:
     ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False``.
   - ``deterministic``: ``torch.use_deterministic_algorithms(True)`` plus
     ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, which torch requires for it.
   - ``no_lt``: ``DISABLE_ADDMM_CUDA_LT=1``. It only switches the addmm-with-bias
     path off, and this call has no bias, so expect no change from it: it is a
     control, and the ALGO lines show whether the kernels moved.

Verdict printed on stdout (exit status 0 for a clean run whatever the verdict):

    ALIGN_DEPENDENT: YES/NO           default setting, any A or B offset differs
    MEMORY_DEPENDENT: YES/NO          default setting, any free-memory run differs
    REPEATS_IDENTICAL: YES/NO         the repeats of every scenario agree
    WSCAP_FIXES (wscap4096): YES/NO/N/A    one line per cap
    NO_REDUCED_PRECISION_FIXES: YES/NO/N/A
    DETERMINISTIC_FIXES: YES/NO/N/A
    NO_LT_FIXES: YES/NO/N/A
    SAME_BYTES_AS_DEFAULT (<setting>): YES/NO   does the setting move the baseline?
    CROSS_PROCESS_REPRODUCIBLE: YES/NO   same baseline bits in every default worker
    DEPENDENT_SHAPES: <M>x<K>x<N> (label), ...   shapes with any dependence
    ALGO: <setting> <M>x<K>x<N> <scenario>: <kernel names>

A ``*_FIXES`` line is N/A when the default setting shows no dependence, YES when the
setting shows none, NO when it still does. ``NOT_RUN`` means the setting or the
scenario group was not selected, ``UNKNOWN`` that the data cannot tell (a worker
failed, or fewer than two scenarios ran). ALIGN_DEPENDENT and MEMORY_DEPENDENT are
the default setting only. ALGO lines are printed for the baseline scenario of every
setting and shape, and for any scenario whose kernels differ from it
(``--print-all-algo`` prints all); an ``ALGO_SUMMARY`` line per setting and shape
counts the scenarios that ran the baseline kernels, and the JSON report has every
kernel list. One more line per setting and shape gives the scenario count, the
distinct output hashes and the max|diff| values.

Exit status: 0 a clean run; 2 no CUDA device or unusable arguments; 1 a worker
failed. An unreadable config is not an error: the production dimensions are used and
the output says so.

    CUDA_VISIBLE_DEVICES=0 /data/venvs/1cat-m589/bin/python \
        benchmarks/sm70_gemm_alignment_memory_check.py \
        --out /data/bench/gemm_alignment_memory_check.json

Use one idle GPU: the memory scenarios take memory away from the process that runs
the GEMM, and any other process on the device moves the baseline. With the defaults
there are 9 workers per setting (1 alignment, 1 in-process memory, 7 fresh memory)
and 54 in all, each a fresh Python process that hashes every output; the run time is
not measured yet. ``--memory-mode inprocess``, ``--settings``, ``--m`` and
``--no-profile`` shorten it.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_CONFIG = "/data/models/Qwen3.8-Flash-Next-NVFP4-e4m3kv/config.json"
DEFAULT_TP = 4
DEFAULT_M = (17, 1616, 6464, 8080)
DEFAULT_OFFSETS = (0, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_B_OFFSETS = (16,)
DEFAULT_FREE_MIB = (16384, 8192, 4096, 2048, 1024, 512)
MODEL_KEYS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "num_experts",
)
# Used when the config is unreadable or lacks one of MODEL_KEYS.
PRODUCTION_MODEL = {
    "hidden_size": 2560,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "num_experts": 512,
}
DTYPE = torch.float16
ACTIVATION_SCALE = 4.0
MIB = 1 << 20
GIB = 1 << 30
# A scenario whose target is within this much of the current free memory is the
# whole-memory scenario again; skip it.
MIN_DUMMY_BYTES = 32 * MIB
MIN_ALIGN_BASE = 512
RESULT_PREFIX = "RESULT_JSON:"
# The device the workers run on; the CPU tests point it at "cpu".
WORKER_DEVICE = "cuda:0"
ALIGN_BASELINE = "A0"
MEMORY_BASELINE = "whole"
GROUPS = ("align", "memory_inprocess", "memory_fresh")
MEMORY_GROUPS = ("memory_inprocess", "memory_fresh")
BASELINES = {
    "align": ALIGN_BASELINE,
    "memory_inprocess": MEMORY_BASELINE,
    "memory_fresh": MEMORY_BASELINE,
}
# Variables a setting may set; the others are removed from the worker environment
# so the operator's shell cannot turn the default setting into another one.
MANAGED_ENV = ("CUBLAS_WORKSPACE_CONFIG", "DISABLE_ADDMM_CUDA_LT")
NO_CUDA_MESSAGE = (
    "needs a CUDA device: set CUDA_VISIBLE_DEVICES to one idle GPU "
    "(this script measures GPU kernels and has no CPU fallback)"
)
KERNEL_MARKERS = {
    "cutlass": "cutlass",
    "splitk": "splitk",
    "gemv": "gemv",
    "sm70": "sm70",
    "volta": "volta",
}
NOTES = [
    (
        "C alignment is engine-controlled: F.linear has no out=, the output comes "
        "from the caching allocator (512-byte aligned). Not a scenario; the "
        "observed out.data_ptr() modulo 256/512 is recorded per run."
    ),
    (
        "no_lt (DISABLE_ADDMM_CUDA_LT) only affects addmm with a bias; this call "
        "has none, so it is a control. Compare its ALGO lines with default."
    ),
]


# -------------------------------------------------------------------- shapes


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.m}x{self.k}x{self.n}"


def parse_shape_triple(text: str) -> tuple[int, int, int]:
    """``M,K,N`` as three positive integers (argparse ``type``)."""
    parts = text.replace("x", ",").split(",")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        values = ()
    if len(values) != 3 or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            f"shape {text!r} is not M,K,N with three positive integers"
        )
    return values  # type: ignore[return-value]


def read_model_dims(path: str | Path) -> dict[str, int] | None:
    """The model dimensions the default shapes need, or None if absent.

    Looks in ``text_config`` first and then at the top level; all of MODEL_KEYS
    must be integers in the same section.
    """
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    for section in (config.get("text_config"), config):
        if not isinstance(section, dict):
            continue
        if all(isinstance(section.get(k), int) for k in MODEL_KEYS):
            return {k: section[k] for k in MODEL_KEYS}
    return None


def _divide(total: int, tp: int, what: str) -> int:
    if total % tp:
        raise ValueError(f"{what} = {total} is not divisible by --tp {tp}")
    return total // tp


def qkv_gated_width(dims: dict[str, int], tp: int) -> int:
    """Local width of ``QKVParallelLinear`` with the packed sigmoid output gate.

    ``models/qwen4_exp/nvidia/qsa.py:301`` and ``:318``: total Q heads are doubled
    for the gate;
    KV heads are split over ranks, or replicated (one per rank) when ``kv < tp``.
    """
    q_local = _divide(2 * dims["num_attention_heads"], tp, "2 * num_attention_heads")
    kv = dims["num_key_value_heads"]
    if kv >= tp:
        kv_local = _divide(kv, tp, "num_key_value_heads")
    else:
        if tp % kv:
            raise ValueError(f"--tp {tp} is not a multiple of num_key_value_heads {kv}")
        kv_local = 1
    return (q_local + 2 * kv_local) * dims["head_dim"]


def default_projections(dims: dict[str, int], tp: int) -> list[tuple[str, int, int]]:
    """``(label, K, N)`` of the projections to test, per TP rank."""
    hidden = dims["hidden_size"]
    heads, kv, head_dim = (
        dims["num_attention_heads"],
        dims["num_key_value_heads"],
        dims["head_dim"],
    )
    gdn = (
        2 * dims["linear_num_key_heads"] * dims["linear_key_head_dim"]
        + 2 * dims["linear_num_value_heads"] * dims["linear_value_head_dim"]
    )
    return [
        ("qkv_gated", hidden, qkv_gated_width(dims, tp)),
        ("qkv", hidden, _divide((heads + 2 * kv) * head_dim, tp, "qkv width")),
        ("o_proj", _divide(heads * head_dim, tp, "o_proj input width"), hidden),
        ("gdn_qkvz", hidden, _divide(gdn, tp, "GDN in_proj_qkvz width")),
        ("router", hidden, dims["num_experts"]),
    ]


def build_shapes(
    dims: dict[str, int],
    tp: int,
    m_list: Sequence[int],
    n_override: Sequence[int] | None = None,
    explicit: Sequence[tuple[int, int, int]] | None = None,
) -> list[Shape]:
    """The shape list: ``explicit`` as given, else projections x ``m_list``."""
    shapes: list[Shape] = []
    seen: set[str] = set()

    def add(shape: Shape) -> None:
        if shape.key not in seen:
            seen.add(shape.key)
            shapes.append(shape)

    if explicit:
        for m, k, n in explicit:
            add(Shape(m, k, n, "explicit"))
        return shapes
    if n_override:
        projections = [(f"n{n}", dims["hidden_size"], n) for n in n_override]
    else:
        projections = default_projections(dims, tp)
    for label, k, n in projections:
        for m in m_list:
            add(Shape(m, k, n, label))
    return shapes


def resolve_shapes(args: argparse.Namespace) -> tuple[list[Shape], str]:
    """Shapes and a line saying where the dimensions came from."""
    if args.shapes:
        return build_shapes({}, args.tp, args.m, explicit=args.shapes), (
            "explicit --shapes"
        )
    dims = read_model_dims(args.config)
    source = f"config {args.config}"
    if dims is None:
        dims = dict(PRODUCTION_MODEL)
        source = "production defaults (config unreadable or incomplete)"
    shapes = build_shapes(dims, args.tp, args.m, n_override=args.n)
    return shapes, f"{source}, tp={args.tp}"


# ------------------------------------------------------------------ settings


def build_settings() -> dict[str, dict[str, Any]]:
    """The cuBLAS settings to compare. ``env`` is applied to the worker process."""

    def setting(
        env: dict[str, str] | None = None,
        reduced_precision_reduction: bool = True,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        return {
            "env": env or {},
            "allow_fp16_reduced_precision_reduction": reduced_precision_reduction,
            "deterministic_algorithms": deterministic,
        }

    return {
        "default": setting(),
        "wscap4096": setting({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}),
        "wscap16": setting({"CUBLAS_WORKSPACE_CONFIG": ":16:8"}),
        "no_reduced_precision": setting(reduced_precision_reduction=False),
        "deterministic": setting(
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, deterministic=True
        ),
        "no_lt": setting({"DISABLE_ADDMM_CUDA_LT": "1"}),
    }


def _apply_setting(setting: dict[str, Any]) -> dict[str, Any]:
    """Set the torch-side flags of a setting; return what is in force.

    The environment part was set by the driver before the process started. This runs
    before the worker's first CUDA call.
    """
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = setting[
        "allow_fp16_reduced_precision_reduction"
    ]
    torch.use_deterministic_algorithms(setting["deterministic_algorithms"])
    return {
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "env": {name: os.environ.get(name) for name in MANAGED_ENV},
    }


# -------------------------------------------------------- inputs and placement


def derive_seed(seed: int, tag: str, *dims: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{tag}:{','.join(map(str, dims))}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def make_activations(seed: int, m: int, k: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "A", m, k))
    return (torch.randn(m, k, generator=generator) * ACTIVATION_SCALE).to(DTYPE)


def make_weights(seed: int, k: int, n: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "W", k, n))
    return (torch.randn(n, k, generator=generator) * k**-0.5).to(DTYPE)


class InputCache:
    """Seeded inputs on the device, built once per (M, K) and (K, N)."""

    def __init__(self, seed: int, device: torch.device):
        self.seed = seed
        self.device = device
        self._activations: dict[tuple[int, int], torch.Tensor] = {}
        self._weights: dict[tuple[int, int], torch.Tensor] = {}

    def activations(self, m: int, k: int) -> torch.Tensor:
        if (m, k) not in self._activations:
            self._activations[(m, k)] = make_activations(self.seed, m, k).to(
                self.device
            )
        return self._activations[(m, k)]

    def weights(self, k: int, n: int) -> torch.Tensor:
        if (k, n) not in self._weights:
            self._weights[(k, n)] = make_weights(self.seed, k, n).to(self.device)
        return self._weights[(k, n)]


@dataclass
class Placed:
    """A contiguous view ``offset_bytes`` into ``storage``, which it keeps alive."""

    view: torch.Tensor
    storage: torch.Tensor
    offset_bytes: int


def alignment_base(*offset_lists: Iterable[int]) -> int:
    """Power-of-two boundary above twice the largest offset (at least 512 bytes)."""
    largest = max([0, *[o for offsets in offset_lists for o in offsets]])
    base = MIN_ALIGN_BASE
    while base < 2 * largest:
        base *= 2
    return base


def place_with_offset(
    values: torch.Tensor,
    offset_bytes: int,
    base_bytes: int = MIN_ALIGN_BASE,
    device: torch.device | str | None = None,
) -> Placed:
    """Copy ``values`` into a zeroed buffer; return a view at a chosen byte offset.

    The view address is ``offset_bytes`` above an address that is a multiple of
    ``base_bytes``, so ``data_ptr % 256 == offset_bytes % 256`` whatever the
    allocator returned. The view has the shape of ``values`` and is contiguous.
    """
    esize = values.element_size()
    if offset_bytes < 0 or offset_bytes % esize:
        raise ValueError(
            f"offset {offset_bytes} must be a non-negative multiple of the "
            f"element size ({esize} bytes)"
        )
    if base_bytes < 1 or base_bytes % esize:
        raise ValueError(f"base {base_bytes} must be a multiple of {esize} bytes")
    target = values.device if device is None else torch.device(device)
    count = values.numel()
    storage = torch.zeros(
        count + (base_bytes + offset_bytes) // esize, dtype=values.dtype, device=target
    )
    pad = (-storage.data_ptr()) % base_bytes
    if pad % esize:
        raise RuntimeError("the allocator returned an address that is not aligned")
    flat = storage.narrow(0, (pad + offset_bytes) // esize, count)
    flat.copy_(values.reshape(-1))
    return Placed(flat.view(values.shape), storage, offset_bytes)


@dataclass(frozen=True)
class AlignScenario:
    name: str
    a_offset: int
    b_offset: int


def plan_alignment(
    a_offsets: Sequence[int], b_offsets: Sequence[int]
) -> list[AlignScenario]:
    """``A<off>`` scenarios (``A0`` first, the baseline), then ``B<off>`` ones."""
    a_list = [0, *sorted({o for o in a_offsets if o != 0})]
    b_list = sorted({o for o in b_offsets if o != 0})
    return [AlignScenario(f"A{o}", o, 0) for o in a_list] + [
        AlignScenario(f"B{o}", 0, o) for o in b_list
    ]


# ----------------------------------------------------------- memory scenarios


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    # Free memory wanted at the moment the GEMM is called; None means "whole".
    target_free_bytes: int | None
    dummy_bytes: int
    reachable: bool
    reason: str = ""


def scenario_name(target_free_bytes: int | None) -> str:
    if target_free_bytes is None:
        return MEMORY_BASELINE
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

    The workers re-plan each scenario against the free memory they see at that
    moment, because earlier scenarios free their tensors and the caching allocator
    keeps blocks; this list is for planning and for tests.
    """
    free_bytes, _ = mem_get_info()
    targets: list[int | None] = [None, *[m * MIB for m in free_mib]]
    return [plan_scenario(free_bytes, t) for t in targets]


def memory_targets(free_mib: Sequence[int]) -> dict[str, int | None]:
    """Scenario name to target free bytes, ``whole`` first."""
    targets: dict[str, int | None] = {MEMORY_BASELINE: None}
    for mib in free_mib:
        targets[scenario_name(mib * MIB)] = mib * MIB
    return targets


# ------------------------------------------------------------------- measures


def sha256_tensor(t: torch.Tensor) -> str:
    """sha256 of the raw bytes of a (CPU or CUDA) tensor."""
    cpu = t.detach().cpu().contiguous().reshape(-1)
    raw = cpu.view(torch.uint8) if cpu.element_size() > 1 else cpu.to(torch.uint8)
    return hashlib.sha256(raw.numpy()).hexdigest()


def _worse(worst: float, value: float) -> float:
    """The larger of two diffs; NaN wins, since ``max(0.0, nan)`` would hide it."""
    return float("nan") if worst != worst or value != value else max(worst, value)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 24) -> float:
    """Max |a - b| in fp32, in chunks so the temporaries stay small.

    NaN if any element difference is NaN (a NaN or an inf of the same sign).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    worst = 0.0
    for start in range(0, fa.numel(), chunk):
        diff = fa[start : start + chunk].float() - fb[start : start + chunk].float()
        worst = _worse(worst, float(diff.abs().max()))
    return worst


def reference_max_abs_diff(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, chunk_rows: int = 1024
) -> float:
    """Max |out - x.float() @ w.float().T|, the reference in fp32, in row chunks.

    Everything stays on the device ``x`` lives on, in chunks of ``chunk_rows`` rows,
    so the temporaries stay small even under the free-memory scenarios.
    """
    if x.shape[0] != out.shape[0] or w.shape[0] != out.shape[1]:
        raise ValueError("out does not match x @ w.T")
    w32 = w.float().t()
    worst = 0.0
    for start in range(0, x.shape[0], chunk_rows):
        reference = x[start : start + chunk_rows].float() @ w32
        diff = out[start : start + chunk_rows].float() - reference
        worst = _worse(worst, float(diff.abs().max()))
    return worst


class BaselineStore:
    """Baseline outputs per shape key; on disk too when a directory is given."""

    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory else None
        self._memory: dict[str, torch.Tensor] = {}

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{key}.pt"

    def put(self, key: str, tensor: torch.Tensor) -> None:
        if self.directory is None:
            self._memory[key] = tensor
        else:
            torch.save(tensor, self._path(key))

    def get(self, key: str) -> torch.Tensor | None:
        if key in self._memory:
            return self._memory[key]
        if self.directory is not None and self._path(key).exists():
            try:
                return torch.load(self._path(key), weights_only=True)
            except Exception:  # noqa: BLE001 - a missing baseline is reported as None
                return None
        return None


# ---------------------------------------------------------------- the GEMM


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # The engine call: linear.py:508 -> dispatch_unquantized_gemm()(layer, x,
    # layer.weight, bias) -> utils.py:126 torch.nn.functional.linear(x, weight, bias).
    return F.linear(x, w, None)


def _distinct(items: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def classify_kernels(names: Sequence[str]) -> dict[str, bool]:
    """Which of cutlass / splitK / gemv / sm70 / volta appear in the kernel names."""
    lower = [n.lower() for n in names]
    return {
        flag: any(marker in n for n in lower) for flag, marker in KERNEL_MARKERS.items()
    }


def describe_kernels(kernels: Sequence[str] | None) -> str:
    if kernels is None:
        return "unavailable (torch.profiler did not run)"
    if not kernels:
        return "none seen"
    return " | ".join(kernels)


def _kernel_names(run: Callable[[], Any]) -> list[str] | None:
    """Run ``run`` once and return the CUDA kernel names torch.profiler saw.

    ``run`` always executes, with or without a profiler: if the profiler cannot
    start (no CUPTI, a CPU build) the names are None and the call still runs. An
    exception from ``run`` itself, such as an out-of-memory error, propagates.
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
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
        cuda = torch.autograd.DeviceType.CUDA
        names = [
            event.name
            for event in prof.events()
            if event.device_type == cuda
            and "Memcpy" not in event.name
            and "Memset" not in event.name
        ]
        if not names:
            names = [
                event.key
                for event in prof.key_averages()
                if event.device_type == cuda
                and "Memcpy" not in event.key
                and "Memset" not in event.key
            ]
    except Exception:  # noqa: BLE001
        return None
    return _distinct([n[:160] for n in names])


def _device_info(device: torch.device) -> dict[str, Any]:
    """Name and capability of the worker's device (the driver never opens CUDA)."""
    if device.type != "cuda":
        return {"name": str(device), "capability": None}
    return {
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def _failure_status(exc: RuntimeError) -> str:
    """oom for an allocator failure, error for any other failure."""
    return "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"


def _reason(exc: BaseException) -> str:
    lines = str(exc).splitlines()
    return lines[0] if lines else repr(exc)


def _mem_get_info(device: torch.device) -> tuple[int, int]:
    return torch.cuda.mem_get_info(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _take_free_memory(device: torch.device, target: int | None):
    """Plan against the memory free now and allocate the dummy; (plan, dummy, free)."""
    torch.cuda.empty_cache()
    free, _ = _mem_get_info(device)
    plan = plan_scenario(free, target)
    dummy = None
    if plan.reachable and plan.dummy_bytes > 0:
        dummy = torch.empty(plan.dummy_bytes, dtype=torch.uint8, device=device)
    return plan, dummy, _mem_get_info(device)[0]


def run_case(
    x: torch.Tensor,
    w: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    store: BaselineStore,
    key: str,
    is_baseline: bool,
) -> dict[str, Any]:
    """The repeats, hashes, diffs and profiled kernels of one (shape, scenario)."""
    record: dict[str, Any] = {}
    hashes: list[str] = []
    first_cpu: torch.Tensor | None = None
    first_dev: torch.Tensor | None = None
    status = "ok"
    for _ in range(args.repeats):
        try:
            out = _gemm(x, w)
            _synchronize(device)
        except RuntimeError as exc:
            status = _failure_status(exc)
            record["reason"] = _reason(exc)
            break
        cpu = out.cpu()
        if first_dev is None:
            record["c_data_ptr_mod_256"] = out.data_ptr() % 256
            record["c_data_ptr_mod_512"] = out.data_ptr() % 512
            first_dev, first_cpu = out, cpu
        else:
            del out
        hashes.append(sha256_tensor(cpu))
    record["status"] = status
    if status != "ok" or first_cpu is None or first_dev is None:
        return record
    record["sha256"] = hashes
    record["repeats_identical"] = len(set(hashes)) == 1
    if is_baseline:
        store.put(key, first_cpu)
        record["max_abs_diff_vs_baseline"] = 0.0
    else:
        baseline = store.get(key)
        record["max_abs_diff_vs_baseline"] = (
            None if baseline is None else max_abs_diff(first_cpu, baseline)
        )
    try:
        record["max_abs_diff_vs_reference_fp32"] = reference_max_abs_diff(
            x, w, first_dev
        )
        if args.cpu_reference and is_baseline:
            record["max_abs_diff_vs_reference_cpu_fp32"] = reference_max_abs_diff(
                x.cpu(), w.cpu(), first_cpu
            )
    except RuntimeError as exc:
        record["reference_error"] = _reason(exc)
    del first_dev
    if not args.no_profile:
        holder: list[torch.Tensor] = []
        try:
            kernels = _kernel_names(lambda: holder.append(_gemm(x, w)))
        except RuntimeError as exc:
            record["profile_error"] = _reason(exc)
        else:
            record["kernels"] = kernels
            if kernels is not None:
                record["kernel_flags"] = classify_kernels(kernels)
            if holder:
                record["profiled_matches_repeat1"] = (
                    sha256_tensor(holder[0]) == hashes[0]
                )
    return record


# ------------------------------------------------------------------- workers


def _worker_header(
    args: argparse.Namespace, device: torch.device, flags: dict[str, Any], kind: str
) -> dict[str, Any]:
    return {
        "worker": kind,
        "setting": args.setting,
        "device": _device_info(device),
        "torch_flags": flags,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def worker_align(args: argparse.Namespace, shapes: Sequence[Shape]) -> dict[str, Any]:
    """All alignment scenarios of every shape, in this process."""
    device = torch.device(WORKER_DEVICE)
    flags = _apply_setting(build_settings()[args.setting])
    cache = InputCache(args.seed, device)
    scenarios = plan_alignment(args.offsets, args.b_offsets)
    base = alignment_base(args.offsets, args.b_offsets)
    store = BaselineStore(None)
    result = _worker_header(args, device, flags, "align")
    result["shapes"] = {}
    for shape in shapes:
        x_dev = cache.activations(shape.m, shape.k)
        w_dev = cache.weights(shape.k, shape.n)
        records = []
        for scenario in scenarios:
            x = place_with_offset(x_dev, scenario.a_offset, base)
            w = place_with_offset(w_dev, scenario.b_offset, base)
            record: dict[str, Any] = {
                "name": scenario.name,
                "group": "align",
                "a_offset_bytes": scenario.a_offset,
                "b_offset_bytes": scenario.b_offset,
                "a_data_ptr_mod_256": x.view.data_ptr() % 256,
                "b_data_ptr_mod_256": w.view.data_ptr() % 256,
            }
            record.update(
                run_case(
                    x.view,
                    w.view,
                    args,
                    device,
                    store,
                    shape.key,
                    scenario.name == ALIGN_BASELINE,
                )
            )
            records.append(record)
            del x, w
        result["shapes"][shape.key] = {
            "m": shape.m,
            "k": shape.k,
            "n": shape.n,
            "scenarios": records,
        }
        _empty_cache(device)
    return result


def worker_memory(args: argparse.Namespace, shapes: Sequence[Shape]) -> dict[str, Any]:
    """Free-memory scenarios, in this process.

    With ``--worker-scenario`` (fresh mode) exactly that scenario runs and the
    dummy is taken before the first cuBLAS call of the process; without it every
    scenario runs in turn (in-process mode).
    """
    device = torch.device(WORKER_DEVICE)
    flags = _apply_setting(build_settings()[args.setting])
    cache = InputCache(args.seed, device)
    # The inputs are resident before any memory is taken away.
    for shape in shapes:
        cache.activations(shape.m, shape.k)
        cache.weights(shape.k, shape.n)
    targets = memory_targets(args.free_mib)
    if args.worker_scenario is not None:
        if args.worker_scenario not in targets:
            raise SystemExit(
                f"unknown scenario {args.worker_scenario!r}; known: {list(targets)}"
            )
        targets = {args.worker_scenario: targets[args.worker_scenario]}
    base = alignment_base(args.offsets, args.b_offsets)
    store = BaselineStore(args.worker_baseline_dir)
    fresh = args.worker_scenario is not None
    result = _worker_header(args, device, flags, "memory")
    result["mode"] = "fresh" if fresh else "inprocess"
    result["shapes"] = {
        s.key: {"m": s.m, "k": s.k, "n": s.n, "scenarios": []} for s in shapes
    }
    for target in targets.values():
        plan, dummy, free_before = _take_free_memory(device, target)
        for shape in shapes:
            record: dict[str, Any] = {
                "name": plan.name,
                "group": f"memory_{result['mode']}",
                "target_free_mib": None if target is None else target // MIB,
                "dummy_mib": plan.dummy_bytes // MIB,
                "free_mib_before_gemm": free_before // MIB,
                "a_offset_bytes": 0,
                "b_offset_bytes": 0,
            }
            if not plan.reachable:
                record.update(status="skipped", reason=plan.reason)
            else:
                x = place_with_offset(cache.activations(shape.m, shape.k), 0, base)
                w = place_with_offset(cache.weights(shape.k, shape.n), 0, base)
                record["a_data_ptr_mod_256"] = x.view.data_ptr() % 256
                record["b_data_ptr_mod_256"] = w.view.data_ptr() % 256
                record.update(
                    run_case(
                        x.view,
                        w.view,
                        args,
                        device,
                        store,
                        shape.key,
                        plan.name == MEMORY_BASELINE,
                    )
                )
                del x, w
                # Hand the blocks of this shape back so the next one sees the target.
                _empty_cache(device)
            result["shapes"][shape.key]["scenarios"].append(record)
        del dummy
        _empty_cache(device)
    return result


# -------------------------------------------------------------------- verdict


def tri_any(values: Iterable[bool | None]) -> bool | None:
    """True if any is True; None if any is unknown or there are none; else False."""
    items = list(values)
    if any(v is True for v in items):
        return True
    if not items or any(v is None for v in items):
        return None
    return False


def tri_all(values: Iterable[bool | None]) -> bool | None:
    items = list(values)
    if any(v is False for v in items):
        return False
    if not items or any(v is None for v in items):
        return None
    return True


def _yes_no(flag: bool | None, unknown: str = "UNKNOWN") -> str:
    return unknown if flag is None else ("YES" if flag else "NO")


def collect_records(entry: dict[str, Any]) -> dict[str, dict[str, list[dict]]]:
    """``{shape_key: {group: [records]}}`` from one setting's worker results."""
    out: dict[str, dict[str, list[dict]]] = {}

    def add(group: str, result: dict[str, Any] | None) -> None:
        if not result:
            return
        for key, shape in result["shapes"].items():
            out.setdefault(key, {}).setdefault(group, []).extend(shape["scenarios"])

    add("align", entry.get("align"))
    add("memory_inprocess", entry.get("memory_inprocess"))
    for result in entry.get("memory_fresh", {}).values():
        add("memory_fresh", result)
    return out


def display_name(group: str, name: str) -> str:
    return f"fresh-{name}" if group == "memory_fresh" else name


def analyze_group(group: str, records: Sequence[dict]) -> bool | None:
    """Whether any scenario of ``group`` differs from its baseline; None if unknown."""
    ok = [r for r in records if r.get("status") == "ok"]
    if len(ok) < 2:
        return None
    hashes = {r["sha256"][0] for r in ok}
    base = next((r for r in ok if r["name"] == BASELINES[group]), None)
    if base is None:
        return True if len(hashes) > 1 else None
    return any(r["sha256"][0] != base["sha256"][0] for r in ok)


def analyze_shape(
    by_group: dict[str, list[dict]], groups: Sequence[str]
) -> dict[str, Any]:
    records = [r for g in GROUPS for r in by_group.get(g, [])]
    ok = [r for r in records if r.get("status") == "ok"]
    align = (
        analyze_group("align", by_group.get("align", [])) if "align" in groups else None
    )
    present = [g for g in MEMORY_GROUPS if g in by_group]
    memory = (
        tri_any(analyze_group(g, by_group[g]) for g in present)
        if "memory" in groups and present
        else None
    )
    kinds = []
    if "align" in groups:
        kinds.append(align)
    if "memory" in groups:
        kinds.append(memory)

    def worst(field: str) -> float | None:
        values = [r[field] for r in ok if r.get(field) is not None]
        return functools.reduce(_worse, values) if values else None

    def names(status: str) -> list[str]:
        return [
            display_name(r.get("group", ""), r["name"])
            for r in records
            if r.get("status") == status
        ]

    return {
        "scenarios_run": len(ok),
        "distinct_output_hashes": len({r["sha256"][0] for r in ok}),
        "align_dependent": align,
        "memory_dependent": memory,
        "dependent": tri_any(kinds),
        "repeats_identical": tri_all(r["repeats_identical"] for r in ok),
        "max_abs_diff_vs_baseline": worst("max_abs_diff_vs_baseline"),
        "max_abs_diff_vs_reference_fp32": worst("max_abs_diff_vs_reference_fp32"),
        "oom_scenarios": names("oom"),
        "error_scenarios": names("error"),
        "skipped_scenarios": names("skipped"),
    }


def analyze_setting(
    entry: dict[str, Any], groups: Sequence[str] = ("align", "memory")
) -> dict[str, Any]:
    """Cross-scenario summary of one setting's worker results."""
    shapes = {
        key: analyze_shape(by_group, groups)
        for key, by_group in collect_records(entry).items()
    }
    align = (
        tri_any(s["align_dependent"] for s in shapes.values())
        if "align" in groups
        else None
    )
    memory = (
        tri_any(s["memory_dependent"] for s in shapes.values())
        if "memory" in groups
        else None
    )
    kinds = []
    if "align" in groups:
        kinds.append(align)
    if "memory" in groups:
        kinds.append(memory)
    return {
        "shapes": shapes,
        "align_dependent": align,
        "memory_dependent": memory,
        "dependent": tri_any(kinds),
        "repeats_identical": tri_all(s["repeats_identical"] for s in shapes.values()),
        "dependent_shapes": [k for k, s in shapes.items() if s["dependent"] is True],
    }


def baseline_hashes(entry: dict[str, Any]) -> dict[str, list[str]]:
    """Every baseline output hash per shape key (A0, in-process and fresh whole)."""
    out: dict[str, list[str]] = {}
    for key, by_group in collect_records(entry).items():
        for group in GROUPS:
            for r in by_group.get(group, []):
                if r["name"] == BASELINES[group] and r.get("status") == "ok":
                    out.setdefault(key, []).append(r["sha256"][0])
    return out


def compute_verdict(
    results: dict[str, dict[str, Any]],
    groups: Sequence[str] = ("align", "memory"),
) -> dict[str, Any]:
    """Verdict lines from the per-setting worker results.

    ``results[setting]`` holds ``align`` (a worker result), ``memory_inprocess``
    (a worker result) and ``memory_fresh`` (worker results by scenario name). A
    setting "fixes" the dependence when the default setting is dependent and this
    one shows none; it is N/A when the default is not dependent.
    """
    summaries = {s: analyze_setting(entry, groups) for s, entry in results.items()}
    default = summaries.get("default")
    verdict: dict[str, Any] = {"settings": summaries}

    def kind(flag: str, group: str) -> str:
        if group not in groups:
            return "NOT_RUN"
        return "UNKNOWN" if default is None else _yes_no(default[flag])

    verdict["ALIGN_DEPENDENT"] = kind("align_dependent", "align")
    verdict["MEMORY_DEPENDENT"] = kind("memory_dependent", "memory")
    verdict["REPEATS_IDENTICAL"] = _yes_no(
        tri_all(s["repeats_identical"] for s in summaries.values())
    )

    def fixes(setting: str) -> str:
        if setting not in summaries:
            return "NOT_RUN"
        if default is None or default["dependent"] is None:
            return "UNKNOWN"
        if not default["dependent"]:
            return "N/A"
        dependent = summaries[setting]["dependent"]
        return "UNKNOWN" if dependent is None else _yes_no(not dependent)

    verdict["WSCAP_FIXES"] = {s: fixes(s) for s in ("wscap4096", "wscap16")}
    verdict["NO_REDUCED_PRECISION_FIXES"] = fixes("no_reduced_precision")
    verdict["DETERMINISTIC_FIXES"] = fixes("deterministic")
    verdict["NO_LT_FIXES"] = fixes("no_lt")

    hashes = {s: baseline_hashes(entry) for s, entry in results.items()}
    same: dict[str, str] = {}
    for setting, per_shape in hashes.items():
        if setting == "default" or "default" not in hashes:
            continue
        common = [k for k in per_shape if k in hashes["default"]]
        if not common:
            same[setting] = "UNKNOWN"
        else:
            equal = all(per_shape[k][0] == hashes["default"][k][0] for k in common)
            same[setting] = _yes_no(equal)
    verdict["SAME_BYTES_AS_DEFAULT"] = same

    cross: list[bool | None] = []
    for per in hashes.get("default", {}).values():
        if len(per) >= 2:
            cross.append(len(set(per)) == 1)
    verdict["CROSS_PROCESS_REPRODUCIBLE"] = _yes_no(tri_all(cross))
    verdict["DEPENDENT_SHAPES"] = [] if default is None else default["dependent_shapes"]
    return verdict


def algo_lines(
    results: dict[str, dict[str, Any]], print_all: bool = False
) -> list[str]:
    """ALGO lines: the baseline kernels of every (setting, shape) and any change."""
    lines: list[str] = []
    for setting, entry in results.items():
        for key, by_group in collect_records(entry).items():
            ordered = [
                (g, r)
                for g in GROUPS
                for r in by_group.get(g, [])
                if r.get("status") == "ok" and "kernels" in r
            ]
            if not ordered:
                continue
            base = next(
                (r for g, r in ordered if g == "align" and r["name"] == ALIGN_BASELINE),
                ordered[0][1],
            )
            same = 0
            for group, r in ordered:
                unchanged = r["kernels"] == base["kernels"]
                same += unchanged
                if print_all or r is base or not unchanged:
                    suffix = (
                        "" if unchanged or r is base else " (differs from baseline)"
                    )
                    lines.append(
                        f"ALGO: {setting} {key} {display_name(group, r['name'])}: "
                        f"{describe_kernels(r['kernels'])}{suffix}"
                    )
            lines.append(
                f"ALGO_SUMMARY: {setting} {key}: {same} of {len(ordered)} "
                "scenarios run the baseline kernels"
            )
    return lines


# ------------------------------------------------------------------ subprocess


def build_worker_command(
    script: str,
    worker: str,
    setting: str,
    shapes: Sequence[Shape],
    args: argparse.Namespace,
    scenario: str | None = None,
    baseline_dir: str | Path | None = None,
) -> list[str]:
    """Command line of one worker subprocess; the shapes are passed explicitly."""
    command = [
        sys.executable,
        script,
        "--worker",
        worker,
        "--setting",
        setting,
        "--shapes",
        *[f"{s.m},{s.k},{s.n}" for s in shapes],
        "--seed",
        str(args.seed),
        "--repeats",
        str(args.repeats),
        "--offsets",
        ",".join(map(str, args.offsets)),
        "--b-offsets",
        ",".join(map(str, args.b_offsets)),
        "--free-mib",
        *[str(m) for m in args.free_mib],
    ]
    if args.no_profile:
        command.append("--no-profile")
    if args.cpu_reference:
        command.append("--cpu-reference")
    if scenario is not None:
        command += ["--worker-scenario", scenario]
    if baseline_dir is not None:
        command += ["--worker-baseline-dir", str(baseline_dir)]
    return command


def build_worker_env(setting_env: dict[str, str]) -> dict[str, str]:
    """Environment of a worker: the caller's, minus the managed variables, plus
    the setting's own."""
    env = dict(os.environ)
    for name in MANAGED_ENV:
        env.pop(name, None)
    env.update(setting_env)
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
            f"worker exited {completed.returncode}: {' '.join(command[2:6])}\n"
            + "\n".join(tail)
        )
    return extract_result(completed.stdout)


# ---------------------------------------------------------------------- driver


def _int_list(tokens: Sequence[str], flag: str, parser: argparse.ArgumentParser):
    """Integers from tokens that may be comma separated (``0,16 32``)."""
    try:
        return [int(t) for tok in tokens for t in tok.replace(",", " ").split()]
    except ValueError:
        parser.error(f"{flag} takes integers, got {' '.join(tokens)!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument(
        "--m", type=int, nargs="+", help=f"token counts, default {list(DEFAULT_M)}"
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        help="replace the projection list by K=hidden_size and these N values",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape_triple,
        nargs="+",
        metavar="M,K,N",
        help="explicit shapes; replaces the config-derived list",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--offsets",
        nargs="+",
        default=[",".join(map(str, DEFAULT_OFFSETS))],
        help="byte offsets of A inside its buffer (comma or space separated)",
    )
    parser.add_argument(
        "--b-offsets",
        nargs="+",
        default=[",".join(map(str, DEFAULT_B_OFFSETS))],
        help="byte offsets of B (the weight) for the extra B scenarios",
    )
    parser.add_argument(
        "--free-mib",
        type=int,
        nargs="+",
        default=list(DEFAULT_FREE_MIB),
        help="free memory left at the GEMM, in MiB, besides the whole-memory run",
    )
    parser.add_argument(
        "--settings",
        default="all",
        help="comma list of default,wscap4096,wscap16,no_reduced_precision,"
        "deterministic,no_lt, or all",
    )
    parser.add_argument(
        "--groups",
        default="align,memory",
        help="scenario groups to run: align, memory or both",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("inprocess", "fresh", "both"),
        default="both",
        help="inprocess: one worker runs the free-memory scenarios in turn; fresh: "
        "one worker per scenario, the dummy taken before the first cuBLAS call",
    )
    parser.add_argument(
        "--cpu-reference",
        action="store_true",
        help="also compare the baseline output with a CPU fp32 reference",
    )
    parser.add_argument(
        "--no-profile", action="store_true", help="skip the torch.profiler kernel names"
    )
    parser.add_argument(
        "--print-all-algo",
        action="store_true",
        help="print an ALGO line for every scenario, not only the changed ones",
    )
    parser.add_argument(
        "--workdir", help="where fresh-mode workers keep their baseline outputs"
    )
    parser.add_argument("--out", help="write the JSON report here")
    # Internal: one worker subprocess.
    parser.add_argument("--worker", choices=("align", "memory"), help=argparse.SUPPRESS)
    parser.add_argument("--setting", help=argparse.SUPPRESS)
    parser.add_argument("--worker-scenario", help=argparse.SUPPRESS)
    parser.add_argument("--worker-baseline-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    args.offsets = _int_list(args.offsets, "--offsets", parser)
    args.b_offsets = _int_list(args.b_offsets, "--b-offsets", parser)
    esize = torch.empty(0, dtype=DTYPE).element_size()
    for flag, offsets in (("--offsets", args.offsets), ("--b-offsets", args.b_offsets)):
        if any(o < 0 or o % esize for o in offsets):
            parser.error(f"{flag} must be non-negative multiples of {esize} bytes")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.tp < 1:
        parser.error("--tp must be at least 1")
    if args.shapes and (args.m or args.n):
        parser.error("--shapes replaces --m and --n; give one or the other")
    if args.m is not None and any(m < 1 for m in args.m):
        parser.error("--m values must be positive")
    if args.n is not None and any(n < 1 for n in args.n):
        parser.error("--n values must be positive")
    if any(m < 1 for m in args.free_mib):
        parser.error("--free-mib values must be positive")
    if args.m is None:
        args.m = list(DEFAULT_M)
    if args.worker is not None and not args.shapes:
        parser.error("a worker needs --shapes")
    if args.worker is not None and args.setting not in build_settings():
        parser.error(f"a worker needs a known --setting, got {args.setting!r}")
    return args


def _select_names(spec: str, available: Sequence[str]) -> list[str]:
    if spec == "all":
        return list(available)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown name(s) {unknown}; available: {list(available)}")
    return names


def run_driver(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print(NO_CUDA_MESSAGE, file=sys.stderr)
        return 2
    try:
        shapes, source = resolve_shapes(args)
    except ValueError as exc:
        print(f"cannot derive the shapes: {exc}", file=sys.stderr)
        return 2
    settings = build_settings()
    chosen = _select_names(args.settings, list(settings))
    groups = _select_names(args.groups, ("align", "memory"))
    script = os.path.abspath(__file__)
    # The driver never opens CUDA: its context would take memory from the GPU the
    # workers measure. The device is reported from the first worker's result.
    print(
        f"torch {torch.__version__}, CUDA {torch.version.cuda}; {len(shapes)} shapes "
        f"({source}); settings {chosen}; groups {groups}; memory mode "
        f"{args.memory_mode}"
    )
    if torch.cuda.device_count() > 1:
        print(
            "NOTE: several GPUs are visible and the workers use the first; set "
            "CUDA_VISIBLE_DEVICES to one idle GPU"
        )
    scenario_names = list(memory_targets(args.free_mib))
    device_info: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    def launch(label: str, command: list[str], env: dict[str, str]):
        nonlocal device_info
        print(f"worker: {label}", flush=True)
        try:
            result = run_worker(command, env)
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            errors.append(f"{label}: {exc}")
            print(f"  FAILED: {exc}")
            return None
        if device_info is None:
            device_info = result.get("device")
            print(f"device {device_info}")
        return result

    for name in chosen:
        env = build_worker_env(settings[name]["env"])
        entry: dict[str, Any] = {
            "align": None,
            "memory_inprocess": None,
            "memory_fresh": {},
        }
        results[name] = entry
        if "align" in groups:
            command = build_worker_command(script, "align", name, shapes, args)
            entry["align"] = launch(f"{name} align", command, env)
        if "memory" in groups and args.memory_mode in ("inprocess", "both"):
            command = build_worker_command(script, "memory", name, shapes, args)
            entry["memory_inprocess"] = launch(f"{name} memory inprocess", command, env)
        if "memory" in groups and args.memory_mode in ("fresh", "both"):
            with tempfile.TemporaryDirectory(
                prefix="gemm_baseline_", dir=args.workdir
            ) as baseline_dir:
                for scenario in scenario_names:
                    command = build_worker_command(
                        script,
                        "memory",
                        name,
                        shapes,
                        args,
                        scenario=scenario,
                        baseline_dir=baseline_dir,
                    )
                    result = launch(f"{name} memory fresh {scenario}", command, env)
                    if result is not None:
                        entry["memory_fresh"][scenario] = result

    verdict = compute_verdict(results, groups)
    report = {
        "device": device_info,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shapes": [
            {"key": s.key, "label": s.label, "m": s.m, "k": s.k, "n": s.n}
            for s in shapes
        ],
        "shape_source": source,
        "dtype": str(DTYPE),
        "seed": args.seed,
        "repeats": args.repeats,
        "offsets": args.offsets,
        "b_offsets": args.b_offsets,
        "free_mib": args.free_mib,
        "memory_mode": args.memory_mode,
        "groups": groups,
        "settings": {k: v for k, v in settings.items() if k in chosen},
        "results": results,
        "errors": errors,
        "notes": NOTES,
        "verdict": verdict,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print_verdict(verdict, shapes, results, args.print_all_algo, errors)
    return 1 if errors else 0


def print_verdict(
    verdict: dict[str, Any],
    shapes: Sequence[Shape],
    results: dict[str, dict[str, Any]],
    print_all_algo: bool,
    errors: list[str],
) -> None:
    print()
    for key in ("ALIGN_DEPENDENT", "MEMORY_DEPENDENT", "REPEATS_IDENTICAL"):
        print(f"{key}: {verdict[key]}")
    for setting, value in verdict["WSCAP_FIXES"].items():
        print(f"WSCAP_FIXES ({setting}): {value}")
    for key in ("NO_REDUCED_PRECISION_FIXES", "DETERMINISTIC_FIXES", "NO_LT_FIXES"):
        print(f"{key}: {verdict[key]}")
    for setting, value in verdict["SAME_BYTES_AS_DEFAULT"].items():
        print(f"SAME_BYTES_AS_DEFAULT ({setting}): {value}")
    print(f"CROSS_PROCESS_REPRODUCIBLE: {verdict['CROSS_PROCESS_REPRODUCIBLE']}")
    labels = {s.key: s.label for s in shapes}
    dependent = ", ".join(
        f"{k} ({labels.get(k, '')})" for k in verdict["DEPENDENT_SHAPES"]
    )
    print(f"DEPENDENT_SHAPES: {dependent or 'none'}")
    for setting, summary in verdict["settings"].items():
        for key, s in summary["shapes"].items():
            print(
                f"  {setting} {key} ({labels.get(key, '')}): {s['scenarios_run']} "
                f"scenarios, {s['distinct_output_hashes']} distinct output hash(es), "
                f"align {_yes_no(s['align_dependent'], 'n/a')}, "
                f"memory {_yes_no(s['memory_dependent'], 'n/a')}, "
                f"max|diff| vs baseline {s['max_abs_diff_vs_baseline']}, "
                f"vs fp32 reference {s['max_abs_diff_vs_reference_fp32']}, "
                f"repeats identical: {s['repeats_identical']}"
                + (f", oom: {s['oom_scenarios']}" if s["oom_scenarios"] else "")
                + (f", errors: {s['error_scenarios']}" if s["error_scenarios"] else "")
                + (
                    f", skipped: {s['skipped_scenarios']}"
                    if s["skipped_scenarios"]
                    else ""
                )
            )
    for line in algo_lines(results, print_all_algo):
        print(line)
    print(f"NOTE: {NOTES[0]}")
    for error in errors:
        print(f"ERROR: {error}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker is None:
        return run_driver(args)
    # A worker: print one JSON line and exit.
    shapes = [Shape(m, k, n) for m, k, n in args.shapes]
    func = worker_align if args.worker == "align" else worker_memory
    with torch.inference_mode():
        result = func(args, shapes)
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
