# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do the SM70 NVFP4 KV kernels store, gather and decode what the reference says,
and what do they cost next to the E4M3 kernels?

Question under test (plan 071). The CPU tests prove the Triton store, the Triton
gather and the NVFP4 decode route equal their torch references in the Triton
interpreter. They prove nothing about the compiled kernels on a V100: a different
division, a different fused multiply, a launch configuration the interpreter never
sees. This script runs the same chain on one GPU on real-geometry random K/V and
says which links hold, how large the NVFP4 attention error is next to the E4M3
error it is meant to replace, and how long each kernel takes.

The chain, per case (one NVFP4 or E4M3 block size and one row count):

1. Store. Random K and V ``[tokens, kv_heads, head_dim]`` (seeded on the CPU, so
   every run builds the same bytes) are written with ``store_nvfp4_kv_triton`` into
   a cache on the GPU and with ``reshape_and_cache_nvfp4_reference`` into a cache on
   the CPU; the two caches are compared byte for byte, separately for the data
   bytes and the scale bytes. For the E4M3 control the same K/V go through
   ``reshape_and_cache_flash`` with per-tensor scales ``amax / 448``.
2. Gather. ``gather_dequant_nvfp4_kv_triton`` on the reference cache bytes (so a
   store mismatch cannot hide here) against ``gather_dequant_nvfp4_kv`` on the CPU,
   with illegal top-k entries in the selection: negative, past the block table, on
   an unmapped page, on a physical block past the cache, and one row whose request
   does not exist. Every illegal entry must come out as exact zeros.
3. Decode. ``qsa_sparse_paged_attention(kv_cache_dtype="nvfp4")``, the production
   route: gather and decode the selected rows to FP16, then the FP16 split-K kernel
   with the layer scales folded in FP32. Three comparisons:

   - path identity: at unit layer scales the NVFP4 route against the same FP16
     kernel on a plain FP16 paged cache of the dequantized values. The two run the
     same arithmetic on the same operands, so the outputs must be equal bit for bit;
   - quantization error: at the realistic layer scales against the FP16 kernel on
     the original, unquantized K/V (max |diff|, mean |diff| and relative L2);
   - E4M3 control: the same harness on the E4M3 cache against the same FP16
     baseline, for the error NVFP4 has to be read against.

4. Timings, CUDA events, median of ``--iters`` calls per round, ABBA rounds across
   the arms so a drift in clocks or neighbours moves every arm: store (NVFP4 and
   E4M3), gather, the whole NVFP4 decode, the FP16 kernel alone on the dequantized
   cache, and the E4M3 decode, for each row count (default 1 and 5, the decode
   steps of a plain and an MTP4 target).

5. The fused reader (``--arms gather fused``, the default; ``--arms gather`` skips it).
   ``qsa_sparse_paged_attention(kv_cache_dtype="nvfp4", nvfp4_fused_reader=True)``
   decodes the packed pages inside the split-K kernel (``KV_NVFP4``) instead of
   gathering them to FP16 first. Its QK is two K = 128 dots, so it is NOT bit-equal to
   the gather route: ``FUSED_MATCHES_GATHER`` accepts a relative L2 of at most
   ``FUSED_REL_L2_TOLERANCE`` (FP16 output rounding level) against the gather route,
   on the clean selection at the realistic layer scales and on the selection with
   illegal entries at unit scales (illegal rows and empty rows must be exact zeros).
   Lines: ``FUSED_VS_GATHER <case>`` (max, mean, relL2), ``FUSED_VS_FP16 <case>`` (the
   quantization error of the fused route against the unquantized K/V, next to
   ``DECODE_VS_FP16``) and ``FUSED_ROUNDTRIP_US`` (per-launch time of the fused
   decode over the gather decode and the E4M3 decode). The two mutated caches must also
   be FLAGGED through the fused route (``<mutation> fused_decode``). The timed arm is
   ``decode_fused_nvfp4_B<block>``.

6. The prefill scratch route (``--arms prefill_scratch``, plan 071 option B'; not in the
   default arms because it needs a prefill-sized ``--rows``, 64 or more for the CUDA
   part). One prefill chunk: ``rows`` consecutive positions at the end of one
   ``--context``-token request, selected the way the grouped planner reads them (4-token
   groups plus a tail). ``prefill_scratch_decode`` decodes the whole prefix once into
   an FP16 scratch (``dequant_nvfp4_prefix_triton``) and ``SCRATCH_MATCHES_GATHER``
   requires it to equal ``gather_dequant_nvfp4_kv`` at unit layer scale bit for bit on
   every token, and to be exact +0.0 past the sequence length; the two mutated caches pushed
   through the same decode must differ (``<mutation> scratch_decode``). On a GPU
   the route itself (scratch decode + FP16 grouped page4 + V scale) runs on the chunk:
   ``SCRATCH_VS_FUSED`` (the whole chunk) and ``SCRATCH_VS_GATHER`` (the first 64 rows,
   the gather route is slow) must be within ``FUSED_REL_L2_TOLERANCE``
   (``SCRATCH_ROUTE_WITHIN_TOLERANCE``; UNKNOWN, and not required, off a GPU). Timed
   arms: ``prefill_scratch_decode_nvfp4_B<block>`` (table + decode kernel alone),
   ``prefill_scratch_route_nvfp4_B<block>`` (the whole route, GPU), ``prefill_fused_
   nvfp4_B<block>`` (the Triton fused reader on the same chunk) and
   ``prefill_e4m3_B<block>`` (the E4M3 grouped route on the same chunk shape, GPU);
   ``PREFILL_ROUNDTRIP_US`` gives scratch route and fused over E4M3. NOT established by
   any CPU run: every one of those times.

Geometry. Read from ``--config`` for ``--tp`` ranks, as the engine builds it: head
size, query heads per rank, KV heads per rank (replicated below the TP size),
selection width ``indexer_budget + indexer_compress_ratio - 1``. The production
values at TP4 are head 256, 6 query heads, 1 KV head, top-k 2051. Block sizes are
the ones the allocator derives (``docs/design/c4140_qsa_nvfp4_kv.md``): 2784 and
2864 for NVFP4 without and with MTP4, 1616 for E4M3 with MTP4 (the control).

Host memory and reference rows. The GPU arms (store, gather, zero-fill, decode, fused,
scratch route, every timing) always run on every row of the batch. The CPU side does
not scale that way: the reference gather costs ~19 KiB of host RAM per row (an int64
index and an FP32 lookup per nibble, ~18.6 bytes per output element), so a whole-batch
reference at 5568 rows needed ~51 GiB (W1b of 2026-10-05, OOM-killed in a 48G fence),
and the CPU attention model costs ~70 ms per row (~7 min per case at 5568 rows).
``--ref-rows`` (default 256) bounds both: up to that many rows every row is checked
(the 5- and 64-row runs are unchanged, bit for bit); above it a deterministic subset is
(rows 0, 1, the last two, evenly spaced between) and the comparison reads those rows of
the GPU's full-batch output, ``--ref-chunk`` (64) rows at a time. ``ZERO_FILL_OK`` still
looks at every row. The verdict says how many rows the reference saw
(``REFERENCE_ROWS <case>: ... on <k> of <M> rows``), and every case's JSON entry carries
``ref_rows``. Before each case the run prints ``HOST_MEM_EST: rows=<M> est_peak_GB=<x>``
(this process's RSS plus a per-tensor model, ``host_mem_model``, validated against
measured peak RSS), records it in the JSON (``host_mem_est``, plus the measured
``VmHWM`` after the case) and skips the case with ``SKIPPED`` (a ``CASE_SKIPPED`` verdict
line, ``SELF_CHECK: FAIL``) instead of dying if the estimate exceeds ``--max-host-gb``
(24).

Route knobs. Every arm names its own route, so the run pins
``VLLM_SM70_QSA_NVFP4_FUSED_READER=0``, ``VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH=0`` and
``VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS=64`` for its whole duration (the fused arm passes
``nvfp4_fused_reader=True``, the scratch arm calls the route directly) and restores the
environment after. An exported serving environment therefore cannot change what the
controls and the timed ``decode_nvfp4`` / ``gather`` arms measure: with
``FUSED_READER=1`` exported they ran the fused reader (``ZERO_FILL_OK`` and
``DECODE_PATH_IDENTICAL`` NO; W1 of plan 071, 2026-10-05). A run that overrode an
ambient value prints ``ROUTE_KNOBS ambient ...``; the JSON carries ``route_knobs``.

Verdict printed on stdout:

    STORE_MATCHES_REFERENCE: YES/NO          data and scale bytes, every NVFP4 case
    GATHER_MATCHES_REFERENCE: YES/NO         FP16 outputs equal, every NVFP4 case
    ZERO_FILL_OK: YES/NO                     illegal entries are exact zeros
    DECODE_PATH_IDENTICAL: YES/NO            unit scales: NVFP4 route == FP16 kernel
    FUSED_MATCHES_GATHER: YES/NO/UNKNOWN     fused reader within tolerance of gather
    DECODE_VS_FP16 <case>: max|d| .. mean|d| .. relL2 ..
    E4M3_CONTROL <case>: max|d| .. mean|d| .. relL2 ..
    NVFP4_VS_E4M3_REL_L2 M=<rows>: <ratio>   NVFP4 error over E4M3 error
    TIMING_US <case> <kernel>: median .. p10 .. p90 .. (round spread ..%)
    DECODE_ROUNDTRIP_US M=<rows>: nvfp4 .. e4m3 .. ratio ..

``NO`` lines carry the evidence (the count and first position of differing bytes,
the largest difference). ``UNKNOWN`` means a case failed before it could say.

Self-check. One command line is enough to know whether the run can be trusted. Every
run also validates itself and ends with a verdict:

    POSITIVE_CONTROL <name>: PASS/FAIL/SKIPPED
    NEGATIVE_CONTROL <mutation> <link>: FLAGGED/MISSED/SKIPPED
    STAGE <name>: RAN/SKIPPED
    SELF_CHECK: PASS
    SELF_CHECK: FAIL (<what failed>)

Positive controls must PASS: ``STORE_MATCHES_REFERENCE``, ``GATHER_MATCHES_REFERENCE``,
``ZERO_FILL_OK`` and ``DECODE_PATH_IDENTICAL`` (the four verdicts above, bit-exact),
``TIE_STORE_MATCHES_REFERENCE`` (the GPU store on engineered round-to-nearest-even
ties, E2M1 midpoints and E4M3 scale midpoints, equals the reference), and
``NVFP4_ERROR_IN_BAND`` / ``E4M3_ERROR_IN_BAND``: the relative L2 error of the
attention output measured through the harness must lie between 0.7 and 1.4 times the
value a CPU model predicts for the same data (FP32 attention on K/V quantized by the
reference quantizer, or by the float8_e4m3fn cast that ``reshape_and_cache_flash``
performs). The E4M3 arm is the harness's known-answer check: if it does not reproduce
its band the comparison itself cannot be trusted.

Negative controls must be FLAGGED: three mutated arms are compared with the same
comparisons that judge the real arm, and each comparison must report a difference.
``nibble_order_swapped`` (the two nibbles of every data byte exchanged) and
``scale_placement_wrong`` (the scale bytes permuted as SM100's V swizzle does, which is
not the V100 layout) are checked at the store, gather and decode links;
``tie_rule_removed_e2m1`` and ``tie_rule_removed_e4m3`` (round half up instead of half
to even, in the nibble or in the scale byte) are checked on the tie store. A mutation
that is MISSED means the data or the comparison cannot see that class of error.
A MISSED can also follow a failed positive control: a store that is wrong in exactly
the mutated way makes the mutated reference equal to it.

Stages store, gather, zero_fill, decode, e4m3_control, tie_control, negative_controls
and timing must all be RAN. A skipped stage is a failed self-check, so ``--no-timing``
and ``--no-e4m3`` make ``SELF_CHECK`` FAIL by design; a case that raised does too.

Exit status: 0 a clean run with ``SELF_CHECK: PASS``; 1 ``SELF_CHECK: FAIL`` (a failed
control, a skipped stage or a case that raised); 2 no CUDA device, a GPU that already
holds memory (unless ``--allow-busy``), unusable arguments, or a ``vllm`` on
``sys.path`` that lacks the NVFP4 modules (the p071 tree, or a venv synced to it).

    cd <worktree> && CUDA_VISIBLE_DEVICES=<idle gpu> PYTHONPATH=$PWD \\
        /data/venvs/1cat-p070/bin/python benchmarks/sm70_nvfp4_kv_kernel_check.py \\
        --out /data/bench/nvfp4_kv_kernel_check.json

Use one idle GPU and never one a server holds: the script allocates a few hundred
MiB, and any other process on the device moves the timings. Without ``--allow-busy``
it refuses a GPU that already holds more than ``--max-used-mib`` (1024). The first
call of every Triton kernel compiles it, so the run starts with warmups; with the
defaults it takes a few minutes, most of it compilation. The printed ``SOURCE``
lines name the tree the kernels were imported from.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_CONFIG = "/data/models/Qwen3.8-Flash-Next-NVFP4/config.json"
# Production values at TP4, used when the config cannot be read.
PRODUCTION_GEOMETRY = {"head_dim": 256, "q_heads": 6, "kv_heads": 1, "topk": 2051}
DEFAULT_NVFP4_BLOCKS = (2784, 2864)
DEFAULT_E4M3_BLOCK = 1616
DEFAULT_ROWS = (1, 5)
# Overlay-like layer scales (the Flash-Next overlay holds K 0.0186..0.0367 and
# V 0.0171..0.0841); powers of two would hide a scale folded on the wrong side.
DEFAULT_K_SCALE = 0.0213623046875
DEFAULT_V_SCALE = 0.0404924675822258
# Largest |value| of the random K/V: with these scales the block scales stay below
# the E4M3 maximum (amax / 6 / layer_scale <= 448 needs amax <= 53.7).
MAX_ABS_VALUE = 40.0
E4M3_MAX = 448.0
# The fused reader sums QK as two K = 128 dots, so against the gather route it is
# equal only up to FP32 summation order and the FP16 rounding of the output: relative
# L2 of about 3e-4 is the rounding level (2**-11 per element); 2e-3 leaves a margin
# and still fails a wrong nibble, scale or layer-scale placement (those are O(0.1..1)).
FUSED_REL_L2_TOLERANCE = 2e-3
DEFAULT_ARMS = ("gather", "fused")
ARMS = ("gather", "fused", "prefill_scratch")
# The CUDA part of the scratch route is a prefill route: the grouped page4 planner
# works on groups of 8 rows and the knob's default minimum is 64.
PREFILL_ROUTE_MIN_ROWS = 64
# The gather route (31 rows per launch group) is compared on this many rows only.
PREFILL_VS_GATHER_ROWS = 64
# Host memory. The GPU arms always run at the full row count; what is bounded is the
# CPU side. The float64/FP16 CPU reference (``gather_dequant_nvfp4_kv``) costs about
# 18.6 bytes per output element of host RAM (an int64 index and a float32 grid
# lookup per nibble), i.e. ~19 KiB per row at topk 2051 x head_dim 256 for the K side
# alone: M = 5568 rows needed ~54 GiB (W1b of 2026-10-05, OOM-killed in a 48G fence).
# So the CPU references and the CPU attention model run on ``--ref-rows`` rows of the
# batch (all of them when the batch is that small), ``REF_CHUNK_ROWS`` rows at a time.
DEFAULT_REF_ROWS = 256
REF_CHUNK_ROWS = 64
DEFAULT_MAX_HOST_GB = 24.0
GIB = float(1 << 30)
# Bytes of the CPU reference gather per output element ``E = rows * topk * heads * d``,
# per ``nvfp4_kv.gather_dequant_nvfp4_kv`` (see ``ref_gather_peak_bytes``).
REF_GATHER_BYTES_PER_ELEMENT = 18.5625
RESULT_KEYS = ("store", "gather", "zero_fill", "decode_path", "decode_quality")
# The measured attention error must lie within this factor of the CPU model's.
BAND_LOW, BAND_HIGH = 0.7, 1.4
# Midpoints of the E2M1 grid (0, .5, 1, 1.5, 2, 3, 4, 6) and 6 x midpoints of E4M3
# neighbours (1.0625, 1.3125, 1.5625, 2.125, 0.53125): fp16-exact values that sit
# exactly on a rounding tie after the store's scaling at unit layer scale.
E2M1_TIE_VALUES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_TIE_AMAX = (6.375, 7.875, 9.375, 12.75, 3.1875)
TIE_TOKENS = 32
MUTATIONS = ("nibble_order_swapped", "scale_placement_wrong")
TIE_MUTATIONS = ("tie_rule_removed_e2m1", "tie_rule_removed_e4m3")
LINKS = ("store", "gather", "decode")
POSITIVE_CONTROLS = (
    "STORE_MATCHES_REFERENCE",
    "GATHER_MATCHES_REFERENCE",
    "ZERO_FILL_OK",
    "DECODE_PATH_IDENTICAL",
    "TIE_STORE_MATCHES_REFERENCE",
    "NVFP4_ERROR_IN_BAND",
    "E4M3_ERROR_IN_BAND",
)
STAGES = (
    "store",
    "gather",
    "zero_fill",
    "decode",
    "e4m3_control",
    "tie_control",
    "negative_controls",
    "timing",
)
# The device the cases run on; the CPU tests point it at "cpu" (Triton interpreter).
WORKER_DEVICE = "cuda:0"
# The route-selecting knobs of ``qsa_sparse_paged_attention``. A call that does not pass
# ``nvfp4_fused_reader`` follows ``VLLM_SM70_QSA_NVFP4_FUSED_READER``, and a chunk of
# enough rows with query positions takes the scratch route when the scratch knob is on.
# The harness names the route of every arm itself (the fused arm passes
# ``nvfp4_fused_reader=True``, the scratch route is called directly), so a run pins the
# knobs for its whole duration: an exported serving environment must not change what
# "decode", "gather" or "prefill" mean. Without the pin, FUSED_READER=1 made the
# gather-route controls run the fused reader (ZERO_FILL_OK and DECODE_PATH_IDENTICAL
# NO, and the timed ``decode_nvfp4`` arm equal to ``decode_fused_nvfp4``).
ROUTE_KNOBS = {
    "VLLM_SM70_QSA_NVFP4_FUSED_READER": "0",
    "VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH": "0",
    "VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS": str(PREFILL_ROUTE_MIN_ROWS),
}


def ambient_route_knobs() -> dict[str, str | None]:
    """The route knobs as the process environment has them (None: not set)."""
    return {name: os.environ.get(name) for name in ROUTE_KNOBS}


@contextlib.contextmanager
def pinned_route_knobs():
    """Run with every route knob at the harness's value; restore the environment after.

    ``vllm.envs`` reads ``os.environ`` at each access (the engine's cache is not
    enabled here), so the pin takes effect on the next call.
    """
    saved = ambient_route_knobs()
    os.environ.update(ROUTE_KNOBS)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ------------------------------------------------------------------ geometry


@dataclass(frozen=True)
class Geometry:
    head_dim: int
    q_heads: int
    kv_heads: int
    topk: int
    source: str

    @property
    def group(self) -> int:
        return self.q_heads // self.kv_heads


def read_geometry_from_config(path: str | Path, tp: int) -> dict[str, int] | None:
    """Per-rank attention geometry from a model config, or None if unreadable."""
    try:
        config = json.loads(Path(path).read_text())
        text = config.get("text_config", config)
        heads = int(text["num_attention_heads"])
        kv_heads = int(text["num_key_value_heads"])
        head_dim = int(text["head_dim"])
        topk = int(text["indexer_budget"]) + int(text["indexer_compress_ratio"]) - 1
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if tp <= 0 or heads % tp or head_dim <= 0 or head_dim % 16 or topk <= 0:
        return None
    return {
        "head_dim": head_dim,
        "q_heads": heads // tp,
        "kv_heads": max(1, kv_heads // tp),
        "topk": topk,
    }


def resolve_geometry(args: argparse.Namespace) -> Geometry:
    found = read_geometry_from_config(args.config, args.tp)
    values = dict(found) if found else dict(PRODUCTION_GEOMETRY)
    source = f"config {args.config} tp={args.tp}" if found else "production defaults"
    for key in ("head_dim", "q_heads", "kv_heads", "topk"):
        override = getattr(args, key)
        if override is not None:
            values[key] = override
            source += f" (+{key} override)"
    if values["q_heads"] % values["kv_heads"]:
        raise ValueError("q_heads must be a multiple of kv_heads")
    return Geometry(source=source, **values)


# ------------------------------------------------------------------ planning


@dataclass(frozen=True)
class Case:
    kind: str  # "nvfp4" or "e4m3"
    block_size: int
    rows: int

    @property
    def label(self) -> str:
        return f"{self.kind}-B{self.block_size}-M{self.rows}"


def plan_cases(
    nvfp4_blocks: Sequence[int],
    e4m3_block: int | None,
    rows: Sequence[int],
) -> list[Case]:
    """Every NVFP4 block size and, if given, the E4M3 control, per row count."""
    if not rows or any(r <= 0 for r in rows):
        raise ValueError("row counts must be positive")
    if any(b <= 0 or b % 4 for b in nvfp4_blocks) or (
        e4m3_block is not None and (e4m3_block <= 0 or e4m3_block % 4)
    ):
        raise ValueError("block sizes must be positive multiples of 4")
    cases = [Case("nvfp4", b, r) for r in rows for b in nvfp4_blocks]
    if e4m3_block is not None:
        cases += [Case("e4m3", e4m3_block, r) for r in rows]
    if not cases:
        raise ValueError("nothing to run")
    return cases


@dataclass(frozen=True)
class Layout:
    block_size: int
    pages: int  # logical pages per request
    requests: int
    num_blocks: int  # physical blocks of the cache, two of them never mapped
    tokens_per_request: int


def plan_layout(block_size: int, rows: int, topk: int, context: int) -> Layout:
    """Cache geometry for a batch of ``rows`` rows over one or two requests."""
    tokens = max(context, topk + 1)
    pages = -(-tokens // block_size)
    requests = 1 if rows == 1 else 2
    return Layout(
        block_size=block_size,
        pages=pages,
        requests=requests,
        num_blocks=requests * pages + 2,
        tokens_per_request=tokens,
    )


def make_block_table(layout: Layout, seed: int) -> torch.Tensor:
    """Distinct physical blocks per logical page, int32 ``[requests, pages]``."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(layout.num_blocks, generator=generator)
    mapped = order[: layout.requests * layout.pages]
    return mapped.reshape(layout.requests, layout.pages).to(torch.int32)


def slot_mapping(table: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Slot of every token of every request, request-major, int64."""
    tokens = torch.arange(layout.tokens_per_request)
    pages = (tokens // layout.block_size).long()
    offsets = tokens % layout.block_size
    slots = table.long()[:, pages] * layout.block_size + offsets
    return slots.reshape(-1)


def make_kv(
    tokens: int, kv_heads: int, head_dim: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random K and V with the spread of real activations, fp16 ``[t, h, d]``.

    A normal draw times a per-token log-uniform magnitude (three decades) with a
    few heavy-tailed outliers, clamped to ``MAX_ABS_VALUE``.
    """
    generator = torch.Generator().manual_seed(seed)

    def one() -> torch.Tensor:
        base = torch.randn(tokens, kv_heads, head_dim, generator=generator)
        magnitude = 10.0 ** (
            torch.rand(tokens, kv_heads, 1, generator=generator) * 2 - 1
        )
        outliers = torch.randn(tokens, kv_heads, head_dim, generator=generator)
        outlier_mask = (
            torch.rand(tokens, kv_heads, head_dim, generator=generator) < 0.01
        )
        x = base * magnitude + outlier_mask * outliers * 8.0
        return x.clamp(-MAX_ABS_VALUE, MAX_ABS_VALUE).half()

    return one(), one()


@dataclass(frozen=True)
class Selection:
    indices: torch.Tensor  # int32 [rows, topk]
    table: torch.Tensor  # int32 [requests, pages]
    token_to_req: torch.Tensor  # int32 [rows]
    illegal: torch.Tensor  # bool [rows, topk], the independent expectation
    empty_rows: tuple[int, ...]  # rows whose every entry is illegal


def build_selection(
    layout: Layout,
    table: torch.Tensor,
    rows: int,
    topk: int,
    seed: int,
    *,
    inject_illegal: bool,
) -> Selection:
    """Top-k selections: distinct legal tokens per row, optionally with illegal ones.

    Rows alternate over the requests. With ``inject_illegal`` the first row gets a
    negative entry, an entry past the block table and a huge entry; an unmapped
    page of request 0 and a physical block past the cache of request 1 (when there
    is one) are placed under selected tokens; and, with three or more rows, the
    last row belongs to a request that does not exist. ``illegal`` is computed here
    by the rule's definition, independently of ``nvfp4_entry_validity``.
    """
    generator = torch.Generator().manual_seed(seed)
    table = table.clone()
    token_to_req = (torch.arange(rows) % layout.requests).to(torch.int32)
    span = layout.pages * layout.block_size
    indices = torch.empty((rows, topk), dtype=torch.int32)
    for row in range(rows):
        indices[row] = torch.randperm(layout.tokens_per_request, generator=generator)[
            :topk
        ].to(torch.int32)
    empty: list[int] = []
    if inject_illegal:
        if topk < 8:
            raise ValueError("illegal injection needs a selection width of 8 or more")
        indices[0, 0] = -1
        indices[0, 1] = span  # first token past the block table
        indices[0, 2] = 2**30
        if layout.pages > 1:
            table[0, 1] = -1  # request 0 never got its second page
            indices[0, 3] = layout.block_size + 1  # a token on that page
        if layout.requests > 1:
            table[1, 0] = layout.num_blocks + 5  # past the cache
            indices[1, 3] = 2
            indices[1, 4] = -7
        if rows >= 3:
            token_to_req[rows - 1] = -1
            empty.append(rows - 1)
    illegal = expected_illegal(layout, table, token_to_req, indices)
    return Selection(indices, table, token_to_req, illegal, tuple(empty))


def expected_illegal(
    layout: Layout,
    table: torch.Tensor,
    token_to_req: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """bool ``[rows, topk]``: the entries the kernel must treat as illegal.

    The rule by its definition, written out independently of ``nvfp4_entry_validity``:
    the request is out of range, or the token is negative, or its logical page is past
    the block table, or the mapped physical block is out of ``[0, num_blocks)``. It
    runs on whole tensors (the per-entry Python loop it replaces cost ~22 us per entry,
    i.e. four minutes for one 5568 x 2051 selection); the tests keep the loop as the
    oracle.
    """
    request = token_to_req.long()[:, None]
    token = indices.long()
    bad = (request < 0) | (request >= layout.requests) | (token < 0)
    page = token.div(layout.block_size, rounding_mode="floor")
    bad |= page >= layout.pages
    block = table.long()[
        request.clamp(0, layout.requests - 1), page.clamp(0, layout.pages - 1)
    ]
    bad |= (block < 0) | (block >= layout.num_blocks)
    return bad


@dataclass(frozen=True)
class PrefillSelection:
    indices: torch.Tensor  # int32 [rows, topk]
    table: torch.Tensor  # int32 [1, pages]: request 0 only
    token_to_req: torch.Tensor  # int32 [rows], all zero
    positions: torch.Tensor  # int64 [rows], the last ``rows`` positions
    seq_lens: torch.Tensor  # int32 [1]


def build_prefill_selection(
    layout: Layout, table: torch.Tensor, rows: int, topk: int, seed: int
) -> PrefillSelection:
    """One prefill chunk of request 0: causal 4-token groups plus the partial tail.

    Row ``r`` sits at position ``seq_len - rows + r`` and sees ``position + 1`` tokens.
    It selects up to ``(topk - 3) // 4`` random complete 4-token groups below that, in
    ascending order, then the 1-3 tokens of its last partial group at the slot after
    them: the layout the grouped planner reads (``2051 = 512 * 4 + 3``). Every entry is
    legal and causal; the rest of the row is ``-1``.
    """
    seq_len = layout.tokens_per_request
    if rows > seq_len:
        raise ValueError(f"a chunk of {rows} rows needs --context of at least {rows}")
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(seq_len - rows, seq_len, dtype=torch.int64)
    max_groups = max(0, (topk - 3) // 4)
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    for row, position in enumerate(positions.tolist()):
        visible = position + 1
        full = visible // 4
        count = min(full, max_groups)
        groups = torch.randperm(full, generator=generator)[:count].sort().values
        span = (groups[:, None] * 4 + torch.arange(4)).reshape(-1)
        indices[row, : span.numel()] = span.to(torch.int32)
        tail = min(visible - full * 4, topk - span.numel())
        if tail > 0:
            indices[row, count * 4 : count * 4 + tail] = (
                full * 4 + torch.arange(tail)
            ).to(torch.int32)
    return PrefillSelection(
        indices,
        table[:1].clone(),
        torch.zeros(rows, dtype=torch.int32),
        positions,
        torch.tensor([seq_len], dtype=torch.int32),
    )


# ------------------------------------------------------------- reference rows


def reference_rows(rows: int, limit: int) -> list[int]:
    """Which rows of a batch get the CPU reference: all of them up to ``limit``.

    Above ``limit`` (``limit <= 0`` means no limit) a deterministic subset of at most
    ``limit`` rows: the rows the selection builder treats specially (0 and 1 carry the
    illegal injections of both requests, ``rows - 2`` and ``rows - 1`` are the last
    row of each request, the very last being the empty one) and evenly spaced rows
    over the whole batch, so a kernel that fails only in the far rows (an offset
    that overflows 32 bits at 5568 x 2051 x 256 elements) is still seen. At most
    ``max(limit, 4)`` rows.
    """
    if limit <= 0 or rows <= limit:
        return list(range(rows))
    special = sorted({r for r in (0, 1, rows - 2, rows - 1) if 0 <= r < rows})
    spaced = max(1, limit - len(special))
    even = {(i * (rows - 1)) // max(1, spaced - 1) for i in range(spaced)}
    return sorted(set(special) | even)


def ref_gather_peak_bytes(entries: int, heads: int, head_dim: int) -> int:
    """Peak host bytes of one ``gather_dequant_nvfp4_kv`` over ``entries`` top-k entries.

    ``E = entries * heads * head_dim`` output elements. The reference decodes K, then V
    (``dequantize_kv_nvfp4``), and its second pass runs with the first side's FP16
    output (``2E``) and the loop variable that still holds the first side's
    dequantized tensor (``2E``) alive. In ``e2m1_decode`` the biggest live set is the
    nibble codes (``E``), the int64 table index (``8E``) and the FP32 lookup (``4E``)
    = 13E, and then the nibble codes, the lookup, its negation, a mask and the
    ``where`` result = 14E; the packed bytes (``E/2``) and scale bytes (``E/16``)
    stay. Total ``4E + 14E + E/2 + E/16 = 18.5625E``, plus the int64 index tensors of
    the validity rule (about 48 bytes per entry).
    """
    elements = entries * heads * head_dim
    return round(REF_GATHER_BYTES_PER_ELEMENT * elements + 48 * entries)


def host_mem_model(
    kind: str,
    rows: int,
    geometry: Geometry,
    block_size: int,
    context: int,
    *,
    ref_rows: int = DEFAULT_REF_ROWS,
    ref_chunk: int = REF_CHUNK_ROWS,
    prefill: bool = False,
) -> dict[str, int]:
    """Per-term host-RAM model of one case, bytes; ``peak`` is the case's peak.

    ``N = rows * topk`` selected entries, ``E = N * heads * head_dim`` elements.
    Retained for the whole case: the clean and dirty selections (int32 indices plus a
    bool mask, ``5N`` each), the prefill selection (int32, ``4N``, with the
    ``prefill_scratch`` arm) and the random K/V (FP16). On top of that the largest of
    the case's phases:

    * the reference gather: the reference K and V of the checked rows, kept in FP16
      (``4 * E_s``, ``E_s`` the elements of the ``S`` checked rows), written chunk by
      chunk, plus one chunk of ``ref_chunk`` rows through ``ref_gather_peak_bytes``;
    * the comparison: all of the reference plus one chunk's device output copied to the
      host (FP16 ``2 E_c``) and its FP32 operands and difference (``3 * 4 E_c``);
    * the whole-cache dequantization of the model check (FP32), and the CPU attention
      model over the checked rows (one row at a time, ~25 MB).

    The device tensors (the Triton gather's ``[rows, topk, heads, d]`` K and V, ~5.9 GiB
    each at 5568 rows) live on the GPU and are not host RAM.
    """
    g = geometry
    n = rows * g.topk
    layout = plan_layout(block_size, rows, g.topk, context)
    tokens = layout.requests * layout.tokens_per_request
    sampled = len(reference_rows(rows, ref_rows))
    chunk = max(1, min(ref_chunk, sampled))
    per_entry = g.kv_heads * g.head_dim
    selections = (5 + 5 + (4 if prefill else 0)) * n
    kv_inputs = 2 * 2 * tokens * per_entry
    cache_elements = layout.num_blocks * layout.block_size * per_entry
    cache_phase = round(REF_GATHER_BYTES_PER_ELEMENT * cache_elements) + (
        2 * 4 * tokens * per_entry
    )
    # One row at a time: K/V gathered to FP32 and repeated per query head (~25 MB).
    attention_model = 64 << 20
    parts = {
        "selections": selections,
        "kv_inputs": kv_inputs,
        # ``expected_illegal`` keeps five int64 tensors of the batch's shape alive.
        "build_phase": 40 * n,
    }
    if kind == "nvfp4":
        chunk_entries = chunk * g.topk
        parts["ref_cache"] = 4 * sampled * g.topk * per_entry
        parts["ref_chunk_phase"] = ref_gather_peak_bytes(
            chunk_entries, g.kv_heads, g.head_dim
        )
        parts["compare_phase"] = (2 + 3 * 4) * chunk_entries * per_entry
        parts["model_check_phase"] = cache_phase + attention_model
        # The reference arrays are filled chunk by chunk, so the last chunk's gather
        # meets the earlier chunks' output only; the comparison meets all of it.
        written = 4 * (sampled - chunk) * g.topk * per_entry
        parts["reference_phase"] = max(
            written + parts["ref_chunk_phase"],
            parts["ref_cache"] + parts["compare_phase"],
        )
        phase = max(
            parts["reference_phase"],
            parts["model_check_phase"],
            parts["build_phase"],
        )
    else:
        parts["model_check_phase"] = 4 * 4 * tokens * per_entry + attention_model
        phase = max(parts["model_check_phase"], parts["build_phase"])
    parts["peak"] = selections + kv_inputs + phase
    return parts


def process_rss_bytes(field: str = "VmRSS") -> int:
    """This process's ``VmRSS`` / ``VmHWM`` in bytes (0 where /proc is unavailable)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


# ------------------------------------------------------------------ statistics


def compare_bytes(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    """Byte-level comparison of two uint8 tensors of the same shape."""
    if reference.shape != actual.shape:
        raise ValueError(
            f"shape mismatch {tuple(reference.shape)} {tuple(actual.shape)}"
        )
    different = reference != actual
    count = int(different.sum())
    first = None
    if count:
        first = [int(i) for i in different.nonzero()[0]]
    return {"bytes": reference.numel(), "different": count, "first_index": first}


def compare_cache_regions(
    reference_views: tuple[torch.Tensor, torch.Tensor],
    actual_views: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Compare the (data, scale) views of one cache side, region by region."""
    return {
        "data": compare_bytes(reference_views[0], actual_views[0]),
        "scale": compare_bytes(reference_views[1], actual_views[1]),
    }


def merge_region_comparisons(sides: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Add up the per-side ``compare_cache_regions`` results (K then V)."""
    merged: dict[str, Any] = {}
    for part in ("data", "scale"):
        parts = [side[part] for side in sides]
        merged[part] = {
            "bytes": sum(p["bytes"] for p in parts),
            "different": sum(p["different"] for p in parts),
            "first_index": next(
                (p["first_index"] for p in parts if p["different"]), None
            ),
        }
    return merged


def error_stats(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """max |d|, mean |d| and relative L2 of ``actual`` against ``reference``."""
    a, r = actual.double(), reference.double()
    diff = (a - r).abs()
    norm = float(r.norm())
    return {
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rel_l2": float((a - r).norm()) / norm if norm > 0 else float("nan"),
        "reference_norm": norm,
    }


def summarize_timing(rounds: Sequence[Sequence[float]]) -> dict[str, float]:
    """Median, p10, p90 of all samples and the spread of the per-round medians."""
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
    return {
        "median": statistics.median(samples),
        "p10": percentile(0.1),
        "p90": percentile(0.9),
        "round_spread_pct": spread,
        "samples": len(samples),
    }


def abba_order(arms: Sequence[str], rounds: int) -> list[str]:
    """A B ... B A per round, reversed on every other round, so no arm always
    runs first or last."""
    order: list[str] = []
    for index in range(rounds):
        order += list(arms) if index % 2 == 0 else list(reversed(arms))
    return order


# ------------------------------------------------------------------ controls


@contextlib.contextmanager
def patched(module: Any, **replacements: Any):
    """Replace attributes of ``module`` for the duration of the block."""
    saved = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def swap_nibbles(data: torch.Tensor) -> torch.Tensor:
    """Every uint8 byte with its two nibbles exchanged: the order mutation."""
    return ((data >> 4) | ((data & 0x0F) << 4)).to(torch.uint8)


def sm100_scale_swizzle(
    block_size: int, scale_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source and destination flat positions of SM100's V scale swizzle.

    ``nvfp4_kv_cache_kernels.cu`` ``swizzle_scale_offset``: with ``g = S / 4``,
    ``(t, s)`` moves to ``((t / 4) * 4 + s / g, (s % g) * 4 + t % 4)``. A bijection of
    the ``block_size * scale_dim`` positions of one block and head; V100 stores scales
    linearly, so applying it is a placement error.
    """
    if block_size % 4 or scale_dim % 4:
        raise ValueError("the swizzle needs block_size and scale_dim divisible by 4")
    t = torch.arange(block_size)[:, None]
    s = torch.arange(scale_dim)[None, :]
    group = scale_dim // 4
    swizzled_t = (t // 4) * 4 + s // group
    swizzled_s = (s % group) * 4 + t % 4
    source = (t * scale_dim + s).reshape(-1)
    destination = (swizzled_t * scale_dim + swizzled_s).reshape(-1)
    return source, destination


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """The placement mutation applied to ``(blocks, block_size, heads, scale_dim)``."""
    blocks, block_size, heads, scale_dim = scales.shape
    source, destination = sm100_scale_swizzle(block_size, scale_dim)
    flat = scales.permute(0, 2, 1, 3).reshape(blocks, heads, block_size * scale_dim)
    moved = torch.empty_like(flat)
    moved[..., destination] = flat[..., source]
    return moved.reshape(blocks, heads, block_size, scale_dim).permute(0, 2, 1, 3)


def mutate_cache(
    cache: torch.Tensor, mutation: str, split_views: Callable
) -> torch.Tensor:
    """A copy of a 5-D NVFP4 ``cache`` with one mutation applied to both sides."""
    out = cache.clone()
    (k_data, v_data), (k_scale, v_scale) = split_views(out)
    if mutation == "nibble_order_swapped":
        for data in (k_data, v_data):
            data.copy_(swap_nibbles(data))
    elif mutation == "scale_placement_wrong":
        for scale in (k_scale, v_scale):
            scale.copy_(swizzle_scales(scale))
    else:
        raise ValueError(f"unknown mutation {mutation}")
    return out


def e2m1_encode_half_up(x: torch.Tensor) -> torch.Tensor:
    """E2M1 code with ties rounded up in magnitude (the nibble tie-rule mutation)."""
    magnitude = x.float().abs()
    code = sum(
        (magnitude >= threshold).to(torch.uint8)
        for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    )
    negative = (x.float() < 0) & (code != 0)
    return code | (negative.to(torch.uint8) << 3)


def e4m3_encode_half_up(
    x: torch.Tensor, encode: Callable, decode: Callable
) -> torch.Tensor:
    """E4M3 byte with ties rounded up, from the round-to-nearest-even ``encode``.

    ``encode`` and ``decode`` are the reference functions, passed in so the mutation can
    replace them without recursing. Block scales are non-negative.
    """
    rne = encode(x)
    code = rne & 0x7F
    value = decode(code)
    upper = decode(torch.clamp(code + 1, max=0x7E).to(torch.uint8))
    magnitude = x.float().clamp(0.0, E4M3_MAX)
    tie_up = (magnitude > value) & (magnitude == (value + upper) / 2)
    return rne + tie_up.to(torch.uint8)


def tie_group_values(token: int, group: int) -> list[float]:
    """16 fp16-exact values of one group, on rounding ties at unit layer scale.

    Even ``token + group``: amax 6 (block scale exactly 1) with every E2M1 midpoint
    and its negative. Odd: an amax of 6 times an E4M3 midpoint, so the scale byte
    itself is a tie, with the other values at multiples of amax / 16.
    """
    if (token + group) % 2 == 0:
        return [6.0, *E2M1_TIE_VALUES, *(-v for v in E2M1_TIE_VALUES), -6.0]
    amax = E4M3_TIE_AMAX[((token + group) // 2) % len(E4M3_TIE_AMAX)]
    return [amax] + [amax * (-1) ** j * j / 16.0 for j in range(1, 16)]


def make_tie_kv(
    tokens: int, kv_heads: int, head_dim: int, shift: int = 0
) -> torch.Tensor:
    """fp16 ``[tokens, kv_heads, head_dim]`` whose groups sit on rounding ties."""
    if head_dim % 16:
        raise ValueError("head_dim must be a multiple of 16")
    out = torch.zeros(tokens, kv_heads, head_dim)
    for token in range(tokens):
        for head in range(kv_heads):
            for group in range(head_dim // 16):
                values = tie_group_values(token + shift, group)
                out[token, head, group * 16 : (group + 1) * 16] = torch.tensor(values)
    return out.half()


def model_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selection: Selection,
    tokens_per_request: int,
    rows: Sequence[int] | None = None,
) -> torch.Tensor:
    """FP32 attention, one row at a time, over each row's selected tokens.

    ``keys`` and ``values`` are ``[requests * tokens_per_request, heads, d]``, request
    major, the order ``slot_mapping`` stores them in. Illegal entries are not modelled
    (use the clean selection). ``rows`` limits the model to those rows (the result is
    ``[len(rows), q_heads, d]`` in that order); the default is every row. The model costs
    ~70 ms per row on the CPU, so a 5568-row batch must not run it whole.
    """
    _, q_heads, head_dim = q.shape
    chosen = range(q.shape[0]) if rows is None else list(rows)
    group = q_heads // keys.shape[1]
    out = torch.zeros((len(chosen), q_heads, head_dim), dtype=torch.float32)
    for position, row in enumerate(chosen):
        request = int(selection.token_to_req[row])
        index = selection.indices[row].long() + request * tokens_per_request
        k = keys[index].float().repeat_interleave(group, dim=1)
        v = values[index].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), k) * head_dim**-0.5
        out[position] = torch.einsum("hk,khd->hd", torch.softmax(scores, dim=-1), v)
    return out


def error_in_band(measured: float, model: float) -> bool:
    """Is the measured relative L2 within ``BAND_LOW..BAND_HIGH`` of the model's?"""
    if not (model > 0 and measured > 0):  # also false for NaN
        return False
    return BAND_LOW <= measured / model <= BAND_HIGH


# ------------------------------------------------------------------ verdicts


def _yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def fused_within_tolerance(fused: dict[str, Any]) -> bool:
    """The fused reader against the gather route: rounding-level error, exact zeros."""
    return bool(
        fused["vs_gather"]["rel_l2"] <= FUSED_REL_L2_TOLERANCE
        and fused["vs_gather_illegal"]["rel_l2"] <= FUSED_REL_L2_TOLERANCE
        and fused["zero_fill_ok"]
    )


def scratch_decode_matches_gather(prefill: dict[str, Any]) -> bool:
    """The scratch decode equals the gather bit for bit, and its tail is +0.0."""
    decode = prefill["decode"]
    return bool(decode["different"] == 0 and decode["tail_zero_ok"])


def scratch_route_within_tolerance(route: dict[str, Any]) -> bool:
    """The whole scratch route against the fused reader and the gather route."""
    return bool(
        route["vs_fused"]["rel_l2"] <= FUSED_REL_L2_TOLERANCE
        and route["vs_gather"]["rel_l2"] <= FUSED_REL_L2_TOLERANCE
    )


def compute_verdict(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verdict lines from the per-case results.

    A result is ``{"case": label, "kind": ..., "rows": ..., "error": str | None,
    "store": ..., "gather": ..., "zero_fill": ..., "decode_path": ...,
    "decode_quality": ...}``; keys a case did not reach are absent or None. A line is
    UNKNOWN when no NVFP4 case produced its measurement.
    """
    nvfp4 = [r for r in results if r["kind"] == "nvfp4"]
    verdict: dict[str, Any] = {}

    def over(key: str, ok: Callable[[Any], bool]) -> str:
        """NO if any case that got there failed, else UNKNOWN if any case did not
        get there (or none exists), else YES."""
        reached = [r for r in nvfp4 if r.get(key) is not None]
        if any(not ok(r[key]) for r in reached):
            return "NO"
        if not nvfp4 or len(reached) < len(nvfp4):
            return "UNKNOWN"
        return "YES"

    verdict["STORE_MATCHES_REFERENCE"] = over(
        "store",
        lambda s: s["data"]["different"] == 0 and s["scale"]["different"] == 0,
    )
    verdict["GATHER_MATCHES_REFERENCE"] = over("gather", lambda g: g["different"] == 0)
    verdict["ZERO_FILL_OK"] = over("zero_fill", lambda z: z["ok"])
    verdict["DECODE_PATH_IDENTICAL"] = over(
        "decode_path", lambda d: d["max_abs"] == 0.0
    )
    if any("fused" in r for r in nvfp4):
        verdict["FUSED_MATCHES_GATHER"] = over("fused", fused_within_tolerance)
    if any("prefill_scratch" in r for r in nvfp4):
        verdict["SCRATCH_MATCHES_GATHER"] = over(
            "prefill_scratch", scratch_decode_matches_gather
        )
        routed = [
            r
            for r in nvfp4
            if "vs_fused" in ((r.get("prefill_scratch") or {}).get("route") or {})
        ]
        if any(
            not scratch_route_within_tolerance(r["prefill_scratch"]["route"])
            for r in routed
        ):
            verdict["SCRATCH_ROUTE_WITHIN_TOLERANCE"] = "NO"
        else:
            verdict["SCRATCH_ROUTE_WITHIN_TOLERANCE"] = "YES" if routed else "UNKNOWN"
    verdict["errors"] = {r["case"]: r["error"] for r in results if r.get("error")}
    verdict["skipped"] = {r["case"]: r["skipped"] for r in results if r.get("skipped")}
    verdict["REFERENCE_ROWS"] = {
        r["case"]: f"{r['ref_rows']['checked']}/{r['ref_rows']['total']}"
        for r in results
        if r.get("ref_rows") and not r.get("skipped")
    }
    ratios: dict[str, float] = {}
    for rows in sorted({r["rows"] for r in results}):
        fp4 = [
            r["decode_quality"]["rel_l2"]
            for r in nvfp4
            if r["rows"] == rows and r.get("decode_quality")
        ]
        e4 = [
            r["decode_quality"]["rel_l2"]
            for r in results
            if r["kind"] == "e4m3" and r["rows"] == rows and r.get("decode_quality")
        ]
        if fp4 and e4 and e4[0] > 0:
            ratios[f"M={rows}"] = max(fp4) / e4[0]
    verdict["NVFP4_VS_E4M3_REL_L2"] = ratios
    return verdict


def decode_roundtrip_lines(timings: dict[str, dict[str, Any]]) -> list[str]:
    """``DECODE_ROUNDTRIP_US`` lines from ``timings[<group>][<arm>]`` summaries."""
    lines = []
    for group, arms in timings.items():
        fp4 = [v["median"] for k, v in arms.items() if k.startswith("decode_nvfp4")]
        e4_arms = [v for k, v in arms.items() if k.startswith("decode_e4m3")]
        e4 = e4_arms[0] if e4_arms else None
        if fp4 and e4:
            lines.append(
                f"DECODE_ROUNDTRIP_US {group}: nvfp4 "
                + "/".join(f"{x:.1f}" for x in fp4)
                + f" e4m3 {e4['median']:.1f} ratio "
                + "/".join(f"{x / e4['median']:.2f}" for x in fp4)
            )
        fused = [v["median"] for k, v in arms.items() if k.startswith("decode_fused")]
        if fused and fp4:
            lines.append(
                f"FUSED_ROUNDTRIP_US {group}: fused "
                + "/".join(f"{x:.1f}" for x in fused)
                + " gather "
                + "/".join(f"{x:.1f}" for x in fp4)
                + " ratio "
                + "/".join(f"{f / g:.2f}" for f, g in zip(fused, fp4, strict=False))
            )
        scratch = [
            v["median"]
            for k, v in arms.items()
            if k.startswith("prefill_scratch_route")
        ]
        prefill_fused = [
            v["median"] for k, v in arms.items() if k.startswith("prefill_fused")
        ]
        prefill_e4m3 = [v for k, v in arms.items() if k.startswith("prefill_e4m3")]
        if (scratch or prefill_fused) and prefill_e4m3:
            base = prefill_e4m3[0]["median"]
            lines.append(
                f"PREFILL_ROUNDTRIP_US {group}: e4m3 {base:.1f} scratch route "
                + ("/".join(f"{x:.1f}" for x in scratch) or "n/a")
                + " fused "
                + ("/".join(f"{x:.1f}" for x in prefill_fused) or "n/a")
                + " ratio to e4m3 (scratch route, fused) "
                + ("/".join(f"{x / base:.2f}" for x in scratch) or "n/a")
                + ", "
                + ("/".join(f"{x / base:.2f}" for x in prefill_fused) or "n/a")
            )
    return lines


def exit_code_for(results: Sequence[dict[str, Any]]) -> int:
    return 1 if any(r.get("error") for r in results) else 0


def format_verdict(
    verdict: dict[str, Any], results: Sequence[dict[str, Any]]
) -> list[str]:
    lines = []
    for key in (
        "STORE_MATCHES_REFERENCE",
        "GATHER_MATCHES_REFERENCE",
        "ZERO_FILL_OK",
        "DECODE_PATH_IDENTICAL",
        "FUSED_MATCHES_GATHER",
        "SCRATCH_MATCHES_GATHER",
        "SCRATCH_ROUTE_WITHIN_TOLERANCE",
    ):
        if key in verdict:
            lines.append(f"{key}: {verdict[key]}")
    for r in results:
        scratch = r.get("prefill_scratch")
        if not scratch:
            continue
        d = scratch["decode"]
        lines.append(
            f"SCRATCH_DECODE {r['case']}: {d['different']} of {d['elements']} "
            f"elements differ from the gather, tail +0.0 "
            f"{_yes_no(d['tail_zero_ok'])}"
        )
        route = scratch.get("route") or {}
        if "skipped" in route:
            lines.append(f"SCRATCH_ROUTE {r['case']}: skipped ({route['skipped']})")
        for name, key in (
            ("SCRATCH_VS_FUSED", "vs_fused"),
            ("SCRATCH_VS_GATHER", "vs_gather"),
        ):
            if key in route:
                e = route[key]
                lines.append(
                    f"{name} {r['case']}: max|d| {e['max_abs']:.3g} "
                    f"mean|d| {e['mean_abs']:.3g} relL2 {e['rel_l2']:.3g}"
                )
    for r in results:
        if (
            r.get("store")
            and r["store"]["data"]["different"] + r["store"]["scale"]["different"]
        ):
            d, sc = r["store"]["data"], r["store"]["scale"]
            lines.append(
                f"STORE_DIFF {r['case']}: data {d['different']}/{d['bytes']} "
                f"first {d['first_index']}, scale {sc['different']}/{sc['bytes']} "
                f"first {sc['first_index']}"
            )
        if r.get("gather") and r["gather"]["different"]:
            lines.append(
                f"GATHER_DIFF {r['case']}: {r['gather']['different']} of "
                f"{r['gather']['elements']} elements, "
                f"max|d| {r['gather']['max_abs']:.3g}"
            )
        if r.get("decode_path") and r["decode_path"]["max_abs"] != 0.0:
            lines.append(
                f"DECODE_PATH_DIFF {r['case']}: "
                f"max|d| {r['decode_path']['max_abs']:.3g}"
            )
    for r in results:
        q = r.get("decode_quality")
        if not q:
            continue
        name = "E4M3_CONTROL" if r["kind"] == "e4m3" else "DECODE_VS_FP16"
        lines.append(
            f"{name} {r['case']}: max|d| {q['max_abs']:.4g} mean|d| {q['mean_abs']:.4g}"
            f" relL2 {q['rel_l2']:.4g}"
        )
    for r in results:
        fused = r.get("fused")
        if not fused:
            continue
        for name, key in (("FUSED_VS_GATHER", "vs_gather"), ("FUSED_VS_FP16", "quality")):
            q = fused[key]
            lines.append(
                f"{name} {r['case']}: max|d| {q['max_abs']:.4g} "
                f"mean|d| {q['mean_abs']:.4g} relL2 {q['rel_l2']:.4g}"
            )
    for group, ratio in verdict["NVFP4_VS_E4M3_REL_L2"].items():
        lines.append(f"NVFP4_VS_E4M3_REL_L2 {group}: {ratio:.2f}")
    for case, error in verdict["errors"].items():
        lines.append(f"CASE_ERROR {case}: {error}")
    for case, reason in verdict["skipped"].items():
        lines.append(f"CASE_SKIPPED {case}: {reason}")
    for case, coverage in verdict["REFERENCE_ROWS"].items():
        checked, total = coverage.split("/")
        lines.append(
            f"REFERENCE_ROWS {case}: CPU reference and model on {checked} of {total} "
            f"rows ({'every row' if checked == total else 'a deterministic subset'});"
            " kernels, zero-fill and timings on every row"
        )
    return lines


def _over(
    cases: Sequence[dict[str, Any]],
    get: Callable[[dict[str, Any]], Any],
    ok: Callable[[Any], bool],
    good: str,
    bad: str,
) -> str:
    """``bad`` if any case that has the measurement fails it, else SKIPPED if a case
    lacks it (or there is no case), else ``good``."""
    got = [get(r) for r in cases]
    if any(g is not None and not ok(g) for g in got):
        return bad
    if not cases or any(g is None for g in got):
        return "SKIPPED"
    return good


def _differs(comparison: dict[str, Any], region: str) -> bool:
    return comparison[region]["different"] > 0


def _control_flagged(mutation: str, link: str, control: dict[str, Any]) -> bool:
    if link == "store":
        region = "data" if mutation == "nibble_order_swapped" else "scale"
        return _differs(control["store"], region)
    if link == "gather":
        return control["gather"]["different"] > 0
    return control["decode"]["max_abs"] > 0.0


def compute_self_check(
    results: Sequence[dict[str, Any]],
    timings: dict[str, dict[str, Any]],
    *,
    no_timing: bool = False,
    fused_arm: bool | None = None,
    prefill_arm: bool | None = None,
    prefill_route_required: bool = False,
) -> dict[str, Any]:
    """The positive controls, negative controls and stages, and the overall verdict.

    ``fused_arm`` says whether the fused reader was asked for (``--arms``); ``None``
    infers it from the results, which cannot see a fused arm that raised everywhere.
    ``prefill_arm`` is the same for the prefill scratch route; its CUDA part must have
    run (``SCRATCH_ROUTE_WITHIN_TOLERANCE``) only when ``prefill_route_required``
    (the run is on a GPU).
    """
    nvfp4 = [r for r in results if r["kind"] == "nvfp4"]
    e4m3 = [r for r in results if r["kind"] == "e4m3"]
    verdict = compute_verdict(results)
    positive: dict[str, str] = {}
    for name in POSITIVE_CONTROLS[:4]:
        positive[name] = {"YES": "PASS", "NO": "FAIL"}.get(verdict[name], "SKIPPED")

    def band(r: dict[str, Any]) -> Any:
        check = r.get("model_check")
        return None if check is None else check

    def in_band(check: dict[str, Any]) -> bool:
        return error_in_band(check["measured_rel_l2"], check["model_rel_l2"])

    positive["TIE_STORE_MATCHES_REFERENCE"] = _over(
        nvfp4,
        lambda r: r.get("tie_control"),
        lambda t: not _differs(t["correct"], "data")
        and not _differs(t["correct"], "scale"),
        "PASS",
        "FAIL",
    )
    positive["NVFP4_ERROR_IN_BAND"] = _over(nvfp4, band, in_band, "PASS", "FAIL")
    positive["E4M3_ERROR_IN_BAND"] = _over(e4m3, band, in_band, "PASS", "FAIL")

    fused_run = (
        any("fused" in r for r in nvfp4) if fused_arm is None else fused_arm
    )
    if fused_run:
        positive["FUSED_MATCHES_GATHER"] = _over(
            nvfp4, lambda r: r.get("fused"), fused_within_tolerance, "PASS", "FAIL"
        )

    prefill_run = (
        any("prefill_scratch" in r for r in nvfp4)
        if prefill_arm is None
        else prefill_arm
    )
    if prefill_run:
        positive["SCRATCH_MATCHES_GATHER"] = _over(
            nvfp4,
            lambda r: r.get("prefill_scratch"),
            scratch_decode_matches_gather,
            "PASS",
            "FAIL",
        )
        if prefill_route_required:
            applicable = [
                r
                for r in nvfp4
                if not ((r.get("prefill_scratch") or {}).get("route") or {}).get(
                    "not_applicable"
                )
            ]
            positive["SCRATCH_ROUTE_WITHIN_TOLERANCE"] = _over(
                applicable,
                lambda r: (r.get("prefill_scratch") or {})
                .get("route", {})
                .get("vs_fused")
                and r["prefill_scratch"]["route"],
                scratch_route_within_tolerance,
                "PASS",
                "FAIL",
            )

    negative: dict[str, str] = {}
    for mutation in MUTATIONS:
        for link in LINKS:
            negative[f"{mutation} {link}"] = _over(
                nvfp4,
                lambda r, m=mutation: (r.get("controls") or {}).get(m),
                lambda c, m=mutation, link=link: _control_flagged(m, link, c),
                "FLAGGED",
                "MISSED",
            )
    if fused_run:
        for mutation in MUTATIONS:
            negative[f"{mutation} fused_decode"] = _over(
                nvfp4,
                lambda r, m=mutation: ((r.get("fused") or {}).get("controls") or {}).get(
                    m
                ),
                lambda c, m=mutation: _control_flagged(m, "decode", c),
                "FLAGGED",
                "MISSED",
            )
    if prefill_run:
        for mutation in MUTATIONS:
            negative[f"{mutation} scratch_decode"] = _over(
                nvfp4,
                lambda r, m=mutation: (
                    ((r.get("prefill_scratch") or {}).get("controls") or {}).get(m)
                ),
                lambda c: c["decode"]["different"] > 0,
                "FLAGGED",
                "MISSED",
            )
    for mutation, key, region in (
        ("tie_rule_removed_e2m1", "e2m1_mutant", "data"),
        ("tie_rule_removed_e4m3", "e4m3_mutant", "scale"),
    ):
        negative[f"{mutation} tie_store"] = _over(
            nvfp4,
            lambda r, k=key: (r.get("tie_control") or {}).get(k),
            lambda c, region=region: _differs(c, region),
            "FLAGGED",
            "MISSED",
        )

    def ran(cases: Sequence[dict[str, Any]], *keys: str) -> str:
        done = bool(cases) and all(
            all(r.get(k) is not None for k in keys) for r in cases
        )
        return "RAN" if done else "SKIPPED"

    timing_ran = (
        not no_timing
        and bool(timings)
        and all(
            arms and all(a["median"] > 0 for a in arms.values())
            for arms in timings.values()
        )
    )
    stages = {
        "store": ran(nvfp4, "store"),
        "gather": ran(nvfp4, "gather"),
        "zero_fill": ran(nvfp4, "zero_fill"),
        "decode": ran(nvfp4, "decode_path", "decode_quality"),
        "e4m3_control": ran(e4m3, "decode_quality", "model_check"),
        "tie_control": ran(nvfp4, "tie_control"),
        "negative_controls": ran(nvfp4, "controls"),
        "timing": "RAN" if timing_ran else "SKIPPED",
    }
    if fused_run:
        stages["decode_fused"] = ran(nvfp4, "fused")
    if prefill_run:
        stages["prefill_scratch"] = ran(nvfp4, "prefill_scratch")
    reasons = [f"{n} {v}" for n, v in positive.items() if v != "PASS"]
    reasons += [f"{n} {v}" for n, v in negative.items() if v != "FLAGGED"]
    reasons += [f"stage {n} {v}" for n, v in stages.items() if v != "RAN"]
    reasons += [f"{r['case']} raised" for r in results if r.get("error")]
    reasons += [
        f"{r['case']} skipped by the host-memory guard"
        for r in results
        if r.get("skipped")
    ]
    return {
        "positive": positive,
        "negative": negative,
        "stages": stages,
        "reasons": reasons,
        "verdict": "FAIL" if reasons else "PASS",
    }


def format_self_check(
    check: dict[str, Any], results: Sequence[dict[str, Any]]
) -> list[str]:
    lines = []
    for r in results:
        m = r.get("model_check")
        if m:
            ratio = (
                m["measured_rel_l2"] / m["model_rel_l2"]
                if m["model_rel_l2"]
                else float("nan")
            )
            name = "E4M3_BAND" if r["kind"] == "e4m3" else "NVFP4_BAND"
            lines.append(
                f"{name} {r['case']}: measured relL2 {m['measured_rel_l2']:.4g} "
                f"model {m['model_rel_l2']:.4g} ratio {ratio:.3f} "
                f"(band {BAND_LOW}..{BAND_HIGH})"
            )
    lines += [f"POSITIVE_CONTROL {n}: {v}" for n, v in check["positive"].items()]
    lines += [f"NEGATIVE_CONTROL {n}: {v}" for n, v in check["negative"].items()]
    lines += [f"STAGE {n}: {v}" for n, v in check["stages"].items()]
    if check["verdict"] == "PASS":
        lines.append("SELF_CHECK: PASS")
    else:
        lines.append(f"SELF_CHECK: FAIL ({'; '.join(check['reasons'])})")
    return lines


def exit_code_for_self_check(check: dict[str, Any]) -> int:
    return 0 if check["verdict"] == "PASS" else 1


# ------------------------------------------------------------------ the measurement


def load_kernels() -> dict[str, Any]:
    """Import the kernels lazily: planning and verdict code needs no vLLM."""
    from vllm import _custom_ops as ops
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv as nv
    from vllm.models.qwen4_exp.nvidia.ops import nvfp4_kv_triton as nvt
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_nvfp4 as qsa_nvfp4_ops

    return {"ops": ops, "nv": nv, "nvt": nvt, "qsa": qsa_ops, "qn": qsa_nvfp4_ops}


def _sync() -> None:
    if WORKER_DEVICE.startswith("cuda"):
        torch.accelerator.synchronize()


def time_arms(
    arms: dict[str, Callable[[], Any]], *, rounds: int, iters: int, warmup: int
) -> dict[str, dict[str, float]]:
    """Time each arm; CUDA events on a GPU, ``perf_counter`` on the CPU tests.

    Arms are interleaved A B B A across ``rounds`` rounds (``abba_order``), each
    turn running ``iters`` calls and recording the median call time of that turn.
    """
    for fn in arms.values():
        for _ in range(warmup):
            fn()
    _sync()
    use_events = WORKER_DEVICE.startswith("cuda")
    per_arm: dict[str, list[list[float]]] = {name: [] for name in arms}
    for name in abba_order(list(arms), rounds):
        fn = arms[name]
        samples: list[float] = []
        for _ in range(iters):
            if use_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0)
            else:
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1e6)
        per_arm[name].append(samples)
    return {name: summarize_timing(r) for name, r in per_arm.items()}


class CaseRun:
    """One case: build the caches, verify the kernels, expose timing closures."""

    def __init__(
        self,
        case: Case,
        geometry: Geometry,
        kernels: dict[str, Any],
        *,
        seed: int,
        context: int,
        k_scale: float,
        v_scale: float,
        fused: bool = False,
        prefill_scratch: bool = False,
        ref_rows: int = DEFAULT_REF_ROWS,
        ref_chunk: int = REF_CHUNK_ROWS,
    ) -> None:
        self.case, self.geometry, self.k = case, geometry, kernels
        self.fused = fused
        self.prefill_scratch = prefill_scratch
        # The rows the CPU references and the CPU model run on (see DEFAULT_REF_ROWS).
        self.checked = reference_rows(case.rows, ref_rows)
        self.ref_chunk = max(1, ref_chunk)
        self.seed, self.k_scale, self.v_scale = seed, k_scale, v_scale
        self.device = torch.device(WORKER_DEVICE)
        self.layout = plan_layout(case.block_size, case.rows, geometry.topk, context)
        self.table = make_block_table(self.layout, seed)
        self.slots = slot_mapping(self.table, self.layout)
        total = self.layout.requests * self.layout.tokens_per_request
        self.key, self.value = make_kv(
            total, geometry.kv_heads, geometry.head_dim, seed
        )
        generator = torch.Generator().manual_seed(seed + 1)
        self.q = torch.randn(
            case.rows, geometry.q_heads, geometry.head_dim, generator=generator
        ).half()
        self.clean = build_selection(
            self.layout,
            self.table,
            case.rows,
            geometry.topk,
            seed + 2,
            inject_illegal=False,
        )
        self.dirty = build_selection(
            self.layout,
            self.table,
            case.rows,
            geometry.topk,
            seed + 3,
            inject_illegal=True,
        )
        self.prefill: PrefillSelection | None = None
        if prefill_scratch:
            self.prefill = build_prefill_selection(
                self.layout, self.table, case.rows, geometry.topk, seed + 4
            )

    # -- helpers ---------------------------------------------------------
    def _dev(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device)

    def _fp16_cache(self, key: torch.Tensor, value: torch.Tensor):
        """A plain FP16 paged cache holding ``key`` / ``value`` at the stored slots."""
        shape = (
            self.layout.num_blocks,
            self.layout.block_size,
            self.geometry.kv_heads,
            self.geometry.head_dim,
        )
        k16 = torch.zeros(shape, dtype=torch.float16)
        v16 = torch.zeros(shape, dtype=torch.float16)
        flat_k = k16.view(-1, self.geometry.kv_heads, self.geometry.head_dim)
        flat_v = v16.view(-1, self.geometry.kv_heads, self.geometry.head_dim)
        flat_k[self.slots] = key
        flat_v[self.slots] = value
        return self._dev(k16), self._dev(v16)

    def _attention(self, selection: Selection, **kwargs):
        return self.k["qsa"].qsa_sparse_paged_attention(
            self._dev(self.q),
            kwargs.pop("k_cache"),
            kwargs.pop("v_cache"),
            self._dev(selection.indices),
            self._dev(selection.table),
            self._dev(selection.token_to_req),
            **kwargs,
        )

    # -- the NVFP4 case --------------------------------------------------
    def run_nvfp4(self) -> dict[str, Any]:
        nv, nvt = self.k["nv"], self.k["nvt"]
        g, layout = self.geometry, self.layout
        shape = (
            layout.num_blocks,
            2,
            layout.block_size,
            g.kv_heads,
            nv.nvfp4_kv_row_bytes(g.head_dim),
        )
        ks, vs = self.k_scale, self.v_scale
        reference = torch.zeros(shape, dtype=torch.uint8)
        nv.reshape_and_cache_nvfp4_reference(
            self.key, self.value, reference, self.slots, k_scale=ks, v_scale=vs
        )
        actual = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        nvt.store_nvfp4_kv_triton(
            self._dev(self.key),
            self._dev(self.value),
            actual,
            self._dev(self.slots),
            k_scale=torch.tensor(ks, device=self.device),
            v_scale=torch.tensor(vs, device=self.device),
        )
        _sync()
        ref_views = nv.nvfp4_kv_split_views(reference)
        actual_cpu = actual.cpu()
        act_views = nv.nvfp4_kv_split_views(actual_cpu)
        result: dict[str, Any] = {
            "store": merge_region_comparisons(
                [
                    compare_cache_regions(
                        (ref_views[0][side], ref_views[1][side]),
                        (act_views[0][side], act_views[1][side]),
                    )
                    for side in (0, 1)
                ]
            )
        }

        cache = self._dev(reference)  # the reference bytes: gather and decode are
        # tested on them, so a store mismatch cannot hide behind a later one.
        sel = self.dirty
        # The kernel runs on every row; the CPU reference only on ``self.checked``
        # rows, a chunk at a time (a whole-batch reference needs ~19 KiB of host RAM
        # per row), and the comparison reads those rows of the kernel's output.
        keys, values = nvt.gather_dequant_nvfp4_kv_triton(
            cache,
            self._dev(sel.table),
            self._dev(sel.token_to_req),
            self._dev(sel.indices),
            k_scale=ks,
            v_scale=vs,
        )
        ref_keys, ref_values = self._ref_gather(nv, reference, sel)
        result["gather"] = self._gather_diff(keys, values, ref_keys, ref_values)
        zeros_ok = self._zero_fill_ok(keys, values, sel.illegal)
        del keys, values

        # Path identity: a cache stored at unit layer scales, decoded at unit scales,
        # against the FP16 kernel on the dequantized values of that same cache.
        unit = torch.zeros(shape, dtype=torch.uint8)
        nv.reshape_and_cache_nvfp4_reference(self.key, self.value, unit, self.slots)
        (u_k, u_v), (u_ks, u_vs) = nv.nvfp4_kv_split_views(unit)
        unit_cache = self._dev(unit)
        nvfp4_out = self._attention(
            sel,
            k_cache=unit_cache[:, 0],
            v_cache=unit_cache[:, 1],
            kv_cache_dtype="nvfp4",
        )
        fp16_out = self._attention(
            sel,
            k_cache=self._dev(nv.dequantize_kv_nvfp4(u_k, u_ks)),
            v_cache=self._dev(nv.dequantize_kv_nvfp4(u_v, u_vs)),
        )
        nvfp4_out, fp16_out = nvfp4_out.cpu(), fp16_out.cpu()
        decode_ok = bool(torch.equal(nvfp4_out, fp16_out))
        empty_ok = all(not nvfp4_out[row].any() for row in sel.empty_rows)
        result["zero_fill"] = {"ok": bool(zeros_ok and empty_ok and decode_ok)}
        result["decode_path"] = error_stats(nvfp4_out, fp16_out)

        # Negative controls: the same three comparisons, on mutated caches.
        controls: dict[str, Any] = {}
        for mutation in MUTATIONS:
            mutated = mutate_cache(reference, mutation, nv.nvfp4_kv_split_views)
            m_views = nv.nvfp4_kv_split_views(mutated)
            store_cmp = merge_region_comparisons(
                [
                    compare_cache_regions(
                        (m_views[0][side], m_views[1][side]),
                        (act_views[0][side], act_views[1][side]),
                    )
                    for side in (0, 1)
                ]
            )
            m_keys, m_values = nvt.gather_dequant_nvfp4_kv_triton(
                self._dev(mutated),
                self._dev(sel.table),
                self._dev(sel.token_to_req),
                self._dev(sel.indices),
                k_scale=ks,
                v_scale=vs,
            )
            gather_cmp = {
                "different": self._gather_diff(m_keys, m_values, ref_keys, ref_values)[
                    "different"
                ]
            }
            del m_keys, m_values
            m_unit = self._dev(mutate_cache(unit, mutation, nv.nvfp4_kv_split_views))
            m_out = self._attention(
                sel,
                k_cache=m_unit[:, 0],
                v_cache=m_unit[:, 1],
                kv_cache_dtype="nvfp4",
            ).cpu()
            controls[mutation] = {
                "store": store_cmp,
                "gather": gather_cmp,
                "decode": error_stats(m_out, fp16_out),
            }
        result["controls"] = controls
        del ref_keys, ref_values
        result["tie_control"] = self._tie_control(nv, nvt)

        # Quantization error at the realistic scales against the unquantized K/V.
        clean = self.clean
        k_orig, v_orig = self._fp16_cache(self.key, self.value)
        baseline = self._attention(clean, k_cache=k_orig, v_cache=v_orig).cpu()
        quantized = self._attention(
            clean,
            k_cache=cache[:, 0],
            v_cache=cache[:, 1],
            kv_cache_dtype="nvfp4",
            k_scale=ks,
            v_scale=vs,
        ).cpu()
        result["decode_quality"] = error_stats(quantized, baseline)
        (k_data, v_data), (k_scales, v_scales) = nv.nvfp4_kv_split_views(reference)
        heads, head_dim = g.kv_heads, g.head_dim
        k_model = nv.dequantize_kv_nvfp4(
            k_data, k_scales, layer_scale=ks, out_dtype=torch.float32
        ).reshape(-1, heads, head_dim)[self.slots]
        v_model = nv.dequantize_kv_nvfp4(
            v_data, v_scales, layer_scale=vs, out_dtype=torch.float32
        ).reshape(-1, heads, head_dim)[self.slots]
        result["model_check"] = self._model_check(k_model, v_model, quantized, baseline)
        if self.fused:
            result["fused"] = self._run_fused(
                nv,
                sel,
                unit_cache,
                unit,
                nvfp4_out,
                fp16_out,
                clean,
                cache,
                quantized,
                baseline,
            )
        if self.prefill_scratch:
            result["prefill_scratch"] = self._run_prefill_scratch(
                nv, nvt, reference, cache
            )
        self._cache = cache
        # What the FP16 kernel alone reads once the gather is done (timing only).
        self._k_deq = self._dev(
            nv.dequantize_kv_nvfp4(k_data, k_scales, layer_scale=ks)
        )
        self._v_deq = self._dev(
            nv.dequantize_kv_nvfp4(v_data, v_scales, layer_scale=vs)
        )
        return result

    def _run_fused(
        self,
        nv,
        sel: Selection,
        unit_cache: torch.Tensor,
        unit: torch.Tensor,
        gather_unit_out: torch.Tensor,
        fp16_unit_out: torch.Tensor,
        clean: Selection,
        cache: torch.Tensor,
        gather_quantized: torch.Tensor,
        baseline: torch.Tensor,
    ) -> dict[str, Any]:
        """The fused reader against the gather route (tolerance, not bit equality).

        ``vs_gather``: the clean selection at the realistic layer scales, the same
        comparison as ``decode_quality`` but against the gather route's output.
        ``vs_gather_illegal``: the selection with illegal entries at unit scales,
        against the gather route's output (``gather_unit_out``). ``quality``: the
        quantization error of the fused route against the unquantized K/V. The mutated
        caches go through the fused route and must differ from the FP16 kernel.
        """
        ks, vs = self.k_scale, self.v_scale

        def fused(selection: Selection, kv: torch.Tensor, **scales) -> torch.Tensor:
            return self._attention(
                selection,
                k_cache=kv[:, 0],
                v_cache=kv[:, 1],
                kv_cache_dtype="nvfp4",
                nvfp4_fused_reader=True,
                **scales,
            ).cpu()

        realistic = fused(clean, cache, k_scale=ks, v_scale=vs)
        unit_out = fused(sel, unit_cache)
        controls = {
            mutation: {
                "decode": error_stats(
                    fused(
                        sel,
                        self._dev(mutate_cache(unit, mutation, nv.nvfp4_kv_split_views)),
                    ),
                    fp16_unit_out,
                )
            }
            for mutation in MUTATIONS
        }
        return {
            "vs_gather": error_stats(realistic, gather_quantized),
            "vs_gather_illegal": error_stats(unit_out, gather_unit_out),
            "quality": error_stats(realistic, baseline),
            "zero_fill_ok": all(not unit_out[row].any() for row in sel.empty_rows),
            "controls": controls,
        }

    # -- the prefill scratch route (plan 071 option B') --------------------
    def _prefill_call(self, rows: slice = slice(None), **kwargs) -> torch.Tensor:
        """``qsa_sparse_paged_attention`` on the prefill chunk (``rows`` of it)."""
        p = self.prefill
        assert p is not None
        return self.k["qsa"].qsa_sparse_paged_attention(
            self._dev(self.q[rows]),
            kwargs.pop("k_cache"),
            kwargs.pop("v_cache"),
            self._dev(p.indices[rows]),
            self._dev(p.table),
            self._dev(p.token_to_req[rows]),
            query_positions=self._dev(p.positions[rows]),
            sequence_lengths=self._dev(p.seq_lens),
            max_sequence_length=int(p.seq_lens[0]),
            **kwargs,
        )

    def _scratch_geometry(self) -> tuple[int, int, int]:
        p = self.prefill
        assert p is not None
        seq_len = int(p.seq_lens[0])
        return seq_len, self.layout.block_size, -(-seq_len // self.layout.block_size)

    def _decode_prefix(self, nvt, qn, cache: torch.Tensor):
        """Scratch decode of request 0's whole prefix; returns the CPU K and V."""
        p, g = self.prefill, self.geometry
        assert p is not None
        seq_len, block, pages = self._scratch_geometry()
        table, lens = self._dev(p.table), self._dev(p.seq_lens)
        _, offsets = qn.nvfp4_prefix_scratch_table(table, lens, block, pages)
        shape = (pages, block, g.kv_heads, g.head_dim)
        # A sentinel, not zeros: a slot the kernel forgets to write must show.
        scratch_k = torch.full(shape, 7.0, dtype=torch.float16, device=self.device)
        scratch_v = torch.full(shape, 7.0, dtype=torch.float16, device=self.device)
        nvt.dequant_nvfp4_prefix_triton(
            cache[:, 0],
            cache[:, 1],
            table,
            lens,
            offsets,
            scratch_k,
            scratch_v,
            max_pages=pages,
        )
        _sync()
        return scratch_k.cpu(), scratch_v.cpu()

    def _compare_scratch(self, nv, reference: torch.Tensor, scratch) -> dict[str, Any]:
        """Bit comparison with the gather at unit layer scale; the tail must be +0.0."""
        p, g = self.prefill, self.geometry
        assert p is not None
        seq_len, block, pages = self._scratch_geometry()
        want_k, want_v = nv.gather_dequant_nvfp4_kv(
            reference,
            p.table,
            torch.zeros(1, dtype=torch.int32),
            torch.arange(seq_len, dtype=torch.int32).view(1, -1),
            k_scale=1.0,
            v_scale=1.0,
        )
        different = 0
        tail_ok = True
        for got, want in zip(scratch, (want_k, want_v), strict=True):
            flat = got.reshape(pages * block, g.kv_heads, g.head_dim)
            different += int(
                (flat[:seq_len].view(torch.int16) != want[0].view(torch.int16)).sum()
            )
            tail_ok &= not bool(flat[seq_len:].view(torch.int16).any())
        return {
            "elements": 2 * seq_len * g.kv_heads * g.head_dim,
            "different": different,
            "tail_zero_ok": tail_ok,
        }

    def _route_reason(self) -> str | None:
        """Why the CUDA part of the route cannot run here, or None."""
        if self.device.type != "cuda":
            return "not a CUDA device: the grouped page4 route is a V100 kernel"
        if self.case.rows < PREFILL_ROUTE_MIN_ROWS:
            return (
                f"{self.case.rows} rows (the route starts at {PREFILL_ROUTE_MIN_ROWS})"
            )
        return self.k["qsa"]._qsa_nvfp4_prefill_extension_reason()

    def _install_scratch(self):
        """This case's scratch as the device's, reserving it on first use."""
        qn, g = self.k["qn"], self.geometry
        seq_len, block, _ = self._scratch_geometry()
        if getattr(self, "_scratch", None) is None:
            self._scratch = qn.ensure_nvfp4_prefill_scratch(
                self.device,
                block_size=block,
                num_kv_heads=g.kv_heads,
                head_size=g.head_dim,
                capacity_tokens=seq_len,
            )
        qn.install_nvfp4_prefill_scratch(self.device, self._scratch)

    def _route_call(self, cache: torch.Tensor) -> torch.Tensor | None:
        """The whole scratch route on the chunk (``None`` when it declines)."""
        p = self.prefill
        assert p is not None
        self._install_scratch()
        q = self._dev(self.q)
        return self.k["qn"].qsa_sparse_attention_nvfp4_prefill(
            q,
            cache[:, 0],
            cache[:, 1],
            self._dev(p.indices),
            self._dev(p.table),
            self._dev(p.token_to_req),
            torch.empty_like(q),
            None,
            self.k_scale,
            self.v_scale,
            query_positions=self._dev(p.positions),
            sequence_lengths=self._dev(p.seq_lens),
            max_sequence_length=int(p.seq_lens[0]),
        )

    def _run_prefill_scratch(self, nv, nvt, reference, cache) -> dict[str, Any]:
        qn = self.k["qn"]
        out: dict[str, Any] = {
            "decode": self._compare_scratch(
                nv, reference, self._decode_prefix(nvt, qn, cache)
            )
        }
        out["controls"] = {
            mutation: {
                "decode": self._compare_scratch(
                    nv,
                    reference,
                    self._decode_prefix(
                        nvt,
                        qn,
                        self._dev(
                            mutate_cache(reference, mutation, nv.nvfp4_kv_split_views)
                        ),
                    ),
                )
            }
            for mutation in MUTATIONS
        }
        reason = self._route_reason()
        if reason is not None:
            # A chunk too short for the CUDA route is not a failure of the route: the
            # self-check on a GPU only asks the route of the cases that can run it.
            out["route"] = {
                "skipped": reason,
                "not_applicable": self.case.rows < PREFILL_ROUTE_MIN_ROWS,
            }
            return out
        routed = self._route_call(cache)
        if routed is None:
            out["route"] = {"skipped": "the route declined the chunk (see the log)"}
            return out
        scales = {"k_scale": self.k_scale, "v_scale": self.v_scale}
        fused = self._prefill_call(
            k_cache=cache[:, 0],
            v_cache=cache[:, 1],
            kv_cache_dtype="nvfp4",
            nvfp4_fused_reader=True,
            **scales,
        )
        subset = slice(0, min(self.case.rows, PREFILL_VS_GATHER_ROWS))
        gathered = self._prefill_call(
            subset,
            k_cache=cache[:, 0],
            v_cache=cache[:, 1],
            kv_cache_dtype="nvfp4",
            nvfp4_fused_reader=False,
            **scales,
        )
        out["route"] = {
            "vs_fused": error_stats(routed.cpu(), fused.cpu()),
            "vs_gather": error_stats(routed[subset].cpu(), gathered.cpu()),
        }
        return out

    # -- the E4M3 control ------------------------------------------------
    def run_e4m3(self) -> dict[str, Any]:
        g, layout = self.geometry, self.layout
        shape = (layout.num_blocks, layout.block_size, g.kv_heads, g.head_dim)
        ks = float(self.key.abs().max()) / E4M3_MAX
        vs = float(self.value.abs().max()) / E4M3_MAX
        k_cache = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        v_cache = torch.zeros_like(k_cache)
        self._e4m3_store(
            k_cache,
            v_cache,
            self._dev(self.key),
            self._dev(self.value),
            self._dev(self.slots),
            torch.tensor(ks, device=self.device),
            torch.tensor(vs, device=self.device),
        )
        _sync()
        clean = self.clean
        k_orig, v_orig = self._fp16_cache(self.key, self.value)
        baseline = self._attention(clean, k_cache=k_orig, v_cache=v_orig).cpu()
        e4m3 = self._attention(
            clean,
            k_cache=k_cache,
            v_cache=v_cache,
            kv_cache_dtype="fp8_e4m3",
            k_scale=ks,
            v_scale=vs,
        ).cpu()
        self._e4m3 = (k_cache, v_cache, ks, vs)
        quality = error_stats(e4m3, baseline)
        # What reshape_and_cache_flash does per element: scale, clamp, cast to E4M3.
        k_model = (self.key.float() / ks).clamp(-E4M3_MAX, E4M3_MAX)
        v_model = (self.value.float() / vs).clamp(-E4M3_MAX, E4M3_MAX)
        k_model = k_model.to(torch.float8_e4m3fn).float() * ks
        v_model = v_model.to(torch.float8_e4m3fn).float() * vs
        return {
            "decode_quality": quality,
            "model_check": self._model_check(k_model, v_model, e4m3, baseline),
        }

    def _model_check(
        self,
        k_model: torch.Tensor,
        v_model: torch.Tensor,
        measured_out: torch.Tensor,
        baseline_out: torch.Tensor,
    ) -> dict[str, float]:
        """The CPU model's relative L2 for these dequantized K/V, and the measured.

        Both are taken over the same rows (``self.checked``; every row of a batch up
        to ``--ref-rows``, where this equals ``decode_quality``'s relative L2).
        """
        tpr = self.layout.tokens_per_request
        every = len(self.checked) == self.case.rows
        rows = None if every else self.checked
        reference = model_attention(
            self.q, self.key, self.value, self.clean, tpr, rows
        )
        quantized = model_attention(self.q, k_model, v_model, self.clean, tpr, rows)
        if every:
            measured = error_stats(measured_out, baseline_out)["rel_l2"]
        else:
            measured = error_stats(measured_out[rows], baseline_out[rows])["rel_l2"]
        return {
            "model_rel_l2": error_stats(quantized, reference)["rel_l2"],
            "measured_rel_l2": measured,
        }

    # -- the CPU reference of the gather, a chunk of rows at a time ----------
    def _ref_gather(self, nv, reference: torch.Tensor, sel: Selection):
        """The reference gather's FP16 K and V ``[len(checked), topk, heads, d]``."""
        g = self.geometry
        rows = self.checked
        shape = (len(rows), g.topk, g.kv_heads, g.head_dim)
        keys = torch.empty(shape, dtype=torch.float16)
        values = torch.empty(shape, dtype=torch.float16)
        for lo in range(0, len(rows), self.ref_chunk):
            part = torch.tensor(rows[lo : lo + self.ref_chunk])
            k, v = nv.gather_dequant_nvfp4_kv(
                reference,
                sel.table,
                sel.token_to_req[part],
                sel.indices[part],
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
            keys[lo : lo + len(part)] = k
            values[lo : lo + len(part)] = v
            del k, v
        return keys, values

    def _gather_diff(
        self,
        got_k: torch.Tensor,
        got_v: torch.Tensor,
        ref_k: torch.Tensor,
        ref_v: torch.Tensor,
    ) -> dict[str, Any]:
        """Elements that differ, elements compared and max |diff| of a device gather
        output (all rows) against the reference of ``self.checked`` rows."""
        different = elements = 0
        max_abs = 0.0
        rows = self.checked
        for lo in range(0, len(rows), self.ref_chunk):
            part = torch.tensor(rows[lo : lo + self.ref_chunk], device=got_k.device)
            for got, ref in ((got_k, ref_k), (got_v, ref_v)):
                chunk = got.index_select(0, part).cpu()
                want = ref[lo : lo + len(part)]
                different += int((chunk != want).sum())
                elements += chunk.numel()
                largest = float((chunk.float() - want.float()).abs().max())
                if math.isnan(largest):
                    max_abs = largest
                elif not math.isnan(max_abs) and largest > max_abs:
                    max_abs = largest
                del chunk
        return {"different": different, "elements": elements, "max_abs": max_abs}

    def _zero_fill_ok(
        self, got_k: torch.Tensor, got_v: torch.Tensor, illegal: torch.Tensor
    ) -> bool:
        """Is every illegal entry of the kernel's output exactly zero (all rows)?"""
        ok = True
        for lo in range(0, got_k.shape[0], self.ref_chunk):
            legal = ~self._dev(illegal[lo : lo + self.ref_chunk])[..., None, None]
            for got in (got_k, got_v):
                ok &= bool((got[lo : lo + self.ref_chunk].masked_fill(legal, 0) == 0).all())
        return ok

    def _tie_control(self, nv, nvt) -> dict[str, Any]:
        """The store on engineered ties against the reference and two tie mutants."""
        g, layout = self.geometry, self.layout
        tokens = min(TIE_TOKENS, layout.block_size)
        shape = (
            1,
            2,
            layout.block_size,
            g.kv_heads,
            nv.nvfp4_kv_row_bytes(g.head_dim),
        )
        key = make_tie_kv(tokens, g.kv_heads, g.head_dim)
        value = make_tie_kv(tokens, g.kv_heads, g.head_dim, shift=1)
        slots = torch.arange(tokens)

        def reference_cache() -> torch.Tensor:
            cache = torch.zeros(shape, dtype=torch.uint8)
            nv.reshape_and_cache_nvfp4_reference(key, value, cache, slots)
            return cache

        correct = reference_cache()
        rne_encode, decode = nv.e4m3_encode, nv.e4m3_decode
        with patched(nv, e2m1_encode=e2m1_encode_half_up):
            e2m1_mutant = reference_cache()
        with patched(
            nv, e4m3_encode=lambda x: e4m3_encode_half_up(x, rne_encode, decode)
        ):
            e4m3_mutant = reference_cache()
        actual = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        nvt.store_nvfp4_kv_triton(
            self._dev(key), self._dev(value), actual, self._dev(slots)
        )
        _sync()
        act_views = nv.nvfp4_kv_split_views(actual.cpu())

        def against_actual(cache: torch.Tensor) -> dict[str, Any]:
            views = nv.nvfp4_kv_split_views(cache)
            return merge_region_comparisons(
                [
                    compare_cache_regions(
                        (views[0][side], views[1][side]),
                        (act_views[0][side], act_views[1][side]),
                    )
                    for side in (0, 1)
                ]
            )

        return {
            "correct": against_actual(correct),
            "e2m1_mutant": against_actual(e2m1_mutant),
            "e4m3_mutant": against_actual(e4m3_mutant),
        }

    def _e4m3_store(self, k_cache, v_cache, key, value, slots, ks, vs) -> None:
        """Write device tensors ``key`` / ``value`` at ``slots`` as E4M3 bytes."""
        if self.device.type == "cuda":
            self.k["ops"].reshape_and_cache_flash(
                key, value, k_cache, v_cache, slots, "fp8_e4m3", ks, vs
            )
            return
        # CPU (the tests): the same conversion the CUDA kernel performs.
        flat_k = k_cache.view(-1, *k_cache.shape[2:])
        flat_v = v_cache.view(-1, *v_cache.shape[2:])
        flat_k[slots] = (
            (key.float() / float(ks)).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        flat_v[slots] = (
            (value.float() / float(vs)).to(torch.float8_e4m3fn).view(torch.uint8)
        )

    # -- timing closures ---------------------------------------------------
    def timing_arms(self) -> dict[str, Callable[[], Any]]:
        """Closures to time; the case must have been verified (caches are built).

        Every tensor is moved to the device once, so a timed call is the kernel.
        """
        rows = self.case.rows
        label = f"B{self.case.block_size}"
        arms: dict[str, Callable[[], Any]] = {}
        sel = self.clean
        indices = self._dev(sel.indices)
        table = self._dev(sel.table)
        token_to_req = self._dev(sel.token_to_req)
        q = self._dev(self.q)
        slots = self._dev(self.slots[:rows])
        key = self._dev(self.key[:rows])
        value = self._dev(self.value[:rows])
        attention = self.k["qsa"].qsa_sparse_paged_attention
        if self.case.kind == "nvfp4":
            nvt = self.k["nvt"]
            cache = self._cache
            scratch = torch.zeros_like(cache)
            ks_t = torch.tensor(self.k_scale, device=self.device)
            vs_t = torch.tensor(self.v_scale, device=self.device)
            ks, vs = self.k_scale, self.v_scale
            k16, v16 = self._k_deq, self._v_deq
            arms[f"store_nvfp4_{label}"] = lambda: nvt.store_nvfp4_kv_triton(
                key, value, scratch, slots, k_scale=ks_t, v_scale=vs_t
            )
            arms[f"gather_nvfp4_{label}"] = lambda: nvt.gather_dequant_nvfp4_kv_triton(
                cache, table, token_to_req, indices, k_scale=ks, v_scale=vs
            )
            arms[f"decode_nvfp4_{label}"] = lambda: attention(
                q,
                cache[:, 0],
                cache[:, 1],
                indices,
                table,
                token_to_req,
                kv_cache_dtype="nvfp4",
                k_scale=ks,
                v_scale=vs,
            )
            arms[f"decode_fp16_dequantized_{label}"] = lambda: attention(
                q, k16, v16, indices, table, token_to_req
            )
            if self.fused:
                arms[f"decode_fused_nvfp4_{label}"] = lambda: attention(
                    q,
                    cache[:, 0],
                    cache[:, 1],
                    indices,
                    table,
                    token_to_req,
                    kv_cache_dtype="nvfp4",
                    k_scale=ks,
                    v_scale=vs,
                    nvfp4_fused_reader=True,
                )
            if self.prefill is not None:
                arms.update(self._prefill_timing_arms(label, cache, ks, vs))
        else:
            k_cache, v_cache, ks, vs = self._e4m3
            scratch_k, scratch_v = torch.zeros_like(k_cache), torch.zeros_like(v_cache)
            ks_t = torch.tensor(ks, device=self.device)
            vs_t = torch.tensor(vs, device=self.device)
            arms[f"store_e4m3_{label}"] = lambda: self._e4m3_store(
                scratch_k, scratch_v, key, value, slots, ks_t, vs_t
            )
            arms[f"decode_e4m3_{label}"] = lambda: attention(
                q,
                k_cache,
                v_cache,
                indices,
                table,
                token_to_req,
                kv_cache_dtype="fp8_e4m3",
                k_scale=ks,
                v_scale=vs,
            )
            if self.prefill is not None and self._route_reason() is None:
                # The same chunk shape through the E4M3 grouped page4 route: the
                # number the scratch route is measured against.
                p = self.prefill
                chunk = (
                    self._dev(p.indices),
                    self._dev(p.table),
                    self._dev(p.token_to_req),
                )
                positions, seq_lens = self._dev(p.positions), self._dev(p.seq_lens)
                arms[f"prefill_e4m3_{label}"] = lambda: attention(
                    q,
                    k_cache,
                    v_cache,
                    *chunk,
                    query_positions=positions,
                    sequence_lengths=seq_lens,
                    kv_cache_dtype="fp8_e4m3",
                    k_scale=ks,
                    v_scale=vs,
                )
        return arms

    def _prefill_timing_arms(
        self, label: str, cache: torch.Tensor, ks: float, vs: float
    ) -> dict[str, Callable[[], Any]]:
        """Closures of the prefill chunk: scratch decode, whole route, fused reader."""
        p, qn, nvt = self.prefill, self.k["qn"], self.k["nvt"]
        assert p is not None
        _, block, pages = self._scratch_geometry()
        table, lens = self._dev(p.table), self._dev(p.seq_lens)
        scratch = tuple(
            torch.zeros(
                (pages, block, self.geometry.kv_heads, self.geometry.head_dim),
                dtype=torch.float16,
                device=self.device,
            )
            for _ in "kv"
        )

        def decode_only() -> None:
            _, offsets = qn.nvfp4_prefix_scratch_table(table, lens, block, pages)
            nvt.dequant_nvfp4_prefix_triton(
                cache[:, 0],
                cache[:, 1],
                table,
                lens,
                offsets,
                scratch[0],
                scratch[1],
                max_pages=pages,
            )

        arms: dict[str, Callable[[], Any]] = {
            f"prefill_scratch_decode_nvfp4_{label}": decode_only,
            f"prefill_fused_nvfp4_{label}": lambda: self._prefill_call(
                k_cache=cache[:, 0],
                v_cache=cache[:, 1],
                kv_cache_dtype="nvfp4",
                nvfp4_fused_reader=True,
                k_scale=ks,
                v_scale=vs,
            ),
        }
        if self._route_reason() is None:
            arms[f"prefill_scratch_route_nvfp4_{label}"] = lambda: self._route_call(
                cache
            )
        return arms


def run_all(
    cases: Sequence[Case],
    geometry: Geometry,
    kernels: dict[str, Any],
    args: argparse.Namespace,
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Verify every case, then time the arms of each row count together.

    The route knobs are pinned (``ROUTE_KNOBS``) for the whole run, timing included.
    """
    ambient = {k: v for k, v in ambient_route_knobs().items() if v is not None}
    if ambient:
        log(f"ROUTE_KNOBS ambient {ambient} overridden by {ROUTE_KNOBS}")
    with pinned_route_knobs():
        return _run_all_pinned(cases, geometry, kernels, args, log)


def _run_all_pinned(
    cases: Sequence[Case],
    geometry: Geometry,
    kernels: dict[str, Any],
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    runs: dict[str, CaseRun] = {}
    ref_rows = getattr(args, "ref_rows", DEFAULT_REF_ROWS)
    ref_chunk = getattr(args, "ref_chunk", REF_CHUNK_ROWS)
    max_host_gb = getattr(args, "max_host_gb", DEFAULT_MAX_HOST_GB)
    prefill_arm = "prefill_scratch" in getattr(args, "arms", ("gather",))
    for case in cases:
        entry: dict[str, Any] = {
            "case": case.label,
            "kind": case.kind,
            "rows": case.rows,
        }
        checked = len(reference_rows(case.rows, ref_rows))
        entry["ref_rows"] = {"checked": checked, "total": case.rows}
        model = host_mem_model(
            case.kind,
            case.rows,
            geometry,
            case.block_size,
            args.context,
            ref_rows=ref_rows,
            ref_chunk=ref_chunk,
            prefill=prefill_arm,
        )
        resident = process_rss_bytes()
        estimate = resident + model["peak"]
        entry["host_mem_est"] = {
            "rows": case.rows,
            "est_peak_gib": round(estimate / GIB, 3),
            "process_rss_gib": round(resident / GIB, 3),
            "model_gib": {k: round(v / GIB, 4) for k, v in model.items()},
            "max_host_gb": max_host_gb,
        }
        log(
            f"HOST_MEM_EST: rows={case.rows} est_peak_GB={estimate / GIB:.2f} "
            f"case={case.label} limit_GB={max_host_gb:g} (process {resident / GIB:.2f}"
            f" + case {model['peak'] / GIB:.2f}) reference_rows={checked}/{case.rows}"
        )
        if max_host_gb > 0 and estimate > max_host_gb * GIB:
            entry["skipped"] = (
                f"estimated host peak {estimate / GIB:.1f} GB exceeds --max-host-gb "
                f"{max_host_gb:g}; lower --ref-rows or raise the limit"
            )
            log(f"SKIPPED {case.label}: {entry['skipped']}")
            results.append(entry)
            continue
        try:
            run = CaseRun(
                case,
                geometry,
                kernels,
                seed=args.seed,
                context=args.context,
                k_scale=args.k_scale,
                v_scale=args.v_scale,
                fused="fused" in getattr(args, "arms", ("gather",)),
                prefill_scratch=prefill_arm,
                ref_rows=ref_rows,
                ref_chunk=ref_chunk,
            )
            entry.update(run.run_nvfp4() if case.kind == "nvfp4" else run.run_e4m3())
            runs[case.label] = run
            entry["host_mem_est"]["measured_vmhwm_gib"] = round(
                process_rss_bytes("VmHWM") / GIB, 3
            )
            log(f"verified {case.label}")
        except Exception as exc:  # noqa: BLE001 - reported, the other cases still run
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log(f"FAILED {case.label}: {entry['error']}")
        results.append(entry)

    timings: dict[str, dict[str, Any]] = {}
    if not args.no_timing:
        for rows in sorted({c.rows for c in cases}):
            arms: dict[str, Callable[[], Any]] = {}
            for case in cases:
                if case.rows == rows and case.label in runs:
                    arms.update(runs[case.label].timing_arms())
            if not arms:
                continue
            try:
                timings[f"M={rows}"] = time_arms(
                    arms, rounds=args.rounds, iters=args.iters, warmup=args.warmup
                )
            except Exception as exc:  # noqa: BLE001
                log(f"TIMING FAILED M={rows}: {type(exc).__name__}: {exc}")
                timings[f"M={rows}"] = {}
    return results, timings


# ------------------------------------------------------------------ driver


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--head-dim", dest="head_dim", type=int, default=None)
    parser.add_argument("--q-heads", dest="q_heads", type=int, default=None)
    parser.add_argument("--kv-heads", dest="kv_heads", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--nvfp4-blocks", type=int, nargs="+", default=list(DEFAULT_NVFP4_BLOCKS)
    )
    parser.add_argument("--e4m3-block", type=int, default=DEFAULT_E4M3_BLOCK)
    parser.add_argument("--no-e4m3", action="store_true", help="skip the E4M3 control")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(DEFAULT_ARMS),
        help="NVFP4 routes to verify and time: the gather route (production), the "
        "fused reader and/or the prefill scratch route (plan 071 option B'; needs a "
        "prefill-sized --rows); `--arms gather` is the run without the fused reader",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument(
        "--ref-rows",
        type=int,
        default=DEFAULT_REF_ROWS,
        help="rows per case that get the CPU reference and the CPU attention model "
        "(all rows of a batch this small; a deterministic subset of this many rows "
        "otherwise, 0 = every row). The kernels, the zero-fill check and the timings "
        "always run on every row. The reference costs ~19 KiB of host RAM per row",
    )
    parser.add_argument(
        "--ref-chunk",
        type=int,
        default=REF_CHUNK_ROWS,
        help="rows per step of the CPU reference (host RAM is linear in this)",
    )
    parser.add_argument(
        "--max-host-gb",
        type=float,
        default=DEFAULT_MAX_HOST_GB,
        help="skip (SKIPPED line, not a crash) a case whose estimated host peak, "
        "this process's RSS plus the case's model, exceeds this many GiB; 0 = no limit",
    )
    parser.add_argument("--context", type=int, default=8192, help="tokens per request")
    parser.add_argument("--k-scale", type=float, default=DEFAULT_K_SCALE)
    parser.add_argument("--v-scale", type=float, default=DEFAULT_V_SCALE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50, help="calls per timing turn")
    parser.add_argument("--rounds", type=int, default=4, help="ABBA timing rounds")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--out", help="write the JSON report here")
    args = parser.parse_args(argv)
    if args.iters <= 0 or args.rounds <= 0 or args.warmup < 0:
        parser.error("--iters and --rounds must be positive, --warmup non-negative")
    if args.context <= 0 or args.k_scale <= 0 or args.v_scale <= 0:
        parser.error("--context and the layer scales must be positive")
    if args.ref_rows < 0 or args.ref_chunk <= 0 or args.max_host_gb < 0:
        parser.error(
            "--ref-rows and --max-host-gb must not be negative, --ref-chunk positive"
        )
    return args


def refuse_busy_gpu(max_used_mib: int) -> str | None:
    """A message if the GPU already holds more than ``max_used_mib``, else None."""
    free, total = torch.cuda.mem_get_info()
    used_mib = (total - free) / (1 << 20)
    if used_mib > max_used_mib:
        return (
            f"the GPU already holds {used_mib:.0f} MiB (limit {max_used_mib}); run "
            "on an idle GPU, never one a server holds, or pass --allow-busy"
        )
    return None


def device_info() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_mib": props.total_memory // (1 << 20),
        "torch": torch.__version__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("needs a CUDA device (set CUDA_VISIBLE_DEVICES to an idle GPU)")
        return 2
    if not args.allow_busy:
        message = refuse_busy_gpu(args.max_used_mib)
        if message:
            print(message)
            return 2
    try:
        geometry = resolve_geometry(args)
        cases = plan_cases(
            args.nvfp4_blocks, None if args.no_e4m3 else args.e4m3_block, args.rows
        )
    except ValueError as exc:
        print(f"unusable arguments: {exc}")
        return 2
    try:
        kernels = load_kernels()
    except ImportError as exc:
        print(
            f"cannot import the NVFP4 kernels ({exc}); the vllm on sys.path lacks "
            "them: "
            "run from a tree that has them with PYTHONPATH=<worktree>, or sync the "
            "venv to the p071 tip first"
        )
        return 2
    info = device_info()
    print(
        f"device {info['name']} capability {info['capability']}, torch {info['torch']}"
    )
    if info["capability"] != [7, 0]:
        print("NOTE: not an sm_70 device; the numbers are not V100 numbers")
    print(
        f"geometry head_dim={geometry.head_dim} q_heads={geometry.q_heads} "
        f"kv_heads={geometry.kv_heads} topk={geometry.topk} ({geometry.source})"
    )
    for name in ("nv", "nvt", "qsa"):
        print(f"SOURCE {name}: {kernels[name].__file__}")
    print(f"cases: {', '.join(c.label for c in cases)}")

    with torch.inference_mode():
        results, timings = run_all(cases, geometry, kernels, args)
    verdict = compute_verdict(results)
    check = compute_self_check(
        results,
        timings,
        no_timing=args.no_timing,
        fused_arm="fused" in args.arms,
        prefill_arm="prefill_scratch" in args.arms,
        prefill_route_required=WORKER_DEVICE.startswith("cuda"),
    )
    lines = format_verdict(verdict, results)
    for group, arms in timings.items():
        for arm, summary in arms.items():
            lines.append(
                f"TIMING_US {group} {arm}: median {summary['median']:.1f} "
                f"p10 {summary['p10']:.1f} p90 {summary['p90']:.1f} "
                f"(round spread {summary['round_spread_pct']:.1f}%)"
            )
    lines += decode_roundtrip_lines(timings)
    lines += format_self_check(check, results)
    print("\n".join(lines))
    if args.out:
        report = {
            "device": info,
            "geometry": asdict(geometry),
            "args": vars(args),
            "route_knobs": {"pinned": ROUTE_KNOBS, "ambient": ambient_route_knobs()},
            "cases": results,
            "timings": timings,
            "verdict": verdict,
            "self_check": check,
            "lines": lines,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")
    return max(exit_code_for(results), exit_code_for_self_check(check))


if __name__ == "__main__":
    sys.exit(main())
