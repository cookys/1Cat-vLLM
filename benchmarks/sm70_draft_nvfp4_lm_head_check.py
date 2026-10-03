# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check for the draft-only NVFP4 lm_head (VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD).

Loads lm_head rows from the Flash-Next checkpoint, quantizes them with the
production helper, runs the real QPN2 ops and the real rerank, and reports:

* the weight and logits relative L2 error (NVFP4 vs the FP16 GEMM),
* a packing-convention check of the real prepare+GEMM kernels,
* ``disagreement_rate`` per rerank size R: the fraction of hidden-state rows
  whose final draft token of the NVFP4+rerank path differs from the FP16 path
  (R=0 is the raw NVFP4 argmax, tier-3 diagnostic),
* the rank histogram of the FP16 argmax inside its own shard's NVFP4 logits,
* CUDA-event timings (eager and CUDA-graph replay) of the FP16 GEMM, the QPN2
  GEMM for split_k in {8, 16, 32} and the whole head, for M in {1, 2, 4},
* resident bytes, quantize time and peak extra memory of building the head.

Scope.  ``--tp-rank <r>`` loads one shard (about 1 GB of VRAM); the final token
is then the shard-local winner.  ``--tp-rank all`` loads all ``--tp-size``
shards on the one GPU (about 2 GB) and reproduces the production TP reduction,
so the disagreement is that of the real global draft token.  A single-shard,
shard-local comparison is a proxy for the TP4 global token: the global winner is
the largest of the shard-local winners, so it can differ between the two paths
only if some shard's local winner differs (or, rarely, when two shards' top
values sit within one FP16 rounding step of each other).

``fp16_near_tie_rate`` is the share of rows whose top-2 FP16 logits differ by
less than one FP16 ulp of the maximum (exact ties included).  On those rows the
FP16 argmax itself is a coin flip, so a disagreement there is FP16 rounding, not
a 4-bit effect; ``disagreement_rate_not_near_tie`` is the disagreement over the
remaining rows.  Padding rows of a shard are masked in both paths.

Hidden states.  ``--hidden-dump`` takes (a) a directory written by
``VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR`` (files ``rank*_step*.pt``, dicts with a
``hidden`` tensor; the files are filtered by ``--tp-rank``, i.e.
``rank<r>_step*.pt`` (``rank0_step*.pt`` for ``--tp-rank all``), because the
hidden states are replicated across TP ranks; ``--hidden-glob`` overrides),
(b) one such file, or (c) a plain ``torch.save``'d float16 tensor [n, 2560].
Without it the rows are random N(0, --sigma), which makes the numbers
pessimistic: random vectors have no dominant token.

    CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/sm70_draft_nvfp4_lm_head_check.py \
        --tp-rank all --hidden-dump /data/bench/draft_hidden \
        --out /data/bench/draft_nvfp4_lm_head.json

It needs one idle SM70 GPU and under a minute; pick it with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm._C  # noqa: F401  (registers torch.ops._C.*)

from vllm.v1.worker.gpu.spec_decode.eagle import draft_nvfp4_lm_head as dh

HIDDEN_SIZE = 2560
RANK_EDGES = (1, 2, 4, 8, 16, 32, 64, 128)
RANK_LABELS = ("1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/data/models/Qwen3.8-Flash-Next-NVFP4")
    )
    parser.add_argument(
        "--tp-rank",
        type=str,
        default="0",
        help='shard index, or "all" to emulate the whole TP reduction on one GPU',
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument(
        "--hidden-dump",
        type=Path,
        default=None,
        help="dump directory / file from VLLM_SM70_MTP_DRAFT_HIDDEN_DUMP_DIR, or a "
        "torch.save'd float16 tensor [n, 2560]",
    )
    parser.add_argument(
        "--hidden-glob",
        type=str,
        default=None,
        help="files of a --hidden-dump directory; default rank<tp-rank>_step*.pt "
        "(rank0_step*.pt for --tp-rank all)",
    )
    parser.add_argument(
        "--max-hidden-rows", type=int, default=0, help="cap on rows used (0 = all)"
    )
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=str, default="0,8,32,64")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("draft_nvfp4_lm_head.json"))
    return parser.parse_args()


def load_shard(model_dir: Path, tp_rank: int, tp_size: int) -> tuple[torch.Tensor, int]:
    """Rows of ``lm_head.weight`` owned by ``tp_rank`` (float16, CPU), start row."""
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_file = model_dir / index["weight_map"]["lm_head.weight"]
    with safe_open(str(shard_file), framework="pt", device="cpu") as handle:
        weight = handle.get_slice("lm_head.weight")
        vocab, hidden = weight.get_shape()
        if vocab % tp_size or hidden != HIDDEN_SIZE:
            raise SystemExit(f"unexpected lm_head shape {(vocab, hidden)}")
        rows = vocab // tp_size
        start = tp_rank * rows
        shard = weight[start : start + rows, :]
    return shard.to(torch.float16), start


def load_hidden_dump(path: Path, glob: str) -> tuple[torch.Tensor, int]:
    """Hidden states [n, 2560] float16 (CPU) from a dump directory, file or tensor."""
    files = sorted(path.glob(glob)) if path.is_dir() else [path]
    if not files:
        raise SystemExit(
            f"no files matching {glob!r} in {path} (override with --hidden-glob)"
        )
    parts = []
    for file in files:
        loaded = torch.load(file, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded["hidden"]
        parts.append(loaded.reshape(-1, HIDDEN_SIZE).to(torch.float16))
    return torch.cat(parts), len(files)


def make_hidden(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, str, int]:
    files = 0
    if args.hidden_dump is not None:
        glob = args.hidden_glob or (
            "rank0_step*.pt"
            if args.tp_rank == "all"
            else f"rank{int(args.tp_rank)}_step*.pt"
        )
        hidden, files = load_hidden_dump(args.hidden_dump, glob)
        source = f"dump:{args.hidden_dump} ({files} file(s))"
    else:
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        hidden = torch.randn(args.num_hidden, HIDDEN_SIZE, generator=gen) * args.sigma
        hidden = hidden.to(torch.float16)
        source = f"random_normal_sigma={args.sigma}"
    if args.max_hidden_rows > 0:
        hidden = hidden[: args.max_hidden_rows]
    if hidden.shape[0] < 4:
        raise SystemExit("need at least 4 hidden-state rows")
    return hidden.to(device).contiguous(), source, files


@torch.no_grad()
def convention_check(weight: torch.Tensor, device: torch.device) -> dict[str, float]:
    """Real prepare+GEMM kernels vs a dequantized matmul on a 4096-row slice.

    A wrong nibble order or scale layout shows up here as a relative error of
    order one; a correct convention is limited by the FP16 output rounding.
    """
    sub = weight[:4096]
    packed, scales, g = dh.quantize_fp16_to_nvfp4_packed(sub)
    codes, prepared = dh._prepare_qpn2(packed, scales)
    gen = torch.Generator(device="cpu").manual_seed(123)
    hidden = torch.randn(4, HIDDEN_SIZE, generator=gen).to(torch.float16).to(device)
    out = torch.empty(4, sub.shape[0], dtype=torch.float16, device=device)
    split_k, chains = dh.default_split_config(HIDDEN_SIZE, sub.shape[0])
    dh._qpn2_gemm_out(out, hidden, codes, prepared, g, split_k, chains)
    reference = (
        hidden.float() @ dh.unpack_nvfp4_packed(packed, scales, g, kernel_exact=True).T
    )
    diff = (out.float() - reference).norm() / reference.norm()
    return {
        "rel_l2": float(diff),
        "max_abs": float((out.float() - reference).abs().max()),
    }


@torch.no_grad()
def weight_error(weight: torch.Tensor) -> float:
    packed, scales, g = dh.quantize_fp16_to_nvfp4_packed(weight)
    num = torch.zeros((), dtype=torch.float64, device=weight.device)
    den = torch.zeros_like(num)
    for begin in range(0, weight.shape[0], 8192):
        end = min(begin + 8192, weight.shape[0])
        deq = dh.unpack_nvfp4_packed(packed[begin:end], scales[begin:end], g)
        ref = weight[begin:end].float()
        num += ((deq - ref) ** 2).sum().double()
        den += (ref**2).sum().double()
    return float((num / den).sqrt())


def f16_ulp(value: torch.Tensor) -> torch.Tensor:
    """FP16 spacing at ``|value|``: 2**(floor(log2|v|) - 10), 2**-24 below 2**-14."""
    exponent = torch.floor(torch.log2(value.float().abs().clamp_min(2.0**-14)))
    return torch.exp2(exponent - 10)


@torch.no_grad()
def evaluate(
    heads: list[dh.DraftNvfp4LMHead],
    weights: list[torch.Tensor],
    hidden: torch.Tensor,
    ranks: list[int],
) -> dict:
    """FP16 path vs NVFP4+rerank path on every hidden-state row.

    With several heads the final token is reduced across them exactly like the
    TP all-gather does (largest value wins, first shard on a tie).
    """
    num_shards, n = len(heads), heads[0].rows
    starts = torch.tensor(
        [h.org_vocab_start_index for h in heads], device=hidden.device
    )
    disagree = dict.fromkeys(ranks, 0)
    sq_err = torch.zeros((), dtype=torch.float64, device=hidden.device)
    sq_ref = torch.zeros_like(sq_err)
    far_disagree = dict.fromkeys(ranks, 0)
    all_ranks, gaps, ties, near_ties, total = [], [], 0, 0, 0
    for begin in range(0, hidden.shape[0], dh.MAX_ROWS):
        h = hidden[begin : begin + dh.MAX_ROWS].contiguous()
        m = h.shape[0]
        rows = torch.arange(m, device=h.device)
        refs = [h @ w.t() for w in weights]  # FP16 GEMM: the target's numerics
        qs = []
        for head in heads:
            q = torch.empty(m, n, dtype=torch.float16, device=h.device)
            dh._qpn2_gemm_out(
                q,
                h,
                head.codes,
                head.scales,
                head.global_scale,
                head.split_k,
                head.accumulator_chains,
            )
            qs.append(q)
        for head, ref, q in zip(heads, refs, qs):
            valid = n - head.num_org_vocab_padding
            sq_err += (
                ((q[:, :valid].float() - ref[:, :valid].float()) ** 2).sum().double()
            )
            sq_ref += (ref[:, :valid].float() ** 2).sum().double()
            if head.num_org_vocab_padding:  # same padding mask as the head applies
                ref[:, valid:] = float("-inf")
                q[:, valid:] = float("-inf")

        full = torch.cat(refs, dim=1)
        top2 = full.float().topk(2, dim=-1).values
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
        ties += int((top2[:, 0] == top2[:, 1]).sum())
        near_tie = (top2[:, 0] - top2[:, 1]) < f16_ulp(top2[:, 0])
        near_ties += int(near_tie.sum())
        ref_index = full.argmax(dim=-1)  # index into the concatenated shards
        ref_shard, ref_local = ref_index // n, ref_index % n
        ref_token = ref_local + starts[ref_shard]

        # rank of the FP16 argmax inside its own shard's NVFP4 logits
        winner_q = torch.stack(qs)[ref_shard, rows]
        target = winner_q.gather(1, ref_local[:, None])
        all_ranks.append(((winner_q.float() > target.float()).sum(dim=-1) + 1).cpu())

        for r in ranks:
            values, ids = [], []
            for head in heads:
                head.rerank_k = r
                value, token = head.local_top_tokens(h)
                values.append(value)
                ids.append(token)
            winner = torch.stack(values).argmax(dim=0)
            token = torch.stack(ids).gather(0, winner[None])[0]
            differs = token != ref_token
            disagree[r] += int(differs.sum())
            far_disagree[r] += int((differs & ~near_tie).sum())
        total += m

    ranks_t = torch.cat(all_ranks)
    counts, previous = [], 0
    for edge in RANK_EDGES:
        counts.append(int((ranks_t <= edge).sum()) - previous)
        previous += counts[-1]
    counts.append(int(ranks_t.numel()) - previous)
    gaps_t = torch.cat(gaps)
    return {
        "scope": "global" if num_shards > 1 else "shard",
        "shards": num_shards,
        "rows": total,
        "logits_rel_l2": float((sq_err / sq_ref).sqrt()),
        "disagreement_rate": {str(r): disagree[r] / total for r in ranks},
        "disagreement_rows": {str(r): disagree[r] for r in ranks},
        "disagreement_rate_not_near_tie": {
            str(r): (
                far_disagree[r] / (total - near_ties) if total > near_ties else None
            )
            for r in ranks
        },
        "disagreement_rows_not_near_tie": {str(r): far_disagree[r] for r in ranks},
        "fp16_near_tie_rate": near_ties / total,
        "fp16_near_tie_rows": near_ties,
        "rank_hist": dict(zip(RANK_LABELS, counts)),
        "rank_p50": float(ranks_t.float().median()),
        "rank_p99": float(ranks_t.float().quantile(0.99)),
        "rank_max": int(ranks_t.max()),
        "fp16_top1_top2_gap_p10": float(gaps_t.quantile(0.10)),
        "fp16_top1_top2_gap_p50": float(gaps_t.quantile(0.50)),
        "fp16_exact_top1_ties": ties,
    }


def _time_events(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def time_call(fn, warmup: int, iters: int) -> dict[str, float | str | None]:
    """Average microseconds per call, eager and as a CUDA graph replay."""
    result: dict[str, float | str | None] = {
        "eager_us": _time_events(fn, warmup, iters)
    }
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        result["graph_us"] = _time_events(graph.replay, warmup, iters)
    except Exception as err:  # keep the eager numbers even if capture fails
        result["graph_us"] = None
        result["graph_error"] = f"{type(err).__name__}: {err}"
    return result


@torch.no_grad()
def _timings_for_rows(
    head: dh.DraftNvfp4LMHead,
    weight_t: torch.Tensor,
    h: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict:
    default_split, default_rerank = head.split_k, head.rerank_k
    buffer = torch.empty(h.shape[0], head.rows, dtype=torch.float16, device=h.device)
    entry: dict = {
        "f16_torch_mm": time_call(
            lambda: torch.mm(h, weight_t, out=buffer), warmup, iters
        )
    }
    for split_k in dh.SUPPORTED_SPLIT_K:
        entry[f"qpn2_gemm_split{split_k}"] = time_call(
            lambda sk=split_k: dh._qpn2_gemm_out(
                buffer,
                h,
                head.codes,
                head.scales,
                head.global_scale,
                sk,
                head.accumulator_chains,
            ),
            warmup,
            iters,
        )
        head.split_k = split_k
        for r in (0, default_rerank):
            head.rerank_k = r
            entry[f"head_split{split_k}_R{r}"] = time_call(
                lambda: head.local_top_tokens(h), warmup, iters
            )
    head.split_k, head.rerank_k = default_split, default_rerank
    return entry


def timings(
    head: dh.DraftNvfp4LMHead,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict:
    weight_t = weight.t()
    return {
        str(m): _timings_for_rows(
            head, weight_t, hidden[:m].contiguous(), warmup, iters
        )
        for m in (1, 2, 4)
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "needs a CUDA device (set CUDA_VISIBLE_DEVICES to an idle GPU)"
        )
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    torch.manual_seed(args.seed)
    ranks = sorted({int(r) for r in args.ranks.split(",")})
    if not set(ranks) <= set(dh.RERANK_K_CHOICES):
        raise SystemExit(f"--ranks must be a subset of {dh.RERANK_K_CHOICES}")
    shard_ids = (
        list(range(args.tp_size)) if args.tp_rank == "all" else [int(args.tp_rank)]
    )
    config = SimpleNamespace(method="mtp", draft_sample_method="greedy")
    name = torch.cuda.get_device_name(device)
    print(f"device {name} {capability}; shards {shard_ids} of tp_size={args.tp_size}")

    weights, heads, builds = [], [], []
    for shard in shard_ids:
        shard_cpu, start = load_shard(args.model_dir, shard, args.tp_size)
        weight = shard_cpu.to(device)
        del shard_cpu
        lm_head = SimpleNamespace(
            weight=weight,
            bias=None,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=start, num_org_vocab_padding=0
            ),
        )
        ok, reason = dh.DraftNvfp4LMHead.eligible(lm_head, config, device)
        print(f"shard {shard}: gate eligible={ok} ({reason})")
        if not ok:
            raise SystemExit(2)
        torch.accelerator.synchronize()
        allocated_before = torch.accelerator.memory_allocated(device)
        head = dh.DraftNvfp4LMHead(lm_head, rerank_k=64)
        torch.accelerator.synchronize()
        builds.append(
            {
                "shard": shard,
                "quantize_seconds": head.quantize_seconds,
                "peak_extra_bytes": head.peak_extra_bytes,
                "resident_bytes": head.resident_bytes(),
                "allocated_delta_bytes": torch.accelerator.memory_allocated(device)
                - allocated_before,
                "global_scale": head.global_scale,
            }
        )
        weights.append(weight)
        heads.append(head)

    hidden, hidden_source, hidden_files = make_hidden(args, device)
    print(f"hidden: {hidden_source} rows={hidden.shape[0]}")

    summary: dict = {
        "device": name,
        "capability": list(capability),
        "tp_size": args.tp_size,
        "shards": shard_ids,
        "rows_per_shard": heads[0].rows,
        "k": heads[0].hidden_size,
        "split_k": heads[0].split_k,
        "accumulator_chains": heads[0].accumulator_chains,
        "hidden_source": hidden_source,
        "hidden_files": hidden_files,
        "num_hidden": int(hidden.shape[0]),
        "builds": builds,
        "resident_bytes": builds[0]["resident_bytes"],
        "quantize_seconds": builds[0]["quantize_seconds"],
        "peak_extra_bytes": builds[0]["peak_extra_bytes"],
        "weight_rel_l2": weight_error(weights[0]),
        "convention_check": convention_check(weights[0], device),
    }
    print(
        f"per-rank resident={summary['resident_bytes'] / 1e6:.1f} MB "
        f"quantize={summary['quantize_seconds']:.2f}s "
        f"peak_extra={summary['peak_extra_bytes'] / 1e6:.1f} MB"
    )
    print(
        f"weight rel-L2={summary['weight_rel_l2']:.4f} "
        f"kernel-vs-dequant convention check={summary['convention_check']}"
    )

    summary["accuracy"] = evaluate(heads, weights, hidden, ranks)
    for head in heads:
        head.rerank_k = 64  # evaluate() leaves the last tested R behind
    acc = summary["accuracy"]
    print(f"logits rel-L2={acc['logits_rel_l2']:.4f} scope={acc['scope']}")
    print(f"disagreement_rate per R (NVFP4+rerank vs FP16): {acc['disagreement_rate']}")
    print(
        f"fp16_near_tie_rate={acc['fp16_near_tie_rate']:.4f}; disagreement_rate "
        f"excluding near ties: {acc['disagreement_rate_not_near_tie']}"
    )
    print(f"rank histogram of the FP16 argmax in NVFP4 logits: {acc['rank_hist']}")

    summary["timings_us"] = timings(
        heads[0], weights[0], hidden, args.warmup, args.iters
    )
    for m, entry in summary["timings_us"].items():
        line = {
            k: (
                round(v["graph_us"], 1)
                if v.get("graph_us")
                else round(v["eager_us"], 1)
            )
            for k, v in entry.items()
        }
        print(f"M={m} graph-replay us (eager if capture failed): {line}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
