# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check for the draft-only NVFP4 lm_head (VLLM_SM70_MTP_DRAFT_NVFP4_LM_HEAD).

Loads the rows of one tensor-parallel lm_head shard from the Flash-Next
checkpoint, quantizes them with the production helper, runs the real QPN2 ops
and reports:

* the weight and logits relative L2 error (NVFP4 vs the FP16 GEMM),
* a bit-convention check of the packing against the real kernel,
* top-1 agreement with the FP16 argmax for several rerank sizes R,
* the rank histogram of the FP16 argmax inside the NVFP4 logits,
* CUDA-event timings (eager and CUDA-graph replay) of the FP16 GEMM, the QPN2
  GEMM for split_k in {8, 16, 32} and the whole head, for M in {1, 2, 4},
* resident bytes and the allocator delta of building the head.

It needs one idle SM70 GPU (about 2 GB of VRAM, under a minute); pick it with
CUDA_VISIBLE_DEVICES. Random hidden states make the rank histogram pessimistic
(random vectors have no dominant token); use ``--hidden-dump`` with real draft
hidden states ([n, 2560] float16 saved with torch.save) for the numbers that
matter.

    CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/sm70_draft_nvfp4_lm_head_check.py \
        --tp-rank 0 --out /data/bench/draft_nvfp4_lm_head_rank0.json
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
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument(
        "--hidden-dump",
        type=Path,
        default=None,
        help="torch.save'd float16 tensor [n, 2560] of real draft hidden states",
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


def make_hidden(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, str]:
    if args.hidden_dump is not None:
        loaded = torch.load(args.hidden_dump, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded["hidden"]
        hidden = loaded.reshape(-1, HIDDEN_SIZE).to(torch.float16)
        return hidden.to(device).contiguous(), f"dump:{args.hidden_dump}"
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden = torch.randn(args.num_hidden, HIDDEN_SIZE, generator=gen) * args.sigma
    return hidden.to(torch.float16).to(
        device
    ).contiguous(), f"random_normal_sigma={args.sigma}"


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


@torch.no_grad()
def accuracy(
    head: dh.DraftNvfp4LMHead,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    ranks: list[int],
) -> dict:
    start = head.org_vocab_start_index
    agree = {r: 0 for r in ranks}
    sq_err = torch.zeros((), dtype=torch.float64, device=hidden.device)
    sq_ref = torch.zeros_like(sq_err)
    all_ranks, gaps, ties = [], [], 0
    rows_total = 0
    weight_t = weight.t()
    for begin in range(0, hidden.shape[0], dh.MAX_ROWS):
        h = hidden[begin : begin + dh.MAX_ROWS].contiguous()
        m = h.shape[0]
        ref = h @ weight_t  # FP16 GEMM, the numerics of the target lm_head
        ref_arg = ref.argmax(dim=-1)
        top2 = ref.float().topk(2, dim=-1).values
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
        ties += int((top2[:, 0] == top2[:, 1]).sum())
        q = torch.empty(m, head.rows, dtype=torch.float16, device=h.device)
        dh._qpn2_gemm_out(
            q,
            h,
            head.codes,
            head.scales,
            head.global_scale,
            head.split_k,
            head.accumulator_chains,
        )
        sq_err += ((q.float() - ref.float()) ** 2).sum().double()
        sq_ref += (ref.float() ** 2).sum().double()
        target = q.float().gather(1, ref_arg[:, None])
        all_ranks.append(((q.float() > target).sum(dim=-1) + 1).cpu())
        for r in ranks:
            head.rerank_k = r
            _, ids = head.local_top_tokens(h)
            agree[r] += int((ids - start == ref_arg).sum())
        rows_total += m
    ranks_t = torch.cat(all_ranks)
    counts, previous = [], 0
    for edge in RANK_EDGES:
        counts.append(int((ranks_t <= edge).sum()) - previous)
        previous += counts[-1]
    counts.append(int(ranks_t.numel()) - previous)
    gaps_t = torch.cat(gaps)
    return {
        "rows": rows_total,
        "logits_rel_l2": float((sq_err / sq_ref).sqrt()),
        "top1_agreement": {str(r): agree[r] / rows_total for r in ranks},
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
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
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

    shard_cpu, start = load_shard(args.model_dir, args.tp_rank, args.tp_size)
    weight = shard_cpu.to(device)
    del shard_cpu
    lm_head = SimpleNamespace(
        weight=weight,
        bias=None,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=0
        ),
    )
    config = SimpleNamespace(method="mtp", draft_sample_method="greedy")
    ok, reason = dh.DraftNvfp4LMHead.eligible(lm_head, config, device)
    name = torch.cuda.get_device_name(device)
    print(f"gate: eligible={ok} ({reason}); device {name} {capability}")
    if not ok:
        raise SystemExit(2)

    torch.accelerator.synchronize()
    allocated_before = torch.accelerator.memory_allocated(device)
    torch.accelerator.reset_peak_memory_stats(device)
    head = dh.DraftNvfp4LMHead(lm_head, rerank_k=64)
    torch.accelerator.synchronize()
    allocated_delta = torch.accelerator.memory_allocated(device) - allocated_before
    build_peak = torch.accelerator.max_memory_allocated(device) - allocated_before

    hidden, hidden_source = make_hidden(args, device)
    print(f"hidden: {hidden_source} shape={tuple(hidden.shape)}")

    summary: dict = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "tp_rank": args.tp_rank,
        "tp_size": args.tp_size,
        "rows": head.rows,
        "k": head.hidden_size,
        "global_scale": head.global_scale,
        "split_k": head.split_k,
        "accumulator_chains": head.accumulator_chains,
        "hidden_source": hidden_source,
        "num_hidden": int(hidden.shape[0]),
        "resident_bytes": head.resident_bytes(),
        "allocated_delta_bytes": allocated_delta,
        "build_peak_over_baseline_bytes": build_peak,
        "weight_rel_l2": weight_error(weight),
        "convention_check": convention_check(weight, device),
    }
    print(
        f"resident={summary['resident_bytes'] / 1e6:.1f} MB "
        f"allocator delta={allocated_delta / 1e6:.1f} MB "
        f"build peak={build_peak / 1e6:.1f} MB"
    )
    print(
        f"weight rel-L2={summary['weight_rel_l2']:.4f} "
        f"kernel-vs-dequant convention check={summary['convention_check']}"
    )

    summary["accuracy"] = accuracy(head, weight, hidden, ranks)
    head.rerank_k = 64  # accuracy() leaves the last tested R behind
    acc = summary["accuracy"]
    print(f"logits rel-L2={acc['logits_rel_l2']:.4f}")
    print(f"top-1 agreement with the FP16 argmax per R: {acc['top1_agreement']}")
    print(f"rank histogram of the FP16 argmax in NVFP4 logits: {acc['rank_hist']}")

    summary["timings_us"] = timings(head, weight, hidden, args.warmup, args.iters)
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
