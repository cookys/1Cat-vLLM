# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the tiny SYNTHETIC expert-routing dump fixture (CPU only).

    python tests/v1/worker/fixtures/make_expert_routing_fixture.py \
        tests/v1/worker/fixtures/expert_routing_dump

The routing is random, not measured: meta.json says ``synthetic: true``.  It
goes through the real dumper, so file names, key names, dtypes and shapes are the
final ones.  Six step indices at nominal c=2, MTP4 (5 rows per request):

    0 1 2 5   pure decode, 2 active requests, 10 rows
    3         dropped (layer 17 never staged): a hole in step_idx
    4         mixed: one decoding request (5 rows) + a 4-row prefill chunk
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe import expert_routing_dump as erd

LAYERS, DRAFT_LAYERS, DRAFT_STEPS, K, EXPERTS = 48, 1, 4, 10, 512
MAX_ROWS = 16
STEPS = [
    ("decode", [5, 5], [False, False]),
    ("decode", [5, 5], [False, False]),
    ("decode", [5, 5], [False, False]),
    ("hole", [5, 5], [False, False]),
    ("mixed", [5, 4], [False, True]),
    ("decode", [5, 5], [False, False]),
]


def _ids(rows: int, gen: torch.Generator) -> torch.Tensor:
    rows_ids = [torch.randperm(EXPERTS, generator=gen)[:K] for _ in range(rows)]
    return torch.stack(rows_ids).to(torch.int32)


def build(out_dir: str) -> None:
    gen = torch.Generator().manual_seed(20261005)
    dumper = erd.ExpertRoutingDumper(
        out_dir,
        rank=0,
        num_target_layers=LAYERS,
        num_draft_layers=DRAFT_LAYERS,
        num_draft_steps=DRAFT_STEPS,
        top_k=K,
        num_experts=EXPERTS,
        steps_per_file=3,
        max_rows=MAX_ROWS,
        with_weights=True,
        device="cpu",
        concurrency=2,
        cell_label="fixture-synthetic-c2",
        meta_extra={
            "model": "synthetic-fixture",
            "tp": 4,
            "synthetic": True,
            "num_speculative_tokens": DRAFT_STEPS,
        },
    )
    start = 100
    for kind, per_req, prefilling in STEPS:
        n = sum(per_req)
        pad = 2
        batch = types.SimpleNamespace(
            num_tokens=n,
            num_tokens_after_padding=n + pad,
            req_ids=["fixture-req-a", "fixture-req-b"],
            num_scheduled_tokens=np.array(per_req, dtype=np.int32),
            positions=torch.cat(
                [torch.arange(start, start + n), torch.zeros(pad, dtype=torch.int64)]
            ).to(torch.int64),
            is_prefilling_np=np.array(prefilling),
            num_draft_tokens=4 if kind != "mixed" else 4,
        )
        start += 5
        for layer in range(LAYERS):
            if kind == "hole" and layer == 17:
                continue
            ids = _ids(n + pad, gen)
            dumper.stage(
                dumper.slot_for(f"model.layers.{layer}.mlp.experts"),
                ids,
                torch.softmax(torch.randn(n + pad, K, generator=gen), dim=-1),
            )
        dumper.record_target_step(batch)
        for d in range(DRAFT_STEPS):
            rows = n + pad if d == 0 else len(per_req)
            dumper.stage(
                dumper.slot_for(f"mtp.layers.{LAYERS}.mlp.experts"),
                _ids(rows, gen),
                torch.rand(rows, K, generator=gen),
            )
            dumper.record_draft_step(d, len(per_req))
    dumper.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "expert_routing_dump"
    os.makedirs(target, exist_ok=True)
    build(target)
