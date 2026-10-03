# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU replay check for the SM70 sampling CUDA graph (VLLM_SM70_SAMPLING_CUDAGRAPH).

A model-free harness around the real ``Sampler``, ``RejectionSampler``,
``RequestState`` and ``SamplingCudaGraphManager``. Two identical rigs run the
same speculative-decode rounds in lock step: one through the eager chain the
model runner uses today (rejection sampling, ``get_num_sampled_and_rejected``,
``post_update``, the model-state postprocess), the other through the graph
manager. Every round it asserts, bit for bit:

* ``num_sampled``, ``num_rejected`` and the first ``num_sampled`` ``sampled``
  tokens of every request are identical (the entries after the first rejection
  are never written and never read, so they are not compared);
* the state the round writes is identical: ``num_computed_tokens``,
  ``total_len``, ``last_sampled_tokens``, ``all_token_ids``, the penalty bin
  counts and the model-state ``num_accepted`` buffer.

Rounds use the production plumbing: ``idx_mapping`` and the expanded tensors
are fresh per round, ``logits_indices`` comes from the real
``combine_sampled_and_draft_tokens`` kernel, positions come from
``prepare_pos_seq_lens`` and advance with every accepted token, and the UVA
sampling-state arrays are re-pointed at the next pool buffer by
``apply_staged_writes()`` every round, exactly as the runner does. Halfway
through, a request's seed, top-p and top-k change, so a graph that had baked
those pointers or values would diverge from the eager rig.

Then it checks that replay does not freeze the RNG: with fixed logits and fixed
draft tokens, only the positions move; both rigs must keep matching and the
sampled rows must not all be equal (``frozen_rng`` false).

The captured request slots (``0..num_reqs-1``) are the slots the live requests
use, so the run also covers the claim that the capture-time warmup does not
disturb a request that later takes the same slot (the eager rig had no
warmup).

The default run covers both ways the model-state postprocess can run: inside
the graph, and as the eager tail the runner uses when the Mamba align context
did not exist at capture time (``--model-state both``).

It also reports the graph pool bytes (the capture-time reserved delta against
the design's ~30 MB estimate for the (1, 5) shape), the static input buffer
bytes, and host-enqueue and GPU time per round for eager and replay.

The eager rig runs with ``VLLM_SM70_TOPK_TOPP_BRANCHFREE=2`` (the reference
path the graph reproduces); ``--eager-branchfree 0`` compares against today's
compact path instead, where a last-ulp difference in a top-p margin row is a
legitimate finding, not a harness bug.

Exit status: 0 all checks passed; 1 a mismatch or a failed check; 2 the
environment cannot run it (no CUDA device). One JSON line is printed and
written to ``--out``.

    # From the worktree, inside the serving venv (the installed wheel carries
    # the overlay files); use one idle SM70 GPU, about a minute:
    PY=/data/venvs/1cat-m589/bin/python
    cd /data/src/1cat-wt-fable && CUDA_VISIBLE_DEVICES=<idle gpu> \
        $PY benchmarks/sm70_sampling_graph_replay_check.py \
        --rounds 40 --out /data/bench/sampling_graph_replay_check.json

    # Against the worktree sources instead of an overlaid wheel:
    cd /data/src/1cat-wt-fable && CUDA_VISIBLE_DEVICES=<idle gpu> \
        PYTHONPATH=/data/src/1cat-wt-fable $PY \
        benchmarks/sm70_sampling_graph_replay_check.py --rounds 40

More shapes and signatures: ``--num-reqs 2`` (captures num_reqs=2, num_logits=10),
``--temperature 0.7`` (captures the temperature signature). Engine level, with
the real model: run the server twice, ``VLLM_SM70_SAMPLING_CUDAGRAPH=0`` and
``=1``, send the same prompts with a fixed ``seed`` and compare the token
streams; the ``1`` log must contain ``SM70 sampling CUDA graph replay active``.
This script does not start an engine.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import vllm._C  # noqa: F401  (registers torch.ops._C.*)

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample import cudagraph as cg
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

NUM_SPEC = 4  # MTP4
PER_REQ = NUM_SPEC + 1
MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 64
DESIGN_POOL_ESTIMATE_MIB = 30.0  # design doc, (1, 5) shape at 248320 vocab
MIB = float(1 << 20)
WARM_ROUNDS = 3  # excluded from the timing medians


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rounds", type=int, default=40, help="rounds, at least 20")
    parser.add_argument(
        "--position-rounds",
        type=int,
        default=20,
        help="rounds of the fixed-logits experiment where only positions move",
    )
    parser.add_argument("--num-reqs", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=2.0,
        help="std of the random logits; larger means peakier distributions",
    )
    parser.add_argument(
        "--model-state",
        choices=("in-graph", "tail", "both"),
        default="both",
        help="run the model-state postprocess inside the graph, as an eager "
        "tail after the replay, or check both",
    )
    parser.add_argument(
        "--eager-branchfree",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="VLLM_SM70_TOPK_TOPP_BRANCHFREE for the eager rig; 2 is the "
        "reference path the graph reproduces",
    )
    parser.add_argument("--out", help="also write the JSON summary here")
    args = parser.parse_args()
    if args.rounds < 20:
        parser.error("--rounds must be at least 20")
    if not 1 <= args.num_reqs <= MAX_NUM_REQS:
        parser.error(f"--num-reqs must be in [1, {MAX_NUM_REQS}]")
    return args


# ----------------------------------------------------------------- the rig


class _StubModelState:
    """Stands in for MambaHybridModelState.postprocess_state.

    It runs the real scatter kernel into a ``num_accepted`` buffer. With
    ``tail`` the Mamba-align attributes say the context does not exist yet, so
    the manager must leave the call out of the graph.
    """

    def __init__(self, num_accepted: torch.Tensor, tail: bool):
        self.num_accepted = num_accepted
        self._align_mode = tail
        self._mamba_ctx = None

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            _scatter_num_accepted_kernel,
        )

        _scatter_num_accepted_kernel[(idx_mapping.shape[0],)](
            idx_mapping, num_sampled, self.num_accepted
        )


@dataclasses.dataclass
class RoundOutputs:
    sampled: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    host_us: float
    gpu_us: float


class HarnessManager(cg.SamplingCudaGraphManager):
    """The production manager; only the outer capture scope may be skipped."""

    outer_scope: bool = True

    def _capture_scope(self):
        if self.outer_scope:
            return super()._capture_scope()
        return contextlib.nullcontext()


def _init_single_rank_distributed() -> str:
    """Initialize a one-rank distributed environment for graph_capture()."""
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )

        port = 29500 + (os.getpid() % 2000)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=0,
            )
            ensure_model_parallel_initialized(1, 1)
        return "tp/pp groups initialized (world size 1)"
    except Exception as exc:  # noqa: BLE001 - report and carry on without the scope
        return f"unavailable ({type(exc).__name__}: {exc})"


class Rig:
    """One model-free instance of the V2 sampling state.

    The post-update methods are the model runner's own functions, so the eager
    rig and the graph's ``post_update_fn`` run exactly the production code.
    """

    _post_update_sampled = GPUModelRunner._post_update_sampled
    _postprocess_model_state = GPUModelRunner._postprocess_model_state
    postprocess_sampled = GPUModelRunner.postprocess_sampled

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        *,
        use_graph: bool,
        tail: bool,
        outer_scope: bool,
    ):
        self.args = args
        self.device = device
        self.use_graph = use_graph
        self.num_reqs = args.num_reqs
        vocab = args.vocab_size
        self.max_model_len = (
            args.prompt_len + (args.rounds + args.position_rounds + 8) * PER_REQ + 64
        )
        # The runner methods the graph shares with the eager path read these.
        self.is_last_pp_rank = True
        self.req_states = RequestState(
            max_num_reqs=MAX_NUM_REQS,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=MAX_NUM_TOKENS,
            num_speculative_steps=NUM_SPEC,
            vocab_size=vocab,
            device=device,
        )
        # Requests take slots 0..num_reqs-1, the slots the capture warmup uses.
        self.req_states.free_indices = list(reversed(range(MAX_NUM_REQS)))
        self.input_buffers = InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, device)
        self.sampler = Sampler(
            MAX_NUM_REQS,
            vocab,
            device,
            self.req_states,
            logprobs_mode="raw_logprobs",
            num_speculative_tokens=PER_REQ,
        )
        spec_config = SimpleNamespace(
            num_speculative_tokens=NUM_SPEC,
            rejection_sample_method="standard",
            synthetic_acceptance_rates=None,
        )
        self.rejection_sampler = RejectionSampler(self.sampler, spec_config, device)
        self.num_accepted = torch.ones(MAX_NUM_REQS, dtype=torch.int32, device=device)
        self.model_state = _StubModelState(self.num_accepted, tail)

        self.manager: HarnessManager | None = None
        self.capture_info: dict[str, Any] = {}
        if use_graph:
            signature = cg.SamplingSignature(
                top_k=args.top_k != vocab,
                top_p=args.top_p != 1.0,
                temperature=args.temperature not in (0.0, 1.0),
                min_p=args.min_p != 0.0,
            )
            self.manager = HarnessManager(
                sampler=self.sampler,
                rejection_sampler=self.rejection_sampler,
                req_states=self.req_states,
                input_buffers=self.input_buffers,
                model_state=self.model_state,
                post_update_fn=self._post_update_sampled,
                postprocess_state_fn=self._postprocess_model_state,
                device=device,
                max_num_reqs=MAX_NUM_REQS,
                num_speculative_steps=NUM_SPEC,
                vocab_size=vocab,
                max_graph_reqs=args.num_reqs,
                signatures=(signature,),
            )
            self.manager.outer_scope = outer_scope
            self.signature = signature
            self._capture()
        self._add_requests()

    def _capture(self) -> None:
        assert self.manager is not None
        torch.cuda.synchronize(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        free_before = torch.cuda.mem_get_info(self.device)[0]
        start = time.perf_counter()
        self.manager.capture()
        torch.cuda.synchronize(self.device)
        self.capture_info = {
            "capture_seconds": round(time.perf_counter() - start, 3),
            "reserved_delta_mib": round(
                (torch.cuda.memory_reserved(self.device) - reserved_before) / MIB, 3
            ),
            "free_memory_delta_mib": round(
                (free_before - torch.cuda.mem_get_info(self.device)[0]) / MIB, 3
            ),
            "pool_reserved_mib": round(self.manager.pool_reserved_bytes() / MIB, 3),
            "static_input_mib": round(self.manager.static_nbytes() / MIB, 3),
            "model_state_in_graph": self.manager.model_state_in_graph,
            "keys": [str(k) for k in self.manager.keys],
        }

    def _add_requests(self) -> None:
        args = self.args
        generator = np.random.default_rng(args.seed)
        self.slots = []
        for i in range(self.num_reqs):
            prompt = generator.integers(0, args.vocab_size, args.prompt_len).tolist()
            req_id = f"req{i}"
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=args.prompt_len,
                all_token_ids=prompt,
                num_computed_tokens=args.prompt_len,
                max_tokens=self.max_model_len,
            )
            slot = self.req_states.req_id_to_index[req_id]
            self.slots.append(slot)
            self.sampler.add_request(
                slot,
                args.prompt_len,
                SamplingParams(
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    seed=args.seed + i,
                ),
            )
        self.slots_np = np.array(self.slots, dtype=np.int32)
        self.slots_t = torch.tensor(self.slots, dtype=torch.int64, device=self.device)
        self.req_states.apply_staged_writes()
        self.sampler.apply_staged_writes()

    # ------------------------------------------------------------- rounds

    def change_sampling_params(self) -> None:
        """A seed, top-p and top-k change, as a new request would stage."""
        states = self.sampler.sampling_states
        slot = self.slots[0]
        states.seeds.np[slot] = self.args.seed + 7777
        if self.args.top_p != 1.0:
            states.top_p.np[slot] = max(0.5, self.args.top_p - 0.05)
        if self.args.top_k != self.args.vocab_size:
            states.top_k.np[slot] = max(2, self.args.top_k // 2)

    def set_computed_tokens(self, value: int) -> None:
        self.req_states.num_computed_tokens.gpu[self.slots_t] = value

    def _prepare_batch(self, drafts: torch.Tensor) -> InputBatch:
        r = self.num_reqs
        n = r * PER_REQ
        device = self.device
        self.req_states.draft_tokens[self.slots_t] = drafts
        idx_mapping = async_copy_to_gpu(self.slots_np, device=device)
        cu_np = np.arange(r + 1, dtype=np.int32) * PER_REQ
        cu_num_logits = async_copy_to_gpu(cu_np, device=device)
        expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
            idx_mapping, n, cu_num_logits, PER_REQ
        )
        query_start_loc_np = np.full(MAX_NUM_REQS + 1, n, dtype=np.int32)
        query_start_loc_np[: r + 1] = cu_np
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc = self.input_buffers.query_start_loc[: r + 1]
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:r]
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            n,
        )
        return InputBatch(
            req_ids=[f"req{i}" for i in range(r)],
            num_reqs=r,
            num_reqs_after_padding=r,
            idx_mapping=idx_mapping,
            idx_mapping_np=self.slots_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=np.full(r, PER_REQ, dtype=np.int32),
            num_tokens=n,
            num_tokens_after_padding=n,
            num_draft_tokens=r * NUM_SPEC,
            num_draft_tokens_per_req=np.full(r, NUM_SPEC, dtype=np.int32),
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np[: r + 1],
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.zeros(r, dtype=torch.int32),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(r, dtype=np.int32),
            prefill_len_np=np.full(r, self.args.prompt_len, dtype=np.int32),
            num_computed_prefill_tokens_np=np.full(
                r, self.args.prompt_len, dtype=np.int32
            ),
            is_prefilling_np=np.zeros(r, dtype=np.bool_),
            max_seq_len_np=None,
            input_ids=self.input_buffers.input_ids[:n],
            positions=self.input_buffers.positions[:n],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_np,
            has_structured_output_reqs=False,
        )

    def step(self, logits: torch.Tensor, drafts: torch.Tensor) -> RoundOutputs:
        """One round, in the order the runner uses."""
        # The runner applies the sampler's staged writes every step, which
        # re-points the UVA .gpu views; prefill_len rotates with the request
        # state writes.
        self.sampler.apply_staged_writes()
        self.req_states.apply_staged_writes()
        batch = self._prepare_batch(drafts)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        host_start = time.perf_counter()
        if self.use_graph:
            assert self.manager is not None
            out = self.manager.run(logits, batch, None, None)
            if out is None:
                raise RuntimeError(
                    "the batch was not admitted to the graph; see the warning above"
                )
            sampled = out.sampler_output.sampled_token_ids
            num_sampled, num_rejected = out.num_sampled, out.num_rejected
            if not out.model_state_done:
                self._postprocess_model_state(batch.idx_mapping, num_sampled)
        else:
            sampler_output = self.rejection_sampler(logits, batch, None)
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                sampler_output.num_sampled,
                batch.seq_lens,
                batch.cu_num_logits,
                batch.idx_mapping,
                self.req_states.prefill_len.gpu,
            )
            sampled = sampler_output.sampled_token_ids
            self.postprocess_sampled(
                batch.idx_mapping,
                sampled,
                num_sampled,
                num_rejected,
                batch.query_start_loc,
            )
        host_us = (time.perf_counter() - host_start) * 1e6
        end_event.record()
        end_event.synchronize()
        return RoundOutputs(
            sampled=sampled.clone(),
            num_sampled=num_sampled.clone(),
            num_rejected=num_rejected.clone(),
            host_us=host_us,
            gpu_us=start_event.elapsed_time(end_event) * 1e3,
        )

    def state_tensors(self) -> dict[str, torch.Tensor]:
        slots = self.slots_t
        states = self.req_states
        return {
            "num_computed_tokens": states.num_computed_tokens.gpu[slots].clone(),
            "total_len": states.total_len.gpu[slots].clone(),
            "last_sampled_tokens": states.last_sampled_tokens[slots].clone(),
            "all_token_ids": states.all_token_ids.gpu[slots].clone(),
            "output_bin_counts": self.sampler.penalties_state.output_bin_counts[
                slots
            ].clone(),
            "num_accepted": self.num_accepted[slots].clone(),
        }


# --------------------------------------------------------------- comparison


def compare_round(
    rnd: int, eager: Rig, eager_out: RoundOutputs, graph: Rig, graph_out: RoundOutputs
) -> dict[str, Any] | None:
    """First field that differs, or None when the round is bit-identical.

    ``sampled`` is compared only up to ``num_sampled`` per request: the kernels
    leave the entries after the first rejection unwritten (``new_empty``), the
    consumers ignore them, and they hold whatever the allocator recycled.
    """
    valid = (
        torch.arange(eager_out.sampled.shape[1], device=eager_out.sampled.device)[None]
        < eager_out.num_sampled[:, None]
    )
    pairs = {
        "num_sampled": (eager_out.num_sampled, graph_out.num_sampled),
        "num_rejected": (eager_out.num_rejected, graph_out.num_rejected),
        "sampled": (
            torch.where(valid, eager_out.sampled, 0),
            torch.where(valid, graph_out.sampled, 0),
        ),
    }
    eager_state, graph_state = eager.state_tensors(), graph.state_tensors()
    pairs.update({k: (eager_state[k], graph_state[k]) for k in eager_state})
    for name, (want, got) in pairs.items():
        if not torch.equal(want, got):
            flat_want, flat_got = want.flatten(), got.flatten()
            differing = (flat_want != flat_got).nonzero().flatten()[:4].tolist()
            return {
                "round": rnd,
                "field": name,
                "first_differing_flat_indices": differing,
                "eager": flat_want[differing].tolist() if differing else None,
                "graph": flat_got[differing].tolist() if differing else None,
            }
    return None


def make_round_inputs(
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
    host_rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh random logits [L, V] fp16 and draft tokens [R, K] int64."""
    n = args.num_reqs * PER_REQ
    logits = (
        torch.randn(
            n,
            args.vocab_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * args.logit_scale
    ).to(torch.float16)
    return logits, draft_tokens_for(args, logits, host_rng)


def draft_tokens_for(
    args: argparse.Namespace, logits: torch.Tensor, host_rng: np.random.Generator
) -> torch.Tensor:
    """Drafts from the top of each verifying row: a mix of accepts and rejects."""
    top = torch.topk(logits.float(), 5, dim=-1).indices.cpu().numpy()
    rows = top.reshape(args.num_reqs, PER_REQ, 5)[:, :NUM_SPEC]
    pick = np.where(
        host_rng.random(rows.shape[:2]) < 0.5,
        0,
        host_rng.integers(0, 5, rows.shape[:2]),
    )
    chosen = np.take_along_axis(rows, pick[..., None], axis=-1)[..., 0]
    return torch.from_numpy(chosen.astype(np.int64)).to(logits.device)


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 1) if values else float("nan")


def run_variant(
    args: argparse.Namespace, device: torch.device, tail: bool, outer_scope: bool
) -> dict[str, Any]:
    """One lock-step run of the eager rig against the graph rig."""
    torch.manual_seed(args.seed)
    eager = Rig(args, device, use_graph=False, tail=tail, outer_scope=outer_scope)
    graph = Rig(args, device, use_graph=True, tail=tail, outer_scope=outer_scope)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    host_rng = np.random.default_rng(args.seed)

    mismatches: list[dict[str, Any]] = []
    eager_outs: list[RoundOutputs] = []
    graph_outs: list[RoundOutputs] = []

    def lock_step(rnd: int, logits: torch.Tensor, drafts: torch.Tensor) -> RoundOutputs:
        e = eager.step(logits, drafts)
        g = graph.step(logits, drafts)
        eager_outs.append(e)
        graph_outs.append(g)
        diff = compare_round(rnd, eager, e, graph, g)
        if diff is not None:
            mismatches.append(diff)
        return g

    # 1. Random logits every round, parameters change halfway.
    accepted_lengths = []
    for rnd in range(args.rounds):
        if rnd == args.rounds // 2:
            eager.change_sampling_params()
            graph.change_sampling_params()
        logits, drafts = make_round_inputs(args, device, generator, host_rng)
        g = lock_step(rnd, logits, drafts)
        accepted_lengths.append(g.num_sampled.float().mean().item())

    # 2. Fixed logits and drafts; only the positions move.
    logits, drafts = make_round_inputs(args, device, generator, host_rng)
    rows = []
    for j in range(args.position_rounds):
        for rig in (eager, graph):
            rig.set_computed_tokens(args.prompt_len + 17 * (j + 1))
        g = lock_step(args.rounds + j, logits, drafts)
        accepted = int(g.num_sampled[0])
        rows.append(tuple(g.sampled[0, :accepted].tolist()))
    distinct = len(set(rows))

    timed_e = eager_outs[WARM_ROUNDS : args.rounds]
    timed_g = graph_outs[WARM_ROUNDS : args.rounds]
    assert graph.manager is not None
    result = {
        "model_state": "eager-tail" if tail else "in-graph",
        "signature": str(graph.signature),
        "signature_is_default": graph.signature in cg.DEFAULT_CAPTURE_SIGNATURES,
        "rounds_compared": len(eager_outs),
        "mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "mean_num_sampled": round(statistics.mean(accepted_lengths), 3),
        "distinct_outputs_with_fixed_logits": distinct,
        "position_rounds": args.position_rounds,
        "frozen_rng": distinct < 2,
        "graph_replays": graph.manager.num_replays,
        "graph_fallbacks": graph.manager.num_fallbacks,
        "capture": graph.capture_info,
        "pool_reserved_mib_vs_design_estimate": [
            graph.capture_info["pool_reserved_mib"],
            DESIGN_POOL_ESTIMATE_MIB,
        ],
        "host_enqueue_us_median": {
            "eager": _median([o.host_us for o in timed_e]),
            "replay": _median([o.host_us for o in timed_g]),
        },
        "gpu_us_median": {
            "eager": _median([o.gpu_us for o in timed_e]),
            "replay": _median([o.gpu_us for o in timed_g]),
        },
    }
    result["pass"] = (
        not mismatches
        and not result["frozen_rng"]
        and result["graph_fallbacks"] == 0
        and result["graph_replays"] == len(graph_outs)
        and graph.manager.model_state_in_graph == (not tail)
    )
    return result


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA device (set CUDA_VISIBLE_DEVICES to an idle GPU)")
        return 2
    os.environ["VLLM_SM70_TOPK_TOPP_BRANCHFREE"] = str(args.eager_branchfree)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    scope = _init_single_rank_distributed()
    outer_scope = scope.startswith("tp/pp")
    print(
        f"device {torch.cuda.get_device_name(device)} {capability}; "
        f"graph_capture scope: {scope}; "
        f"eager VLLM_SM70_TOPK_TOPP_BRANCHFREE={envs.VLLM_SM70_TOPK_TOPP_BRANCHFREE}"
    )
    if capability != (7, 0):
        print(
            "note: not an SM70 device. Eager top-k/top-p runs the Triton kernel "
            "there, not the reference path the graph captures, so mismatches are "
            "expected; the runner does not capture the graph on such devices."
        )

    variants = {"in-graph": [False], "tail": [True], "both": [False, True]}[
        args.model_state
    ]
    results = []
    with torch.inference_mode():
        for tail in variants:
            results.append(run_variant(args, device, tail, outer_scope))
    summary = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "num_reqs": args.num_reqs,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": args.min_p,
        },
        "eager_branchfree": args.eager_branchfree,
        "graph_capture_outer_scope": scope,
        "default_capture_signatures": [str(s) for s in cg.DEFAULT_CAPTURE_SIGNATURES],
        "variants": results,
        "pass": all(r["pass"] for r in results),
    }
    line = json.dumps(summary)
    print(line)
    if args.out:
        with open(args.out, "w") as f:
            f.write(line + "\n")
    for r in results:
        verdict = "PASS" if r["pass"] else "FAIL"
        print(
            f"[{verdict}] model_state={r['model_state']}: "
            f"{r['rounds_compared']} rounds, {r['mismatches']} mismatches, "
            f"frozen_rng={r['frozen_rng']} "
            f"(distinct={r['distinct_outputs_with_fixed_logits']}/"
            f"{r['position_rounds']}), pool "
            f"{r['capture']['pool_reserved_mib']} MiB reserved "
            f"(design ~{DESIGN_POOL_ESTIMATE_MIB:.0f}; whole capture "
            f"{r['capture']['reserved_delta_mib']} MiB incl. warmup cache, "
            f"static inputs {r['capture']['static_input_mib']} MiB), host enqueue "
            f"{r['host_enqueue_us_median']['eager']} -> "
            f"{r['host_enqueue_us_median']['replay']} us, gpu "
            f"{r['gpu_us_median']['eager']} -> {r['gpu_us_median']['replay']} us"
        )
        if r["first_mismatch"] is not None:
            print(f"  first mismatch: {r['first_mismatch']}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
