# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph for the post-target sampling chain of the V2 spec-decode path.

After the target verify graph the V2 model runner launches about a hundred small
kernels from the host before the first draft graph can start: the fp32 logits
copy, temperature / min-p / top-k / top-p, the four rejection-sampling kernels,
``get_num_sampled_and_rejected``, ``post_update`` and the model-state
postprocess. The segment is host-launch bound. ``SamplingCudaGraphManager``
captures it as one CUDA graph per admitted key and replays it instead
(``VLLM_SM70_SAMPLING_CUDAGRAPH=1``).

What is inside the graph (everything reads graph-owned static buffers):

* ``input_ids[logits_indices]`` and ``positions[logits_indices]`` gathers;
* temperature, min-p and top-k/top-p on the fp32 logits, in this order;
* ``RejectionSampler.rejection_sample_processed``;
* ``get_num_sampled_and_rejected`` and ``post_update``;
* the model-state ``postprocess_state`` only when that call is provably the
  same at replay time as at capture time (see
  ``model_state_postprocess_is_capture_safe``); otherwise the model runner
  runs it eagerly after the replay.

What stays outside, on the main stream, before the replay: the NCCL logits
gather (``compute_logits``), the fp16 -> fp32 copy of the logits into the
static buffer, and ``copy_`` of the per-step tensors (``idx_mapping``,
``cu_num_logits``, ``expanded_idx_mapping``, ``expanded_local_pos``,
``logits_indices``) and of the sampling-state arrays into static buffers. After
the replay the outputs are cloned out of the graph pool, so the async output
copy on another stream never races with the next replay.

Why the sampling-state arrays are staged: ``SamplingStates.temperature`` and
friends are ``UvaBackedTensor`` whose ``.gpu`` view is re-pointed at the next
buffer of a round-robin pool by every ``apply_staged_writes()`` (every step),
and ``RequestState.prefill_len`` likewise on every step that adds a request.
A graph that baked those pointers would read stale buffers.

Numerics. Under capture ``apply_top_k_top_p_triton`` already routes to the
full-vocabulary reference (``apply_top_k_top_p_pytorch``). That is exactly what
eager execution does with ``VLLM_SM70_TOPK_TOPP_BRANCHFREE=2`` when top-k and
top-p are both set, so replay is bit-identical to eager mode 2. With top-k only
or top-p only the eager path is the Triton ``_topk_topp_kernel``, not the
reference, so those signatures are not admitted. The draws are counter-based
Philox keyed by (seed, position, token) read from the static ``seeds`` and
``pos`` buffers, so a replay with refreshed inputs draws fresh numbers; no
``torch.Generator`` state is involved.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

_MIB = float(1 << 20)

# Reasons a batch is not admitted. Each is logged once, so keep them free of
# per-batch numbers.
REASON_NOT_UNIFORM = (
    "the batch is not a uniform speculative-decode batch "
    "(every request must carry exactly num_speculative_tokens drafts)"
)
REASON_TOO_MANY_REQS = "num_reqs exceeds VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS"
REASON_PREFILL = "the batch contains prefilling requests"
REASON_GRAMMAR = "structured output (grammar bitmask) is active"
REASON_LOGPROBS = "logprobs or per-request logprob token ids are requested"
REASON_LOGIT_BIAS = "logit bias, allowed token ids or min_tokens is active"
REASON_PENALTIES = "repetition, frequency or presence penalty is active"
REASON_BAD_WORDS = "bad words are active"
REASON_COMPUTE_NANS = "VLLM_COMPUTE_NANS_IN_LOGITS is enabled"
REASON_DRAFT_LOGITS = "draft logits are in use (probabilistic drafting)"
REASON_ONE_SIDED = (
    "only one of top-k and top-p is set: eager execution runs the Triton "
    "top-k/top-p kernel for that case, which is not the reference path the "
    "graph captures"
)
REASON_SIGNATURE = "the sampling signature was not captured"
REASON_LOGITS_SHAPE = "the logits shape does not match the captured shape"
REASON_NO_GRAPH_FOR_SHAPE = "no graph was captured for this (num_reqs, num_logits)"


@dataclass(frozen=True)
class SamplingSignature:
    """The host-side sampling decisions that a captured graph freezes.

    Each field mirrors a host branch of the eager path (``SamplingStates`` and
    ``Sampler.apply_sampling_params``), evaluated over the requests in the batch.
    """

    # Any request has top_k != vocab_size.
    top_k: bool
    # Any request has top_p != 1.0.
    top_p: bool
    # Any request has temperature not in {0, 1}; otherwise the eager path skips
    # the temperature kernel.
    temperature: bool
    # Any request has min_p != 0.
    min_p: bool

    def __str__(self) -> str:
        return (
            f"top_k={int(self.top_k)},top_p={int(self.top_p)},"
            f"temperature={int(self.temperature)},min_p={int(self.min_p)}"
        )


@dataclass(frozen=True)
class SamplingGraphKey:
    num_reqs: int
    num_logits: int
    signature: SamplingSignature

    def __str__(self) -> str:
        return (
            f"(num_reqs={self.num_reqs}, num_logits={self.num_logits}, "
            f"{self.signature})"
        )


# The signature the SM70 bench and SWE harness sample with: temperature 1.0,
# top_k 20, top_p 0.95, no min_p. Temperature 1.0 is in {0, 1}, so the eager
# path skips the temperature kernel and so does the graph. Adding a signature
# here captures one more graph per shape into the same pool.
DEFAULT_CAPTURE_SIGNATURES: tuple[SamplingSignature, ...] = (
    SamplingSignature(top_k=True, top_p=True, temperature=False, min_p=False),
)


@dataclass
class SamplingGraphOutput:
    """Outputs of one replay, already cloned out of the graph pool."""

    sampler_output: SamplerOutput
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    # True when ``post_update`` ran inside the graph (always).
    post_update_done: bool
    # True when the model-state ``postprocess_state`` ran inside the graph. When
    # False the caller must run it eagerly after the replay.
    model_state_done: bool


def model_state_postprocess_is_capture_safe(model_state: Any) -> bool:
    """Whether ``model_state.postprocess_state`` may be frozen into a graph.

    A graph freezes every host-side decision of the call. The only host state
    the known model states consult is the Mamba align context, which
    ``MambaHybridModelState`` creates lazily inside ``preprocess_state`` of the
    first real forward (dummy runs never reach it). A graph captured before
    that would silently drop the align state copy for the lifetime of the
    server, so in align mode the call is capturable only when the context is
    already initialized. An override this function does not know is not
    capturable.
    """
    postprocess = getattr(type(model_state), "postprocess_state", None)
    if postprocess is ModelState.postprocess_state:
        return True  # The interface default does nothing.
    align_mode = getattr(model_state, "_align_mode", None)
    if align_mode is None:
        return False
    if not align_mode:
        return True
    ctx = getattr(model_state, "_mamba_ctx", None)
    return ctx is not None and bool(getattr(ctx, "is_initialized", False))


def compute_signature(
    sampler: Sampler, idx_mapping_np: np.ndarray
) -> SamplingSignature:
    """Evaluate the eager path's host branches for this batch."""
    states = sampler.sampling_states
    temperature = states.temperature.np[idx_mapping_np]
    return SamplingSignature(
        top_k=bool(np.any(states.top_k.np[idx_mapping_np] != states.vocab_size)),
        top_p=bool(np.any(states.top_p.np[idx_mapping_np] != 1.0)),
        temperature=not bool(np.all((temperature == 0.0) | (temperature == 1.0))),
        min_p=bool(np.any(states.min_p.np[idx_mapping_np] != 0.0)),
    )


class _StaticBuffers:
    """Graph-owned inputs, sized for the largest captured key.

    Every key uses views ``[:n]`` of the same buffers; the base pointer is the
    same for all of them. ``logits`` is fp32 so the fp16 -> fp32 conversion is
    the staging copy, and the graph processes it in place.
    """

    def __init__(
        self,
        max_graph_reqs: int,
        max_logits: int,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
    ):
        def zeros(n: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(n, dtype=dtype, device=device)

        self.logits = torch.zeros(
            max_logits, vocab_size, dtype=torch.float32, device=device
        )
        self.idx_mapping = zeros(max_graph_reqs, torch.int32)
        self.cu_num_logits = zeros(max_graph_reqs + 1, torch.int32)
        self.expanded_idx_mapping = zeros(max_logits, torch.int32)
        self.expanded_local_pos = zeros(max_logits, torch.int32)
        self.logits_indices = zeros(max_logits, torch.int64)
        # Mirrors of the UVA sampling-state arrays, [max_num_reqs] each.
        self.temperature = zeros(max_num_reqs, torch.float32)
        self.seeds = zeros(max_num_reqs, torch.int64)
        self.top_k = torch.full(
            (max_num_reqs,), vocab_size, dtype=torch.int32, device=device
        )
        self.top_p = torch.ones(max_num_reqs, dtype=torch.float32, device=device)
        self.min_p = zeros(max_num_reqs, torch.float32)
        self.prefill_len = zeros(max_num_reqs, torch.int32)

    def tensors(self) -> list[torch.Tensor]:
        return list(vars(self).values())

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclass
class _CapturedGraph:
    graph: Any
    # (sampled [R, K+1] int64, num_sampled [R] int32, num_rejected [R] int32),
    # allocated from the graph pool and kept alive here.
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    pool_reserved_bytes: int = 0
    pool_allocated_bytes: int = 0


class SamplingCudaGraphManager:
    """Captures and replays the sampling chain, one graph per admitted key.

    The key is ``(num_reqs, num_logits, SamplingSignature)``. For every
    ``num_reqs`` in ``1..max_graph_reqs`` and every signature in ``signatures``
    one graph is captured for the uniform speculative shape
    ``num_logits = num_reqs * (num_speculative_steps + 1)``. All graphs share
    one dedicated pool (``torch.cuda.graph_pool_handle()``): the gumbel and
    sort temporaries of this chain collided with the model pool in the same way
    they do for the Eagle managers. Sharing is safe because replays are
    sequential and every replay is self-contained; the held outputs are never
    reused by a later capture.
    """

    def __init__(
        self,
        *,
        sampler: Sampler,
        rejection_sampler: RejectionSampler,
        req_states: RequestState,
        input_buffers: InputBuffers,
        model_state: Any,
        post_update_fn: Callable[..., None],
        postprocess_state_fn: Callable[[torch.Tensor, torch.Tensor], None],
        device: torch.device,
        max_num_reqs: int,
        num_speculative_steps: int,
        vocab_size: int,
        max_graph_reqs: int = 1,
        signatures: tuple[SamplingSignature, ...] = DEFAULT_CAPTURE_SIGNATURES,
    ):
        """
        Args:
            post_update_fn: ``(idx_mapping, sampled, num_sampled, num_rejected,
                query_start_loc) -> None``; the model runner's ``post_update``
                call, shared with the eager path so the two cannot drift.
            postprocess_state_fn: ``(idx_mapping, num_sampled) -> None``; the
                model runner's model-state postprocess.
        """
        if num_speculative_steps < 1:
            raise ValueError("a sampling graph needs num_speculative_steps >= 1")
        self.sampler = sampler
        self.rejection_sampler = rejection_sampler
        self.req_states = req_states
        self.input_buffers = input_buffers
        self.model_state = model_state
        self.post_update_fn = post_update_fn
        self.postprocess_state_fn = postprocess_state_fn
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.max_graph_reqs = max(1, min(max_graph_reqs, max_num_reqs))
        if self.max_graph_reqs != max_graph_reqs:
            logger.warning(
                "VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS=%d is outside [1, "
                "max_num_seqs=%d]; using %d.",
                max_graph_reqs,
                max_num_reqs,
                self.max_graph_reqs,
            )
        self.signatures = tuple(signatures)

        self.logits_per_req = num_speculative_steps + 1
        self.keys: tuple[SamplingGraphKey, ...] = tuple(
            SamplingGraphKey(r, r * self.logits_per_req, signature)
            for r in range(1, self.max_graph_reqs + 1)
            for signature in self.signatures
        )
        self._key_set = frozenset(self.keys)
        self._shapes = frozenset((k.num_reqs, k.num_logits) for k in self.keys)

        self.bufs: _StaticBuffers | None = None
        self.pool: Any = None
        self.graphs: dict[SamplingGraphKey, _CapturedGraph] = {}
        self.model_state_in_graph = False
        self.num_replays = 0
        self.num_fallbacks = 0
        self._warned: set[str] = set()
        self._logged_first_replay = False

    # ------------------------------------------------------------------ CUDA
    # Seams for the CPU tests; the defaults are the real CUDA calls.

    def _new_graph(self) -> Any:
        return torch.cuda.CUDAGraph()

    def _new_pool(self) -> Any:
        return torch.cuda.graph_pool_handle()

    def _capture_scope(self) -> contextlib.AbstractContextManager:
        # Same outer context the model CudaGraphManager uses: it moves the
        # capture to a side stream and wraps the TP/PP collective capture state.
        return graph_capture(device=self.device)

    def _graph_scope(self, graph: Any) -> contextlib.AbstractContextManager:
        return torch.cuda.graph(graph, self.pool)

    def _memory_probe(self) -> tuple[int, int]:
        """(reserved, allocated) bytes of the device, (0, 0) off CUDA."""
        if self.device.type != "cuda":
            return 0, 0
        return (
            torch.cuda.memory_reserved(self.device),
            torch.cuda.memory_allocated(self.device),
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # --------------------------------------------------------------- capture

    @property
    def captured(self) -> bool:
        return bool(self.graphs)

    def static_nbytes(self) -> int:
        return 0 if self.bufs is None else self.bufs.nbytes()

    def pool_reserved_bytes(self) -> int:
        return sum(g.pool_reserved_bytes for g in self.graphs.values())

    def capture(self) -> None:
        """Warm up and capture every key. Call once, inside ``capture_model``.

        An exception propagates: a half-captured manager must not serve, and a
        failed capture is worth a loud startup failure for an opt-in feature.
        """
        if self.graphs:
            return
        self.model_state_in_graph = model_state_postprocess_is_capture_safe(
            self.model_state
        )
        if not self.model_state_in_graph:
            logger.info(
                "SM70 sampling CUDA graph: the model-state postprocess is not "
                "capture-safe yet (%s), so it runs eagerly after each replay.",
                type(self.model_state).__name__,
            )
        self.bufs = _StaticBuffers(
            self.max_graph_reqs,
            self.max_graph_reqs * self.logits_per_req,
            self.max_num_reqs,
            self.vocab_size,
            self.device,
        )
        self.pool = self._new_pool()
        free_before = self._free_bytes()
        with self._capture_scope():
            for key in self.keys:
                self._capture_key(key)
        free_after = self._free_bytes()
        logger.info(
            "SM70 sampling CUDA graph captured %d key(s): %s. Graph pool "
            "reserved %.2f MiB (sum of per-key deltas; shared pool), static "
            "input buffers %.2f MiB, device free memory delta %.2f MiB "
            "including warmup temporaries. Model-state postprocess in graph: %s.",
            len(self.graphs),
            ", ".join(str(k) for k in self.keys),
            self.pool_reserved_bytes() / _MIB,
            self.static_nbytes() / _MIB,
            (free_before - free_after) / _MIB,
            self.model_state_in_graph,
        )
        top_k_top_p_mode = int(envs.VLLM_SM70_TOPK_TOPP_BRANCHFREE)
        if top_k_top_p_mode != 2:
            logger.warning(
                "SM70 sampling CUDA graph: top-k/top-p inside the graph is the "
                "full-vocabulary reference, which equals eager "
                "VLLM_SM70_TOPK_TOPP_BRANCHFREE=2. The eager fallback path "
                "runs with BRANCHFREE=%d, so batches that are not admitted may "
                "differ from admitted ones in the last ulp of a top-p boundary.",
                top_k_top_p_mode,
            )

    def _free_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return torch.cuda.mem_get_info(self.device)[0]

    def _capture_key(self, key: SamplingGraphKey) -> None:
        self._fill_dummy_inputs(key)
        # Eager warmup: JIT-compile every Triton kernel of the chain and let
        # ATen set up its sort/scan workspaces before the capture.
        self._run_body(key)
        self._synchronize()
        graph = self._new_graph()
        # torch.cuda.graph.__enter__ calls empty_cache(), which releases the
        # blocks the eager warmup cached. Probe inside the scope so that release
        # is not subtracted from the pool's own growth.
        with self._graph_scope(graph):
            reserved_before, allocated_before = self._memory_probe()
            outputs = self._run_body(key)
            reserved_after, allocated_after = self._memory_probe()
        self.graphs[key] = _CapturedGraph(
            graph=graph,
            outputs=outputs,
            pool_reserved_bytes=max(0, reserved_after - reserved_before),
            pool_allocated_bytes=max(0, allocated_after - allocated_before),
        )
        logger.debug(
            "SM70 sampling CUDA graph %s: pool reserved %.2f MiB, allocated "
            "(held outputs) %.2f MiB.",
            key,
            self.graphs[key].pool_reserved_bytes / _MIB,
            self.graphs[key].pool_allocated_bytes / _MIB,
        )

    def _fill_dummy_inputs(self, key: SamplingGraphKey) -> None:
        """Fill the static buffers with a valid uniform batch for warmup/capture.

        ``InputBatch.make_dummy`` writes ``seq_lens``, ``query_start_loc``,
        ``input_ids`` and ``positions`` of the persistent input buffers (the
        same side effect as every other capture) and supplies the request
        slots. The speculative layout is then written explicitly because
        ``make_dummy`` models one logit per request.

        The warmup run updates request slots ``0..num_reqs-1`` (num_computed,
        total_len, last_sampled, accepted counts). Nothing is live while
        ``capture_model`` runs and ``add_request`` overwrites every one of
        those fields when a slot is first used.
        """
        assert self.bufs is not None
        b = self.bufs
        r, n = key.num_reqs, key.num_logits
        per_req = self.logits_per_req
        dummy = InputBatch.make_dummy(r, n, self.input_buffers)
        device = self.device
        local_pos = torch.arange(per_req, dtype=torch.int32, device=device)
        slots = torch.arange(r, dtype=torch.int32, device=device)
        b.idx_mapping[:r].copy_(dummy.idx_mapping)
        b.cu_num_logits[: r + 1].copy_(
            torch.arange(r + 1, dtype=torch.int32, device=device) * per_req
        )
        b.expanded_idx_mapping[:n].copy_(slots.repeat_interleave(per_req))
        b.expanded_local_pos[:n].copy_(local_pos.repeat(r))
        b.logits_indices[:n].copy_(torch.arange(n, dtype=torch.int64, device=device))
        b.logits.zero_()
        sig = key.signature
        b.temperature.fill_(0.7 if sig.temperature else 1.0)
        b.seeds.copy_(torch.arange(self.max_num_reqs, dtype=torch.int64, device=device))
        b.top_k.fill_(20 if sig.top_k else self.vocab_size)
        b.top_p.fill_(0.95 if sig.top_p else 1.0)
        b.min_p.fill_(0.05 if sig.min_p else 0.0)
        b.prefill_len.zero_()

    # ------------------------------------------------------------------ body

    def _apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        signature: SamplingSignature,
    ) -> torch.Tensor:
        """``Sampler.apply_sampling_params`` for an admitted batch.

        Admission guarantees that logit bias, penalties and bad words are
        no-ops in the eager path, so only the steps that follow them remain.
        ``logits`` is the static fp32 buffer and is processed in place, as the
        eager path processes its fp32 copy. The step order and the argument
        tensors mirror ``Sampler.apply_sampling_params`` and ``SamplingStates``;
        a CPU test pins that equivalence.
        """
        assert self.bufs is not None
        b = self.bufs
        if signature.temperature:
            apply_temperature(logits, expanded_idx_mapping, b.temperature)
        if signature.min_p:
            apply_min_p(logits, expanded_idx_mapping, b.min_p)
        if signature.top_k or signature.top_p:
            top_k = b.top_k[expanded_idx_mapping] if signature.top_k else None
            top_p = b.top_p[expanded_idx_mapping] if signature.top_p else None
            # Under capture this takes the branch-free reference path.
            logits = apply_top_k_top_p(logits, top_k, top_p)
        return logits

    def _run_body(
        self, key: SamplingGraphKey
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The captured computation; also what the eager warmup runs."""
        assert self.bufs is not None
        b = self.bufs
        r, n = key.num_reqs, key.num_logits
        idx_mapping = b.idx_mapping[:r]
        cu_num_logits = b.cu_num_logits[: r + 1]
        expanded_idx_mapping = b.expanded_idx_mapping[:n]
        expanded_local_pos = b.expanded_local_pos[:n]
        logits_indices = b.logits_indices[:n]

        draft_sampled = self.input_buffers.input_ids[logits_indices]
        pos = self.input_buffers.positions[logits_indices]
        processed_logits = self._apply_sampling_params(
            b.logits[:n], expanded_idx_mapping, key.signature
        )
        sampled, num_sampled = self.rejection_sampler.rejection_sample_processed(
            processed_logits,
            None,
            draft_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            b.temperature,
            b.seeds,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            self.input_buffers.seq_lens[:r],
            cu_num_logits,
            idx_mapping,
            b.prefill_len,
        )
        self.post_update_fn(
            idx_mapping,
            sampled,
            num_sampled,
            num_rejected,
            self.input_buffers.query_start_loc[: r + 1],
        )
        if self.model_state_in_graph:
            self.postprocess_state_fn(idx_mapping, num_sampled)
        return sampled, num_sampled, num_rejected

    # ------------------------------------------------------------- admission

    def check_admission(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any,
        draft_logits: torch.Tensor | None,
    ) -> tuple[SamplingGraphKey | None, str | None]:
        """Return ``(key, None)`` for an admitted batch or ``(None, reason)``."""
        if not self.graphs:
            return None, REASON_NO_GRAPH_FOR_SHAPE
        num_reqs = input_batch.num_reqs
        per_req = input_batch.num_draft_tokens_per_req
        if (
            input_batch.num_draft_tokens <= 0
            or per_req is None
            or len(per_req) != num_reqs
            or not bool(np.all(per_req == self.num_speculative_steps))
        ):
            return None, REASON_NOT_UNIFORM
        if num_reqs > self.max_graph_reqs:
            return None, REASON_TOO_MANY_REQS
        if draft_logits is not None:
            return None, REASON_DRAFT_LOGITS
        if grammar_output is not None or input_batch.has_structured_output_reqs:
            return None, REASON_GRAMMAR
        if self.sampler.compute_nans:
            return None, REASON_COMPUTE_NANS
        # Every row is a decode row: with a prefilling request post_update here
        # would also run before prompt logprobs read num_computed_tokens.
        if bool(np.any(input_batch.is_prefilling_np)):
            return None, REASON_PREFILL

        idx_mapping_np = input_batch.idx_mapping_np
        sampler = self.sampler
        if (
            sampler.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
            or sampler.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) != 0
        ):
            return None, REASON_LOGPROBS
        if bool(np.any(sampler.logit_bias_state.use_logit_bias[idx_mapping_np])):
            return None, REASON_LOGIT_BIAS
        if bool(np.any(sampler.penalties_state.use_penalty[idx_mapping_np])):
            return None, REASON_PENALTIES
        if int(sampler.bad_words_state.num_bad_words.np[idx_mapping_np].max()) != 0:
            return None, REASON_BAD_WORDS

        signature = compute_signature(sampler, idx_mapping_np)
        if signature.top_k != signature.top_p:
            return None, REASON_ONE_SIDED
        num_logits = input_batch.num_draft_tokens + num_reqs
        key = SamplingGraphKey(num_reqs, num_logits, signature)
        if key not in self._key_set:
            if (num_reqs, num_logits) not in self._shapes:
                return None, REASON_NO_GRAPH_FOR_SHAPE
            return None, REASON_SIGNATURE
        if tuple(logits.shape) != (num_logits, self.vocab_size):
            return None, REASON_LOGITS_SHAPE
        return key, None

    def _warn_once(self, reason: str) -> None:
        if reason in self._warned:
            return
        self._warned.add(reason)
        logger.warning(
            "SM70 sampling CUDA graph is not used for some batches: %s. Those "
            "batches run the eager sampling path (logged once per reason).",
            reason,
        )

    def run(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any,
        draft_logits: torch.Tensor | None,
    ) -> SamplingGraphOutput | None:
        """Replay the graph for an admitted batch, or return None (eager)."""
        key, reason = self.check_admission(
            logits, input_batch, grammar_output, draft_logits
        )
        if key is None:
            assert reason is not None
            self.num_fallbacks += 1
            self._warn_once(reason)
            return None
        return self.replay(key, logits, input_batch)

    # ---------------------------------------------------------------- replay

    def _stage_inputs(
        self,
        key: SamplingGraphKey,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        """Copy everything the graph reads into its static buffers.

        All copies are enqueued on the current (main) stream, before the replay.
        """
        assert self.bufs is not None
        b = self.bufs
        r, n = key.num_reqs, key.num_logits
        sig = key.signature

        # Per-step tensors (device -> device) and the logits (fp16 -> fp32).
        b.logits[:n].copy_(logits)
        b.idx_mapping[:r].copy_(input_batch.idx_mapping)
        b.cu_num_logits[: r + 1].copy_(input_batch.cu_num_logits)
        b.expanded_idx_mapping[:n].copy_(input_batch.expanded_idx_mapping)
        b.expanded_local_pos[:n].copy_(input_batch.expanded_local_pos)
        b.logits_indices[:n].copy_(input_batch.logits_indices)
        # UVA arrays whose .gpu pointer rotates every step.
        states = self.sampler.sampling_states
        b.temperature.copy_(states.temperature.gpu)
        b.seeds.copy_(states.seeds.gpu)
        if sig.top_k:
            b.top_k.copy_(states.top_k.gpu)
        if sig.top_p:
            b.top_p.copy_(states.top_p.gpu)
        if sig.min_p:
            b.min_p.copy_(states.min_p.gpu)
        b.prefill_len.copy_(self.req_states.prefill_len.gpu)

    def replay(
        self, key: SamplingGraphKey, logits: torch.Tensor, input_batch: InputBatch
    ) -> SamplingGraphOutput:
        """Stage the inputs, replay, and hand out clones of the outputs."""
        captured = self.graphs[key]
        self._stage_inputs(key, logits, input_batch)
        captured.graph.replay()
        self.num_replays += 1
        if not self._logged_first_replay:
            self._logged_first_replay = True
            logger.info("SM70 sampling CUDA graph replay active, first key %s.", key)

        # The outputs live in the graph pool and the next replay overwrites
        # them, while the async output copy reads them on another stream.
        sampled, num_sampled, num_rejected = (t.clone() for t in captured.outputs)
        return SamplingGraphOutput(
            sampler_output=SamplerOutput(
                sampled_token_ids=sampled,
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled,
            ),
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            post_update_done=True,
            model_state_done=self.model_state_in_graph,
        )
