# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the SM70 sampling CUDA graph manager.

The CUDA graph object, the graph pool and the Triton kernels are replaced by
recorders, so nothing here touches a GPU. What is pinned:

* the admission rule: which batches get a graph, and the reason for every one
  that does not, logged once;
* the key/signature computation, which must mirror the eager host branches;
* the static input buffers: sizes, staging copies, and that the UVA
  sampling-state arrays are re-read every step (their ``.gpu`` pointer rotates);
* the capture protocol (warmup then capture, one shared pool, memory log);
* that the graph body applies the same sampling steps, in the same order and
  with the same arguments, as ``Sampler.apply_sampling_params``;
* the model-runner wiring (env gate, graph-or-eager in ``sample`` and the
  post-update work skipped in ``sample_tokens``).

Run on a GPU-less shell. The production venv has no pytest, so borrow one with
uv without modifying the venv:
    cd <worktree> && CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> \
        uv run --no-project --python <venv>/bin/python --with pytest -- \
        python -m pytest --noconftest tests/v1/sample/test_sm70_sampling_cudagraph.py -q
"""

import contextlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample import cudagraph as cg
from vllm.v1.worker.gpu.sample import states as states_module
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_module
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

CPU = torch.device("cpu")
K = 4  # MTP4: num_speculative_tokens
PER_REQ = K + 1
VOCAB = 64
MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 64
SIG = cg.DEFAULT_CAPTURE_SIGNATURES[0]
MIB = 1 << 20


@pytest.fixture(autouse=True)
def _fresh_env_cache():
    # The server freezes env lookups after startup; tests need live reads.
    envs.disable_envs_cache()


# --------------------------------------------------------------------- fakes


class FakeUva:
    """Stand-in for UvaBackedTensor: host array plus a rotating device view."""

    def __init__(self, values: Any, dtype: Any):
        self.np = np.array(values, dtype=dtype)
        self.gpu = torch.from_numpy(self.np.copy())

    def rotate(self) -> None:
        """What apply_staged_writes() does: re-point .gpu at a fresh buffer."""
        self.gpu = torch.from_numpy(self.np.copy())


def make_sampler(vocab: int = VOCAB, max_num_reqs: int = MAX_NUM_REQS):
    """A sampler whose per-request state encodes the bench signature.

    ``sampling_states`` is a real ``SamplingStates`` (built without __init__,
    which needs pinned/UVA memory) so its real host branches run.
    """
    states = object.__new__(SamplingStates)
    states.max_num_reqs = max_num_reqs
    states.vocab_size = vocab
    states.temperature = FakeUva(np.ones(max_num_reqs), np.float32)
    states.top_k = FakeUva(np.full(max_num_reqs, 20), np.int32)
    states.top_p = FakeUva(np.full(max_num_reqs, 0.95), np.float32)
    states.min_p = FakeUva(np.zeros(max_num_reqs), np.float32)
    states.seeds = FakeUva(np.arange(max_num_reqs) + 1000, np.int64)
    states.num_logprobs = np.full(max_num_reqs, NO_LOGPROBS, dtype=np.int32)
    token_ids = {"n": 0}
    sampler = SimpleNamespace(
        compute_nans=False,
        use_fp64_gumbel=False,
        logprobs_mode="raw_logprobs",
        sampling_states=states,
        logprob_token_ids_state=SimpleNamespace(
            max_num_token_ids=lambda _idx: token_ids["n"], state=token_ids
        ),
        logit_bias_state=SimpleNamespace(
            use_logit_bias=np.zeros(max_num_reqs, dtype=bool),
            apply_logit_bias=lambda *a, **k: None,
        ),
        penalties_state=SimpleNamespace(
            use_penalty=np.zeros(max_num_reqs, dtype=bool),
            apply_penalties=lambda *a, **k: None,
            output_bin_counts=torch.arange(
                max_num_reqs * vocab, dtype=torch.int32
            ).view(max_num_reqs, vocab),
        ),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.zeros(max_num_reqs, dtype=np.int32)),
            apply_bad_words=lambda *a, **k: None,
        ),
    )
    return sampler


def make_batch(num_reqs: int = 1, k: int = K, slots: list[int] | None = None):
    """A uniform speculative batch, shaped like InputBatch."""
    slots = list(range(num_reqs)) if slots is None else slots
    per_req = k + 1
    n = num_reqs * per_req
    idx_np = np.array(slots, dtype=np.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=num_reqs * k,
        num_draft_tokens_per_req=np.full(num_reqs, k, dtype=np.int32),
        has_structured_output_reqs=False,
        is_prefilling_np=np.zeros(num_reqs, dtype=bool),
        idx_mapping_np=idx_np,
        idx_mapping=torch.from_numpy(idx_np.copy()),
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32) * per_req,
        expanded_idx_mapping=torch.from_numpy(idx_np.copy()).repeat_interleave(per_req),
        expanded_local_pos=torch.arange(per_req, dtype=torch.int32).repeat(num_reqs),
        logits_indices=torch.arange(n, dtype=torch.int64) + 3,
    )


class FakeGraph:
    def __init__(self):
        self.replays = 0
        self.captured = False

    def replay(self) -> None:
        self.replays += 1


class FakeRejection:
    num_speculative_steps = K

    def __init__(self, calls: list):
        self.calls = calls

    def rejection_sample_processed(self, *args, **kwargs):
        self.calls.append(("rejection", args, kwargs))
        cu_num_logits = args[4]
        r = cu_num_logits.shape[0] - 1
        return (
            torch.zeros(r, PER_REQ, dtype=torch.int64),
            torch.ones(r, dtype=torch.int32),
        )


class CpuManager(cg.SamplingCudaGraphManager):
    """The manager with every CUDA call replaced by a recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.pools: list[Any] = []
        self.made: list[FakeGraph] = []
        self.probe_calls = 0
        self.pool_growth = 25 * MIB

    def _new_graph(self):
        graph = FakeGraph()
        self.made.append(graph)
        return graph

    def _new_pool(self):
        return object()

    def _capture_scope(self):
        self.events.append("outer-scope")
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _graph_scope(self, graph):
        self.pools.append(self.pool)
        self.events.append("capture-begin")
        yield
        graph.captured = True
        self.events.append("capture-end")

    def _memory_probe(self):
        # Each capture probes twice; the graph pool grows by pool_growth.
        self.probe_calls += 1
        self.events.append("probe")
        base = 100 * MIB
        return (base + (self.probe_calls % 2 == 0) * self.pool_growth, 0)

    def _synchronize(self):
        self.events.append("sync")

    def _restore_regions(self, regions, saved):
        self.events.append("restore")
        super()._restore_regions(regions, saved)

    def _free_bytes(self):
        return 0


class Recorder:
    """Collects the calls of every kernel the graph body touches."""

    def __init__(self):
        self.calls: list[tuple] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self.calls

        def temperature(logits, expanded_idx_mapping, temperature):
            calls.append(
                (
                    "temperature",
                    tuple(logits.shape),
                    logits.dtype,
                    expanded_idx_mapping.tolist(),
                    temperature.tolist(),
                )
            )

        def min_p(logits, expanded_idx_mapping, min_p):
            calls.append(
                (
                    "min_p",
                    tuple(logits.shape),
                    logits.dtype,
                    expanded_idx_mapping.tolist(),
                    min_p.tolist(),
                )
            )

        def top_k_top_p(logits, k, p, **kwargs):
            calls.append(
                (
                    "top_k_top_p",
                    tuple(logits.shape),
                    logits.dtype,
                    None if k is None else k.tolist(),
                    None if p is None else p.tolist(),
                )
            )
            return logits

        def num_sampled_and_rejected(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
        ):
            calls.append(
                (
                    "num_sampled_and_rejected",
                    seq_lens,
                    cu_num_logits,
                    idx_mapping,
                    prefill_len,
                )
            )
            return num_sampled, torch.zeros_like(num_sampled)

        for module in (cg, states_module):
            monkeypatch.setattr(module, "apply_temperature", temperature)
            monkeypatch.setattr(module, "apply_min_p", min_p)
            monkeypatch.setattr(module, "apply_top_k_top_p", top_k_top_p)
        monkeypatch.setattr(
            cg, "get_num_sampled_and_rejected", num_sampled_and_rejected
        )

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeLogger:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def debug(self, *args, **kwargs):
        pass


class Env(SimpleNamespace):
    pass


def make_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_graph_reqs: int = 1,
    model_state: Any = None,
    capture: bool = True,
    vocab: int = VOCAB,
    signatures: tuple = cg.DEFAULT_CAPTURE_SIGNATURES,
    branchfree: str = "2",
    scribble: bool = False,
) -> Env:
    # Mode 2 is the eager path whose numerics the graph reproduces; any other
    # mode makes capture() warn, which several tests count warnings around.
    monkeypatch.setenv("VLLM_SM70_TOPK_TOPP_BRANCHFREE", branchfree)
    recorder = Recorder()
    recorder.install(monkeypatch)
    fake_logger = FakeLogger()
    monkeypatch.setattr(cg, "logger", fake_logger)
    sampler = make_sampler(vocab)
    rejection = FakeRejection(recorder.calls)
    post_update_calls: list[tuple] = []
    model_state_calls: list[tuple] = []
    # Distinct non-zero values: a slot may hold live state that must survive.
    req_states = SimpleNamespace(
        prefill_len=FakeUva(np.arange(MAX_NUM_REQS) + 50, np.int32),
        num_computed_tokens=SimpleNamespace(
            gpu=torch.arange(MAX_NUM_REQS, dtype=torch.int32) + 100
        ),
        total_len=SimpleNamespace(
            gpu=torch.arange(MAX_NUM_REQS, dtype=torch.int32) + 200
        ),
        last_sampled_tokens=torch.arange(MAX_NUM_REQS, dtype=torch.int64).view(-1, 1)
        + 300,
        all_token_ids=SimpleNamespace(
            gpu=torch.arange(MAX_NUM_REQS * 32, dtype=torch.int32).view(
                MAX_NUM_REQS, 32
            )
        ),
    )
    input_buffers = InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, CPU)
    if model_state is None:
        model_state = StaticModelState()

    def post_update_fn(*args):
        post_update_calls.append(args)
        if scribble:
            # What post_update does to the dummy slots during the warmup.
            idx = args[0].long()
            bins = sampler.penalties_state.output_bin_counts
            bins[idx, 3] += 1
            req_states.num_computed_tokens.gpu[idx] += 5
            req_states.total_len.gpu[idx] += 5
            req_states.last_sampled_tokens[idx] = 9
            req_states.all_token_ids.gpu[idx, :5] = 4

    def postprocess_state_fn(*args):
        model_state_calls.append(args)
        accepted = getattr(model_state, "num_accepted_tokens_gpu", None)
        if scribble and accepted is not None:
            accepted[args[0].long()] = 5

    manager = CpuManager(
        sampler=sampler,
        rejection_sampler=rejection,
        req_states=req_states,
        input_buffers=input_buffers,
        model_state=model_state,
        post_update_fn=post_update_fn,
        postprocess_state_fn=postprocess_state_fn,
        device=CPU,
        max_num_reqs=MAX_NUM_REQS,
        num_speculative_steps=K,
        vocab_size=vocab,
        max_graph_reqs=max_graph_reqs,
        signatures=signatures,
    )
    if capture:
        manager.capture()
    return Env(
        manager=manager,
        sampler=sampler,
        recorder=recorder,
        logger=fake_logger,
        rejection=rejection,
        req_states=req_states,
        input_buffers=input_buffers,
        post_update_calls=post_update_calls,
        model_state_calls=model_state_calls,
        batch=make_batch(1),
        logits=torch.zeros(PER_REQ, vocab, dtype=torch.float16),
        draft_logits=None,
        grammar=None,
    )


class StaticModelState:
    """A model state whose postprocess is the interface no-op."""

    postprocess_state = ModelState.postprocess_state


def admission(env: Env):
    return env.manager.check_admission(
        env.logits, env.batch, env.grammar, env.draft_logits
    )


# ------------------------------------------------------- signature and env


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "min_p", "expected"),
    [
        # The bench / SWE harness: temperature 1.0, top_k 20, top_p 0.95.
        ([1.0], [20], [0.95], [0.0], (True, True, False, False)),
        # temperature 0 and 1 are both "no temperature kernel".
        ([0.0, 1.0], [20, 20], [0.95, 0.95], [0.0, 0.0], (True, True, False, False)),
        ([0.7], [20], [0.95], [0.0], (True, True, True, False)),
        # Any request in the batch with a non-trivial value flips the flag.
        ([1.0, 0.6], [20, 20], [0.95, 0.95], [0.0, 0.0], (True, True, True, False)),
        ([1.0], [VOCAB], [0.95], [0.0], (False, True, False, False)),
        ([1.0], [20], [1.0], [0.0], (True, False, False, False)),
        ([1.0], [VOCAB], [1.0], [0.0], (False, False, False, False)),
        ([1.0], [20], [0.95], [0.05], (True, True, False, True)),
        ([1.0, 1.0], [VOCAB, 20], [1.0, 1.0], [0.0, 0.0], (True, False, False, False)),
    ],
)
def test_compute_signature_mirrors_the_eager_host_branches(
    temperature, top_k, top_p, min_p, expected
):
    sampler = make_sampler()
    n = len(temperature)
    states = sampler.sampling_states
    states.temperature.np[:n] = temperature
    states.top_k.np[:n] = top_k
    states.top_p.np[:n] = top_p
    states.min_p.np[:n] = min_p
    signature = cg.compute_signature(sampler, np.arange(n, dtype=np.int32))
    assert signature == cg.SamplingSignature(*expected)


def test_signature_ignores_requests_outside_the_batch():
    sampler = make_sampler()
    sampler.sampling_states.temperature.np[5] = 0.3  # a slot not in the batch
    signature = cg.compute_signature(sampler, np.array([0], dtype=np.int32))
    assert signature == SIG


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0), ("0", 0), ("1", 1)],
)
def test_env_sampling_cudagraph_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_SM70_SAMPLING_CUDAGRAPH", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH", value)
    assert expected == envs.VLLM_SM70_SAMPLING_CUDAGRAPH
    assert isinstance(envs.VLLM_SM70_SAMPLING_CUDAGRAPH, int)


@pytest.mark.parametrize("value", ["2", "-1", "true", "yes", "", "1.0", "on"])
def test_env_sampling_cudagraph_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH", value)
    with pytest.raises(ValueError, match="VLLM_SM70_SAMPLING_CUDAGRAPH"):
        _ = envs.VLLM_SM70_SAMPLING_CUDAGRAPH


@pytest.mark.parametrize(("value", "expected"), [(None, 1), ("1", 1), ("4", 4)])
def test_env_max_reqs_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS", value)
    assert expected == envs.VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS


@pytest.mark.parametrize("value", ["0", "-3", "abc", "", "1.5"])
def test_env_max_reqs_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS", value)
    with pytest.raises(ValueError):
        _ = envs.VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS


# ------------------------------------------------------------------ admission


def test_admits_the_bench_batch_with_the_expected_key(monkeypatch):
    env = make_env(monkeypatch)
    key, reason = admission(env)
    assert reason is None
    assert key == cg.SamplingGraphKey(1, PER_REQ, SIG)
    assert env.manager.keys == (key,)


def test_greedy_temperatures_mixed_with_one_are_admitted(monkeypatch):
    env = make_env(monkeypatch)
    env.sampler.sampling_states.temperature.np[0] = 0.0
    key, reason = admission(env)
    assert reason is None and key is not None


def _set(path: str, value):
    def mutate(env: Env):
        obj = env
        *parents, leaf = path.split(".")
        for name in parents:
            obj = getattr(obj, name)
        if isinstance(obj, np.ndarray):
            obj[int(leaf)] = value
        else:
            setattr(obj, leaf, value)

    return mutate


def _shape_logits(env: Env):
    env.logits = torch.zeros(PER_REQ, VOCAB + 1, dtype=torch.float16)


def _fewer_drafts(env: Env):
    env.batch.num_draft_tokens_per_req = np.array([K - 1], dtype=np.int32)
    env.batch.num_draft_tokens = K - 1


def _no_drafts(env: Env):
    env.batch.num_draft_tokens_per_req = None
    env.batch.num_draft_tokens = 0


def _token_ids(env: Env):
    env.sampler.logprob_token_ids_state.state["n"] = 3


def _top_k_only(env: Env):
    env.sampler.sampling_states.top_p.np[0] = 1.0


def _top_p_only(env: Env):
    env.sampler.sampling_states.top_k.np[0] = VOCAB


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_fewer_drafts, cg.REASON_NOT_UNIFORM),
        (_no_drafts, cg.REASON_NOT_UNIFORM),
        (_set("draft_logits", torch.zeros(1)), cg.REASON_DRAFT_LOGITS),
        (_set("grammar", object()), cg.REASON_GRAMMAR),
        (_set("batch.has_structured_output_reqs", True), cg.REASON_GRAMMAR),
        (_set("sampler.compute_nans", True), cg.REASON_COMPUTE_NANS),
        (_set("batch.is_prefilling_np.0", True), cg.REASON_PREFILL),
        (_set("sampler.sampling_states.num_logprobs.0", 0), cg.REASON_LOGPROBS),
        (_set("sampler.sampling_states.num_logprobs.0", 5), cg.REASON_LOGPROBS),
        (_token_ids, cg.REASON_LOGPROBS),
        (
            _set("sampler.logit_bias_state.use_logit_bias.0", True),
            cg.REASON_LOGIT_BIAS,
        ),
        (_set("sampler.penalties_state.use_penalty.0", True), cg.REASON_PENALTIES),
        (
            _set("sampler.bad_words_state.num_bad_words.np.0", 2),
            cg.REASON_BAD_WORDS,
        ),
        (_top_k_only, cg.REASON_ONE_SIDED),
        (_top_p_only, cg.REASON_ONE_SIDED),
        # Signatures other than the captured one.
        (_set("sampler.sampling_states.temperature.np.0", 0.7), cg.REASON_SIGNATURE),
        (_set("sampler.sampling_states.min_p.np.0", 0.1), cg.REASON_SIGNATURE),
        (_shape_logits, cg.REASON_LOGITS_SHAPE),
    ],
    ids=lambda v: getattr(v, "__name__", None) or str(v)[:48],
)
def test_every_rejected_batch_reports_its_reason(monkeypatch, mutate, reason):
    env = make_env(monkeypatch)
    # Sanity: the unmutated batch is admitted.
    assert admission(env)[1] is None
    mutate(env)
    key, got = admission(env)
    assert key is None
    assert got == reason


def test_unset_top_k_and_top_p_signature_is_not_captured_by_default(monkeypatch):
    env = make_env(monkeypatch)
    env.sampler.sampling_states.top_k.np[0] = VOCAB
    env.sampler.sampling_states.top_p.np[0] = 1.0
    assert admission(env) == (None, cg.REASON_SIGNATURE)


def test_extra_signatures_can_be_captured(monkeypatch):
    sampled = cg.SamplingSignature(
        top_k=True, top_p=True, temperature=True, min_p=False
    )
    env = make_env(monkeypatch, signatures=(SIG, sampled))
    assert [k.signature for k in env.manager.keys] == [SIG, sampled]
    env.sampler.sampling_states.temperature.np[0] = 0.7
    key, reason = admission(env)
    assert reason is None and key.signature == sampled


def test_nothing_is_admitted_before_capture(monkeypatch):
    env = make_env(monkeypatch, capture=False)
    assert admission(env) == (None, cg.REASON_NO_GRAPH_FOR_SHAPE)


# --------------------------------------------------------------- MAX_REQS bound


def test_max_reqs_default_captures_only_the_single_request_shape(monkeypatch):
    env = make_env(monkeypatch)
    assert [(k.num_reqs, k.num_logits) for k in env.manager.keys] == [(1, 5)]
    assert set(env.manager.graphs) == set(env.manager.keys)


def test_max_reqs_bound_admits_up_to_the_bound_and_rejects_above(monkeypatch):
    env = make_env(monkeypatch, max_graph_reqs=3)
    assert [(k.num_reqs, k.num_logits) for k in env.manager.keys] == [
        (1, 5),
        (2, 10),
        (3, 15),
    ]
    for num_reqs in (1, 2, 3):
        env.batch = make_batch(num_reqs)
        env.logits = torch.zeros(num_reqs * PER_REQ, VOCAB, dtype=torch.float16)
        key, reason = admission(env)
        assert reason is None
        assert (key.num_reqs, key.num_logits) == (num_reqs, num_reqs * PER_REQ)
    env.batch = make_batch(4)
    env.logits = torch.zeros(4 * PER_REQ, VOCAB, dtype=torch.float16)
    assert admission(env) == (None, cg.REASON_TOO_MANY_REQS)


def test_max_reqs_is_clamped_to_max_num_seqs(monkeypatch):
    env = make_env(monkeypatch, max_graph_reqs=100, capture=False)
    assert env.manager.max_graph_reqs == MAX_NUM_REQS
    assert len(env.manager.keys) == MAX_NUM_REQS
    assert any("outside" in w for w in env.logger.warnings)


# ------------------------------------------------------------ one-time fallback


def test_each_fallback_reason_is_warned_once(monkeypatch):
    env = make_env(monkeypatch)
    env.sampler.compute_nans = True
    for _ in range(5):
        assert env.manager.run(env.logits, env.batch, None, None) is None
    assert env.manager.num_fallbacks == 5
    reasons = [w for w in env.logger.warnings if cg.REASON_COMPUTE_NANS in w]
    assert len(reasons) == 1
    assert len(env.logger.warnings) == 1

    # A different reason is a new line, repeated calls are not.
    env.sampler.compute_nans = False
    env.sampler.penalties_state.use_penalty[0] = True
    for _ in range(3):
        assert env.manager.run(env.logits, env.batch, None, None) is None
    assert len(env.logger.warnings) == 2
    assert cg.REASON_PENALTIES in env.logger.warnings[1]
    assert env.manager.num_replays == 0


def test_the_warning_text_is_the_same_for_every_batch(monkeypatch):
    """Per-batch numbers in the message would defeat the once-per-reason log."""
    env = make_env(monkeypatch)
    env.sampler.penalties_state.use_penalty[0] = True
    for slots in ([0], [1], [2]):
        env.batch = make_batch(1, slots=slots)
        env.sampler.penalties_state.use_penalty[slots[0]] = True
        assert env.manager.run(env.logits, env.batch, None, None) is None
    assert len(env.logger.warnings) == 1


def test_admitted_batches_log_no_warning(monkeypatch):
    env = make_env(monkeypatch)
    out = env.manager.run(env.logits, env.batch, None, None)
    assert out is not None
    assert env.logger.warnings == []
    assert any("replay active" in m for m in env.logger.infos)


# ------------------------------------------------------------- static buffers


def test_static_buffer_sizes_for_the_single_request_shape(monkeypatch):
    vocab = 248320  # the real vocabulary; 5 x 248320 fp32 is only 5 MB
    env = make_env(monkeypatch, vocab=vocab)
    bufs = env.manager.bufs
    assert bufs.logits.shape == (PER_REQ, vocab)
    assert bufs.logits.dtype == torch.float32
    assert bufs.idx_mapping.shape == (1,) and bufs.idx_mapping.dtype == torch.int32
    assert bufs.cu_num_logits.shape == (2,)
    assert bufs.expanded_idx_mapping.shape == (PER_REQ,)
    assert bufs.expanded_local_pos.shape == (PER_REQ,)
    assert bufs.logits_indices.shape == (PER_REQ,)
    assert bufs.logits_indices.dtype == torch.int64
    for name in ("temperature", "seeds", "top_k", "top_p", "min_p", "prefill_len"):
        assert getattr(bufs, name).shape == (MAX_NUM_REQS,)
    expected = (
        PER_REQ * vocab * 4  # logits
        + 1 * 4  # idx_mapping
        + 2 * 4  # cu_num_logits
        + PER_REQ * 4  # expanded_idx_mapping
        + PER_REQ * 4  # expanded_local_pos
        + PER_REQ * 8  # logits_indices
        + MAX_NUM_REQS * (4 + 8 + 4 + 4 + 4 + 4)  # sampling-state mirrors
    )
    assert env.manager.static_nbytes() == expected
    assert expected == 4_966_400 + 4 + 8 + 20 + 20 + 40 + MAX_NUM_REQS * 28


def test_static_buffers_are_shared_across_shapes_and_sized_for_the_largest(
    monkeypatch,
):
    env = make_env(monkeypatch, max_graph_reqs=3)
    bufs = env.manager.bufs
    assert bufs.logits.shape == (3 * PER_REQ, VOCAB)
    assert bufs.idx_mapping.shape == (3,)
    assert bufs.cu_num_logits.shape == (4,)


def test_replay_stages_every_input_into_the_static_buffers(monkeypatch):
    env = make_env(monkeypatch)
    sampler = env.sampler
    states = sampler.sampling_states
    states.temperature.np[:] = 1.0
    states.seeds.np[2] = 424242
    states.temperature.rotate()
    states.seeds.rotate()
    env.batch = make_batch(1, slots=[2])
    env.logits = torch.randn(PER_REQ, VOCAB).to(torch.float16)
    out = env.manager.run(env.logits, env.batch, None, None)
    assert out is not None
    bufs = env.manager.bufs
    assert torch.equal(bufs.logits, env.logits.to(torch.float32))
    assert bufs.idx_mapping.tolist() == [2]
    assert bufs.cu_num_logits.tolist() == [0, PER_REQ]
    assert bufs.expanded_idx_mapping.tolist() == [2] * PER_REQ
    assert bufs.expanded_local_pos.tolist() == list(range(PER_REQ))
    assert bufs.logits_indices.tolist() == [3, 4, 5, 6, 7]
    assert bufs.seeds[2].item() == 424242
    assert torch.equal(bufs.temperature, states.temperature.gpu)
    assert torch.equal(bufs.top_k, states.top_k.gpu)
    assert torch.equal(bufs.top_p, states.top_p.gpu)
    assert torch.equal(bufs.prefill_len, env.req_states.prefill_len.gpu)
    assert env.manager.made[0].replays == 1


def test_sampling_state_arrays_are_restaged_on_every_step(monkeypatch):
    """UvaBackedTensor.gpu is re-pointed every step; a baked pointer goes stale.

    The graph must read the staged copy of the *current* ``.gpu`` each replay.
    """
    env = make_env(monkeypatch)
    states = env.sampler.sampling_states
    bufs = env.manager.bufs

    assert env.manager.run(env.logits, env.batch, None, None) is not None
    assert bufs.seeds[0].item() == 1000
    old_gpu = states.seeds.gpu

    # Step 2: a new request changed the seed; apply_staged_writes rotated .gpu.
    states.seeds.np[0] = 777
    states.top_p.np[0] = 0.9
    states.seeds.rotate()
    states.top_p.rotate()
    assert states.seeds.gpu.data_ptr() != old_gpu.data_ptr()
    assert env.manager.run(env.logits, env.batch, None, None) is not None
    assert bufs.seeds[0].item() == 777
    assert bufs.top_p[0].item() == pytest.approx(0.9)

    # prefill_len rotates when a request is added.
    env.req_states.prefill_len.np[0] = 99
    env.req_states.prefill_len.rotate()
    assert env.manager.run(env.logits, env.batch, None, None) is not None
    assert bufs.prefill_len[0].item() == 99


def test_graph_only_stages_the_arrays_its_signature_reads(monkeypatch):
    env = make_env(monkeypatch)
    bufs = env.manager.bufs
    bufs.min_p.fill_(-1.0)  # canary: min_p is not in the captured signature
    states = env.sampler.sampling_states
    states.min_p.np[:] = 0.0
    states.min_p.rotate()
    assert env.manager.run(env.logits, env.batch, None, None) is not None
    assert torch.all(bufs.min_p == -1.0)


def test_outputs_are_clones_not_graph_pool_tensors(monkeypatch):
    env = make_env(monkeypatch)
    manager = env.manager
    key = manager.keys[0]
    held = manager.graphs[key].outputs
    held[0].fill_(7)
    held[1].fill_(3)
    held[2].fill_(1)

    out = manager.run(env.logits, env.batch, None, None)
    assert out is not None
    sampled, num_sampled, num_rejected = (
        out.sampler_output.sampled_token_ids,
        out.num_sampled,
        out.num_rejected,
    )
    assert sampled.data_ptr() != held[0].data_ptr()
    assert num_sampled.data_ptr() != held[1].data_ptr()
    assert num_rejected.data_ptr() != held[2].data_ptr()
    assert sampled.tolist() == [[7] * PER_REQ]
    assert num_sampled.tolist() == [3]

    # The next replay overwrites the held outputs; the handed-out ones survive.
    for t in held:
        t.fill_(99)
    assert sampled.tolist() == [[7] * PER_REQ]
    assert num_sampled.tolist() == [3]
    assert num_rejected.tolist() == [1]
    # SamplerOutput and the returned num_sampled are one tensor, as in eager.
    assert out.sampler_output.num_sampled is out.num_sampled
    assert out.sampler_output.logprobs_tensors is None
    assert out.sampler_output.num_nans is None


# ------------------------------------------------------------- capture protocol


def test_capture_warms_up_then_captures_each_key_in_one_shared_pool(monkeypatch):
    env = make_env(monkeypatch, max_graph_reqs=2, capture=False)
    manager = env.manager
    manager.capture()

    assert manager.events[0] == "outer-scope"
    # Per key: eager warmup, sync, capture.
    body_markers = [e for e in manager.events if e not in ("outer-scope", "probe")]
    # Per key: warmup, sync, capture, then the restore of the warmup's writes
    # (which synchronizes the capture side stream).
    assert body_markers == [
        "sync",
        "capture-begin",
        "capture-end",
        "restore",
        "sync",
        "sync",
        "capture-begin",
        "capture-end",
        "restore",
        "sync",
    ]
    rejections = [c for c in env.recorder.calls if c[0] == "rejection"]
    assert len(rejections) == 2 * len(manager.keys)  # warmup + capture per key
    assert len(manager.made) == len(manager.keys)
    assert all(g.captured for g in manager.made)
    assert len({id(p) for p in manager.pools}) == 1  # one dedicated pool
    assert manager.pool is manager.pools[0]
    assert set(manager.graphs) == set(manager.keys)
    for key, captured in manager.graphs.items():
        sampled, num_sampled, num_rejected = captured.outputs
        assert sampled.shape == (key.num_reqs, PER_REQ)
        assert num_sampled.shape == (key.num_reqs,)
        assert num_rejected.shape == (key.num_reqs,)
    # Capturing twice is a no-op.
    manager.capture()
    assert len(manager.made) == len(manager.keys)


def test_capture_logs_the_measured_pool_bytes_once_at_info(monkeypatch):
    env = make_env(monkeypatch, capture=False)
    env.manager.pool_growth = 25 * MIB
    env.manager.capture()
    assert env.manager.pool_reserved_bytes() == 25 * MIB
    messages = [m for m in env.logger.infos if "captured" in m]
    assert len(messages) == 1
    assert "Graph pool reserved 25.00 MiB" in messages[0]
    assert "static input buffers" in messages[0]


def _full_state(env: Env, model_state) -> dict[str, torch.Tensor]:
    state = {
        "output_bin_counts": env.sampler.penalties_state.output_bin_counts,
        "num_computed_tokens": env.req_states.num_computed_tokens.gpu,
        "total_len": env.req_states.total_len.gpu,
        "last_sampled_tokens": env.req_states.last_sampled_tokens,
        "all_token_ids": env.req_states.all_token_ids.gpu,
    }
    if hasattr(model_state, "num_accepted_tokens_gpu"):
        state["num_accepted_tokens_gpu"] = model_state.num_accepted_tokens_gpu
    return {name: tensor.clone() for name, tensor in state.items()}


@pytest.mark.parametrize("max_graph_reqs", [1, 3])
@pytest.mark.parametrize("align", [False, True], ids=["in-graph", "eager-tail"])
def test_capture_restores_the_persistent_state_the_warmup_wrote(
    monkeypatch, max_graph_reqs, align
):
    """Regression: the warmup's post_update left a count in output_bin_counts.

    PenaltiesState.add_request only rewrites the row for requests that use
    penalties, so the residue stayed forever and the eager rig, which never ran
    a warmup, differed from the graph rig in the GPU replay check.
    """
    model_state = HybridLikeModelState(align=align)
    env = make_env(
        monkeypatch,
        max_graph_reqs=max_graph_reqs,
        model_state=model_state,
        capture=False,
        scribble=True,
    )
    before = _full_state(env, model_state)
    env.manager.capture()
    after = _full_state(env, model_state)

    assert env.manager.model_state_in_graph is (not align)
    # The warmup really ran post_update (and the model-state postprocess when it
    # is in the graph), once per key for the warmup and once during the capture.
    assert len(env.post_update_calls) == 2 * len(env.manager.keys)
    assert len(env.model_state_calls) == (0 if align else 2 * len(env.manager.keys))
    for name, want in before.items():
        assert torch.equal(after[name], want), name


def test_without_the_restore_the_scribbled_state_would_remain(monkeypatch):
    """Negative control: the test above is only meaningful if scribbling shows."""
    model_state = HybridLikeModelState()
    env = make_env(monkeypatch, model_state=model_state, capture=False, scribble=True)
    monkeypatch.setattr(env.manager, "_restore_regions", lambda regions, saved: None)
    before = _full_state(env, model_state)
    env.manager.capture()
    after = _full_state(env, model_state)
    for name in before:
        assert not torch.equal(after[name], before[name]), name


def test_the_restored_list_covers_every_region_post_update_writes(monkeypatch):
    model_state = HybridLikeModelState()
    env = make_env(monkeypatch, model_state=model_state, capture=False)
    env.manager.model_state_in_graph = True
    regions = env.manager._persistent_regions(2)
    got = {tensor.data_ptr(): rows for tensor, rows in regions}
    want = {
        "output_bin_counts": env.sampler.penalties_state.output_bin_counts,
        "num_computed_tokens": env.req_states.num_computed_tokens.gpu,
        "total_len": env.req_states.total_len.gpu,
        "last_sampled_tokens": env.req_states.last_sampled_tokens,
        "all_token_ids": env.req_states.all_token_ids.gpu,
        "num_accepted_tokens_gpu": model_state.num_accepted_tokens_gpu,
    }
    assert len(regions) == len(want)
    for name, tensor in want.items():
        assert got[tensor.data_ptr()] == slice(0, 2), name
    # The model-state buffer is left out when its postprocess is not in the graph.
    env.manager.model_state_in_graph = False
    tail_regions = env.manager._persistent_regions(2)
    assert len(tail_regions) == len(want) - 1
    assert model_state.num_accepted_tokens_gpu.data_ptr() not in {
        tensor.data_ptr() for tensor, _ in tail_regions
    }


def test_output_bin_counts_is_restored_for_the_dummy_slots(monkeypatch):
    """The exact field the GPU replay check flagged, one dummy slot."""
    env = make_env(monkeypatch, capture=False, scribble=True)
    bins = env.sampler.penalties_state.output_bin_counts
    before = bins.clone()
    env.manager.capture()
    assert torch.equal(bins, before)
    assert env.manager._persistent_regions(1)[0][0] is bins


def test_model_state_regions_follow_the_postprocess_writes():
    scratch = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    accepted = torch.ones(MAX_NUM_REQS, dtype=torch.int32)
    ready = HybridLikeModelState(align=True, ctx=_Ctx(True))
    ready.num_accepted_tokens_gpu = accepted
    ready._mamba_ctx.num_accepted_tokens_out = scratch
    regions = cg.model_state_persistent_regions(ready, 2)
    assert [(t.data_ptr(), r) for t, r in regions] == [
        (accepted.data_ptr(), slice(0, 2)),
        (scratch.data_ptr(), slice(None)),
    ]
    # No context yet (align mode before the first forward): only the counts.
    ready._mamba_ctx = None
    assert len(cg.model_state_persistent_regions(ready, 2)) == 1
    assert cg.model_state_persistent_regions(StaticModelState(), 2) == []


def test_pool_memory_is_probed_inside_the_graph_scope(monkeypatch):
    """torch.cuda.graph.__enter__ runs empty_cache(), which would be subtracted
    from the pool's growth if the probe ran before entering the scope."""
    env = make_env(monkeypatch, max_graph_reqs=2, capture=False)
    env.manager.capture()
    marked = [e for e in env.manager.events if e != "outer-scope"]
    for start in range(0, len(marked), 7):
        assert marked[start : start + 7] == [
            "sync",
            "capture-begin",
            "probe",
            "probe",
            "capture-end",
            "restore",
            "sync",
        ]
    assert marked.count("probe") == 2 * len(env.manager.keys)


@pytest.mark.parametrize("mode", ["0", "1"])
def test_capture_warns_when_eager_top_k_top_p_is_not_the_reference(monkeypatch, mode):
    env = make_env(monkeypatch, branchfree=mode)
    assert any(f"BRANCHFREE={mode}" in w for w in env.logger.warnings)


def test_capture_is_quiet_when_eager_matches_the_graph_numerics(monkeypatch):
    env = make_env(monkeypatch, branchfree="2")
    assert env.logger.warnings == []


def test_the_body_reads_only_static_buffers_and_persistent_input_buffers(
    monkeypatch,
):
    env = make_env(monkeypatch)
    manager = env.manager
    bufs = manager.bufs
    buffers = env.input_buffers
    key = manager.keys[0]
    buffers.input_ids[:] = torch.arange(MAX_NUM_TOKENS, dtype=torch.int32) * 3
    buffers.positions[:] = torch.arange(MAX_NUM_TOKENS, dtype=torch.int64) + 1000
    env.batch = make_batch(1)
    manager._stage_inputs(key, env.logits, env.batch)
    env.recorder.calls.clear()
    env.post_update_calls.clear()

    manager._run_body(key)

    rejection = next(c for c in env.recorder.calls if c[0] == "rejection")
    (
        processed,
        draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seeds,
    ) = rejection[1]
    assert processed.data_ptr() == bufs.logits.data_ptr()  # processed in place
    assert draft_logits is None
    # Gathers use the static logits_indices (3..7 from the staged batch).
    assert draft_sampled.tolist() == [9, 12, 15, 18, 21]
    assert pos.tolist() == [1003, 1004, 1005, 1006, 1007]
    for got, static in (
        (cu_num_logits, bufs.cu_num_logits),
        (idx_mapping, bufs.idx_mapping),
        (expanded_idx_mapping, bufs.expanded_idx_mapping),
        (expanded_local_pos, bufs.expanded_local_pos),
        (temperature, bufs.temperature),
        (seeds, bufs.seeds),
    ):
        assert got.data_ptr() == static.data_ptr()

    ns = next(c for c in env.recorder.calls if c[0] == "num_sampled_and_rejected")
    _, seq_lens, cu, idx, prefill_len = ns
    assert seq_lens.data_ptr() == buffers.seq_lens.data_ptr()  # persistent buffer
    assert seq_lens.shape == (1,)
    assert cu.data_ptr() == bufs.cu_num_logits.data_ptr()
    assert idx.data_ptr() == bufs.idx_mapping.data_ptr()
    # UvaBackedTensor.gpu rotates, so prefill_len must be the staged copy.
    assert prefill_len.data_ptr() == bufs.prefill_len.data_ptr()
    assert prefill_len.data_ptr() != env.req_states.prefill_len.gpu.data_ptr()

    (post_update,) = env.post_update_calls
    p_idx, p_sampled, p_num_sampled, p_num_rejected, p_qsl = post_update
    assert p_idx.data_ptr() == bufs.idx_mapping.data_ptr()
    assert p_qsl.data_ptr() == buffers.query_start_loc.data_ptr()
    assert p_qsl.shape == (2,)


# ------------------------------------------------- model-state postprocess tail


class _Ctx:
    def __init__(self, initialized: bool):
        self.is_initialized = initialized


class AlignLikeModelState:
    """The attributes MambaHybridModelState.postprocess_state consults."""

    def __init__(self, align: bool, ctx: Any):
        self._align_mode = align
        self._mamba_ctx = ctx

    def postprocess_state(self, *args):  # an override, not the interface no-op
        pass


class HybridLikeModelState(AlignLikeModelState):
    """Adds the buffer MambaHybridModelState.postprocess_state writes."""

    def __init__(self, align: bool = False, ctx: Any = None):
        super().__init__(align, ctx)
        self.num_accepted_tokens_gpu = torch.arange(MAX_NUM_REQS, dtype=torch.int32) + 7


class UnknownModelState:
    def postprocess_state(self, *args):
        pass


@pytest.mark.parametrize(
    ("model_state", "expected"),
    [
        (StaticModelState(), True),
        (AlignLikeModelState(False, None), True),
        # Align mode before the first real forward: the context does not exist.
        (AlignLikeModelState(True, None), False),
        (AlignLikeModelState(True, _Ctx(False)), False),
        (AlignLikeModelState(True, _Ctx(True)), True),
        (UnknownModelState(), False),
    ],
    ids=[
        "interface-noop",
        "mamba-no-align",
        "align-ctx-missing",
        "align-ctx-uninitialized",
        "align-ctx-ready",
        "unknown-override",
    ],
)
def test_model_state_postprocess_capture_safety(model_state, expected):
    assert cg.model_state_postprocess_is_capture_safe(model_state) is expected


def test_probe_on_the_real_mamba_hybrid_state_class():
    mamba = pytest.importorskip("vllm.v1.worker.gpu.model_states.mamba_hybrid")
    state = object.__new__(mamba.MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    assert cg.model_state_postprocess_is_capture_safe(state) is False
    state._mamba_ctx = _Ctx(True)
    assert cg.model_state_postprocess_is_capture_safe(state) is True
    state._align_mode = False
    state._mamba_ctx = None
    assert cg.model_state_postprocess_is_capture_safe(state) is True


def test_postprocess_state_is_captured_only_when_safe(monkeypatch):
    safe = make_env(monkeypatch, model_state=AlignLikeModelState(False, None))
    assert safe.manager.model_state_in_graph is True
    # Warmup and capture each ran the model-state postprocess once.
    assert len(safe.model_state_calls) == 2
    out = safe.manager.run(safe.logits, safe.batch, None, None)
    assert out.post_update_done and out.model_state_done

    unsafe = make_env(monkeypatch, model_state=AlignLikeModelState(True, None))
    assert unsafe.manager.model_state_in_graph is False
    assert unsafe.model_state_calls == []
    assert len(unsafe.post_update_calls) == 2  # post_update is always in the graph
    out = unsafe.manager.run(unsafe.logits, unsafe.batch, None, None)
    assert out.post_update_done and not out.model_state_done
    assert any("runs eagerly after each replay" in m for m in unsafe.logger.infos)


# ---------------------------------------------- body == Sampler.apply_sampling_params


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "min_p"),
    [
        (1.0, 20, 0.95, 0.0),
        (0.7, 20, 0.95, 0.0),
        (0.7, 20, 0.95, 0.05),
        (1.0, VOCAB, 1.0, 0.0),
        (0.0, 20, 0.95, 0.0),
        (1.0, 20, 1.0, 0.0),
        (1.0, VOCAB, 0.8, 0.1),
    ],
)
def test_graph_body_applies_the_same_steps_as_the_eager_sampler(
    monkeypatch, temperature, top_k, top_p, min_p
):
    """Pin the body to Sampler.apply_sampling_params: order and arguments.

    The real ``Sampler.apply_sampling_params`` and the real ``SamplingStates``
    host branches run against recorded kernels; the graph body must make the
    same sequence of calls with equal arguments.
    """
    signature = None
    env = make_env(monkeypatch, capture=False)
    states = env.sampler.sampling_states
    states.temperature.np[0] = temperature
    states.top_k.np[0] = top_k
    states.top_p.np[0] = top_p
    states.min_p.np[0] = min_p
    for uva in (states.temperature, states.top_k, states.top_p, states.min_p):
        uva.rotate()
    idx_np = np.array([0], dtype=np.int32)
    signature = cg.compute_signature(env.sampler, idx_np)

    batch = make_batch(1)
    logits = torch.randn(PER_REQ, VOCAB).to(torch.float16)
    pos = torch.arange(PER_REQ, dtype=torch.int64)
    input_ids = torch.arange(PER_REQ, dtype=torch.int32)

    env.recorder.calls.clear()
    Sampler.apply_sampling_params(
        env.sampler,
        logits,
        batch.expanded_idx_mapping,
        idx_np,
        pos,
        input_ids,
        batch.expanded_local_pos,
    )
    eager_calls = list(env.recorder.calls)

    key = cg.SamplingGraphKey(1, PER_REQ, signature)
    manager = env.manager
    manager.bufs = cg._StaticBuffers(1, PER_REQ, MAX_NUM_REQS, VOCAB, CPU)
    manager._stage_inputs(key, logits, batch)
    env.recorder.calls.clear()
    manager._apply_sampling_params(
        manager.bufs.logits[:PER_REQ],
        manager.bufs.expanded_idx_mapping[:PER_REQ],
        signature,
    )
    graph_calls = list(env.recorder.calls)

    assert graph_calls == eager_calls
    expected_names = []
    if signature.temperature:
        expected_names.append("temperature")
    if signature.min_p:
        expected_names.append("min_p")
    if signature.top_k or signature.top_p:
        expected_names.append("top_k_top_p")
    assert [c[0] for c in graph_calls] == expected_names


# ------------------------------------------------ RejectionSampler refactor


def test_rejection_sampler_call_and_the_processed_helper_agree(monkeypatch):
    """``__call__`` must pass rejection_sample exactly what the helper passes."""
    seen: list[tuple] = []

    def fake_rejection_sample(*args, **kwargs):
        seen.append((args, kwargs))
        return torch.zeros(1, PER_REQ, dtype=torch.int64), torch.ones(
            1, dtype=torch.int32
        )

    monkeypatch.setattr(rs_module, "rejection_sample", fake_rejection_sample)
    sampler = make_sampler()
    sampler.apply_sampling_params = lambda logits, *a: logits.to(torch.float32)
    rejection = object.__new__(RejectionSampler)
    rejection.sampler = sampler
    rejection.num_speculative_steps = K
    rejection.synthetic_conditional_rates = None

    batch = make_batch(1)
    batch.input_ids = torch.arange(MAX_NUM_TOKENS, dtype=torch.int32)
    batch.positions = torch.arange(MAX_NUM_TOKENS, dtype=torch.int64) + 500
    logits = torch.randn(PER_REQ, VOCAB)
    out = rejection(logits, batch, None)
    assert isinstance(out, SamplerOutput)
    assert out.logprobs_tensors is None

    draft_sampled = batch.input_ids[batch.logits_indices]
    pos = batch.positions[batch.logits_indices]
    rejection.rejection_sample_processed(
        logits.to(torch.float32),
        None,
        draft_sampled,
        pos,
        batch.cu_num_logits,
        batch.idx_mapping,
        batch.expanded_idx_mapping,
        batch.expanded_local_pos,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
    )
    (call_args, call_kwargs), (helper_args, helper_kwargs) = seen
    assert call_kwargs == helper_kwargs == {"use_fp64": False}
    assert len(call_args) == len(helper_args) == 12
    for got, want in zip(call_args, helper_args):
        if isinstance(got, torch.Tensor):
            assert torch.equal(got, want)
        else:
            assert got == want
    # rejection_sample's signature order: logits, draft_logits, draft_sampled,
    # cu_num_logits, pos, idx_mapping, expanded_idx_mapping, expanded_local_pos,
    # temperature, seed, num_speculative_steps, synthetic_conditional_rates.
    assert torch.equal(call_args[3], batch.cu_num_logits)
    assert torch.equal(call_args[4], pos)
    assert call_args[10] == K and call_args[11] is None


# ----------------------------------------------------------- model runner wiring


def _bare_runner(**attrs):
    runner = object.__new__(model_runner_module.GPUModelRunner)
    # Upstream's step timer, read at the end of every sample_tokens.
    runner.mixed_prefill_timer = SimpleNamespace(
        begin=lambda *_: None, finish=lambda: None
    )
    runner.lora_config = None
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def test_runner_defaults_keep_the_eager_path():
    runner = _bare_runner()
    assert runner.sampling_graph is None
    assert runner._sampling_graph_output is None


def test_capture_is_a_noop_when_the_env_is_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_SAMPLING_CUDAGRAPH", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("the manager must not be built when the env is 0")

    monkeypatch.setattr(model_runner_module, "SamplingCudaGraphManager", boom)
    runner = _bare_runner(sampling_graph="stale")
    runner._capture_sampling_graph()
    assert runner.sampling_graph is None


def test_capture_names_every_blocker_and_stays_eager(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH", "1")
    fake_logger = FakeLogger()
    monkeypatch.setattr(model_runner_module, "logger", fake_logger)
    monkeypatch.setattr(
        model_runner_module,
        "SamplingCudaGraphManager",
        lambda **kw: pytest.fail("must not be built"),
    )
    runner = _bare_runner(
        sampler=SimpleNamespace(compute_nans=True),
        rejection_sampler=None,
        speculator=None,
        num_speculative_steps=0,
        use_pp=True,
        device=CPU,
    )
    runner._capture_sampling_graph()
    assert runner.sampling_graph is None
    (warning,) = fake_logger.warnings
    for part in (
        "does not sample",
        "speculative decoding is off",
        "pipeline parallelism",
        "not CUDA",
        "COMPUTE_NANS",
    ):
        assert part in warning


def test_capture_builds_and_captures_the_manager_from_the_env(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH_MAX_REQS", "3")
    built: list[Any] = []

    class FakeManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.captured = False
            built.append(self)

        def capture(self):
            self.captured = True

    monkeypatch.setattr(model_runner_module, "SamplingCudaGraphManager", FakeManager)
    monkeypatch.setattr(
        model_runner_module.current_platform, "is_device_capability", lambda c: True
    )
    # A CUDA-typed device name without touching CUDA.
    device = SimpleNamespace(type="cuda")
    sampler = SimpleNamespace(compute_nans=False)
    runner = _bare_runner(
        sampler=sampler,
        rejection_sampler=object(),
        speculator=SimpleNamespace(draft_logits=None),
        num_speculative_steps=K,
        use_pp=False,
        device=device,
        req_states=object(),
        input_buffers=object(),
        model_state=object(),
        max_num_reqs=MAX_NUM_REQS,
        vocab_size=VOCAB,
    )
    runner._capture_sampling_graph()
    (manager,) = built
    assert runner.sampling_graph is manager and manager.captured
    kwargs = manager.kwargs
    assert kwargs["max_graph_reqs"] == 3
    assert kwargs["num_speculative_steps"] == K
    assert kwargs["vocab_size"] == VOCAB
    assert kwargs["post_update_fn"] == runner._post_update_sampled
    assert kwargs["postprocess_state_fn"] == runner._postprocess_model_state


def test_capture_skips_probabilistic_drafting(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH", "1")
    fake_logger = FakeLogger()
    monkeypatch.setattr(model_runner_module, "logger", fake_logger)
    monkeypatch.setattr(
        model_runner_module.current_platform, "is_device_capability", lambda c: True
    )
    runner = _bare_runner(
        sampler=SimpleNamespace(compute_nans=False),
        rejection_sampler=object(),
        speculator=SimpleNamespace(draft_logits=torch.zeros(1)),
        num_speculative_steps=K,
        use_pp=False,
        device=SimpleNamespace(type="cuda"),
    )
    runner._capture_sampling_graph()
    assert runner.sampling_graph is None
    assert "probabilistic drafting" in fake_logger.warnings[0]


def test_capture_is_limited_to_sm70(monkeypatch):
    """Off SM70 eager top-k/top-p is not the reference path the graph captures."""
    monkeypatch.setenv("VLLM_SM70_SAMPLING_CUDAGRAPH", "1")
    fake_logger = FakeLogger()
    monkeypatch.setattr(model_runner_module, "logger", fake_logger)
    monkeypatch.setattr(
        model_runner_module,
        "SamplingCudaGraphManager",
        lambda **kw: pytest.fail("must not be built off SM70"),
    )
    capabilities = []

    def is_device_capability(capability):
        capabilities.append(capability)
        return False

    monkeypatch.setattr(
        model_runner_module.current_platform,
        "is_device_capability",
        is_device_capability,
    )
    runner = _bare_runner(
        sampler=SimpleNamespace(compute_nans=False),
        rejection_sampler=object(),
        speculator=SimpleNamespace(draft_logits=None),
        num_speculative_steps=K,
        use_pp=False,
        device=SimpleNamespace(type="cuda"),
    )
    runner._capture_sampling_graph()
    assert runner.sampling_graph is None
    assert capabilities == [70]
    assert "not SM70" in fake_logger.warnings[0]


class _SampleHarness:
    """Runs GPUModelRunner.sample() on a bare runner for a spec-decode batch."""

    def __init__(self, monkeypatch, graph):
        self.eager_calls: list[str] = []
        self.sampler_output = SamplerOutput(
            sampled_token_ids=torch.full((1, PER_REQ), 5),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.full((1,), 2, dtype=torch.int32),
        )

        def rejection_sampler(logits, batch, draft_logits):
            self.eager_calls.append("rejection_sampler")
            return self.sampler_output

        monkeypatch.setattr(
            model_runner_module, "try_dflash2_sparse_target_rejection", lambda *a, **k: None
        )
        monkeypatch.setattr(
            model_runner_module,
            "get_num_sampled_and_rejected",
            lambda num_sampled, *a: (num_sampled, torch.zeros_like(num_sampled)),
        )
        self.runner = _bare_runner(
            model=SimpleNamespace(
                compute_logits=lambda h: torch.zeros(
                    PER_REQ, VOCAB, dtype=torch.float16
                )
            ),
            sampling_graph=graph,
            rejection_sampler=rejection_sampler,
            speculator=SimpleNamespace(draft_logits=None),
            sampler=None,
            req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=None)),
        )
        self.batch = make_batch(1)
        self.batch.seq_lens = torch.tensor([PER_REQ], dtype=torch.int32)
        self.hidden = torch.zeros(16, 4)

    def sample(self):
        return model_runner_module.GPUModelRunner.sample(
            self.runner, self.hidden, self.batch, None
        )


def test_sample_returns_the_graph_outputs_and_skips_the_eager_chain(monkeypatch):
    graph_out = cg.SamplingGraphOutput(
        sampler_output=SamplerOutput(
            sampled_token_ids=torch.full((1, PER_REQ), 7),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.full((1,), 3, dtype=torch.int32),
        ),
        num_sampled=torch.full((1,), 3, dtype=torch.int32),
        num_rejected=torch.full((1,), 2, dtype=torch.int32),
        post_update_done=True,
        model_state_done=True,
    )
    seen: list[tuple] = []

    class Graph:
        def run(self, logits, batch, grammar_output, draft_logits):
            seen.append((tuple(logits.shape), grammar_output, draft_logits))
            return graph_out

    harness = _SampleHarness(monkeypatch, Graph())
    sampler_output, num_sampled, num_rejected = harness.sample()
    assert sampler_output is graph_out.sampler_output
    assert num_sampled is graph_out.num_sampled
    assert num_rejected is graph_out.num_rejected
    assert harness.eager_calls == []
    assert harness.runner._sampling_graph_output is graph_out
    assert seen == [((PER_REQ, VOCAB), None, None)]


def test_sample_falls_back_to_the_eager_chain_when_the_graph_declines(monkeypatch):
    class Graph:
        def run(self, *args):
            return None

    harness = _SampleHarness(monkeypatch, Graph())
    sampler_output, num_sampled, num_rejected = harness.sample()
    assert sampler_output is harness.sampler_output
    assert harness.eager_calls == ["rejection_sampler"]
    assert harness.runner._sampling_graph_output is None
    assert num_rejected.tolist() == [0]


def test_sample_clears_a_stale_graph_marker(monkeypatch):
    harness = _SampleHarness(monkeypatch, None)
    harness.runner._sampling_graph_output = "stale"
    harness.sample()
    assert harness.runner._sampling_graph_output is None


def test_postprocess_sampled_is_post_update_then_model_state(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(
        model_runner_module, "post_update", lambda *a: order.append("post_update")
    )
    runner = _bare_runner(
        is_last_pp_rank=True,
        sampler=SimpleNamespace(
            penalties_state=SimpleNamespace(output_bin_counts=torch.zeros(1))
        ),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=torch.zeros(1)),
            last_sampled_tokens=torch.zeros(1),
            all_token_ids=SimpleNamespace(gpu=torch.zeros(1)),
            total_len=SimpleNamespace(gpu=torch.zeros(1)),
        ),
        model_state=SimpleNamespace(
            postprocess_state=lambda *a: order.append("model_state")
        ),
    )
    runner.postprocess_sampled(
        torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), None
    )
    assert order == ["post_update", "model_state"]


class _SampleTokensHarness:
    """Runs GPUModelRunner.sample_tokens() on a bare runner, no speculator."""

    def __init__(self, monkeypatch, graph_output):
        self.events: list[str] = []
        events = self.events

        class FakeAsyncOutput:
            def __init__(self, **kwargs):
                events.append("async_output")
                self.sampler_output = kwargs["sampler_output"]

            def get_output(self):
                return "output"

        monkeypatch.setattr(model_runner_module, "AsyncOutput", FakeAsyncOutput)
        batch = make_batch(1)
        batch.req_ids = ["r0"]
        batch.query_start_loc = torch.tensor([0, PER_REQ], dtype=torch.int32)
        sampler_output = SamplerOutput(
            sampled_token_ids=torch.zeros(1, PER_REQ, dtype=torch.int64),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones(1, dtype=torch.int32),
        )
        self.batch = batch

        def fake_sample(hidden_states, input_batch, grammar_output):
            events.append("sample")
            runner._sampling_graph_output = graph_output
            return (
                sampler_output,
                sampler_output.num_sampled,
                torch.zeros(1, dtype=torch.int32),
            )

        runner = _bare_runner(
            execute_model_state=SimpleNamespace(
                input_batch=batch,
                attn_metadata=None,
                slot_mappings_by_layer=None,
                hidden_states=torch.zeros(PER_REQ, 4),
                aux_hidden_states=None,
                finished_req_ids=set(),
            ),
            _sm70_v2_mtp_profile_pending=None,
            is_last_pp_rank=True,
            speculator=None,
            speculative_config=None,
            pp_handler=None,
            num_speculative_steps=0,
            use_async_scheduling=False,
            main_stream=object(),
            output_copy_stream=object(),
            model=SimpleNamespace(compute_logits=None),
            prompt_logprobs_worker=SimpleNamespace(
                compute_prompt_logprobs=lambda *a, **k: {}
            ),
            req_states=SimpleNamespace(
                all_token_ids=SimpleNamespace(gpu=None),
                num_computed_tokens=SimpleNamespace(gpu=None),
                prompt_len=SimpleNamespace(np=None, gpu=None),
                prefill_len=SimpleNamespace(np=None),
                num_computed_prefill_tokens=None,
            ),
            kv_connector=SimpleNamespace(post_forward=lambda ids: None),
            eplb=SimpleNamespace(step=lambda **kwargs: None),
            device=CPU,
        )
        runner.sample = fake_sample
        runner.postprocess_sampled = lambda *a: events.append("postprocess_sampled")
        runner._postprocess_model_state = lambda *a: events.append("model_state")
        self.runner = runner

    def run(self):
        return model_runner_module.GPUModelRunner.sample_tokens(self.runner, None)


def _graph_output(model_state_done: bool) -> cg.SamplingGraphOutput:
    return cg.SamplingGraphOutput(
        sampler_output=None,  # unused by sample_tokens
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        post_update_done=True,
        model_state_done=model_state_done,
    )


def test_sample_tokens_eager_path_runs_the_full_postprocess(monkeypatch):
    harness = _SampleTokensHarness(monkeypatch, None)
    assert harness.run() == "output"
    assert harness.events == ["sample", "async_output", "postprocess_sampled"]


def test_sample_tokens_skips_all_state_updates_the_graph_already_ran(monkeypatch):
    harness = _SampleTokensHarness(monkeypatch, _graph_output(model_state_done=True))
    assert harness.run() == "output"
    assert harness.events == ["sample", "async_output"]
    assert harness.runner._sampling_graph_output is None


def test_sample_tokens_runs_only_the_model_state_tail_when_not_captured(monkeypatch):
    harness = _SampleTokensHarness(monkeypatch, _graph_output(model_state_done=False))
    assert harness.run() == "output"
    # post_update ran in the graph; the model-state postprocess follows the
    # async output copy, as the eager path orders it.
    assert harness.events == ["sample", "async_output", "model_state"]
