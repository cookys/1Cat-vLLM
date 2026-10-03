# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only integration of real pool/managers/coordinator/scheduler methods.

Run directly (no vLLM imports, native extension or CUDA initialization):
  OMP_NUM_THREADS=1 .venv/bin/python tests/v1/core/test_mtp_prefix_snapshot_cpu.py

Only config/request containers and import dependencies are substituted. AST
extraction executes the production method bodies, including pool eviction and
Mamba speculative-block relocation. Device numerics still need serving tests.
"""

from __future__ import annotations

import ast
import collections
import collections.abc
import ctypes
import dataclasses
import hashlib
import importlib.util
import itertools
import math
import sys
import types
import typing
import unittest
from abc import ABC, abstractmethod
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_policy():
    spec = importlib.util.spec_from_file_location(
        "mtp_snapshot_cpu_policy", ROOT / "vllm/v1/core/mtp_prefix_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


POLICY = load_policy()


@dataclasses.dataclass(frozen=True)
class FullSpec:
    block_size: int
    prefix_cacheable: bool = True


@dataclasses.dataclass(frozen=True)
class MambaSpec:
    block_size: int
    prefix_cacheable: bool = True
    mamba_cache_mode: str = "align"
    num_speculative_blocks: int = 4
    mamba_type: str = "gdn"


@dataclasses.dataclass(frozen=True)
class CircularSpec:
    block_size: int
    prefix_cacheable: bool = False


class Request:
    def __init__(self, name, tokens):
        self.request_id = name
        self.prompt_token_ids = list(tokens)
        self.prompt_embeds = None
        self.mm_features = []
        self.lora_request = None
        self.resumable = False
        self.num_preemptions = 0
        self.num_prompt_tokens = len(tokens)
        self.num_output_tokens = 0
        self.num_output_placeholders = 0
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.num_tokens = len(tokens)
        self.shared_prefix_boundary = 0
        self.skip_reading_prefix_cache = False
        self.all_token_ids = self.prompt_token_ids
        self.block_hashes = [
            hashlib.sha256(bytes(tokens[:end])).digest()
            for end in range(8, len(tokens) + 1, 8)
        ]


def load_runtime():
    # Fakes represent inert configuration types, never allocator behavior.
    ns = dict(
        ABC=ABC,
        abstractmethod=abstractmethod,
        dataclass=dataclasses.dataclass,
        defaultdict=collections.defaultdict,
        Sequence=collections.abc.Sequence,
        NamedTuple=typing.NamedTuple,
        overload=typing.overload,
        itertools=itertools,
        groupby=itertools.groupby,
        lcm=math.lcm,
        cdiv=lambda a, b: (a + b - 1) // b,
        FullAttentionSpec=FullSpec,
        MambaSpec=MambaSpec,
        CircularBufferSpec=CircularSpec,
        MTPPrefixCertificate=POLICY.MTPPrefixCertificate,
        MTPPrefixSnapshots=POLICY.MTPPrefixSnapshots,
        MTPPrefixBlockPoolView=POLICY.MTPPrefixBlockPoolView,
        eligible_request=POLICY.eligible_request,
        BlockHash=lambda x: x,
        BlockHashWithGroupId=lambda x: x,
        make_block_hash_with_group_id=lambda h, g: h + g.to_bytes(4, "big"),
        logger=types.SimpleNamespace(info=lambda *a: None, debug=lambda *a: None),
    )
    for name in (
        "CrossAttentionManager",
        "SlidingWindowSpec",
        "ChunkedLocalAttentionSpec",
        "TQFullAttentionSpec",
        "MLAAttentionSpec",
        "HiddenStateCacheSpec",
        "SinkFullAttentionSpec",
        "PrefixAnchoredSWASpec",
        "SlidingWindowMLASpec",
        "KpoolTailSpec",
        "CrossAttentionSpec",
    ):
        ns[name] = type(name, (), {})

    class StripImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if (node.module or "").startswith("vllm") else node

    def extract(path, names, methods=()):
        tree = ast.parse((ROOT / path).read_text())
        body = [n for n in tree.body if getattr(n, "name", "") in names]
        for cls, method, alias in methods:
            node = next(n for n in tree.body if getattr(n, "name", "") == cls)
            fn = next(n for n in node.body if getattr(n, "name", "") == method)
            fn.name = alias
            body.append(fn)
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *body,
            ],
            type_ignores=[],
        )
        module = StripImports().visit(module)
        exec(compile(ast.fix_missing_locations(module), path, "exec"), ns)

    extract(
        "vllm/v1/core/kv_cache_utils.py",
        {
            "KVCacheBlock",
            "FreeKVCacheBlockQueue",
            "BlockHashListWithBlockSize",
        },
    )
    extract("vllm/v1/core/block_pool.py", {"BlockPool", "BlockHashToBlockMap"})
    extract(
        "vllm/v1/core/single_type_kv_cache_manager.py",
        {
            "SingleTypeKVCacheManager",
            "FullAttentionManager",
            "MambaManager",
            "CircularBufferManager",
            "get_manager_for_kv_cache_spec",
        },
    )
    ns["spec_manager_map"] = {
        FullSpec: ns["FullAttentionManager"],
        MambaSpec: ns["MambaManager"],
        CircularSpec: ns["CircularBufferManager"],
    }
    extract(
        "vllm/v1/core/kv_cache_coordinator.py",
        {
            "KVCacheCoordinator",
            "HybridKVCacheCoordinator",
            "SpecGroup",
        },
    )
    extract(
        "vllm/v1/core/sched/scheduler.py",
        set(),
        [
            ("Scheduler", "_mamba_block_aligned_split", "split_chunk"),
            ("Scheduler", "_update_after_schedule", "update_after_schedule"),
        ],
    )
    extract(
        "vllm/v1/core/kv_cache_manager.py",
        {"KVCacheBlocks"},
        [
            ("KVCacheManager", "free", "free_request"),
            ("KVCacheManager", "get_computed_blocks", "get_computed_blocks"),
            ("KVCacheManager", "allocate_slots", "allocate_slots"),
        ],
    )
    ns["torch"] = torch
    ns["is_conv_state_dim_first"] = lambda: False
    extract(
        "vllm/model_executor/layers/mamba/mamba_utils.py",
        {
            "MambaCopySpec",
            "get_conv_copy_spec",
            "get_temporal_copy_spec",
        },
    )
    ns["MambaAttentionBackendEnum"] = types.SimpleNamespace(
        GDN_ATTN="gdn", SHORT_CONV="ple"
    )
    extract(
        "vllm/v1/core/mtp_prefix_snapshot.py",
        {
            "configure_mtp_prefix_snapshots",
        },
    )
    ns["RequestStatus"] = types.SimpleNamespace(
        FINISHED_STOPPED="stop", FINISHED_LENGTH_CAPPED="length"
    )
    return ns


NS = load_runtime()


def coordinator(enabled=True, blocks=600, *, rings=False):
    groups = [
        types.SimpleNamespace(kv_cache_spec=s, is_eagle_group=False, layer_names=[])
        for s in (
            FullSpec(8),
            FullSpec(16),
            MambaSpec(16),
            MambaSpec(16, mamba_type="ple"),
        )
    ]
    groups[0].layer_names = ["mtp.layers.60.self_attn.attn"]
    groups[1].layer_names = ["mtp.layers.60.self_attn.indexer.compressed_key_cache"]
    if rings:
        groups.append(
            types.SimpleNamespace(
                kv_cache_spec=CircularSpec(8),
                is_eagle_group=False,
                layer_names=["mtp.layers.60.self_attn.indexer.raw_key_cache"],
            )
        )
    result = NS["HybridKVCacheCoordinator"](
        kv_cache_config=types.SimpleNamespace(
            num_blocks=blocks, kv_cache_groups=groups
        ),
        max_model_len=256,
        max_in_flight_tokens=64,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=8,
    )
    result.configure_prefix_cache_retention(0)
    if enabled:
        result.mtp_prefix_snapshots = POLICY.MTPPrefixSnapshots(
            16,
            cacheable_group_ids=tuple(
                i for i, g in enumerate(groups) if g.kv_cache_spec.prefix_cacheable
            ),
        )
    return result


def prefill(c, r, *, finish="stop", budget=64):
    # Certificate/eviction fixtures below explicitly represent an already-warm
    # producer. Admission integration tests use run_request instead; those call
    # the real public lookup and allocator and never inject a warm decision.
    if c.mtp_prefix_snapshots is not None:
        policy = c.mtp_prefix_snapshots
        boundary = (r.num_prompt_tokens - 1) // policy.alignment * policy.alignment
        policy.observe_lookup(r, max(policy.alignment, boundary - policy.alignment))
        c.mtp_prefix_snapshots.admit(r)
    scheduler = types.SimpleNamespace(
        use_eagle=True,
        mamba_state_block_size=16,
        cache_config=types.SimpleNamespace(block_size=8),
        mamba_state_retention_interval=0,
        mamba_dense_boundaries_on_contention=False,
        kv_cache_manager=types.SimpleNamespace(coordinator=c),
    )
    ends = []
    while r.num_computed_tokens < r.num_prompt_tokens:
        n = NS["split_chunk"](
            scheduler, r, min(budget, r.num_prompt_tokens - r.num_computed_tokens)
        )
        assert n > 0
        end = r.num_computed_tokens + n
        c.new_step_starts()
        for m in c.single_type_managers:
            m.remove_skipped_blocks(r.request_id, r.num_computed_tokens)
            m.allocate_new_blocks(r.request_id, end + 4, end)
        c.cache_blocks(r, end)
        r.num_computed_tokens = end
        ends.append(end)
    if finish is not None:
        r.status = finish
        NS["free_request"](types.SimpleNamespace(coordinator=c), r)
    return ends


def lookup(c, r):
    baseline = c.find_longest_cache_hit(r.block_hashes, r.num_tokens - 1)
    if c.mtp_prefix_snapshots is None:
        return baseline, baseline
    return baseline, c.find_committed_prefix_hit(r, baseline[1]) or baseline


def cache_manager(c):
    manager = types.SimpleNamespace(
        coordinator=c,
        block_pool=c.block_pool,
        enable_caching=True,
        log_stats=False,
        max_model_len=256,
        empty_kv_cache_blocks=NS["KVCacheBlocks"](
            tuple(() for _ in c.single_type_managers)
        ),
        create_kv_cache_blocks=NS["KVCacheBlocks"],
    )
    for method in ("get_computed_blocks", "allocate_slots", "free_request"):
        setattr(manager, method, types.MethodType(NS[method], manager))
    return manager


def splitter(c):
    return types.SimpleNamespace(
        use_eagle=True,
        mamba_state_block_size=16,
        cache_config=types.SimpleNamespace(block_size=8),
        mamba_state_retention_interval=0,
        mamba_dense_boundaries_on_contention=False,
        kv_cache_manager=cache_manager(c),
    )


def run_request(c, r, *, budget=64, finish="stop"):
    """Real lookup -> split -> allocate -> cache -> free, ordinary warmup."""
    scheduler = splitter(c)
    manager = scheduler.kv_cache_manager
    cached, hit = manager.get_computed_blocks(r)
    initial_hit = hit
    ends = []
    while r.num_computed_tokens < r.num_prompt_tokens:
        c.new_step_starts()
        n = NS["split_chunk"](
            scheduler,
            r,
            min(budget, r.num_prompt_tokens - r.num_computed_tokens - hit),
            hit,
        )
        assert n > 0
        allocated = manager.allocate_slots(r, n, hit, cached, num_lookahead_tokens=4)
        assert allocated is not None
        r.num_computed_tokens += hit + n
        ends.append(r.num_computed_tokens)
        cached, hit = None, 0
    if finish is not None:
        r.status = finish
        manager.free_request(r)
    return initial_hit, ends


class WarmAdmissionTests(unittest.TestCase):
    def test_cold_on_off_chunking_and_blocks_match(self):
        for size in (15, 16, 17, 31, 32, 33, 43, 64, 65, 127, 161):
            for budget in (8, 16, 33, 64):
                with self.subTest(size=size, budget=budget):
                    off, on = coordinator(False), coordinator()
                    tokens = list(range(size))
                    expected = run_request(off, Request("cold", tokens), budget=budget)
                    actual = run_request(on, Request("cold", tokens), budget=budget)
                    self.assertEqual(actual, expected)
                    policy = on.mtp_prefix_snapshots
                    self.assertEqual(policy.admissions, {})
                    self.assertEqual(policy.pending, {})
                    self.assertFalse(
                        any(b.mtp_prefix_certificate for b in on.block_pool.blocks)
                    )
                    self.assertEqual(policy.pool_stats(on.block_pool)[0], 0)

                    # Include physical block assignments / cache hashes, not only
                    # chunk lengths. Host policy must not allocate an extra page.
                    def layout(c):
                        return [
                            (b.block_id, b.block_hash, b.ref_cnt)
                            for b in c.block_pool.blocks
                        ]

                    self.assertEqual(layout(on), layout(off))

    def test_three_requests_bootstrap_cold_warm_certified(self):
        c = coordinator()
        tokens = list(range(43))
        self.assertEqual(run_request(c, Request("cold", tokens)), (0, [16, 43]))
        self.assertEqual(run_request(c, Request("warm", tokens)), (16, [32, 43]))
        self.assertEqual(run_request(c, Request("hit", tokens)), (32, [43]))
        self.assertEqual(c.mtp_prefix_snapshots.lookup_saved_tokens, 16)
        self.assertEqual(c.mtp_prefix_snapshots.admissions, {})

    def test_admission_logs_distinguish_cold_near_boundary_and_candidate(self):
        c = coordinator()
        c.mtp_prefix_snapshots.telemetry = True
        lines = []
        old_logger = NS["logger"]
        NS["logger"] = types.SimpleNamespace(
            info=lambda fmt, *args: lines.append(fmt % args), debug=lambda *args: None
        )
        try:
            for name in ("cold", "warm", "hit"):
                run_request(c, Request(name, list(range(43))))
        finally:
            NS["logger"] = old_logger
        admissions = [line for line in lines if "MTP_COMMITTED_PREFIX admit " in line]
        self.assertEqual(len(admissions), 3)  # Once per request, not per chunk.
        for line, reason in zip(admissions, ("cold", "near_boundary", "candidate")):
            self.assertIn(f"cut_reason={reason}", line)
        self.assertIn("warm_producer=False", admissions[0])
        self.assertIn("candidate_tokens=32", admissions[-1])

    def test_shared_system_prefix_keeps_off_chunking(self):
        c = coordinator()
        off = coordinator(False)
        for state in (c, off):
            run_request(state, Request("system-prefix-seed", list(range(43))))
        request = Request("new-conversation", list(range(101)))
        hit, ends = run_request(c, request, finish=None)
        control = run_request(
            off, Request("new-conversation", list(range(101))), finish=None
        )
        self.assertEqual((hit, ends), control)
        self.assertEqual(hit, 16)
        self.assertNotIn(96, ends)
        self.assertEqual(
            c.mtp_prefix_snapshots.cut_reason(request), "prefix_below_threshold"
        )
        self.assertFalse(c.mtp_prefix_snapshots.is_warm_producer(request))
        self.assertNotIn(request.request_id, c.mtp_prefix_snapshots.pending)
        self.assertFalse(any(b.mtp_prefix_certificate for b in c.block_pool.blocks))
        self.assertEqual(
            [(b.block_id, b.block_hash, b.ref_cnt) for b in c.block_pool.blocks],
            [(b.block_id, b.block_hash, b.ref_cnt) for b in off.block_pool.blocks],
        )
        request.status = "stop"
        cache_manager(c).free_request(request)
        # This also represents a real long append. V2b knowingly gives up the
        # B=96 certificate that v2 would let the next request use.
        self.assertEqual(lookup(c, Request("next", list(range(105))))[1][1], 80)

    def test_chain_warm_reasons_and_zero_threshold(self):
        policy = POLICY.MTPPrefixSnapshots(16)
        r = Request("r", list(range(101)))  # B=96, B-A=80.
        for baseline, candidate, reason, warm in (
            (0, 0, "cold", False),
            (16, 0, "prefix_below_threshold", False),
            (64, 0, "prefix_below_threshold", False),
            (80, 0, "near_boundary", True),
            (0, 32, "candidate", True),
            (16, 32, "candidate", True),
        ):
            with self.subTest(baseline=baseline, candidate=candidate):
                policy.observe_lookup(r, baseline, candidate)
                self.assertEqual(policy.cut_reason(r), reason)
                self.assertEqual(policy.is_warm_producer(r), warm)
        short = Request("short", list(range(17)))  # B-A=0 must not admit a miss.
        policy.observe_lookup(short, 0)
        self.assertEqual(policy.cut_reason(short), "cold")
        self.assertEqual(policy.extra_boundaries(short), ())
        policy.observe_lookup(r, 80)
        policy.admit(r)
        policy.observe_lookup(r, 0)
        self.assertEqual(policy.cut_reason(r), "near_boundary")

    def test_failed_admission_then_eviction_reclassifies_cold(self):
        for reserve_full in (False, True):
            with self.subTest(reserve_full=reserve_full):
                c = coordinator()
                run_request(c, Request("seed", list(range(43))))
                r = Request("waiting", list(range(43)))
                manager = cache_manager(c)
                cached, hit = manager.get_computed_blocks(r)
                self.assertEqual(hit, 16)
                # Keep cached pages visible but consume all free capacity. No
                # allocator mock: touching them changes the real refcounts.
                pressure = [b for b in c.block_pool.blocks if not b.is_null]
                c.block_pool.touch(pressure)
                c.new_step_starts()
                self.assertIsNone(
                    manager.allocate_slots(
                        r,
                        16,
                        hit,
                        cached,
                        num_lookahead_tokens=4,
                        full_sequence_must_fit=reserve_full,
                    )
                )
                self.assertFalse(
                    c.mtp_prefix_snapshots.admissions[r.request_id].scheduled
                )
                c.block_pool.free_blocks(pressure)
                evict = c.block_pool.get_new_blocks(c.block_pool.get_num_free_blocks())
                c.block_pool.free_blocks(evict)
                self.assertEqual(run_request(c, r), (0, [16, 43]))
                self.assertEqual(c.mtp_prefix_snapshots.pending, {})

    def test_retry_can_become_warm_but_scheduled_cold_cannot(self):
        c = coordinator()
        r = Request("waiting", list(range(43)))
        manager = cache_manager(c)
        self.assertEqual(manager.get_computed_blocks(r)[1], 0)
        run_request(c, Request("seed", list(range(43))))
        self.assertEqual(run_request(c, r), (16, [32, 43]))
        cold = Request("other", list(range(99, 142)))
        manager.get_computed_blocks(cold)
        c.new_step_starts()
        self.assertIsNotNone(manager.allocate_slots(cold, 16, num_lookahead_tokens=4))
        cold.num_computed_tokens = 16
        policy = c.mtp_prefix_snapshots
        # Later hits / computed progress cannot promote this request.
        policy.observe_lookup(cold, 32)
        self.assertFalse(policy.is_warm_producer(cold))
        self.assertEqual(policy.extra_boundaries(cold), ())
        cold.status = "abort"
        manager.free_request(cold)

    def test_finish_abort_preempt_clear_admission_and_id_reuse(self):
        for status in ("stop", "length", "abort", "error", "running", "ignored"):
            with self.subTest(status=status):
                c = coordinator()
                run_request(c, Request("seed", list(range(43))))
                r = Request("reused", list(range(43)))
                run_request(c, r, finish=status)
                policy = c.mtp_prefix_snapshots
                self.assertNotIn(r.request_id, policy.admissions)
                self.assertNotIn(r.request_id, policy.pending)
                other = Request(r.request_id, list(range(99, 142)))
                self.assertEqual(run_request(c, other), (0, [16, 43]))
        c = coordinator()
        run_request(c, Request("seed", list(range(43))))
        r = Request("preempted", list(range(43)))
        r.num_preemptions = 1
        off = coordinator(False)
        run_request(off, Request("seed", list(range(43))))
        control = Request("preempted", list(range(43)))
        control.num_preemptions = 1
        self.assertEqual(run_request(c, r), run_request(off, control))
        self.assertFalse(any(b.mtp_prefix_certificate for b in c.block_pool.blocks))

    def test_skip_read_is_cold_but_candidate_only_can_be_chain_warm(self):
        c = coordinator()
        for name in ("seed", "warm"):
            run_request(c, Request(name, list(range(43))))
        r = Request("skip", list(range(43)))
        r.skip_reading_prefix_cache = True
        self.assertEqual(run_request(c, r), (0, [16, 43]))
        # A certificate can remain when an older legacy state is evicted.
        # V2b can use the certificate as its chain-warm signal even when the
        # ordinary lookup hits zero; it may then create a later boundary.
        c = coordinator()
        for name in ("seed", "warm"):
            run_request(c, Request(name, list(range(43))))
        for block in c.block_pool.blocks:
            if (
                block.block_hash is not None
                and not block.mtp_prefix_only
                and block.mtp_prefix_certificate is None
                and block.block_hash[-4:] in (b"\x00\x00\x00\x02", b"\x00\x00\x00\x03")
            ):
                c.block_pool._maybe_evict_cached_block(block)
        r = Request("candidate-only", list(range(61)))
        manager = cache_manager(c)
        _, hit = manager.get_computed_blocks(r)
        self.assertEqual(hit, 32)
        self.assertEqual(
            c.mtp_prefix_snapshots.admissions[r.request_id].baseline_tokens, 0
        )
        self.assertTrue(c.mtp_prefix_snapshots.is_warm_producer(r))
        self.assertEqual(c.mtp_prefix_snapshots.cut_reason(r), "candidate")
        self.assertEqual(run_request(c, r), (32, [48, 61]))

    def test_cold_production_geometry_ignores_later_progress(self):
        for length in (1615, 1616, 3249, 16177, 65553):
            for budget in (8192, 16384):
                traces = []
                for enabled in (False, True):
                    c = coordinator(enabled)
                    scheduler = splitter(c)
                    scheduler.mamba_state_block_size = 1616
                    r = Request("cold", [1])
                    r.num_tokens = r.num_prompt_tokens = length
                    if enabled:
                        c.mtp_prefix_snapshots.alignment = 1616
                        c.mtp_prefix_snapshots.observe_lookup(r, 0)
                    ends = []
                    while r.num_computed_tokens < length:
                        n = NS["split_chunk"](
                            scheduler, r, min(budget, length - r.num_computed_tokens)
                        )
                        self.assertGreater(n, 0)
                        if enabled:
                            c.mtp_prefix_snapshots.admit(r)
                        r.num_computed_tokens += n
                        ends.append(r.num_computed_tokens)
                    traces.append(ends)
                with self.subTest(length=length, budget=budget):
                    self.assertEqual(*traces)
                if length == 16177 and budget == 8192:
                    self.assertEqual(traces[1], [8080, 14544, 16177])

    def test_cold_and_warm_decisions_stay_request_local(self):
        c = coordinator()
        run_request(c, Request("seed", list(range(43))))
        manager = cache_manager(c)
        cold = Request("cold", list(range(99, 142)))
        warm = Request("warm", list(range(43)))
        _, cold_hit = manager.get_computed_blocks(cold)
        cached, warm_hit = manager.get_computed_blocks(warm)
        self.assertEqual((cold_hit, warm_hit), (0, 16))
        c.new_step_starts()
        self.assertIsNotNone(
            manager.allocate_slots(warm, 16, warm_hit, cached, num_lookahead_tokens=4)
        )
        self.assertIsNotNone(manager.allocate_slots(cold, 16, num_lookahead_tokens=4))
        warm.num_computed_tokens, cold.num_computed_tokens = 32, 16
        policy = c.mtp_prefix_snapshots
        self.assertTrue(policy.is_warm_producer(warm))
        self.assertFalse(policy.is_warm_producer(cold))
        self.assertEqual(policy.extra_boundaries(cold), ())
        for r in (warm, cold):
            r.status = "abort"
            manager.free_request(r)
        self.assertEqual(policy.admissions, {})


class SnapshotTests(unittest.TestCase):
    def test_scheduled_telemetry_covers_prompt_only(self):
        c = coordinator()
        c.mtp_prefix_snapshots.telemetry = True
        r = Request("r", list(range(43)))
        r.use_structured_output = False
        r.num_computed_tokens = 32
        scheduler = types.SimpleNamespace(
            requests={"r": r},
            kv_cache_manager=types.SimpleNamespace(coordinator=c),
            enable_return_routed_experts=False,
            finished_req_ids={"old"},
        )
        output = types.SimpleNamespace(
            num_scheduled_tokens={"r": 14}, has_structured_output_requests=False
        )
        lines = []
        old_logger = NS["logger"]
        NS["logger"] = types.SimpleNamespace(
            info=lambda fmt, *args: lines.append(fmt % args)
        )
        try:
            NS["update_after_schedule"](scheduler, output)
            self.assertEqual(len(lines), 1)
            self.assertIn("start=32 end=43 prefill_tokens=11", lines[0])
            self.assertEqual(r.num_computed_tokens, 46)
            NS["update_after_schedule"](scheduler, output)
            self.assertEqual(len(lines), 1)  # Ordinary decode is not prefill.
        finally:
            NS["logger"] = old_logger

    def test_public_lookup_counters_junction_and_skip_read(self):
        c = coordinator()
        prefill(c, Request("producer", list(range(43))))
        c.mtp_prefix_snapshots.telemetry = True
        manager = types.SimpleNamespace(
            coordinator=c,
            block_pool=c.block_pool,
            enable_caching=True,
            log_stats=False,
            empty_kv_cache_blocks=(),
            create_kv_cache_blocks=lambda blocks: blocks,
        )
        r = Request("consumer", list(range(43)))
        _, length = NS["get_computed_blocks"](manager, r)
        self.assertEqual(length, 32)
        self.assertEqual(c.mtp_prefix_snapshots.lookup_hits, 1)
        self.assertEqual(c.mtp_prefix_snapshots.lookup_saved_tokens, 16)
        # Candidate lookups must not leak their junction into ordinary state
        # retention. This remains the baseline lookup's reported junction.
        self.assertEqual(r.shared_prefix_boundary, c.shared_prefix_boundary)
        r.skip_reading_prefix_cache = True
        self.assertEqual(NS["get_computed_blocks"](manager, r), ((), 0))
        self.assertEqual(c.mtp_prefix_snapshots.lookup_hits, 1)
        self.assertEqual(r.shared_prefix_boundary, 0)

    def test_supported_geometry_gate_and_fail_closed(self):
        def config():
            return types.SimpleNamespace(
                use_v2_model_runner=True,
                model_config=types.SimpleNamespace(
                    hf_text_config=types.SimpleNamespace(
                        model_type="qwen4_exp_text",
                        mtp_num_hidden_layers=1,
                        num_hidden_layers=60,
                        indexer_compress_ratio=4,
                    )
                ),
                speculative_config=types.SimpleNamespace(
                    method="mtp",
                    num_speculative_tokens=4,
                ),
                cache_config=types.SimpleNamespace(
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                ),
                scheduler_config=types.SimpleNamespace(async_scheduling=False),
                parallel_config=types.SimpleNamespace(
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=1,
                ),
            )

        gate = NS["configure_mtp_prefix_snapshots"]
        self.assertEqual(
            gate(config(), coordinator(rings=True), has_connector=False).alignment, 16
        )
        for field, value in (("method", "eagle"), ("num_speculative_tokens", 5)):
            cfg = config()
            setattr(cfg.speculative_config, field, value)
            with self.assertRaises(ValueError):
                gate(cfg, coordinator(rings=True), has_connector=False)
        with self.assertRaises(ValueError):
            gate(config(), coordinator(rings=True), has_connector=True)
        cfg = config()
        cfg.scheduler_config.async_scheduling = True
        with self.assertRaises(ValueError):
            gate(cfg, coordinator(rings=True), has_connector=False)
        c = coordinator(rings=True)
        c.single_type_managers = (
            *c.single_type_managers[:3],
            c.single_type_managers[4],
        )
        with self.assertRaises(ValueError):
            gate(config(), c, has_connector=False)
        for suffix in ("attn", "indexer.compressed_key_cache", "indexer.raw_key_cache"):
            c = coordinator(rings=True)
            for group in c.kv_cache_config.kv_cache_groups:
                group.layer_names = [
                    name
                    for name in group.layer_names
                    if name != "mtp.layers.60.self_attn." + suffix
                ]
            with self.subTest(missing=suffix), self.assertRaises(ValueError):
                gate(config(), c, has_connector=False)
        for ratio in (3, 0):
            cfg = config()
            cfg.model_config.hf_text_config.indexer_compress_ratio = ratio
            with self.assertRaises(ValueError):
                gate(cfg, coordinator(rings=True), has_connector=False)

    def test_finished_prefill_reuses_one_more_page(self):
        c = coordinator()
        r = Request("producer", list(range(43)))
        self.assertEqual(prefill(c, r), [16, 32, 43])
        base, hit = lookup(c, Request("consumer", list(range(43))))
        self.assertEqual(base[1], 16)
        self.assertEqual(hit[1], 32)
        self.assertTrue(all(g[-1].mtp_prefix_certificate.committed for g in hit[0]))

    def test_disabled_retention_and_lookup_unchanged(self):
        c = coordinator(False)
        self.assertEqual(prefill(c, Request("p", list(range(43)))), [16, 43])
        self.assertEqual(lookup(c, Request("r", list(range(43))))[1][1], 16)

    def test_pending_abort_preempt_and_error_do_not_publish(self):
        for status in (None, "abort", "error", "running", "ignored"):
            with self.subTest(status=status):
                c = coordinator()
                prefill(c, Request("p", list(range(43))), finish=status)
                base, hit = lookup(c, Request("r", list(range(43))))
                self.assertEqual(hit[1], base[1])
                self.assertEqual(hit[1], 16)

    def test_mtp_shift_guard_rejects_different_next_token(self):
        c = coordinator()
        prefill(c, Request("p", list(range(43))))
        tokens = list(range(43))
        tokens[32] = 99
        r = Request("other-sibling", tokens)
        base, hit = lookup(c, r)
        self.assertEqual(hit[1], base[1])
        self.assertLess(hit[1], 32)

    def test_every_group_must_keep_same_certificate_generation(self):
        for group in range(4):
            with self.subTest(group=group):
                c = coordinator()
                prefill(c, Request("p", list(range(43))))
                r = Request("r", list(range(43)))
                blocks, n = lookup(c, r)[1]
                self.assertEqual(n, 32)
                block = blocks[group][-1]
                original_hash = block.block_hash
                c.block_pool._maybe_evict_cached_block(block)
                self.assertIsNone(block.mtp_prefix_certificate)
                self.assertFalse(block.mtp_prefix_only)
                # Same address AND same hash are insufficient after reuse.
                block.block_hash = original_hash
                c.block_pool.cached_block_hash_to_block.insert(original_hash, block)
                self.assertFalse(c.mtp_prefix_snapshots.accepts(r, blocks, n))

    def test_mixed_producer_certificates_are_rejected(self):
        c = coordinator()
        prefill(c, Request("p", list(range(43))))
        r = Request("r", list(range(43)))
        blocks, n = lookup(c, r)[1]
        blocks[2][-1].mtp_prefix_certificate = POLICY.MTPPrefixCertificate(32, 32, True)
        self.assertFalse(c.mtp_prefix_snapshots.accepts(r, blocks, n))

    def test_qsa_ring_is_private_and_excluded_from_certificates(self):
        c = coordinator(rings=True)
        r = Request("p", list(range(43)))
        prefill(c, r)
        consumer = Request("r", list(range(43)))
        groups, length = lookup(c, consumer)[1]
        self.assertEqual(length, 32)
        self.assertEqual(groups[-1], [])
        c.allocate_new_computed_blocks(consumer.request_id, groups, length, 0)
        ring = c.single_type_managers[-1].req_to_blocks[consumer.request_id]
        self.assertEqual(len(ring), 1)
        self.assertIsNone(ring[0].block_hash)
        self.assertIsNone(ring[0].mtp_prefix_certificate)
        c.free(consumer.request_id)
        self.assertEqual(
            c.block_pool.get_num_free_blocks(), c.block_pool.num_gpu_blocks - 1
        )

    def test_ordinary_pool_cannot_see_experimental_states(self):
        c = coordinator()
        prefill(c, Request("p", list(range(43))))
        r = Request("r", list(range(43)))
        blocks, _ = lookup(c, r)[1]
        block = blocks[2][-1]
        self.assertTrue(block.mtp_prefix_only)
        key = block.block_hash
        mapping = c.block_pool.cached_block_hash_to_block
        self.assertIsNone(mapping.get_one_block(key))
        self.assertIs(mapping.get_one_block(key, allow_mtp_prefix=True), block)
        # A later ordinary duplicate remains visible through the same hash.
        duplicate = c.block_pool.get_new_blocks(1)[0]
        duplicate.block_hash = key
        mapping.insert(key, duplicate)
        self.assertIs(mapping.get_one_block(key), duplicate)

    def test_candidate_view_prefers_committed_duplicate(self):
        c = coordinator()
        prefill(c, Request("p", list(range(43))))
        blocks, _ = lookup(c, Request("r", list(range(43))))[1]
        certified = blocks[2][-1]
        key = certified.block_hash
        mapping = c.block_pool.cached_block_hash_to_block
        mapping.pop(key, certified.block_id)
        pending = c.block_pool.get_new_blocks(1)[0]
        pending.block_hash = key
        mapping.insert(key, pending)
        mapping.insert(key, certified)
        self.assertIs(mapping.get_one_block(key), pending)
        self.assertIs(mapping.get_one_block(key, allow_mtp_prefix=True), certified)

    def test_only_true_prefill_and_known_lookahead_can_be_recorded(self):
        for changes in (
            {"num_output_tokens": 1},
            {"num_output_placeholders": 1},
            {"num_preemptions": 1},
            {"resumable": True},
            {"prompt_embeds": object()},
            {"mm_features": [object()]},
            {"lora_request": object()},
            {"num_prompt_tokens": 32},
            {"num_computed_tokens": 32},
        ):
            with self.subTest(changes=changes):
                c = coordinator()
                r = Request("p", list(range(43)))
                c.mtp_prefix_snapshots.observe_lookup(r, 16)
                c.mtp_prefix_snapshots.admit(r)
                for k, v in changes.items():
                    setattr(r, k, v)
                self.assertFalse(
                    c.mtp_prefix_snapshots.record_prefill(r, 32, c.single_type_managers)
                )

    def test_all_boundary_remainders_and_extension(self):
        for size in range(33, 65):
            for append in (0, 1, 16, 33):
                with self.subTest(size=size, append=append):
                    c = coordinator()
                    prefill(c, Request("p", list(range(size))))
                    _, hit = lookup(c, Request("r", list(range(size + append))))
                    self.assertGreaterEqual(hit[1], (size - 1) // 16 * 16)
                    self.assertLessEqual(hit[1], size)

    def test_decode_scratch_relocation_does_not_mutate_snapshot(self):
        c = coordinator()
        r = Request("p", list(range(43)))
        prefill(c, r, finish=None)
        # Existing align allocation/relocation runs, not a duplicate allocator.
        for m in c.single_type_managers[2:]:
            kept = c.block_pool.get_cached_block(
                NS["BlockHashListWithBlockSize"](r.block_hashes, 8, 16)[1],
                [m.kv_cache_group_id],
                allow_mtp_prefix=True,
            )[0]
            original = kept.mtp_prefix_certificate
            for end in (46, 50, 55, 63, 68):
                m.remove_skipped_blocks(r.request_id, end - 5)
                m.allocate_new_blocks(r.request_id, end + 4, end)
                state_col = (end + 15) // 16 - 1
                self.assertIsNot(m.req_to_blocks[r.request_id][state_col], kept)
                self.assertIs(kept.mtp_prefix_certificate, original)

    def test_certified_hit_uses_normal_touch_and_release(self):
        c = coordinator()
        prefill(c, Request("p", list(range(43))))
        r = Request("r", list(range(43)))
        groups, length = lookup(c, r)[1]
        for blocks in groups:
            self.assertEqual(blocks[-1].ref_cnt, 0)
        c.allocate_new_computed_blocks(r.request_id, groups, length, 0)
        for blocks in groups:
            self.assertEqual(blocks[-1].ref_cnt, 1)
        c.free(r.request_id)
        for blocks in groups:
            self.assertEqual(blocks[-1].ref_cnt, 0)
        self.assertEqual(
            c.block_pool.get_num_free_blocks(), c.block_pool.num_gpu_blocks - 1
        )

    def test_experimental_blocks_are_evictable_with_no_capacity_leak(self):
        c = coordinator()
        prefill(c, Request("p", list(range(43))))
        exclusive, evicted, free = c.mtp_prefix_snapshots.pool_stats(c.block_pool)
        self.assertEqual(exclusive, 2)  # One GDN and one PLE metadata group.
        self.assertEqual(evicted, 0)
        self.assertEqual(free, c.block_pool.num_gpu_blocks - 1)
        blocks = c.block_pool.get_new_blocks(free)
        self.assertEqual(c.mtp_prefix_snapshots.pool_stats(c.block_pool), (0, 2, 0))
        c.block_pool.free_blocks(blocks)
        self.assertEqual(lookup(c, Request("r", list(range(43))))[1][1], 0)

    def test_older_certificate_survives_newer_uncertified_candidate(self):
        c = coordinator()
        prefill(c, Request("p", list(range(61))))
        r = Request("r", list(range(61)))
        newest = lookup(c, r)[1]
        self.assertEqual(newest[1], 48)
        newest[0][0][-1].mtp_prefix_certificate.committed = False
        older = c.find_committed_prefix_hit(r, 0)
        self.assertIsNotNone(older)
        self.assertEqual(older[1], 32)

    def test_real_copy_specs_preserve_conv_and_temporal_bytes(self):
        # Execute upstream Python pointer/size selectors on CPU and perform
        # their byte copy. This validates the transport contract, not Triton
        # code generation or equality to a separately recomputed FP state.
        for kind, shape, dtype in (
            ("conv", (8, 7, 16), torch.float16),
            ("temporal", (8, 4, 4), torch.float32),
        ):
            for accepted in range(1, 6):
                with self.subTest(kind=kind, accepted=accepted):
                    states = torch.empty(shape, dtype=dtype)
                    raw = states.view(torch.uint8).reshape(-1)
                    raw.copy_((torch.arange(raw.numel()) % 256).to(torch.uint8))
                    before = raw.clone()
                    ids = [2, 3, 4, 5, 6]
                    copy = NS[f"get_{kind}_copy_spec"](states, ids, 0, accepted)
                    destination = states[0].view(torch.uint8).reshape(-1)
                    count = copy.num_elements * states.element_size()
                    expected = ctypes.string_at(copy.start_addr, count)
                    ctypes.memmove(states[0].data_ptr(), copy.start_addr, count)
                    self.assertEqual(bytes(destination[:count].tolist()), expected)
                    # Destination cannot corrupt any of the committed inputs.
                    start = states[0].numel() * states.element_size()
                    self.assertTrue(torch.equal(raw[start:], before[start:]))


if __name__ == "__main__":
    unittest.main()
