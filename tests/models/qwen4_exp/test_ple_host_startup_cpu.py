# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU tests; run this file with the worktree's .venv/bin/python."""

import ast
import importlib.util
import json
import logging
import mmap
import multiprocessing
import os
import struct
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm/models/qwen4_exp/common/ple_host_startup.py"
spec = importlib.util.spec_from_file_location("ple_startup_cpu_test_module", HELPER)
startup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = startup
spec.loader.exec_module(startup)
PREFIX = "VLLM_QWEN4EXP_PLE_PIN_"


def _lock_worker(path, events, duration):
    with startup.host_pin_lock(path, 5):
        events.put(("enter", os.getpid(), time.monotonic()))
        time.sleep(duration)
        events.put(("exit", os.getpid(), time.monotonic()))


def _source_functions():
    path = ROOT / "vllm/models/qwen4_exp/nvidia/ple_layer.py"
    tree = ast.parse(path.read_text())
    alloc = next(
        n for n in tree.body if getattr(n, "name", "") == "_ple_pinned_host_empty"
    )
    cls = next(
        n for n in tree.body if getattr(n, "name", "") == "Qwen4ExpPinnedHostEmbedding"
    )
    load = next(n for n in cls.body if getattr(n, "name", "") == "load_shard")
    materialize = next(
        n for n in cls.body if getattr(n, "name", "") == "materialize_tables"
    )
    budget = next(
        n for n in cls.body if getattr(n, "name", "") == "_resolve_host_budget"
    )
    common = ast.parse((ROOT / "vllm/models/qwen4_exp/common/ple.py").read_text())
    nodes = [
        n
        for n in common.body
        if getattr(n, "name", "")
        in {
            "PLEShardOverlap",
            "compute_ple_shard_overlap",
            "copy_ple_embedding_shard_",
            "copy_ple_embedding_shard_split_",
            "PLEPlacement",
            "plan_ple_placement",
        }
    ]
    ns = {
        "torch": torch,
        "os": os,
        "dataclass": startup.dataclass,
        "PinStartupOptions": startup.PinStartupOptions,
        "pin_startup_guard": startup.pin_startup_guard,
        "nullcontext": nullcontext,
        "logger": logging.getLogger("ple-test"),
        "format_gib": str,
        "_PLE_REGISTERED_HOST_TABLES": [],
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes
        + [alloc, load, materialize, budget],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clean_env = patch.dict(os.environ, {}, clear=True)
        self.clean_env.start()

    def tearDown(self):
        self.clean_env.stop()
        self.temp.cleanup()

    def _checkpoint(self):
        names = [f"model.ple.ngram_embedding.shard_{i}.weight" for i in range(2)]
        header = {
            names[0]: {"data_offsets": [0, 8192]},
            "other.weight": {"data_offsets": [8192, 12288]},
            names[1]: {"data_offsets": [12288, 20480]},
        }
        raw = json.dumps(header).encode().ljust(4088, b" ")
        path = self.root / "weights.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(20480))
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {k: path.name for k in header},
                }
            )
        )
        return path

    def test_defaults_and_invalid_options(self):
        options = startup.PinStartupOptions.from_env()
        self.assertFalse(options.serialize or options.drop_cache)
        for key, value in [
            ("SERIALIZE", "yes"),
            ("TIMEOUT_S", "nan"),
            ("PAUSE_MS", "-1"),
        ]:
            with (
                self.subTest(key=key),
                patch.dict(os.environ, {PREFIX + key: value}),
                self.assertRaises(ValueError),
            ):
                startup.PinStartupOptions.from_env()

    def test_headers_only_and_exact_shard_ranges(self):
        path = self._checkpoint()
        ranges = startup.ple_checkpoint_ranges(str(self.root))
        self.assertEqual(ranges, {str(path): [(4096, 8192), (16384, 8192)]})
        with patch.object(os, "posix_fadvise") as advice:
            total = startup.drop_ple_checkpoint_cache(str(self.root))
        self.assertEqual(total, 16384)
        self.assertEqual(
            [(c.args[1], c.args[2]) for c in advice.call_args_list],
            [(4096, 8192), (16384, 8192)],
        )

    def test_advice_alignment_and_merge(self):
        path = self._checkpoint()
        with patch.object(os, "posix_fadvise") as advice:
            total = startup.advise_file_ranges(str(path), [(3, 9000), (4096, 8192)])
        self.assertEqual(total, 8192)
        self.assertEqual(advice.call_args.args[1:3], (4096, 8192))
        with self.assertRaises(ValueError):
            startup.advise_file_ranges(str(path), [(0, 999999)])

    def test_invalid_header_advice_fails_open_without_touching_data(self):
        path = self._checkpoint()
        path.write_bytes(struct.pack("<Q", 2**40))
        with patch.object(os, "posix_fadvise") as advice:
            self.assertEqual(startup.drop_ple_checkpoint_cache(str(self.root)), 0)
        advice.assert_not_called()

    def test_file_cache_advice_preserves_private_cow_bytes(self):
        path = self.root / "cow.bin"
        original = bytes(range(256)) * 128
        path.write_bytes(original)
        with path.open("rb") as source:
            mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            mapped[8192:8200] = b"modified"
            expected = mapped[:]
            self.assertEqual(
                startup.advise_file_ranges(str(path), [(0, len(mapped))]), len(mapped)
            )
            self.assertEqual(mapped[:], expected)
            self.assertEqual(path.read_bytes(), original)
        finally:
            mapped.close()

    def test_lock_timeout_exception_release_and_symlink(self):
        path = str(self.root / "pin.lock")
        with (
            startup.host_pin_lock(path, 1),
            self.assertRaises(TimeoutError),
            startup.host_pin_lock(path, 0.03),
        ):
            self.fail("lock entered concurrently")
        with (
            self.assertRaisesRegex(RuntimeError, "body failed"),
            startup.host_pin_lock(path, 1),
        ):
            raise RuntimeError("body failed")
        with startup.host_pin_lock(path, 1):
            pass
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(OSError), startup.host_pin_lock(str(link), 1):
            self.fail("symlink accepted")

    def test_multiprocess_register_windows_do_not_overlap(self):
        ctx = multiprocessing.get_context("spawn")
        events = ctx.Queue()
        path = str(self.root / "pin.lock")
        processes = [
            ctx.Process(target=_lock_worker, args=(path, events, 0.04))
            for _ in range(3)
        ]
        try:
            for process in processes:
                process.start()
            records = [events.get(timeout=30) for _ in range(6)]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            ordered = sorted(records, key=lambda r: r[2])
            self.assertEqual([r[0] for r in ordered], ["enter", "exit"] * 3)
            for i in range(0, 6, 2):
                self.assertEqual(ordered[i][1], ordered[i + 1][1])
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            events.close()

    def test_terminated_owner_releases_lock(self):
        ctx = multiprocessing.get_context("spawn")
        events = ctx.Queue()
        path = str(self.root / "pin.lock")
        process = ctx.Process(target=_lock_worker, args=(path, events, 60))
        process.start()
        try:
            self.assertEqual(events.get(timeout=30)[0], "enter")
        finally:
            process.terminate()
            process.join(timeout=10)
            events.close()
        with startup.host_pin_lock(path, 0.5):
            pass

    def test_cache_advice_precedes_single_registration_under_lock(self):
        ns = _source_functions()
        events = []
        path = str(self.root / "pin.lock")
        tensor = SimpleNamespace(
            data_ptr=lambda: 1234, nbytes=8192, is_pinned=lambda: True
        )

        def register(pointer, size, flags):
            self.assertEqual((pointer, size, flags), (1234, 8192, 3))
            with self.assertRaises(TimeoutError), startup.host_pin_lock(path, 0.01):
                self.fail("registration was outside the lock")
            events.append("register")
            return 0

        def empty(*args, **kwargs):
            events.append("allocate")
            return tensor

        ns["torch"] = SimpleNamespace(
            empty=empty,
            cuda=SimpleNamespace(
                cudart=lambda: SimpleNamespace(cudaHostRegister=register)
            ),
        )
        with (
            patch.dict(
                os.environ,
                {
                    PREFIX + "SERIALIZE": "1",
                    PREFIX + "DROP_CACHE": "1",
                    PREFIX + "LOCK_PATH": path,
                },
            ),
            patch.object(
                startup,
                "drop_ple_checkpoint_cache",
                side_effect=lambda _: events.append("advice") or 4096,
            ),
            self.assertLogs(startup.logger, level="INFO") as logs,
        ):
            result = ns["_ple_pinned_host_empty"](
                (64, 128), torch.uint8, model_path="test-checkpoint"
            )
        self.assertIs(result, tensor)
        self.assertEqual(events, ["advice", "allocate", "register"])
        self.assertEqual(ns["_PLE_REGISTERED_HOST_TABLES"], [tensor])
        self.assertIn(f"lock_path={path!r}", logs.output[0])

    def test_auto_host_budget_uses_retained_config(self):
        ns = _source_functions()
        ns["get_current_vllm_config"] = Mock(
            side_effect=AssertionError("no ambient model configuration")
        )
        ns["_ple_host_budget_bytes"] = lambda: None
        ns["available_host_bytes"] = lambda: None
        ns["total_host_bytes"] = lambda: None
        ns["_ple_vram_reserve_bytes"] = lambda _: 7
        ns["kv_cache_bytes_for_max_model_len"] = Mock(return_value=16)
        ns["auto_ple_host_budget_bytes"] = Mock(return_value=20)
        ns["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(mem_get_info=lambda _: (48, 64))
        )
        config = SimpleNamespace(
            cache_config=SimpleNamespace(gpu_memory_utilization=0.9),
            model_config=SimpleNamespace(max_model_len=128),
        )
        owner = SimpleNamespace(
            _meta_weight_shape=(8, 4), embedding_dim=4, _ple_vllm_config=config
        )
        for hybrid, expected in ((True, 32), (False, 20)):
            with self.subTest(hybrid=hybrid):
                ns["envs"] = SimpleNamespace(VLLM_SM70_QWEN38_HYBRID_PLE=hybrid)
                self.assertEqual(ns["_resolve_host_budget"](owner, "cpu"), expected)
        ns["kv_cache_bytes_for_max_model_len"].assert_called_once_with(config)
        ns["auto_ple_host_budget_bytes"].assert_called_once_with(
            table_bytes=32,
            device_total_bytes=64,
            device_allocated_bytes=16,
            gpu_memory_utilization=0.9,
            kv_cache_bytes=16,
            reserve_bytes=7,
        )
        ns["get_current_vllm_config"].assert_not_called()

    def test_materialization_does_not_require_ambient_model_config(self):
        ns = _source_functions()
        ns["available_host_bytes"] = lambda: None
        ns["get_current_vllm_config"] = Mock(
            side_effect=AssertionError("no ambient model configuration")
        )
        ns["torch"] = SimpleNamespace(
            device=lambda *args: torch.device("cpu"),
            accelerator=SimpleNamespace(current_device_index=lambda: 0),
            empty=torch.empty,
        )
        ns["_ple_pinned_host_empty"] = Mock(
            side_effect=lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype)
        )
        owner = SimpleNamespace(
            ple_device_table=None,
            _meta_weight_shape=(8, 4),
            embedding_dim=4,
            _meta_weight_dtype=torch.uint8,
            _resolve_host_budget=lambda _: 12,
            _pin_checkpoint_model_path=None,
        )
        ns["materialize_tables"](owner)
        self.assertEqual(owner.ple_device_table.shape, (5, 4))
        self.assertEqual(owner.ple_host_storage.shape, (3, 4))
        ns["_ple_pinned_host_empty"].assert_called_once_with(
            (3, 4), torch.uint8, model_path=None
        )
        pointer = owner.ple_host_storage.data_ptr()
        ns["materialize_tables"](owner)
        self.assertEqual(owner.ple_host_storage.data_ptr(), pointer)
        ns["get_current_vllm_config"].assert_not_called()

    def test_paced_registration_failure_does_not_allocate_larger_fallback(self):
        ns = _source_functions()
        tensor = SimpleNamespace(
            data_ptr=lambda: 1234, nbytes=8192, is_pinned=lambda: False
        )
        runtime = SimpleNamespace(
            cudaHostRegister=Mock(return_value=2),
            cudaHostUnregister=Mock(return_value=0),
        )
        fake = SimpleNamespace(
            empty=Mock(return_value=tensor),
            cuda=SimpleNamespace(cudart=lambda: runtime),
        )
        ns["torch"] = fake
        with (
            patch.dict(
                os.environ,
                {
                    PREFIX + "SERIALIZE": "1",
                    PREFIX + "LOCK_PATH": str(self.root / "pin.lock"),
                },
            ),
            self.assertRaisesRegex(RuntimeError, "refuses a larger"),
        ):
            ns["_ple_pinned_host_empty"]((64, 128), torch.uint8)
        self.assertEqual(fake.empty.call_count, 1)
        runtime.cudaHostUnregister.assert_not_called()

    def test_successful_register_but_failed_pin_check_is_unregistered(self):
        ns = _source_functions()
        tensor = SimpleNamespace(
            data_ptr=lambda: 1234, nbytes=8192, is_pinned=lambda: False
        )
        runtime = SimpleNamespace(
            cudaHostRegister=Mock(return_value=0),
            cudaHostUnregister=Mock(return_value=0),
        )
        ns["torch"] = SimpleNamespace(
            empty=Mock(return_value=tensor),
            cuda=SimpleNamespace(cudart=lambda: runtime),
        )
        ns["_ple_pinned_host_empty"]((64, 128), torch.uint8)
        runtime.cudaHostUnregister.assert_called_once_with(1234)
        self.assertEqual(ns["_PLE_REGISTERED_HOST_TABLES"], [])

    def test_copy_parity_without_post_copy_advice(self):
        ns = _source_functions()
        source = torch.arange(96, dtype=torch.uint8).reshape(24, 4)
        for enabled in ("0", "1"):
            for split in (0, 3, 12):
                for rank in (0, 1):
                    with (
                        self.subTest(enabled=enabled, split=split, rank=rank),
                        patch.dict(os.environ, {PREFIX + "DROP_CACHE": enabled}),
                        patch.object(os, "posix_fadvise") as advice,
                    ):
                        owner = SimpleNamespace(
                            materialize_tables=lambda: None,
                            ple_device_table=torch.empty((split, 4), dtype=torch.uint8),
                            ple_host_storage=torch.empty(
                                (12 - split, 4), dtype=torch.uint8
                            ),
                        )
                        for start in range(0, 24, 8):
                            ns["load_shard"](
                                owner,
                                source[start : start + 8],
                                checkpoint_start=start,
                                tp_start=rank * 12,
                                tp_end=(rank + 1) * 12,
                            )
                        actual = torch.cat(
                            (owner.ple_device_table, owner.ple_host_storage)
                        )
                        self.assertTrue(
                            torch.equal(actual, source[rank * 12 : (rank + 1) * 12])
                        )
                        self.assertTrue(owner._checkpoint_shard_loaded)
                        advice.assert_not_called()


if __name__ == "__main__":
    unittest.main()
