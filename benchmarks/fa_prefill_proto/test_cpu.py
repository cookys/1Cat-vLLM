# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the rejected prototypes' extraction and resource screen.

Run directly with the serving venv Python; no torch/driver import or GPU use.
These checks do not establish GPU bitwise parity or synchronization correctness.
"""

import re
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate
import probe
import resources
import study

ROOT = Path(__file__).resolve().parents[2]


class ExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "flash-attention-v100/kernel/"
                      "fused_mha_forward_paged.cu").read_text()

    def test_baseline_body_is_verbatim(self):
        original = self.source[self.source.index(generate.BEGIN):
                               self.source.index(generate.END)]
        restored = generate.core(self.source, 0).replace(
            "astra_fa_proto0_kernel", generate.KERNEL)
        self.assertEqual(restored, original)

    def test_source_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "kernel source changed"):
            generate.core(self.source + "\n", 1)

    def test_ambiguous_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected one source anchor"):
            generate.replace_once("twice twice", "twice", "replacement")

    def test_bad_variant_rejected(self):
        with self.assertRaisesRegex(ValueError, "variant must"):
            generate.core(self.source, 3)

    def test_each_variant_retains_cta_barriers(self):
        baseline = generate.core(self.source, 0)
        for variant in (1, 2):
            with self.subTest(variant=variant):
                result = generate.core(self.source, variant)
                self.assertEqual(result.count("__syncthreads();"),
                                 baseline.count("__syncthreads();"))
                self.assertIn("__launch_bounds__(D256_BM32_PHASE_THREADS, 2)",
                              result)

    def test_p1_final_layout_assertions_do_not_cascade(self):
        result = generate.core(self.source, 1)
        sizes = re.findall(r"static_assert\(sizeof\([^;]*?==\s*(\d+)", result)
        self.assertEqual(sizes, ["37472", "44000", "44000"])
        self.assertIn("volatile int total_n_blocks;", result)
        self.assertNotIn("const int total_n_blocks", result)
        self.assertNotIn("shared.shared.", result)

    def test_p2_keeps_reduction_k_order(self):
        qk = (generate.HERE / "qk16.cuh").read_text()
        self.assertIn("k_offset = 0; k_offset < D256_BM32_PHASE_D;", qk)
        self.assertIn("k_offset += WMMA_K", qk)
        self.assertEqual(qk.count("mma_sync(qk, a_fragment, b_fragment, qk)"), 1)

    def test_free_arms_only_change_launch_bound_and_symbols(self):
        for variant in (1, 2):
            with self.subTest(variant=variant):
                fixed = generate.generate(self.source, variant)
                free = generate.generate(self.source, variant, True)
                restored = free.replace(f"astra{variant}_free", f"astra{variant}")
                restored = restored.replace(f"proto{variant}_free", f"proto{variant}")
                restored = restored.replace(
                    "__launch_bounds__(D256_BM32_PHASE_THREADS)",
                    "__launch_bounds__(D256_BM32_PHASE_THREADS, 2)")
                self.assertEqual(fixed, restored)

    def test_no_free_baseline(self):
        with self.assertRaisesRegex(ValueError, "only P1/P2"):
            generate.generate(self.source, 0, True)


class ScratchLayoutTest(unittest.TestCase):
    def test_p1_every_float_has_one_owner_inside_score_strip(self):
        owners = {}
        for warp in range(8):
            words = set()
            for lane in range(32):
                for half in range(2):
                    for pair in range(4):
                        offset = ((half * 16 + pair * 4 + (lane >> 3)) * 144
                                  + warp * 16 + (lane & 7) * 2)
                        for part in (0, 1):
                            address = offset + part
                            self.assertNotIn(address, owners)
                            owners[address] = (warp, lane, half, pair, part)
                            words.add(address)
            expected = {row * 144 + col for row in range(32)
                        for col in range(warp * 16, (warp + 1) * 16)}
            self.assertEqual(words, expected)
        self.assertEqual(len(owners), 32 * 128)

    def test_p1_halfwarp_vector_transaction_bank_bijection(self):
        for warp in range(8):
            for pair in range(4):
                for lane_start in (0, 16):
                    banks = [((pair * 4 + (lane >> 3)) * 144
                              + warp * 16 + (lane & 7) * 2 + part) % 32
                             for lane in range(lane_start, lane_start + 16)
                             for part in (0, 1)]
                    self.assertEqual(sorted(banks), list(range(32)))

    def test_p2_quadrants_partition_existing_score_storage(self):
        seen = set()
        for warp in range(16):
            rows = range((warp >> 3) * 16, ((warp >> 3) + 1) * 16)
            cols = range((warp & 7) * 16, ((warp & 7) + 1) * 16)
            quadrant = {r * 128 + c for r in rows for c in cols}
            self.assertTrue(seen.isdisjoint(quadrant))
            seen.update(quadrant)
        self.assertEqual(seen, set(range(32 * 128)))


class ResourceGateTest(unittest.TestCase):
    def parse(self, reg=64, shared=41936, stack=0, local=0, stores=0,
              loads=0, sass="HMMA.884.F32.F32.STEP0;"):
        return resources.parse(
            f"REG:{reg} STACK:{stack} SHARED:{shared} LOCAL:{local}",
            f"{stores} bytes spill stores, {loads} bytes spill loads", sass)

    def test_baseline_passes_cpu_gate_only(self):
        result = self.parse()
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["resource_ctas_per_sm"], 2)

    def test_64_registers_does_not_hide_spill(self):
        for nbytes, shared in ((4, 43984), (4, 44000), (8, 41936)):
            with self.subTest(nbytes=nbytes, shared=shared):
                result = self.parse(shared=shared, stack=8, stores=nbytes,
                                    loads=nbytes,
                                    sass="HMMA.884; STL [R1], R0; LDL.LU R0, [R1];")
                self.assertFalse(result["gate_pass"])
                self.assertEqual(result["ldl_stl_static"], 2)

    def test_each_resource_veto_is_independent(self):
        for kwargs in ({"reg": 65}, {"shared": 49153}, {"local": 4},
                       {"stack": 8}, {"stores": 4}, {"loads": 4},
                       {"sass": "HMMA.884; LDL R0, [R1];"},
                       {"sass": "FFMA R0, R1, R2, R3;"}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.parse(**kwargs)["gate_pass"])

    def test_missing_evidence_fails_closed(self):
        resource = "REG:64 STACK:0 SHARED:41936 LOCAL:0"
        log = "0 bytes spill stores, 0 bytes spill loads"
        with self.assertRaisesRegex(ValueError, "missing PTXAS"):
            resources.parse(resource, "", "HMMA.884;")
        for field in ("REG:64", "STACK:0", "SHARED:41936", "LOCAL:0"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "missing resource"):
                    resources.parse(resource.replace(field, ""), log, "HMMA.884;")

    def test_free_p1_resource_limit_is_one_cta(self):
        result = self.parse(reg=68, shared=44000)
        self.assertEqual(result["resource_ctas_per_sm"], 1)
        self.assertFalse(result["gate_pass"])  # Legacy test, not GPU admission.


class StudyTest(unittest.TestCase):
    def blocks(self, pairs):
        return [[{"arm": arm, "event_ms": [value] * 8}
                 for arm, value in zip("ABBA", values)] for values in pairs]

    def test_abba_ratio_uses_paired_geometric_denominator(self):
        result = study.summarize(self.blocks([(100, 80, 320, 400)] * 6))
        self.assertAlmostEqual(result["paired_ratio_median"], 0.8)
        self.assertAlmostEqual(result["relative_p0_percent"], -20)
        self.assertEqual(result["aa_max_abs_drift_percent"], 300)
        self.assertAlmostEqual(result["bootstrap_upper95_one_sided"], 0.8)

    def test_abba_keeps_noise_instead_of_picking_fastest(self):
        result = study.summarize(self.blocks(
            [(100, 80, 80, 100), (100, 120, 120, 100)] * 3))
        self.assertEqual(result["noise_ratio_min_max"], [0.8, 1.2])
        self.assertEqual(result["paired_ratio_median"], 1.0)
        self.assertGreater(result["bootstrap_upper95_one_sided"], 1.0)

    def test_abba_rejects_bad_order_and_invalid_times(self):
        bad = self.blocks([(100, 90, 90, 100)] * 2)
        bad[0][0]["arm"] = "B"
        with self.assertRaisesRegex(ValueError, "not ABBA"):
            study.summarize(bad)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    study.summarize(self.blocks([(100, value, 90, 100)] * 2))
        with self.assertRaisesRegex(ValueError, "two ABBA"):
            study.summarize(self.blocks([(100, 90, 90, 100)]))

    def test_numeric_diff_signed_zero_and_nonfinite(self):
        import numpy as np
        ref = np.array([0, 1], dtype=np.float16)
        val = np.array([-0.0, 1], dtype=np.float16)
        result = study.numerical_metrics(ref, val, ref.tobytes(), val.tobytes())
        self.assertFalse(result["bitwise"])
        self.assertEqual(result["max_abs_diff"], 0)
        val[1] = 1.5
        result = study.numerical_metrics(ref, val, ref.tobytes(), val.tobytes())
        self.assertEqual(result["max_abs_diff"], 0.5)
        self.assertEqual(result["max_rel_diff"], 0.5)
        val[0] = float("nan")
        result = study.numerical_metrics(ref, val, ref.tobytes(), val.tobytes())
        self.assertFalse(result["finite"])
        self.assertIsNone(result["max_abs_diff"])

    def test_decision_keeps_numerical_difference_pending(self):
        row = {"paired_ratio_median": 0.88, "bootstrap_upper95_one_sided": 0.89}
        stats = {"96": row.copy(), "4032": row.copy()}
        self.assertEqual(study.classify(stats, True, True, True),
                         "E1_MICROBENCH_CANDIDATE")
        self.assertEqual(study.classify(stats, False, True, True), "E2_PENDING")
        self.assertEqual(study.classify(stats, True, False, True),
                         "INVALID_NONFINITE")
        self.assertIn("BASELINE_MISMATCH", study.classify(stats, True, True, False))
        stats["4032"]["bootstrap_upper95_one_sided"] = 0.91
        self.assertEqual(study.classify(stats, True, True, True), "INCONCLUSIVE")
        stats["96"]["paired_ratio_median"] = 1.01
        self.assertEqual(study.classify(stats, True, True, True), "SPEED_FAIL")

    def test_json_and_md_preserve_failed_progress(self):
        data = {"status": "FAILED", "error": "fake failure", "timings": {}}
        with tempfile.TemporaryDirectory(dir="/data/tmp") as tmp:
            study.save(data, tmp)
            self.assertEqual(json.loads((Path(tmp) / "results.json").read_text()),
                             data)
            self.assertIn("fake failure", (Path(tmp) / "results.md").read_text())

    def fake_manifest(self, directory):
        variants = {}
        for arm in study.ARMS:
            hashes = {}
            for suffix in ("cu", "so", "resources.txt", "build.log", "sass"):
                path = directory / f"proto{arm}.{suffix}"
                path.write_bytes(b"not a real CUDA artifact")
                hashes[suffix] = resources.digest(path)
            variants[arm] = {"files_sha256": hashes,
                             "library": str(directory / f"proto{arm}.so")}
        manifest = {"schema_version": 2, "variants": variants}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_manifest_tampering_rejected_before_library_load(self):
        with tempfile.TemporaryDirectory(dir="/data/tmp") as tmp:
            root = Path(tmp)
            expected = self.fake_manifest(root)
            self.assertEqual(probe.checked_manifest(root), expected)
            (root / "proto2.so").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                probe.checked_manifest(root)

    def test_missing_source_evidence_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory(dir="/data/tmp") as tmp:
            root = Path(tmp)
            manifest = self.fake_manifest(root)
            del manifest["variants"]["1"]["files_sha256"]["cu"]
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "artifact types"):
                probe.checked_manifest(root)

    def test_describe_imports_no_torch_and_loads_no_cuda_library(self):
        with tempfile.TemporaryDirectory(dir="/data/tmp") as tmp:
            self.fake_manifest(Path(tmp))
            code = (
                "import sys,ctypes; "
                "ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(Exception('CDLL')); "
                "import probe; "
                f"probe.describe(__import__('pathlib').Path({tmp!r})); "
                "assert 'torch' not in sys.modules; print('CPU_ONLY_OK')"
            )
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
                   "PYTHONPATH": str(generate.HERE)}
            process = subprocess.run([sys.executable, "-c", code], env=env,
                                     capture_output=True, text=True, check=True)
            self.assertIn("CPU_ONLY_OK", process.stdout)

    def test_gpu_requires_opt_in_before_import(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "",
                                          "FA_RUN_GPU": "0"}):
            with self.assertRaisesRegex(RuntimeError, "requires FA_RUN_GPU"):
                probe.run_gpu(None, {})

    def test_pointer_abi_arguments_match_wrapper(self):
        with mock.patch.object(probe.ctypes, "CDLL") as loader:
            loader.return_value.fa_launch.return_value = 0
            lib = probe.Library(Path("/unused"))
            tensors = [mock.Mock() for _ in range(5)]
            q, k, v, table, lengths = tensors
            q.shape = (1, 6, 96, 256)
            k.shape, table.shape = (257, 784, 1, 256), (1, 257)
            k.stride.return_value = v.stride.return_value = (200704, 256, 256, 1)
            for i, tensor in enumerate(tensors):
                tensor.data_ptr.return_value = 1000 + 100 * i
            output, lse, stream = mock.Mock(), mock.Mock(), mock.Mock()
            output.data_ptr.return_value, lse.data_ptr.return_value = 2000, 3000
            stream.cuda_stream = 42
            lib.launcher(tensors, output, lse, stream)()
            self.assertEqual(loader.return_value.fa_launch.call_args.args,
                             (1000, 1100, 1200, 2000, 3000, 1300, 1400,
                              1, 6, 96, 257, 1, 200704, 256, 256,
                              200704, 256, 256, 0.0625, 42))


if __name__ == "__main__":
    unittest.main(verbosity=2)
