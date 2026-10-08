# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the rejected prototypes' extraction and resource screen.

Run directly with the serving venv Python; no torch/driver import or GPU use.
These checks do not establish GPU bitwise parity or synchronization correctness.
"""

import re
import unittest
from pathlib import Path

import generate
import resources

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
