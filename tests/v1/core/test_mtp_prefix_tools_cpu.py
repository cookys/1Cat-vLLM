# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline telemetry and HTTP probe checks; no server/GPU requests."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZE = load("tools/qwen4_exp/analyze_committed_prefix.py")
PROBE = load("benchmarks/benchmark_mtp_committed_prefix.py")


class ToolTests(unittest.TestCase):
    def test_compare_rejects_partial_record_and_logprob_drift(self):
        record = {
            "complete": True,
            "alignment": 1616,
            "seed": 1,
            "max_tokens": 1,
            "cases": {
                "case": {
                    "prompt_sha256": "fixed",
                    "token_ids": [12],
                    "logprobs": [-1.0],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            a.write_text(json.dumps(record))
            b.write_text(json.dumps(record))
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(PROBE.compare(a, b), 0)
                record["cases"]["case"]["logprobs"] = [-1.01]
                b.write_text(json.dumps(record))
                self.assertEqual(PROBE.compare(a, b), 1)
            record["complete"] = False
            b.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                PROBE.compare(a, b)

    def test_length_proxy_skips_trimmed_missing_and_too_short(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(
                json.dumps(
                    {
                        "transcript": [
                            {
                                "role": "assistant",
                                "prompt_tokens": p,
                                "completion_tokens": 5,
                                "trimmed": trimmed,
                            }
                            for p, trimmed in [
                                (9, False),
                                (33, False),
                                (43, False),
                                (48, True),
                            ]
                        ]
                    }
                )
            )
            result = ANALYZE.swe_proxy([path], 16)
        self.assertEqual(result["calls"], 4)
        self.assertEqual(result["previous_prompt_too_short"], 1)
        self.assertEqual(result["trimmed_or_shrinking"], 1)
        self.assertEqual(result["latest_boundary_in_previous_prefill_proxy"], 1)

    def test_retry_dedup_and_scheduled_normal_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            rows = []
            for request, hit in [("r", 16), ("r", 32), ("aborted", 32)]:
                rows.append(
                    f"MTP_COMMITTED_PREFIX lookup request={request} prompt_tokens=43 "
                    f"baseline_tokens=16 hit_tokens={hit} saved_tokens={hit - 16} "
                    "exclusive_blocks=2 evicted_blocks=1 free_blocks=590"
                )
            rows += [
                "MTP_COMMITTED_PREFIX scheduled request=r prefill_tokens=5",
                "MTP_COMMITTED_PREFIX scheduled request=r prefill_tokens=6",
                "MTP_COMMITTED_PREFIX scheduled request=aborted prefill_tokens=11",
                "MTP_COMMITTED_PREFIX finish request=r publish=True certificates=1",
                (
                    "MTP_COMMITTED_PREFIX finish request=aborted "
                    "publish=False certificates=0"
                ),
            ]
            path.write_text("\n".join(rows))
            result = ANALYZE.log_accounting([path])
        self.assertEqual(result["lookup_attempts"], 3)
        self.assertEqual(result["last_lookup_saved_tokens"], 32)
        self.assertEqual(result["scheduled_prefill_tokens"], 22)
        self.assertEqual(result["normal_finished_scheduled_prefill_tokens"], 11)
        self.assertEqual(result["published_certificates"], 1)

    def test_streamed_numeric_ids_and_first_token(self):
        events = [
            {"choices": [{"text": "", "token_ids": []}]},
            {
                "choices": [
                    {"token_ids": [4, 5], "logprobs": {"token_logprobs": [-1.0, -2.0]}}
                ]
            },
            {"choices": [{"token_ids": [], "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 43}},
        ]
        data = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        with patch.object(
            PROBE, "post", return_value=io.BytesIO(data + b"data: [DONE]\n")
        ):
            result = PROBE.completion("unused", {})
        self.assertEqual(result["token_ids"], [4, 5])
        self.assertEqual(result["logprobs"], [-1.0, -2.0])
        self.assertEqual(result["finish_reason"], "length")
        self.assertIsNotNone(result["ttft_s"])

    def test_guard_changes_exact_lookahead_not_prefix(self):
        cases = {name: tokens for name, _, tokens in PROBE.cases([10, 11, 12], 32, 7)}
        for multiplier in (2, 10):
            for family in ("a", "b"):
                key = f"m{multiplier}-{family}"
                p, miss = cases[key + "-producer"], cases[key + "-guard-miss"]
                boundary = multiplier * 32
                self.assertEqual(p[:boundary], miss[:boundary])
                self.assertNotEqual(p[boundary], miss[boundary])
                for append in (1024, 2048, 4096):
                    self.assertEqual(cases[key + f"-append{append}"][: len(p)], p)


if __name__ == "__main__":
    unittest.main()
