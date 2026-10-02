# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for PLE metadata staging, without vLLM/native imports.

Run directly with a CPU PyTorch installation:
  OMP_NUM_THREADS=1 .venv/bin/python tests/v1/attention/test_ple_metadata_async_cpu.py

The actual builder and block-table selection run on CPU. Only configuration,
metadata containers and the pinned H2D transport are substituted. GPU timing,
stream behavior and CUDA graph replay still require a serving smoke test.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm/v1/attention/backends/short_conv_attn.py"


def _load_functions():
    namespace = {
        "torch": torch,
        "PleShortConvAttentionMetadata": SimpleNamespace,
        "MambaSpec": SimpleNamespace,
        "NULL_BLOCK_ID": 0,
        "async_tensor_h2d": Mock(),
    }
    tree = ast.parse(SOURCE.read_text())
    helper = next(
        n for n in tree.body if getattr(n, "name", "") == "_async_ple_request_indices"
    )
    builder = next(
        n
        for n in tree.body
        if getattr(n, "name", "") == "PleShortConvAttentionMetadataBuilder"
    )
    build = next(n for n in builder.body if getattr(n, "name", "") == "build")
    utils = ROOT / "vllm/v1/attention/backends/utils.py"
    block_selector = next(
        n
        for n in ast.parse(utils.read_text()).body
        if getattr(n, "name", "") == "mamba_get_block_table_tensor"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            helper,
            block_selector,
            build,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


def _builder(full_graph, mode):
    result = SimpleNamespace(
        use_spec_decode=True,
        use_full_cuda_graph=full_graph,
        num_spec=4,
        decode_cudagraph_max_bs=8,
        decode_cudagraph_max_tokens=40,
        kv_cache_spec=SimpleNamespace(block_size=8, num_speculative_blocks=4),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode=mode)
        ),
        _build_non_spec_metadata=Mock(return_value="non-spec-fallback"),
    )
    for name, dtype, size in (
        ("spec_state_indices_tensor", torch.int32, 8),
        ("spec_sequence_masks", torch.bool, 8),
        ("spec_query_start_loc", torch.int32, 9),
        ("num_accepted_tokens", torch.int32, 8),
    ):
        setattr(result, name, torch.full((size,), 99, dtype=dtype))
    return result


def _batch(lengths):
    query = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    seq_lens = torch.tensor([16 + n if n else 0 for n in lengths], dtype=torch.int32)
    block_table = torch.arange(len(lengths) * 12, dtype=torch.int32).reshape(-1, 12)
    return SimpleNamespace(
        query_start_loc=query.clone(),
        query_start_loc_cpu=query,
        num_reqs=len(lengths),
        num_actual_tokens=sum(lengths),
        seq_lens=seq_lens,
        block_table_tensor=block_table,
        compute_num_computed_tokens=lambda: seq_lens - torch.tensor(lengths),
    )


class PleMetadataTests(unittest.TestCase):
    def setUp(self):
        self.ns = _load_functions()

    def test_cpu_indices_keep_identity(self):
        indices = torch.tensor([3, 1, 7], dtype=torch.int64)
        result = self.ns["_async_ple_request_indices"](indices, torch.device("cpu"))
        self.assertIs(result, indices)
        self.ns["async_tensor_h2d"].assert_not_called()

    def test_accelerator_indices_use_async_staging(self):
        # CUDA devices are descriptors only; no CUDA operation is performed.
        for values in ([], [0], [3, 1, 7]):
            with self.subTest(values=values):
                copy = self.ns["async_tensor_h2d"]
                copy.reset_mock()
                indices = torch.tensor(values, dtype=torch.int64)
                result = self.ns["_async_ple_request_indices"](
                    indices, torch.device("cuda:2")
                )
                copy.assert_called_once_with(
                    values, dtype=torch.int64, device=torch.device("cuda:2")
                )
                self.assertIs(result, copy.return_value)

    def _check_batch(self, lengths, drafts, full_graph, mode):
        builder = _builder(full_graph, mode)
        batch = _batch(lengths)
        accepted = torch.tensor(
            [i % 5 + 1 for i in range(len(lengths))], dtype=torch.int32
        )
        indices = self.ns["_async_ple_request_indices"]
        calls = []

        def record(tensor, device):
            calls.append((tensor.tolist(), device))
            return indices(tensor, device)

        self.ns["_async_ple_request_indices"] = record
        try:
            result = self.ns["build"](
                builder,
                0,
                batch,
                num_accepted_tokens=accepted,
                num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
            )
        finally:
            self.ns["_async_ple_request_indices"] = indices

        spec = [i for i, n in enumerate(drafts) if n >= 0]
        decode = [i for i, n in enumerate(lengths) if drafts[i] < 0 and n == 1]
        prefill = [i for i, n in enumerate(lengths) if drafts[i] < 0 and n > 1]
        non_spec = decode + prefill
        self.assertEqual(result.num_spec_decodes, len(spec))
        self.assertEqual(result.num_decodes, len(decode))
        self.assertEqual(result.num_prefills, len(prefill))
        self.assertEqual(result.num_spec_decode_tokens, sum(lengths[i] for i in spec))
        self.assertEqual(result.num_prefill_tokens, sum(lengths[i] for i in prefill))
        self.assertEqual(
            result.num_accepted_tokens[: len(spec)].tolist(), accepted[spec].tolist()
        )
        expected_slots = []
        for i in spec:
            column = max((int(batch.seq_lens[i]) - 1) // 8, 0) if mode == "align" else 0
            expected_slots.append(int(batch.block_table_tensor[i, column]))
        self.assertEqual(
            result.spec_state_indices_tensor[: len(spec)].tolist(), expected_slots
        )
        offsets = batch.query_start_loc_cpu.tolist()
        spec_tokens = [j for i in spec for j in range(offsets[i], offsets[i + 1])]
        non_spec_tokens = [
            j for i in non_spec for j in range(offsets[i], offsets[i + 1])
        ]
        self.assertEqual(result.spec_token_indx.tolist(), spec_tokens)
        self.assertEqual(result.non_spec_token_indx.tolist(), non_spec_tokens)
        expected_starts = [0]
        for i in spec:
            expected_starts.append(expected_starts[-1] + lengths[i])
        self.assertEqual(
            result.spec_query_start_loc[: len(spec) + 1].tolist(), expected_starts
        )
        # One device-index request per group; accepted counts and computed
        # lengths reuse these tensors instead of initiating another transfer.
        self.assertEqual([x[0] for x in calls[:2]], [spec, non_spec])
        self.assertEqual(len(calls), 3 if non_spec else 2)
        self.ns["async_tensor_h2d"].assert_not_called()
        if full_graph and not non_spec:
            self.assertEqual(
                result.spec_state_indices_tensor[len(spec) :].tolist(),
                [0] * (len(lengths) - len(spec)),
            )
            self.assertEqual(
                result.num_accepted_tokens[len(spec) :].tolist(),
                [1] * (len(lengths) - len(spec)),
            )
            self.assertEqual(
                result.spec_query_start_loc[len(spec) + 1 :].tolist(),
                [sum(lengths)] * (len(lengths) - len(spec)),
            )
            self.assertEqual(
                result.num_accepted_tokens.data_ptr(),
                builder.num_accepted_tokens.data_ptr(),
            )

    def test_pure_mixed_and_padded_requests(self):
        cases = [
            ([5], [4]),
            ([5, 0, 0, 0], [4, -1, -1, -1]),
            ([5, 5, 0, 0], [4, 4, -1, -1]),
            ([1, 5, 3, 0], [-1, 4, -1, -1]),
            ([5, 1, 5, 7], [4, -1, 4, -1]),
            ([1, 5, 1, 5], [-1, 4, -1, 4]),
        ]
        for lengths, drafts in cases:
            for full_graph in (False, True):
                for mode in ("none", "all", "align"):
                    with self.subTest(
                        lengths=lengths, full_graph=full_graph, mode=mode
                    ):
                        self._check_batch(lengths, drafts, full_graph, mode)

    def test_no_speculation_keeps_existing_fallback(self):
        builder = _builder(True, "align")
        for drafts in (None, torch.tensor([-1, -1], dtype=torch.int32)):
            result = self.ns["build"](
                builder, 0, _batch([1, 3]), num_decode_draft_tokens_cpu=drafts
            )
            self.assertEqual(result, "non-spec-fallback")


if __name__ == "__main__":
    unittest.main()
