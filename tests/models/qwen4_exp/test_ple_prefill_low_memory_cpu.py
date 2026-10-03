# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numeric/storage contracts for the actual PLE batched prefill method.

Run directly with CPU PyTorch; no vLLM/native imports or CUDA initialization.
Only the layer shell, metadata and env registry are substituted. A separate
real-import smoke test covers the env registry. These tests do not prove CUDA
allocation peaks or cuDNN parity.
"""

import ast
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm/models/qwen4_exp/nvidia/ple_layer.py"
FLAG = "VLLM_QWEN4EXP_PLE_PREFILL_LOW_MEMORY"


def load_layer():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if getattr(n, "name", "") == "Qwen4ExpPLELayer")
    names = {"_short_conv_dilated_prefill_batched", "_gather_prefill_conv_tail"}
    methods = [n for n in cls.body if getattr(n, "name", "") in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    environment = SimpleNamespace(**{FLAG: False})
    namespace = {
        "torch": torch,
        "F": F,
        "envs": environment,
        "NULL_BLOCK_ID": 0,
        "logger": Mock(),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    layer_type = type("CpuPleLayer", (), {name: namespace[name] for name in names})
    return layer_type, environment


LAYER_TYPE, ENVS = load_layer()


class StorageTrace(TorchDispatchMode):
    """Track live tensor storage, including aliases, without retaining tensors.

    This is logical ATen tensor liveness on CPU, not CUDA allocator reserved
    memory or a convolution-library workspace measurement.
    """

    def __init__(self, inputs):
        super().__init__()
        self.excluded = {
            t.untyped_storage()._cdata
            for t in tree_leaves(inputs)
            if isinstance(t, torch.Tensor)
        }
        self.references = []
        self.peak_bytes = 0
        self.convolution_live_bytes = []

    def live_bytes(self):
        storages = {}
        for ref in self.references:
            tensor = ref()
            if tensor is not None:
                storage = tensor.untyped_storage()
                if storage._cdata not in self.excluded:
                    storages[storage._cdata] = storage.nbytes()
        return sum(storages.values())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("prefill must not read a tensor scalar on the host")
        if func in (torch.ops.aten.conv1d.default, torch.ops.aten.convolution.default):
            self.convolution_live_bytes.append(self.live_bytes())
        result = func(*args, **(kwargs or {}))
        self.references.extend(
            weakref.ref(t) for t in tree_leaves(result) if isinstance(t, torch.Tensor)
        )
        live_storages = {}
        surviving = []
        for ref in self.references:
            tensor = ref()
            if tensor is None:
                continue
            surviving.append(ref)
            storage = tensor.untyped_storage()
            if storage._cdata not in self.excluded:
                live_storages[storage._cdata] = storage.nbytes()
        self.references = surviving
        self.peak_bytes = max(self.peak_bytes, sum(live_storages.values()))
        return result


def make_case(lengths, dtype=torch.float16, channels=16, kernel=4, dilation=3):
    generator = torch.Generator().manual_seed(69)
    layer = LAYER_TYPE()
    layer.conv_state_len = (kernel - 1) * dilation
    layer.short_conv_dilation = dilation
    tokens = sum(lengths)
    inputs = torch.randn(tokens, channels, generator=generator).to(dtype)
    weights = torch.randn(channels, kernel, generator=generator).to(dtype) / 4
    state = torch.randn(
        len(lengths) + 3,
        channels,
        layer.conv_state_len + 4,
        generator=generator,
    ).to(dtype)
    # Nonzero decode prefix verifies subtraction of num_decode_tokens and
    # slicing only the final num_prefills entries of query_start_loc.
    starts = [0, 1, 2]
    for length in lengths:
        starts.append(starts[-1] + length)
    metadata = SimpleNamespace(
        non_spec_query_start_loc=torch.tensor(starts, dtype=torch.int32),
        has_initial_states_p=torch.tensor(
            [i % 2 == 0 for i in range(len(lengths))], dtype=torch.bool
        ),
        max_prefill_query_len=max(lengths, default=0),
    )
    indices = torch.arange(len(lengths), 0, -1, dtype=torch.int32)
    return layer, inputs, metadata, state, weights, indices


def run_case(case, enabled):
    layer, inputs, metadata, state, weights, indices = case
    state = state.clone()
    setattr(ENVS, FLAG, enabled)
    tracker = StorageTrace((inputs, state, weights, indices, vars(metadata)))
    with torch.inference_mode(), tracker:
        output = layer._short_conv_dilated_prefill_batched(
            inputs,
            metadata,
            state,
            weights,
            indices,
            len(indices),
            2,
            len(inputs),
        )
    return output, state, tracker


class PrefillTests(unittest.TestCase):
    def assert_bytes_equal(self, left, right):
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.dtype, right.dtype)
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    def check_case(self, case):
        before_input = case[1].clone()
        before_weight = case[4].clone()
        baseline, old_state, old_trace = run_case(case, False)
        candidate, new_state, new_trace = run_case(case, True)
        self.assert_bytes_equal(baseline, candidate)
        self.assert_bytes_equal(old_state, new_state)
        self.assert_bytes_equal(case[1], before_input)
        self.assert_bytes_equal(case[4], before_weight)
        self.assertEqual(candidate.stride(), baseline.stride())
        self.assertNotEqual(candidate.data_ptr(), case[1].data_ptr())
        return old_trace, new_trace

    def test_parity_shapes_dtypes_and_state_history(self):
        cases = 0
        for dtype in (torch.float16, torch.float32):
            for lengths in ([17], [17, 2], [3, 31, 2, 5], [0, 23, 0, 2]):
                for kernel, dilation in ((1, 1), (4, 1), (4, 3)):
                    with self.subTest(
                        dtype=dtype, lengths=lengths, kernel=kernel, dilation=dilation
                    ):
                        self.check_case(
                            make_case(lengths, dtype, kernel=kernel, dilation=dilation)
                        )
                        cases += 1
        self.assertEqual(cases, 24)

    def test_null_and_noncontiguous_input_and_state(self):
        case = list(make_case([2, 11, 1]))
        # Preserve layout contracts with strided source tensors.
        case[1] = case[1].t().contiguous().t()
        case[3] = case[3].transpose(1, 2).contiguous().transpose(1, 2)
        case[5][0] = 0
        self.check_case(case)
        output, state, _ = run_case(case, True)
        self.assert_bytes_equal(output[:2], torch.zeros_like(output[:2]))
        self.assert_bytes_equal(state[0], case[3][0])
        self.assert_bytes_equal(state[-1], case[3][-1])

    def test_state_tail_matches_independent_history_oracle(self):
        lengths = [2, 19, 0, 7]
        case = make_case(lengths)
        layer, inputs, metadata, state, _, indices = case
        _, updated, _ = run_case(case, True)
        start = 0
        for row, length in enumerate(lengths):
            index = int(indices[row])
            initial = state[index, :, : layer.conv_state_len]
            if not metadata.has_initial_states_p[row]:
                initial = torch.zeros_like(initial)
            history = torch.cat((initial, inputs[start : start + length].t()), dim=1)
            expected = (
                history[:, -layer.conv_state_len :]
                if length
                else (state[index, :, : layer.conv_state_len])
            )
            self.assert_bytes_equal(updated[index, :, : layer.conv_state_len], expected)
            self.assert_bytes_equal(
                updated[index, :, layer.conv_state_len :],
                state[index, :, layer.conv_state_len :],
            )
            start += length

    def test_empty_and_no_state_storage(self):
        for lengths in ([], [0, 0], [7, 2]):
            case = list(make_case(lengths))
            case[3] = case[3][:0]
            baseline, old_state, _ = run_case(case, False)
            candidate, new_state, _ = run_case(case, True)
            self.assert_bytes_equal(baseline, candidate)
            self.assert_bytes_equal(old_state, new_state)

    def test_special_values(self):
        case = list(make_case([13, 3], torch.float32))
        case[1][0, :6] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-30]
        )
        self.check_case(case)

    def test_mixed_state_dtype(self):
        case = list(make_case([23, 2]))
        case[3] = case[3].float()
        self.check_case(case)

    def test_convolution_arguments_and_layout_are_unchanged(self):
        case = make_case([37, 3, 0, 2])
        seen = []
        original = F.conv1d

        def capture(inputs, weights, **kwargs):
            seen.append((inputs.clone(), weights.clone(), inputs.stride(), kwargs))
            return original(inputs, weights, **kwargs)

        with patch.object(F, "conv1d", side_effect=capture):
            self.check_case(case)
        self.assertEqual(len(seen), 2)
        self.assert_bytes_equal(seen[0][0], seen[1][0])
        self.assert_bytes_equal(seen[0][1], seen[1][1])
        self.assertEqual(seen[0][2:], seen[1][2:])

    def test_activation_and_indexing_use_legacy_operator_and_layout(self):
        class LayoutTrace(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.operations = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in (
                    torch.ops.aten.silu.default,
                    torch.ops.aten.silu_.default,
                    torch.ops.aten.masked_fill_.Scalar,
                    torch.ops.aten.index.Tensor,
                ):
                    value = args[0]
                    self.operations.append(
                        (str(func), tuple(value.shape), value.stride())
                    )
                return func(*args, **(kwargs or {}))

        # The CUDA contract includes op identity and memory layout, not only
        # CPU output values. The old in-place/view path passes byte parity but
        # fails this check. Include the dense transposed x_p return layout.
        for transposed in (False, True):
            case = list(make_case([37, 3, 0, 2]))
            if transposed:
                case[1] = case[1].t().contiguous().t()
            traces = []
            for enabled in (False, True):
                trace = LayoutTrace()
                with trace:
                    run_case(case, enabled)
                traces.append(trace.operations)
            self.assertEqual(traces[0], traces[1])
            self.assertTrue(any(op == "aten.silu.default" for op, _, _ in traces[1]))
            self.assertFalse(any(op == "aten.silu_.default" for op, _, _ in traces[1]))

    def test_convolution_entry_live_storage_matches_legacy(self):
        # Releasing packed_tokens or delaying output before F.conv1d changes
        # the workspace headroom seen by library algorithm selection. This is
        # distinct from testing convolution inputs or post-conv byte parity.
        for lengths in ([37], [37, 3, 0, 2]):
            baseline, candidate = self.check_case(make_case(lengths))
            self.assertEqual(len(baseline.convolution_live_bytes), 1)
            self.assertEqual(
                baseline.convolution_live_bytes, candidate.convolution_live_bytes
            )

    def test_production_lengths_reduced_channels_and_full_width_short_case(self):
        # Exact failing scheduler shape, scaled only in C to stay CPU-cheap.
        self.check_case(make_case([8080, 107], channels=8))
        self.check_case(make_case([97, 3, 2, 1], channels=32))
        self.check_case(make_case([11, 2], channels=10240))

    def test_padded_storage_peak_is_bounded(self):
        for lengths in ([2048], [2048, 17], [2048, 17, 3, 2]):
            case = make_case(lengths, channels=32)
            old, new = self.check_case(case)
            # Require a material storage reduction, not just a matching flag
            # or source pattern. Views must not hide a retained backing slab.
            # The conservative revision keeps the legacy live set until
            # convolution: it saves two padded slabs, rather than the first
            # version's three slabs plus unpadded output. Allow small state
            # and indexing buffers, but require at least 1.95 slabs saved.
            slab_bytes = len(lengths) * max(lengths) * case[1].shape[1] * 2
            self.assertGreaterEqual(old.peak_bytes - new.peak_bytes, 1.95 * slab_bytes)
            print(
                f"lengths={lengths}: logical tensor peak "
                f"{old.peak_bytes} -> {new.peak_bytes} bytes"
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
