# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feedback control of prefill work sharing a step with resident decoders."""

import math

from vllm.v1.outputs import MixedPrefillTiming


class MixedPrefillBudget:
    """Learn a token budget from completed GPU steps, without host-side waits.

    The first mixed step uses a conservative seed. Subsequent samples adapt to
    the actual model, device, TP size and context. This is a latency target,
    not a hard deadline: even a decode-only step can exceed the target.
    """

    def __init__(
        self, max_tokens: int, target_ms: float, min_tokens: int = 128
    ) -> None:
        if target_ms > 0 and min_tokens > max_tokens:
            raise ValueError(
                "mixed_prefill_min_tokens must not exceed the effective maximum "
                "(mixed_prefill_max_tokens and scheduler token budget)"
            )
        self.max_tokens = max_tokens
        self.min_tokens = min(min_tokens, max_tokens)
        self.target_ms = target_ms
        self.tokens = max(self.min_tokens, min(512, max_tokens))
        self.ms_per_token: float | None = None

    def update(self, sample: MixedPrefillTiming | None) -> None:
        if (
            sample is None
            or self.target_ms <= 0
            or self.min_tokens == self.max_tokens
            or sample.prefill_tokens <= 0
            or sample.decode_tokens <= 0
            or not math.isfinite(sample.elapsed_ms)
            or sample.elapsed_ms <= 0
        ):
            return
        # A tiny boundary/tail chunk is dominated by fixed decode overhead.
        # It must not teach the controller that a full chunk is equally costly.
        if sample.prefill_tokens < sample.budget_tokens * 0.75:
            return
        cost = sample.elapsed_ms / sample.prefill_tokens
        self.ms_per_token = (
            cost if self.ms_per_token is None else 0.5 * self.ms_per_token + 0.5 * cost
        )
        desired = self.target_ms / self.ms_per_token
        # Grow slowly on fast devices; react promptly to a long-context stall.
        desired = min(desired, self.tokens * 1.25)
        quantum = min(16, self.max_tokens)
        self.tokens = max(
            self.min_tokens,
            min(self.max_tokens, int(desired) // quantum * quantum),
        )
