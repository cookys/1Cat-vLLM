# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded spacing of prefill batches, without altering their token budget."""


class PrefillCadence:
    def __init__(self, decode_steps: int) -> None:
        self.decode_steps = decode_steps
        self.remaining = 0

    @property
    def enabled(self) -> bool:
        return self.decode_steps > 0

    def hold(self, has_eligible_decoder: bool) -> bool:
        if not has_eligible_decoder:
            # No wall-clock sleep and no credit carried across an idle period.
            self.remaining = 0
        return self.remaining > 0

    def scheduled(self, prefill_rows: int, decode_rows: int) -> None:
        if not self.enabled:
            return
        if prefill_rows:
            self.remaining = self.decode_steps
        elif decode_rows:
            self.remaining = max(0, self.remaining - 1)
        else:
            # The eligible decoder may have failed KV allocation / been
            # preempted. Release the gate for the next call rather than return
            # empty batches forever. This is a fallback, not a decode credit.
            self.remaining = 0
