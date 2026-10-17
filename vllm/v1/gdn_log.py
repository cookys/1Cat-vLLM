# SPDX-License-Identifier: Apache-2.0
"""plan 075 G-LOG: GDN speculative state as one snapshot plus update logs.

DFlash2 verification used to store ``1 + K`` full FP32 recurrent-state
snapshots per GDN layer (one per verified token), each in its own cache page.
With ``VLLM_DFLASH2_GDN_LOG=1`` a request owns one page per GDN layer instead.
The page keeps

* the conv state (unchanged),
* one FP32 state ``H1`` (the state after the first token of the last verified
  round, or the canonical state when nothing is pending),
* two banks of FP32 rank-1 update records for steps 1..K of the last round
  (``decay``, ``key[K]``, ``delta[V]`` per value head), and
* a small header whose ``epoch`` word selects the bank a round writes.

A GDN step is ``h <- h * decay; h <- h + delta (x) key``. Replaying the
recorded ``decay``/``delta``/``key`` of steps 1..a-1 onto ``H1`` reproduces the
accepted state with the same FP32 operations the verifier performed.

Everything in this module is pure Python so the allocator, the runner and the
tests share one definition of the byte layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import MambaSpec

LAYOUT_VERSION = 1
# Step slots kept per bank. Slot 0 is never written (H1 is the page state) but
# is kept so that step index == draft position.
LOG_STEPS = 8
LOG_BANKS = 2
ALIGN_BYTES = 256
HEADER_BYTES = 256
# Header word offsets (int32 words).
HDR_WORD_VERSION = 0
HDR_WORD_EPOCH = 1


def _align(value: int, to: int = ALIGN_BYTES) -> int:
    return (value + to - 1) // to * to


def gdn_log_enabled() -> bool:
    """Raw flag. Use :func:`gdn_log_active` to also check the configuration."""
    return bool(envs.VLLM_DFLASH2_GDN_LOG)


@dataclass(frozen=True)
class GdnLogLayout:
    """Byte layout of one GDN page. Offsets are relative to the page start,
    ``*_from_ssm`` offsets relative to the start of the FP32 SSM state."""

    conv_bytes: int
    hv: int
    k: int
    v: int
    page_size_bytes: int
    ssm_itemsize: int = 4

    @property
    def ssm_bytes(self) -> int:
        return self.hv * self.v * self.k * self.ssm_itemsize

    @property
    def ssm_offset(self) -> int:
        return self.conv_bytes

    @property
    def record_words(self) -> int:
        """FP32 words of one (step, value head) record: decay + key + delta."""
        return 1 + self.k + self.v

    @property
    def step_words(self) -> int:
        return self.hv * self.record_words

    @property
    def payload_bytes(self) -> int:
        return LOG_STEPS * self.step_words * 4

    @property
    def bank_stride_bytes(self) -> int:
        return _align(self.payload_bytes)

    @property
    def bank0_offset(self) -> int:
        return _align(self.ssm_offset + self.ssm_bytes)

    def bank_offset(self, bank: int) -> int:
        assert 0 <= bank < LOG_BANKS
        return self.bank0_offset + bank * self.bank_stride_bytes

    @property
    def header_offset(self) -> int:
        return self.bank0_offset + LOG_BANKS * self.bank_stride_bytes

    @property
    def used_bytes(self) -> int:
        return self.header_offset + HEADER_BYTES

    # Offsets in FP32/int32 words from the start of the SSM state: this is how
    # the kernels address the page (they hold a pointer to the SSM view).
    @property
    def bank0_words_from_ssm(self) -> int:
        return (self.bank0_offset - self.ssm_offset) // 4

    @property
    def bank_stride_words(self) -> int:
        return self.bank_stride_bytes // 4

    @property
    def header_words_from_ssm(self) -> int:
        return (self.header_offset - self.ssm_offset) // 4

    def check_fits(self) -> None:
        if self.used_bytes > self.page_size_bytes:
            raise ValueError(
                f"G-LOG needs {self.used_bytes} bytes per GDN page "
                f"(conv {self.conv_bytes} + ssm {self.ssm_bytes} + 2 banks of "
                f"{self.bank_stride_bytes} + header {HEADER_BYTES}) but the page "
                f"is {self.page_size_bytes} bytes"
            )

    def record_word_index(self, bank: int, step: int, head: int) -> int:
        """Word index (from the SSM start) of the (bank, step, head) record."""
        assert 0 <= step < LOG_STEPS and 0 <= head < self.hv
        return (
            self.bank0_words_from_ssm
            + bank * self.bank_stride_words
            + step * self.step_words
            + head * self.record_words
        )


def layout_from_mamba_spec(spec: MambaSpec) -> GdnLogLayout:
    """Derive the layout from a GDN ``MambaSpec`` (conv, temporal) pair."""
    import torch

    from vllm.utils.torch_utils import get_dtype_size

    if len(spec.shapes) != 2:
        raise ValueError(f"G-LOG expects (conv, ssm) states, got {spec.shapes}")
    conv_shape, ssm_shape = spec.shapes
    if len(ssm_shape) != 3:
        raise ValueError(f"G-LOG expects ssm shape (HV, V, K), got {ssm_shape}")
    if spec.dtypes[1] != torch.float32:
        raise ValueError(
            f"G-LOG requires an FP32 SSM state, got {spec.dtypes[1]}; the "
            "update logs replay FP32 arithmetic"
        )
    conv_elems = 1
    for dim in conv_shape:
        conv_elems *= dim
    conv_bytes = conv_elems * get_dtype_size(spec.dtypes[0])
    hv, v, k = ssm_shape
    layout = GdnLogLayout(
        conv_bytes=conv_bytes,
        hv=hv,
        k=k,
        v=v,
        page_size_bytes=spec.page_size_bytes,
    )
    layout.check_fits()
    return layout


def gdn_log_unmet_conditions(vllm_config: VllmConfig) -> list[str]:
    """Reasons the G-LOG route cannot serve this configuration (empty = ok)."""
    from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

    problems: list[str] = []
    spec_config = vllm_config.speculative_config
    if not uses_dflash_selector_engine(vllm_config):
        problems.append("speculative method must be the DFlash selector engine")
    if vllm_config.cache_config.mamba_cache_mode != "align":
        problems.append("mamba_cache_mode must be 'align'")
    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        problems.append("VLLM_USE_V2_MODEL_RUNNER=1 is required")
    if spec_config is not None:
        k = spec_config.num_speculative_state_tokens()
        if not 1 <= k < LOG_STEPS:
            problems.append(
                f"num_speculative_state_tokens={k} must be in [1, {LOG_STEPS - 1}]"
            )
    return problems


def gdn_log_active(vllm_config: VllmConfig) -> bool:
    """Whether this engine runs the G-LOG route.

    Off when the flag is unset or there is no speculative config (the nospec
    engine keeps its exact layout). A set flag on a configuration the route
    cannot serve is an error: the page count must be decided before any page is
    allocated, never after a fallback.
    """
    if not gdn_log_enabled() or vllm_config.speculative_config is None:
        return False
    problems = gdn_log_unmet_conditions(vllm_config)
    if problems:
        raise ValueError(
            "VLLM_DFLASH2_GDN_LOG=1 is not supported here: " + "; ".join(problems)
        )
    return True
