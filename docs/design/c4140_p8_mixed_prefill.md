# P8: opt-in mixed-prefill budget

Python-only adaptation of upstream `1cabc7ce9` on P7 `104c33e55`, with
prefix-retention fix `f6707f55b` cherry-picked as `4a470cc3a`. Native kernels,
KV geometry, outer block-size validation, and speculative admission are unchanged.

## Controls

| CLI suffix | Default | Meaning |
|---|---:|---|
| `--mixed-prefill-step-latency-ms` | 0 | Zero disables the entire controller. |
| `--mixed-prefill-max-tokens` | 512 | Aggregate prefill cap, excluding reserved decode/verify rows. |
| `--mixed-prefill-min-tokens` | 128 | Adaptive allowance floor; must fit the effective cap/batch budget when ON. |

With target > 0, min=max selects a static cap without CUDA timing events.
With target > 0 and min<max, completed GPU step timings adapt the allowance.
For example, ten q8 requests reserve 80 rows; a 512-row prefill allowance permits
592 total rows if the global token budget permits. This is not a 512-row batch cap.
Pure prefill retains its existing token budget and Mamba alignment splits.

The adaptive seed is 512, clamped to the configured range. The EWMA uses total
mixed-step elapsed time per prefill row; it does not estimate a decode-only floor.
Its lower allowance bound prevents an unattainable target from shrinking chunks
to 16 rows indefinitely. Alignment, token availability and KV pressure can still
schedule fewer rows than the floor. The target is not a latency guarantee.
The controller ignores undersized boundary/tail samples, grows by at most 25%
per update, and uses 16-row quantization above its floor.

## Scheduler and feedback contract

Reservation covers eligible resident query rows, including speculation and async
placeholders, before either running or waiting prefill is capped. It leaves FCFS
queue order unchanged. Replays after preemption count as prefill. A failing prefill
allocation skips that request instead of evicting a resident decoder. **Token
reservation does not guarantee KV allocation:** a decoder's own allocation can
still invoke the existing preemption policy; `Mixed prefill KV pressure:` records
that event. Full-input admission and lookahead allocation remain unchanged.

Supported controller scope: CUDA, chunked prefill, PP=1, non-encoder-decoder,
`disable_chunked_mm_input=False`, no DDTree tree verification. Unsupported cases
retain the original scheduler. Multimodal encoder correctness/performance has not
been validated; the intended GPU experiment is text-only 27B DFlash2.

V1 and V2 runners time only adaptive mixed steps on the current stream. Up to eight
pending event pairs are retained, readiness is queried without synchronization,
and a delayed sample carries its own counts. If several samples become ready,
the most recent completed one is returned. Pure decode does not record events.
Counters count completed scheduled mixed steps, independently of delayed timing:

- `vllm:mixed_prefill_steps_total`
- `vllm:mixed_prefill_tokens_total`

`Mixed prefill control active:` is logged once on the first mixed step. It is not
a claim that the cap actually shortened that particular step.

## Validation and numerical scope

CPU tests use real Scheduler/AsyncScheduler and KV managers, local model config
only. OFF traces cover eight combinations (sync/async, spec 0/7, ample/pressured KV)
and 20 steps each, checked against the original base schedule/state/refcount trace.
Their reference hashes are in `tests/v1/core/mixed_prefill_off_reference.json`.
Sub-block tests check real worker copy decisions with a scalar recurrence, retained
snapshot immutability, and real scheduler prefix-hit/EAGLE splits for block
2048/4096 and a nondividing cap of 320. Additional cases cover ten q8 decoders plus
a 100K prefill, adaptive floor, static cap, allocation failures, async in-flight
steps, full-input admission, CLI parsing, and Prometheus counter draining.

These checks prove host scheduling invariants, not GPU floating-point equality.
ON changes chunk/batch shapes and may change accumulation paths and sampled token
sequences. No E1 or strict distribution-equivalence claim is made for ON. Compare
OFF to the existing stack, and ON/on plus ON/off on loader instances before any
quality decision. GPU validation and performance measurements remain outstanding.

Run the three `test_mixed_prefill_*.py` files and `test_mamba_sparse_retention.py`
with `pytest --noconftest`, `CUDA_VISIBLE_DEVICES=` and a CPU-capable serving Python.
No weights, proposer, CUDA events or GPU kernels are constructed by these tests.
