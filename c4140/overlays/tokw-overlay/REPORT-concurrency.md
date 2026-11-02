# REPORT-concurrency (renew-tokworkers-serve-v1, 2026-10-31)
Hypothesis (Fable): one _batch_encode_loop per queue key awaits one run_in_executor inline, so --renderer-num-workers 4 cannot parallelize non-batching renewals. **Confirmed.**

Test: 4 distinct ~254K-token prompts, 0.1 s stagger, one loop, real HF tokenizer (maybe_make_thread_pool as HfRenderer), KW=add_special_tokens=False,truncation,max_length=262145. 3 reps, medians. Solo encode 0.68-0.75 s.

| arm | workers | median wall | per-request latency (median, req0..3) |
|---|---|---|---|
| plain loop | 4 | 2.737 s (2.74/2.69/3.02) | 0.655 / 2.635 / 2.535 / 2.435 |
| plain loop | 1 | 3.172 s | 0.82 / 3.07 / 2.97 / 2.87 |
| overlay | 4 | **0.889 s** (0.96/0.89/0.89) | 0.661 / 0.662 / 0.648 / 0.587 |
| overlay | 1 | 2.649 s (serialized) | 0.648 / 1.176 / 1.75 / 2.348 |

Reading: the flag alone is falsified (4 workers = 2.74 s, same as serial). The 1-worker plain arm is a bit slower than the 4-worker plain arm (3.17 vs 2.74), so extra threads only help marginally (executor hand-off), not as parallelism. With the overlay the 4 encodes overlap almost perfectly (0.89 s wall, each ~0.65 s), so the Rust tokenizer does run concurrently across threads. Machine load avg ~9 on 112 CPUs during runs (mple2 co-tenant), so absolute numbers carry some noise.

Correctness through the patched loop (test_bit.py, 21-text 20-renewal conversation, ids compared to direct tok()): seq w4 21/21, concurrent w4 21/21, seq w1 21/21, concurrent w1 21/21 (also plain 21/21). Exception in the tokenizer still reaches the caller (ValueError). Batching semantics unchanged: dbg.py shows identical queue keys and call shapes with and without overlay (KW key = 'other' -> per-prompt str calls; truncation-only key -> one batched list call).
Note: test_bit.py "batch_same_kwargs" shows str,str,str under overlay but list,str in the plain run of the same script; dbg.py (clean, isolated) shows identical shapes in both modes, so I attribute it to timing of the 2 ms window in the multi-test script, not to a semantic difference. Teardown prints "Task destroyed but it is pending" / "cannot reuse already awaited coroutine" in both plain and overlay (pre-existing loop shutdown noise).

Overlay (stage untouched; chainable shim, re-entry guard, TOKW_OVERLAY=0 disables; sitecustomize chains to the next sitecustomize, so it can sit beside the tokcache shim):
- /data/bench/explore/renew-tokworkers-serve-v1/concurrency/tokw_overlay.py sha8 8df9a5fe
- /data/bench/explore/renew-tokworkers-serve-v1/concurrency/site/sitecustomize.py sha8 0cd1080e
- tests: test_conc.py 3797b1f6, test_bit.py ab576c20; raw: plain.json overlay.json bit_plain.json bit_overlay.json
Design implication: the lever needs overlay (or a stage patch) in addition to the two argv flags; plus tokcache chaining must be tested together (not done here). Pack unchanged and unsealed.
