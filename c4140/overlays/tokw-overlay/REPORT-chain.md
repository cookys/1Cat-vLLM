# REPORT-chain (renew-tokworkers-serve-v1, 2026-10-31)
Chain tokw overlay + tokcache overlay (840e0ede) through both sitecustomize shims, 4 workers, real tokenizer, stage on sys.path, CUDA hidden. Raw: REPORT-chain.json, concurrency/chain/*.out|err.

## Finding first: tokw v1 did not chain (fixed in v2)
With v1 (meta_path finder, same pattern as tokcache's), only ONE overlay installed per order: tokw-first -> tokcache only; tokcache-first -> tokw only. Each finder resolves the spec via PathFinder itself and never consults later finders, so the finder at meta_path[0] (the last shim run) wins. Fix = tokw v2 patches SourceFileLoader.exec_module at class level (order-independent; tokcache's own finder still works, its exec_module wrap calls the patched class method). tokcache_overlay.py itself was NOT modified. Anyone chaining a third overlay with a finder-style hook hits the same trap.

## Results (v2, sha8 da16969e; runs sequential, load avg ~8)
PYTHONPATH tokw-first = <lane>/concurrency/site:/data/bench/explore/renew-tokcache-cpu-v1/site ; tokcache-first = reverse.
| | tokw-first | tokcache-first |
|---|---|---|
| (c) install lines (tokw / tokcache) | 1 / 1 | 1 / 1 |
| (a) 4 staggered distinct, median wall | 1.227 s (reps 1.27/1.23/1.20) | 1.228 s (1.23/1.32/1.20) |
| (a) hit/miss/skip per rep | 0 / 4 / 0 | 0 / 4 / 0 |
| (a) per-request latency rep0 | 1.03/1.02/1.07/0.91 | 1.02/0.90/1.01/0.92 |
| (b) 4 conv x 21 req, bit-equal vs plain | 84/84 | 84/84 |
| (b) hits / misses / skip | 80 / 4 / 0 | 80 / 4 / 0 |
| (b) median renewal request | 0.062 s (p90 0.067) | 0.065 s (p90 0.068) |
| (b) median first request | 0.77 s | 0.82 s |
| (b) wall | 2.01 s | 2.07 s |
| receipt counters == in-process STATS | yes (hit 80, miss 4, reused 11181450, verify_fail 0) | yes (same) |

## Deviations from expectation
- (a) wall is 1.23 s, not about 0.9 s: tokcache's miss path costs about 0.3 s more per 254K encode (per-request 0.9-1.07 s vs 0.65 s plain); the overlay-only figure was 0.889 s. Still 2.2x better than the 2.74 s plain 4-worker serial. I did not isolate the cause (likely offset-mapping on the miss path); not measured.
- (b) hits are 0.062-0.065 s, about 2x the 0.024-0.03 s single-stream figure: 4 concurrent streams share the GIL / CPU with the Python-side cache work. Hit count 80/84 and bit-equality are as expected.
- Both orders behave identically within noise.
Pack unchanged, unsealed. v1 kept at concurrency/chain/tokw_overlay.v1.py (8df9a5fe); the earlier REPORT-concurrency numbers were produced with v1 (no other overlay present, so valid).
