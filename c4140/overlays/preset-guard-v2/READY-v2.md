# CPU_READY: preset-guard-v2

Manifest sha16 `1c73ab72bab238c4`. Any nospec/dflash2 preset at GMU ≥0.88 rejects effective non-null prompt_logprobs with HTTP400 and request/client log. GMU<0.88 observes. Engine config takes precedence over launcher metadata. No algorithm, stage or native binary changes.

14 CPU FastAPI/ASGI tests PASS (real protocol, nospec .92 selfcheck, rejection counterexamples); launch plans verify last GMU and middleware. Evidence in ../preset-guard-v2-cpu/, pinned by manifest. No TCP listener or GPU use.

Next lead-owned nospec .92 boot:
```sh
GMU=0.92 FLASH_V100_PVH=1 bash /data/bench/explore/preset-guard-v2/launch-nospec-guard-v2.sh
/data/venvs/1cat-main-integp7pr-p8-gvB-integ/bin/python -B /data/bench/explore/preset-guard-v2/selfcheck-v2.py --execute --preset nospec --gmu 0.92 --base http://127.0.0.1:8021 --out /data/bench/explore/preset-check-nospec-gmu92-v2.json
```

Dflash2: launch-dflash2-guard-v2.sh, selfcheck --preset dflash2 and a fresh output path. No standalone GPU window requested; attach to next scheduled boot. High-GMU selfcheck requires matching guard400 receipt plus three healthy probes; 200 even with a living server is FAIL. Live startup check remains pending.

All code/evidence/dependencies pinned in manifest-v2.json; sha256.txt pins manifest and READY. Frozen package: future test logs outside this directory. README-v2.md has full usage; LIMITS-v2.md records unchanged remaining limits.
