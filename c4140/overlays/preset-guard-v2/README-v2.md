# preset-guard-v2

任何模式 **gmu≥0.88**，effective `prompt_logprobs is not None` → HTTP400 + request id/client/gmu log；**gmu<0.88** observe並原樣放行。包含0/False/-1及echo隱式路徑。模式僅記錄，不參與拒絕判定。源於nospec .92 OOM；不是dflash2專屬限制。

新目錄，原guard/launcher/stage不動，沒有core或.so改動。`preset_guard_v2.PresetGuard` 用現成ASGI middleware hook。實際 `scope['app'].state.vllm_config.cache_config.gpu_memory_utilization` 存在時優先；否則用launcher的 `VLLM_TEAM_PRESET_GMU`。缺失／非有限／不在(0,1]使啟用配置失敗。engine config在stage-FIX `api_server.py:358` 已寫入state；版本漂移若沒有該欄仍有同源launch env。自檢400收據含x-preset-guard-gmu，要求與預期一致。

兩新launcher沿用舊stage選擇：nospec=PVH serving-v2、dflash2=stage-FIX。共同 `GMU` 預設 .92；同一值進 `--gpu-memory-utilization` 與guard env，guard在這兩個preset固定開啟。上層serve.sh固定.80在前，新參數最後覆寫；拒絕額外CLI gmu覆寫，避免設定分離。`C4140_PRINT_ARGV=1`只列argv，不啟動。

```sh
GMU=0.92 FLASH_V100_PVH=1 C4140_PRINT_ARGV=1 bash /data/bench/explore/preset-guard-v2/launch-nospec-guard-v2.sh
GMU=0.92 C4140_PRINT_ARGV=1 bash /data/bench/explore/preset-guard-v2/launch-dflash2-guard-v2.sh
```

lead有窗後去掉PRINT_ARGV冷啟，再掛以下selfcheck。此包未自行啟停server或送GPU請求。

```sh
/data/venvs/1cat-main-integp7pr-p8-gvB-integ/bin/python -B /data/bench/explore/preset-guard-v2/selfcheck-v2.py --execute --preset nospec --gmu 0.92 --base http://127.0.0.1:8021 --out /data/bench/explore/preset-check-nospec-gmu92-v2.json
```

自檢送16,384 token IDs、prompt_logprobs1、max_tokens1。高gmu **必須本guard的400收據**（param/code/request_id/gmu一致）及之後三次health200，否則FAIL；200但存活同樣policy FAIL。低gmu保留200或400且存活的原規則。預設plan-only；`--gmu`必填，輸出不能放封包內/覆寫既有檔。dflash2改preset及輸出檔；不以切換模式放寬策略。

CPU14案PASS，`../preset-guard-v2-cpu/tests.log`：真FastAPI/ASGI、真1Cat protocol、兩模式.80/.879999/.88/.92/1邊界、nospec.92 selfcheck直接走middleware並驗backend零呼叫、錯放200/錯receipt/503會FAIL、engine config .92覆蓋stale env .80、stream/echo/chunk replay。未開TCP listener，CVD空；live startup selfcheck仍待lead。

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/bench/s2-impl/c1fix/stage-FIX PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/venvs/1cat-main-integp7pr-p8-gvB-integ/bin/python -B /data/bench/explore/preset-guard-v2/test_guard_v2.py > /tmp/preset-guard-v2-tests.log 2>&1
```

其他n/top-k/best_of/output/prompt上限未改，見LIMITS-v2.md；封版後測試/自檢輸出只寫包外。
