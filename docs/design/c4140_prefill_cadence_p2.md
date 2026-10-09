# P2 · Prefill cadence 原型與 GPU 事前登記（2026-10-09）

狀態：CPU 原型；GPU 未跑、未採用。engine branch `p072-astra-prefill-cadence`，基底 `de01359b8`（P7 + prefix retention + P8 預設關閉 + V2 dispatch stats）。工作樹 `/data/src/1cat-wt-astra-cadence`。本稿的 `vllm/`、`tests/`、`benchmarks/` 皆指這個 worktree；正式知識副本在 llm-playground `doc/plan/072-prefill-cadence-p2-prereg-20261009.md`。

## 1. 動機與範圍

P0 m44 以每秒 iteration 差分得 M/D ≥9.0，只是下界；超過一秒的混合步被量化成1000ms，不能用來驗證 cadence。來源：llm-playground `doc/plan/072-concurrency-merged-ranking-20261009.md:70–82`，原始 `/data/bench/m44/{summary.json,metrics-*.jsonl,client/*.jsonl}`。本原型不估已得收益，保留現行4096 align/chunk路線，在chunk之間安排k個純decode批次。

它不能縮短一次長prefill kernel，也不減少prefill總工作量；可能延長cold TTFT。改變的是decode進度的分配，以及純decode是否能回到FULL graph。不得把P0理想公式或排程比例直接列為實測收益。

## 2. 實作契約

- `--prefill-cadence-decode-steps 0` 預設OFF；實驗ON為1、3（合法0…64）。`--prefill-cadence-step-log` 獨立預設OFF。OFF+logOFF不掃描cadence metadata、不建CUDA event。設定與CLI：`vllm/config/scheduler.py:102–113,259`、`vllm/engine/arg_utils.py:562,1447,2315`。
- cadence>0且`--mixed-prefill-step-latency-ms >0`在config階段拒絕，包含P8 min=max的靜態模式。原型拒PP/DP>1、encoder-decoder、非chunked、disable_chunked_mm_input、tree verifier；逐步timer要求V2。不能無聲退成OFF。TP4可以，所有rank接收同一SchedulerOutput。
- 任何有prefill列的**成功非空排程批次**把剩餘額度設k；接下來k個非空純decode批次各扣1。hold時running partial-prefill留原位置，waiting/skipped queues完全不取出，確保短prefill/tail也不混入「純decode」步；額度完了走原有完整budget。這也會作用於短prefill的最後一步，不是只判 `rows==4096`，避免尾段與warm append鑽過gate。來源：`core/sched/scheduler.py:597–615,657,970,1394–1419`、`prefill_cadence.py:15–32`。
- **async語義**：額度在KV配置成功、preemption處理後、`_update_after_schedule`前保留，算的是FIFO將要執行的批次；不是scheduler呼叫次數，也不等待GPU完成才再計數。engine `core.py:606–687`依序submit、appendleft/pop消費；`async_scheduler.py:35–59`管理placeholders。這是對早期提案「成功完成」文字的精確化：沒有新增GPU fence；回報不重複扣額度。若execute致命失敗，原引擎錯誤政策不變，不再繼續宣稱完成了這些步。
- 沒有eligible decoder時立即放行prefill並清舊額度；eligible decoder在配置期間被preempt/失敗導致空批次，也清gate供下次重試。空批次不被算作成功decode。沒有sleep、按秒節流或新KV保留策略。
- **prefill進度保底**：在已有可配置的prefill、原排程器budget/KV/seq槽可前進的前提下，cadence至多延後k個非空decode批次，隨後必有一次不受cadence限制的prefill機會；固定c/無資源失敗測試中每k+1批次跨一個4096邊界。不能保證過量admission、持續KV不足或原priority策略下的無條件wall-time SLA。這些不能當作cadence成功，也不新增繞過FCFS/priority的餓死風險。
- `_mamba_block_aligned_split`、align4096的config斷言、KV manager、attention/CUDA kernel均不改。來源：scheduler原函式`449–513`無diff；4096 MambaSpec測試與原始OFF schedule/refcount/queue hash回歸。

**沒有 oldest-age 覆寫**：早期提案的按年齡強制放行控制器未實作。本版以 **k+1 結構保底**取代，沒有年齡計時器、wall-time上限或額外priority排序；cold queue與TTFT仍須獨立過閘。

### 數值界線

OFF的排程序列/KV refcounts/queue/request state逐值等於固定P7+retention基底fixture（新的`cadence_step=None`欄位不計入fixture）；不把CPU排程等價說成GPU已驗E1。ON不改token/quant/算子，但批次形狀、接受數與state更新分段可能改變浮點結果，**不得預先宣稱E1或嚴格分布等價**。GPU要求OFF對生產基線token/logprob parity；ON記錄相同seed的token、logprob差異及self-replay底線。若數值變動，保留為需owner核准的候選，不直接採用。

## 3. 主量測：逐步server時間，不用一秒iteration倒數

`--prefill-cadence-step-log`兩臂都開。每個TP rank，在V2非dummy `execute_model`前record start，在sampling、postprocess、draft與connector post-forward排入main stream後record end。稍後`event.query()`就緒才讀`elapsed_time`，**不呼叫synchronize**；最多128 pending，超額/unfinished明印`PREFILL_CADENCE_TIMING_GAP`，不是靜默丟樣本。來源：`worker/cadence_step_timer.py:16–70`、`worker/gpu/model_runner.py:1497,1544,2001`。

每步兩種紀錄：

1. `PREFILL_CADENCE_STEP` JSON：step_id/rank、prefill/decode列、人數、pending_prefill_reqs、hold/remaining、FULL/NONE路由、padding、**stream_ms**、stream_gap_ms、host_enqueue_ms、host_begin_ns。stream_ms含兩event間的GPU串流等待/host提交空隙，並非kernel-only busy time；gap為前一個end至本step start。copy stream在main-stream event以外的尾端不包含。host_enqueue_ms是CPU enqueue跨度，不能當GPU完成時間。
2. scheduler `PREFILL_CADENCE_COMPLETE`：同step_id的**實際commit給EngineCoreOutput**的decode token數；已做stop/abort/rejection處理，排除被丟棄的raw sampled token。來源：scheduler`2067–2087`。不記request字串、token IDs或prompt。

`benchmarks/prefill_cadence_readout.py`以step_id對齊四rank與completion，時間取同step四rank的max。主rate為 `Σdecode_committed_tokens / Σ(decode_reqs × (max_rank_stream_ms + max_rank_gap_ms)/1000)`；這是server decoder人秒rate，不等同client SSE p50。無gap版並列供辨識host idle。使用max gap+max duration是保守帳，不宣稱它是同步全機wall精確值。

**干擾口徑預先固定**：step開始時仍有running prefill或waiting/skipped新請求即`pending_prefill_reqs>0`；有decoder的步歸interfered，包含cadence插入的pure-decode步。這是engine待處理prefill區間代理，不等同client first-token前區間；blocked waiting也計入，需保留計數。pending=0的decoder步歸steady。另按真正mixed/decode/prefill三類列p50/p90/p99，不能把插入的decode誤分類成steady來抬數字。

空批不納入。測試覆蓋4099.065ms仍完整保留、延後event與缺rank/重複step會判INCONCLUSIVE。每格只含一個engine lifetime；記測試前後step_id，結尾發短drain請求直到最後正式step四rank皆寫出，再用`--first-step/--last-step`排除warmup/drain。出現未補齊step、timing gap、rank metadata不一致 → INCONCLUSIVE。**必需physical phases=mixed/decode，各≥20步；必需regimes=steady/interfered，各≥20步**，四個計數各自檢查；pure-prefill沒有decoder分母，只診斷，不設20步門檻。比較欄固定`regimes.interfered.per_decoder_tok_s_with_gap`、`regimes.steady.per_decoder_tok_s_with_gap`，不得換用phase的rate或無gap欄。

readout還要求每格`128K`與`200K`兩個cold cohort各≥10筆；少一筆、cache未知或缺outcome → `INCONCLUSIVE`，timeout/error → `FAIL`且不刪樣本。輸入`--cold-json`契約為`{"cohorts":{"128K":[{"ok":true,"ttft_s":12.3,"cached_tokens":0}],"200K":[...]}}`，陣列保留該格**所有預定cold請求**，包括失敗列。這是wrapper從原始回應整理的證據檔，不是把team summary當樣本。cache須來自usage明確值或已驗證的server-details省略零契約，不可把null填零。p90欄=`cold.cohorts.{128K,200K}.ttft_s_p90`，兩種長度分開判。`status=COMPLETE`只表示本格樣本齊全，不代表跨臂性能PASS。以上規則皆在`benchmarks/prefill_cadence_readout.py`實作；測試驗19/20步與9/10筆cold邊界。

timer會改變allocator/event footprint，故兩臂同開，另用相同workload各一格logOFF檢查方向，不能把logON快當timer收益。補量規則見§4.1。

離線命令（CPU）：

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 /data/venvs/1cat-main-integp7pr-p8/bin/python \
 /data/src/1cat-wt-astra-cadence/benchmarks/prefill_cadence_readout.py \
 /data/bench/<window>/<cell>.serve.log --ranks 4 \
 --first-step <first> --last-step <last> \
 --cold-json /data/bench/<window>/<cell>.cold.json \
 --out /data/bench/<window>/<cell>.steps.json
```

## 4. GPU前凍結的P2門檻

決策對象為27B NVFP4 KV、TP4、DFlash2 K7、P7既有batched生產路線，P8=OFF；不夾帶P9、kernel prototype或parking變更。沿用同model/.so/venv、block/mamba4096、同max-num-seqs/batch budget、同pool IDs、同arrival/plan/seed與cache暖機；兩臂都先完成AOT後重啟成loader。先驗OFF token/logprob ==同基底生產底線；pool若不同先釘相同ID數。每格記完整argv、commit、.so hash、cache命中、preempt、KV peak、FULL比例、RAM/las swap。另必記 **so_mapped**：暖機後、正式流量前，從四個TP worker的`/proc/<pid>/maps`擷取已載入.so的realpath、device/inode與磁碟SHA256，含flash_attn_v100/flash_qla/vllm自訂ops，保存PID→rank。只hash預計安裝檔不算；deleted mapping、無法讀maps、兩臂binary內容不同 → INCONCLUSIVE。不能用「同venv」推定同binary。

**矩陣**：OFF(k0)對k1、OFF(k0)對k3分開判。每組採`A B B A / B A A B`，每臂4格（超過≥3），每個c10與c10fit plan分層；共4組32格，OFF不跨組借用；不得把k1/k3挑最佳單格、或兩個plan混成一個數。每格用同一份team_traffic_workload plan、同到達序；另跑固定arrival cold128K/200K各≥10個請求/**格**，所有預定請求均保留，符合§3機械閘。若窗不夠，只能先交INCONCLUSIVE的smoke，不能改門檻。

每一plan × k的**所有合併門檻**：

| 門檻 | 事前定義 |
|---|---|
| interfered | 各格server人秒rate中位數 ON/OFF **≥1.25**（§3主rate）；client windows p50/p10同時附列，不混換分母 |
| steady | 同口徑server rate中位数 ON/OFF **≥0.95** |
| whole-turn | 現行workload `summary.aggregate.per_stream_tok_s.p50`的跨格中位数 ON/OFF **≥1.15**；且`ratio−1 > max(spread_OFF, spread_ON)`，spread明定`(max_cell−min_cell)/min_cell`；這是既有whole-turn decode rate，TTFT另量 |
| cold TTFT | 固定arrival冷請求（cached_tokens≤1 block）的TTFT p90：128K/200K各自的各格p90跨格中位數 ON/OFF **≤1.50**；所有請求保留，timeout/error不得剔除。這是lead本輪裁定，取代早期提案1.25；**1.25僅敏感度註記，不計分**。樣本不足或cache欄空 → INCONCLUSIVE |
| completed goodput | 每格固定plan的成功完成request數 /（最後一個預定request完成或deadline − 第一個原始arrival），含TTFT/排隊/drain，跨格中位數 ON/OFF **≥0.95**。不以decode-only tok/s替代；timeout/error仍計入wall分母且本格存活閘失敗。team與fixed-arrival cold兩cohort分開都須過 |
| cold queue | §4.1固定arrivalcohort的`Q(t)=已到達但未見first token的cold請求數`，預定穩定到達區間OLS斜率 **≤0**、末樣本 **≤** 首樣本，最後arrival+900s內Q回0且全部完成；任一不過 → NO-GO。不能只憑closed-loop最終drain宣稱queue不增長 |
| 進度/存活 | 所有已admit prefill最終完成；无engine error/OOM、无新增preemption；以event/log驗證k間隔，資源不足另外標記而非當成功純decode步 |
| 數值 | OFF serving parity及self-replay通過；ON對OFF IDs/logprob差分＋ON self-replay；非bitwise需owner看數值/任務品質後另決定，速度PASS不自動授權部署 |

whole-turn欄位來源：llm-playground `scripts/c4140-ab/team_traffic_workload.py:804–815,860–863`；不可改成aggregate總tok/s替代。同時列whole p10、TTFT p50/p99、client interfered時間比例、p99 token gap。任一性能/TTFT門檻失敗 → 該k/plan NO-GO；缺證據 → INCONCLUSIVE；兩組都過才可宣稱整體通過。此處沒有「先量再改」的容忍帶。

CLI差異（由lead的GPU wrapper執行，本人未啟動服務）：

```text
兩臂共用：--mixed-prefill-step-latency-ms 0 --prefill-cadence-step-log
A：--prefill-cadence-decode-steps 0
B1：--prefill-cadence-decode-steps 1
B3：--prefill-cadence-decode-steps 3
```

### 4.1 固定到達、時間預算與加格規則

cold cohort：每格保持同樣c10 resident decode負載，t=0起每60秒送一筆salted cold，128K與200K交替，共20筆(各10)。兩臂用同一到達表，原始arrival不隨前一筆完成時間延後；若容量不支持既定cohort，記前置NO-GO，不臨場改admission。每筆deadline=arrival+900s，最後到達t=1140s，最晚t=2040s結束。每秒由原始arrival/first-token事件重建Q；對t=300,360,…,1140的每個時間點，用其前10秒的Q均值作OLS樣本，避免與瞬間arrival排序相撞。此有限窗口的queue不增長不等於長期穩定性的證明。正式step分析區間涵蓋team及cold流量，排除warmup/drain。

**時長是工程預算，尚未實測cadence**：m34 `off{1,2,3}.json` 原始requests的max(t_end_rel)−min(t_send_rel)=1483.7–1500.6s，即team約25分鐘。預留每格team 25–35min + cold 20–34min +啟動/loader/reset 5–15min，約50–84min/格；一個k×plan的8格約6.7–11.2h，完整32格約27–45h，另加初次編譯/parity與logOFF方向檢查。這不是一個短GPU窗：lead可先排一組，未跑的組仍INCONCLUSIVE。每格hard cap=90min，外層依格數另加還原餘裕，不共用單一短alarm。

**加格只准一次，且成對**：初始8格全部保留。若不足20步/10個cold、時間缺口或有明確infra失敗，不挑好段補入；整組追加一次`A B B A`，上限12格/組(每臂6)。失敗格原樣留檔；所有樣本齊全的格都進統計，不能因速度慢排除。若只有spread令whole-turn門檻未過，也只准同樣一次預定ABBA追加；完整性能或TTFT硬門檻明確失敗則NO-GO，不以持續加格救結果。追加後仍缺樣本 → INCONCLUSIVE，仍未過門檻 → NO-GO。若改workload/到達率，須另立新版本預登記，不能與本組混算。

## 5. Overlay、CPU驗證與交付

相對base需覆蓋7個runtime Python檔：

```text
vllm/config/scheduler.py
vllm/engine/arg_utils.py
vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/prefill_cadence.py (new)
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/cadence_step_timer.py (new)
```

無.so/CUDA/allocator改動；只在lead新venv副本overlay，勿將整個worktree當PYTHONPATH遮掉flash_qla。混入其他新補丁時按diff合併，不用舊整檔覆寫更新版。

CPU命令（驗證venv、完整命令應與log一起保存）：

```sh
cd /data/src/1cat-wt-astra-cadence
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 env PYTHONPATH=/data/src/1cat-wt-astra-cadence:/data/bench/astra-host-parking/test-deps \
 /data/venvs/1cat-main-integp7pr-p8/bin/python -m pytest --noconftest \
 tests/v1/core/test_prefill_cadence.py \
 tests/v1/core/test_mixed_prefill_budget.py \
 tests/v1/core/test_mixed_prefill_off_reference.py \
 tests/v1/worker/test_gpu_model_runner_v2_cudagraph_metrics.py \
 tests/v1/worker/test_gpu_model_runner_v2_prefill_dispatch.py \
 tests/v1/worker/test_cadence_step_timer_dispatch.py -q
```

結果：原版133項；本次審查修訂 **143 passed**，`/data/bench/astra-cadence/review-tests.log`；含實際CLI/import且`torch.cuda.is_initialized()==False`、k1/3 × sync/async × spec0/7、4096真MambaSpec、pending async、abort/alloc失敗、running/waiting queue、原始OFF fixture、V2真execute/sample entrypoints的CPU mock。修補一個既有V2 dispatch fixture缺少的P8 timer=None欄位，未放寬production判斷。GPU數值、吞吐與每step timer開銷皆未驗。
