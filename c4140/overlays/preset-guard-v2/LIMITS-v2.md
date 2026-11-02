# Team preset：gmu guard v2 與單請求上限

源碼基底 `S=/data/bench/s2-impl/c1fix/stage-FIX`，以下 `S/…:line` 均為此固定基底；manifest 釘相關源碼 SHA。本輪只落地 prompt-logprob API guard；其餘為上限建議，未改現役 preset。

| 路徑 | 查到的現行行為／證據 | team preset 建議 |
|---|---|---|
| `prompt_logprobs != None` | `S/vllm/v1/worker/gpu/sample/prompt_logprob.py:24` 判 **is not None**，所以 0 也會計算；`:226–242` 固定 CHUNK_SIZE=1024 呼叫 full-vocab logits。`S/vllm/model_executor/layers/vocab_parallel_embedding.py:495` FP32 logits；`S/vllm/config/vllm.py:2137` dflash2 SM70 defaults。已發生 HTTP500 + engine death 的記錄見 lmhead-s1-v4/READY-v4.md §1。 | **任何模式 gmu≥0.88 一律400**（含nospec/dflash2）；gmu<0.88只log。評分另用低gmu、低併發；這不是低gmu絕不OOM的保證。 |
| `echo` 的隱式 prompt logprob | `S/vllm/entrypoints/openai/completion/protocol.py:273` echo→logprobs；`…/chat_completion/protocol.py:575` echo→top_logprobs（預設0）。CPU 實讀 request.to_sampling_params 對齊。 | gmu≥0.88 的任何模式同等拒絕，含 `echo:true,logprobs:0`、chat 僅 `echo:true`（本包已實作）。不改普通 echo 且 effective prompt_logprobs=None 的 completion。 |
| decode logprobs／top-k 極大值 | `S/vllm/config/model.py:234` max_logprobs 預設20；`S/vllm/sampling_params.py:714–772` 同時限 sample/prompt；-1 解為整個 vocab。CPU：20通過、21與-1拒絕。 | 明釘 `--max-logprobs 20`，不要設 -1／vocab。一般 team 請求預設不回 logprobs；20 是既有品質評分口徑，仍非完整 logits 計算的消除開關。 |
| `n>1` | `S/vllm/sampling_params.py:487–497` 有上限，但 `S/vllm/envs.py:1643` 預設 **16384**。大量輸出可擴增 child requests／狀態／回傳；max_num_seqs 限同時執行，不限一個 API 請求的總 fanout。 | `VLLM_MAX_N_SEQUENCES=1`（現成 hook，CPU n1 PASS／n2 reject）。允許多答案時分成獨立、受配額的請求。不聲稱 n=2 本身已重現 crash。 |
| `best_of` | 此 build 的普通 CompletionRequest／ChatCompletionRequest 無正式 best_of 欄位；`S/vllm/entrypoints/openai/engine/protocol.py:30–60` extra=allow 但忽略；CPU best_of=999 產生 SamplingParams.n=1、無 best_of。BatchChat 有 legacy 欄位，轉一般 request，不是已啟用 best-of 引擎實作。 | 明確拒絕 best_of>1（或整個參數），不要讓客戶把「被忽略」當作執行成功。 |
| beam search / batch prompts | `S/vllm/entrypoints/openai/completion/protocol.py:234` 與 chat `:539` 用 n 作 beam_width；`…/completion/serving.py:194–219` 對多 prompt 分別 generate。這些可繞過「n=1 就只有一個工作」的想像。 | `use_beam_search=false`，每 HTTP 最多1 prompt；停用 team 的 `/v1/chat/completions/batch` 或限制1組。beam 需獨立排程，不只靠 VLLM_MAX_N_SEQUENCES。 |
| 未設 max_tokens／max_completion_tokens | completion 預設16（protocol.py:58），chat 預設None（chat protocol.py:193–199）；`S/vllm/entrypoints/utils.py:175–210` 取 context餘額/請求值/override 的 min，並非真正無限，但 16K prompt 可產生 **245760** tokens。`S/vllm/v1/engine/input_processor.py:316` 亦有餘額 fallback。 | 客戶預設2048，server硬上限6000；現成 `--override-generation-config '{"max_new_tokens":6000}'`。completion serving.py:79–82/chat serving.py:155–158 取此 override，再由 get_max_tokens 的 min 強制限住；CPU override6000已驗。 |
| prompt > max_model_len | renderer params.py:416–440 token_len_check 在渲染階段丟 VLLMValidationError；input_processor.py:400–432 對 >max 及 generate剛好=max再檢查；get_max_tokens 對262145>262144亦CPU拒絕。預期為API驗證錯誤，不進GPU prefill；本輪未做GPU重現。 | team上限 prompt **254000**、output≤6000，且總和≤262144；拒絕超限，不靜默截斷。請求中的 truncate_prompt_tokens 會改語意，不能把截斷後成功當原輸入成功。 |
| HTTP bytes / prompt批次過大 | token上限前需 JSON parse/tokenization；本 guard 同樣先讀 JSON，未新增 HTTP byte cap；`S/vllm/entrypoints/openai/completion/serving.py:219` 多prompt總數沒有本輪查到的team小上限。 | 前端body上限 **16 MiB**（含工具schema），每HTTP一個prompt，避免 tokenizer 前 host RAM/CPU 放大。這是待另行套用的API/gateway規則。 |

根因大小口徑修正：`1024 × 248320 × 4 = 1,017,118,720 bytes = 970 MiB`，每 rank local `[1024,62080]` 是242.5 MiB（不是254 MiB）；full-vocab all_gather 本身已很大。但現碼 `S/vllm/v1/worker/gpu/sample/logprob.py:78–100` 對小top-k使用 fused logsumexp/subset kernel，只有 full-vocab 分支 `:90` 才 `torch.log_softmax(logits.float())`。因此 READY-v4 的「額外完整 log_softmax >1.2GiB／總峰值>2.2GiB」不能直接套到 top20；需 memory trace 才能定實峰。這不影響已觀測的 crash 與本次 API 擋法。動態 chunk／FP16／rank-local top-k 依指示留後。

歷史CPU scan 的實際輸出在 `../preset-guard/evidence/scan.json`（本輪未改其餘上限）；它驗的是現有函式與API contract，不冒稱 GPU 無 OOM。此包沒有套用表中其餘建議的數值限制，避免改動已凍結測試臂。

本輪事故來源：`/data/bench/explore/p-gemm-nospec/out-pgemm-nospec-20261018-r2/FIX/server.log:1593–1639`，nospec gmu .92 的 prompt logits all_gather 需要486MiB，GPU2僅404.75MiB可用並OOM。依Fable/lead 01M57NYS5XTMKKJ19M8TQD7064，拒絕條件只看gmu、與spec模式無關。Decimal精確比較0.88；boot實際engine config優先於env，同一個GMU值傳launcher與API metadata。
