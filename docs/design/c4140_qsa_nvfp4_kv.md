# C4140 plan 071：QSA main K/V 的 NVFP4 KV cache（階段 1，僅 CPU）

基底 `c4140-p070`（`1b5731f90`）。階段 1 做參考實作、dtype 准入、cache spec 與 allocator 推導；
之後依序接上 global scale（步驟 A）、Triton store 與呼叫點契約（步驟 B），並記錄編譯路徑與 reader 計畫（步驟 C）。
還沒有 attention reader，所以 `--kv-cache-dtype nvfp4` 在 `_run_qsa` 仍明確拒絕執行。

## 已定決策（Fable）

- **版面採選項 A**：放大 block，讓 NVFP4 頁剛好放得下 GDN state 頁。不改 allocator、platform、worker。
  選項 C 留作團隊並發的後續：把兩個 NVFP4 頁打包進一個實體頁，不開 MTP 時 block 為 1392，重用
  DCP2 已有的 packed 頁機制，但要改 platform 的 block 規則。選項 B（state 獨立頁池）不做。
- **MTP 順序**：不開 MTP 的 A；接著 draft 也用 NVFP4 的 A；最後才做 target NVFP4 加 draft E4M3 的混合版面。
  混合版面延後，不是被排除：「混合版面會打破 allocator」這個前提來自 Astra 的基底 `25361ff5f`，
  在 p070 不成立。p070 的 allocator 有逐 owner 頁清單與 packed 頁（PR #823），dry-run 顯示該配置被接受；
  剩下的阻礙只是 worker 與 platform 層的單一全域 cache dtype。
- **dtype 名稱**：沿用既有的 `nvfp4`，不另取新名。

## dtype 名稱與 SM100 約定（SM70 參考必須與之一致）

名稱 `nvfp4`：`config/cache.py:34`（`CacheDType`）、`kv_cache_interface.py:65-66`（`get_kv_quant_mode`）、
頁大小 `:186-204`、`:299-319`、儲存 dtype `utils/torch_utils.py:98`（uint8）、列位元組
`:464-466`（`head//2 + head//16`）、SM100 分派 `csrc/libtorch_stable/cache_kernels.cu:840-857`。

SM100/SM120 的 store（`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`、`quantization/fp4/nvfp4_utils.cuh`）：

| 項目 | 約定 | 位置 |
|---|---|---|
| 頁版面 | `[K_data \| K_scale \| V_data \| V_scale]`，每側先資料後 scale | `nvfp4_kv_cache_kernels.cu:8`、`:218-245` |
| 打包 | 16 值打成 8 byte；偶數索引在低 nibble | `:139-145`；`nvfp4_utils.cuh:118-153`（`cvt.rn.satfinite.e2m1x2 b0, %3, %2`，第一個來源進高 nibble） |
| scale 位元組 | `sf = (1/k_scale) × (amax × rcp(6))`，E4M3 飽和 RNE；`out = rcp(sf_q × rcp(1/k_scale))` | `nvfp4_kv_cache_kernels.cu:95`；`nvfp4_utils.cuh:240-271` |
| K 與 V 的 scale 排列 | K 線性，V 為 SM100 的 4×4 swizzle。V100 兩者都用線性 | `nvfp4_kv_cache_kernels.cu:161-171`、`:33-39` |
| 位元級對拍 | 參考實作用同樣的運算順序，除法取代 `rcp.approx.ftz`，差異只在 fp32 中間值貼著捨入邊界時 | 見測試 |

## global scale 的決定

**決定**：每層 K、V 各一個 fp32 純量，語意是反量化乘數，`dequant = e2m1 × block × global`。
checkpoint 有就照用，沒有就是 1.0。這與 fleet 的 SGLang 實作和 SM100 store 一致，沒有語意衝突，
所以不採用「由校準 amax 推導」的替代方案。上游測試的 `amax/448` 只是測試取值，不是約定。

| 面向 | SGLang（`cookys/sglang-qsa-nvfp4-kv` @1118e72） | 1Cat 本分支 |
|---|---|---|
| 反量化 | 乘 global：`patches/qsa_nvfp4_kv.py:173-174`、`:257-262` | `nvfp4_kv.py:353-375`，`block = e4m3_decode(scale) × layer_scale` |
| 取得 | `layer_global_scale`（`:120-135`）以全域 layer id 索引，越界回 1.0；取用在 `:188-189` | `layer._k_scale`／`_v_scale`，由 `qsa.py:801-840` 從 checkpoint 槽位複製 |
| 缺值 | 寫入端存 1.0（`docs/mechanism.md:62-71`），驗收就在 1.0 下跑 | 保留預設 1.0；`model.py:168`、`:211` 對 nvfp4 只記 info，不拒絕 |
| store 端 | `1/global` 當 SM100 的 `SFScaleVal` | 相同：`scale_inverse = 1.0 / layer_scale` |

E4M3 沒有校準就拒絕啟動，NVFP4 不拒絕：block scale 已逐 16 值吸收動態範圍，
測試顯示 `amax/2688`、`amax/448`、`amax/200` 與 1.0 的 relative L2 都在 8.5% 到 10.5%。
E4M3 overlay 的 24 個純量為 K 0.0186 到 0.0367、V 0.0171 到 0.0841，它們的倍數是 448 還是 200 不在 repo 與 provenance 內，[未驗證]。

守衛：載入值必須有限、為正，且不超過 `NVFP4_KV_LAYER_SCALE_MAX = 10`（`nvfp4_kv.py:78-84`、`:253-269`，
錯誤訊息帶層名）。group 的 block scale 是 `amax16 / (6 × layer_scale)`，低於 2^-6 就成為 E4M3 次正規數、失去尾數位；
amax 至少 1 的層在 layer_scale 超過 10.67 時會發生。overlay 的 K／V amax 都大於 3，所以 10 留有餘裕，
又能擋掉為別種格式準備的 scale。`nvfp4_layer_scale_from_amax`（`amax/2688`）與 `nvfp4_block_scale_range`
留作需要推導時使用，目前沒有呼叫端。測試在 `test_nvfp4_kv_scale.py`，含從 SGLang 測試移植的案例。

**SM100 編譯守衛**：store 內唯一 SM100 以上才有的內建指令是 `cvt.rn.satfinite.e2m1x2.f32`
（`nvfp4_utils.cuh:70-153`，無條件的 inline asm，沒有 `__CUDA_ARCH__` 守衛，測試已釘住）。其餘是一般運算：
half2 的 abs 與 max、`__nv_fp8_e4m3(float)`、`rcp.approx.ftz.f32`、`__shfl_xor_sync`、整數存取。
`CMakeLists.txt:1123-1127`、`:1157-1161` 只對 10.0a／10.1a／10.3a／12.0a／12.1a 加入此檔，
`cache_kernels.cu:841-856` 其餘架構報錯。若要編 sm_70：把 `fp32_vec8_to_e2m1`／`fp32_vec16_to_e2m1` 換成軟體
捨入（`e2m1_encode` 即規格）、排除 bf16 實例化（bf16 的 half2 運算需 sm_80 以上，[未驗證]）、加 CMake 架構項與分派巨集。
這對階段 1 不需要：Triton store 用軟體 e2m1 捨入，不需要 nvcc；CUDA store 只有在 reader 存在之後才有用。沒有建置任何東西。

## 寫入路徑（store）：契約與 SM100 分支對照

呼叫點在 `Qwen4ExpQSAAttention._run_qsa`：`impl.do_kv_cache_update(self, key, value, self.kv_cache, main_metadata.slot_mapping)`，
prefill、decode、draft 都走這裡。`Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update`（`qsa.py:290-329`）對 nvfp4 轉給
`store_nvfp4_kv_triton`，其他 dtype 仍是 `reshape_and_cache_flash`。`_run_qsa` 更早就對 nvfp4 拋 `NotImplementedError`，
所以真實路徑仍是關的，store 目前只被測試呼叫。

| 參數 | 契約 |
|---|---|
| `key`、`value` | `[rows, num_kv_heads, head_dim]`，fp16（fp32、bf16 也可）。只要求最後一維連續；`value` 是 QKV 輸出的切片，token 與 head stride 照傳。`rows` 可以大於 slot 數（CUDA graph padding），多出的列不寫 |
| `slot_mapping` | int64 `[num_actual_tokens]`，`block × block_size + offset`。負值跳過（padding，或 DCP 下別的 rank 擁有的 token）；大於等於 `num_blocks × block_size` 也跳過 |
| `kv_cache` | 5 維 uint8 `(num_blocks, 2, block_size, heads, 144)`，worker 建立的 view。packed member view 的 block stride 較大，照 stride 寫，不碰鄰居 |
| `k_scale`、`v_scale` | `layer._k_scale`／`_v_scale`，0 維 fp32 device 純量，內核內讀取，所以 CUDA graph 可擷取 |
| 重複 slot | 同一次呼叫內重複的 slot，內核結果未定義，與 SM100 相同，scheduler 不會產生；參考 writer 規定後寫者勝 |

SM100 的 NVFP4 分支（`cache_kernels.cu:840-857` 分派，`nvfp4_kv_cache_kernels.cu` 內核與檢查）做什麼，以及 SM70 store 是否對齊：

| SM100 行為 | 位置 | SM70 Triton store |
|---|---|---|
| token 數取 `slot_mapping.size(0)`，key 多餘的列不碰 | `cache_kernels.cu:825-833`；`nvfp4_kv_cache_kernels.cu:191` | 相同 |
| `slot < 0` 跳過，沒有上界檢查 | `:77` | 相同，另加上界；對合法輸入沒有差別 |
| 讀 `*k_scale_ptr`、`*v_scale_ptr` 為 device 純量，用 `1/scale` | `:95` | 相同 |
| 一個 CTA 一個 token，K 與 V 在同一個內核 | `:256` | 一個 program 一個 token 與 head，K、V 各啟動一次；每層每次 forward 多一次小啟動，成本沒量 |
| 只用 `key.stride(0)`，假設 head stride 等於 head_size | `:103-104`、`:272` | 取實際 head stride，較寬鬆 |
| 只收 fp16、bf16 | `:263` | fp16、bf16、fp32 |
| `block_size % 4 == 0`，因為 V 的 scale 有 4×4 swizzle | `:210` | V100 為線性，沒有這個要求；2784、2864、1392 本來也都是 4 的倍數 |
| V 的 scale 做 swizzle，K 線性 | `:161-171` | 兩者都線性 |
| 以 stride 判斷 HND 或 NHD，scale 區緊接在資料區後 | `:216`、`:230` | 由 `nvfp4_kv_split_views` 給 stride，版面相同 |
| e2m1 轉換用 `cvt.rn.satfinite.e2m1x2` | `nvfp4_utils.cuh:118-153` | 軟體比較；tie 規則對 torch 規格窮舉驗證，與 PTX 的 tie 行為[未驗證] |

## 推導公式與結果

`Platform._align_hybrid_block_size`（`vllm/platforms/interface.py:672-708`）：

```text
attn_block_size = align × cdiv(state_page, align × attn_page_size_1_token)
align           = max(kernel block 16, cache_config.block_size)
attn_page_size_1_token = 2 × heads × (head//2 + head//16) × 1 B   # NVFP4：288；E4M3：512
```

state 頁是 `MambaSpec.real_page_size_bytes`，TP4、fp16 conv、fp32 SSM：GDN 801,792 B（K=0）、
822,272 B（MTP4）；PLE conv 184,320／266,240 B，較小。

| | 每 token 每層 | K=0 的 block | K=4 的 block |
|---|---:|---:|---:|
| E4M3（現況） | 512 | 1568 | 1616 |
| **NVFP4** | **288** | **2784** | **2864** |

- 288 B/token 推得**整數**。K=0：`2784×288 = 801,792`，頁與 state 相等，**零 padding**。
  K=4：`2864×288 = 824,832`，每頁 pad 2,560 B（0.31%）。不需退回補 padding 的做法。
- 推導與 allocator 沒有寫死 1616 之類的假設；用真正的 `_align_hybrid_block_size`（QSA backend）與真正的
  `get_kv_cache_groups` 在 CPU 上驗證。
- 代價 [未實測]：align 模式的 prefill chunk 在 `max_num_batched_tokens=8192` 下由 7840／8080 縮到 5568／5728；
  prefix 重用粒度變粗。明確傳 `--block-size` 只會放大。

## 容量：兩個 KV 預算

真 allocator 在 CPU 上算的 `GPU KV cache size`。p069 預算是 292 個 ID（3.2907 GiB/rank）。
p070 預算由實測錨點反推：E4M3 加 MTP4、B=1616 的 263,538 tokens 剛好是 189 個 ID，
所以預算在 2.1299 到 2.1412 GiB 之間；下表用下界，上界的結果只差 1 到 3 個 ID。
「262K 等價」是 `IDs / (ceil(262144/B)+F)`；每人 ctx 是 N 人同時 active 時的 allocator 模型預算，不是 SLA。

| 配置 | B | p069：tokens／262K 等價／N=16 每人 ctx | p070：tokens／262K 等價／N=16 每人 ctx |
|---|---:|---|---|
| MTP4 E4M3 | 1616 | 407,159／1.553／0 | 263,538／1.005／0 |
| MTP4 A | 2864 | 602,707／2.299／0 | 389,855／1.487／0 |
| MTP4 C1（NVFP4 打包加 E4M3 draft） | 1616 | 579,007／2.209／0 | 374,127／1.427／0 |
| 不開 MTP，E4M3 | 1568 | 482,818／1.842／17,248 | 312,499／1.192／6,272 |
| 不開 MTP，A | 2784 | 756,184／2.885／25,056 | 488,999／1.865／8,352 |
| 不開 MTP，C | 1392 | 771,011／2.941／30,624 | 498,587／1.902／12,528 |

p070 的 N=8 每人 ctx：MTP4 三種皆為 0；不開 MTP 為 26,656（E4M3）、41,760（A）、45,936（C）。
不開 MTP 時沒有計入回收 draft 權重的預算；Astra 的 1.247 GiB 是 p069 的估計，p070 的權重不同，未重估。
p070 上 MTP4 在 N≥8 時 GDN state 稅就吃光預算，4-bit KV 解決不了。

p070 欄反映目前 p070 的 pool，錨點是 263,538 tokens。這個 pool 比 p069 少，已識別的原因是上游 `b75d3cd0f`
把五個 batch pack 開關的預設由 0 改為 1，可逐項核算的額外副本約 1.23 GiB 每 rank
（llm-playground repo 的 `doc/survey/2026-10-03-c4140-p069-astra.md` §I，`:3003-3012`、`:3068-3072`）。
Fable 在 GPU 上驗證五個開關全關後，預期 pool 回到約 411K 到 414K tokens；這是預期值，不是量測。
本欄會在該實測 pool 出來後重算，這裡不自行估計。

## 已交付（皆 CPU）

| 項目 | 位置 |
|---|---|
| 純 torch 參考：量化、反量化、頁版面、writer、含非法索引 zero-fill 的 sparse gather，運算順序同 SM100 store | `vllm/models/qwen4_exp/nvidia/ops/nvfp4_kv.py` |
| dtype 准入（backend、impl、owner）、`get_kv_cache_shape`（末維 144）、執行 guard | `vllm/models/qwen4_exp/nvidia/qsa.py` |
| global scale 接線：載入、次正規數守衛、`nvfp4` 不要求校準 | `qsa.py`、`model.py`、`nvfp4_kv.py` |
| Triton fused unpack+scale gather（直譯器逐位元等於參考） | `nvidia/ops/nvfp4_kv_triton.py` |
| Triton store（量化、打包、寫 scale）與 `do_kv_cache_update` 契約（直譯器逐位元組等於參考 writer） | `nvfp4_kv_triton.py`、`qsa.py` |
| sm_70 編譯閘門：不需 GPU，檢查 PTX 與暫存器 | `tests/models/qwen4_exp/test_nvfp4_kv_sm70_compile.py` |

放在 `nvidia/ops/` 與 `qsa.py`、`qsa_kv_calibration.py` 並列，後續 kernel 從這裡取格式常數；
`nvfp4_kv_cache_full_dim` 與 `nvfp4_kv_cache_split_views` 留在 upstream 的 `utils/torch_utils.py`，只匯入不修改。
寫入端不產生負零 nibble（0x8）。非法 top-k 索引（負值、超出 block table、未映射或越界 block、無效 request）
在兩個區都 zero-fill，且不讀取 cache。

逐元素誤差（8192×256，fp16 輸入，global scale 1.0）：

| 輸入 | NVFP4 max abs／mean abs／relative L2 | E4M3 逐張量 relative L2 |
|---|---|---:|
| 高斯 | 0.619／0.0715／9.5% | 2.65% |
| Student-t(3) 重尾（max 161） | 6.14／0.112／9.1% | 2.64% |

## 測試

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD uv run --no-project --python /data/venvs/1cat-p070/bin/python \
  --with pytest -- python -m pytest --noconftest tests/models/qwen4_exp/test_nvfp4_kv_{reference,admission,scale,triton,store,sm70_compile}.py -q
275 passed, 51 skipped   # 參考 123、准入與 spec 與 allocator 65、scale 66、編譯閘門 21；Triton gather 19 與 store 32 預設略過
TRITON_INTERPRET=1 同上   # 305 passed, 21 skipped；編譯閘門需要真的編譯器，直譯器下略過
```

- 參考測試含「SM100 約定釘選」：讀 `csrc` 原始碼字串，斷言頁版面、`1/k_scale`、scale 運算順序、
  e2m1 運算元順序與「store 內只有 e2m1 一種 cvt、沒有 `__CUDA_ARCH__`」；上游改了約定就會失敗。
  另有「運算順序決定位元組」的邊界案例（layer scale 7 時 30 個 fp16 amax 會分岔）。
  上游自己的 SM100 測試容差很鬆（`test_cache.py:383-387`，atol 1.5、rtol 0.5），且對 K 也套用 V 的 swizzle 解碼
  （`:358-374`），而 kernel 寫 K 為線性（`nvfp4_kv_cache_kernels.cu:161-164`）；這些測試釘不住位元，[未驗證其影響]。
- store 測試（32 個）逐位元組比對參考 writer：隨機、邊界、飽和、次正規、9 個數量級的動態範圍、padding 與越界 slot、
  多餘列、strided value、fp32 輸入、三種 head size、packed member view、`do_kv_cache_update` 契約、store 接 gather、
  輸入驗證，以及兩個內核內編碼器對 torch 在每個 tie 與每個 fp16 值上的窮舉。
  突變檢查：換 nibble 順序 17 個失敗、拿掉 tie 規則 13 個失敗、拿掉資料寫入的遮罩使行程崩潰、
  改成 `amax/6` 的運算順序 2 個失敗、輸出 scale 用錯 3 個失敗。
- 編譯閘門（21 個）：Triton 3.6.0 內附的 ptxas 對 sm_70 編 store 與 gather，斷言目標 sm_70、PTX 無 bf16 或 fp8 轉換、
  無 e2m1、wgmma、`cp.async`、新架構 mma，store 無 atomic，暫存器不超過 64 且不溢出到 local memory。
  把 store 的 block scale 改成原生 fp8 轉換會讓 13 個失敗。它只證明能編譯、資源小，不證明能跑或跑多快。
- worktree 內 `vllm/*.so` 與 `_version.py` 是指向 `/data/venvs/1cat-p070` 的符號連結（已被 git 忽略，沒有建置），
  讓 worktree 的原始碼能載入 `vllm._C`；沒有修改任何 venv。
- 沒有可見 GPU 時，匯入 `qsa.py` 會因 `flash_attn.py` 向 GPU 詢問 FlashAttention 版本而失敗；
  相關測試檔在 `torch.cuda.is_available()` 為假時用 FA2 的答案取代（Volta 本來就是 FA2）。
- `test_config.py` 的 4 個失敗在 `c4140-p070` 本來就失敗，與本案無關。
- 提交使用 `--no-verify`：共用的 `.git/hooks` 被裝上 pre-commit，首次執行要下載全部工具環境而卡住。
  改為直接跑 ruff 0.14.0 的 `check` 與 `format`、typos、markdownlint-cli2 0.21.0，以及 `tools/pre_commit` 內的
  spdx、forbidden-imports、torch-cuda、boolean-context-manager、lazy-imports、env-registration 腳本。mypy 未跑。

## 編譯路徑與 reader（步驟 C，只有計畫，沒有 reader 程式碼）

### sm_70 編譯實測

不需要 GPU：Triton 3.6.0 內附的 ptxas 把內核編到 sm_70，再用 `cuobjdump --dump-resource-usage` 讀資源。
E4M3 那三列的 constexpr 是照內核簽名手動填的（TOPK 2051、PAGE_SIZE 1616、BLOCK_M 8、GROUP_SIZE 6、HEAD_DIM 256），不是從實際啟動擷取。

| 內核 | warps | 暫存器 | stack | shared（Triton） |
|---|---|---:|---:|---:|
| NVFP4 store，head 256，fp16 輸入 | 1／2／4 | 40／28／24 | 8／8／0 B | 128／128／256 B |
| NVFP4 store，head 256，bf16 輸入 | 1 | 38 | 8 B | 128 B |
| NVFP4 gather，head 256 | 1／2／4 | 64／32／30 | 0 | 0 |
| 現行 E4M3 split-K，BLOCK_N 16 | 4 | 241 | 0 | 24,576 B |
| 現行 E4M3 split-K，BLOCK_N 16 | 2 | 255 | 288 B | 24,576 B |
| 現行 E4M3 split-K，BLOCK_N 64 | 2 | 32 | 9,232 B | 73,728 B |

store 與 gather 都不需要 shared memory 的調校，一個 warp 就夠；它們不是瓶頸候選。
reader 才是：E4M3 split-K 在最好的 profile 已用 241 個暫存器，2 個 warp 就溢出。

### decode 的融合 reader 需要什麼

型態直接仿 `ops/qsa.py:617-828` 的 `_qsa_sparse_paged_gqa_splitk_kernel`：
索引 → block table → masked load 原始位元組 → 軟體解碼 → `tl.dot` → online softmax，再用 `_qsa_merge_splitk_kernel` 合併。
加一個 `KV_NVFP4` constexpr 分支，不是新內核。

- **tile 與 warps**：沿用 `_qsa_sparse_launch_profile`（`ops/qsa.py:2522-2556`）的 pre-Ampere 設定，BLOCK_N 16、4 warps。
  該處註解的理由：D=256 的 64 欄 tile 超過 Turing 的 64 KiB shared，2 個 warp 在 V100 上讓 tensor core 工作序列化。
  上表也顯示 BLOCK_N 64 的 stack 是 9,232 B。NVFP4 每個元素的解碼工作比 E4M3 多，所以不要從 2 個 warp 起步。
- **sm_70 沒有的東西**：沒有原生 fp8 或 fp4 轉換，所以用整數位元解碼，E4M3 的 scale byte 重用
  `deepseek_v4/common/ops/fp8_software.py` 的 `fp8_e4m3fn_bits_to_fp32`，e2m1 用位元組裝；
  沒有 bf16 tensor core，所以只做 fp16 query（`qsa_e4m3_capability_reason`，`ops/qsa.py:2262-2270`，V100 本來就跑 fp16）；
  沒有 `cp.async`、wgmma 與 m16n8 mma，這些由編譯閘門守住。
- **解碼在 fp16 是精確的**：e2m1（0、0.5、1、1.5、2、3、4、6）乘 E4M3 block scale，乘積最多 5 位有效數字，
  最大 2688，最小非零 2^-10（fp16 的正規數），對全部 8×127 個組合在 CPU 上驗證為精確。
  所以 reader 可以先解碼成 fp16 而不乘 global scale，K 的 global scale 乘在分數上，V 的 global scale 乘在正規化後的輸出上，
  和 E4M3 路徑同位置（`ops/qsa.py:686-688`、`:776-778`），符合 Astra §E 對 scale 位置的限制。
- **不做 interleave**：K tile 讀成低 nibble 與高 nibble 兩塊 `[128, BLOCK_N]`，`scores = dot(q_even, k_even) + dot(q_odd, k_odd)`，
  query 用 stride 2 的指標載入；V 產生偶數與奇數兩半累加器，最後用 stride 2 的指標寫回。
  block scale 索引對兩塊都是 `j // 8`。累加器總大小不變，但 QK 變成兩段 fp32 部分和，
  捨入順序與 K=256 的單一 dot 不同，所以與 gather 加 FP16 路徑比對的目標是 fp16 捨入內，不是逐位元。
- **遮罩**：與 E4M3 相同，無效的 request、token、page 載入 0 並貢獻 0 機率；NVFP4 的無效 lane 要同時把 nibble 與 scale byte 載成 0。
  page 偏移用 int64，與 E4M3 相同。
- **定址**：資料區與 scale 區在頁內不同位置，用 `nvfp4_kv_split_views` 給的 stride，與 gather 內核一致。

### prefill 計畫

QSA 沒有稠密 prefill。prefill 與 decode 都是對 `[rows, 2051]` 個選中 token 做稀疏注意力，差別只有列數。
分派在 `ops/qsa.py:1733-1762`：列數不少於 `_SM70_QSA_XQA_PAGE4_MIN_ROWS`，或 uint8 cache 且列數超過 16，走 XQA page4 CUDA；
再來是 grouped page4；其餘走 Triton split-K。目前 decode 的 target M5 與 draft M1 都在 Triton split-K
（Astra §E 已確認該內核是 gather、解碼、注意力融合在一起）。

NVFP4 現在被三層擋住，reader 上線時要逐層放行：`_xqa_page4_shape_supported` 要求 cache 形狀為 `(…, 1, 256)`，
NVFP4 的末維是 144 所以被擋；`qsa_sparse_paged_attention` 對 dtype 字串 `nvfp4` 拋 `ValueError`（`ops/qsa.py:2321-2323`）；
`forward_qsa` 要求非 E4M3 的 cache dtype 等於 query dtype（`qsa.py:370-376`），uint8 會拋 `RuntimeError`。再加上 `_run_qsa` 的 guard。

| 選項 | 做法 | 成本與風險 |
|---|---|---|
| a | 同一個 Triton NVFP4 split-K 服務所有列數 | 改動最小，先得到端到端正確性。與 E4M3 的 CUDA page4 prefill 誰快沒有資料，`#441` 的 1.16 到 2.6 倍是 Triton 對舊 Triton profile，不是對 CUDA |
| b | 先解碼成 fp16 scratch，再用現成的 FP16 page4 CUDA 路由 | gather 內核已逐位元正確。scratch 每個選中 token 1,024 B（K 與 V 各 fp16 512 B），而儲存是 288 B。依列逐一展開不可行：16 列 33.6 MB、64 列 134 MB、2048 列 4.3 GB。要依 page 去重：不重複 token 最多 `min(rows×2051, context+rows)`，64K context 約 64 MiB，262K 約 256 MiB，對 p070 每 rank 約 2.1 GiB 的 KV 預算不是小數。XQA page4 已有壓縮 page 索引（`_qsa_xqa_page4_block_table`），可借其表重映射 |
| c | 在 `flash-attention-v100/kernel/flash_decode_paged.cu` 加 NVFP4 的 CUDA reader | 仿 E4M3 分支（`KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3`，`:809-909`；`fp8_kv_utils.cuh`；grouped page4 ABI v2 已帶 `kv_cache_dtype` 與 scale）。需要建置 flash_attn_v100 擴充，不能在純 CPU 階段做 |

建議：先 a（decode 與 prefill 一起），量 prefill 對 E4M3 CUDA page4 的差距；只有 a 大幅落後且列數大於 16 的流量佔多數時才做 b；
團隊並發（約 20 人、256K context）若 prefill 成為瓶頸才投資 c。

### 工作量估計

以下是估計，不是量測。

| 工作 | 估計 | 需要 GPU |
|---|---|---|
| split-K 與 merge 內核加 NVFP4 分支，直譯器對 `gather_dequant_nvfp4_kv` 加 FP16 參考注意力做 parity | 2 到 3 天 | 否 |
| 放行三層守衛與 `_run_qsa`、`forward_qsa` 的 dtype 分支、page4 排除 | 0.5 天 | 否 |
| V100 上編譯與資源調校（暫存器、溢出、BLOCK_N、warps、偶奇兩段 dot 的精度） | 1 到 2 天 | 是 |
| store 到 reader 的往返、`kv-logprob-probe`（含 bf16 重跑組）、ABBA 序列化 A/B；只有勝出者才跑 SWE-34 | 2 到 3 天 | 是 |
| 里程碑 2：draft 也用 NVFP4 | 約 1 天 | 是 |

合計單人約 1.5 到 2 週，其中約 1 週要占 GPU。

## 品質驗證：SGLang 的證據不是 1Cat 的上界

fleet SGLang 的品質數字，nvfp4 對 bf16 KV 的 ΔNLL 為 +0.0042 到 +0.0096 nats/token
（llm-playground repo 的 `engines/patches/sglang-qsa-nvfp4-kv/README.md:327-332`），**不能當 1Cat 的上界**。

SGLang 的第一個 4096-token prefill chunk 內，位置從不讀 pooled KV，所以 2K 視窗在所有 KV dtype 間逐位元相同，
較長視窗的平均 ΔNLL 也被這段零誤差的位置稀釋（同檔 `:345-346`；`notes/methodology/kv-cache-quality-probe.md` 規則三）。
1Cat 的 QSA 在同一個 forward 內先寫後讀：`_run_qsa` 先呼叫 `do_kv_cache_update`（`qsa.py:943`），
再呼叫 `forward_qsa`（`qsa.py:950`），讀的就是剛量化寫入的 cache。所以量化誤差從第一個 chunk 就存在。

`scripts/eval/kv-logprob-probe.py` 的計畫因此要這樣訂：

1. **第一個 prefill chunk 的視窗是必備的**，打分位置全落在第一個 chunk 內。保留 2K 視窗：它們在 SGLang 上沒有訊號，
   在 1Cat 上卻是第一個 chunk 的直接量測。chunk 邊界取 1Cat 的 scheduler chunk 大小，從 serve 設定讀，不沿用 SGLang 的 4096。
2. 第一個 chunk 的視窗、≥32K 視窗與 96K 桶分開報告，不併成一個 pooled 數字。
3. 每個比較都含未量化 KV 的重跑組，先看重跑組的 CI，小於它的差距無法解析（規則一）。1Cat 的每個 server 行程
   會落在兩種數值狀態之一（`notes/methodology/serving-determinism-floor.md`），所以每一組啟動後先過 12 案
   `--no-logprobs` parity，確認落在同一個狀態，否則重跑組量到的是啟動狀態差，不是 KV 精度。
4. 閾值沿用工具的提案規則：ΔNLL 95% CI 上界不超過 +0.005 nats 才算通過。
5. [未驗證] 工具的說明提到 `/flush_cache` 與 `logprob_start_len`，是 SGLang 的介面；能否直接對 1Cat vLLM 抓 logprob 沒有查，
   可能要用 vLLM 的 prompt logprobs 做 adapter。

## 未完成與開放問題

1. 沒有 attention reader。`_run_qsa` 拒絕執行，page4、ops 入口、`forward_qsa` 另有三層擋下，reader 上線時要逐層放行並補明確的 `kv_cache_dtype` 守衛。
2. Triton store 與 gather 只在直譯器跑過；sm_70 只證明能編譯、資源小，沒有在 V100 執行，沒有速度。
   store 的捨入平手無法與 SM100 的 PTX 對拍。
3. 融合 reader 的 QK 是兩段部分和，與 gather 加 FP16 的路徑不會逐位元相同；可接受的差距要用 `kv-logprob-probe` 對重跑組定，不能用 SWE-34。
4. draft 的 scale 載入對 nvfp4 只由 `model.py` 的旗標放行，沒有 draft 實測，里程碑 2 要驗。
5. 首個 prefill chunk 也讀量化 KV（QSA 先寫後讀），SGLang 的 ΔNLL 證據不能當上界；驗證計畫見上一節。
6. packed 頁在 DCP1 下與 `KVBlockZeroer`、offload 的互動沒有驗證；測試只涵蓋 member view 的寫入不碰鄰居。
7. store 的 K、V 各一次啟動的實際成本，以及 `rows` 大時的 grid 形狀，沒有量。
