# S1-v4-A5 — prototype 05，固定 shared 位址與 causal interior fastpath

這是舊 `s1-v4-A` CPU_STOP 封包外的新候選。kernel source **逐byte等於 prototype-05-source.cuh** (`3b2c0bf8f4dcaac8e41babd05734549853d5b94bd52536e4bb3e1980aeb09767`)；不是第八版的 phase tid 重算。BASE / A+A1B / S1-v3-OV / 舊CPU_STOP皆不動。只離線CPU編譯，GPU須由lead排窗。

## 判準與 deviation

Fable `01M58KDGJ9SDYEE96QX7YEYE5C` 明確調整舊 **STACK0** 閘：目的為熱迴圈無spill，允許序曲／尾聲local。因此本版採 **REG≤128、STACK≤16 B、KV hot-loop及outlined helpers的LDL/STL=0**。第五版已知STACK8 B，全核LDL1/STL2保留並揭露，不稱全核零spill；也不把NCU的全核local sectors要求為零。

正式Stage-1不變：B8/L253953 eager partial **≤2.86 ms/層**，native finite、vs BASE maxabs≤2e-3，partial+combine DRAM **≤1.2×同窗BASE**，加上述資源閘。L2僅報告、不設95%閘。Stage-2≤1.60 ms僅印距離。

本版預期 **≤2.45 ms/層**（相對chain53 OV 2.88394約−15%）仍是未驗假說。比較同窗BASE、OV、A5三臂，所有shape與每輪中位數完整列出。`event_increment`對OV的≥15%、逐輪方向與spread為歸因診斷，不額外改Stage-1。stall百分比不當wall省時；primary為未profile Events/eager partial。

## PC資源證據

`inspect_native.py`以cuobjdump離線抽取實際新.so；`resource_gate.py`找含HMMA的最長backward BRA，自loop入口到該backedge作inclusive PC範圍，另保守掃描main terminal EXIT後**所有**outlined helper指令。第五版：主迴圈 **0x25f0–0x9140**，0x9140 `@P0 BRA 0x25f0`；0x9150起為epilogue，main terminal EXIT=0x9ce0。helper區0x9cf0後亦零LDL/STL。

三個局部記憶體指令全部在迴圈外：序曲 `0x1120 STL [R1],R52`；尾聲 `0x94b0 STL [R1],R8`、`0x9bc0 LDL.LU R5,[R1]`。NCU local ld/st sectors仍逐臂列出，資訊欄名明寫whole-kernel，不能據此反推hot loop spill。`inspection.json`同時保存全核opcodes、instruction SHA與精確PC區間。argon獨立讀SASS的核對收據另附；未齊不算GPU_READY。

## 實作與數值

沿用OV：512T/grid=(1,80,B)、N64、48rows、D256、80splits、不跨request，shared=97024 B；production NVFP4頁4096/每token288B、ABI/原combiner不變。

只改 `kernel/s1_verify.cuh`、新增 `kernel/s1_v4_address.cuh`。兩個packed relative-coordinate uint32在loop前計算；QK/PV各在相位入口以opaque register identity解包、加shared base。QK K16使用XOR低位+固定立即偏移；PV P M16固定2048B、V row8 XOR64；消除重複generic位址與swizzle重算。與第五版完全一致，不額外加S2R或軟體改序。

Full causal fastpath只在 `valid==64 && t+63<=max(0,length-8)` 使用；末尾/causal保持原式，softmax數學與MMA K順序、raw producer、alias壽命不改。每tile三個full CTA barrier加六QKwarps一個named192；initial/epilogue各一個full barrier。這次不再砍同步。

CPU覆蓋：actual helper C++（僅PTX LDS換host byte-read）逐槽20480 calls/786432 uint32 operands；33278976 causal/tail組合與三個錯位址/unsafe-Full突變；四shape B4/B8×L131065/253953及13 edges數值；原碼producer/epilogue/softmax恆等；新資源閘測STACK16/17、hotloop及outlined helper local負例、序曲尾聲local正例。CPU BLAS/FP16模型不模擬HMMA/MUFU，不冒稱native parity。GPU第一步CUDA_LAUNCH_BLOCKING=1 production writer fixture：BASE finite與B1=batch必成立，再驗候選finite/error，才計時。

## 窗與完整呼叫

單GPU0、總cap **30min**（執行27min+清理3min）。五shape另含B8/32K；≥20warm、三輪×20 graph Events、20次eager partial/combine；第二輪反向交錯。NCU每臂只B8/L253953 partial+combine兩kernel，單次≤285s。沿用STOP/Xid/ECC/空卡/清cache規則；輸出在封包外。

```sh
CUDA_VISIBLE_DEVICES='' bash /data/bench/explore/d1/s1-v4-A5/run.sh
bash /data/bench/explore/d1/s1-v4-A5/run.sh --execute --out /data/bench/explore/d1/out-s1-v4-A5-20261019
```

NCU沿既有METRICS（DRAM/L2/tensor/occupancy/XU/LSU/stall/shared actual-ideal/local/opcode），加已離線確認的 `smsp__issue_active.avg.pct_of_peak_sustained_active` / `smsp__inst_issued.sum`。CPU只讀重核：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 python3 /data/bench/explore/d1/s1-v4-A5/test_cpu.py
```

封版後不在此目錄重跑會寫receipt的build/address/numeric/inspection腳本；需重跑先複本到/tmp並釘回原native路徑。microbench過後仍需獨立serving驗證，不在本窗部署。
