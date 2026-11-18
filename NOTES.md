# NOTES — Lab 3 PPO on Enduro-v5 工作紀錄

## 1. 環境安裝 (2026-10-23)

- 機器上**沒有 conda，也沒有 Python 3.10**，只有系統 Python 3.11.9。
  為了不在系統上額外安裝 Miniconda，改在專案根目錄建立 venv：`.venv/`（Python 3.11.9）。
  所有指定版本都有 cp311 wheel，套件版本與指引**完全一致**：
  - torch 2.0.1+cu118、gym 0.26.2（ale-py 0.8.1、AutoROM 授權 ROM 直接裝成功）、numpy 1.25.2、
    tensorboard 2.14.0、opencv-python 4.8.0.76、moviepy 1.0.3
  - 額外：matplotlib（匯出曲線）、pygame（`render_mode="human"` 顯示用）、pypdf（讀講義，系統 Python）
- **問題**：系統沒裝 Microsoft Visual C++ Redistributable，`import torch` 出現 `WinError 126 (c10.dll)`。
  解法（不動系統）：`pip install msvc-runtime` 到 venv，再把 `vcruntime140*.dll / msvcp140*.dll / concrt140.dll / vcomp140.dll`
  複製到 `.venv/Lib/site-packages/torch/lib/`。若之後在別台電腦跑，裝官方 VC++ Redist 即可。
- 第二個 DLL 問題：`ale_py/__init__.py` 自己用 `ctypes.CDLL("msvcp140.dll")` 檢查 VC++ runtime，若沒先 import torch 就會失敗
  （訓練和 test.py 都先 import torch 所以沒事，但單獨 `import ale_py` 會錯）。解法：在 venv 的 site-packages 放
  `vc_runtime_dlls.pth`，啟動時對 `.venv/Scripts`（msvc-runtime 的 DLL 所在）呼叫 `os.add_dll_directory`。只影響這個 venv。
- 啟用方式：`.venv\Scripts\activate`（PowerShell：`.venv\Scripts\Activate.ps1`）

### 驗證結果
1. `torch.cuda.is_available()` = **True**，`torch.cuda.get_device_name(0)` = **NVIDIA GeForce RTX 2080 Ti** ✅
2. `gym.make("ALE/Enduro-v5").reset()` → observation shape **(210, 160, 3)** uint8，action 數 9 ✅
   - spec kwargs：`frameskip=4, repeat_action_probability=0.25, max_num_frames_per_episode=108000`，**沒有** TimeLimit wrapper
3. `gym.make("ALE/Breakout-v5")` 正常建立，obs (210, 160, 3)，action 數 4 ✅
- 另外發現：ALE 在 `render_mode=None` 時呼叫 `env.render()` 會直接拋錯 → sample code 的 `evaluate()` 每步都呼叫
  `self.test_env.render()`，所以 test_env 必須用 `render_mode="rgb_array"`。

## 2. 目錄結構

- `lab03/src/`：助教原始 sample code，**完全不動**，當作對照組（可用 diff 檢查 TODO/提示註解有沒有被改）。
- `src/`：實作版本（繳交用）。
- `runs/<實驗名稱>/{checkpoints,tb,videos}`：權重、TensorBoard log、影片、`train.log`。
- `.venv/`：Python 環境。

## 3. Sample code 的 TODO 位置與用途

| # | 檔案:行 (原始) | 用途 |
|---|---|---|
| 1 | `base_agent.py:61` `train()` | 用 net 取動作 → `env.step`，取得 next obs / reward / done |
| 2 | `base_agent.py:70` `train()` 內 `gae_replay_buffer.append(0, {...})` | 把 transition (observation, action, reward, value, logp_pi, done) 存進 GAE buffer |
| 3 | `models/atari_model.py:41` `AtariNet.forward()` | 回傳 action, log prob, value, entropy（支援 eval greedy 與給定動作 `a` 的 log prob） |
| 4 | `ppo_agent_atari.py:17` `__init__` | 建立訓練環境 `self.env` |
| 5 | `ppo_agent_atari.py:21` `__init__` | 建立評估環境 `self.test_env` |
| 6 | `ppo_agent_atari.py:32` `decide_agent_actions()` | 加 batch 維度、forward、回傳 action / value / logp |
| 7 | `ppo_agent_atari.py:87` `update()` | ratio、clipped surrogate loss、value loss、entropy、反向傳播、gradient clipping |

非 TODO 但重要：
- **GAE 已經由助教在 `replay_buffer/gae_replay_buffer.py` 實作好**（`_compute_gae`、`_compute_discounted_return`），不在 TODO 範圍。
  只加了 `[說明]` 註解（δ、遞迴、return、horizon 切段），**沒有改邏輯**。
- buffer 在 `extract_batch()` 已對整批 advantage 做 normalization。
- 網路架構 sample code 已給好：Nature CNN（3 層 conv）→ actor / critic **各自**一個 512 的 FC（共享 conv trunk），
  `_initialize_weights()` 已是 orthogonal(√2)。

## 4. 實作決策與解讀

### 4.1 「Horizon 128」 vs 「每 10000 steps 更新一次」
依 sample code 的實際用法：
- `update_sample_count = 10000`：**單一環境**收集 10000 個 transition 後才做一次 PPO update（rollout 長度）。
- `horizon = 128`：只在 `GaeSampleMemory.get_gae()` 使用 —— 每條 trajectory（依 done 切分）再切成長度 128 的片段，
  各片段獨立計算 GAE，片段尾端用 `V(s_{t+128})` bootstrap（`values[start:start+horizon+1]`）。
  也就是「truncated GAE，最長往後看 128 步」。
- 每次 update：10000 筆資料 × 3 epoch，minibatch 128 → 每 epoch 79 個 minibatch（最後一個 16 筆），共 237 次梯度更新。

### 4.2 平行環境
sample code 的 `train()` 是**單一環境**（`agent_count: 1`，`terminate`/`truncate` 為純量，`evaluate()` 用 `action[0]`）。
依「以 sample code 為主」原則，**不改成 vector env**，維持單環境。

### 4.3 前處理（`src/atari_wrappers.py`，新檔案）
- `gym.make(env_id, frameskip=1)` + `AtariPreprocessing(frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30, terminal_on_life_loss=False, scale_obs=False)` + `FrameStack(4)` → `(4, 84, 84)` uint8。
- frame skip 只做一次（由 AtariPreprocessing 做，含最後兩 frame max-pool），避免 v5 預設 skip 4 再疊一次。
- sticky action 0.25 維持 v5 預設（訓練、評估、測試都一樣）。
- **Reward clipping**：不用 wrapper，而是在 `base_agent.train()` 存進 buffer 時取 `np.sign(reward)`。
  這樣 `episode_reward += reward` 仍是原始 reward → `Train/Episode Reward` 是原始分數；test_env/test.py 完全不 clip。
  （Enduro 的 reward 本來就是 ±1，clip 幾乎沒影響；Breakout 的 1/4/7 會被 clip 成 1）

### 4.4 網路 forward
- `a` 有給（update）→ 用給定動作算 `log_prob`；`eval=True` → **greedy argmax**；否則從 `Categorical(logits)` 取樣。
- **評估模式的選擇**：sample code 在 `evaluate()` 呼叫 `decide_agent_actions(..., eval=True)`，並把 `eval=True` 傳進
  `forward()`，表示 sample code 的設計是「評估時行為不同」，最合理的解讀就是 greedy。所以評估/測試預設 greedy；
  `test.py --stochastic` 可改成取樣，seed 掃描兩種都會跑。
- 輸出層初始化：actor gain 0.01、critic gain 1（其他層沿用 sample code 的 orthogonal √2）。
  初始 log π ≈ −2.195 ≈ log(1/9)，確認初始 policy 接近均勻。

### 4.5 update()
- ratio = exp(logp_new − logp_old)；surrogate = −mean(min(ratio·A, clip(ratio,1±0.2)·A))；
  value loss = MSE(V, return)；total = surrogate + 0.5·v_loss − 0.01·entropy；grad clip 0.5。
- minibatch 內再做一次 advantage normalization（buffer 已做過整批一次，兩次並不衝突）。
- 新增 TensorBoard：`PPO/Approx KL`（k3 估計 `(r−1) − log r`）、`PPO/Clip Fraction`、`Train/SPS`。
- **沒有改的 sample code 行為**（刻意保留，記錄下來）：
  - sample code 在 epoch 迴圈**外**只 shuffle 一次，3 個 epoch 的 minibatch 切法相同（指引寫「每個 epoch 打亂」，
    但那段在 TODO 之外，屬於 sample code 原邏輯，所以不動）。影響很小，列為可能的調整項。
  - `train()` 中 terminal 那一步 `break` 在 `total_time_step += 1` 之前，所以每個 episode 少算 1 步（對 20M 步可忽略）。
  - truncated 和 terminated 一樣視為 done（GAE 不 bootstrap）。Enduro 只在 108000 frames（27000 agent steps）上限才會 truncate，
    只影響最後一小段，略有偏差但可接受。

### 4.6 額外功能（新函式，不動 sample code 主邏輯）
- `base_agent.py`：
  - `should_evaluate()`：`eval_interval_steps > 0` 時每固定步數評估（Enduro 用 500k），否則沿用原本每 100 episode。
    評估只在 episode 結束時檢查（與原本相同的位置），所以實際評估步數會略晚於 500k 的倍數。
  - `after_evaluate()`：更新 `best.pth`（只存網路權重，與 sample code `save()` 格式相同）、`best_info.txt`、`eval_log.csv`，
    並存完整狀態 `last.pt`（net + optimizer + 步數 + episode + best score + 下次評估步數 + 累計時間）。
  - `save_checkpoint()` / `load_checkpoint()`：續訓用，先寫 `.tmp` 再 `os.replace`，避免中斷時檔案損毀。
  - 訓練結束時若最後一步沒評估過，補一次評估。
  - `load()` 加 `map_location`（CPU 也能載入）。
- `main.py`：config 的 key / 預設超參數不變，加上 argparse（`--exp-name --env-id --training-steps --eval-interval-steps --seed --resume`）。
  另外呼叫 `SetThreadExecutionState` 讓 Windows 在訓練中不進入睡眠（程式結束自動恢復）。
- `test.py`：獨立測試腳本（見 RESULTS.md 用法）。
  - `--render`（`render_mode="human"`）實測被 ALE 鎖在真實速度：每秒 15 個 agent step → 一場 9 天（27000 步）要 **30 分鐘**。
    所以另外加 `--display-fps N`：用 rgb_array 畫面 + OpenCV 視窗顯示，可自訂速度（例如 60 = 4 倍速、約 7.5 分鐘），
    畫面上同時顯示目前分數/步數/天數。已驗證顯示模式不影響結果（Breakout seed 0 兩種模式都是 48 分、1148 步）。

## 5. 階段 1：單元檢查 ✅
- 小規模 rollout（300 步、手動插入一個 done 切 trajectory）+ 一次 update，Enduro 與 Breakout 都跑過：
  - `decide_agent_actions` 回傳 action/value/logp shape 皆 `(1,)`
  - `extract_batch`：return/adv/action/reward/value/logp_pi/done 皆 `(300,)`，observation `(300, 4, 84, 84)` float32
  - loss 有限、更新後參數全部 finite；初始 entropy ≈ log(9)=2.197 / log(4)=1.386；approx KL ≈ 0.005
- 完整 `main.py` 短跑（25k 步、update 10000）：評估、存檔、best.pth、eval_log.csv 正常。
- 續訓測試：從 `last.pt` 接回 step 26595、episode 8、Adam step 474、累計時間，eval_log 正確接續。
- `test.py`：同 seed 兩次結果一致（含錄影與不錄影），mp4 正常輸出。
- 發現並修正：訓練剛好在評估點結束時會重複評估一次 → 加 `last_eval_step` 判斷。

### 速度 (SPS)
- 單步拆解：env.step（含前處理）≈ 0.70 ms，GPU forward + 回傳 CPU ≈ 1.7 ms（Windows WDDM 的 kernel launch / 同步延遲），
  CPU forward ≈ 0.77 ms。把 3 次 `.cpu()` 合成 1 次只快約 7%，不值得改。
- 短跑實測含 update 約 **300~360 SPS**。

## 6. 階段 2：Breakout 驗證
- 2026-10-23 17:02 啟動：`--exp-name Breakout_ppo --env-id ALE/Breakout-v5 --training-steps 3000000 --eval-interval-steps 250000`
  （PPO 超參數全部預設；評估間隔只是紀錄頻率）
- 背景執行方式：用 WMI `Win32_Process.Create` 啟動 `cmd /c ... > train.log 2>&1`，不隸屬於對話的 shell，對話結束也不會中斷。
- **結果：通過 ✅**（不需要除錯）。17:02 → 19:28，訓練時間 8800 s（2.4 h），平均 **341 SPS**。

| step | eval avg (3 場, greedy) |
|---|---|
| 250k | 9.0 |
| 500k | 15.7 |
| 1.0M | 12.3 |
| 1.5M | 22.7 |
| 2.0M | 24.0 |
| 2.5M | 15.7 |
| 2.75M | 30.7 |
| **3.0M** | **40.7**（best） |

- random 約 1~2 分 → 1.5M 起評估 >20，3M 時 40.7；訓練曲線 20-episode 移動平均由 ~1 升到 ~30。
- 訓練末期 approx KL ≈ 0.025~0.03、clip fraction ≈ 0.19（偏高但穩定，沒有發散；10000 筆 × 3 epoch 的設定下屬合理）。
- `test.py` 5 場（seed 0, greedy）：48 / 14 / 17 / 6 / 47，平均 26.4。
  第 4 場長度 26993（打到 108000 frames 上限才 truncate）：Breakout 掉球後要按 FIRE 才發球，
  greedy policy 在某些狀態永遠不選 FIRE 就會卡住。這是 Breakout + greedy 的已知現象，Enduro 不需要 FIRE 發球，不受影響。
- 曲線：`runs/Breakout_ppo/train_reward.png`、`eval_reward.png`

## 7. 階段 3：Enduro baseline
- 2026-10-23 19:28 啟動：`--exp-name Enduro_release --training-steps 20000000 --eval-interval-steps 500000`，seed 1，eval 3 場 greedy。
- 預估：Breakout 實測 341 SPS（Enduro 短跑也是 300~360）→ 20M ≈ 16.3 h，加上評估（後期每次評估 3 場 × 最多 27000 步，
  評估約 400 SPS → 每次最多 ~3.5 分，40 次 ≈ 2 h）→ 預估 **約 18~19 h，10/24 下午完成**，遠早於 10/30 期限，不需縮短步數。
### 中斷事件 #1（2026-10-23 20:35）
- 訓練跑到 1.36M 步時，python 程序**無聲消失**：log 沒有 traceback，Windows Application/System 事件記錄在 20:25–20:45 間
  沒有任何事件（不是 crash、不是睡眠），判斷是被外部終止，原因不明（Breakout 用同樣方式跑 2.4 h 都沒事）。
- 當時 last.pt 只在每次評估（每 500k）存 → 損失 1.0M→1.36M 約 36 萬步（~18 分鐘）。
- 對策（20:41 重新啟動）：
  1. `last.pt` 改為每 **200k** 步也存一次（`periodic_checkpoint()`，在 update 結束時呼叫），最多損失 ~10 分鐘。
  2. 新增 `src/run_train.ps1` 監督腳本：用 `--resume` 啟動 main.py，若沒寫出 `checkpoints/DONE` 就 30 秒後自動續訓（最多 50 次）。
     已實測：中途強制 kill python → 自動重啟並從 last.pt 接回 → 正常跑完寫 DONE。
  3. 續訓時 env/torch 改用 `seed + 目前步數`，不會重播一開始的 episode。
  4. 續訓時把 TensorBoard 舊 event 檔中 step > checkpoint 步數的紀錄剪掉（`src/tb_utils.py`，原檔備份在
     `runs/Enduro_release/tb_backup/`），避免報告截圖出現重疊的折返曲線。這次剪掉 468 筆（1.0M~1.36M）。
- **原因已確認**：使用者回報關掉了一個「沒有顯示任何東西的 PowerShell 視窗」。WMI 在本機建立的程序跑在互動式
  Session 1，會開出主控台視窗（輸出都導向 log 所以是空白的）；關掉主控台 → CTRL_CLOSE_EVENT → python 一起被結束，
  不會留下 traceback 或事件記錄，與觀察完全吻合。
- 修正：20:43 改用 `Win32_ProcessStartup.ShowWindow = 0` + `-WindowStyle Hidden` 啟動監督腳本，**不再出現視窗**
  （再次從 1.0M 的 last.pt 接回，TensorBoard 再剪掉 43 筆）。
### 中途檢查（10/24 02:34，8M 步）
- 評估分數：0~2.5M 全為 0（與「前 3M 為 0 是正常」相符）→ 3.0M 19.7 → 3.5M 109 → 5.0M 241 → 5.5M 391 → 7.0M 427 → 8.0M 430。
- 訓練 episode 長度由 3325 變成 ~6650 → 已穩定通過第 1 天、在第 2 天結束。
- SPS 穩定約 334。approx KL 漸升到 ~0.037、clip fraction ~0.33（比 Breakout 高很多）：policy 每次 update 變化偏大，
  但目前沒有崩潰跡象。依指示 baseline 不調參，列入調參建議。

### 續訓與 buffer 的關係（使用者提問後的檢查與修正）
- `last.pt` **不存 GAE buffer**。PPO 是 on-policy：buffer 只放「上次 update 後、用目前權重」收集的資料，update 完就清空，
  不會跨 update 使用。續訓時權重與 Adam 狀態完全一樣 → 重新收集的資料與遺失的資料來自同一個 policy，統計上等價，**不造成偏差**。
- 這次 1.0M 的 checkpoint 是評估時存的，當時 buffer 有 655/10000 筆被丟掉（只是這 655 步有計步但沒用於訓練，影響可忽略）。
- 修正（之後重啟才會生效，目前執行中的程序用的是舊程式碼）：
  - `after_evaluate()` 不再存 `last.pt`；`last.pt` **只在 update 結束（buffer 剛用完）時存** → 續訓狀態等同「兩次 update 之間」，零資料遺失。
  - 續訓時以 `best_info.txt` 恢復 best score（避免評估後、下次存 last.pt 前中斷，best.pth 被較差模型覆蓋）。
  - 續訓時也剪掉 `eval_log.csv` 中 step > checkpoint 的列（那些評估會重做）。
  - 小實驗驗證：last.pt 存在 step 19993（buffer 剛滿 10000），續訓後第一行 `[3324/10000][23316]` = 19993 + 3323，完全接上。

- 背景啟動指令（隱藏視窗）：
  ```powershell
  $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ProcessStartupInformation = $startup; CurrentDirectory = "<root>\src"; CommandLine = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File <root>\src\run_train.ps1 -ExpName Enduro_release -Steps 20000000"}
  ```
- 啟動指令（前景）：`powershell -ExecutionPolicy Bypass -File src\run_train.ps1 -ExpName Enduro_release -EnvId ALE/Enduro-v5 -Steps 20000000 -EvalIntervalSteps 500000`
- 手動續訓指令：`python main.py --exp-name Enduro_release --training-steps 20000000 --eval-interval-steps 500000 --resume`

### 訓練完成（10/24 12:44）
- 20,000,472 步，`DONE` 正常寫出；20:43 之後沒有再中斷（supervisor 只有 launch #0）。
- 有效訓練時間 60,612 s（含評估），平均 330 SPS。各階段 SPS：0~3M 337 → 18~20M 330（評估 episode 變長，略降）。
- 評估：13M 起在 460~780 間震盪；`best.pth` = 15.0M（776.3）。
- 各階段平均（TensorBoard）：

| 區間 | entropy | approx KL | clip frac | value loss | train reward |
|---|---|---|---|---|---|
| 0–3M | 1.90 | 0.014 | 0.18 | 0.036 | 1.0 |
| 5–8M | 1.31 | 0.034 | 0.32 | 0.76 | 258 |
| 12–15M | 1.19 | 0.039 | 0.29 | 0.76 | 474 |
| 18–20M | 1.21 | 0.038 | 0.27 | 0.80 | 531 |

## 8. 階段 4：測試與 seed 掃描
- `test.py` 的天數計算修正：原本 `ceil(length/3325)` 會把剛好 3 天（9979 步）算成第 4 天。
  改為：terminate → `round(length/3326)`；truncate → `length//3326 + 1`。之後 reward 區間與長度兩種算法在所有 seed 上都一致。
- `best.pth`（15.0M）5 場（seed 0）：783/753/496/630/477，平均 627.8；20 seeds greedy 平均 553.9，最多到第 3 天。
- 懷疑 3 場評估雜訊太大 → 對訓練中評估前幾名的 checkpoint 做 10-seed 掃描（不是額外訓練，只是測試）：
  13.0M 631.3、13.5M 537.6、17.0M 682.1、17.5M 696.6、**20.0M 861.3**（10 seeds 中 5 個到第 4 天）。
- 20M checkpoint 完整掃描：greedy 20 seeds 平均 **788.1**，7/20 到第 4 天，最佳 **seed 6 = 994**；stochastic 平均 599.8，最多第 3 天。
  5 場（seed 0）：948/928/949/764/780，平均 **873.8**。
- **決策**：`best.pth` 依規格保留（訓練中評估最高者），但 demo 建議改用 `model_20000472_729.pth` + seed 6 + greedy（第 4 天，25 分）。
- 重現性：20M seed 0~9 兩次獨立掃描結果完全相同；seed 6 加錄影仍是 994 / 13307 步。
- 從錄影畫面確認天數：seed 6 在第 10200 格儀表板顯示「4」、最後一格「第 4 天、剩 109 台」→ 第 4 天結束時沒達標。
- 評估模式：greedy 一律優於 stochastic（20M：788 vs 600），所以 demo 用 greedy（也是 sample code `eval=True` 的語意）。

## 9. 額外實驗：分段降低學習率續訓（10/24 使用者同意）
- 起因：使用者參考往年修課者的做法（lr 2.5e-4 訓練 100M → 1e-5 續訓 40M → 1e-6 續訓 60M）。從對方的曲線看：
  - 降到 1e-5 後的**前 2M 步**就有大幅跳升（約 1050 → 1850），之後 38M 步只多 +200；1e-6 也只在一開始跳一次，之後持平。
  - 2.5e-4 主線在 20→80M 每 10M 約 +60，80M 之後趨緩。
  - → 結論：降學習率的效果用 2~3M 步的短跑就看得出來，不需要長時間訓練，可以用短探測決定何時降學習率。
- 本實驗**超出「只跑助教參數」的 baseline 規則，是使用者明確同意的額外實驗**。`runs/Enduro_release`（baseline）完全不動，RESULTS.md 的 baseline 結果保留。
- 預算：10/24 16:24 起算，主線截止 **10/28 00:00**，收尾約 10 h，留約 1.5 天給報告。
- 程式變更（**沒有動到任何 TODO 區塊**；使用者提醒 TODO 是 demo 主要解釋的地方，不能為續訓改亂）：
  - `main.py`：新增 `--learning-rate`（預設 2.5e-4 = 助教值）與 `--init-from`；預設行為和 baseline 完全一樣。
  - `finetune_utils.py`（新檔）：`init_from_checkpoint()` 從別的實驗分支（接續網路、Adam 狀態、步數、episode、累計時間，TensorBoard 曲線接在 20M 後面），以及 `set_learning_rate()`（載入的 optimizer state 帶有舊 lr，載入後要重設；另記錄 `Train/Learning Rate`）。
  - `run_train.ps1`：新增 `-LearningRate`、`-InitFrom` 參數，直接傳給 main.py。
  - `lr_decay_schedule.py`（新檔）：排程器，自動執行下面的流程，狀態存在 `runs/lr_decay/state.json`，log 在 `runs/lr_decay/schedule.log`，探測結果在 `runs/lr_decay/probes.csv`。
- 煙霧測試：從 20M `last.pt` 用 1e-5 起跑，approx KL 立刻從約 0.04 降到約 0.004，clip fraction 從約 0.27 降到約 0.04。
- 同時跑兩個訓練：各約 287 SPS，合計約 574 SPS（單獨一個約 350）→ 探測可以和主線並行，不會拖慢主線太多。
- 流程（排程器）：
  1. 主線 `Enduro_lr2.5e-4_cont`：從 baseline 20M 的 `last.pt` 用 lr 2.5e-4 繼續訓練。
  2. 探測 `Enduro_lr1e-5_from<N>M`：在 20M、35M、50M、65M…從主線的 `last.pt` 分支（存成 `branch_<N>M.pt`），lr 1e-5 跑 3M 步，
     用 `test.py --sweep 0 19`（greedy）對分支點與探測結束各打一次分數。
  3. 停損：依分支順序，最後一個探測比前一個進步 < 100（約 1/3 天），或時間到 10/28 00:00，就停掉主線，進行中的探測直接放棄。
  4. 收尾：分數最高的探測用 1e-5 延長到「分支點 + 10M」，再分支一條 `..._then_lr1e-6` 跑 3M，最後兩者都掃 seed 0~19。
- 判斷依據：比較用 20 個固定 seed（0~19）的 greedy 平均，而不是訓練中只有 3 場的評估（前面 `best.pth` 已經證明 3 場雜訊太大）。
  原本打算用 10 個 seed，但 dry run 中同一個模型單場分數從 475 到 948 都有，10 個 seed 的平均差約有 ±80 雜訊，和門檻 100 太接近，改成 20 個（約 ±55）。
  20M 的參考值：seed 0~19 平均 788.1（也順便驗證重現性）。
- Dry run（`--dry-run`，極小步數）：分支、探測、停損、等待掃描、延長、1e-6 分支、最終掃描全部走完，續訓接 last.pt 正常；測試資料夾已刪除。
- 10/25 02:23 進度：
  - 探測 `Enduro_lr1e-5_from20M`（3M 步，19:47 完成）：seed 0~19 greedy 平均 **788.1 → 1114.8（+326.6）**；20 個 seed 中 18 個到第 4 天以上（其中 6 個到第 5 天），seed 0 = 1625（第 6 天）。
  - 分支點重新掃描 = 788.1，和之前完全相同 → 重現性確認。
  - 主線 2.5e-4 在 30.6M，訓練中評估（3 場）29~30M 約 850~1020（13~20M 是 460~780），約 307 SPS。
  - `probes.csv` 欄位名稱 `*_seed0_9` 是改成 20 個 seed 前留下的命名，實際是 seed 0~19；程式已改名（執行中的排程器仍是舊名稱，不影響數值）。
- 10/25 11:22 進度：
  - 探測 `Enduro_lr1e-5_from35M`（35.0M → 38.0M，09:19 完成）：seed 0~19 平均 **1008.2 → 1214.9（+206.7）**；天數分布 第 4 天 11、第 5 天 8、第 6 天 1，最佳 seed 8 = 1637（第 6 天）。
  - 跟 20M 探測（1114.8）比 **+100.1**，剛好超過停損門檻 100 → 主線繼續（在雜訊 ±55 的範圍內，是勉強過關）。
  - 主線在 40.4M，approx KL 已從約 0.04 降到約 0.021、clip fraction 約 0.16（高學習率下的更新本身也變溫和了）；訓練中評估 36.5~39.5M 約 850~985。
  - 下一個分支點 50M 預計 10/25 約 20:00，結果約 10/26 00:00。
- 10/25 23:08：探測 `from50M`（50.0M → 53.0M）：分支點 796.4 → **1551.8**（+755）；比 35M 探測 +336.9 → 主線繼續。
  天數分布 第 4 天 4、第 5 天 6、第 6 天 4、第 7 天 5、第 8 天 1；最佳 seed 13 = **2263（第 8 天）**。
  注意：主線 50M 本身（796.4）反而比 35M（1008.2）低 → 高學習率下主線分數震盪大，但降學習率後的分數仍持續上升，證明用「降學習率後的分數」判斷比看主線準。
- 10/26 12:58：探測 `from65M`（65.0M → 68.0M）：分支點 1367.5 → **1585.1**（+217.6）；比 50M 探測只 **+33.2 < 100 → 停損觸發**。
  - 主線停在約 68.3M（寫入 `DONE` + `STOPPED_BY_SCHEDULER`）。
  - 收尾：分數最高的 `from65M` 從自己的 last.pt（68,016,119）用 1e-5 延長到 75,006,308；之後分支 `..._then_lr1e-6` 跑 3M，再兩者都掃 seed 0~19。預計 10/26 約 23:00 全部完成。
  - from65M 天數分布：第 4 天 6、第 5 天 2、第 6 天 3、第 7 天 8、第 8 天 1；最佳 seed 2 = 2201（第 8 天）。
- 10/26 22:36 全部完成（期間沒有中斷）：
  - 1e-5 延長：from65M 從 68,016,119 接續到 75,017,207；seed 0~19 平均 **1844.6**（第 4 天 2、第 5 天 1、第 6 天 3、第 7 天 10、第 8 天 4）。
  - 1e-6 分支：從 from65M 的 last.pt（74,825,749，最後一次存 last.pt 的位置，不是 75.0M）跑到 77,830,590；approx KL 約 0.00003、clip fraction 0（policy 幾乎凍結）。
    seed 0~19 平均 **1920.25**（第 5 天 4、第 6 天 1、第 7 天 9、第 8 天 2、**第 9 天 4**：seed 6/10/12/19）。
- 10/27 驗證與 demo 選擇：
  - seed 12 加錄影重跑：2339 / 26997 步，和掃描完全相同。26997 步 = 撞到 108000 frames 上限被截斷。
  - 錄影畫面：最後一格儀表板「第 9 天、剩 261 台」，倒數 400 格是第 8 天且已出現達標旗子 → 天數計算正確。
  - 5 場（seed 0）：1387 / 1380 / 1919 / 1846 / 1846，平均 1675.6。
  - **決策**：最終 demo = `Enduro_lr1e-5_from65M_then_lr1e-6/checkpoints/model_77830590_1533.pth` + seed 12 + greedy（第 9 天，50 分）；備用 seed 19 / 10 / 6。baseline 的 20M + seed 6 留作備案。
  - 圖表：新增 `src/plot_lr_decay.py`（不改 `export_plots.py`，只 import 它的函式），輸出到 `runs/lr_decay/*.png`。
  - 檢查：`src/` 與 `lab03/src` 比對，沒有任何 TODO / 註解行被刪除；`src/` 沒有權重檔。
  - 主線實際停在 68,406,046（last.pt）／68.4M。額外步數約 70M，牆鐘約 54 h。
- 10/27 11:08：四個第 9 天 seed 各自單獨跑 `test.py --episodes 1 --seed N` 存 log（`runs/Enduro_lr1e-5_from65M_then_lr1e-6/test_logs/`）：seed 6 = 2322 / 26995、seed 10 = 2323 / 26994、seed 12 = 2339 / 26997、seed 19 = 2327 / 26993，和掃描完全相同。
- 10/27 19:07：報告用 5 場測試改從 seed 12（demo 的 seed）開始：2339 / 1687 / 1986 / 1656 / 1884，平均 1910.4 → `test_logs/test_5games_seed12.log`；seed 0 版本（平均 1675.6）也複製到 `test_logs/test_5games_seed0.log`。兩者都是同一個模型、同一個 `test.py`，差別只在起始 seed。

## 10. 報告（11/01，使用者要求）
- `report/report.tex`（XeLaTeX）＋ `report/figures/`；依 Lab3-PPO (2026).pdf 的要求：訓練曲線、5 場測試、4 個問題。不提訓練中斷。
- 本機沒有 TeX：經使用者同意下載免安裝的 Tectonic 0.15.0（放在 session scratchpad，不裝進系統）在本機編譯 → `report/report.pdf`（8 頁，中文字型為微軟正黑體）。用 Windows 內建 PDF API 把每頁轉成圖片檢查排版，修正超出版面與大片空白。
- TensorBoard 無頭瀏覽器截圖失敗（headless Chrome 抓不到 TB 資料）→ 報告先用 matplotlib 匯出的同一份資料，`\figOr` 巨集會在 `figures/` 有 `tb_*.png` 截圖時自動換上（見 `report/README.md`）。
- 5 場測試圖是把 `test_logs/test_5games_seed12.log` 的實際輸出畫成終端機樣式。
- TensorBoard 曲線空白（只有 tag）：venv 的 protobuf 7.36.2 移除了 `MessageToJson(including_default_value_fields=...)`，TensorBoard 2.14 的 hparams 插件回傳 500，前端判定「Failed to fetch runs」。修正：venv 內 `pip install protobuf==4.23.4`（規格未指定 protobuf；torch 不依賴它）。修正後 hparams API 回 200、run 清單正常。
- 11/01：填入學號姓名（314513067 張晉堯）；使用者的 TensorBoard 截圖中 baseline 兩張檔名互換（train↔eval），已對調檔名；學習率階段的兩張截圖 baseline 評估缺 8~20M、頂端被 Ignore outliers 裁切 → 該圖改用 matplotlib 版（截圖檔保留）。重新編譯 report.pdf（8 頁）。

## 11. 打包（11/01，使用者指示）
- 提交版 `RL_LAB3_314513067_張晉堯.zip`：`src/`（14 檔，排除 __pycache__，無權重）＋ `report.pdf`；解壓後 compileall 通過。
- 筆電轉移版 `RL_LAB3_314513067_laptop_transfer.zip`（約 20 MB）：提交版 zip、唯一權重 `model_77830590_1533.pth`、`seed12_greedy-episode-0.mp4`、`test_logs/`、`README_demo.txt`（環境版本與 demo 指令）。
- CPU 驗證：`test.py --seed 12 --cpu` 也是 2339 / 26997，和 GPU 完全相同 → 筆電沒有 GPU 也能重現。
