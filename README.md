# RL Lab 3：PPO on Enduro-v5

用 Proximal Policy Optimization (PPO) 玩 Atari `ALE/Enduro-v5`。

## 課程資訊

| | |
|---|---|
| 課程 | Selected Topics in Reinforcement Learning（Fall 2026） |
| 開課單位 | 國立陽明交通大學（NYCU）CGI Lab |
| 作業 | Lab 3：Proximal Policy Optimization (PPO) |

## 結果

| 模型 | 步數 | seed 0~19 平均分數（greedy） | 最遠天數 |
|---|---|---|---|
| Baseline（助教超參數，lr 2.5e-4） | 20M | 788.1 | 第 4 天 |
| 分段降低學習率（2.5e-4 → 1e-5 → 1e-6） | 77.8M | **1920.3** | **第 9 天** |

最好的一場：seed 12 拿到 **2339 分**，跑到第 9 天，直到碰上 108000 frames 的時間上限。

完整數據見 [RESULTS.md](RESULTS.md)，開發紀錄見 [NOTES.md](NOTES.md)，報告在 [report/report.pdf](report/report.pdf)。

## 目錄

```
src/          實作（以助教 sample code 為基礎，完成所有 TODO）
  ppo_agent_atari.py        建立環境、選動作、PPO update（clipped surrogate + value loss + entropy）
  models/atari_model.py     Actor-critic CNN 的 forward
  base_agent.py             收集 rollout、存進 GAE buffer、評估與存檔
  replay_buffer/            GAE buffer（助教提供）
  atari_wrappers.py         前處理：灰階、84x84、frame skip 4、frame stack 4
  test.py                   測試、seed 掃描、錄影、即時顯示
  finetune_utils.py, lr_decay_schedule.py   分段降低學習率的實驗
lab03/src/    助教原始 sample code（對照用，沒有修改）
report/       報告（LaTeX 原始檔、PDF、圖）
runs/         訓練 log、評估紀錄、seed 掃描結果、曲線圖（不含權重與 TensorBoard 檔）
```

## 環境

Python 3.11、torch 2.0.1、gym[atari] 0.26.2、ale-py 0.8.1、numpy 1.25.2、opencv-python 4.8.0.76、moviepy 1.0.3、tensorboard 2.14.0（protobuf 4.23.4）

## 使用方式（在 `src/` 底下執行）

```bash
# 訓練（超參數用助教的預設值）
python main.py --exp-name Enduro_release --env-id ALE/Enduro-v5 --training-steps 20000000 --eval-interval-steps 500000

# 測試：1 場、seed 12、greedy，開視窗顯示
python test.py --exp-name <exp> --ckpt <path/to/model.pth> --episodes 1 --seed 12 --display-fps 60

# 5 場測試 / seed 掃描
python test.py --exp-name <exp> --ckpt <path/to/model.pth> --episodes 5 --seed 12
python test.py --exp-name <exp> --ckpt <path/to/model.pth> --sweep 0 19
```

模型權重沒有放進 repo。
