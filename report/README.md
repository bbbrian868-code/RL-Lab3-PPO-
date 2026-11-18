# report/ — Lab 3 報告 LaTeX 原始檔

- `report.tex`：報告本體，需用 **XeLaTeX** 編譯（Overleaf：Menu → Compiler → XeLaTeX）。
  - 開頭的 `\studentid`、`\studentname` 請填入學號與姓名。
  - 字型自動選擇：Overleaf 用 Noto Serif CJK TC，Windows 本機退回微軟正黑體。
- `figures/`：報告用的圖。

## 換成 TensorBoard 截圖
作業要求「Screenshot of Tensorboard training curve」。目前報告裡的曲線是用 matplotlib 從同一份 TensorBoard event 資料畫的；
把截圖用下列檔名放進 `figures/`，重新編譯就會自動換掉（`\figOr` 巨集會優先用截圖）：

| 檔名 | 內容 | TensorBoard 指令 |
|---|---|---|
| `tb_baseline_train.png` | Baseline 的 `Train/Episode Reward` | `.venv\Scripts\tensorboard.exe --logdir runs\Enduro_release\tb` |
| `tb_baseline_eval.png` | Baseline 的 `Evaluate/Episode Reward` | 同上 |
| `tb_lr_decay_train.png` | 所有學習率階段的 `Train/Episode Reward` | `.venv\Scripts\tensorboard.exe --logdir runs`（取消勾選 Breakout_ppo） |
| `tb_lr_decay_eval.png` | 所有學習率階段的 `Evaluate/Episode Reward` | 同上 |
| `test_5games_screenshot.png` | 5 場測試的終端機截圖（可選） | 見下方 |

5 場測試（在 `src/` 下執行，結果與 `figures/test_5games_seed12.png` 相同）：
```
python test.py --exp-name Enduro_lr1e-5_from65M_then_lr1e-6 --ckpt ..\runs\Enduro_lr1e-5_from65M_then_lr1e-6\checkpoints\model_77830590_1533.pth --episodes 5 --seed 12
```
