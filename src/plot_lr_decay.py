import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from export_plots import RUNS_DIR, INK, INK_2, GRID, SURFACE, load_scalars, style, save

# [說明] 額外實驗「分段降低學習率續訓」的圖 (報告用 PNG)，輸出到 runs/lr_decay/：
# [說明]   lr_decay_train_reward.png：各學習率階段的 Train/Episode Reward (移動平均)
# [說明]   lr_decay_eval_reward.png ：各學習率階段的 Evaluate/Episode Reward (訓練中 3 場平均)
# [說明]   lr_decay_probe_scores.png：各分支點「降學習率前 / 後」的 seed 0~19 平均 (停損判斷依據)
# [說明] 用法：python plot_lr_decay.py

OUT_DIR = os.path.join(RUNS_DIR, "lr_decay")
MAIN = ["Enduro_release", "Enduro_lr2.5e-4_cont"]
PROBES = ["Enduro_lr1e-5_from20M", "Enduro_lr1e-5_from35M", "Enduro_lr1e-5_from50M", "Enduro_lr1e-5_from65M"]
FINAL = "Enduro_lr1e-5_from65M_then_lr1e-6"
# [說明] 學習率 -> 顏色 (固定對應，不依排名)
C_HIGH, C_MID, C_LOW = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#898781"
SMOOTH = 20


def tb(exp):
	return os.path.join(RUNS_DIR, exp, "tb")


def chain(exps, tag):
	steps, values = [], []
	for exp in exps:
		s, v = load_scalars(tb(exp), tag)
		steps.append(s)
		values.append(v)
	return np.concatenate(steps), np.concatenate(values)


def moving_average(steps, values, k=SMOOTH):
	if len(values) < k:
		return steps, values
	return steps[k - 1:], np.convolve(values, np.ones(k) / k, mode="valid")


def day_lines(ax, top):
	# [說明] 天數分界 (助教評分表：第 2 天 200、第 3 天 500，之後每 300 一天)，在右側標示
	for day, r in enumerate([200, 500, 800, 1100, 1400, 1700, 2000, 2300], start=2):
		if r <= top:
			ax.axhline(r, color=GRID, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
			ax.annotate(f"day {day}", (1, r), xycoords=("axes fraction", "data"), xytext=(4, 0),
						textcoords="offset points", va="center", fontsize=8, color=MUTED)


def plot_curves(tag, ylabel, fname, smooth):
	fig, ax = plt.subplots(figsize=(10, 5))
	s, v = chain(MAIN, tag)
	if smooth:
		s, v = moving_average(s, v)
	ax.plot(s, v, color=C_HIGH, linewidth=2, label="lr 2.5e-4 (baseline 0-20M + continued)")
	for i, exp in enumerate(PROBES):
		s, v = load_scalars(tb(exp), tag)
		if smooth:
			s, v = moving_average(s, v)
		ax.plot(s, v, color=C_MID, linewidth=2, label="lr 1e-5 (probes; 65M probe extended to 75M)" if i == 0 else None,
				marker=None if smooth else "o", markersize=3)
	s, v = load_scalars(tb(FINAL), tag)
	if smooth:
		s, v = moving_average(s, v)
	ax.plot(s, v, color=C_LOW, linewidth=2, label="lr 1e-6 (from 74.8M)", marker=None if smooth else "o", markersize=3)
	ax.set_ylim(bottom=0)
	day_lines(ax, ax.get_ylim()[1])
	ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
	style(ax, tag + (f"  (moving average, {SMOOTH} episodes)" if smooth else "  (avg of 3 eval episodes)"), ylabel)
	save(fig, os.path.join(OUT_DIR, fname))


def sweep_mean(exp, ckpt_name):
	path = os.path.join(RUNS_DIR, exp, f"seed_sweep_{ckpt_name}_greedy.csv")
	with open(path) as f:
		return float(np.mean([float(r["reward"]) for r in csv.DictReader(f)]))


def final_sweep(exp):
	path = max(glob.glob(os.path.join(RUNS_DIR, exp, "seed_sweep_model_*_greedy.csv")),
			   key=lambda p: int(os.path.basename(p).split("_")[3]))
	step = int(os.path.basename(path).split("_")[3])
	with open(path) as f:
		return step, float(np.mean([float(r["reward"]) for r in csv.DictReader(f)]))


def plot_probe_scores():
	with open(os.path.join(OUT_DIR, "probes.csv")) as f:
		rows = list(csv.DictReader(f))
	branch = np.array([float(r["branch_ckpt"].split("_")[1][:-4]) * 1e6 for r in rows])
	pre = np.array([float(r[[k for k in r if k.startswith("pre_mean")][0]]) for r in rows])
	post = np.array([float(r[[k for k in r if k.startswith("post_mean")][0]]) for r in rows])
	post_x = branch + 3e6

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(branch, pre, color=C_HIGH, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
			markeredgewidth=2, label="lr 2.5e-4 main line at branch point")
	for x0, y0, x1, y1 in zip(branch, pre, post_x, post):
		ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
	ax.plot(post_x, post, color=C_MID, linewidth=0, marker="o", markersize=8, markeredgecolor=SURFACE,
			markeredgewidth=2, label="after 3M steps at lr 1e-5")
	for x, y in zip(post_x, post):
		ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
	s1, m1 = final_sweep(PROBES[-1])
	s2, m2 = final_sweep(FINAL)
	ax.plot([s1], [m1], color=C_MID, linewidth=0, marker="D", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
			label="65M probe extended to 75M (lr 1e-5)")
	ax.plot([s2], [m2], color=C_LOW, linewidth=0, marker="D", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
			label="then 3M steps at lr 1e-6")
	for x, y in ((s1, m1), (s2, m2)):
		ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
	ax.set_ylim(0, 2400)
	ax.set_xlim(15e6, 82e6)
	day_lines(ax, 2400)
	ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
	style(ax, "Greedy test score, mean of seeds 0-19", "Mean episode reward (raw)")
	save(fig, os.path.join(OUT_DIR, "lr_decay_probe_scores.png"))


if __name__ == "__main__":
	plot_curves("Train/Episode Reward", "Episode reward (raw)", "lr_decay_train_reward.png", smooth=True)
	plot_curves("Evaluate/Episode Reward", "Avg reward over eval episodes (raw)", "lr_decay_eval_reward.png", smooth=False)
	plot_probe_scores()
