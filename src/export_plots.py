import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# [說明] 從 TensorBoard event 檔匯出 Train/Episode Reward 與 Evaluate/Episode Reward 曲線 (PNG 備份)
# [說明] 用法：python export_plots.py --exp-name Enduro_release  -> runs/<exp>/train_reward.png, eval_reward.png

RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"


def load_scalars(tb_dir, tag):
	# [說明] 續訓會產生多個 event 檔，全部讀進來後依 step 排序、去除重複 step
	points = {}
	for path in sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))):
		acc = EventAccumulator(path, size_guidance={"scalars": 0})
		acc.Reload()
		if tag in acc.Tags()["scalars"]:
			for e in acc.Scalars(tag):
				points[e.step] = e.value
	steps = np.array(sorted(points))
	return steps, np.array([points[s] for s in steps])


def style(ax, title, ylabel):
	ax.set_facecolor(SURFACE)
	ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
	ax.set_xlabel("Environment steps", color=INK_2)
	ax.set_ylabel(ylabel, color=INK_2)
	ax.grid(True, color=GRID, linewidth=0.8)
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(GRID)
	ax.tick_params(colors=INK_2)
	ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 or x == 0 else f"{x / 1e3:g}k"))


def save(fig, path):
	fig.patch.set_facecolor(SURFACE)
	fig.tight_layout()
	fig.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(fig)
	print(f"saved {path}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp-name", type=str, default="Enduro_release")
	parser.add_argument("--smooth", type=int, default=20, help="moving-average window (episodes)")
	args = parser.parse_args()
	exp_dir = os.path.join(RUNS_DIR, args.exp_name)
	tb_dir = os.path.join(exp_dir, "tb")

	steps, values = load_scalars(tb_dir, "Train/Episode Reward")
	if len(steps):
		fig, ax = plt.subplots(figsize=(9, 4.5))
		ax.plot(steps, values, color=BLUE_LIGHT, linewidth=1, label="per episode")
		if len(values) >= args.smooth:
			kernel = np.ones(args.smooth) / args.smooth
			smoothed = np.convolve(values, kernel, mode="valid")
			ax.plot(steps[args.smooth - 1:], smoothed, color=BLUE, linewidth=2, label=f"moving average ({args.smooth} episodes)")
		ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
		style(ax, f"{args.exp_name} - Train/Episode Reward", "Episode reward (raw)")
		save(fig, os.path.join(exp_dir, "train_reward.png"))

	steps, values = load_scalars(tb_dir, "Evaluate/Episode Reward")
	if len(steps):
		fig, ax = plt.subplots(figsize=(9, 4.5))
		ax.plot(steps, values, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
		ax.margins(y=0.15)
		best = int(np.argmax(values))
		ax.annotate(f"best {values[best]:.0f} @ {steps[best] / 1e6:.2f}M", (steps[best], values[best]),
					textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=9)
		style(ax, f"{args.exp_name} - Evaluate/Episode Reward", "Avg reward over eval episodes (raw)")
		save(fig, os.path.join(exp_dir, "eval_reward.png"))


if __name__ == "__main__":
	main()
