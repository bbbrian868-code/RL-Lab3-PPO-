import csv
import ctypes
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import torch

# [說明] 額外實驗「分段降低學習率續訓」的排程器 (與助教 TODO 無關，只負責啟動 / 監看 main.py 與 test.py)
# [說明] 流程：
# [說明]   1. 主線：從 baseline 20M 的 last.pt 用 lr 2.5e-4 繼續訓練
# [說明]   2. 探測：在 20M、35M、50M ... 從主線分支，用 lr 1e-5 跑 3M 步，再用 test.py 掃 seed 0~9 (greedy) 打分數
# [說明]   3. 停損：某次探測分數比上一次進步不到 STOP_GAIN，或時間超過 MAIN_DEADLINE，就停掉主線
# [說明]   4. 收尾：分數最高的探測用 lr 1e-5 延長到分支點 +10M，再用 lr 1e-6 跑 3M，最後掃 seed 0~19
# [說明] 狀態存在 runs/lr_decay/state.json，排程器中斷後重新執行會接續
# [說明] 用法 (背景執行)：.venv\Scripts\python.exe src\lr_decay_schedule.py

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
RUNS_DIR = os.path.join(ROOT, "runs")
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SCHED_DIR = os.path.join(RUNS_DIR, "lr_decay")
STATE_PATH = os.path.join(SCHED_DIR, "state.json")
LOG_PATH = os.path.join(SCHED_DIR, "schedule.log")
RESULT_CSV = os.path.join(SCHED_DIR, "probes.csv")

BASELINE_CKPT = os.path.join(RUNS_DIR, "Enduro_release", "checkpoints", "last.pt")
MAIN_EXP = "Enduro_lr2.5e-4_cont"
MAIN_LR = "2.5e-4"
MAIN_STEPS_CAP = 200_000_000
FIRST_BRANCH = 20_000_000
BRANCH_EVERY = 15_000_000
PROBE_LR = "1e-5"
PROBE_STEPS = 3_000_000
FINISH_STEPS = 10_000_000       # [說明] 收尾的 1e-5 階段：分支點 + 10M
FINAL_LR = "1e-6"
FINAL_STEPS = 3_000_000
STOP_GAIN = 100.0               # [說明] 探測分數進步小於此值 (約 1/3 天) 就停掉主線
MAIN_DEADLINE = datetime.datetime(2026, 10, 28, 0, 0)
POLL_SECONDS = 120
# [說明] 探測用 20 個 seed 打分數：單場分數差異很大，10 個 seed 的平均差約有 ±80 的雜訊，和停損門檻 100 太接近
SWEEP_LAST, FINAL_SWEEP_LAST = 19, 19
CKPT_INTERVAL = 200_000
EXP_PREFIX = ""
CREATE_NO_WINDOW = 0x08000000

# [說明] --dry-run：用極小的步數把整個流程 (分支、探測、停損、收尾、掃 seed) 走一遍，檢查排程器本身沒有錯
if "--dry-run" in sys.argv:
	SCHED_DIR = os.path.join(RUNS_DIR, "_dry_lr_decay")
	STATE_PATH = os.path.join(SCHED_DIR, "state.json")
	LOG_PATH = os.path.join(SCHED_DIR, "schedule.log")
	RESULT_CSV = os.path.join(SCHED_DIR, "probes.csv")
	EXP_PREFIX = "_dry_"
	MAIN_EXP = EXP_PREFIX + MAIN_EXP
	MAIN_STEPS_CAP = FIRST_BRANCH + 60_472
	BRANCH_EVERY, PROBE_STEPS, FINISH_STEPS, FINAL_STEPS = 30_000, 20_000, 40_000, 20_000
	POLL_SECONDS = 15
	SWEEP_LAST, FINAL_SWEEP_LAST = 0, 1
	CKPT_INTERVAL = 10_000


def log(msg):
	line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
	print(line, flush=True)
	with open(LOG_PATH, "a", encoding="utf-8") as f:
		f.write(line + "\n")


def load_state():
	if os.path.exists(STATE_PATH):
		with open(STATE_PATH) as f:
			return json.load(f)
	return {"phase": "search", "main": None, "probes": [], "next_branch": FIRST_BRANCH, "finish": {}}


def save_state(state):
	tmp = STATE_PATH + ".tmp"
	with open(tmp, "w") as f:
		json.dump(state, f, indent=2)
	os.replace(tmp, STATE_PATH)


# [說明] Windows 上用 OpenProcess / GetExitCodeProcess 檢查程序是否還在 (os.kill(pid, 0) 在 Windows 會直接結束程序，不能用)
def pid_alive(pid):
	if not pid:
		return False
	handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
	if not handle:
		return False
	code = ctypes.c_ulong()
	ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
	ctypes.windll.kernel32.CloseHandle(handle)
	return code.value == 259  # STILL_ACTIVE


def ckpt_dir(exp):
	return os.path.join(RUNS_DIR, exp, "checkpoints")


def is_done(exp):
	return os.path.exists(os.path.join(ckpt_dir(exp), "DONE"))


# [說明] 透過 run_train.ps1 (會自動從 last.pt 續訓) 在背景啟動一個訓練實驗，不開視窗
def launch_training(exp, steps, lr, init_from=""):
	cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File",
		   os.path.join(SRC_DIR, "run_train.ps1"), "-ExpName", exp, "-Steps", str(int(steps)), "-LearningRate", lr,
		   "-CheckpointIntervalSteps", str(CKPT_INTERVAL)]
	if init_from:
		cmd += ["-InitFrom", init_from]
	p = subprocess.Popen(cmd, cwd=SRC_DIR, creationflags=CREATE_NO_WINDOW)
	log(f"launch training {exp}: steps={int(steps)} lr={lr} init={init_from or '-'} pid={p.pid}")
	return p.pid


def stop_training(exp, pid):
	# [說明] 先寫 DONE 讓 run_train.ps1 不再重啟，再結束整個程序樹
	open(os.path.join(ckpt_dir(exp), "DONE"), "w").close()
	open(os.path.join(ckpt_dir(exp), "STOPPED_BY_SCHEDULER"), "w").close()
	if pid_alive(pid):
		subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)
	log(f"stopped {exp} (pid {pid})")


# [說明] 用 test.py 掃 seed，結果寫到 runs/<exp>/seed_sweep_<ckpt名>_greedy.csv
def launch_sweep(exp, ckpt, first, last):
	out = os.path.join(SCHED_DIR, f"sweep_{exp}_{os.path.splitext(os.path.basename(ckpt))[0]}.log")
	cmd = [PYTHON, "-u", "test.py", "--exp-name", exp, "--ckpt", ckpt, "--sweep", str(first), str(last)]
	with open(out, "w") as f:
		p = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=f, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
	log(f"launch sweep {exp} {os.path.basename(ckpt)} seeds {first}-{last} pid={p.pid}")
	return p.pid


def sweep_csv(exp, ckpt):
	return os.path.join(RUNS_DIR, exp, f"seed_sweep_{os.path.splitext(os.path.basename(ckpt))[0]}_greedy.csv")


def sweep_mean(exp, ckpt):
	path = sweep_csv(exp, ckpt)
	if not os.path.exists(path):
		return None
	with open(path) as f:
		rewards = [float(r["reward"]) for r in csv.DictReader(f)]
	return sum(rewards) / len(rewards) if rewards else None


# [說明] 實驗中步數最大的 model_<step>_<score>.pth (訓練結束時一定會存一個)
def final_model(exp):
	files = glob.glob(os.path.join(ckpt_dir(exp), "model_*.pth"))
	return max(files, key=lambda p: int(os.path.basename(p).split("_")[1])) if files else None


def main_step():
	path = os.path.join(ckpt_dir(MAIN_EXP), "eval_log.csv")
	if not os.path.exists(path):
		return 0
	with open(path) as f:
		rows = [line.split(",") for line in f if line.strip()]
	return int(rows[-1][0]) if rows else 0


# [說明] 主線過了分支點後，複製當下的 last.pt 當作分支起點 (last.pt 只在 update 結束時寫入，buffer 是空的)
def snapshot_branch(target):
	src = os.path.join(ckpt_dir(MAIN_EXP), "last.pt")
	if not os.path.exists(src):
		return None
	dst = os.path.join(ckpt_dir(MAIN_EXP), f"branch_{target / 1_000_000:g}M.pt")
	tmp = dst + ".tmp"
	shutil.copyfile(src, tmp)
	step = torch.load(tmp, map_location="cpu")["total_time_step"]
	if step < target:
		os.remove(tmp)
		return None
	os.replace(tmp, dst)
	log(f"snapshot {os.path.basename(dst)} at step {step}")
	return dst


def write_results(state):
	with open(RESULT_CSV, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["probe_exp", "branch_ckpt", "pre_mean", "post_ckpt", "post_mean", "gain"])
		for p in state["probes"]:
			pre, post = p.get("pre_mean"), p.get("post_mean")
			gain = f"{post - pre:.1f}" if pre is not None and post is not None else ""
			w.writerow([p["exp"], os.path.basename(p["branch_ckpt"]), f"{pre:.1f}" if pre is not None else "",
						os.path.basename(p.get("post_ckpt") or ""), f"{post:.1f}" if post is not None else "", gain])


# [說明] 推進一個探測：分支 -> 訓練 3M -> 掃 seed (分支點與探測結束各掃一次)
def step_probe(p):
	if p["status"] == "training":
		if not pid_alive(p.get("pid")) and not is_done(p["exp"]):
			p["pid"] = launch_training(p["exp"], p["end_step"], PROBE_LR, p["branch_ckpt"])
		elif is_done(p["exp"]):
			p["post_ckpt"] = final_model(p["exp"])
			p["post_pid"] = launch_sweep(p["exp"], p["post_ckpt"], 0, SWEEP_LAST)
			p["status"] = "sweeping"
	if "pre_pid" not in p:
		p["pre_pid"] = launch_sweep(MAIN_EXP, p["branch_ckpt"], 0, SWEEP_LAST)
	if p["status"] == "sweeping":
		if not pid_alive(p.get("post_pid")):
			p["post_mean"] = sweep_mean(p["exp"], p["post_ckpt"])
			if p["post_mean"] is None:
				p["post_pid"] = launch_sweep(p["exp"], p["post_ckpt"], 0, SWEEP_LAST)
			else:
				p["status"] = "done"
				log(f"probe {p['exp']} done: post mean {p['post_mean']:.1f}")
	if p.get("pre_mean") is None and not pid_alive(p.get("pre_pid")):
		p["pre_mean"] = sweep_mean(MAIN_EXP, p["branch_ckpt"])
		if p["pre_mean"] is None:
			p["pre_pid"] = launch_sweep(MAIN_EXP, p["branch_ckpt"], 0, SWEEP_LAST)


def tick(state):
	now = datetime.datetime.now()

	if state["phase"] == "search":
		main = state["main"]
		if main is None:
			os.makedirs(ckpt_dir(MAIN_EXP), exist_ok=True)
			branch20 = os.path.join(ckpt_dir(MAIN_EXP), "branch_20M.pt")
			if not os.path.exists(branch20):
				shutil.copyfile(BASELINE_CKPT, branch20)
			state["main"] = main = {"pid": launch_training(MAIN_EXP, MAIN_STEPS_CAP, MAIN_LR, branch20)}
		elif not pid_alive(main["pid"]) and not is_done(MAIN_EXP):
			main["pid"] = launch_training(MAIN_EXP, MAIN_STEPS_CAP, MAIN_LR, os.path.join(ckpt_dir(MAIN_EXP), "branch_20M.pt"))

		# [說明] 新的分支點：20M 直接用 baseline 的 checkpoint，之後等主線跑過分支點再取 last.pt
		target = state["next_branch"]
		if target == FIRST_BRANCH:
			branch = os.path.join(ckpt_dir(MAIN_EXP), "branch_20M.pt")
		elif main_step() >= target:
			branch = snapshot_branch(target)
		else:
			branch = None
		if branch:
			step = torch.load(branch, map_location="cpu")["total_time_step"]
			exp = f"{EXP_PREFIX}Enduro_lr{PROBE_LR}_from{target / 1_000_000:g}M"
			state["probes"].append({"exp": exp, "branch_ckpt": branch, "branch_step": step,
									"end_step": step + PROBE_STEPS, "status": "training",
									"pid": launch_training(exp, step + PROBE_STEPS, PROBE_LR, branch)})
			state["next_branch"] = target + BRANCH_EVERY

	for p in state["probes"]:
		if p["status"] != "done" or p.get("pre_mean") is None:
			step_probe(p)
	write_results(state)

	if state["phase"] == "search":
		# [說明] 依分支順序取「已完成的前幾個探測」，比較最後兩個的分數
		done = []
		for p in state["probes"]:
			if p["status"] != "done":
				break
			done.append(p)
		stop_reason = None
		if len(done) >= 2 and done[-1]["post_mean"] - done[-2]["post_mean"] < STOP_GAIN:
			stop_reason = f"gain {done[-1]['post_mean'] - done[-2]['post_mean']:.1f} < {STOP_GAIN}"
		elif now >= MAIN_DEADLINE:
			stop_reason = f"deadline {MAIN_DEADLINE}"
		if stop_reason:
			log(f"stop main line: {stop_reason}")
			stop_training(MAIN_EXP, state["main"]["pid"])
			# [說明] 進行中的探測直接放棄，讓收尾階段獨占 GPU
			for p in state["probes"]:
				if p["status"] == "training":
					stop_training(p["exp"], p["pid"])
					p["status"] = "abandoned"
			state["phase"] = "finish_wait"

	if state["phase"] == "finish_wait":
		if any(p["status"] == "sweeping" for p in state["probes"]):
			return
		scored = [p for p in state["probes"] if p.get("post_mean") is not None]
		best = max(scored, key=lambda p: p["post_mean"])
		log(f"best probe {best['exp']} (mean {best['post_mean']:.1f}); extend lr {PROBE_LR} to branch +{FINISH_STEPS}")
		os.remove(os.path.join(ckpt_dir(best["exp"]), "DONE"))
		state["finish"] = {"exp1": best["exp"], "end1": best["branch_step"] + FINISH_STEPS,
						   "pid1": launch_training(best["exp"], best["branch_step"] + FINISH_STEPS, PROBE_LR)}
		state["phase"] = "finish_1e-5"

	f = state["finish"]
	if state["phase"] == "finish_1e-5":
		if is_done(f["exp1"]):
			f["exp2"] = f"{f['exp1']}_then_lr{FINAL_LR}"
			init = os.path.join(ckpt_dir(f["exp1"]), "last.pt")
			step = torch.load(init, map_location="cpu")["total_time_step"]
			f["end2"] = step + FINAL_STEPS
			f["pid2"] = launch_training(f["exp2"], f["end2"], FINAL_LR, init)
			state["phase"] = "finish_1e-6"
		elif not pid_alive(f["pid1"]):
			f["pid1"] = launch_training(f["exp1"], f["end1"], PROBE_LR)

	if state["phase"] == "finish_1e-6":
		if is_done(f["exp2"]):
			f["final1"], f["final2"] = final_model(f["exp1"]), final_model(f["exp2"])
			f["sweep1"] = launch_sweep(f["exp1"], f["final1"], 0, FINAL_SWEEP_LAST)
			f["sweep2"] = launch_sweep(f["exp2"], f["final2"], 0, FINAL_SWEEP_LAST)
			state["phase"] = "final_sweep"
		elif not pid_alive(f["pid2"]):
			f["pid2"] = launch_training(f["exp2"], f["end2"], FINAL_LR, os.path.join(ckpt_dir(f["exp1"]), "last.pt"))

	if state["phase"] == "final_sweep" and not pid_alive(f["sweep1"]) and not pid_alive(f["sweep2"]):
		f["mean1"], f["mean2"] = sweep_mean(f["exp1"], f["final1"]), sweep_mean(f["exp2"], f["final2"])
		log(f"FINAL seed 0-19 mean: {f['exp1']} {f['mean1']}, {f['exp2']} {f['mean2']}")
		state["phase"] = "done"


def main():
	os.makedirs(SCHED_DIR, exist_ok=True)
	log(f"scheduler start (pid {os.getpid()})")
	while True:
		state = load_state()
		if state["phase"] == "done":
			log("all phases done")
			break
		try:
			tick(state)
		except Exception as e:
			log(f"tick error: {e!r}")
		save_state(state)
		time.sleep(POLL_SECONDS)


if __name__ == "__main__":
	main()
