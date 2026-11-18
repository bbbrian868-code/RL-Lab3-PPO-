import argparse
import csv
import os
import time
import warnings

import cv2
import numpy as np
import torch
from gym.wrappers import RecordVideo

from atari_wrappers import make_atari_env
from models.atari_model import AtariNet

warnings.filterwarnings("ignore")

# [說明] 測試腳本：載入 checkpoint，用「原始 reward」測 N 場，可指定 seed 重現、錄影或即時顯示
# [說明] 範例：
# [說明]   python test.py --exp-name Enduro_release                      (best.pth 測 5 場)
# [說明]   python test.py --exp-name Enduro_release --episodes 1 --seed 7 --render   (demo：只測 1 場，即時顯示)
# [說明]   python test.py --exp-name Enduro_release --episodes 1 --seed 7 --record   (錄影到 runs/<exp>/videos/)
# [說明]   python test.py --exp-name Enduro_release --sweep 0 19         (掃 seed 0~19，每個 seed 1 場)

RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
# [說明] Enduro 一天約 3326 個 agent step (frame skip 4)；第 1 天要超 200 台、之後每天 300 台
STEPS_PER_DAY = 3326


# [說明] 依助教評分表由 reward 換算天數與 demo 分數：第 1 天 0~200、第 2 天 200~500、之後每 300 一天
def day_from_reward(reward):
	if reward < 200:
		return 1
	return min(2 + int((reward - 200) // 300), 9)


def demo_points(day):
	return {1: 0, 2: 10, 3: 20}.get(day, min(25 + 5 * (day - 4), 50))


# [說明] 由 episode 長度推算結束時在第幾天：遊戲只會在「某天結束時沒達標」(terminate) 結束，
# [說明] 所以長度約為天數的整數倍 (每天 3321~3328 步不等，用四捨五入)；
# [說明] 若是撞到 108000 frames 上限 (truncate)，則是在下一天的途中被截斷
def day_from_length(length, truncated=False):
	if truncated:
		return length // STEPS_PER_DAY + 1
	return max(1, round(length / STEPS_PER_DAY))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp-name", type=str, default="Enduro_release")
	parser.add_argument("--env-id", type=str, default="ALE/Enduro-v5")
	parser.add_argument("--ckpt", type=str, default=None, help="default: runs/<exp-name>/checkpoints/best.pth")
	parser.add_argument("--episodes", type=int, default=5)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy argmax")
	parser.add_argument("--render", action="store_true", help="show the game window (render_mode='human', real-time speed)")
	parser.add_argument("--display-fps", type=int, default=0, help="show the game in an OpenCV window at this many agent steps/s (real-time = 15); faster than --render")
	parser.add_argument("--record", action="store_true", help="save mp4 to runs/<exp-name>/videos/")
	parser.add_argument("--video-fps", type=int, default=30)
	parser.add_argument("--sweep", type=int, nargs=2, metavar=("FIRST", "LAST"), help="run 1 episode for each seed in [FIRST, LAST]")
	parser.add_argument("--cpu", action="store_true")
	return parser.parse_args()


def load_net(args, n_actions, device):
	ckpt = args.ckpt or os.path.join(RUNS_DIR, args.exp_name, "checkpoints", "best.pth")
	net = AtariNet(n_actions).to(device)
	state = torch.load(ckpt, map_location=device)
	# [說明] 同時支援只有權重的 .pth 和含 optimizer 的完整 checkpoint (last.pt)
	net.load_state_dict(state["net"] if "net" in state else state)
	net.eval()
	print(f"Loaded {ckpt}")
	return net


def make_test_env(args, seed, video_tag):
	# [說明] human：ALE 自己開視窗，速度固定為真實時間 (每秒 15 個 agent step，9 天約 30 分鐘)
	# [說明] --display-fps：改用 rgb_array 畫面 + OpenCV 視窗顯示，可自訂速度 (demo 省時間用)
	render_mode = "human" if args.render else ("rgb_array" if args.record or args.display_fps > 0 else None)
	env = make_atari_env(args.env_id, render_mode=render_mode)
	if args.record:
		env.metadata["render_fps"] = args.video_fps
		video_dir = os.path.join(RUNS_DIR, args.exp_name, "videos")
		env = RecordVideo(env, video_dir, episode_trigger=lambda e: True, name_prefix=video_tag)
	return env


# [說明] 用 OpenCV 顯示原始 RGB 畫面 (放大 3 倍)，並在畫面上顯示目前分數與步數
_last_show = [0.0]


def show_frame(env, total_reward, length, fps):
	frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
	frame = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
	cv2.putText(frame, f"reward {total_reward:.0f}  step {length}  day {length // STEPS_PER_DAY + 1}", (8, 22),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.imshow("PPO Enduro", frame)
	cv2.waitKey(1)
	wait = 1.0 / fps - (time.time() - _last_show[0])
	if wait > 0:
		time.sleep(wait)
	_last_show[0] = time.time()


# [說明] 固定 torch / numpy / 環境的 seed：同一個 seed 會得到同樣的結果
def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)


def run_episodes(net, args, seed, n_episodes, device, video_tag):
	set_seed(seed)
	env = make_test_env(args, seed, video_tag)
	results = []
	for ep in range(n_episodes):
		# [說明] 只有第一場 reset 時給 seed，後面幾場沿用同一個 RNG 繼續 (整組結果仍可重現)
		observation, info = env.reset(seed=seed) if ep == 0 else env.reset()
		total_reward, length = 0.0, 0
		while True:
			obs = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0).to(device)
			with torch.no_grad():
				action, _, _, _ = net(obs, eval=not args.stochastic)
			observation, reward, terminate, truncate, info = env.step(int(action.item()))
			total_reward += reward
			length += 1
			if args.display_fps > 0:
				show_frame(env, total_reward, length, args.display_fps)
			if terminate or truncate:
				break
		results.append((total_reward, length, truncate))
		print(f"Episode: {ep + 1}\tLength: {length}\tTotal reward: {total_reward:.2f}", flush=True)
	env.close()
	if args.display_fps > 0:
		cv2.destroyAllWindows()
	return results


def main():
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	# [說明] 關掉 cuDNN 的非確定性演算法，讓 GPU 上同一個 seed 每次結果一致
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	probe = make_atari_env(args.env_id)
	n_actions = probe.action_space.n
	probe.close()
	net = load_net(args, n_actions, device)
	mode = "stochastic" if args.stochastic else "greedy"

	if args.sweep:
		first, last = args.sweep
		rows = []
		for seed in range(first, last + 1):
			print(f"--- seed {seed}")
			(reward, length, truncated), = run_episodes(net, args, seed, 1, device, f"seed{seed}_{mode}")
			rows.append((seed, reward, length, day_from_reward(reward), day_from_length(length, truncated)))
		ckpt_name = os.path.splitext(os.path.basename(args.ckpt or "best.pth"))[0]
		out_csv = os.path.join(RUNS_DIR, args.exp_name, f"seed_sweep_{ckpt_name}_{mode}.csv")
		with open(out_csv, "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["seed", "reward", "length", "day_by_reward", "day_by_length", "demo_points"])
			for r in rows:
				w.writerow([r[0], f"{r[1]:.0f}", r[2], r[3], r[4], demo_points(r[3])])
		print(f"\n{'seed':>4} {'reward':>8} {'length':>7} {'day(reward)':>11} {'day(length)':>11} {'points':>6}")
		for r in rows:
			print(f"{r[0]:>4} {r[1]:>8.0f} {r[2]:>7} {r[3]:>11} {r[4]:>11} {demo_points(r[3]):>6}")
		best = max(rows, key=lambda r: r[1])
		print(f"\nmean reward: {np.mean([r[1] for r in rows]):.1f}   best seed: {best[0]} (reward {best[1]:.0f}, day {best[3]})")
		print(f"saved {out_csv}")
	else:
		results = run_episodes(net, args, args.seed, args.episodes, device, f"seed{args.seed}_{mode}")
		avg = sum(r[0] for r in results) / len(results)
		print(f"average score: {avg}")


if __name__ == "__main__":
	main()
