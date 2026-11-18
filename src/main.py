import argparse
import os
import sys
import numpy as np
import torch
from ppo_agent_atari import AtariPPOAgent
from tb_utils import trim_tb_events, trim_eval_log
from finetune_utils import init_from_checkpoint, set_learning_rate

# [說明] 專案根目錄下的 runs/<實驗名稱>/{checkpoints,tb,videos}，權重與 log 不放進 src/
RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))


# [說明] Windows：訓練期間阻止系統進入睡眠 (程式結束後自動恢復)，避免長時間訓練被中斷
def keep_system_awake():
	if sys.platform == "win32":
		import ctypes
		ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
		ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp-name", type=str, default="Enduro_release")
	parser.add_argument("--env-id", type=str, default="ALE/Enduro-v5")
	parser.add_argument("--training-steps", type=float, default=1e8)
	parser.add_argument("--update-sample-count", type=int, default=10000)
	parser.add_argument("--eval-interval-steps", type=int, default=500000, help="evaluate every N steps; 0 = every eval_interval episodes")
	parser.add_argument("--eval-episode", type=int, default=3)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--checkpoint-interval-steps", type=int, default=200000, help="save resumable last.pt every N steps")
	parser.add_argument("--resume", action="store_true", help="resume from runs/<exp-name>/checkpoints/last.pt (if it exists)")
	# [說明] 以下兩個參數用於「分段降低學習率續訓」的額外實驗；不給時就是助教的 baseline 設定
	parser.add_argument("--learning-rate", type=float, default=2.5e-4)
	parser.add_argument("--init-from", type=str, default=None, help="start a new experiment from another experiment's checkpoint (used only when this experiment has no last.pt yet)")
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	exp_dir = os.path.join(RUNS_DIR, args.exp_name)

	config = {
		"gpu": True,
		"training_steps": args.training_steps,
		"update_sample_count": args.update_sample_count,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": os.path.join(exp_dir, "tb"),
		"update_ppo_epoch": 3,
		"learning_rate": args.learning_rate,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": args.env_id,
		"eval_interval": 100,
		"eval_episode": args.eval_episode,
		# [說明] 以下為額外新增的設定
		"eval_interval_steps": args.eval_interval_steps,
		"checkpoint_dir": os.path.join(exp_dir, "checkpoints"),
		"clip_reward": True,
		"seed": args.seed,
		"checkpoint_interval_steps": args.checkpoint_interval_steps,
	}
	print(config, flush=True)
	keep_system_awake()
	last_ckpt = os.path.join(config["checkpoint_dir"], "last.pt")
	if args.resume and os.path.exists(last_ckpt):
		# [說明] 先把 TensorBoard 舊紀錄中超過 checkpoint 步數的部分剪掉 (避免曲線重疊)
		resume_step = torch.load(last_ckpt, map_location="cpu")["total_time_step"]
		trim_tb_events(config["logdir"], resume_step, os.path.join(exp_dir, "tb_backup"))
		trim_eval_log(os.path.join(config["checkpoint_dir"], "eval_log.csv"), resume_step)
	agent = AtariPPOAgent(config)
	resumed = args.resume and os.path.exists(last_ckpt)
	if resumed:
		agent.load_checkpoint(last_ckpt)
	elif args.init_from:
		init_from_checkpoint(agent, args.init_from)
	if resumed or args.init_from:
		# [說明] 載入的 optimizer state 帶有舊的 lr，以本次指定的學習率為準
		set_learning_rate(agent, args.learning_rate)
		# [說明] 續訓時換一組 seed (seed + 目前步數)，避免重播訓練一開始的同一批 episode
		resume_seed = args.seed + agent.total_time_step
		torch.manual_seed(resume_seed)
		np.random.seed(resume_seed % (2 ** 32))
		agent.env.reset(seed=resume_seed)
	agent.train()
	# [說明] 正常跑完才寫 DONE 標記，run_train.ps1 看到它就不再重啟
	open(os.path.join(config["checkpoint_dir"], "DONE"), "w").close()
