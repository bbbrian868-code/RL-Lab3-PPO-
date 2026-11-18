import torch

# [說明] 額外實驗「分段降低學習率續訓」用的工具 (與助教 TODO 無關，獨立成檔案)：
# [說明] 從某個實驗的 checkpoint 分支出新實驗，並改用指定的學習率繼續訓練


# [說明] 接續網路、optimizer (Adam 動量)、步數、episode 編號與累計時間，讓 TensorBoard 曲線接在原步數後面；
# [說明] best_score 不接續 (新實驗有自己的 best.pth)，下一次評估 / 存檔從目前步數重新起算
def init_from_checkpoint(agent, load_path):
	ckpt = torch.load(load_path, map_location=agent.device)
	if "net" not in ckpt:
		# [說明] 只有權重的 .pth：只載入網路，optimizer 與步數從頭開始
		agent.net.load_state_dict(ckpt)
		print(f"Initialized weights from {load_path}")
		return
	agent.net.load_state_dict(ckpt["net"])
	agent.optim.load_state_dict(ckpt["optim"])
	agent.total_time_step = ckpt["total_time_step"]
	agent.episode_idx = ckpt["episode_idx"]
	agent.elapsed_before = ckpt["elapsed_time"]
	agent.next_eval_step = agent.total_time_step + agent.eval_interval_steps
	agent.next_checkpoint_step = agent.total_time_step + agent.checkpoint_interval_steps
	print(f"Initialized from {load_path}: step {agent.total_time_step}, episode {agent.episode_idx}")


# [說明] 覆寫 optimizer 的學習率 (載入 optimizer state 時會一併載入舊的 lr，所以載入後要再設一次)
def set_learning_rate(agent, lr):
	for group in agent.optim.param_groups:
		group["lr"] = lr
	agent.writer.add_scalar('Train/Learning Rate', lr, agent.total_time_step)
	print(f"Learning rate set to {lr}")
