import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]

		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": 1,
			})

		self.writer = SummaryWriter(config["logdir"])

		# [說明] 以下為額外新增的設定 (sample code 沒有)：
		# [說明] clip_reward：訓練時存進 buffer 的 reward 取 sign -> {-1, 0, +1}；episode reward 紀錄仍用原始分數
		# [說明] eval_interval_steps > 0 時改成「每固定步數」評估 + 存檔，否則沿用原本「每 eval_interval 個 episode」
		# [說明] checkpoint_dir：權重存放處 (與 TensorBoard log 分開)
		self.clip_reward = config.get("clip_reward", True)
		self.eval_interval_steps = int(config.get("eval_interval_steps", 0))
		self.checkpoint_dir = config.get("checkpoint_dir", self.writer.log_dir)
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		# [說明] 續訓需要的狀態：episode 編號、最佳評估分數、下一次評估的步數、累計訓練時間
		self.episode_idx = 0
		self.best_score = -float("inf")
		self.next_eval_step = self.eval_interval_steps
		self.last_eval_step = -1
		self.checkpoint_interval_steps = int(config.get("checkpoint_interval_steps", 0))
		self.next_checkpoint_step = self.checkpoint_interval_steps
		self.elapsed_before = 0.0
		self.session_start_time = time.time()
		self.session_start_step = 0

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		# [說明] 續訓時 episode_idx 從 checkpoint 接續 (全新訓練為 0)
		episode_idx = self.episode_idx
		self.session_start_time = time.time()
		self.session_start_step = self.total_time_step
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			self.episode_idx = episode_idx
			while True:
				### TODO ###
				# get action from net and get next information from env
				# [說明] rollout 階段：用目前的 policy π_θ_old 對 observation (4, 84, 84) 取樣動作，
				# [說明] 同時記下 V(s_t) 和 log π_old(a_t|s_t)：之後算 GAE 需要 V，算 PPO ratio 需要舊的 log prob
				# [說明] decide_agent_actions 回傳的三者 shape 都是 (1,) (batch 維度)，所以 env.step 傳 action[0]
				action, value, logp_pi = self.decide_agent_actions(observation)
				next_observation, reward, terminate, truncate, info = self.env.step(action[0])
				# [說明] reward clipping：只有存進 buffer 給 PPO 學習的 reward 取 sign，
				# [說明] 下面 episode_reward += reward 仍累加原始 reward (TensorBoard 看到的是原始分數)
				train_reward = float(np.sign(reward)) if self.clip_reward else float(reward)

				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				# [說明] 每筆 transition 存成純量：action / reward / value / logp_pi / done 維度一致 (皆為 0 維)，
				# [說明] extract_batch() 串接後各自變成 (N,)；observation 包成 dict，key 為 "observation_2d"，(4, 84, 84)
				obs = {}
				obs["observation_2d"] = np.asarray(observation, dtype=np.float32)
				self.gae_replay_buffer.append(0, {
						### TODO ###
						# store the transition into gae_replay_buffer
						# [說明] done = terminate or truncate：buffer 用 done 切分 trajectory，且 GAE 在 done 處不往後 bootstrap
						# [說明] (truncate 也視為終止是 sample buffer 的做法；Enduro 只在 108000 frames 上限才會 truncate)
						"observation": obs,
						"action": action[0],
						"reward": train_reward,
						"value": value[0],
						"logp_pi": logp_pi[0],
						"done": terminate or truncate,
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()

				episode_reward += reward
				episode_len += 1
				
				if terminate or truncate:
					self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len}")
					break
					
				observation = next_observation
				self.total_time_step += 1
				
			# [說明] 原本是 episode_idx % eval_interval == 0；改由 should_evaluate() 判斷 (支援每固定步數評估)
			if self.should_evaluate(episode_idx):
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.checkpoint_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				# [說明] 額外：更新 best.pth、寫評估紀錄、存可續訓的 last.pt
				self.after_evaluate(avg_score)

		# [說明] 訓練結束：若最後一步還沒評估過，再評估一次並存檔
		if self.last_eval_step != self.total_time_step:
			avg_score = self.evaluate()
			self.save(os.path.join(self.checkpoint_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
			self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
			self.after_evaluate(avg_score)
		self.writer.flush()

	# [說明] 判斷這個 episode 結束後要不要評估
	def should_evaluate(self, episode_idx):
		if self.eval_interval_steps > 0:
			if self.total_time_step >= self.next_eval_step:
				while self.next_eval_step <= self.total_time_step:
					self.next_eval_step += self.eval_interval_steps
				return True
			return False
		return episode_idx % self.eval_interval == 0

	# [說明] 評估後：分數創新高就存 best.pth (只有網路權重，test.py 讀這個)；
	# [說明] 並把 (步數, 分數) 附加到 eval_log.csv
	def after_evaluate(self, avg_score):
		self.last_eval_step = self.total_time_step
		if avg_score > self.best_score:
			self.best_score = avg_score
			self.save(os.path.join(self.checkpoint_dir, "best.pth"))
			with open(os.path.join(self.checkpoint_dir, "best_info.txt"), "w") as f:
				f.write(f"step={self.total_time_step}\nscore={avg_score}\nepisode={self.episode_idx}\n")
			print(f"New best score {avg_score} at step {self.total_time_step}, saved best.pth")
		with open(os.path.join(self.checkpoint_dir, "eval_log.csv"), "a") as f:
			f.write(f"{self.total_time_step},{avg_score},{self.elapsed_time():.1f}\n")
		# [說明] 這裡不存 last.pt：評估發生在 episode 結束時，buffer 可能還有尚未 update 的資料，
		# [說明] last.pt 只在 update 結束 (buffer 剛用完) 時由 periodic_checkpoint() 存，續訓才不會丟資料
		self.writer.flush()

	# [說明] 每 checkpoint_interval_steps 步存一次 last.pt (在 update 結束、buffer 清空時呼叫)，
	# [說明] 程序被意外中止時最多損失這段步數
	def periodic_checkpoint(self):
		if self.checkpoint_interval_steps > 0 and self.total_time_step >= self.next_checkpoint_step:
			while self.next_checkpoint_step <= self.total_time_step:
				self.next_checkpoint_step += self.checkpoint_interval_steps
			self.save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"))

	# [說明] 累計訓練時間 (含續訓前的時間)
	def elapsed_time(self):
		return self.elapsed_before + (time.time() - self.session_start_time)

	# [說明] 本次執行的平均 SPS (steps per second)，寫進 TensorBoard
	def log_sps(self):
		dt = time.time() - self.session_start_time
		sps = (self.total_time_step - self.session_start_step) / max(dt, 1e-6)
		self.writer.add_scalar('Train/SPS', sps, self.total_time_step)
		return sps

	# [說明] 完整 checkpoint：網路 + optimizer 狀態 + 步數等，讓訓練中斷後可以接續
	def save_checkpoint(self, save_path):
		tmp_path = save_path + ".tmp"
		torch.save({
			"net": self.net.state_dict(),
			"optim": self.optim.state_dict(),
			"total_time_step": self.total_time_step,
			"episode_idx": self.episode_idx,
			"best_score": self.best_score,
			"next_eval_step": self.next_eval_step,
			"elapsed_time": self.elapsed_time(),
		}, tmp_path)
		# [說明] 先寫暫存檔再改名，避免存到一半被中斷導致 checkpoint 損毀
		os.replace(tmp_path, save_path)

	def load_checkpoint(self, load_path):
		ckpt = torch.load(load_path, map_location=self.device)
		self.net.load_state_dict(ckpt["net"])
		self.optim.load_state_dict(ckpt["optim"])
		self.total_time_step = ckpt["total_time_step"]
		self.episode_idx = ckpt["episode_idx"]
		self.best_score = ckpt["best_score"]
		self.next_eval_step = ckpt["next_eval_step"]
		self.elapsed_before = ckpt["elapsed_time"]
		self.next_checkpoint_step = self.total_time_step + self.checkpoint_interval_steps
		# [說明] best.pth 可能比 last.pt 新 (評估後、下次存 last.pt 前被中斷)，以 best_info.txt 的分數為準，避免被較差的模型覆蓋
		best_info = os.path.join(self.checkpoint_dir, "best_info.txt")
		if os.path.exists(best_info):
			with open(best_info) as f:
				info = dict(line.strip().split("=", 1) for line in f if "=" in line)
			self.best_score = max(self.best_score, float(info["score"]))
		print(f"Resumed from {load_path}: step {self.total_time_step}, episode {self.episode_idx}, best {self.best_score}")

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				self.test_env.render()
				action, _, _ = self.decide_agent_actions(observation, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action[0])
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path, map_location=self.device))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


	

