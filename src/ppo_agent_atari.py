import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
from atari_wrappers import make_atari_env
import gym


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		# [說明] 訓練用環境：灰階 + resize 84x84 + frame skip 4 + frame stack 4，observation shape (4, 84, 84)
		# [說明] 前處理細節見 atari_wrappers.py；reward clip 在 base_agent.train() 存 buffer 時才做
		self.env = make_atari_env(config["env_id"])

		### TODO ###
		# initialize test_env
		# self.test_env = ???
		# [說明] 評估用環境：前處理與訓練相同，但 reward 不 clip (看原始分數)
		# [說明] render_mode="rgb_array" 是因為 evaluate() 每步會呼叫 render()，ALE 在 render_mode=None 時會報錯
		self.test_env = make_atari_env(config["env_id"], render_mode="rgb_array")

		# [說明] 固定亂數種子 (torch / numpy / 環境)，讓實驗可重現；env 只在第一次 reset 時給 seed，之後沿用其 RNG
		seed = config.get("seed", None)
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)
			self.env.reset(seed=seed)
			self.env.action_space.seed(seed)
			self.test_env.reset(seed=seed + 10000)

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		
		# [說明] observation 是 FrameStack 的 LazyFrames，轉成 tensor 並加上 batch 維度: (4, 84, 84) -> (1, 4, 84, 84)
		# [說明] 這裡不除以 255，正規化在 AtariNet.forward() 裡做
		observation = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0).to(self.device)

		# if eval:
		# 	with torch.no_grad():
		# 		???, ???, ???, _ = self.net(observation, eval=True)
		# else:
		# 	???, ???, ???, _ = self.net(observation)
		# [說明] 評估時 eval=True -> greedy (argmax)；收集訓練資料時從 π(·|s) 取樣
		# [說明] 收集資料時也包 no_grad：這裡算出的 value、logp 只是要存進 buffer 當「舊值」，
		# [說明] 不需要計算圖 (效果等同 .detach())，梯度只在 update() 重新 forward 時才計算
		if eval:
			with torch.no_grad():
				action, logp_pi, value, _ = self.net(observation, eval=True)
		else:
			with torch.no_grad():
				action, logp_pi, value, _ = self.net(observation)

		# [說明] 轉回 numpy，三者 shape 統一為 (1,)；value 在 forward 被 squeeze 成 0 維，所以 reshape(-1)
		action = action.cpu().numpy().reshape(-1)
		value = value.cpu().numpy().reshape(-1)
		logp_pi = logp_pi.cpu().numpy().reshape(-1)

		return action, value, logp_pi

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0
		total_approx_kl = 0
		total_clip_frac = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				# ???, ???, ???, ??? = self.net(...)
				# [說明] 用「目前的」網路 θ 重新 forward 同一批 (s, a)：傳入 a=舊動作，取得 log π_θ(a|s)、V_θ(s)、entropy
				# [說明] ob: (B, 4, 84, 84)，ac: (B,) -> logp_new / value_new / entropy: (B,)  (B = batch_size = 128，最後一批可能較小)
				_, logp_new, value_new, entropy = self.net(ob_train_batch, a=ac_train_batch)
				value_new = value_new.reshape(-1)

				# calculate policy loss
				# ratio = ???
				# surrogate_loss = ???
				# [說明] minibatch 內再做一次 advantage normalization (buffer 已對整批做過一次)，讓每個 minibatch 的梯度尺度一致
				adv_train_batch = (adv_train_batch - adv_train_batch.mean()) / (adv_train_batch.std() + 1e-8)
				# [說明] ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ - log π_θ_old)，shape (B,)
				ratio = torch.exp(logp_new - logp_pi_train_batch)
				# [說明] PPO-clip：L = E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ]，要最大化 L，所以 loss 取負號
				# [說明] A>0 時 ratio 最多被推到 1+ε，A<0 時最多被壓到 1-ε，超出範圍梯度為 0 -> 限制每次更新幅度
				surr1 = ratio * adv_train_batch
				surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.min(surr1, surr2).mean()

				# calculate value loss
				# value_criterion = nn.MSELoss()
				# v_loss = value_criterion(...)
				# [說明] critic 回歸 GAE return (= A_t + V_old(s_t))：v_loss = mean((V_θ(s) - return)^2)，兩者 shape 皆 (B,)
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value_new, return_train_batch)
				# [說明] entropy 取 minibatch 平均 (純量)，作為探索獎勵
				entropy = entropy.mean()

				# calculate total loss
				# loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
				# [說明] total = policy loss + c1·value loss - c2·entropy (c1=0.5, c2=0.01)；entropy 越大 loss 越小 -> 鼓勵探索
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				# self.optim.zero_grad()
				# loss.backward()
				# nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				# self.optim.step()
				# [說明] gradient clipping：所有參數梯度的總 L2 norm 超過 0.5 就等比例縮小，避免單次更新過大
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				# [說明] 監控用指標 (不參與梯度)：
				# [說明] approx KL ≈ E[(ratio - 1) - log ratio]，新舊 policy 差多少；clip fraction = 被 clip 的樣本比例
				with torch.no_grad():
					log_ratio = logp_new - logp_pi_train_batch
					approx_kl = ((ratio - 1.0) - log_ratio).mean()
					clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

				# total_surrogate_loss += surrogate_loss.item()
				# total_v_loss += v_loss.item()
				# total_entropy += entropy.item()
				# total_loss += loss.item()
				# loss_counter += 1
				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				total_approx_kl += approx_kl.item()
				total_clip_frac += clip_frac.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		# [說明] 額外紀錄：approx KL、clip fraction、SPS (每秒環境步數)
		self.writer.add_scalar('PPO/Approx KL', total_approx_kl / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Clip Fraction', total_clip_frac / loss_counter, self.total_time_step)
		sps = self.log_sps()
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
		print(f"Approx KL: {total_approx_kl / loss_counter:.5f}\tClip Fraction: {total_clip_frac / loss_counter:.4f}\tSPS: {sps:.1f}")
		# [說明] 定期存可續訓的 last.pt
		self.periodic_checkpoint()
	



