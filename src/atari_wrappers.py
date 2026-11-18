import gym
from gym.wrappers import AtariPreprocessing, FrameStack


# [說明] Atari 前處理：(210, 160, 3) RGB -> 灰階 + resize (84, 84) -> 疊 4 張 -> (4, 84, 84)
# [說明] ALE/*-v5 預設 frameskip=4、sticky action 0.25；AtariPreprocessing 自己會做 frame skip，
# [說明] 所以 gym.make 必須傳 frameskip=1，避免 skip 被做兩次 (變成每個動作 16 frames)。
# [說明] sticky action (repeat_action_probability=0.25) 維持 v5 預設，訓練/測試都一樣。
# [說明] reward clip 不在這裡做：訓練時在 base_agent.train() 存進 buffer 前才取 sign，
# [說明] 這樣 Train/Episode Reward 記錄的仍是原始分數；評估/測試一律用原始 reward。
def make_atari_env(env_id, render_mode=None, frame_stack=4, noop_max=30):
	env = gym.make(env_id, frameskip=1, render_mode=render_mode)
	# [說明] 每個 agent step = 4 個 ALE frame，觀測取最後兩個 frame 的 max (去除閃爍)
	# [說明] noop_max=30：reset 後隨機做 0~30 個 NOOP，增加起始狀態多樣性
	# [說明] Enduro 沒有 lives，所以 terminal_on_life_loss=False；scale_obs=False 保持 uint8 0~255，
	# [說明] 由網路 forward() 裡的 x / 255 做正規化
	env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=4, screen_size=84,
							terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False)
	# [說明] 疊 4 張連續畫面讓網路看得到速度/方向，輸出 LazyFrames，np.asarray 後 shape = (4, 84, 84)
	env = FrameStack(env, frame_stack)
	return env
