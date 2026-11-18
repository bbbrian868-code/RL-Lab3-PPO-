import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 1)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, a=[]):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        value = self.value(x)
        value = torch.squeeze(value)

        logits = self.action_logits(x)
        
        dist = Categorical(logits=logits)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy
        # [說明] 共享 CNN trunk 後分兩個 head：actor 輸出 logits (B, num_actions)，critic 輸出 V(s)
        # [說明] x: (B, 4, 84, 84) -> cnn -> (B, 64, 7, 7) -> flatten (B, 3136)
        # [說明] value 經 squeeze 後為 (B,)；B=1 時會變 0 維，呼叫端再 reshape
        # [說明] 三種用法：
        # [說明]   1. 收集資料 (eval=False, a 為空)：從 π(·|s) 取樣動作 (探索)
        # [說明]   2. 評估/測試 (eval=True)：取機率最大的動作 (greedy, 結果較穩定)
        # [說明]   3. PPO update (傳入 a)：不取樣，計算「舊動作 a 在新 policy 下」的 log π_θ(a|s)，用於 ratio
        if len(a) > 0:
            action = a
        elif eval:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        # [說明] log π(a|s)，shape (B,)；回傳 log prob 而非機率本身，ratio = exp(logp_new - logp_old) 數值較穩定
        logp_pi = dist.log_prob(action)
        # [說明] 每個樣本的 entropy H(π(·|s))，shape (B,)；update 時取 mean 當作探索獎勵
        entropy = dist.entropy()

        return action, logp_pi, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # [說明] 輸出層另外初始化 (參考 "37 Implementation Details of PPO")：
        # [說明] actor 輸出層 gain 0.01 -> 初始 logits 接近 0，各動作機率接近均勻，初期探索充分
        # [說明] critic 輸出層 gain 1 -> 初始 V(s) 尺度合理
        nn.init.orthogonal_(self.action_logits[-1].weight, 0.01)
        nn.init.constant_(self.action_logits[-1].bias, 0.0)
        nn.init.orthogonal_(self.value[-1].weight, 1.0)
        nn.init.constant_(self.value[-1].bias, 0.0)



