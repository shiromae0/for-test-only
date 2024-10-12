import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy

from shapezenv import ShapezEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from stable_baselines3.common.callbacks import BaseCallback


class ActionMaskCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(ActionMaskCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        action_mask = self.env.get_action_mask()
        self.model.policy.set_action_mask(action_mask)
        return True


class MaskedMultiInputPolicy(MultiInputPolicy):
    def __init__(self, *args, **kwargs):
        super(MaskedMultiInputPolicy, self).__init__(*args, **kwargs)
        self.action_mask = None

    def set_action_mask(self, action_mask):
        self.action_mask = action_mask

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        # 获取策略和价值的潜在向量
        latent_pi, latent_vf = self.mlp_extractor(features)
        # 获取动作分布
        distribution = self._get_action_dist_from_latent(latent_pi)
        # 使用动作掩码过滤无效动作
        if self.action_mask is not None:
            # 对于离散动作空间，修改分布的概率
            masked_probs = distribution.distribution.probs * torch.tensor(self.action_mask, device=self.device)
            # 避免除以零
            masked_probs_sum = masked_probs.sum(dim=-1, keepdim=True)
            # 确保总概率为 1
            masked_probs = masked_probs / masked_probs_sum
            # 更新分布的概率
            distribution.distribution.probs = masked_probs

        # 根据是否是确定性动作选择
        if deterministic:
            actions = distribution.get_actions(deterministic=True)
        else:
            actions = distribution.get_actions()
        # 计算 log 概率
        log_prob = distribution.log_prob(actions)
        # 计算价值估计
        values = self.value_net(latent_vf)

        return actions, values, log_prob


resource = np.full((20, 20), 0)
build = np.full((20, 20), -1)
resource[19, 19] = 11
build[0,0] = 2100
build[0,1] = 2100


def linear_schedule(initial_value):
    def func(progress_remaining):
        # progress_remaining 从 1 到 0，1 表示训练开始，0 表示训练结束
        return initial_value * progress_remaining  # 线性减小学习率

    return func


# 创建自定义环境
env = ShapezEnv(build, resource, target_shape=11)
env.reset()
# 创建PPO模型，使用多层感知机策略

# model = model.load("ppo_shapez_model")

# 开始训练
callback = ActionMaskCallback(env)
model = PPO(MaskedMultiInputPolicy, env, verbose=1)
model.set_env(env)
model.learn(total_timesteps=100000, callback=callback)

# 保存模型
model.save("ppo_shapez_model")

# 测试模型
obs, info = env.reset()
for step in range(5000):
    action, _states = model.predict(obs)
    result = env.step(action)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        if truncated == True:
            print("Truncated")
        elif done == True:
            print("Goal reached!", "Reward:", reward)
            # 假设 info 是一个列表，提取第一个字典中的 'terminal_observation'
            terminal_observation = info[0]['terminal_observation']
            print(terminal_observation)
            break

# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")
