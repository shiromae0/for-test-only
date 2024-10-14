from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy

from ShapezEnv import ShapezEnv
import torch
from stable_baselines3.common.callbacks import BaseCallback
import getmap
import torch as th

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

class ActionMaskCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(ActionMaskCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        action_mask = self.env.get_action_mask()
        self.model.policy.set_action_mask(action_mask)
        return True
class MaskedMultiInputPolicy(MultiInputPolicy):
    def __init__(self, *args, callback=None, **kwargs):
        super(MaskedMultiInputPolicy, self).__init__(*args, **kwargs)
        self.action_mask = None
        self.callback = callback  # 将 callback 存储为类的成员

    def set_action_mask(self, action_mask):
        self.action_mask = action_mask

    # def get_distribution(self, obs: PyTorchObs) -> Distribution:
    #     """
    #     Get the current policy distribution given the observations, with optional action masking.
    #
    #     :param obs: The input observations
    #     :return: The action distribution with applied action mask
    #     """
    #     # 提取特征
    #     features = super().extract_features(obs, self.pi_features_extractor)
    #
    #     # 获取 actor 的潜在特征
    #     latent_pi = self.mlp_extractor.forward_actor(features)
    #
    #     # 生成动作分布
    #     action_distribution = self._get_action_dist_from_latent(latent_pi)
    #
    #     # 如果存在动作掩码，应用到动作分布上
    #     if self.action_mask is not None:
    #         # 获取分布的 logits，并应用掩码
    #         logits = action_distribution.distribution.logits
    #         action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32).to(logits.device)
    #         masked_logits = logits + (action_mask_tensor - 1) * 1e9  # 将非法动作的概率设置为极低
    #         action_distribution.distribution = th.distributions.Categorical(logits=masked_logits)
    #     print(action_distribution)
    #     return action_distribution

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes, with optional action masking.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if self.callback is not None:
            self.callback._on_step()
        if isinstance(self.action_dist, DiagGaussianDistribution):
            print("diagGaussian")
            return self.action_dist.proba_distribution(mean_actions, self.log_std)

        elif isinstance(self.action_dist, CategoricalDistribution):
            # print("Cate")
            # Here mean_actions are the logits before the softmax
            action_logits = mean_actions

            # 如果存在动作掩码，将其转换为 PyTorch 张量并应用掩码
            if self.action_mask is not None:
                # one_indices = [index for index, value in enumerate(self.action_mask) if value == 1]
                # print(f"Indices of 1s in action_mask: {one_indices}")
                # 确保 action_mask 位于与 action_logits 相同的设备上
                action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32).to(action_logits.device)
                action_logits = action_logits + (action_mask_tensor - 1) * 1e9
                # valid_indices = (action_logits != -1e9).nonzero(as_tuple=True)
                # 打印不为 -1e9 的索引
                # print(valid_indices)
            else:
                print("error")
            return self.action_dist.proba_distribution(action_logits=action_logits)

        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            print("Mutil")
            action_logits = mean_actions

            # 如果存在动作掩码，将其转换为 PyTorch 张量并应用掩码
            if self.action_mask is not None:
                action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32).to(action_logits.device)
                action_logits = action_logits + (action_mask_tensor - 1) * 1e9

            return self.action_dist.proba_distribution(action_logits=action_logits)

        elif isinstance(self.action_dist, BernoulliDistribution):
            print("Berno")
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            print("state")
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)

        else:
            raise ValueError("Invalid action distribution")




resource = getmap.load_shared_arrays()[0]
build = getmap.load_shared_arrays()[1]
target_shape = getmap.load_needed_shape()
print(resource)
print(build)
build[build != -1] *= 100

# resource = np.array([
#     [11, 0, 0, 0, 0],             # (0,0) 位置的资源形状为 11
#     [0, 0, 0, 0, 0],
#     [11, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

def linear_schedule(initial_value):
    def func(progress_remaining):
        # progress_remaining 从 1 到 0，1 表示训练开始，0 表示训练结束
        return initial_value * progress_remaining  # 线性减小学习率

    return func


# 创建自定义环境
env = ShapezEnv(build, resource, target_shape=13)
env.reset()
act_list = env.action_list
# 创建PPO模型，使用多层感知机策略
# model = model.load("ppo_shapez_model")

# 开始训练
callback = ActionMaskCallback(env)
model = PPO(MaskedMultiInputPolicy, env, verbose=1, policy_kwargs={'callback': callback})
model.set_env(env)
model.learn(total_timesteps=100000, callback=callback)

# 保存模型
model.save("ppo_shapez_model")

# 测试模型

def get_agent_act_list():
    obs, info = env.reset()
    callback = ActionMaskCallback(env)
    agent_act = []
    for step in range(30000):
        if step == 0:
            print("start")
        action, _states = model.predict(obs)
        # print("action =",act_list[action])
        result = env.step(action)
        obs, reward, done, truncated, info = env.step(action)
        agent_act.append(act_list[action])
        # print(obs["grid"])
        if truncated == True:
            env.reset()
            agent_act = []
            print("Truncated")
        elif done == True:
            print("Goal reached!", "Reward:", reward)
            # 假设 info 是一个列表，提取第一个字典中的 'terminal_observation'
            print(obs["grid"])
            break
    return agent_act
print(get_agent_act_list())
# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")
