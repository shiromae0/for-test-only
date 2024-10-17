import sys

import numpy as np
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
    def __init__(self, *args, model=None, **kwargs):
        super(MaskedMultiInputPolicy, self).__init__(*args, **kwargs)
        self.model = model  # 保存 model


    def set_action_mask(self, action_mask):
        self.action_mask = action_mask

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes, with optional action masking.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        original_env = self.model.env.envs[0].unwrapped
        self.action_mask = original_env.get_action_mask()
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)

        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            action_logits = mean_actions
            if self.action_mask is not None:
                # one_indices = [index for index, value in enumerate(self.action_mask) if value == 1]
                # print(f"Indices of 1s in action_mask: {one_indices}")
                action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32).to(action_logits.device)
                action_logits = action_logits + (action_mask_tensor - 1) * 1e9
            return self.action_dist.proba_distribution(action_logits=action_logits)

        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            action_logits = mean_actions
            if self.action_mask is not None:
                action_mask_tensor = th.tensor(self.action_mask, dtype=th.float32).to(action_logits.device)
                action_logits = action_logits + (action_mask_tensor - 1) * 1e9

            return self.action_dist.proba_distribution(action_logits=action_logits)

        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")




print(getmap.load_shared_arrays()[1])

def get_agent_act_list():

    resource = getmap.load_shared_arrays()[0]
    build = getmap.load_shared_arrays()[1]
    target_shape = getmap.load_needed_shape()
    # print(resource)
    # print(np.array2string(build, max_line_width=200))
    env =ShapezEnv(build, resource, target_shape=target_shape)
    env.reset()
    act_list = env.action_list
    # model = PPO(MaskedMultiInputPolicy, env=env, verbose=1, policy_kwargs={'model': None},gamma=0.98)
    model = PPO.load("ppo_model.zip",env=env,gamma = 0.98, custom_objects={"policy_class": MaskedMultiInputPolicy})

    model.set_env(env)
    model.policy.model = model
    # model.learn(total_timesteps=10000)
    model.save("ppo_model.zip")

    obs, info = env.reset()
    agent_act = []
    for step in range(10000):

        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action.item())
        agent_act.append(act_list[action])

        if truncated == True:
            env.reset()
            agent_act = []
            print("Truncated")
        elif done == True:
            print("Goal reached!", "Reward:", reward)
            print(obs["grid"])
            break
    return agent_act
# print(get_agent_act_list())
# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")
