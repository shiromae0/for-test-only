import os

import numpy as np
from stable_baselines3 import PPO
from ShapezEnv import ShapezEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import torch.nn.functional as F
# import getmap
class Machine:
    def __init__(self,type,position, direction = -1):
        self.position = position
        self.direction = direction
        self.type = type
class CustomMlpPolicy(MlpPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, extra_param_list, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.action_list = extra_param_list
        self.delete_miner_idx = []
        self.delete_idx = []
        self.miner_idx = []
        self.conveyor_idx = []
        miner_pos = []
        idx = 0
        for i, (a_type, direct) in enumerate(self.action_list):
            for k,pos in enumerate(self.action_list[(a_type,direct)]):
                if a_type == 22:
                    self.miner_idx.append(idx)
                    miner_pos.append((pos[0],pos[1]))
                elif a_type == 0:
                    if pos in miner_pos:
                        self.delete_miner_idx.append(idx)
                    else:
                        self.delete_idx.append(idx)
                idx += 1

    def get_priority_actions_idx(self, obs):
        last_column = obs[:, -1]
        if last_column.item() == 0:
            return self.miner_idx+self.delete_miner_idx
    def forward(self, obs):
    # 调用父类的 forward 方法获取初步的 actions, values, log_prob
        actions, values, log_prob = super(CustomMlpPolicy, self).forward(obs)
        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # 获取动作分布
        distribution = self._get_action_dist_from_latent(latent_pi)

        # 复制 logits，确保可以安全修改
        logits = distribution.distribution.logits.clone()

        # 获取优先考虑的动作索引
        prior_action_idx = self.get_priority_actions_idx(obs)

        mask = torch.zeros_like(logits)  # 创建一个全 0 的张量与 logits 大小相同
        mask[:, prior_action_idx] = 1  # 仅将优先考虑的动作设为 1

        # 将不在优先动作索引中的 logits 置为极小值 (-inf)，在 softmax 中相当于概率为 0
        logits = logits * mask  # 将非优先动作的 logits 置为 0

        # 如果你想让非优先动作完全不可能被选择，可以用 -inf 替换非优先动作
        logits[mask == 0] = float('-inf')  # 使用 -inf 确保这些动作的概率为 0
        adjusted_distribution = torch.distributions.Categorical(logits=logits)

        # 从新的分布中采样动作
        actions = adjusted_distribution.sample()

        # 计算调整后分布的 log_prob
        log_prob = adjusted_distribution.log_prob(actions)

        return actions, values, log_prob


array = []
# resource = getmap.load_shared_arrays()[0]
# build = getmap.load_shared_arrays()[1]
# target_shape = getmap.load_needed_shape()
resource = np.full((5,5),0)
build = np.full((5,5),-1)
resource[3,3] = 11
build [3,4] = 3101
build [4,3] = 3101
build [2,3] = 3101
build [3,2] = 3101

build [1,1] = 21
Height = build.shape[0]
Width = build.shape[1]
build[build != -1] *= 100

Env = ShapezEnv(build,resource,target_shape=11)
act_list = Env.action_list
# 创建自定义环境
env = DummyVecEnv([lambda: ShapezEnv(build, resource, target_shape=11)])
env.reset()
# 创建PPO模型，使用多层感知机策略


policy_kwargs = dict(
    extra_param_list=act_list
)

model = PPO("MlpPolicy", env, verbose=1)

# 开始训练
model.learn(total_timesteps=2000)

# 保存模型
model.save("ppo_shapez_model")

# 测试模型

obs = env.reset()

actions = []
for step in range(10000):
    action, _states = model.predict(obs)
    action = np.atleast_1d(action)
    obs, reward, done,info = env.step(action)
    actions.append(tuple(act_list[action[0]]))

    if done:
        if info[0]["TimeLimit.truncated"] == True:
            obs = info[0]['terminal_observation']
            print("Truncated")
            actions.clear()
        else:
            actions.append(info[0]['path'])
            print("Goal reached!", "Reward:", reward)
            # 假设 info 是一个列表，提取第一个字典中的 'terminal_observation'
            terminal_observation = info[0]['terminal_observation']

            array = terminal_observation[:Height*Width].reshape((Height, Width))
            np.set_printoptions(threshold=100000)
            break
            #print(info)
            # print(terminal_observation)
            # np.save('array_data.npy', array)
            # print("Array saved to array_data.npy!")

            # print(array)
print(actions)
def return_array():
    coordinates = np.where(array != -1)
    for y, x in zip(coordinates[0], coordinates[1]):
        print(f"坐标: ({y}, {x}), 值: {array[y, x]}")
    return array


# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")
