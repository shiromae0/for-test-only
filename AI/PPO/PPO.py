import numpy as np
from stable_baselines3 import PPO
from ShapezEnv import ShapezEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

resource = np.array([
    [0,0,0,11],
    [0,0,0,11],
    [0,0,0,0],
    [0,0,0,0]
])
build = np.full((4,4),-1)
build[3,3] = 21

# 创建自定义环境
env = DummyVecEnv([lambda: ShapezEnv(build, resource, target_shape=11)])
env.reset()
# 创建PPO模型，使用多层感知机策略
model = PPO("MlpPolicy", env, verbose=1)

# 开始训练
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_shapez_model")

# 测试模型
obs = env.reset()
for step in range(5000):
    action, _states = model.predict(obs)
    obs, reward, done,info = env.step(action)
    if done:
        print("Goal reached!", "Reward:", reward)
        #print(info['terminal_observation'])
        # 访问列表中的第一个字典
        first_item = info[0]

        # 通过键 'terminal_observation' 访问数组
        stack = first_item['terminal_observation']
        grid_rsc, grid_bld, grid_direct = np.split(stack, 3, axis=-1)

        # 由于分割后的数组会增加一个维度，需要去掉最后一个维度
        grid_rsc = np.squeeze(grid_rsc, axis=-1)
        grid_bld = np.squeeze(grid_bld, axis=-1)
        grid_direct = np.squeeze(grid_direct, axis=-1)

        # 打印结果以确认
        print("grid_rsc:\n", grid_rsc)
        print("grid_bld:\n", grid_bld)
        print("grid_direct:\n", grid_direct)
        break

# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")
