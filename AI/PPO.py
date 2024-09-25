import numpy as np
from stable_baselines3 import PPO
from ShapezEnv import ShapezEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


resource = np.full((15,15),0)
build = np.full((15,15),-1)
build[3,3] = 21
resource[7,7] = 11
resource[7,8] = 15
resource[5,5] = 11
# 创建自定义环境
env = DummyVecEnv([lambda: ShapezEnv(build, resource, target_shape=11)])
env.reset()
# 创建PPO模型，使用多层感知机策略
model = PPO("MlpPolicy", env, verbose=1)

# 开始训练
model.learn(total_timesteps=50000)

# 保存模型
model.save("ppo_shapez_model")

# 测试模型
obs = env.reset()
for step in range(2000):
    action, _states = model.predict(obs)
    obs, reward, done,info = env.step(action)
    if done:
        if info[0]["TimeLimit.truncated"] == True:
            print("Truncated")
        else:
            print("Goal reached!", "Reward:", reward)
            # 假设 info 是一个列表，提取第一个字典中的 'terminal_observation'
            terminal_observation = info[0]['terminal_observation']
            first_layer = terminal_observation[..., 0]  # 获取第一个维度
            second_layer = terminal_observation[..., 1]  # 获取第二个维度
            third_layer = terminal_observation[..., 2]  # 获取第三个维度
            print("Second layer:")
            print(second_layer)
            print("Third layer:")
            print(third_layer)
            break

# 评估模型
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"平均奖励: {mean_reward} +/- {std_reward}")
