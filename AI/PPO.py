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
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_shapez_model")

# 测试模型
obs = env.reset()
for step in range(3):
    action, _states = model.predict(obs)
    obs, reward, done,info = env.step(action)
    if done:
        print("Goal reached!", "Reward:", reward)
        print(info)
        break

# 评估模型
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"平均奖励: {mean_reward} +/- {std_reward}")
