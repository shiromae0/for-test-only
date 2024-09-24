import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义策略网络和价值网络
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


# PPO 超参数
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    # 计算折扣回报
    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    # 更新策略
    def update(self, memory):
        states, actions, old_log_probs, rewards, dones = memory
        returns = self.compute_returns(rewards, dones, 0)  # 假设最终状态值为 0

        for _ in range(self.k_epochs):
            # 计算优势函数
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            action_probs, state_values = self.model(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # 计算比率（新策略与旧策略的比率）
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = returns - state_values.squeeze()

            # PPO 损失函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values.squeeze(), returns) - 0.01 * entropy

            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


# Memory 用于存储一回合的经验
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []


# 主训练函数
def train_ppo(env_name, episodes=1000, max_timesteps=200):
    env = gym.make(env_name)
    state_dim = len(env.reset())  # 自动获取状态维度
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)
    memory = Memory()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = ppo.model(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()

            next_state, reward, done, _ = env.step(action)
            memory.add(state, action, log_prob, reward, done)
            state = next_state
            total_reward += reward

            if done or t == max_timesteps - 1:
                ppo.update((memory.states, memory.actions, memory.log_probs, memory.rewards, memory.dones))
                memory.clear()
                break

        print(f'Episode {episode+1}/{episodes}, Total Reward: {total_reward}')



if __name__ == "__main__":
    train_ppo('CartPole-v1')  # 使用CartPole环境进行训练
