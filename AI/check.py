import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class GridWorld:
    def __init__(self, start, end, resource_map, build_map):
        self.shape = resource_map.shape
        self.start = start
        self.goal = end
        self.resource_map = resource_map
        self.build_map = build_map
        self.path = []  # 用于存储路径
        self.last_choice = -1
        self.reset()

    # Initialize the position of the agent and return to the initial state.
    def reset(self):
        self.position = self.start
        self.path = [self.position]  # 重置时记录初始位置
        return self.get_state()

    # Returns a flattened array of the current environment state, where the current position of the agent is marked as 1
    def get_state(self):
        state = np.zeros((self.shape[0], self.shape[1]))
        x, y = self.position
        state[x][y] = 1
        return state.flatten()

    def step(self, action):
        x, y = self.position
        old_position = self.position
        if action == 0 and x > 0 and self.resource_map[x-1][y] == 0 and (self.build_map[x-1][y] == -1 or self.build_map[x-1][y] == 21):  # move up
            x -= 1
        elif action == 1 and x < self.shape[0] - 1 and self.resource_map[x + 1][y] == 0 and (self.build_map[x + 1][y] == -1 or self.build_map[x + 1][y] == 21):  # move down
            x += 1
        elif action == 2 and y > 0 and self.resource_map[x][y - 1] == 0 and (self.build_map[x][y - 1] == -1 or self.build_map[x][y - 1] == 21):  # move left
            y -= 1
        elif action == 3 and y < self.shape[1] - 1 and self.resource_map[x][y + 1] == 0 and (self.build_map[x][y + 1] == -1 or self.build_map[x][y + 1] == 21):  # move right
            y += 1
        self.position = (x, y)
        self.path.append(self.position)  # 每次移动都记录路径

        done = self.position == self.goal
        old_distance = np.linalg.norm(np.array(old_position) - np.array(self.goal))
        new_distance = np.linalg.norm(np.array(self.position) - np.array(self.goal))
        reward = -2  # 默认每步的负奖励
        if self.last_choice == action:
            reward+=1
        if new_distance < old_distance:
            reward += 1  # 如果接近目标，则增加奖励
        if self.position == self.goal:
            reward += 50  # 到达目标时的大奖励
        return self.get_state(), reward, done

    def render(self):
        grid = np.zeros((self.shape[0], self.shape[1]), dtype=str)
        grid[grid == ''] = '.'
        x, y = self.position
        grid[x][y] = 'A'  # Represent the agent
        gx, gy = self.goal
        grid[gx][gy] = 'G'  # Represent the goal
        print("\n".join([" ".join(row) for row in grid]))

    def get_path(self):
        """返回智能体走过的路径"""
        return self.path

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(360, 64)  # Adjust input features to match your state size
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(start, end, res, build, train_model=True):  # 增加 train_model 参数
    env = GridWorld(start, end, res, build)

    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = deque(maxlen=1000)

    # 尝试加载已保存的模型
    try:
        model.load_state_dict(torch.load('dqn_model.pth'))
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found.")
        if not train_model:
            print("Model not trained yet, and train_model is False. Cannot proceed.")
            return False, []

    # 如果不需要训练，只进行推理
    if not train_model:
        print("Skipping training, directly using the loaded model for inference.")
        return inference(model, env)  # 调用推理函数

    # 继续训练模型的部分（如前面的代码逻辑）
    episodes = 500
    gamma = 0.9
    epsilon = 1.0  # 开始时随机探索
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        done = False
        max_steps = 50000
        step = 0

        while not done and step < max_steps:
            step += 1
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if step >= max_steps:
                return False, []

            if len(memory) > 20:
                batch = random.sample(memory, 20)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

                batch_state = torch.cat(batch_state)
                batch_next_state = torch.cat(batch_next_state)
                batch_reward = torch.FloatTensor(batch_reward)
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                batch_done = torch.FloatTensor(batch_done)

                current_qs = model(batch_state).gather(1, batch_action).squeeze(1)
                next_qs = model(batch_next_state).max(1)[0]
                targets = batch_reward + gamma * next_qs * (1 - batch_done)

                loss = nn.MSELoss()(current_qs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                print(f'Episode {episode}: Total reward {total_reward}, Epsilon {epsilon}, Loss {loss.item()}')

        if episode % 250 == 0:
            torch.save(model.state_dict(), 'dqn_model.pth')
            print(f"Model saved at episode {episode}")

    torch.save(model.state_dict(), 'dqn_model.pth')
    print("Training completed, final model saved.")
    return True, env.get_path()


# 推理函数，使用已加载的模型进行推理
def inference(model, env):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0
    done = False
    step = 0
    max_steps = 500

    while not done and step < max_steps:
        step += 1
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        state = next_state
        total_reward += reward

    print(f"Inference completed. Total reward: {total_reward}")
    return True, env.get_path()
