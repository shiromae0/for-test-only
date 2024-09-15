import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

"""
This class is the foundation of the whole learning task. It not only defines the dynamics of the task 
(how to change the state according to the action), the goal(when the task is considered complete) and 
feedback (reward mechanism), but also provides a way to reset the environment to the initial state, 
which is neccessary for multiple independent training rounds.
"""
class Machine:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.status = 'idle'
        
    def shift_status(self):
        if self.status == 'idle':
            self.status = 'active'
        else:
            self.status = 'idle'

class Miner(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
    
class Cutter(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)

class Combiner(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)

class ShapezEnv:
    def __init__(self):
        self.grid_size =(10, 10) # TODO 网格大小
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.mach_type_to_id = {
            'miner': 1,
            'cutter': 2,
            'combiner': 3
        }
        self.machines = {}
        self.miner_position = None
        self.conveyor_position = []
        self.target_shape = 'circle'
        self.extracted_shape = None #提取的形状
        self.goal_reached = False
        self.product_goals = []

    def check_goals(self):
        for shape, quant in self.product_goals:
            if 
    def reset(self):
        # 重置环境到初始状态
        self.machine.clear()
        self.grid = np.zeros(self.grid_size)
        self.miner_position = None
        self.conveyor_position = []
        self.extracted_shape = None
        self.goal_reached = False
        self.initial_conditions()

        return self.get_state()
    
    def initial_conditions(self):
        self.product_goals.append(('circle', 10))
        self.resources['circle'] = 0

    def get_state(self):
        # 返回一个当前网格状态的副本，避免外部修改影响内部状态
        return np.copy(self.grid).flatten()
    
    def place_machine(self, machine_type, position, direction):
        if position not in self.machines:
            if machine_type == 'miner':
                self.machines[position] = Miner(position, direction)
            elif machine_type == 'cutter':
                self.machines[position] = Cutter(position, direction)
            elif machine_type == 'combiner':
                self.machines[position] = Combiner(position, direction)
            else:
                return False
            return True
        return False
    
    def step(self, action_type, **params):
        if action_type == 'place_machine':
            result = self.place_machine(params['machine_type'], params['position'], params['direction'])
        elif action_type == 'check_shape':
            #TODO
        # Assume more actions are handled here

        return self.get_state(), -1, False

        
"""    
class GridWorld:
    def __init__(self):
        self.size = 5  # Define the grid size
        self.start = (1, 1)  
        self.goal = (4, 4)  
        self.reset() 

    # Initialize the position of the agent and return to the initial state.
    def reset(self):
        self.position = self.start  
        return self.get_state()

    # Returns a flattened array of the current environment state, where the current position of the agent is marked as 1
    def get_state(self):
        state = np.zeros((self.size, self.size))
        x, y = self.position
        state[x][y] = 1
        return state.flatten()  
    
    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0:  # move up
            x -= 1
        elif action == 1 and x < self.size - 1:  # move down
            x += 1
        elif action == 2 and y > 0:  # move left
            y -= 1
        elif action == 3 and y < self.size - 1:  # move right
            y += 1

        self.position = (x, y)
        done = self.position == self.goal  
        reward = 1 if done else -1  # Reward 1 for reaching the end, otherwise reward -1
        return self.get_state(), reward, done

    def render(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[grid == ''] = '.'
        x, y = self.position
        grid[x][y] = 'A'  # Represent the agent
        gx, gy = self.goal
        grid[gx][gy] = 'G'  # Represent the goal
        print("\n".join([" ".join(row) for row in grid]))
"""

"""

"""
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(100, 64)  # The first layer
        self.fc2 = nn.Linear(64, 32)   # output layer
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        # 应用ReLU激活函数增加非线性，帮助网络学习复杂的函数关系。
        x = torch.relu(self.fc1(x))
        # 计算每个动作的Q值，输出层不应用激活函数，直接输出实数值。
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    env = ShapezEnv()
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = deque(maxlen=1000)
    episodes = 500
    gamma = 0.9
    epsilon = 1.0 # 开始时随机探索
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

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
                epsilon = max(min_epsilon, epsilon * epsilon_decay) # 随着时间的推移降低探索率，让模型更多地依赖于其学习到的经验
                print(f'Episode {episode}: Total reward {total_reward}, Epsilon {epsilon}')

train()

