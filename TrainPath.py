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

# target = 11

# class Machine:
#     def __init__(self, position, direction):
#         self.position = position
#         self.direction = direction
#         self.status = 'idle'
        
#     def shift_status(self):
#         if self.status == 'idle':
#             self.status = 'active'
#         else:
#             self.status = 'idle'

# class Miner(Machine):
#     def __init__(self, position, direction):
#         super().__init__(position, direction)
    
# class Cutter(Machine):
#     def __init__(self, position, direction):
#         super().__init__(position, direction)

# class Combiner(Machine):
#     def __init__(self, position, direction):
#         super().__init__(position, direction)

# class ShapezEnv:
#     def __init__(self):
#         self.grid_size =(10, 10) # TODO 网格大小
#         self.grid = np.zeros(self.grid_size, dtype=int)
#         self.mach_type_to_id = {
#             'miner': 1,
#             'cutter': 2,
#             'combiner': 3

#         }

#         self.machines = {}
#         self.miner_position = None
#         self.conveyor_position = []
#         self.target_shape = 'circle'
#         self.extracted_shape = None #提取的形状
#         self.goal_reached = False
#         self.product_goals = []

#     def check_goals(self):
#         for shape, quant in self.product_goals:
#             if 
#     def reset(self):
#         # 重置环境到初始状态
#         self.machine.clear()
#         self.grid = np.zeros(self.grid_size)
#         self.miner_position = None
#         self.conveyor_position = []
#         self.extracted_shape = None
#         self.goal_reached = False
#         self.initial_conditions()

#         return self.get_state()
    
#     def initial_conditions(self):
#         self.product_goals.append(('circle', 10))
#         self.resources['circle'] = 0

#     def get_state(self):
#         # 返回一个当前网格状态的副本，避免外部修改影响内部状态
#         return np.copy(self.grid).flatten()
    
#     def place_machine(self, machine_type, position, direction):
#         if position not in self.machines:
#             if machine_type == 'miner':
#                 self.machines[position] = Miner(position, direction)
#             elif machine_type == 'cutter':
#                 self.machines[position] = Cutter(position, direction)
#             elif machine_type == 'combiner':
#                 self.machines[position] = Combiner(position, direction)
#             else:
#                 return False
#             return True
#         return False
    
#     def place_belt(self, start, end):


    
#     def step(self, action_type, **params):
#         if action_type == 'place_machine':
#             result = self.place_machine(params['machine_type'], params['position'], params['direction'])
#         elif action_type == 'check_shape':
#             #TODO
#         # Assume more actions are handled here

#         return self.get_state(), -1, False


resource = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 15, 0, 0, 12, 12, 12, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 0, 0, 0, 12, 12, 12, 0, 0],
    [0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 0, 0, 12, 12, 0, 0],
    [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
    [0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 15],
    [0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 11, 11, 0, 0, 0, 15, 0, 15, 15, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 11, 11, 11, 0, 0, 0, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
build = np.full((15, 24), -1)
build[7:9, 12:14] = 21  # 在第8-9行，第13-14列设置为21
build[0,0] = 5
        
class GridWorld:
    def __init__(self, start, end, resource_map, build_map):
        self.shape = resource_map.shape
        self.start = start
        self.goal = end
        self.resource_map = resource_map
        self.build_map = build_map
        self.path = []  # 用于存储路径
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
        reward = -1  # 默认每步的负奖励
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
def train(start,end,res,build):
    env = GridWorld(start,end,res,build)
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = deque(maxlen=1000)
    episodes = 100
    gamma = 0.9
    epsilon = 1.0 # 开始时随机探索
    epsilon_decay = 0.995
    min_epsilon = 0.01
    done_flag = False
    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        done = False
        max_steps = 5000
        step = 0
        while not done and step <max_steps:
            step+=1
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()
            next_state, reward, done = env.step(action)
            if done == True:
                done_flag = True
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if step >= max_steps and done_flag == False:
                return False
            else:
                print(step)
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
    return env.get_path()


print(train((5,5),(10,3),resource,build))