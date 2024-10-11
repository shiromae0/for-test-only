from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
import math
from collections import deque

class Machine:
    def __init__(self, position, direction = -1):
        self.position = position
        self.direction = direction
        self.type = -1
class Miner(Machine):
    def __init__(self, position, direction,shape):
        super().__init__(position, direction)
        self.type = 22
        self.is_conveyed = False #flag for checking if the resource is successfully conveyed to the destination
        self.direction = direction
        self.shape = shape
class Cutter(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 23
class Trash(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 24
class Hub(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 21
class Conveyor(Machine):
    """
    direction
    #define UP 1
    #define DOWN 2
    #define LEFT 3
    #define RIGHT 4
    #define UP_LEFT 5
    #define UP_RIGHT 6
    #define DOWN_LEFT 7
    #define DOWN_RIGHT 8
    #define LEFT_UP 9
    #define RIGHT_UP 10
    #define LEFT_DOWN 11
    #define RIGHT_DOWN 12
    """
    def __init__(self,position,direction):
        super().__init__(position,direction)
        self.type = 31
        self.direction = direction


class ShapezEnv(gymnasium.Env):


    def __init__(self, build, res, target_shape):
        self.max_step = 500
        self.steps = 0
        self.original_bld = np.array(build)
        self.grid_rsc = np.array(res)
        self.grid_bld = self.original_bld.copy()

        self.machines = []
        self.target_shape = target_shape

        grid_shape = self.grid_rsc.shape
        #define valid actions, return the dict:{(machine_type,direction),[pos1,pos2])....}
        self.create_action_space()
        # 定义观测空间'
        grid_obs_space = spaces.Box(low=0, high=3300, shape=(grid_shape), dtype=np.int32)
        stage_obs_space = spaces.Discrete(4)
        low = np.concatenate([grid_obs_space.low.flatten(), [0]])
        high = np.concatenate([grid_obs_space.high.flatten(), [stage_obs_space.n - 1]])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = 0
        self.path = []
    def find_closet_hub(self,cur_pos):
        min_distance = 0x3f3f3f
        closet_pos = None
        for machine in self.machines:
            if machine.type == 21:
                hub_x = machine.position[0]
                hub_y = machine.position[1]
                distance = (cur_pos[0]-hub_x)**2 + (cur_pos[1]-hub_y)**2
                if distance < min_distance:
                    if (self.grid_bld[hub_x-1][hub_y] == -1 or self.grid_bld[hub_x + 1][hub_y] == -1 or
                            self.grid_bld[hub_x][hub_y-1] == -1 or self.grid_bld[hub_x][hub_y + 1] == -1):
                        min_distance = distance
                        closet_pos = (hub_x,hub_y)
        return closet_pos
    def bfs(self,start, direct,goal):
        start_x, start_y = self._get_next_position(start,direct)
        if self.grid_bld[start_x][start_y] != -1:
            return []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([(start_x, start_y)])
        visited = set()
        visited.add((start_x, start_y))
        visited.add(start)
        prev = {(start_x,start_y):start}
        prev[start] = None
        while queue:
            current = queue.popleft()  # 取出队列中的第一个节点
            if current == goal:
                return self.reconstruct_path(prev, start, goal)

            # 遍历四个方向
            for direction in directions:
                new_x, new_y = current[0] + direction[0], current[1] + direction[1]

                # 检查新位置是否在地图范围内，并且是可以通过的路径
                if (0 <= new_x < self.grid_bld.shape[0] and 0 <= new_y < self.grid_bld.shape[1] and
                        (self.grid_bld[new_x][new_y] == -1 or self.grid_bld[new_x][new_y]//100 == 21)):
                    new_pos = (new_x, new_y)
                    if new_pos not in visited:
                        visited.add(new_pos)
                        queue.append(new_pos)
                        prev[new_pos] = current  # 记录前驱节点
        return []

    def reconstruct_path(self,prev, start, end):
        """
        根据前驱节点字典重建从起点到终点的路径
        :param prev: 前驱节点字典
        :param start: 起点
        :param end: 终点
        :return: 重建的路径
        """
        path = []
        at = end
        while at is not None:
            path.append(at)
            at = prev[at]
        path.reverse()
        return path if path[0] == start else []

    def check_direction(self,pre,cur):
        if cur[0] - pre[0] == -1:#up
            return 1
        if cur[0]-pre[0] == 1:#down
            return 2
        if cur[1] - pre[1] == -1:#left
            return 3
        if cur[1] - pre[1] == 1:#right
            return 4
    def handle_direction(self,pre,cur,nxt):
        if self.check_direction(pre,cur) == self.check_direction(cur,nxt):
            return self.check_direction(pre,cur)
        else:
            if self.check_direction(pre,cur) == 1:
                if self.check_direction(cur, nxt) == 3:
                    return 5
                else:
                    return 6
            elif self.check_direction(pre,cur) == 2:
                if self.check_direction(cur, nxt) == 3:
                    return 7
                else:
                    return 8
            elif self.check_direction(pre,cur) == 3:
                if self.check_direction(cur, nxt) == 1:
                    return 9
                else:
                    return 11
            elif self.check_direction(pre,cur) == 4:
                if self.check_direction(cur, nxt) == 1:
                    return 10
                else:
                    return 12
    def place_conveyor(self):
        pre_pos = None
        next_pos = None
        for path in self.path:
            for i,pos in enumerate(path):
                x, y = pos
                if i < len(path) - 1:
                    if self.grid_bld[x][y] // 100 == 22:  # 当前位置是miner
                        pre_pos = pos # start
                    elif self.grid_bld[x][y] == -1:
                        next_pos = path[i + 1]
                        direction = self.handle_direction(pre_pos, pos,next_pos)  # 计算方向
                        pre_pos = pos  # 更新 pre_pos
                        self.grid_bld[x][y] = 31 * 100 + direction  # 设置传送带的方向
                    else:
                        return

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        if options is not None:
            pass
        # reset the env
        self.steps = 0
        self.grid_bld = self.original_bld.copy()  # 确保不会修改原始建筑物矩阵
        self.state = 0
        self.path.clear()
        self.machines.clear()
        hub_positions = np.argwhere(self.grid_bld//100 == 21)
        for pos in hub_positions:
            position = tuple(pos)
            hub = Hub(position=position, direction=0)
            self.machines.append(hub)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        grid_obs = self.grid_bld
        stage_obs = self.state
        return np.concatenate([grid_obs.flatten(), [stage_obs]])
    def create_valid_action_space(self):
        """
        创建有效的动作空间，机器类型和对应方向的组合。
        0 -> 移除（无方向限制）
        1 -> 传送带（12个方向）
        2 -> 矿机（4个方向）
        """
        action_spaces = {}
        all_pos  = []
        res_pos = np.argwhere(self.grid_rsc != 0)


        # handle the miner action spaces

        for direction in range(4):
            valid_action = (22, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(res_pos)


        #handle the conveyor belt action spaces
        for direction in range(12):
            valid_action = (31, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(all_pos)
        for index in np.ndindex(self.grid_rsc.shape):
            all_pos.append(index)

        #handle the remove action spaces
        action_spaces[(0, -1)] = []
        action_spaces[(0, -1)].extend(all_pos)
        return action_spaces

    def create_action_space(self):
        self.valid_actions = self.create_valid_action_space()
        self.action_list = [
            (action_type, pos)
            for action_type, positions in self.valid_actions.items()
            for pos in positions
        ]
        # print(self.action_list)
        self.action_space = spaces.Discrete(len(self.action_list))
        return

    def CanPlaceConveyor(self, position: Tuple[int, int], direction: int) -> bool:
        # direction
        # define UP 1
        # define DOWN 2
        # define LEFT 3
        # define RIGHT 4
        # define UP_LEFT 5
        # define UP_RIGHT 6
        # define DOWN_LEFT 7
        # define DOWN_RIGHT 8
        # define LEFT_UP 9
        # define LEFT_DOWN 10
        # define RIGHT_UP 11
        # define RIGHT_DOWN 12
        x, y = position
        pre_pos = None
        if direction <= 4:
            return True
        elif direction == 5 or direction == 6:
            pre_pos = (x + 1, y)
            if pre_pos in self.machines and isinstance(self.machines[pre_pos], Conveyor) and x + 1 < self.grid_rsc.shape[0]:
                next_conveyor_direction = self.machines[pre_pos].direction
                if next_conveyor_direction in [1, 9, 11]:
                    return True
            return False
        elif direction == 7 or direction == 8:
            pre_pos = (x - 1, y)
            if pre_pos in self.machines and isinstance(self.machines[pre_pos], Conveyor) and x - 1 >= 0 :
                next_conveyor_direction = self.machines[pre_pos].direction
                if next_conveyor_direction in [2, 10, 12]:
                    return True
            return False
        elif direction == 9 or direction == 10:
            pre_pos = (x , y + 1)
            if pre_pos in self.machines and isinstance(self.machines[pre_pos], Conveyor) and y + 1 < self.grid_rsc.shape[1]:
                next_conveyor_direction = self.machines[pre_pos].direction
                if next_conveyor_direction in [3,5,7]:
                    return True
            return False
        elif direction == 11 or direction == 12:
            pre_pos = (x , y - 1)
            if pre_pos in self.machines and isinstance(self.machines[pre_pos], Conveyor) and y - 1 >= 0:
                next_conveyor_direction = self.machines[pre_pos].direction
                if next_conveyor_direction in [4,6,8]:
                    return True
            return False
        # 默认不允许放置
        return False
    def CanPlaceMiner(self,position):
        if self.grid_rsc[position] == 0 or self.grid_bld[position] != -1:
            return False
        else:
            return True
    def extract_buildings(self, position):
        #extract the machine type and direction in the specific position
        machine_type = self.grid_bld[position] // 100
        direction = self.grid_bld[position] % 100
        return machine_type,direction
    def CanRemove(self,position):
        if self.grid_bld[position] != -1 and self.grid_bld[position]//100 != 21:
            return True
        return False

    def get_prepos(self,position,direction):
        if direction == 1 or direction == 5 or direction == 6:
            return (position[0]+1,position[1])
        elif direction == 2 or direction == 7 or direction == 8:
            return (position[0]-1,position[1])
        elif direction == 3 or direction == 9 or direction == 11:
            return (position[0],position[1]+1)
        elif direction == 4 or direction == 10 or direction == 12:
            return (position[0],position[1]-1)
        else:
            return None

    def check_miner(self) -> bool:
        #check if the nearest miner is set successfully and if the number of miner is
        #enough for the goal
        #return:True if the miner is no more need to considerate
        miner_cnt = 0
        for machine in self.machines:
            if isinstance(machine,Miner):
                miner_cnt += 1
        if miner_cnt >= 1:
            return True
        else:
            return False
    def check_conveyor(self) -> bool:
        return True



    def _get_next_position(self, position: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """根据传送带的方向获取下一个位置"""
        """
           direction
           #define UP 1
           #define DOWN 2
           #define LEFT 3
           #define RIGHT 4
           #define UP_LEFT 5
           #define UP_RIGHT 6
           #define DOWN_LEFT 7
           #define DOWN_RIGHT 8
           #define LEFT_UP 9
           #define RIGHT_UP 10
           #define LEFT_DOWN 11
           #define RIGHT_DOWN 12
           """
        x, y = position
        if (direction == 1 or direction == 9 or direction == 10) and x - 1 >= 0:  # 向上
            return (x - 1, y)
        elif (direction == 4 or direction == 6 or direction == 8) and y + 1 < self.grid_rsc.shape[1]:  # 向右
            return (x, y + 1)
        elif (direction == 2 or direction == 11 or direction == 12) and x + 1 < self.grid_rsc.shape[0]:  # 向下
            return (x + 1, y)
        elif (direction == 3 or direction == 5 or direction == 7) and y - 1 >= 0:  # 向左
            return (x, y - 1)
        return position

    def check_goal(self) -> bool:
        """
        检查是否有资源从矿机通过传送带等路径成功到达 hub，并符合目标形状。
        """
        shape = 0
        for path in self.path:
            for pos in path:
                if self.grid_bld[pos] // 100 == 22:
                    shape = self.grid_rsc[pos]
                if self.grid_bld[pos] // 100 == 21:
                    if shape == self.target_shape:
                        return True

    def step(self, action):
        self.steps += 1
        action_type, selected_position = self.action_list[action]
        machine_type = action_type[0]
        direction = action_type[1]
        x,y = selected_position
        reward = 0.0
        done = False
        truncated = False  # 添加 truncated 标记
        info = {}
        # 如果达到最大步数，标记为 truncated
        if self.steps >= self.max_step:
            print("Truncated")
            truncated = True
            done = False# 或者也可以直接标记为 done
            return self._get_obs(), reward, done, truncated, info

        reward = self.handle_place_event(machine_type,(x,y),direction)
        if self.check_goal() == True:
            print("Done")
            done = True  # 如果达到目标状态，标记为完成
            reward += 500
            print(self.path)
            info['path'] = []
            for path in self.path:
                for pos in path:
                    info['path'].append(pos)
            print(info)
        return self._get_obs(), reward, done, truncated,info

    def calculate_miner_reward(self,position,direction):
        hub_pos = self.machines[0].position
        x_pos = True if position[0] < hub_pos[0] else False
        y_pos = True if position[1] < hub_pos[1] else False
        distance = (hub_pos[0] - position[0]) ** 2 + (hub_pos[1] - position[1]) ** 2
        max_distance = (self.grid_bld.shape[0] ** 2) * 2
        distance_reward = max_distance - distance
        normalized_reward = distance_reward / max_distance
        distance_reward = normalized_reward * 20
        direction_reward = 0
        # define UP 1
        # define DOWN 2
        # define LEFT 3
        # define RIGHT 4
        if x_pos == True and y_pos == True:
            if direction == 1 or direction == 4:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos == False and y_pos == True:
            if direction == 1 or direction == 3:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos == True and y_pos == False:
            if direction == 2 or direction == 4:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos == False and y_pos == False:
            if direction == 2 or direction == 3:
                direction_reward = 10
            else:
                direction_reward = -10

        return distance_reward + direction_reward
    def handle_place_event(self,machine_type,position,direction):
            #handle the place event
            #param:machine_type:the number of the machine
            #param:position:the place that we want put the machine
            #param:direction:the machine's direction
            #return:Canplace: to show if we can handle the action successfully
            #return:reward:the reward of the action
            reward = 0
            if self.state == 0:
                if machine_type == 22:
                    if self.CanPlaceMiner(position):
                        dest = self.find_closet_hub(position)
                        if not (self.bfs(position, direction,dest) or self.grid_bld[position] != self.target_shape):
                            print("fail")
                            reward = -20
                        else:
                            self.grid_bld[position] = 2200 + direction
                            new_machine = Miner(position,direction,self.grid_rsc[position])
                            self.machines.append(new_machine)
                            self.state = 1
                            self.path.append(self.bfs(position,direction,dest))
                            self.place_conveyor()
                            reward += self.calculate_miner_reward(position,direction)
                    else:
                        reward = -10
                else:
                    if machine_type == 0:
                        if self.CanRemove(position):
                            reward = 0
                            self.grid_bld[position] = -1
                        else:
                            reward = -5
                    else:
                        reward = -3
                return reward
            elif self.state == 1:
                return reward


