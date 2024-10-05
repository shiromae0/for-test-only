from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

class Machine:
    def __init__(self, position, direction = -1):
        self.position = position
        self.direction = direction
        self.type = -1
class Miner(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 22
        self.is_conveyed = False #flag for checking if the resource is successfully conveyed to the destination
        self.direction = direction
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
        self.grid_direct = np.full(self.grid_bld.shape, -1)

        self.machines = {}
        self.target_shape = target_shape

        # 获取网格的大小
        grid_shape = self.grid_rsc.shape

        # 定义有效的动作组合（机器类型 + 对应的方向）
        self.valid_actions = self._create_valid_action_space()

        # 定义动作空间为 MultiDiscrete，但我们会根据 machine_type 来过滤方向
        # 这里假设 grid_shape 的大小是 15x24
        self.action_space = spaces.MultiDiscrete([
            len(self.valid_actions),  # 有效的动作组合
            grid_shape[0],  # x 位置 (网格的行)
            grid_shape[1]  # y 位置 (网格的列)
        ])

        # 定义观测空间
        self.observation_space = spaces.Box(
            low=0,
            high=np.max([np.max(self.grid_rsc), np.max(self.grid_bld), np.max(self.grid_direct)]),
            shape=(grid_shape[0], grid_shape[1]),  # 2 表示堆叠了 2 个网格
            dtype=np.int32
        )


    def _get_obs(self) -> np.ndarray:
        return self.grid_bld

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
        if self.grid_rsc[position] == 0:
            return False
        else:
            return True

    def extract_buildings(self, position):
        #extract the machine type and direction in the specific position
        machine_type = self.grid_bld[position] // 100
        direction = self.grid_direct[position] % 100
        return machine_type,direction


    def CanRemove(self,position):
        if self.grid_bld[position] != -1 and self.grid_bld[position]//100 != 21:
            return True
        return False
    
    def handle_place(self,machine_type,position,direction):
        #handle the place event
        #param:machine_type:the number of the machine
        #param:position:the place that we want put the machine
        #param:direction:the machine's direction
        #return:Canplace: to show if we can handle the action successfully
        #return:reward:the reward of the action
        new_machine = Machine
        Canplace = False
        reward = 0
        if machine_type == 0: #action is remove
            Canplace = self.CanRemove(position)
            if Canplace:# can remove the building in position
                del self.machines[position]
                self.grid_bld[position] = -1
                reward = -3
            else:#no building in the position, invalid action
                reward = -50
            return Canplace,reward
        else:
            if position in self.machines: #cannot place the building if the position already exist a building
                Canplace = False
                reward = -50
                return Canplace,reward
            if machine_type == 31:
                Canplace = self.CanPlaceConveyor(position, direction)
                new_machine = Conveyor(position,direction)
            if machine_type == 22:
                Canplace = self.CanPlaceMiner(position)
                new_machine = Miner(position,direction)
        if Canplace:
            self.machines[position] = new_machine
            self.grid_bld[position] = machine_type * 100 + direction
            reward = 5
        else:
            reward = -50
        return Canplace,reward


    def check_goal(self) -> bool:
        """
        检查是否有资源从矿机通过传送带等路径成功到达 hub，并符合目标形状。
        """
        for position, machine in self.machines.items():
            if isinstance(machine, Miner):  # 找到矿机，开始从资源生成点追踪
                current_position = position
                current_machine = machine
                current_shape = self.grid_rsc[position]  # 获取资源的初始形状
                positions = []
                while True:
                    if current_position in positions:
                        return False
                    positions.append(current_position)
                    # 获取下一个位置，根据传送带的方向前进
                    if current_position == self._get_next_position(current_position, current_machine.direction):
                        return False
                    next_position = self._get_next_position(current_position, current_machine.direction)
                    #print(current_position,next_position,machine.direction,machine.type)
                    # 检查下一个位置是否有机器
                    if next_position in self.machines:
                        current_machine = self.machines[next_position]
                        if isinstance(current_machine, Conveyor):
                            # 传送带：继续前进
                            current_position = next_position
                        elif isinstance(current_machine, Hub):
                            # 检查资源是否到达 hub，并且形状是否符合目标
                            if current_shape == self.target_shape:
                               # print(f"资源从 {position} 成功到达 hub，形状符合目标")
                                return True
                            else:
                                #print(f"资源到达 hub，但形状不符合目标 {self.target_shape}")
                                return False
                        else:
                            #print("遇到了其他建筑，停止")
                            return False
                    else:
                        # 如果路径中断或没有传送带，停止追踪
                        #print("资源路径中断")
                        break
                #print()
        return False

    def compute_resource_utilization(self):
        # 计算资源的利用率
        self.total_useful_resource = np.argwhere(self.grid_rsc == self.target_shape)
        # 检查资源是否成功被传送到终点
        for machine in self.machines.values():
            if isinstance(machine, Miner):
                if machine.is_conveyed:
                    self.processed_resource += 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        if options is not None:
            pass

        # reset the env
        self.steps = 0
        self.grid_bld = self.original_bld.copy()  # 确保不会修改原始建筑物矩阵
        self.grid_direct = np.full(self.grid_bld.shape, -1)  # 重置 grid_direct
        self.processed_resource = 0
        self.total_useful_resource = 0
        # 清空 machines 字典
        self.machines.clear()
        # 找到所有 Hub（值为 21 的位置）
        hub_positions = np.argwhere(self.grid_bld//100 == 21)
        # 将这些位置的 Hub 添加到 machines 字典中
        for pos in hub_positions:
            position = tuple(pos)  # 将数组转换为 tuple 类型的坐标
            hub = Hub(position=position, direction=0)  # 假设 direction 默认为 0
            self.machines[position] = hub  # 添加 Hub 对象到 machines 字典
        # 返回环境的初始观察值（obs）和一个空字典
        obs = self._get_obs()
        return obs, {}


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

    def _create_valid_action_space(self):
        """
        创建有效的动作空间，机器类型和对应方向的组合。
        0 -> 移除（无方向限制）
        1 -> 传送带（12个方向）
        2 -> 矿机（4个方向）
        """
        valid_actions = []
        # 机器类型 0 -> 移除，无方向
        valid_actions.append((0, -1))  # 移除不需要方向

        # 机器类型 1 -> 传送带，有 12 个方向
        for direction in range(12):
            valid_actions.append((31, direction+1))

        # 机器类型 2 -> 矿机，有 4 个方向
        for direction in range(4):
            valid_actions.append((22, direction+1))

        return valid_actions

    def step(self, action):
        self.steps += 1
        action_idx, x, y = action
        machine_type, direction = self.valid_actions[action_idx]  # 获取机器类型和方向
        #print(machine_type,direction,x,y)
        reward = 0.0
        done = False
        truncated = False  # 添加 truncated 标记
        info = {}
        # 如果达到最大步数，标记为 truncated
        if self.steps >= self.max_step:
            truncated = True
            done = False# 或者也可以直接标记为 done
            return self._get_obs(), reward, done, truncated, info
        CanPlace,reward = self.handle_place(machine_type,(x,y),direction)
        if self.check_goal() == True:
            done = True  # 如果达到目标状态，标记为完成
            reward += 200
            for key,machine in self.machines.items():
                print(key,machine.direction)
        # 返回观察值、奖励、是否结束、是否被截断和信息
        return self._get_obs(), reward, done, truncated,info

# resource = np.array([
#     [0,0,0,11],
#     [0,0,0,11],
#     [0,0,0,0],
#     [0,0,0,0]
# ])
# build = np.array([[-1, -1, 31, 22],
#                    [-1, -1, -1, 31],
#                    [31, -1, 31, 31],
#                    [31, -1, -1, 21]])
# dir  = np.array([[-1, -1,  4,  2],
#                    [-1, -1, -1,  1],
#                    [ 1, -1,  4,  3],
#                    [ 2, -1, -1, -1]])
# env = ShapezEnv(build, resource, target_shape=11)
# env.reset()
# env.grid_direct = dir
# env.machines[0,3] = Miner((0,3),2)
# env.machines[0,2] = Conveyor((0,2),4)
# env.machines[1,3] = Conveyor((1,3),1)
# env.machines[2,2] = Conveyor((2,2),4)
# env.machines[2,3] = Conveyor((2,3),3)
# env.machines[2,0] = Conveyor((2,0),1)
# env.machines[3,0] = Conveyor((3,0),2)
# print(env.check_goal())