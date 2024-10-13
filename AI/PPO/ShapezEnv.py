from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from Machine import Machine,Miner,Hub,Trash,Conveyor,Cutter


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)

class ShapezEnv(gymnasium.Env):
    def __init__(self, build, res, target_shape):
        self.max_step = 100
        self.steps = 0
        self.original_bld = np.array(build)
        self.path = []
        self.path_num = 0
        self.grid_rsc = np.array(res)
        self.grid_bld = self.original_bld.copy()
        self.reward_grid = np.full(self.grid_bld.shape, -1)
        self.machines = {}
        self.target_shape = target_shape
        self.total_reward = 0
        # 获取网格的大小
        grid_shape = self.grid_rsc.shape

        # 定义有效的动作组合（机器类型 + 对应的方向）
        self.create_action_space()
        self.act_dict = {(action, tuple(pos)): idx for idx, (action, pos) in enumerate(self.action_list)}
        self.last_action_index = -1
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0,
                high=np.max([np.max(self.grid_rsc), np.max(self.grid_bld)]),
                shape=(grid_shape[0], grid_shape[1]),
                dtype=np.int32
            ),
            'last_action': spaces.Box(
                low=np.array([0, 0, 0, 0]),  # 动作空间的最小值（机器型号、方向、坐标）
                high=np.array([31, 12, grid_shape[0]-1, grid_shape[1]-1]),  # 最大值
                shape=(4,),  # 机器型号、机器方向、横坐标、纵坐标
                dtype=np.int32
            )
        })

    def _get_obs(self):
        self.last_action = self.action_list[self.last_action_index]

        act_type = self.last_action[0][0]
        direct = self.last_action[0][1]
        pos = self.last_action[1]
        observation = {
            'grid': self.grid_bld,  # 当前网格数据
            'last_action':np.array([act_type,direct,pos[0],pos[1]])   # 上一个动作的位置索引
        }
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        if options is not None:
            pass
        # reset the env
        self.last_action_index = -1
        self.total_reward = 0
        self.reward_grid = np.full(self.grid_bld.shape, -1)
        self.steps = 0
        self.grid_bld = self.original_bld.copy()  # 确保不会修改原始建筑物矩阵
        self.grid_direct = np.full(self.grid_bld.shape, -1)  # 重置 grid_direct
        self.processed_resource = 0
        self.total_useful_resource = 0
        # 清空 machines 字典
        self.machines.clear()
        self.path_num = 0
        self.path.clear()
        # 找到所有 Hub（值为 21 的位置）
        hub_positions = np.argwhere(self.grid_bld // 100 == 21)
        # 将这些位置的 Hub 添加到 machines 字典中
        for pos in hub_positions:
            position = tuple(pos)  # 将数组转换为 tuple 类型的坐标
            hub = Hub(position=position, direction=0)  # 假设 direction 默认为 0
            self.machines[position] = hub  # 添加 Hub 对象到 machines 字典
        # 返回环境的初始观察值（obs）和一个空字典
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # print(self.steps)
        if self.steps == 0:
            self.steps += 1
            return self._get_obs(), 0, False, False, {}
        self.steps += 1
        action_type, position = self.action_list[action]
        position = (position[0],position[1])
        machine_type = action_type[0]
        direction = action_type[1]
        reward = 0.0
        done = False
        truncated = False  # 添加 truncated 标记
        info = {}
        # print("current_act = ", action_type)
        # print(self.grid_bld)
        # 如果达到最大步数，标记为 truncated
        if self.steps >= self.max_step:
            # print(self.total_reward)
            print("Trun")
            print(self.grid_bld)
            truncated = True
            done = False  # 或者也可以直接标记为 done
            return self._get_obs(), reward, done, truncated, info
        reward = self.handle_place(machine_type, position, direction)
        # print(action_type,position)
        # print(self.grid_bld)
        # for machine in self.machines.items():
        #     print(machine[0],machine[1].shape,machine[1].num)
        self.total_reward += reward
        if self.check_goal() == True:
            done = True  # 如果达到目标状态，标记为完成
            reward += 2000
            self.total_reward += reward
            print(self.total_reward)
            print("len =",len(self.machines))
        # 返回观察值、奖励、是否结束、是否被截断和信息
        # mask = self.get_action_mask()
        # print(self.grid_bld)
        # for num,i in enumerate(mask):
        #     if i == 1:
        #         print("valid_act = ",self.action_list[num])
        # print()
        self.last_action_index = action
        return self._get_obs(), reward, done, truncated, info

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
                    # print(current_position,next_position,machine.direction,machine.type)
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
                                # print(f"资源到达 hub，但形状不符合目标 {self.target_shape}")
                                return False
                        elif isinstance(current_machine,Trash):
                            # meet the trash
                            return False
                        else:
                            # print("遇到了其他建筑，停止")
                            return False
                    else:
                        # 如果路径中断或没有传送带，停止追踪
                        # print("资源路径中断")
                        break
                # print()
        return False

    def create_valid_action_space(self):
        """
        创建有效的动作空间，机器类型和对应方向的组合。
        0 -> 移除（无方向限制）
        1 -> 传送带（12个方向）
        2 -> 矿机（4个方向）
        """
        action_spaces = {}
        all_pos = []
        res_pos = np.argwhere(self.grid_rsc != 0)
        # handle the miner action spaces

        for direction in range(4):
            valid_action = (22, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(res_pos)

        # handle the conveyor belt action spaces
        for index in np.ndindex(self.grid_rsc.shape):
            all_pos.append(index)
        for direction in range(12):
            valid_action = (31, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(all_pos)


        # handle the remove action spaces
        action_spaces[(0, -1)] = []
        action_spaces[(0, -1)].extend(all_pos)
        #handle the Trash action spapces
        action_spaces[(24, 0)] = []
        action_spaces[(24, 0)].extend(all_pos)
        # handle the Cutter action spaces
        cutter_act = self.get_possible_cutter_actions()
        for action in cutter_act:
            act,pos = action
            if act not in action_spaces:
                action_spaces[act] = []
            action_spaces[act].append(pos)
        # print("action spaces:")
        # print(action_spaces)
        # print()
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
        # print(self.action_space)
        return
    def extract_buildings(self, position):
        # extract the machine type and direction in the specific position
        machine_type = self.grid_bld[position] // 100
        direction = self.grid_bld[position] % 100
        return machine_type, direction

    def get_start_pos(self,position,direction):
        cur_pos = position
        cur_dir = direction
        # print("cur = ",cur_pos,"cur_dir = ",cur_dir)
        pre_pos = self._get_pre_position(cur_pos,cur_dir)
        if pre_pos == None:  # find the bound
            return cur_pos
        pre_dir = self.extract_buildings(pre_pos)[1]
        while True:
            if pre_pos == None:  # find the bound
                return cur_pos
            if self._get_next_position(pre_pos,pre_dir) != cur_pos:
                return cur_pos
            machine_type = self.extract_buildings(pre_pos)[0]
            if machine_type == 22 or machine_type == 24:
                return pre_pos
            elif self.grid_bld[pre_pos] == -1: # no more building in the pre_pos
                return pre_pos
            cur_pos = pre_pos
            cur_dir = pre_dir
            pre_pos = self._get_pre_position(cur_pos, cur_dir)
            pre_dir = self.extract_buildings(pre_pos)

    def calculate_miner_reward(self,position,direction):
        hub_pos = np.argwhere(self.grid_bld == 2100)[0]
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
        if x_pos == True and y_pos == True:#左上
            if direction == 2 or direction == 4:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos == False and y_pos == True:#左下
            if direction == 1 or direction == 4:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos == True and y_pos == False:#右上
            if direction == 2 or direction == 3:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos == False and y_pos == False:#右下
            if direction == 1 or direction == 3:
                direction_reward = 10
            else:
                direction_reward = -10

        return distance_reward + direction_reward
    def calculate_conveyor_reward(self,position,direction):
        return 1
        reward = 0
        current_pos = position
        current_direct = direction
        # while True:
        #     if self.grid_bld[current_pos] == -1:
        #         #connected to coneveyor but not connected to the start
        #         reward = -3
        #         break
        #     current_machine = self.extract_buildings(current_pos)[0]
        #     if current_machine == 22:
        #         reward += 10
        #         break
        #     pre_pos = self._get_pre_position(current_pos,current_direct)
        #     # print(self.grid_bld)
        #     # print(pre_pos)
        #     if pre_pos == None :
        #         return -1
        #     pre_direct = self.machines[pre_pos].direction
        #     if self._get_next_position(pre_pos,pre_direct) != current_pos:
        #         #the conveyor is neighbor but not connected,wrong direction
        #         # print("not connected")
        #         reward = 0
        #         break
        #     current_pos = pre_pos
        #     current_direct = self.machines[pre_pos].direction
        #     reward += 1
        # hub_pos = self.find_closet_hub(position)
        # next_pos = self._get_next_position(position,direction)
        # if hub_pos == None or next_pos == None:
        #     #no valid closet hub
        #     return 0
        # if distance(hub_pos,position) < distance(hub_pos,next_pos):
        #     reward = reward / 2
        # else:
        #     reward = reward * 2
        # return reward
    def calculate_trash_reward(self,position):
        start = self._get_pre_position(position,0)
        reward = -10
        for pos in start:
            if pos is None or self.grid_bld[pos] == -1:
                continue
            start_pos = self.get_start_pos(pos,self.grid_bld[pos]%100)
            machine_type,direction = self.extract_buildings(start_pos)
            # print(machine_type)
            if machine_type == 31:
                reward += 0
            elif machine_type == 22:
                reward -= 20
            elif machine_type == 23:
                reward += 10
            #todo:handle main and sub entrance for cutter
        return reward
    def get_cutter_pos(self,position,direction):
        main_pos = None
        sub_pos = None
        if self.machines[position].position == position:
            #main pos
            main_pos = position
            direction_map = {
                1: (0, 1),  # 向上，副出口在下
                2: (0, -1),  # 向下，副出口在上
                3: (-1, 0),  # 向左，副出口在右
                4: (1, 0)  # 向右，副出口在左
            }
            dx, dy = direction_map[direction]
            sub_pos = (position[0] + dx, position[1] + dy)
        elif self.machines[position].sub_pos == position:
            sub_pos = position
            direction_map = {
                1: (0, -1),  # 向上，副出口在下
                2: (0, 1),  # 向下，副出口在上
                3: (1, 0),  # 向左，副出口在右
                4: (-1, 0)  # 向右，副出口在左
            }
            dx, dy = direction_map[direction]
            main_pos = (position[0] + dx, position[1] + dy)
        else:
            print("error, not a cutter")
        return main_pos,sub_pos

    def handle_place(self, machine_type, position, direction):
        # handle the place event
        # param:machine_type:the number of the machine
        # param:position:the place that we want put the machine
        # param:direction:the machine's direction
        # return:Canplace: to show if we can handle the action successfully
        # return:reward:the reward of the action
        new_machine = Machine
        reward = 0
        if machine_type == 0:  # action is remove
            machine_type,direction = self.extract_buildings(position)
            if machine_type == 23:
                main_pos,sub_pos = self.get_cutter_pos(position,direction)
                self.grid_bld[main_pos] = -1
                self.grid_bld[sub_pos] = -1
            else:
                self.grid_bld[position] = -1
            reward = -3
            self.reward_grid[position] = -1
            return reward
        elif machine_type == 22: #placing miner
            # reward = self.calculate_miner_reward(position,direction)
            self.reward_grid[position] = reward
            self.grid_bld[position] = 22 * 100 + direction
            new_machine = Miner(position,direction)
            self.machines[position] = new_machine
        elif machine_type == 31:#placing conveyor
            # reward = self.calculate_conveyor_reward(position,direction)
            self.grid_bld[position] = 31 * 100 + direction
            # print("cur act is ",position,direction)
            new_machine = Conveyor(position,direction)
            self.machines[position] = new_machine
            self.reward_grid[position] = reward
        elif machine_type == 24:
            # reward = self.calculate_trash_reward(position)
            new_machine = Trash(position,direction)
            self.machines[position] = new_machine
            self.reward_grid[position] = reward
            self.grid_bld[position] = 24 * 100
        elif machine_type == 23:
            new_machine = Cutter(position,direction)
            self.grid_bld[position] = 23 * 100 + direction
            sub_pos = new_machine.sub_pos
            self.grid_bld[sub_pos] = 23 * 100 + direction
            self.machines[position] = new_machine
            self.machines[sub_pos] = new_machine
        reward -= 1
        return reward



    def compute_resource_utilization(self):
        # 计算资源的利用率
        self.total_useful_resource = np.argwhere(self.grid_rsc == self.target_shape)
        # 检查资源是否成功被传送到终点
        for machine in self.machines.values():
            if isinstance(machine, Miner):
                if machine.is_conveyed:
                    self.processed_resource += 1


    def _get_next_position(self, position: Tuple[int, int], direction: int):
        x, y = position
        if (direction == 1 or direction == 9 or direction == 10) and x - 1 >= 0:  # 向上
            return (x - 1, y)
        elif (direction == 4 or direction == 6 or direction == 8) and y + 1 < self.grid_rsc.shape[1]:  # 向右
            return (x, y + 1)
        elif (direction == 2 or direction == 11 or direction == 12) and x + 1 < self.grid_rsc.shape[0]:  # 向下
            return (x + 1, y)
        elif (direction == 3 or direction == 5 or direction == 7) and y - 1 >= 0:  # 向左
            return (x, y - 1)
        return None

    def handle_direction(self,position,direction):
        #
        x, y = position
        if (direction == 1 or direction == 5 or direction == 6) and x + 1 < self.grid_rsc.shape[0]:  # 向上
            return (x + 1, y)
        elif (direction == 2 or direction == 7 or direction == 8) and x - 1 >= 0:  # 向下
            return (x - 1, y)
        elif (direction == 3 or direction == 9 or direction == 11) and y + 1 < self.grid_rsc.shape[1]:  # 向右
            return (x, y + 1)
        elif (direction == 4 or direction == 10 or direction == 12) and y - 1 >= 0:  # 向左
            return (x, y - 1)
        return None
    def _get_pre_position(self,position,direction):
        #get the preivous position of conveyor
        possible_pos = []
        # print(direction)
        if direction == 0:
            for direct in range(4):
                pre_pos = self.handle_direction(position,direct+1)
                if position == pre_pos:
                    continue
                else:
                    possible_pos.append(pre_pos)
            return tuple(possible_pos)
        else:
            return self.handle_direction(position,direction)

    def start_retrieve(self, start_position, start_direction):
        """
        回溯传送带，找到起点。
        :param start_position: 起始传送带的位置
        :param start_direction: 起始传送带的方向
        :return: 传送带的起点位置
        """
        current_position = start_position
        current_direction = start_direction
        while True:
            pre_positions = self._get_pre_position(current_position, current_direction)
            if not pre_positions:
                return current_position
            # 假设传送带只能从一个方向来（多个方向情况可以进一步处理）
            # 选择第一个有效前置位置作为回溯位置
            previous_position = pre_positions
            # 更新当前位置和方向
            current_position = previous_position
            current_direction = self.grid_bld[current_position] % 100

            # 如果找不到下一个前置位置，或者已经到达起点，结束回溯
            if self.grid_bld[current_position]//100 == 22:
                return current_position


    def get_possible_cutter_actions(self):
        rows, cols = self.grid_bld.shape
        possible_actions = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols and self.grid_bld[r, c] == -1 and self.grid_bld[r, c + 1] == -1:
                    # 水平放置，direction为 1 或者 2,
                    possible_actions.append(((23,1),(r, c)))
                if c - 1 >= 0 and self.grid_bld[r, c] == -1 and self.grid_bld[r, c - 1] == -1:
                    possible_actions.append(((23,2), (r, c)))
                if r - 1 >= 0 and self.grid_bld[r, c] == -1 and self.grid_bld[r - 1, c] == -1:
                    possible_actions.append(((23,3),(r, c)))
                if r + 1 < rows and self.grid_bld[r, c] == -1 and self.grid_bld[r + 1, c] == -1:
                    possible_actions.append(((23,4), (r, c)))
        return possible_actions


    def _is_first_building(self, position, direction):
        #check if the position is the first conyeyor of the path and connected to the miner
        if self._get_next_position(position, direction) == None:#next pos out of bound
            return True
        else:
            machine_type,direction = self.extract_buildings(position)
            if machine_type == 24:
                return True
            nxt_pos = self._get_next_position(position, direction)
            nxt_direct = self.grid_bld[nxt_pos] % 100
            if self.grid_bld[nxt_pos] == -1:
                return True
            else:
                return False

    def CanPlaceConveyor(self, position: Tuple[int, int], direction: int) -> bool:
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
        if self.grid_bld[position] != -1:  # current place has buildings
            return False
        pre_pos = self._get_pre_position(position, direction)
        if pre_pos == None or self.grid_bld[pre_pos] // 100 == 21 or self.grid_bld[pre_pos] == -1:
            # out of boundary or pre_pos is hub or pre_pos is not a building
            return False
        start_pos = self.get_start_pos(position, direction)
        # print(start_pos)
        if start_pos == position:  # not connected to other machines
            return False
        # handle nxt_pos situation
        nxt_pos = self._get_next_position(position, direction)
        if nxt_pos == None:  # next position out of bound
            return False
        if self.grid_bld[nxt_pos] == -1 or self.grid_bld[nxt_pos] // 100 == 21:
            return True
        else:
            return False
        # if nxt_direct <= 4 and nxt_direct == direction:
        #     return True
        # if direction == 1 or direction == 9 or direction == 10: #going up
        #     if nxt_direct != 5 and nxt_direct != 6:
        #         return False
        # elif direction == 2 or direction == 11 or direction == 12:
        #     if nxt_direct != 7 and nxt_direct != 8:
        #         return False
        # elif direction == 3 or direction == 5 or direction == 7:
        #     if nxt_direct != 9 and nxt_direct != 11:
        #         return False
        # elif direction == 4 or direction == 6 or direction == 8:
        #     if nxt_direct != 10 and nxt_direct != 12:
        #         return False

    def CanPlaceMiner(self, position):
        if self.grid_rsc[position] == 0:
            return False
        else:
            return True

    def CanRemove(self, position):
        if self.grid_bld[position] != -1 and self.grid_bld[position] / 100 != 21:
            return True
        return False
    def can_remove(self,position):
        #check if the remove action is valid in position
        #param:position, represents the target delete position
        if self.grid_bld[position] == -1 or self.grid_bld[position]//100 == 21: # no building can't remove or the building is destination
            return False
        #otherwise, there is a buidling except destination
        machine_type,direction = self.extract_buildings(position)
        # print(f"machine = {machine_type},direction = {direction},isfirst = {self._is_first_building(position,direction)}")
        return self._is_first_building(position,direction)
    def CanPlaceCutter(self,position,direction):
        x,y = position
        direction_map = {
            1: (0, 1),  # 向上，副出口在下
            2: (0, -1),  # 向下，副出口在上
            3: (-1, 0),  # 向左，副出口在右
            4: (1, 0)  # 向右，副出口在左
        }
        dx, dy = direction_map[direction]
        main_pos = position
        sub_pos = (x + dx, y + dy)
        if self.grid_bld[main_pos] != -1 or self.grid_bld[sub_pos] != -1:
            return False
        pre_pos = self._get_pre_position(position, direction)
        if pre_pos == None or self.grid_bld[pre_pos] // 100 != 31:
            # can not be out of boundary and the pre_pos machine must be a correct conveyor
            return False
        start_pos = self.get_start_pos(position, direction)
        # print(start_pos)
        if start_pos == position:  # not connected to other machines
            return False
            # handle nxt_pos situation
        nxt_pos = self._get_next_position(position, direction)
        if nxt_pos == None:  # next position out of bound
            return False
        if self.grid_bld[nxt_pos] == -1 or self.grid_bld[nxt_pos] // 100 == 21:
            return True
        else:
            return False
    def check_action_valid(self, machine_type, position, direction):
        if machine_type == 0:
            # print(machine_type,position,direction)
            return self.can_remove(position)
        elif machine_type == 31:
            if self.CanPlaceConveyor(position, direction):
                # print("canplace conveyor at pos",position)
                return True
            # else:
            #     print("cant place conyeor at pos",position,"direct",direction)
        elif machine_type == 23:
            if self.CanPlaceCutter(position,direction):
                return True
            else:
                # print("cant place cutter in pos",position,direction)
                return False
        else:
            #handle other machines
            if self.grid_bld[position] == -1:
                return True
            else:
                return False

    def get_possible_action_idx(self):
        index = []
        all_machine_pos = np.argwhere((self.grid_bld != -1) & (self.grid_bld // 100 != 21))
        for pos in all_machine_pos:
            idx = self.act_dict[(0, -1), (pos[0], pos[1])]
            index.append(idx)
            # if not (pos[0], pos[1]) in self.machines:
            #     print(self.grid_bld)
            #     print("error")
            direct = self.machines[(pos[0], pos[1])].direction
            if self._is_first_building((pos[0],pos[1]),direct) == True:
                next_pos = self._get_next_position((pos[0],pos[1]),direct)
                if next_pos == None:
                    continue
                # print("find first build,", pos,next_pos,direct)
                for direction in range(12):
                    #place possible conveyor
                    # print("possbile next=",next_pos)
                    if direction < 4:
                        if ((23, direction + 1), next_pos) in self.act_dict:
                            idx = self.act_dict[(23, direction + 1), next_pos]
                            index.append(idx)
                    idx = self.act_dict[(31, direction + 1), next_pos]
                    index.append(idx)
                #place possible trasher
                idx = self.act_dict[(24, 0), next_pos]
                index.append(idx)
        res_pos = np.argwhere(self.grid_rsc != 0)
        for pos in res_pos:
            for direction in range(4):
                #place miner nearby
                idx = self.act_dict[(22, direction + 1), (pos[0],pos[1])]
                index.append(idx)
        return index

    def get_action_mask(self):
        # 创建一个与动作空间大小相同的掩码，默认为 1（表示所有动作有效）
        mask = [0] * len(self.action_list)
        # print(self.grid_bld)
        possible_action_idx = self.get_possible_action_idx()
        # flag = True
        for idx in possible_action_idx:
            action = self.action_list[idx]
            machine_type,direction = action[0]
            position = (action[1][0],action[1][1])
            if self.check_action_valid(machine_type, position, direction) == True:
                mask[idx] = 1
                # print("valid act = ",action,position)
                # flag = False
            else:
                mask[idx] = 0
        # print()
        return mask

    def find_closet_hub(self,cur_pos):
        min_distance = 0x3f3f3f
        closet_pos = None
        hub_poses = np.argwhere(self.grid_bld//100 == 21)
        for hub_pos in hub_poses:
            hub_x,hub_y = hub_pos
            distance = (cur_pos[0] - hub_x) ** 2 + (cur_pos[1] - hub_y) ** 2
            if distance < min_distance:
                if (self.grid_bld[hub_x - 1][hub_y] == -1 or self.grid_bld[hub_x + 1][hub_y] == -1 or
                        self.grid_bld[hub_x][hub_y - 1] == -1 or self.grid_bld[hub_x][hub_y + 1] == -1):
                    min_distance = distance
                    closet_pos = (hub_x, hub_y)
        return closet_pos