import os
import sys
import time
from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from Machine import Machine,Miner,Hub,Trash,Conveyor,Cutter,Rotator


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)





class ShapezEnv(gymnasium.Env):
    def __init__(self, build, res, target_shape):
        self.required_routes = 2
        self.success_times = 0
        self.max_step = 1200 * self.required_routes
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
        self.act_list = []
        # 获取网格的大小
        grid_shape = self.grid_rsc.shape
        self.act_mask = None
        # 定义有效的动作组合（机器类型 + 对应的方向）
        self.create_action_space()
        self.act_dict = {(action, tuple(pos)): idx for idx, (action, pos) in enumerate(self.action_list)}
        self.last_action_index = -1
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0,
                high=np.max(5000),
                shape=(grid_shape[0], grid_shape[1]),
                dtype=np.int32
            )
        })

    def _get_obs(self):
        observation = {
            'grid': self.grid_bld,  # 当前网格数据
        }
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
           pass
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
        # 找到所有 Hub（值为 21 的位置）
        for index, value in np.ndenumerate(self.grid_bld):
            machine_type = value // 100
            position = tuple(index)
            direction = value % 100
            if machine_type == 23:  # Cutter
                if position in self.machines:
                    # print(f"警告：位置 {position} 已经被占用，跳过主出口初始化")
                    continue
                cutter = Cutter(position, direction)
                self.machines[position] = cutter
                # print(f"Cutter 主出口初始化成功，位置: {position}, 方向: {direction}")
                # 初始化 Cutter 的副出口
                sub_pos = cutter.sub_pos
                # 检查副出口是否已经被占用
                if sub_pos in self.machines:
                    print(f"警告：副出口位置 {sub_pos} 已经被占用，跳过此位置")
                else:
                    self.machines[sub_pos] = cutter  # 将副出口也加入到 self.machines
                    # print(f"Cutter 副出口初始化成功，位置: {sub_pos}, 方向: {direction}")
            elif machine_type == 22:  # Miner 矿机
                self.machines[position] = Miner(position, direction)
                # print(f"矿机初始化成功，位置: {position}, 方向: {direction}")
            elif machine_type == 31:  # Conveyor 传送带
                self.machines[position] = Conveyor(position, direction)
                # print(f"传送带初始化成功，位置: {position}, 方向: {direction}")

            elif machine_type == 21:  # Hub
                self.machines[position] = Hub(position, direction)
                # print(f"Hub 初始化成功，位置: {position}, 方向: {direction}")

            elif machine_type == 24:  # Trash
                self.machines[position] = Trash(position,0)
                # print(f"垃圾桶初始化成功，位置: {position}")
            # 根据需要添加其他机器类型
        # 返回环境的初始观察值（obs）和一个空字典
        obs = self._get_obs()
        for pos,machine in self.machines.items():
            direct = machine.direction
            if isinstance(machine,Conveyor):
                self.reward_grid[pos] = self.calculate_conveyor_reward(pos,direct)
            elif isinstance(machine,Miner):
                self.reward_grid[pos] = self.calculate_miner_reward(pos,direct)
            elif isinstance(machine, Trash):
                self.reward_grid[pos] = self.calculate_trash_reward(pos)
        return obs, {}

    def step(self, action):
        # print(self.steps)
        reward = 0
        if self.steps == 0:
            self.steps += 1
            return self._get_obs(), 0, False, False, {}
        self.steps += 1
        self.act_list.append(action)
        action_type, position = self.action_list[action]
        position = (position[0],position[1])
        machine_type = action_type[0]
        direction = action_type[1]
        reward = 0
        self.act_mask = self.get_action_mask()
        if self.act_mask[action] != 0:
            reward = self.handle_place(machine_type, position, direction)
        # print(np.array2string(self.grid_bld, max_line_width=200))
        # for idx,value in enumerate(self.act_mask):
        #     if value == 1:
        #         act = self.action_list[idx]
        #         machine_type, direction = act[0]
        #         print("valid act = ",act)
        # print("chosed act",action_type, position, reward,self.total_reward)
        # print()
        # time.sleep(0.5)
        done = False
        truncated = False  # 添加 truncated 标记
        info = {}
        # 如果达到最大步数，标记为 truncated
        if self.steps >= self.max_step:
            # print(self.total_reward)
            print("Trun")
            print(np.array2string(self.grid_bld, max_line_width=200))
            print(self.total_reward)
            truncated = True
            done = False  # 或者也可以直接标记为 done
            return self._get_obs(), reward, done, truncated, info
        # for machine in self.machines.items():
        #     print(machine[0],machine[1].shape,machine[1].num)

        if self.check_goal() == self.required_routes:
            done = True  # 如果达到目标状态，标记为完成
            reward += self.max_step * 10
            self.total_reward += reward
            self.success_times += 1
            print(self.total_reward)
            print(self.success_times)
            print("len =",len(self.machines))
            sys.exit()
        # elif self.check_goal() == 1:
        #     print("1")
            # 返回观察值、奖励、是否结束、是否被截断和信息
            # mask = self.get_action_mask()
            # print(np.array2string(self.grid_bld, max_line_width=200))
            # for num,i in enumerate(mask):
            #     if i == 1:
            #         print("valid_act = ",self.action_list[num])
            # print()
            # time.sleep(1)
        self.last_action_index = action
        step_penalty = 0.01 * np.exp(0.01 * self.steps)
        # print("s_p= ",step_penalty)
        reward -= step_penalty
        self.total_reward += reward
        # print()
        return self._get_obs(), reward, done, truncated, info

    def check_goal(self):
        """
        检查是否有资源从矿机通过传送带等路径成功到达 hub，并符合目标形状。
        """
        cnt = 0
        for position, machine in self.machines.items():
            if isinstance(machine, Miner): # 找到矿机，开始从资源生成点追踪
                current_position = position
                current_shape = self.grid_rsc[position]  # 获取资源的初始形状
                # 使用通用路径追踪函数进行检查
                result = self._track_path_with_rotator(position, current_shape)
                if result == 'hub':
                    cnt += 1
                elif result == 'none':
                    continue  # 检查其他矿机的路径
                elif result == 'trash':
                    continue

        return cnt

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
        for index in np.ndindex(self.grid_bld.shape):
            if self.grid_bld[index] //100 != 21:
                all_pos.append(index)
        action_spaces[(0, -1)] = []
        action_spaces[(0, -1)].extend(all_pos)
        all_pos.clear()
        for index in np.ndindex(self.grid_bld.shape):
            if self.grid_bld[index] //100 != 21 and self.grid_rsc[index] == 0:
                all_pos.append(index)
        for direction in range(12):
            valid_action = (31, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(all_pos)


        # handle the remove action spaces

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
            if self._get_next_position(pre_pos,pre_dir) != cur_pos: # find the first position
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

    def retrieve_path(self,position,direction):
        #回溯节点，返回所经过的路程
        visited = []
        cur_pos = position
        cur_direct = direction
        while True:
            if cur_pos in self.machines:
                visited.append(cur_pos)
            if cur_pos == None:
                break
            if (self.grid_bld[cur_pos] // 100 != 31 and self.grid_bld[cur_pos] // 100 != 23) or self.grid_bld[cur_pos] // 100 == 24 or self.grid_bld[cur_pos] // 100 == 22 :
                #find the destination
                break
            cur_pos = self._get_pre_position(cur_pos,cur_direct)
            cur_direct = self.grid_bld[cur_pos] % 100
            if cur_pos in visited:
                #有环
                break
        return visited
    def check_valid_shape(self,shape):
        #check if the shape is target shape or possible target shape
        if shape == self.target_shape:
            return True
        if self.target_shape == 13 or self.target_shape == 14 or self.target_shape == 60 or self.target_shape == 20:
            if shape == 11:
                return True
            else:
                return False
        if 16 <= self.target_shape <= 19:
            if shape == 12:
                return True
            else:
                return False
    def calculate_miner_reward(self,position,direction):
        hub_positions = np.argwhere(self.grid_bld == 2100)
        if hub_positions.size > 0:
            hub_pos = hub_positions[0]
        else:
            # 处理没有找到 hub 的情况
            print("No hub found in grid_bld.")
            print(self.grid_bld)
            for num, in enumerate(self.act_list):
               print("valid_act = ",self.action_list[num])
            print()
            hub_pos = None  # 或者根据你的需求处理

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
        reward = 0
        path = self.retrieve_path(position,direction)
        start = path[-1]
        current_shape = None
        if self.grid_bld[start] // 100 != 22:
            #not connected to the start
            return -5
        # print(np.array2string(self.grid_bld, max_line_width=200))
        # print(path)

        for p in reversed(path):
            if isinstance(self.machines[p], Miner):
                current_shape = self.grid_rsc[p]
            elif isinstance(self.machines[p], Cutter):
                shapes = self.process_cut(current_shape)
                if shapes[0] != -1:
                    # valid cutter
                    if self.machines[p].position == p:
                        # main pos
                        current_shape = shapes[0]
                    else:
                        #sub_pos
                        current_shape = shapes[1]

        hub_pos = self.find_closet_hub(position)
        next_pos = self._get_next_position(position,direction)
        if hub_pos == None or next_pos == None:
            #no valid closet hub meaningless to continue placing belt
            return -5
        if self.check_valid_shape(current_shape):
            # conveying the correct shape
            reward += (self.grid_bld.shape[0] - distance(hub_pos, position))/2
        else:
            return -20
        if distance(hub_pos,position) < distance(hub_pos,next_pos):
            reward = -5
        else:
            # closer to the hub
            reward += 5
        return reward

    def calculate_trash_reward(self,position):
        start = self._get_pre_position(position,0)
        reward = 0
        for pos in start:
            if pos is None or self.grid_bld[pos] == -1:
                continue
            path = self.retrieve_path(pos,self.grid_bld[pos]%100)
            current_shape = None
            # print("start calculating")
            # print(np.array2string(self.grid_bld, max_line_width=200))
            # print(self.machines)
            # print(path)
            for p in reversed(path):
                if isinstance(self.machines[p],Miner):
                    current_shape = self.grid_rsc[p]
                elif isinstance(self.machines[p],Cutter):
                    shapes = self.process_cut(current_shape)
                    if shapes[0] != -1:
                        #valid cutter
                        if self.machines[p].position == p:
                            #main pos
                            current_shape = shapes[0]
                        else:
                            current_shape = shapes[1]

            if self.check_valid_shape(current_shape) or current_shape is None:
                # print("cur_shape =", current_shape)
                # print("delete the invalid shape")
                reward -= 50
            else:
                # print("cur_shape =", current_shape)
                # print("delete the valid shape")
                reward += 50
            # print(reward)
            # print()
        # sys.exit()
        return reward
    def calculate_cutter_reward(self,position,direction):
        # cur_pos = position
        # cur_direct = direction
        path = self.retrieve_path(position,direction)
        if path[-1] not in self.machines:
            print(path)
            print(self.grid_bld)
        # start_machine = self.machines[path[-1]]
        # print(path)
        # print(np.array2string(self.grid_bld, max_line_width=200))
        # sys.exit()
        current_shape = None
        for pos in path:
            if isinstance(self.machines[pos],Miner):
               current_shape = self.grid_rsc[pos]
            if isinstance(self.machines[pos], Cutter):
                if self.machines[pos].position == pos:
                    #main exit:
                    current_shape = self.process_cut(current_shape)[0]
                else:
                    #sub exit:
                    current_shape = self.process_cut(current_shape)[1]
        if self.check_valid_shape(current_shape):
            return 100
        else:
            return -100



    def handle_place(self, machine_type, position, direction):
        # handle the place event
        # param:machine_type:the number of the machine
        # param:position:the place that we want put the machine
        # param:direction:the machine's direction
        # return:Canplace: to show if we can handle the action successfully
        # return:reward:the reward of the action
        new_machine = Machine
        reward = -1
        if machine_type == 0:  # action is remove
            machine_type,direction = self.extract_buildings(position)
            if machine_type == 23:
                #cutter
                main_pos,sub_pos = self.get_cutter_pos(position,direction)
                self.grid_bld[main_pos] = -1
                self.grid_bld[sub_pos] = -1
                del self.machines[main_pos]
                del self.machines[sub_pos]
            elif machine_type == 31:
                path = self.retrieve_path(position,self.grid_bld[position]%100)
                start_pos = path[-1]
                start_shape = self.grid_rsc[start_pos]
                if self._track_path_with_rotator(start_pos,start_shape)=='hub':
                    return -1
                else:
                    self.grid_bld[position] = -1
                    reward = -(self.reward_grid)[position] \
                        if (self.reward_grid)[position] > 0 else -(self.reward_grid)[position] / 2
                    del self.machines[position]
                    self.reward_grid[position] = -1
            else:
                self.grid_bld[position] = -1
                reward = -(self.reward_grid)[position]\
                    if (self.reward_grid)[position] > 0 else -(self.reward_grid)[position] / 2
                del self.machines[position]
                self.reward_grid[position] = -1
            return reward
        elif machine_type == 22: #placing miner
            count_22 = np.sum(self.grid_bld//100 == 22)
            reward = self.calculate_miner_reward(position,direction)
            self.reward_grid[position] = reward
            self.grid_bld[position] = 22 * 100 + direction
            new_machine = Miner(position,direction)
            self.machines[position] = new_machine
        elif machine_type == 31:#placing conveyor
            self.grid_bld[position] = 31 * 100 + direction
            # print("cur act is ",position,direction)
            new_machine = Conveyor(position,direction)
            self.machines[position] = new_machine
            self.reward_grid[position] = reward
            reward = self.calculate_conveyor_reward(position, direction)
        elif machine_type == 24:
            #place trash
            new_machine = Trash(position,direction)
            self.machines[position] = new_machine
            self.reward_grid[position] = reward
            self.grid_bld[position] = 24 * 100
            reward = self.calculate_trash_reward(position)
        elif machine_type == 23:
            #place cutter
            new_machine = Cutter(position,direction)
            self.grid_bld[position] = 23 * 100 + direction
            sub_pos = new_machine.sub_pos
            self.grid_bld[sub_pos] = 23 * 100 + direction
            self.machines[position] = new_machine
            self.machines[sub_pos] = new_machine
            reward = self.calculate_cutter_reward(position, direction)
            if reward > 0:
                self.reward_grid[position] = reward
                self.reward_grid[sub_pos] = reward
            else:
                main_pos = position
                del self.machines[main_pos]
                del self.machines[sub_pos]
                self.grid_bld[position] = -1
                self.grid_bld[sub_pos] = -1
                return -20
        reward -= 1
        self.reward_grid[position] = reward
        return reward




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

    def handle_pre_direction(self,position,direction):
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
                pre_pos = self.handle_pre_direction(position,direct+1)
                if pre_pos is None:
                    continue
                else:
                    if self.grid_bld[pre_pos] != -1 :
                        if self._get_next_position(pre_pos,self.grid_bld[pre_pos]%100) == position:
                            possible_pos.append(pre_pos)
            return tuple(possible_pos)
        else:
            # handle the cutter previous position
            if position in self.machines:
                n = self.grid_bld.shape[0]
                m = self.grid_bld.shape[1]
                if isinstance(self.machines[position], Cutter):
                    if self.machines[position].sub_pos == position:
                        # current position is the sub_pos of cutter
                        x, y = position
                        if direction == 1 and x + 1 < n and y - 1 >= 0:
                            return (x + 1, y - 1)
                        elif direction == 2 and x - 1 >= 0 and y + 1 < m:
                            return (x - 1, y + 1)
                        elif direction == 3 and x + 1 < n and y + 1 < m:
                            return (x + 1, y + 1)
                        elif direction == 4 and x - 1 >= 0 and y - 1 >= 0:
                            return (x - 1, y - 1)
                        else:
                            return None

            return self.handle_pre_direction(position,direction)

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
                if c + 1 < cols and self.grid_bld[r, c] //100 != 21 and self.grid_bld[r, c + 1]//100 != 21:
                    # 水平放置，direction为 1 或者 2,
                    possible_actions.append(((23,1),(r, c)))
                if c - 1 >= 0 and self.grid_bld[r, c] //100 != 21 and self.grid_bld[r, c - 1]//100 != 21:
                    possible_actions.append(((23,2), (r, c)))
                if r - 1 >= 0 and self.grid_bld[r, c] //100 != 21 and self.grid_bld[r - 1, c] //100 != 21:
                    possible_actions.append(((23,3),(r, c)))
                if r + 1 < rows and self.grid_bld[r, c] //100 != 21 and self.grid_bld[r + 1, c] //100 != 21:
                    possible_actions.append(((23,4), (r, c)))
        return possible_actions

    def _is_first_building(self, position, direction):
        #check if the position is the first conyeyor of the path and connected to the miner
        if self._get_next_position(position, direction) == None: #next pos out of bound
            return True
        else:
            machine_type,direction = self.extract_buildings(position)
            if machine_type == 24:
                #reached the trash
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
        if self.grid_bld[position] != -1 or self.grid_bld[position] // 100 == 21:  # current place has buildings
            return False
        pre_pos = self._get_pre_position(position, direction)
        if pre_pos == None or self.grid_bld[pre_pos] // 100 == 21 or self.grid_bld[pre_pos] == -1:
            # out of boundary or pre_pos is hub or pre_pos is not a building
            return False
        # start_pos = self.get_start_pos(position, direction)
        # if self.grid_bld[start_pos] // 100 != 21:
        #     print(start_pos)
        #     return False
        # # print(start_pos)
        # if start_pos == position:  # not connected to other machines
        #     return False
        pre_direct = self.machines[pre_pos].direction
        start_pos = self.retrieve_path(pre_pos,pre_direct)[-1]
        if self.grid_rsc[start_pos] == 0 or self._get_next_position(pre_pos,pre_direct) != position:
            return False
        # handle nxt_pos situation
        nxt_pos = self._get_next_position(position, direction)
        if nxt_pos == None:  # next position out of bound
            return False
        if self.grid_bld[nxt_pos] == -1 or self.grid_bld[nxt_pos] // 100 == 21:
            return True
        else:
            #next postion has buildings
            nxt_direct = self.grid_bld[nxt_pos] % 100
            if self._get_pre_position(nxt_pos,nxt_direct) == position:
                #can connected
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

    def CanPlaceMiner(self):
        correct_miner_num = 0
        for pos, machine in self.machines.items():
            if isinstance(machine, Miner) and self.check_valid_shape(self.grid_rsc[machine.position]) is True:
                correct_miner_num += 1
        if correct_miner_num < self.required_routes:
            return True
        else:
            return False


    def can_remove(self,position):
            #check if the remove action is valid in position
            #param:position, represents the target delete position
            if self.grid_bld[position]//100 == 24:
                return True
            if self.grid_bld[position] == -1 or self.grid_bld[position]//100 == 21: # no building can't remove or the building is destination
                print("pos", position, "no building")
                return False
            #otherwise, there is a buidling except destination
            machine_type,direction = self.extract_buildings(position)
            # print(f"machine = {machine_type},direction = {direction},isfirst = {self._is_first_building(position,direction)}")
            nxt_pos = self._get_next_position(position,direction)
            if nxt_pos is not None:
                if self.grid_bld[nxt_pos]//100 == 21:
                    return True
                else:
                    return self._is_first_building(position,direction)
            else:
                #out of bound
                return False
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
        if self.grid_bld[main_pos] != -1 or self.grid_bld[sub_pos] != -1 or self.grid_bld[sub_pos]//100 == 21 or self.grid_bld[main_pos]//100 == 21:
            return False
        pre_pos = self._get_pre_position(position, direction)
        if pre_pos == None or self.grid_bld[pre_pos] // 100 != 31:
            # can not be out of boundary and the pre_pos machine must be a correct conveyor
            return False
        start_pos = self.get_start_pos(position, direction)
        if start_pos == position:  # not connected to other machines
            return False
            # handle nxt_pos situation
        nxt_pos = self._get_next_position(position, direction)
        if nxt_pos == None:  # next position out of bound
            return False
        path = self.retrieve_path(pre_pos,direction)
        # print(position,direction)
        # print(np.array2string(self.grid_bld, max_line_width=200))
        target_shape = self.grid_rsc[path[-1]]
        if target_shape == self.target_shape:
            return False
        if self.target_shape not in self.process_cut(target_shape):
            return False
        if self.grid_bld[nxt_pos] == -1 or self.grid_bld[nxt_pos] // 100 == 21:
            return True
        else:
            return False
    def check_action_valid(self, machine_type, position, direction):
        if machine_type == 0:
            # print(machine_type,position,direction)
            # if position == (4,4) and self.can_remove(position) == True:
            #     print("can remove hub,",self.grid_bld)
            return self.can_remove(position)
        elif machine_type == 31:
            if self.CanPlaceConveyor(position, direction):
                # print("canplace conveyor at pos",position)
                # print(self.grid_bld)
                return True
            # else:
            #     print("cant place conyeor at pos",position,"direct",direction)
        elif machine_type == 23:
            if self.CanPlaceCutter(position,direction):
                return True
            else:
                # print("cant place cutter in pos",position,direction)
                return False
        elif machine_type == 22:
            #miner
            return self.check_valid_shape(self.grid_rsc[position])
        else:
            #handle other machines
            if self.grid_bld[position] == -1:
                return True
            else:
                return False

    def get_possible_action_idx(self):
        index = []
        index.clear()
        if self.CanPlaceMiner() is True:
            res_pos = np.argwhere(self.grid_rsc != 0)
            # handle the miner action spaces
            for direction in range(4):
                for pos in res_pos:
                    idx = self.act_dict[(22, direction + 1),(pos[0],pos[1])]
                    index.append(idx)

        all_machine_pos = np.argwhere((self.grid_bld != -1) & (self.grid_bld // 100 != 21))
        for pos in all_machine_pos:
            idx = self.act_dict[(0, -1), (pos[0], pos[1])]
            if idx in index:
                print("duplicated!!!!")
                sys.exit()
            index.append(idx)
            # if not (pos[0], pos[1]) in self.machines:
            #     print(self.grid_bld)
            #     print("error")
            direct = self.machines[(pos[0], pos[1])].direction
            if self._is_first_building((pos[0],pos[1]),direct) == True:
                next_pos = self._get_next_position((pos[0],pos[1]),direct)
                if next_pos == None or self.grid_rsc[next_pos] != 0:
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
                flag = False
            else:
                # print("invalid act = ",action,position)
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

    def process_cut(self, shape):
        #return:main exit output, sub exit output
        if shape == 11:
            return 13, 14  # 主出口输出左半圆13， 副出口输出右半圆14
        else:
            return (-1,-1)

    def process_rotate(self, shape, rotation_count):
        rotation_map = {
            11: 11,  # 如果是圆形，旋转后不变
            12: 12,  # 如果是正方形，旋转后不变
            13: 20,
            14: 60,
            20: 14,
            60: 13,
            16: 18,
            17: 19,
            18: 17,
            19: 16
        }

        for _ in range(rotation_count):
            shape = rotation_map.get(shape, shape)
        return shape
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

    def _track_path_with_rotator(self, position, shape):
        # print(position)
        # print(np.array2string(self.grid_bld, max_line_width=200))
        # print()
        """
        通用路径追踪函数，处理路径中可能遇到的传送带、旋转器、Cutter和Hub。
        """
        cur_pos = position
        rotation_count = 0  # 初始化旋转次数
        while True:
            nxt_pos = self._get_next_position(cur_pos, self.machines[cur_pos].direction)
            if nxt_pos is None or self.grid_bld[nxt_pos] == -1:
                # print(f"路径中断或无效位置: {nxt_pos}")
                return 'none'
            if nxt_pos in self.machines:
                current_machine = self.machines[nxt_pos]
                if isinstance(current_machine, Conveyor):

                    # 继续沿传送带前进
                    cur_pos = nxt_pos
                    # print(f"经过传送带，当前位置: {cur_pos}")

                elif isinstance(current_machine, Rotator):
                    # 经过 Rotator，增加旋转次数

                    rotation_count += 1
                    cur_pos = nxt_pos
                    # print(f"经过 Rotator，旋转次数: {rotation_count}，当前位置: {cur_pos}")

                elif isinstance(current_machine, Cutter):
                    # 如果是 Cutter，处理形状切割
                    # passed_cutter = True
                    cur_pos = nxt_pos
                    # print(f"资源到达 Cutter，位置: {cur_pos}")

                    # 获取主、副出口
                    main_exit_pos, side_exit_pos = self.get_cutter_pos(cur_pos, current_machine.direction)
                    main_exit_shape, side_exit_shape = self.process_cut(shape)
                    # 分别追踪主出口和副出口
                    if main_exit_pos in self.machines:
                        direction = self.machines[main_exit_pos].direction
                        nxt_pos = self._get_next_position(main_exit_pos,direction)
                        machine_type,direction = self.extract_buildings(nxt_pos)
                        if machine_type == 31 and self._get_pre_position(nxt_pos,direction) == main_exit_pos:
                            #connected
                            main_exit_result = self._track_path_with_rotator(main_exit_pos, main_exit_shape)
                        else:
                            main_exit_result = 'none'
                    else:
                        print("main none")
                        main_exit_result = 'none'

                    if side_exit_pos in self.machines:
                        direction = self.machines[side_exit_pos].direction
                        nxt_pos = self._get_next_position(side_exit_pos, direction)
                        machine_type, direction = self.extract_buildings(nxt_pos)
                        if machine_type == 31 and self._get_pre_position(nxt_pos, direction) == side_exit_pos:
                            # connected
                            side_exit_result = self._track_path_with_rotator(side_exit_pos, side_exit_shape)
                        else:
                            side_exit_result = 'none'
                    else:
                        print("side none")
                        side_exit_result = 'none'

                    if main_exit_result == 'hub' or side_exit_result == 'hub':
                        return 'hub'
                    else:
                        return 'none'

                elif isinstance(current_machine, Hub):
                    # 如果到达 Hub，检查形状是否符合目标
                    rotated_shape = self.process_rotate(shape, rotation_count)
                    # print(f"到达 Hub，最终形状: {rotated_shape}，目标形状: {self.target_shape}")
                    if rotated_shape == self.target_shape:
                        return 'hub'
                    else:
                        return 'none'

                elif isinstance(current_machine, Trash):
                    # 到达 Trash，不成功
                    # print(f"到达 Trash，位置: {nxt_pos}")
                    return 'trash'

                else:
                    return 'none'
            else:
                # print(f"未找到下一个机器，路径中断: {nxt_pos}")
                return 'none'