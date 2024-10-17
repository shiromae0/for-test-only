import os
import sys
import time
import random
from typing import Optional, Tuple, Dict, Any

import gymnasium
import numpy as np
from gymnasium import spaces

from numpy import ndarray, dtype

from Machine import Machine, Miner, Hub, Trash, Conveyor, Cutter, Rotator


def distance(point1, point2):
    """
    Calculates the Manhattan distance between two points.
    Args:
        point1 (tuple): A tuple representing the (x, y) coordinates of the first point.
        point2 (tuple): A tuple representing the (x, y) coordinates of the second point.

    Returns:
        int: The Manhattan distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)


def process_rotate(shape, rotation_count):
    rotation_map = {
        11: 11,
        12: 12,
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


def process_cut(shape):
    # return:main exit output, sub exit output
    if shape == 11:
        return 13, 14
    else:
        return -1, -1


class ShapezEnv(gymnasium.Env):
    def __init__(self, build, res, target_shape, required_routes=1):
        """
                Initializes the class with the given parameters.
                Args:
                    build (array): Initial building grid, representing the environment.
                    res (array): Resource grid, representing shape resources available in the environment.
                    target_shape (int): The required shape in current task.
                    required_routes (int, optional): The number of required routes. Defaults to 1.
                Attributes:
                    required_routes (int): Number of routes required.
                    success_times (int): Counter for successful achieveng times during the training. .
                    max_step (int): Maximum number of steps allowed for one round.
                    steps (int): Current step counter.
                    original_bld (numpy array): Original building grid.
                    grid_rsc (numpy array): Resource grid.
                    grid_bld (numpy array): Current state of the building grid.
                    reward_grid (numpy array): Reward grid initialized with -1.
                    machines (dict): Dictionary to hold machine states.
                    target_shape (int): Require shape of the task.
                    total_reward (int): Total accumulated reward in one round.
                    act_mask (NoneType or custom): Mask for valid actions.
                    action_list (list): List of valid action combinations (machine type + direction).
                    act_dict (dict): Dictionary mapping action and position to an index.
                    observation_space (gym.spaces.Dict): Observation space configuration for the environment.
                """
        self.action_list = None
        self.required_routes = required_routes
        self.success_times = 0
        self.max_step = 1200 * self.required_routes
        self.steps = 0
        self.original_bld = np.array(build)
        self.grid_rsc = np.array(res)
        self.grid_bld = self.original_bld.copy()
        self.reward_grid = np.full(self.grid_bld.shape, -1)
        self.machines = {}
        self.target_shape = target_shape
        self.total_reward = 0
        grid_shape = self.grid_rsc.shape
        self.act_mask = None
        self.create_action_space()
        self.act_dict = {(action, tuple(pos)): idx for idx, (action, pos) in enumerate(self.action_list)}
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0,
                high=np.max(5000),
                shape=(grid_shape[0], grid_shape[1]),
                dtype=np.int32
            )
        })

    def _get_obs(self):
        """
        :return:current observation of the game.
        """
        observation = {
            'grid': self.grid_bld,
        }
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[
        dict[str, ndarray[Any, dtype[Any]]], dict[Any, Any]]:
        if seed is not None:
            pass
        if options is not None:
            pass
        """
           Resets the environment to its initial state. This function is responsible for clearing and 
           reinitializing the environment's components (e.g., machines, rewards, grid state) so that a 
           new episode can begin.
           this function will also calculate the reward matrix if there is any exist buildings in the beginning.
           Args:
               seed (Optional[int]): A seed for random number generation, useful for reproducibility.
               options (Optional[dict]): Additional options to customize the reset process.

           Returns:
               tuple:
                   - obs (dict): A dictionary containing the observation of the environment after the reset.
                   - info (dict): An empty dictionary reserved for additional info.
           """
        self.total_reward = 0
        self.reward_grid = np.full(self.grid_bld.shape, -1)
        self.steps = 0
        self.grid_bld = self.original_bld.copy()
        self.machines.clear()
        # random_actions = random.sample(self.action_list, 500)
        # for action in random_actions:
        #     action_type, position = action
        #     machine_type, direction = action_type
        #     position = (position[0], position[1])
        #     if self.check_action_valid(machine_type,position,direction):
        #         self.handle_place(machine_type, position, direction)

        for index, value in np.ndenumerate(self.grid_bld):
            machine_type = value // 100
            position = tuple(index)
            direction = value % 100
            if machine_type == 23:  # Cutter
                if position in self.machines:
                    continue
                cutter = Cutter(position, direction)
                self.machines[position] = cutter
                sub_pos = cutter.sub_pos
                if sub_pos in self.machines:
                    pass
                else:
                    self.machines[sub_pos] = cutter
            elif machine_type == 22:  # Miner
                self.machines[position] = Miner(position, direction)
            elif machine_type == 31:  # Conveyor
                self.machines[position] = Conveyor(position, direction)
            elif machine_type == 21:  # Hub
                self.machines[position] = Hub(position, direction)
            elif machine_type == 24:  # Trash
                self.machines[position] = Trash(position, 0)
        obs = self._get_obs()
        for pos, machine in self.machines.items():
            direct = machine.direction
            if isinstance(machine, Conveyor):
                self.reward_grid[pos] = self.calculate_conveyor_reward(pos, direct)
            elif isinstance(machine, Miner):
                self.reward_grid[pos] = self.calculate_miner_reward(pos, direct)
            elif isinstance(machine, Trash):
                self.reward_grid[pos] = self.calculate_trash_reward(pos)
        return obs, {}

    def step(self, action):
        """
           Performs a single step in the environment by applying the given action. This function updates
           the environment's state,i.e. the observation space.
           Args:
               action (int): The index of the action to be performed. The action is chosen from the action space.
           Returns:
               tuple: A tuple containing:
                   - obs (dict): The observation of the environment after the step.
                   - reward (float): The reward for the current step.
                   - done (bool): A boolean indicating whether the episode is complete.
                   - truncated (bool): A boolean indicating whether the episode was truncated due to exceeding max steps.
                   - info (dict): Additional information (optional, usually empty).
           """
        if self.steps == 0:
            self.steps += 1
            return self._get_obs(), 0, False, False, {}
        self.steps += 1
        (machine_type, direction), position = self.action_list[action]
        position = (position[0], position[1])
        reward = 0
        self.act_mask = self.get_action_mask()
        if self.act_mask[action] != 0:
            # although we set the invalid action's possibility into a small number,
            # sometimes it will  still choose the invalid action
            reward = self.handle_place(machine_type, position, direction)
        done = False
        truncated = False
        info = {}

        if self.steps >= self.max_step:
            # reached the max step and haven't finish the game,return Truncated
            print("Trun")
            # print(np.array2string(self.grid_bld, max_line_width=200))
            print(self.total_reward)
            truncated = True
            done = False
            return self._get_obs(), reward, done, truncated, info
        if self.check_goal() == self.required_routes:
            # complete the game task
            done = True
            reward += self.max_step * 10
            self.total_reward += reward
            self.success_times += 1
            print(self.total_reward)
            print(self.success_times)
            print("len =", len(self.machines))

        step_penalty = 0.01 * np.exp(0.01 * self.steps)
        reward -= step_penalty
        self.total_reward += reward

        return self._get_obs(), reward, done, truncated, info

    def check_goal(self):
        """
        check the number of the correct routes in the buildings map.
        :return: the number of correct routes in the buildings map.
        """
        cnt = 0
        for position, machine in self.machines.items():
            if isinstance(machine, Miner):
                current_shape = self.grid_rsc[position]
                result = self._track_path_with_rotator(position, current_shape)
                if result == 'hub':
                    cnt += 1
                elif result == 'none':
                    continue
                elif result == 'trash':
                    continue

        return cnt

    def create_valid_action_space(self):
        """
        Creates the valid action space for different machine types in the environment. 
        This function generates possible positions for placing machines like Miners, 
        Conveyor belts, Trash, and Cutters based on the current grid state of the environment.
        
        The action space consists of different action types and valid positions where 
        each action can be applied. This helps in limiting the actions to only those 
        that are valid in the current state of the environment.
    
        Returns:
            dict: A dictionary where the keys are actions (represented as tuples of 
                  machine type and direction), and the values are lists of valid positions 
                  (tuples of grid coordinates) where those actions can be performed.
        """
        action_spaces = {}
        all_pos = []
        res_pos = np.argwhere(self.grid_rsc != 0)
        # handle the miner action spaces

        for direction in range(4):
            valid_action = (22, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(res_pos)

        # handle the remove action spaces
        for index in np.ndindex(self.grid_bld.shape):
            if self.grid_bld[index] // 100 != 21:
                all_pos.append(index)
        action_spaces[(0, -1)] = []
        action_spaces[(0, -1)].extend(all_pos)
        all_pos.clear()
        # handle the conveyor belt action spaces
        for index in np.ndindex(self.grid_bld.shape):
            if self.grid_bld[index] // 100 != 21 and self.grid_rsc[index] == 0:
                all_pos.append(index)
        for direction in range(12):
            valid_action = (31, direction + 1)
            action_spaces[valid_action] = []
            action_spaces[valid_action].extend(all_pos)

        # handle the Trash action spaces
        action_spaces[(24, 0)] = []
        action_spaces[(24, 0)].extend(all_pos)

        cutter_act = self.get_possible_cutter_actions()
        for action in cutter_act:
            act, pos = action
            if act not in action_spaces:
                action_spaces[act] = []
            action_spaces[act].append(pos)
        return action_spaces

    def create_action_space(self):
        """
        Creates the complete action space for the environment by first generating valid actions
        for different machines and then constructing a list of all possible actions.

        This function maps each valid action (represented as a tuple of machine type,direction and position)
        into an index-based variable for efficient access during the reinforcement learning process.

        The action space is defined as a discrete space where each action corresponds to a unique
        index from the action list.
        """
        valid_actions = self.create_valid_action_space()
        self.action_list = [
            (action_type, pos)
            for action_type, positions in valid_actions.items()
            for pos in positions
        ]
        self.action_space = spaces.Discrete(len(self.action_list))
        return

    def extract_buildings(self, position):
        # extract the machine type and direction in the specific position
        machine_type = self.grid_bld[position] // 100
        direction = self.grid_bld[position] % 100
        return machine_type, direction

    def get_start_pos(self, position, direction):
        # retreive back to the start position from current position
        cur_pos = position
        cur_dir = direction
        pre_pos = self._get_pre_position(cur_pos, cur_dir)
        if pre_pos == None:  # find the bound
            return cur_pos
        pre_dir = self.extract_buildings(pre_pos)[1]
        while True:
            if pre_pos == None:  # find the bound
                return cur_pos
            if self._get_next_position(pre_pos, pre_dir) != cur_pos:  # find the first position
                return cur_pos
            machine_type = self.extract_buildings(pre_pos)[0]
            if machine_type == 22 or machine_type == 24:
                return pre_pos
            elif self.grid_bld[pre_pos] == -1:  # no more building in the pre_pos
                return pre_pos
            cur_pos = pre_pos
            cur_dir = pre_dir
            pre_pos = self._get_pre_position(cur_pos, cur_dir)
            pre_dir = self.extract_buildings(pre_pos)

    def retrieve_path(self, position, direction):
        """
            Retrieves the path starting from a given position and direction by following the machines on the grid.
            This function traces the path until it either reaches a stopping condition (a machine type that isn't
            a conveyor or cutter, or a machine that serves as a destination like Trash or Miner) or detects a loop.
            Args:
                position (tuple): The starting position on the grid.
                direction (int): The initial direction of movement from the starting position.
            Returns:
                list: A list of visited positions that represent the path followed by the function.
            """
        visited = []
        cur_pos = position
        cur_direct = direction
        while True:
            if cur_pos in self.machines:
                visited.append(cur_pos)
            if cur_pos == None:
                break
            if (self.grid_bld[cur_pos] // 100 != 31 and self.grid_bld[cur_pos] // 100 != 23) or self.grid_bld[
                cur_pos] // 100 == 24 or self.grid_bld[cur_pos] // 100 == 22:
                # find the destination
                break
            cur_pos = self._get_pre_position(cur_pos, cur_direct)
            cur_direct = self.grid_bld[cur_pos] % 100
            if cur_pos in visited:
                break
        return visited

    def check_valid_shape(self, shape):
        """
          Checks whether the given shape matches the target shape or a possible variation of the target shape.
          the possible variation means, for example, if the target shape is half circle, and current shape is circle,
          it will still return True because it is a "potential" correct shape.
          Args:
              shape (int): The shape to check against the target shape.

          Returns:
              bool: True if the shape is valid according to the target shape rules, otherwise False.
          """
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

    def calculate_miner_reward(self, position, direction):
        """
        calculate the miner reward based on the position and direction related to the closet hub
        :param position: the extractor's placing position
        :param direction: the extractor's direction
        :return: the reward of the placing action
        """
        hub_positions = np.argwhere(self.grid_bld // 100 == 21)
        if hub_positions.size > 0:
            hub_pos = hub_positions[0]

        hub_pos = np.argwhere(self.grid_bld // 100 == 21)[0]
        x_pos = True if position[0] < hub_pos[0] else False
        y_pos = True if position[1] < hub_pos[1] else False
        dis = (hub_pos[0] - position[0]) ** 2 + (hub_pos[1] - position[1]) ** 2
        max_distance = (self.grid_bld.shape[0] ** 2) * 2
        distance_reward = max_distance - dis
        normalized_reward = distance_reward / max_distance
        distance_reward = normalized_reward * 20
        direction_reward = 0

        if x_pos is True and y_pos is True:
            if direction == 2 or direction == 4:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos is False and y_pos is True:
            if direction == 1 or direction == 4:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos is True and y_pos is False:
            if direction == 2 or direction == 3:
                direction_reward = 10
            else:
                direction_reward = -10
        elif x_pos is False and y_pos is False:
            if direction == 1 or direction == 3:
                direction_reward = 10
            else:
                direction_reward = -10
        return distance_reward + direction_reward

    def calculate_conveyor_reward(self, position, direction):
        """
         calculate the conveyor belt reward based on its position and direction
         if it is getting closer to the hub and delivering the correct or potential correct shape,
          return a positive reward,else return a negative reward
        :param position:the position that place the building
        :param direction:the direction of the belt
        :return:reward
        """
        reward = 0
        path = self.retrieve_path(position, direction)
        start = path[-1]
        current_shape = None
        if self.grid_bld[start] // 100 != 22:
            # not connected to the start
            return -5

        for p in reversed(path):
            if isinstance(self.machines[p], Miner):
                current_shape = self.grid_rsc[p]
            elif isinstance(self.machines[p], Cutter):
                shapes = process_cut(current_shape)
                if shapes[0] != -1:
                    # valid cutter
                    if self.machines[p].position == p:
                        # main pos
                        current_shape = shapes[0]
                    else:
                        # sub_pos
                        current_shape = shapes[1]

        hub_pos = self.find_closet_hub(position)
        next_pos = self._get_next_position(position, direction)
        if hub_pos == None or next_pos == None:
            # no valid closet hub meaningless to continue placing belt
            return -5
        if self.check_valid_shape(current_shape):
            # conveying the correct shape
            reward += (self.grid_bld.shape[0] - distance(hub_pos, position)) / 2
        else:
            return -20
        if distance(hub_pos, position) < distance(hub_pos, next_pos):
            reward = -5
        else:
            # closer to the hub
            reward += 5
        return reward

    def calculate_trash_reward(self, position):
        """
        calculate the trash reward, this building is only useful when delivering a wrong shape
        :param position: the postion that place the building
        :return: the reward that place the building
        """
        start = self._get_pre_position(position, 0)
        reward = 0
        for pos in start:
            if pos is None or self.grid_bld[pos] == -1:
                continue
            path = self.retrieve_path(pos, self.grid_bld[pos] % 100)
            current_shape = None

            for p in reversed(path):
                if isinstance(self.machines[p], Miner):
                    current_shape = self.grid_rsc[p]
                elif isinstance(self.machines[p], Cutter):
                    shapes = process_cut(current_shape)
                    if shapes[0] != -1:
                        # valid cutter
                        if self.machines[p].position == p:
                            # main pos
                            current_shape = shapes[0]
                        else:
                            current_shape = shapes[1]

            if self.check_valid_shape(current_shape) or current_shape is None:
                reward -= 50
            else:
                reward += 50
        return reward

    def calculate_cutter_reward(self, position, direction):
        """
        calculate the cutter's reward, if it cut the correct shape, return a positive value
        :param position: the main position of the cutter
        :param direction: the direction of the buidling
        :return: the reward of placing the building
        """
        path = self.retrieve_path(position, direction)

        current_shape = None
        for pos in path:
            if isinstance(self.machines[pos], Miner):
                current_shape = self.grid_rsc[pos]
            if isinstance(self.machines[pos], Cutter):
                if self.machines[pos].position == pos:
                    # main exit:
                    current_shape = process_cut(current_shape)[0]
                else:
                    # sub exit:
                    current_shape = process_cut(current_shape)[1]
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
            machine_type, direction = self.extract_buildings(position)
            if machine_type == 23:
                # cutter
                main_pos, sub_pos = self.get_cutter_pos(position, direction)
                self.grid_bld[main_pos] = -1
                self.grid_bld[sub_pos] = -1
                del self.machines[main_pos]
                del self.machines[sub_pos]
            elif machine_type == 31:
                path = self.retrieve_path(position, self.grid_bld[position] % 100)
                start_pos = path[-1]
                start_shape = self.grid_rsc[start_pos]
                if self._track_path_with_rotator(start_pos, start_shape) == 'hub':
                    return -1
                else:
                    self.grid_bld[position] = -1
                    reward = -self.reward_grid[position] \
                        if self.reward_grid[position] > 0 else -(self.reward_grid)[position] / 2
                    del self.machines[position]
                    self.reward_grid[position] = -1
            else:
                self.grid_bld[position] = -1
                reward = -self.reward_grid[position] \
                    if self.reward_grid[position] > 0 else -self.reward_grid[position] / 2
                del self.machines[position]
                self.reward_grid[position] = -1
            return reward
        elif machine_type == 22:  # placing miner

            reward = self.calculate_miner_reward(position, direction)
            self.reward_grid[position] = reward
            self.grid_bld[position] = 22 * 100 + direction
            new_machine = Miner(position, direction)
            self.machines[position] = new_machine
        elif machine_type == 31:  # placing conveyor
            self.grid_bld[position] = 31 * 100 + direction
            new_machine = Conveyor(position, direction)
            self.machines[position] = new_machine
            self.reward_grid[position] = reward
            reward = self.calculate_conveyor_reward(position, direction)
        elif machine_type == 24:
            # place trash
            new_machine = Trash(position, direction)
            self.machines[position] = new_machine
            self.reward_grid[position] = reward
            self.grid_bld[position] = 24 * 100
            reward = self.calculate_trash_reward(position)
        elif machine_type == 23:
            # place cutter
            new_machine = Cutter(position, direction)
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
        """
        get the next position of the building in pos with direction
        :return: the next position of the current building, if it is out of the bound, return None
        """
        x, y = position
        if (direction == 1 or direction == 9 or direction == 10) and x - 1 >= 0:
            return x - 1, y
        elif (direction == 4 or direction == 6 or direction == 8) and y + 1 < self.grid_rsc.shape[1]:
            return x, y + 1
        elif (direction == 2 or direction == 11 or direction == 12) and x + 1 < self.grid_rsc.shape[0]:
            return x + 1, y
        elif (direction == 3 or direction == 5 or direction == 7) and y - 1 >= 0:
            return x, y - 1
        return None

    def handle_pre_direction(self, position, direction):
        """
        return the previous position for the specifc machine
        :param position:
        :param direction:
        :return:
        """
        x, y = position
        if (direction == 1 or direction == 5 or direction == 6) and x + 1 < self.grid_rsc.shape[0]:
            return x + 1, y
        elif (direction == 2 or direction == 7 or direction == 8) and x - 1 >= 0:
            return x - 1, y
        elif (direction == 3 or direction == 9 or direction == 11) and y + 1 < self.grid_rsc.shape[1]:
            return x, y + 1
        elif (direction == 4 or direction == 10 or direction == 12) and y - 1 >= 0:
            return x, y - 1
        return None

    def _get_pre_position(self, position, direction):
        """
        return the previous position of all machines,
        if it is a trash,it will return 4 positions because the trash can receive the shapes from 4 directions.
        :param position:
        :param direction:
        :return:
        """
        possible_pos = []
        # print(direction)
        if direction == 0:
            for direct in range(4):
                pre_pos = self.handle_pre_direction(position, direct + 1)
                if pre_pos is None:
                    continue
                else:
                    if self.grid_bld[pre_pos] != -1:
                        if self._get_next_position(pre_pos, self.grid_bld[pre_pos] % 100) == position:
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

            return self.handle_pre_direction(position, direction)

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
            previous_position = pre_positions

            current_position = previous_position
            current_direction = self.grid_bld[current_position] % 100

            if self.grid_bld[current_position] // 100 == 22:
                return current_position

    def get_possible_cutter_actions(self):
        rows, cols = self.grid_bld.shape
        possible_actions = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols and self.grid_bld[r, c] // 100 != 21 and self.grid_bld[r, c + 1] // 100 != 21:
                    # 水平放置，direction为 1 或者 2,
                    possible_actions.append(((23, 1), (r, c)))
                if c - 1 >= 0 and self.grid_bld[r, c] // 100 != 21 and self.grid_bld[r, c - 1] // 100 != 21:
                    possible_actions.append(((23, 2), (r, c)))
                if r - 1 >= 0 and self.grid_bld[r, c] // 100 != 21 and self.grid_bld[r - 1, c] // 100 != 21:
                    possible_actions.append(((23, 3), (r, c)))
                if r + 1 < rows and self.grid_bld[r, c] // 100 != 21 and self.grid_bld[r + 1, c] // 100 != 21:
                    possible_actions.append(((23, 4), (r, c)))
        return possible_actions

    def _is_first_building(self, position, direction):
        # check if the position is the first conyeyor of the path and connected to the miner
        if self._get_next_position(position, direction) == None:  # next pos out of bound
            return True
        else:
            machine_type, direction = self.extract_buildings(position)
            if machine_type == 24:
                # reached the trash
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

        pre_direct = self.machines[pre_pos].direction
        start_pos = self.retrieve_path(pre_pos, pre_direct)[-1]
        if self.grid_rsc[start_pos] == 0 or self._get_next_position(pre_pos, pre_direct) != position:
            return False
        # handle nxt_pos situation
        nxt_pos = self._get_next_position(position, direction)
        if nxt_pos == None:  # next position out of bound
            return False
        if self.grid_bld[nxt_pos] == -1 or self.grid_bld[nxt_pos] // 100 == 21:
            return True
        else:
            # next postion has buildings
            nxt_direct = self.grid_bld[nxt_pos] % 100
            if self._get_pre_position(nxt_pos, nxt_direct) == position:
                # can connected
                return True
            else:
                return False

    def CanPlaceMiner(self):
        correct_miner_num = 0
        for pos, machine in self.machines.items():
            if isinstance(machine, Miner) and self.check_valid_shape(self.grid_rsc[machine.position]) is True:
                correct_miner_num += 1
        if correct_miner_num < self.required_routes:
            return True
        else:
            return False

    def can_remove(self, position):
        # check if the remove action is valid in position
        # param:position, represents the target delete position
        if self.grid_bld[position] // 100 == 24:
            return True
        if self.grid_bld[position] == -1 or self.grid_bld[
            position] // 100 == 21:  # no building can't remove or the building is destination
            return False
        # otherwise, there is a buidling except destination
        machine_type, direction = self.extract_buildings(position)

        nxt_pos = self._get_next_position(position, direction)
        if nxt_pos is not None:
            if self.grid_bld[nxt_pos] // 100 == 21:
                return True
            else:
                return self._is_first_building(position, direction)
        else:
            # out of bound
            return False

    def CanPlaceCutter(self, position, direction):
        x, y = position
        direction_map = {
            1: (0, 1),
            2: (0, -1),
            3: (-1, 0),
            4: (1, 0)
        }
        dx, dy = direction_map[direction]
        main_pos = position
        sub_pos = (x + dx, y + dy)
        if self.grid_bld[main_pos] != -1 or self.grid_bld[sub_pos] != -1 or self.grid_bld[sub_pos] // 100 == 21 or \
                self.grid_bld[main_pos] // 100 == 21:
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
        path = self.retrieve_path(pre_pos, direction)

        target_shape = self.grid_rsc[path[-1]]
        if target_shape == self.target_shape:
            return False
        if self.target_shape not in process_cut(target_shape):
            return False
        if self.grid_bld[nxt_pos] == -1 or self.grid_bld[nxt_pos] // 100 == 21:
            return True
        else:
            return False

    def check_action_valid(self, machine_type, position, direction):
        if machine_type == 0:
            return self.can_remove(position)
        elif machine_type == 31:
            if self.CanPlaceConveyor(position, direction):
                return True
        elif machine_type == 23:
            if self.CanPlaceCutter(position, direction):
                return True
            else:
                return False
        elif machine_type == 22:
            # miner
            return self.check_valid_shape(self.grid_rsc[position])
        else:
            # handle other machines
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
                    idx = self.act_dict[(22, direction + 1), (pos[0], pos[1])]
                    index.append(idx)

        all_machine_pos = [key for key, machine in self.machines.items() if not isinstance(machine, Hub)]
        all_machine_pos = np.array(all_machine_pos)

        for pos in all_machine_pos:
            idx = self.act_dict[(0, -1), (pos[0], pos[1])]
            if idx in index:
                print("duplicated!!!!")
                sys.exit()
            index.append(idx)

            direct = self.machines[(pos[0], pos[1])].direction
            if self._is_first_building((pos[0], pos[1]), direct) == True:
                next_pos = self._get_next_position((pos[0], pos[1]), direct)
                if next_pos is None or self.grid_rsc[next_pos] != 0:
                    continue
                for direction in range(12):
                    if direction < 4:
                        if ((23, direction + 1), next_pos) in self.act_dict:
                            idx = self.act_dict[(23, direction + 1), next_pos]
                            index.append(idx)
                    idx = self.act_dict[(31, direction + 1), next_pos]
                    index.append(idx)
                # place possible trash
                idx = self.act_dict[(24, 0), next_pos]
                index.append(idx)
        return index

    def get_action_mask(self):

        mask = [0] * len(self.action_list)
        possible_action_idx = self.get_possible_action_idx()
        for idx in possible_action_idx:
            action = self.action_list[idx]
            machine_type, direction = action[0]
            position = (action[1][0], action[1][1])
            if self.check_action_valid(machine_type, position, direction):
                mask[idx] = 1
            else:
                mask[idx] = 0

        return mask

    def find_closet_hub(self, cur_pos):
        min_distance = 0x3f3f3f
        closet_pos = None
        hub_poses = np.argwhere(self.grid_bld // 100 == 21)
        for hub_pos in hub_poses:
            hub_x, hub_y = hub_pos
            distance = (cur_pos[0] - hub_x) ** 2 + (cur_pos[1] - hub_y) ** 2
            if distance < min_distance:
                if (self.grid_bld[hub_x - 1][hub_y] == -1 or self.grid_bld[hub_x + 1][hub_y] == -1 or
                        self.grid_bld[hub_x][hub_y - 1] == -1 or self.grid_bld[hub_x][hub_y + 1] == -1):
                    min_distance = distance
                    closet_pos = (hub_x, hub_y)
        return closet_pos

    def get_cutter_pos(self, position, direction):
        main_pos = None
        sub_pos = None
        if self.machines[position].position == position:
            # main pos
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
        return main_pos, sub_pos

    def _track_path_with_rotator(self, position, shape):

        cur_pos = position
        rotation_count = 0  # 初始化旋转次数
        while True:
            nxt_pos = self._get_next_position(cur_pos, self.machines[cur_pos].direction)
            if nxt_pos is None or self.grid_bld[nxt_pos] == -1:
                return 'none'
            if nxt_pos in self.machines:
                current_machine = self.machines[nxt_pos]
                if isinstance(current_machine, Conveyor):
                    # nxt_direct = self.grid_bld[nxt_pos]%100
                    # if self._get_pre_position(nxt_pos,nxt_direct) != cur_pos:
                    #     print("false")
                    #     return 'none'
                    cur_pos = nxt_pos
                elif isinstance(current_machine, Rotator):
                    rotation_count += 1
                    cur_pos = nxt_pos
                elif isinstance(current_machine, Cutter):
                    cur_pos = nxt_pos
                    main_exit_pos, side_exit_pos = self.get_cutter_pos(cur_pos, current_machine.direction)
                    main_exit_shape, side_exit_shape = process_cut(shape)

                    if main_exit_pos in self.machines:
                        direction = self.machines[main_exit_pos].direction
                        nxt_pos = self._get_next_position(main_exit_pos, direction)
                        machine_type, direction = self.extract_buildings(nxt_pos)
                        if machine_type == 31 and self._get_pre_position(nxt_pos, direction) == main_exit_pos:
                            main_exit_result = self._track_path_with_rotator(main_exit_pos, main_exit_shape)
                        else:
                            main_exit_result = 'none'
                    else:
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
                        side_exit_result = 'none'

                    if main_exit_result == 'hub' or side_exit_result == 'hub':
                        return 'hub'
                    else:
                        return 'none'

                elif isinstance(current_machine, Hub):
                    rotated_shape = process_rotate(shape, rotation_count)

                    if rotated_shape == self.target_shape:
                        return 'hub'
                    else:
                        return 'none'

                elif isinstance(current_machine, Trash):
                    return 'trash'

                else:
                    return 'none'
            else:
                return 'none'
