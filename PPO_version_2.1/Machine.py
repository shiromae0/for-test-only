class Machine:
    def __init__(self, position, direction=-1):
        self.position = position
        self.direction = direction
        self.type = -1


class Miner(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 22

class Rotator(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 27
class Cutter(Machine):
    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 23
        x, y = self.position
        direction_map = {
            1: (0, 1),  # 向上，副出口在下
            2: (0, -1),  # 向下，副出口在上
            3: (-1, 0),  # 向左，副出口在右
            4: (1, 0)  # 向右，副出口在左
        }

        # 通过方向查找对应的坐标偏移
        if self.direction in direction_map:
            dx, dy = direction_map[self.direction]
            self.sub_pos = (x + dx, y + dy)
        else:
            print("error,direction = ",direction)


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

    def __init__(self, position, direction):
        super().__init__(position, direction)
        self.type = 31
        self.direction = direction