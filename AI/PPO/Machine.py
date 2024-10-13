class Machine:
    def __init__(self, position, direction=-1, shape=None, num=0):
        self.position = position
        self.direction = direction
        self.shape = shape  # 添加 shape 属性
        self.num = num      # 添加 num 属性
        self.type = -1


class Miner(Machine):
    def __init__(self, position, direction, shape=0, num=0):
        super().__init__(position, direction, shape, num)
        self.type = 22


class Cutter(Machine):
    def __init__(self, position, direction, shape=0, num=0):
        super().__init__(position, direction, shape, num)
        self.type = 23


class Trash(Machine):
    def __init__(self, position, direction, shape=0, num=0):
        super().__init__(position, direction, shape, num)
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

    def __init__(self, position, direction, shape=0, num=0):
        super().__init__(position, direction, shape, num)
        self.type = 31
        self.direction = direction