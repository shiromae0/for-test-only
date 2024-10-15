import pyautogui
import keyboard
import time
# from start import find_min_location_and_crop
from PIL import Image
from getmap import load_scroll_offset,load_scaleFactor
from PPO import get_agent_act_list
from visualize import visualize
scroll_offset = load_scroll_offset()
# if scroll_offset[0] == 0 and scroll_offset[1] == 0:
#     scroll_offset[0] = 200
#     scroll_offset[1] = 85
scaleFactor = load_scaleFactor()
scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
# cell_size = 50* scaleFactor
Width=1200
Height=730
Horizontal = 32
verticle =20
# machine_list  = [
#     ((31, 1), (6, 5)),
#     ((31, 2), (7, 4)),
#     ((31, 3), (10, 8)),
#     ((31, 4), (15, 21)),
#     ((31, 5), (14, 12)),
#     ((31, 12), (3, 19)),
#     ((31, 1), (5, 10)),
#     ((31, 2), (15, 4)),
#     ((31, 3), (17, 0)),
#     ((31, 4), (12, 15)),
#     ((31, 5), (7, 2)),
#     ((31, 6), (3, 11)),
#     ((31, 7), (16, 19)),
#     ((31, 8), (6, 9)),
#     ((31, 9), (14, 17)),
#     ((31, 10), (8, 3)),
#     ((31, 11), (11, 12)),
#     ((31, 12), (1, 16)),
# ]


# machine_list = [
#     ((23, 2), (15, 18)),
#     ((24, 0), (10, 10)),
#     ((23, 3), (8, 2)),
#     ((23, 4), (2, 2)),
#     ((0, -1), (19, 5)),
#     ((23, 2), (8, 8)), 
#     ((24, 0), (12, 14)),
# ]
data_list =get_agent_act_list()
#
# array = return_array()
# class Machine:
#     def __init__(self, machine_type, direction):
#         self.type = machine_type
#         self.direction = direction
# machine_position = {}
# path = []
#
# def get_machine_position():
#     global machine_position
#     for x in range(array.shape[0]):  # 遍历每一行
#         for y in range(array.shape[1]):  # 遍历每一列
#             value = array[x, y]
#             if value != -1:  # 如果不是空值
#                 # 提取前两位作为类型，后两位作为方向
#                 type_code = value // 100  # 例如 2201 // 100 = 22
#                 direction = value % 100   # 例如 2201 % 100 = 01
#                 if type_code == 22 or type_code == 23 or type_code == 24 or type_code == 31 or type_code == 21:
#                     # 将 (x, y) 坐标和对应的 Machine(type, direction) 存入字典
#                     machine_position[(x, y)] = Machine(type_code, direction)
#
# def calculate_scale_count(scaleFactor):
#     difference = abs(scaleFactor - 1.0)
#     scale_count = int(difference / 0.1)
#     return scale_count
#
# def scale_event(scaleFactor):
#    
#     if scaleFactor == 1.0:
#         print("scaleFactor 为 1.0,无需滚动")
#         return  # 直接返回，不进入滚动逻辑
#
#     count=calculate_scale_count(scaleFactor)
#     print(f"scale count: {count}")
#     for _ in range(count):
#         if scaleFactor > 1.0:
#             pyautogui.scroll(-120)
#         elif scaleFactor < 1.0:
#             pyautogui.scroll(120)
#

def calculate_scale_count():
    difference = abs(scaleFactor - 1.0)
    scale_count = round(difference / 0.1)
    return scale_count

def scale_event():
    global scaleFactor
    time.sleep(2)
    if scaleFactor == 1.0:
        print("scaleFactor 为 1.0,无需滚动")
        return  # 直接返回，不进入滚动逻辑
    
    count=calculate_scale_count()
    print(f"scale count: {count}")
    for _ in range(count):
        if scaleFactor > 1.0:
            pyautogui.scroll(-120)
        elif scaleFactor < 1.0:
            pyautogui.scroll(120)
    scaleFactor = load_scaleFactor()

# def start_place(image):
#     # 如果 position_flag 为 False 且 start_position 还没有赋值
#     start_position = find_min_location_and_crop(image)
#     return start_position

def revise_direction(direction):
    print(f"direction: {direction}")
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
    if direction == 1:
        keyboard.press("w")  # Up
    elif direction == 2:
        keyboard.press("s")  # Down
    elif direction == 3:
        keyboard.press("a")  # Left
    elif direction == 4:
        keyboard.press("d")  # Right
    #define UP_LEFT 5
    #define UP_RIGHT 6
    #define DOWN_LEFT 7
    #define DOWN_RIGHT 8
    #define LEFT_UP 9
    #define RIGHT_UP 10
    #define LEFT_DOWN 11
    #define RIGHT_DOWN 12
    elif direction == 5:
        keyboard.press("3")
    elif direction == 6:
        keyboard.press("1")
    elif direction == 7:
        keyboard.press("7")
    elif direction == 8:
        keyboard.press("5")
    elif direction == 9:
        keyboard.press("4")
    elif direction == 10:
        keyboard.press("2")
    elif direction == 11:
        keyboard.press("8")
    elif direction == 12:
        keyboard.press("6")

def click_belt():
    tool_location = pyautogui.locateOnScreen('belt.png', confidence=0.7)
    if tool_location is not None:
        tool_center = pyautogui.center(tool_location)
        pyautogui.moveTo(tool_center, duration=0.5)
        pyautogui.click()
    else:
        print("Belt tool not found on the screen.")

def click_miner():
    tool_location = pyautogui.locateOnScreen('miner.png', confidence=0.7)
    if tool_location is not None:
        tool_center = pyautogui.center(tool_location)
        pyautogui.moveTo(tool_center, duration=0.5)
        pyautogui.click()
    else:
        print("Miner tool not found on the screen.")

def click_cutter():
    tool_location = pyautogui.locateOnScreen('cutter.png', confidence=0.7)
    if tool_location is not None:
        tool_center = pyautogui.center(tool_location)
        pyautogui.moveTo(tool_center, duration=0.5)
        pyautogui.click()
    else:
        print("Cutter tool not found on the screen.")

def click_trash():
    tool_location = pyautogui.locateOnScreen('trash.png', confidence=0.7)
    if tool_location is not None:
        tool_center = pyautogui.center(tool_location)
        pyautogui.moveTo(tool_center, duration=0.5)
        pyautogui.click()
    else:
        print("Trash tool not found on the screen.")

def get_position(i,j,cell_size,start_position):
    global scroll_offset
    global scroll_offset_scaled
    scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
    scroll_offset_x,scroll_offset_y=scroll_offset_scaled
    print("j:", j)
    print("i:", i)
    print("scroll_offset_x:", scroll_offset_x)
    print("scroll_offset_y:", scroll_offset_y)
    scaled_x = (j * cell_size) + scroll_offset_x
    scaled_y = (i * cell_size) + scroll_offset_y
    print("scaled_x:", scaled_x)
    print("scaled_y:", scaled_y)
    relative_position=(j * cell_size) + cell_size/2,(i * cell_size) + cell_size/2
    position=start_position[0]+scaled_x+cell_size/2,start_position[1]+ scaled_y+cell_size/2
    print(f"position: {position}")
    if(position[0] > start_position[0] + 1200) or (position[1] > start_position[1] + Height) or (position[0] < start_position[0]) or (position[1] < start_position[1]):
        print(f"调用了")
        print(f"scroll_offset: {scroll_offset}")
        print(f"relative_position: {relative_position}")

        if (relative_position[0] > 600 and relative_position[0] < Horizontal * cell_size - 600) and (relative_position[1] > (Height/2) and relative_position[1] < verticle * cell_size - (Height/2)):
            print(f"调用了可拖拽到中心")
            true_position = drag(position, start_position)
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            print(f"true_position: {true_position}")
            return true_position

        if (relative_position[0] < 600 or relative_position[0] > Horizontal * cell_size - 600) and (relative_position[1] > (Height/2) and relative_position[1] < verticle * cell_size - (Height/2)):
            print(f"调用了左右")
            true_position = drag(position, start_position)
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            true_position = (start_position[0] + relative_position[0] + (scroll_offset_scaled[0]), true_position[1])
            print(f"true_position: {true_position}")
            return true_position

        if (relative_position[1] < (Height/2) or relative_position[1] > verticle * cell_size - (Height/2)) and (relative_position[0] > 600 and relative_position[0] < Horizontal * cell_size - 600):
            print(f"调用了上下")
            true_position = drag(position, start_position)
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            #scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
            print(f"scroll_offset: {scroll_offset}")
            print(f"scaleFactor: {scaleFactor}")
            print(f"scroll_offset_scaled: {scroll_offset_scaled}")
            true_position = (true_position[0] , start_position[1] + relative_position[1] + (scroll_offset_scaled[1]))
            print(f"true_position: {true_position}")
            return true_position

        if (relative_position[0] < 600 and relative_position[1] < (Height/2)) or (relative_position[0] < 600 and relative_position[1] > verticle * cell_size - (Height/2)) or (relative_position[0] > Horizontal * cell_size - 600 and relative_position[1] < (Height/2)) or (relative_position[0] > Horizontal * cell_size - 600 and relative_position[1] > verticle * cell_size - (Height/2)):
            print(f"调用了四角")
            true_position = drag(position, start_position)
            print(f"scroll_offset: {scroll_offset}")
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            true_position = (start_position[0] + relative_position[0] + scroll_offset_scaled[0] , start_position[1] + relative_position[1] + scroll_offset_scaled[1])
            print(f"true_position: {true_position}")
            return true_position


    return position

def drag(position,start_position):
    global scroll_offset
    global scaleFactor
    global scroll_offset_scaled
    center=start_position[0] + Width/2, start_position[1] + Height/2
    position_dragto=(scaleFactor * center[0]+center[0]-position[0])/scaleFactor, (scaleFactor * center[1]+center[1]-position[1])/scaleFactor

    if(position_dragto[0] > start_position[0] + 1200) or (position_dragto[1] > start_position[1] + Height) or (position_dragto[0] < start_position[0]) or (position_dragto[1] < start_position[1]):
        position_dragto_temp = list(position_dragto)
        print(f"position_dragto: {position_dragto}")
        print(f"center: {center}")
        print(f"调用了多次拖动")

        if (position_dragto_temp[0] > center[0]) and (position_dragto_temp[1] > center[1]):
            print(f"调用了多次拖动向右下")
            while (position_dragto_temp[0] > center[0]) or (position_dragto_temp[1] > center[1]):
                print(f"进入了多次拖动向右下")

                pyautogui.moveTo(center)  # 先将鼠标移动到center
                pyautogui.mouseDown()  # 按下鼠标左键


                # 计算每次移动的步长（例如50像素）
                step_x = min(600, position_dragto_temp[0] - center[0])
                step_y = min((Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                # 更新临时位置
                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                # 拖动到新的位置
                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp()  # 松开鼠标

            return center


        if (position_dragto_temp[0] < center[0]) and (position_dragto_temp[1] < center[1]):
            print(f"调用了多次拖动向左上")
            while (position_dragto_temp[0] < center[0]) or (position_dragto_temp[1] < center[1]):
                print(f"进入了多次拖动向左上")
                pyautogui.moveTo(center)  # 先将鼠标移动到center
                pyautogui.mouseDown()  # 按下鼠标左键

                # 计算每次移动的步长（例如50像素）
                step_x = max(-600, position_dragto_temp[0] - center[0])
                step_y = max(-(Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                # 更新临时位置
                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                # 拖动到新的位置
                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp()  # 松开鼠标
            return center

        if (position_dragto_temp[0] > center[0]) and (position_dragto_temp[1] < center[1]):
            print(f"调用了多次拖动向右上")
            while (position_dragto_temp[0] > center[0]) or (position_dragto_temp[1] < center[1]):
                print(f"进入了多次拖动向右上")
                pyautogui.moveTo(center)  # 先将鼠标移动到center
                pyautogui.mouseDown()  # 按下鼠标左键

                # 计算每次移动的步长（例如50像素）
                step_x = min(600, position_dragto_temp[0] - center[0])
                step_y = max(-(Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                # 更新临时位置
                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                # 拖动到新的位置
                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp()  # 松开鼠标
            return center

        if (position_dragto_temp[0] < center[0]) and (position_dragto_temp[1] > center[1]):
            print(f"调用了多次拖动向左下")
            while (position_dragto_temp[0] < center[0]) or (position_dragto_temp[1] > center[1]):
                print(f"进入了多次拖动向左下")
                pyautogui.moveTo(center)  # 先将鼠标移动到center
                pyautogui.mouseDown()  # 按下鼠标左键

                # 计算每次移动的步长（例如50像素）
                step_x = max(-600, position_dragto_temp[0] - center[0])
                step_y = min((Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                # 更新临时位置
                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                # 拖动到新的位置
                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp()  # 松开鼠标
            return center

    else:
        pyautogui.moveTo(center)  # 先将鼠标移动到center
        pyautogui.mouseDown()  # 按下鼠标左键
        pyautogui.dragTo(position_dragto[0], position_dragto[1], duration=1)  # 将鼠标拖动到目标位置
        pyautogui.mouseUp()  # 松开鼠标
        scroll_offset = load_scroll_offset()
        scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
        return center

def place_object(data_list, cell_size, start_position):
    # for pos, machine in machine_position.items():
    #     # 计算放置位置
    #     position = get_position(pos[0], pos[1], cell_size, start_position, scroll_offset_scaled)
    #
    #     # 根据 machine.type 调用相应的点击函数
    #     if machine.type == 22:
    #         click_miner()
    #     elif machine.type == 23:
    #         click_cutter()
    #     elif machine.type == 24:
    #         click_trash()
    #
    #     # 调整方向并放置物体
    #     revise_direction(machine.direction)
    #     pyautogui.moveTo(position, duration=2)
    #     pyautogui.click()
    #
    #     if machine.type == 0:
    #         pyautogui.moveTo(position, duration=2)
    #         pyautogui.click(button='right')
    for item in data_list:
        (machine_type, direction),(x, y) = item  # 解包列表中的每个元组

        # 计算放置位置
        position = get_position(x, y, cell_size, start_position)

        # 根据 machine_type 调用相应的点击函数
        if machine_type == 22:
            click_miner()
        elif machine_type == 23:
            click_cutter()
        elif machine_type == 24:
            click_trash()
        elif machine_type == 31:
            click_belt()

        # 调整方向并放置物体
        
        pyautogui.moveTo(position, duration=0.5)
        revise_direction(direction)
        pyautogui.click()

        if machine_type == 0:
            pyautogui.moveTo(position, duration=0.5)
            pyautogui.click(button='right')


def place_single_object( data_list, cell_size):
     # 计算 start_position
     start_position = visualize()
     # 调用 place_object 将对象放置在适当位置
     place_object(data_list, cell_size, start_position)

def run_place_single_object():
    # 初始化变量
    global scaleFactor
    global scroll_offset
    global scroll_offset_scaled
    scaleFactor = load_scaleFactor()
    scale_event()
    # image = Image.open('star.png')
    start_position = visualize()
    center=start_position[0] + Width/2, start_position[1] + Height/2
    print(center)
    pyautogui.moveTo(center, duration=2)
    pyautogui.click()
    print("start_position:", start_position)
    cell_size = 50 * scaleFactor
    scroll_offset = load_scroll_offset()
    scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

    # 假设 machine_position 是定义好的字典，存储了机器的位置和类型
    #machine_position = {}

    # # 遍历二维数组
    # for x in range(array.shape[0]):  # 遍历每一行
    #     for y in range(array.shape[1]):  # 遍历每一列
    #         value = array[x, y]
    #         if value != -1:  # 如果不是空值
    #             # 提取前两位作为类型，后两位作为方向
    #             type_code = value // 100  # 例如 2201 // 100 = 22
    #             direction = value % 100   # 例如 2201 % 100 = 01
    #
    #             if type_code == 22 or type_code == 23 or type_code == 24 or type_code == 0:
    #                 # 将 (x, y) 坐标和对应的 Machine(type, direction) 存入字典
    #                 machine_position[(x, y)] = Machine(type_code, direction)
    #                 # machine_position[(x, y)] = f"Machine({type_code}, {direction})"
    #
    # # 输出 machine_position 字典
    # print(machine_position)

    # 调用 place_single_object 函数
    place_single_object( data_list, cell_size)

def main():
    run_place_single_object()
    # image =Image.open("star.png")
    # start_position = start_place(image)
    # print("start_position:", start_position)
    # print("下面是传送带拖动")
    # coordinates = return_path()
    # print(coordinates)
    # # # 获取所有矿机的传送路径
    # # all_paths = return_all_paths()
    # start_drag = coordinates[0]
    # end_drag = coordinates[-1]
    # scaleFactor = load_scaleFactor()
    # scroll_offset = load_scroll_offset()
    # print("scaleFactor:", scaleFactor)
    # print("scroll_offset:", scroll_offset)
    # cell_size = 50 * scaleFactor
    # # 对每个矿机路径执行处理
    # # for coordinates in all_paths:
    # #     #run_drag_belt_process(coordinates)
    # #     start_drag = coordinates[0]
    # #     end_drag = coordinates[-1]
    # #     drag_place_belt(coordinates, start_drag, end_drag, cell_size, start_position)
    # drag_place_belt(coordinates,start_drag,end_drag,cell_size,start_position)

main()



