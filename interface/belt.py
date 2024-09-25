import pyautogui
import time
from start import find_min_location_and_crop
from PIL import Image
from getmap import load_scroll_offset,load_scaleFactor


#image_files = [Image.open(f"{i}.png") for i in range(1, 5)]
#start_position = find_min_location_and_crop(image_files)
#print("start_position:",start_position)
#scroll_offset = (-50,-50)
#directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#coordinates = [(12,2),(11,2),(10,2),(9,2),(8,2),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10), (7, 11), (7,12)]
#coordinates = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),  (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18),  (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27),  (4, 28), (4, 29)]
coordinates = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
(4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19),
(4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29),
(5, 29), (6, 29), (7, 29), (8, 29), (9, 29), (10, 29), (11, 29), (12, 29), (13, 29), (14, 29),
(14, 30), (14, 31), (14, 32), (14, 33), (14, 34), (14, 35), (14, 36), (14, 37), (14, 38), (14, 39),
(14, 40), (14, 41), (14, 42), (14, 43), (14, 44), (14, 45), (14, 46), (14, 47), (14, 48), (14, 49),
(13, 49), (12, 49), (11, 49), (10, 49), (9, 49), (8, 49), (7, 49), (6, 49), (5, 49)
]
# start_drag = (12, 2)
end_drag=(5, 49)
#end_drag = (7, 12) 
scroll_offset = load_scroll_offset()
print("scroll_offset:",scroll_offset)
scaleFactor = load_scaleFactor()
#scaleFactor = round(scaleFactor_load, 1)
print("scaleFactor:",scaleFactor)
scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
cell_size = 50* scaleFactor
Width=1200
Height=830

matrix= [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 5, 1, 1, 3, 3, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 3, 3, 3, 1, 1],
        [1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1],
        [1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 34, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21, 21, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 21, 21, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1],
        [1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5],
        [1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 3, 3, 1, 1, 1, 5, 1, 5, 5, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 3, 3, 22, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 3, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


#initialize
# def scroll_event(scroll_offset):
#     time.sleep(2)
#     scroll_x,scroll_y=scroll_offset
#     position=pyautogui.position()#鼠标当前位置
#     end=position[0]-scroll_x,position[1]-scroll_y
#     pyautogui.mouseDown
#     pyautogui.dragTo(end, duration=0.5)
#     scroll_offset=(0,0)
#     return scroll_offset

# scroll_offset = scroll_event(scroll_offset)
# print("scroll_offset",scroll_offset)

def calculate_scale_count(scaleFactor):
    difference = abs(scaleFactor - 1.0)
    scrale_count = int(difference / 0.1)
    return scrale_count

def scale_event(scaleFactor):
    time.sleep(2)
    if scaleFactor == 1.0:
        print("scaleFactor 为 1.0,无需滚动")
        return  # 直接返回，不进入滚动逻辑
    
    count=calculate_scale_count(scaleFactor)
    print(f"scale count: {count}")
    for _ in range(count):
        if scaleFactor > 1.0:
            pyautogui.scroll(-120)
        elif scaleFactor < 1.0:
            pyautogui.scroll(120)
    scaleFactor = 1.0

#scaleFactor = scale_event(scaleFactor)


def start_place(image):
    # 如果 position_flag 为 False 且 start_position 还没有赋值
    start_position = find_min_location_and_crop(image)
    return start_position

# 如果之后再调用 start_place，start_position 已经有值，不会再赋值


def revise_direction(direction):
    if direction == 40:
        pyautogui.press("W")  # Up
    elif direction == 34:
        pyautogui.press("D")  # Right
    elif direction == 37:
        pyautogui.press("S")  # Down
    elif direction == 31:
        pyautogui.press("A")  # Left

def click_belt():
    tool_location = pyautogui.locateOnScreen('belt.png',confidence=0.65)
    tool_center = pyautogui.center(tool_location)
    pyautogui.moveTo(tool_center, duration=0.5)
    pyautogui.click()

# def place_signle_belt(position,direction):
#     time.sleep(2)
#     click_belt()
#     revise_direction(direction)
#     pyautogui.moveTo(position,duration=0.5)
#     pyautogui.click()

# def get_position(matrix,cell_size,start_position,scroll_offset_scaled):
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             direction=matrix[i][j]
#             if direction==31 or direction==34 or direction==37 or direction==40:
#                 scroll_offset_x,scroll_offset_y=scroll_offset_scaled
#                 scaled_x = (j * cell_size) + scroll_offset_x
#                 scaled_y = (i * cell_size) + scroll_offset_y
#                 print("scaled_x:", scaled_x)
#                 print("scaled_y:", scaled_y)
#                 position=start_position[0]+scaled_x+cell_size/2,start_position[1]+ scaled_y+cell_size/2
#                 print(f"position: {position}")
#                 #place_signle_belt(position,direction)
#                 if(position[0] > start_position[0] + 1200) or (position[1] > start_position[1] + 830):
#                     print(f"调用了")
#                     center=drag(position,start_position)

#                 place_signle_belt(position,direction)


def get_position(i,j,cell_size,start_position):
    global scroll_offset
    global scroll_offset_scaled
    scroll_offset_x,scroll_offset_y=scroll_offset_scaled
    scaled_x = (j * cell_size) + scroll_offset_x
    scaled_y = (i * cell_size) + scroll_offset_y
    position=start_position[0]+scaled_x+cell_size/2,start_position[1]+ scaled_y+cell_size/2
    #place_signle_belt(position,direction)
    return position
    
def change_position(i,j,start_position):
    relative_position=(j * cell_size) + cell_size/2,(i * cell_size) + cell_size/2
    center=start_position[0] + Width/2, start_position[1] + Height/2
    if (relative_position[0] > 600 and relative_position[0] < 30 * cell_size - 600) and (relative_position[1] > 415 and relative_position[1] < 30 * cell_size - 415):
        print(f"调用了可拖拽到中心")
        print(f"relative_position: {relative_position}")
        true_position = center
        print(f"true_position: {true_position}")

    if (relative_position[0] < 600 or relative_position[0] > 30 * cell_size - 600) and (relative_position[1] > 415 and relative_position[1] < 30 * cell_size - 415):
        print(f"调用了左右")
        print(f"relative_position: {relative_position}")
        true_position = (start_position[0] + relative_position[0] + scroll_offset[0] , center[1])
        print(f"true_position: {true_position}")

    if (relative_position[1] < 415 or relative_position[1] > 30 * cell_size - 415) and (relative_position[0] > 600 and relative_position[0] < 30 * cell_size - 600):
        print(f"调用了上下")
        print(f"relative_position: {relative_position}")
        true_position = (center[0] , start_position[1] + relative_position[1] + scroll_offset[1])
        print(f"true_position: {true_position}")

    if (relative_position[0] < 600 and relative_position[1] < 415) or (relative_position[0] < 600 and relative_position[1] > 30 * cell_size - 415) or (relative_position[0] > 30 * cell_size - 600 and relative_position[1] < 415) or (relative_position[0] > 30 * cell_size - 600 and relative_position[1] > 30 * cell_size - 415):
        print(f"调用了四角")
        print(f"relative_position: {relative_position}")
        true_position = (start_position[0] + relative_position[0] + scroll_offset[0] , start_position[1] + relative_position[1] + scroll_offset[1])
        print(f"true_position: {true_position}")

    return true_position


  

# def drag(position,start_position):

#     center=start_position[0] + Width/2, start_position[1] + Height/2
#     position_dragto=(scaleFactor * center[0]+center[0]-position[0])/scaleFactor, (scaleFactor * center[1]+center[1]-position[1])/scaleFactor


#     if(position_dragto[0] > start_position[0] + 1200) or (position_dragto[1] > start_position[1] + 830) or (position_dragto[0] < start_position[0]) or (position_dragto[1] < start_position[1]):
#         position_dragto_temp = center

#         while (position_dragto_temp[0] < position_dragto[0]) or (position_dragto_temp[1] < position_dragto[1]):
#             # 计算每次移动的步长（例如50像素）
#             step_x = min(50, position_dragto[0] - position_dragto_temp[0])
#             step_y = min(50, position_dragto[1] - position_dragto_temp[1])

#             # 更新临时位置
#             position_dragto_temp[0] += step_x
#             position_dragto_temp[1] += step_y

#             # 拖动到新的位置
#             pyautogui.dragTo(position_dragto_temp[0], position_dragto_temp[1], duration=0.5)

#             # 如果超出边界，则停止
#             if (position_dragto_temp[0] > start_position[0] + 1200) or (position_dragto_temp[1] > start_position[1] + 830) or (position_dragto_temp[0] < start_position[0]) or (position_dragto_temp[1] < start_position[1]):
#                 print("超出边界，停止拖动")
#                 break

#         pyautogui.mouseUp()  # 松开鼠标

#         # while(position_dragto_temp <= position_dragto):
#         #     pyautogui.moveTo(center)  # 先将鼠标移动到center
#         #     pyautogui.mouseDown()  # 按下鼠标左键
#         #     pyautogui.dragTo(position_dragto_temp[0], position_dragto_temp[1], duration=3)
#         #     pyautogui.mouseUp()  # 松开鼠标
#         #     position_dragto = position_dragto - position_dragto_temp
#     else:
#         pyautogui.moveTo(center)  # 先将鼠标移动到center
#         pyautogui.mouseDown()  # 按下鼠标左键
#         pyautogui.dragTo(position_dragto[0], position_dragto[1], duration=3)  # 将鼠标拖动到目标位置
#         pyautogui.mouseUp()  # 松开鼠标
#         return center

def drag(position,start_position):#需要scaleFactor
    global scroll_offset
    global scroll_offset_scaled
    center=start_position[0] + Width/2, start_position[1] + Height/2
    position_dragto=(scaleFactor * center[0]+center[0]-position[0])/scaleFactor, (scaleFactor * center[1]+center[1]-position[1])/scaleFactor
    s_x=(scaleFactor * center[0]+center[0]-position[0])/scaleFactor-start_position[0] -Width/2
    s_y=(scaleFactor * center[1]+center[1]-position[1])/scaleFactor-start_position[1] - Height/2
    print("s_x",s_x)
    print("s_y",s_y)
    pyautogui.moveTo(center)  # 先将鼠标移动到center
    pyautogui.mouseDown()  # 按下鼠标左键
    pyautogui.dragTo(position_dragto[0], position_dragto[1], duration=0.5)  # 将鼠标拖动到目标位置
    scroll_offset = load_scroll_offset()
    scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
    print("scroll_offset:",scroll_offset)
 


def belt_drag(coordinates,end_drag,cell_size,start_position):
    click_belt()
    for x, y in coordinates:
        position=get_position(x,y,cell_size,start_position)
        pyautogui.moveTo(position,duration=0.3)
        pyautogui.mouseDown()
        if(x==end_drag[0] and y==end_drag[1]):
            pyautogui.mouseUp()

def remove_before_first_match(coordinates,end_drag, cell_size, start_position):
    # 遍历列表找到第一个符合条件的坐标
    global scroll_offset
    global scroll_offset_scaled
    previous_position = None
    flag = False
    for index, coord in enumerate(coordinates):
        # 使用 coord 计算 position
        print(f"coord: {coord}")
        position = get_position(coord[0], coord[1], cell_size, start_position)
        print(f"position: {position}")
        # 判断坐标是否超过指定的阈值
        if (position[0] < start_position[0]) or (position[0] > start_position[0] + 1200) or (position[1] > start_position[1] + 830) or (position[1] < start_position[1]):
            # 调用 belt_drag 删除之前的元素
            position = change_position(coord[0], coord[1], start_position)
            print(f"position: {position}")
            belt_drag(coordinates[:index],coordinates[index-1],cell_size, start_position)
            drag(previous_position,start_position)
            # scroll_offset = load_scroll_offset()
            # print("scroll_offset:",scroll_offset)
            print("limit_x:",start_position[0] + 1200)
            print("limit_y:",start_position[1] + 830)
            print("coor_before:",coordinates[:index])
            # 返回剩余的坐标和 flag
            return coordinates[index-1:], flag
        previous_position = position
        print(f"previous_position: {previous_position}")

    # 如果没有找到符合条件的坐标，返回原列表
    flag = True
    belt_drag(coordinates,end_drag,cell_size, start_position)
    print("coordinates:",coordinates)
    return coordinates, flag


def drag_place_belt(coordinates,end_drag,cell_size,start_position):
    # 调用函数
    new_coordinates,flag = remove_before_first_match(coordinates,end_drag,cell_size,start_position)
    while not flag:
        new_coordinates,flag= remove_before_first_match(new_coordinates,end_drag,cell_size,start_position)


def main(coordinates,end_drag):
    # global scaleFactor
    # global scroll_offset
    image =Image.open("star.png")
    start_position = start_place(image)
    print("start_position:", start_position)
    coordinates = coordinates
    end_drag = end_drag
    #scaleFactor_round = round(scaleFactor, 1)
    #scale_event(scaleFactor_round)
    # scaleFactor = load_scaleFactor()
    # scroll_offset = load_scroll_offset()
    drag_place_belt(coordinates,end_drag,cell_size,start_position)

main(coordinates,end_drag)

#belt_drag(coordinates,cell_size,start_position,scroll_offset_scaled)
#get_position(matrix,cell_size,start_position,scroll_offset_scaled)



