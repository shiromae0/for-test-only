import pyautogui
import time
from start import find_min_location_and_crop
from PIL import Image
from getmap import load_scroll_offset,load_scaleFactor


image = Image.open("star.png")
#start_position = find_min_location_and_crop(image_files)
#print("start_position:",start_position)

#scroll_offset = (-50,-50)
scroll_offset = load_scroll_offset()
print("scroll_offset:",scroll_offset)
scaleFactor = load_scaleFactor()
print("scaleFactor:",scaleFactor)
cell_size=50*scaleFactor
scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
coordinates = [(12,2),(11,2),(10,2),(9,2),(8,2),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10), (7, 11), (7,12)]
start_drag = (12, 2)
end_drag = (7, 12) 
Width=1200
Height=830
target_type=22

class Machine:
    def __init__(self, machine_type, direction):
        self.type = machine_type
        self.direction = direction

# 假设字典 position 已经存储了若干个 Machine 类实例
machine_position = {
    (149,0): Machine(22, 1)
    # (12,3): Machine(22, 4)
}

def revise_direction(direction):
    if direction == 1:
        pyautogui.press("W")
    if direction == 2:
        pyautogui.press("S")
    if direction == 3:
        pyautogui.press("A")
    if direction == 4:
        pyautogui.press("D")

#initialize
def scroll_event(scroll_offset):
    time.sleep(2)
    scroll_x,scroll_y=scroll_offset
    position=pyautogui.position()#鼠标当前位置
    end=position[0]-scroll_x,position[1]-scroll_y
    pyautogui.mouseDown()
    pyautogui.dragTo(end, duration=0.5)
    scroll_offset=(0,0)
    return scroll_offset

#scroll_offset = scroll_event(scroll_offset)
print("scroll_offset",scroll_offset)

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
    return scaleFactor

#scaleFactor = scale_event(scaleFactor)
print("scaleFactor:",scaleFactor)


def start_place(image):
    # 如果 position_flag 为 False 且 start_position 还没有赋值
    start_position = find_min_location_and_crop(image)
    return start_position

# 如果之后再调用 start_place，start_position 已经有值，不会再赋值
start_position = start_place(image)
print("start_position:", start_position)

# def revise_direction(direction):
#     if direction == 40:
#         pyautogui.press("W")  # Up
#     elif direction == 34:
#         pyautogui.press("D")  # Right
#     elif direction == 37:
#         pyautogui.press("S")  # Down
#     elif direction == 31:
#         pyautogui.press("A")  # Left

def click_miner():
    tool_location = pyautogui.locateOnScreen('belt.png',confidence=0.65)
    tool_center = pyautogui.center(tool_location)
    pyautogui.moveTo(tool_center, duration=0.5)
    pyautogui.click()

def place_miner(machine_position,cell_size,start_position,target_type):
    global scroll_offset_scaled
    for pos, machine in machine_position.items():
        direction=machine.direction
        if machine.type == target_type:
            x,y=pos
            position=get_position(x,y,cell_size,start_position)
            #time.sleep(2)
            if(position[0] < start_position[0]) or (position[0] > start_position[0] + 1200) or (position[1] > start_position[1] + 830) or (position[1] < start_position[1]):
                print(f"调用了")
                drag(position,start_position)
            position=get_position(x,y,cell_size,start_position)
            click_miner()
            pyautogui.moveTo(position,duration=0.5)
            revise_direction(direction)
            time.sleep(1)
            pyautogui.click()

def get_position(i,j,cell_size,start_position):
    global scroll_offset
    global scroll_offset_scaled
    scroll_offset_x,scroll_offset_y=scroll_offset_scaled
    scaled_x = (j * cell_size) + scroll_offset_x
    scaled_y = (i * cell_size) + scroll_offset_y
    print("scaled_x:", scaled_x)
    print("scaled_y:", scaled_y)

    position=start_position[0]+scaled_x+cell_size/2,start_position[1]+ scaled_y+cell_size/2
    print(f"position: {position}")
    #place_signle_belt(position,direction)
    return position
    #place_signle_belt(center,direction)



# def drag(position,start_position):
#     global scroll_offset
#     global scroll_offset_scaled
#     center=start_position[0] + Width/2, start_position[1] + Height/2
#     position_dragto=(scaleFactor * center[0]+center[0]-position[0])/scaleFactor, (scaleFactor * center[1]+center[1]-position[1])/scaleFactor
#     pyautogui.moveTo(center)  # 先将鼠标移动到center
#     pyautogui.mouseDown()  # 按下鼠标左键
#     pyautogui.dragTo(position_dragto[0], position_dragto[1], duration=1)  # 将鼠标拖动到目标位置
#     pyautogui.mouseUp()  # 松开鼠标
#     scroll_offset = load_scroll_offset()
#     scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

def drag(position,start_position):
    global scroll_offset
    global scroll_offset_scaled
    center=start_position[0] + Width/2, start_position[1] + Height/2
    position_dragto=(scaleFactor * center[0]+center[0]-position[0])/scaleFactor, (scaleFactor * center[1]+center[1]-position[1])/scaleFactor


    if(position_dragto[0] > start_position[0] + 1200) or (position_dragto[1] > start_position[1] + 830) or (position_dragto[0] < start_position[0]) or (position_dragto[1] < start_position[1]):
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
                step_y = min(415, position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                # 更新临时位置
                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                # 拖动到新的位置
                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                # 如果超出边界，则停止
                # if (position_dragto_temp[0] > start_position[0] + 1200) or (position_dragto_temp[1] > start_position[1] + 830) or (position_dragto_temp[0] < start_position[0]) or (position_dragto_temp[1] < start_position[1]):
                #     print("超出边界，停止拖动")
                #     break

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
                step_y = max(-415, position_dragto_temp[1] - center[1])
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
                step_y = max(-415, position_dragto_temp[1] - center[1])
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
                step_y = min(415, position_dragto_temp[1] - center[1])
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

        # while(position_dragto_temp <= position_dragto):
        #     pyautogui.moveTo(center)  # 先将鼠标移动到center
        #     pyautogui.mouseDown()  # 按下鼠标左键
        #     pyautogui.dragTo(position_dragto_temp[0], position_dragto_temp[1], duration=3)
        #     pyautogui.mouseUp()  # 松开鼠标
        #     position_dragto = position_dragto - position_dragto_temp
    else:
        pyautogui.moveTo(center)  # 先将鼠标移动到center
        pyautogui.mouseDown()  # 按下鼠标左键
        pyautogui.dragTo(position_dragto[0], position_dragto[1], duration=1)  # 将鼠标拖动到目标位置
        pyautogui.mouseUp()  # 松开鼠标
        scroll_offset = load_scroll_offset()
        scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
        return center

#belt_drag(coordinates,cell_size,start_position,scroll_offset_scaled)
#get_position(matrix,cell_size,start_position,scroll_offset_scaled)
place_miner(machine_position,cell_size,start_position,target_type)



