import numpy as np
import pyautogui
import time
from getmap import load_scroll_offset,load_scaleFactor,load_shared_arrays,load_needed_shape
from PPO import get_agent_act_list
from visualize import visualize
# scroll_offset = load_scroll_offset()
# scaleFactor = load_scaleFactor()
# scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
scroll_offset = None
scaleFactor = None
scroll_offset_scaled = None

Width=1200
Height=830
Horizontal = 32
verticle =20

#data_list = get_agent_act_list()

# def get_map():
#     resource = load_shared_arrays()[0]
#     build = load_shared_arrays()[1]
#     target_shape = load_needed_shape()
#     print(resource)
#     print(np.array2string(build, max_line_width=200))
#     return resource,build,target_shape 

def calculate_scale_count():
    difference = abs(scaleFactor - 1.0)
    scale_count = round(difference / 0.1)
    return scale_count

def scale_event():
    global scaleFactor
    time.sleep(2)
    if scaleFactor == 1.0:
        print("scaleFactor 为 1.0,no scaling needed.")
        return  
    
    count=calculate_scale_count()
    print(f"scale count: {count}")
    for _ in range(count):
        if scaleFactor > 1.0:
            pyautogui.scroll(-120)
        elif scaleFactor < 1.0:
            pyautogui.scroll(120)
    scaleFactor = load_scaleFactor()


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
        pyautogui.press("w")  # Up
    elif direction == 2:
        pyautogui.press("s")  # Down
    elif direction == 3:
        pyautogui.press("a")  # Left
    elif direction == 4:
        pyautogui.press("d")  # Right
    elif direction == 5:
        pyautogui.press("3")
    elif direction == 6:
        pyautogui.press("1")
    elif direction == 7:
        pyautogui.press("7")
    elif direction == 8:
        pyautogui.press("5")
    elif direction == 9:
        pyautogui.press("4")
    elif direction == 10:
        pyautogui.press("2")
    elif direction == 11:
        pyautogui.press("8")
    elif direction == 12:
        pyautogui.press("6")

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
    if(position[0] > start_position[0] + Width-cell_size) or (position[1] > start_position[1] + Height-2*cell_size-10) or (position[0] < start_position[0]+cell_size) or (position[1] < start_position[1]+cell_size):
        print(f"scroll_offset: {scroll_offset}")
        print(f"relative_position: {relative_position}")

        if (relative_position[0] > (Width/2) and relative_position[0] < Horizontal * cell_size - (Width/2)) and (relative_position[1] > (Height/2) and relative_position[1] < verticle * cell_size - (Height/2)):
            print(f"center")
            true_position = drag(position, start_position)
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            print(f"true_position: {true_position}")
            return true_position

        if (relative_position[0] < (Width/2) or relative_position[0] > Horizontal * cell_size - (Width/2)) and (relative_position[1] > (Height/2) and relative_position[1] < verticle * cell_size - (Height/2)):
            print(f"left and right")
            true_position = drag(position, start_position)
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            true_position = (start_position[0] + relative_position[0] + (scroll_offset_scaled[0]), true_position[1])
            print(f"true_position: {true_position}")
            return true_position

        if (relative_position[1] < (Height/2) or relative_position[1] > verticle * cell_size - (Height/2)) and (relative_position[0] > (Width/2) and relative_position[0] < Horizontal * cell_size - (Width/2)):
            print(f"up and down")
            true_position = drag(position, start_position)
            print(f"true_position: {true_position}")
            print(f"relative_position: {relative_position}")
            scroll_offset = load_scroll_offset()
            print(f"scroll_offset: {scroll_offset}")
            print(f"scaleFactor: {scaleFactor}")
            print(f"scroll_offset_scaled: {scroll_offset_scaled}")
            true_position = (true_position[0] , start_position[1] + relative_position[1] + (scroll_offset_scaled[1]))
            print(f"true_position: {true_position}")
            return true_position

        if (relative_position[0] < (Width/2) and relative_position[1] < (Height/2)) or (relative_position[0] < (Width/2) and relative_position[1] > verticle * cell_size - (Height/2)) or (relative_position[0] > Horizontal * cell_size - (Width/2) and relative_position[1] < (Height/2)) or (relative_position[0] > Horizontal * cell_size - (Width/2) and relative_position[1] > verticle * cell_size - (Height/2)):
            print(f"four corners")
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

    if(position_dragto[0] > start_position[0] + Width-50) or (position_dragto[1] > start_position[1] + Height-2*50-10) or (position_dragto[0] < start_position[0]+50) or (position_dragto[1] < start_position[1]+50):
        position_dragto_temp = list(position_dragto)
        print(f"position_dragto: {position_dragto}")
        print(f"center: {center}")
        print(f"multiple dragging")

        if (position_dragto_temp[0] > center[0]) and (position_dragto_temp[1] > center[1]):
            print(f"multiple dragging right down")
            while (position_dragto_temp[0] > center[0]) or (position_dragto_temp[1] > center[1]):
                print(f"begin multiple dragging right down")

                pyautogui.moveTo(center)  
                pyautogui.mouseDown()  


                step_x = min((Width/2), position_dragto_temp[0] - center[0])
                step_y = min((Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp() 

            return center


        if (position_dragto_temp[0] < center[0]) and (position_dragto_temp[1] < center[1]):
            print(f"multiple dragging left up")
            while (position_dragto_temp[0] < center[0]) or (position_dragto_temp[1] < center[1]):
                print(f"begin multiple dragging left up")
                pyautogui.moveTo(center)  
                pyautogui.mouseDown() 

                step_x = max(-(Width/2), position_dragto_temp[0] - center[0])
                step_y = max(-(Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp() 
            return center

        if (position_dragto_temp[0] > center[0]) and (position_dragto_temp[1] < center[1]):
            print(f"multiple dragging right up")
            while (position_dragto_temp[0] > center[0]) or (position_dragto_temp[1] < center[1]):
                print(f"begin multiple dragging right up")
                pyautogui.moveTo(center)  
                pyautogui.mouseDown()  

                step_x = min((Width/2), position_dragto_temp[0] - center[0])
                step_y = max(-(Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp() 
            return center

        if (position_dragto_temp[0] < center[0]) and (position_dragto_temp[1] > center[1]):
            print(f"multiple dragging left down")
            while (position_dragto_temp[0] < center[0]) or (position_dragto_temp[1] > center[1]):
                print(f"begin multiple dragging left down")
                pyautogui.moveTo(center) 
                pyautogui.mouseDown()  

                step_x = max(-(Width/2), position_dragto_temp[0] - center[0])
                step_y = min((Height/2), position_dragto_temp[1] - center[1])
                print(f"step_x: {step_x}")
                print(f"step_y: {step_y}")

                position_dragto_temp[0] -= step_x
                position_dragto_temp[1] -= step_y
                print(f"position_dragto_temp: {position_dragto_temp}")

                pyautogui.dragTo(center[0] + step_x, center[1] + step_y, duration=1)

                scroll_offset = load_scroll_offset()
                scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)

            pyautogui.mouseUp()
            return center

    else:
        pyautogui.moveTo(center) 
        pyautogui.mouseDown() 
        pyautogui.dragTo(position_dragto[0], position_dragto[1], duration=1)  
        pyautogui.mouseUp() 
        scroll_offset = load_scroll_offset()
        scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
        return center

def place_object(data_list, cell_size, start_position):

    for item in data_list:
        (machine_type, direction),(x, y) = item 

        position = get_position(x, y, cell_size, start_position)

        if machine_type == 22:
            click_miner()
        elif machine_type == 23:
            click_cutter()
        elif machine_type == 24:
            click_trash()
        elif machine_type == 31:
            click_belt()

        
        pyautogui.moveTo(position, duration=0.5)
        revise_direction(direction)
        pyautogui.click()

        if machine_type == 0:
            pyautogui.moveTo(position, duration=0.5)
            pyautogui.click(button='right')


def place_single_object( data_list, cell_size):
     start_position = visualize()
     place_object(data_list, cell_size, start_position)

def run_place_single_object():
    global scaleFactor
    global scroll_offset
    global scroll_offset_scaled
    scaleFactor = load_scaleFactor()
    scale_event()
    start_position = visualize()
    center=start_position[0] + Width/2, start_position[1] + Height/2
    print(center)
    pyautogui.moveTo(center, duration=0.5)
    pyautogui.click()
    pyautogui.dragTo(center[0]+1, center[1]+1, duration=0.5)

    print("start_position:", start_position)
    cell_size = 50 * scaleFactor
    scroll_offset = load_scroll_offset()
    scroll_offset_scaled = (scroll_offset[0] * scaleFactor, scroll_offset[1] * scaleFactor)
    data_list = get_agent_act_list()
    place_single_object( data_list, cell_size)

def main():
    run_place_single_object()

main()



