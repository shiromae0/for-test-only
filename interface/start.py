import pyautogui

def find_min_location_and_crop(image_file):
    # 使用 locateOnScreen 来查找 star.png 的位置
    location = pyautogui.locateOnScreen(image_file,confidence=0.8)

    if location:
        # 获取左上角的 x 和 y 坐标
        left, top = location.left, location.top
        print(f"Star.png found at: ({left}, {top})")
        left = left - 77
        top = top - 746
        return (left, top)
    else:
        print("Star.png not found on the screen.")
        return None
