import ctypes
from ctypes import wintypes
def visualize():
    # 定义 WinAPI 函数
    user32 = ctypes.windll.user32

    # 获取窗口句柄，"GameWindowTitle" 是游戏窗口的标题
    hwnd = user32.FindWindowW(None, "SHAPEZ")

    # 定义 RECT 结构体，用于获取窗口矩形
    rect = wintypes.RECT()
    if hwnd:
        # 获取窗口的矩形
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        print(f"窗口位置和尺寸: 左:{rect.left}, 上:{rect.top}, 右:{rect.right}, 下:{rect.bottom}")
        left=rect.left+7
        top=rect.top+30
        return (left,top)



