import pygame
import sys

# 初始化pygame
pygame.init()

# 定义变量
size = width, height = 1000, 800

# 加载图像
background = pygame.image.load("C://Users//23081//Desktop//shapez_login.jpg")
menu = pygame.image.load("C://Users//23081//Desktop//shapez2.jpg")  # 另一个背景

# 缩放图像到窗口大小
bg1 = pygame.transform.scale(background, (width, height))
bg2 = pygame.transform.scale(menu, (width, height))

# 获取图像的位置
position = background.get_rect()

# 创建一个主窗口
screen = pygame.display.set_mode(size)

# 标题
pygame.display.set_caption("SHAPEZ")

# 初始化背景
current_bg = bg1

# 初始位置
site = [0, 0]

# 创建游戏主循环
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        # 图像移动 KEYDOWN 键盘按下事件
        # 通过 key 属性对应按键
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                site[1] -= 8
            if event.key == pygame.K_DOWN:
                site[1] += 8
            if event.key == pygame.K_LEFT:
                site[0] -= 8
            if event.key == pygame.K_RIGHT:
                site[0] += 8
            # 切换背景
            if event.key == pygame.K_RETURN:  
                if current_bg == bg1:
                    current_bg = bg2
                else:
                    current_bg = bg1

    # 移动图像
    position = position.move(site)

    # 填充当前背景图像
    screen.blit(current_bg, (0, 0))

    # 放置图片（如果需要）
    # screen.blit(img, position)

    # 更新显示界面
    pygame.display.flip()
