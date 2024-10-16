from PIL import Image

# 定义新的图片路径和文件名
image_files = [
    "belt_S_W_0.png", "belt_S_W_1.png", "belt_S_W_2.png", 
    "belt_S_W_3.png", "belt_S_W_4.png", "belt_S_W_5.png", 
    "belt_S_W_6.png", "belt_S_W_7.png", "belt_S_W_8.png", 
    "belt_S_W_9.png"
]

# 打开所有图片并存储到列表中
images = [Image.open(img) for img in image_files]

# 将第一张图片保存为GIF，同时添加其他图片为帧
images[0].save('belt_S_W.gif', 
               save_all=True, 
               append_images=images[1:], 
               duration=33,  # 每帧的持续时间，单位是毫秒，33毫秒即30帧/秒
               loop=0)  # 循环次数，0表示无限循环
