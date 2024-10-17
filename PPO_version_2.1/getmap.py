import glob
from multiprocessing import shared_memory
import numpy as np

def load_shared_arrays():
    try:
        existing_shm = shared_memory.SharedMemory(name='SharedResource')
        build_shm = shared_memory.SharedMemory(name='SharedBuild')

        HEIGHT = 20
        WIDTH = 32
        resource_array = np.ndarray((HEIGHT, WIDTH), dtype=int, buffer=existing_shm.buf).copy()
        buildingsmap_array = np.ndarray((HEIGHT, WIDTH), dtype=int, buffer=build_shm.buf).copy()
        belt_mapping = {
            31: 3103,
            32: 3111,
            33: 3109,
            34: 3104,
            35: 3112,
            36: 3110,
            37: 3102,
            38: 3107,
            39: 3108,
            40: 3101,
            41: 3105,
            42: 3106,
        }
        # 遍历 buildingsmap_array 并修改 item
        # 遍历 buildingsmap_array 并修改每个值
        for i in range(buildingsmap_array.shape[0]):  # 遍历每一行
            for j in range(buildingsmap_array.shape[1]):  # 遍历每一列
                item = buildingsmap_array[i, j]  # 获取元素
                if item in belt_mapping:
                    buildingsmap_array[i, j] = belt_mapping[item]  # 使用映射值替换

        return resource_array, buildingsmap_array

    finally:
        existing_shm.close()
        build_shm.close()
def load_needed_shape():
    try:
        existing_shm = shared_memory.SharedMemory(name='need_shape_name')
        shape = np.ndarray((1,), dtype=int, buffer=existing_shm.buf)[0]
        return shape

    finally:
        existing_shm.close()
def load_scroll_offset():
    try:
        existing_shm = shared_memory.SharedMemory(name='scrolloffset')
        scrolloffset_x = np.ndarray((2,), dtype=int, buffer=existing_shm.buf)[0]
        scrolloffset_y = np.ndarray((2,), dtype=int, buffer=existing_shm.buf)[1]
        return scrolloffset_x,scrolloffset_y

    finally:
        existing_shm.close()
def load_scaleFactor():
    existing_shm = None
    try:
        existing_shm = shared_memory.SharedMemory(name='scaleFactor')
        scaleFactor= np.frombuffer(existing_shm.buf, dtype=np.double)[0]
        return scaleFactor
    finally:
        if existing_shm is not None:
            existing_shm.close()
for line in load_shared_arrays()[0]:
    print("[",end="")
    for item in line:
        print(item,end=",")
    print("]")
# print(np.array2string(load_shared_arrays()[0], max_line_width=200))
# print(np.array2string(load_shared_arrays()[1], max_line_width=200))
# print(load_shared_arrays()[1])
print(load_scroll_offset())
print(load_scaleFactor())