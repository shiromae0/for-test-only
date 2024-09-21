import glob
from multiprocessing import shared_memory
import numpy as np

def load_shared_arrays():
    try:
        existing_shm = shared_memory.SharedMemory(name='SharedResource')
        build_shm = shared_memory.SharedMemory(name='SharedBuild')

        HEIGHT = 15
        WIDTH = 24

        resource_array = np.ndarray((HEIGHT, WIDTH), dtype=int, buffer=existing_shm.buf).copy()
        buildingsmap_array = np.ndarray((HEIGHT, WIDTH), dtype=int, buffer=build_shm.buf).copy()


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
#print(load_shared_arrays()[1])
print(load_needed_shape())