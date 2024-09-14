#include <windows.h>
#include <cstdio>

int *shared_c;

int main() {
    // 创建共享内存
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,    // 使用分页文件
        NULL,                    // 默认安全属性
        PAGE_READWRITE,          // 读/写权限
        0,                       // 文件的高32位大小
        sizeof(int),             // 文件的低32位大小（int 的大小）
        L"SharedMemoryC"         // 使用宽字符字符串作为共享内存名称
        );
    if (hMapFile == NULL) {
        printf("Could not create file mapping object (%d).\n", GetLastError());
        return 1;
    }

    // 将内存映射到共享变量
    shared_c = (int*) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int));
    if (shared_c == NULL) {
        printf("Could not map view of file (%d).\n", GetLastError());
        CloseHandle(hMapFile);
        return 1;
    }

    // 修改共享变量
    *shared_c = 3;
    printf("C++: shared_c = %d\n", *shared_c);

    // 保持程序运行，让 Python 能够访问共享变量
    while(1){
    }

    // 释放资源
    UnmapViewOfFile(shared_c);
    CloseHandle(hMapFile);
    return 0;
}
