#include <windows.h>
#include <cstdio>
#include "GameMap.h"
#include "Building.h"
#include <cstdlib>

int (*GameMap::Resource)[WIDTH] = nullptr;
int (*GameMap::Buildingsmap)[WIDTH] = nullptr;
void GameMap::CreateMapFile(){
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,                    // 默认安全属性
        PAGE_READWRITE,          // 读/写权限
        0,                       // 文件的高32位大小
        sizeof(int)*HEIGHT*WIDTH,             // 文件的低32位大小（int 的大小）
        L"SharedResource"         // 使用宽字符字符串作为共享内存名称
        );
    Resource = (int(*)[WIDTH]) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int) * HEIGHT*WIDTH);

    if (Resource == NULL) {
        printf("Could not map view of file (%d).\n", GetLastError());
        CloseHandle(hMapFile);
    }
    HANDLE buildmapfile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(int)*HEIGHT*WIDTH,
        L"SharedBuild"
        );
    Buildingsmap = (int(*)[WIDTH]) MapViewOfFile(buildmapfile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int) * HEIGHT*WIDTH);
}
GameMap::GameMap()
{
    CreateMapFile();
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
        {
            Resource[i][j] = NONE;
        }
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
        {
            BuildingsMap[i][j] = nullptr;
            Buildingsmap[i][j] = -1;
        }
}
int* GameMap::getResource() {
    return &Resource[0][0];  // 返回 Resource 数组的首地址
}


void GameMap::FirstMap()
{
    Resource[1][3] = CYCLE;
    Resource[1][4] = CYCLE;
    Resource[1][5] = CYCLE;
    Resource[1][6] = CYCLE;
    Resource[2][3] = CYCLE;
    Resource[3][3] = CYCLE;
    Resource[3][4] = CYCLE;
    Resource[3][5] = CYCLE;
    Resource[3][6] = CYCLE;
    Resource[4][6] = CYCLE;
    Resource[5][6] = CYCLE;
    Resource[5][3] = CYCLE;
    Resource[5][4] = CYCLE;
    Resource[5][5] = CYCLE;

    // 设置 RECT 资源
    Resource[1][25] = RECT;
    Resource[1][28] = RECT;
    Resource[2][25] = RECT;
    Resource[2][28] = RECT;
    Resource[3][25] = RECT;
    Resource[3][28] = RECT;
    Resource[4][25] = RECT;
    Resource[4][28] = RECT;
    Resource[5][25] = RECT;
    Resource[5][28] = RECT;
    Resource[3][26] = RECT;
    Resource[3][27] = RECT;

    Resource[14][3] = CYCLE;
    Resource[14][4] = CYCLE;
    Resource[14][5] = CYCLE;
    Resource[14][6] = CYCLE;
    Resource[15][3] = CYCLE;
    Resource[15][6] = CYCLE;
    Resource[16][3] = CYCLE;
    Resource[16][4] = CYCLE;
    Resource[16][5] = CYCLE;
    Resource[16][6] = CYCLE;
    Resource[17][3] = CYCLE;
    Resource[17][6] = CYCLE;
    Resource[17][6] = CYCLE;
    Resource[18][3] = CYCLE;
    Resource[18][6] = CYCLE;

    // 设置 RECT 资源
    Resource[14][25] = RECT;
    Resource[14][26] = RECT;
    Resource[14][27] = RECT;
    Resource[14][28] = RECT;
    Resource[15][25] = RECT;
    Resource[15][28] = RECT;
    Resource[16][25] = RECT;
    Resource[16][28] = RECT;
    Resource[17][25] = RECT;
    //Resource[17][28] = RECT;
    Resource[16][26] = RECT;
    Resource[16][27] = RECT;
    Resource[18][25] = RECT;
    //Resource[18][28] = RECT;






    /*
    int centerX = 75;  // 中心行
    int centerY = 120; // 中心列
    // 初始化资源
    // 设置距离中心 10-20 个格子距离的 CYCLE 资源
    Resource[centerX - 15][centerY - 10] = CYCLE;
    Resource[centerX - 15][centerY - 9] = CYCLE;
    Resource[centerX - 15][centerY - 8] = CYCLE;
    Resource[centerX - 16][centerY - 9] = CYCLE;
    Resource[centerX - 16][centerY - 8] = CYCLE;
    Resource[centerX - 17][centerY - 8] = CYCLE;
    Resource[centerX - 14][centerY - 10] = CYCLE;
    Resource[centerX - 14][centerY - 9] = CYCLE;
    Resource[centerX - 13][centerY - 10] = CYCLE;
    Resource[centerX - 14][centerY - 8] = CYCLE;
    Resource[centerX - 17][centerY - 9] = CYCLE;
    Resource[centerX - 16][centerY - 10] = CYCLE;
    Resource[centerX - 15][centerY - 7] = CYCLE;
    Resource[centerX - 15][centerY - 6] = CYCLE;
    Resource[centerX - 16][centerY - 7] = CYCLE;
    Resource[centerX - 17][centerY - 7] = CYCLE;
    Resource[centerX - 14][centerY - 7] = CYCLE;
    Resource[centerX - 13][centerY - 9] = CYCLE;
    Resource[centerX - 12][centerY - 10] = CYCLE;
    Resource[centerX - 13][centerY - 8] = CYCLE;
    Resource[centerX - 12][centerY - 8] = CYCLE;
    Resource[centerX - 11][centerY - 9] = CYCLE;
    Resource[centerX - 10][centerY - 10] = CYCLE;
    Resource[centerX - 12][centerY - 9] = CYCLE;
    Resource[centerX - 17][centerY - 6] = CYCLE;
    Resource[centerX - 16][centerY - 6] = CYCLE;
    Resource[centerX - 15][centerY - 5] = CYCLE;
    Resource[centerX - 14][centerY - 6] = CYCLE;
    Resource[centerX - 13][centerY - 7] = CYCLE;
    Resource[centerX - 13][centerY - 6] = CYCLE;
    Resource[centerX - 12][centerY - 7] = CYCLE;
    Resource[centerX - 16][centerY - 5] = CYCLE;
    Resource[centerX - 17][centerY - 5] = CYCLE;
    Resource[centerX - 18][centerY - 5] = CYCLE;
    Resource[centerX - 19][centerY - 6] = CYCLE;
    Resource[centerX - 20][centerY - 6] = CYCLE;
    Resource[centerX - 19][centerY - 7] = CYCLE;
    Resource[centerX - 18][centerY - 7] = CYCLE;
    Resource[centerX - 19][centerY - 8] = CYCLE;
    Resource[centerX - 20][centerY - 7] = CYCLE;
    Resource[centerX - 21][centerY - 7] = CYCLE;
    Resource[centerX - 20][centerY - 8] = CYCLE;
    Resource[centerX - 19][centerY - 9] = CYCLE;
    Resource[centerX - 18][centerY - 10] = CYCLE;
    Resource[centerX - 17][centerY - 11] = CYCLE;
    Resource[centerX - 16][centerY - 11] = CYCLE;

    // 新增的 CYCLE 资源
    Resource[centerX - 10][centerY + 15] = CYCLE;
    Resource[centerX - 11][centerY + 14] = CYCLE;
    Resource[centerX - 12][centerY + 13] = CYCLE;
    Resource[centerX - 13][centerY + 12] = CYCLE;
    Resource[centerX - 14][centerY + 11] = CYCLE;
    Resource[centerX - 15][centerY + 10] = CYCLE;
    Resource[centerX - 16][centerY + 9] = CYCLE;
    Resource[centerX - 17][centerY + 8] = CYCLE;
    Resource[centerX - 18][centerY + 7] = CYCLE;
    Resource[centerX - 10][centerY - 15] = CYCLE;
    Resource[centerX - 11][centerY - 14] = CYCLE;
    Resource[centerX - 12][centerY - 13] = CYCLE;
    Resource[centerX - 13][centerY - 12] = CYCLE;
    Resource[centerX - 14][centerY - 11] = CYCLE;
    Resource[centerX - 15][centerY - 10] = CYCLE;
    Resource[centerX - 16][centerY - 9] = CYCLE;

    // 设置特定位置的 RECT 资源，调整到地图中心 10-20 个格子距离
    Resource[centerX - 63][centerY - 100] = RECT;
    Resource[centerX - 63][centerY - 99] = RECT;
    Resource[centerX - 63][centerY - 98] = RECT;
    Resource[centerX - 62][centerY - 101] = RECT;
    Resource[centerX - 62][centerY - 100] = RECT;
    Resource[centerX - 62][centerY - 99] = RECT;
    Resource[centerX - 61][centerY - 100] = RECT;
    Resource[centerX - 61][centerY - 99] = RECT;

    // 添加其他区域资源，调整到地图中心 10-20 格子的距离
    Resource[centerX - 15][centerY + 20] = CYCLE;
    Resource[centerX - 14][centerY + 20] = CYCLE;
    Resource[centerX - 13][centerY + 20] = CYCLE;
    Resource[centerX - 12][centerY + 21] = CYCLE;
    Resource[centerX - 12][centerY + 22] = CYCLE;
    Resource[centerX - 11][centerY + 20] = CYCLE;
    Resource[centerX - 10][centerY + 20] = CYCLE;
    Resource[centerX - 9][centerY + 20] = CYCLE;
    Resource[centerX - 8][centerY + 20] = CYCLE;
    Resource[centerX - 15][centerY + 21] = CYCLE;
    Resource[centerX - 15][centerY + 22] = CYCLE;
    Resource[centerX - 15][centerY + 23] = CYCLE;
    Resource[centerX - 15][centerY + 24] = CYCLE;
    Resource[centerX - 14][centerY + 24] = CYCLE;
    Resource[centerX - 13][centerY + 24] = CYCLE;
    Resource[centerX - 12][centerY + 24] = CYCLE;
    Resource[centerX - 11][centerY + 24] = CYCLE;

    // 新增的 RECT 资源
    Resource[centerX + 35][centerY - 20] = RECT;
    Resource[centerX + 36][centerY - 20] = RECT;
    Resource[centerX + 37][centerY - 21] = RECT;
    Resource[centerX + 38][centerY - 22] = RECT;
    Resource[centerX + 39][centerY - 23] = RECT;
    Resource[centerX + 40][centerY - 24] = RECT;
    Resource[centerX + 41][centerY - 25] = RECT;

    // 再设置一个区域
    Resource[centerX + 15][centerY + 10] = RECT;
    Resource[centerX + 16][centerY + 10] = RECT;
    Resource[centerX + 17][centerY + 10] = RECT;
    Resource[centerX + 18][centerY + 11] = RECT;
    Resource[centerX + 19][centerY + 11] = RECT;
    Resource[centerX + 20][centerY + 10] = RECT;
    Resource[centerX + 15][centerY + 11] = RECT;
    Resource[centerX + 15][centerY + 12] = RECT;
    Resource[centerX + 15][centerY + 13] = RECT;
    Resource[centerX + 16][centerY + 12] = RECT;
    Resource[centerX + 17][centerY + 12] = RECT;
    Resource[centerX + 18][centerY + 12] = RECT;
    Resource[centerX + 19][centerY + 13] = RECT;
    Resource[centerX + 20][centerY + 12] = RECT;

    // 设置更多区域，调整到地图中心 10-20 格子的距离
    Resource[centerX + 25][centerY - 20] = CYCLE;
    Resource[centerX + 26][centerY - 20] = CYCLE;
    Resource[centerX + 27][centerY - 19] = CYCLE;
    Resource[centerX + 28][centerY - 18] = CYCLE;
    Resource[centerX + 29][centerY - 17] = CYCLE;
    Resource[centerX + 30][centerY - 16] = CYCLE;
    Resource[centerX + 31][centerY - 15] = CYCLE;

    Resource[centerX - 5][centerY - 30] = RECT;
    Resource[centerX - 4][centerY - 30] = RECT;
    Resource[centerX - 3][centerY - 31] = RECT;
    Resource[centerX - 2][centerY - 32] = RECT;
    Resource[centerX - 1][centerY - 33] = RECT;
    Resource[centerX][centerY - 34] = RECT;
    Resource[centerX + 1][centerY - 35] = RECT;
    // 添加更多距离中心 5-10 格子的资源块
    Resource[centerX + 5][centerY + 5] = CYCLE;
    Resource[centerX + 6][centerY + 5] = CYCLE;
    Resource[centerX + 7][centerY + 6] = CYCLE;
    Resource[centerX + 8][centerY + 7] = CYCLE;
    Resource[centerX + 9][centerY + 8] = CYCLE;
    Resource[centerX + 10][centerY + 9] = CYCLE;
    Resource[centerX + 11][centerY + 10] = CYCLE;
    Resource[centerX - 5][centerY - 5] = CYCLE;
    Resource[centerX - 6][centerY - 5] = CYCLE;
    Resource[centerX - 7][centerY - 6] = CYCLE;
    Resource[centerX - 8][centerY - 7] = CYCLE;
    Resource[centerX - 9][centerY - 8] = CYCLE;
    Resource[centerX - 10][centerY - 9] = CYCLE;
    Resource[centerX - 11][centerY - 10] = CYCLE;

    Resource[centerX + 6][centerY - 6] = RECT;
    Resource[centerX + 7][centerY - 7] = RECT;
    Resource[centerX + 8][centerY - 8] = RECT;
    Resource[centerX + 9][centerY - 9] = RECT;
    Resource[centerX + 10][centerY - 10] = RECT;

    Resource[centerX - 6][centerY + 6] = RECT;
    Resource[centerX - 7][centerY + 7] = RECT;
    Resource[centerX - 8][centerY + 8] = RECT;
    Resource[centerX - 9][centerY + 9] = RECT;
    Resource[centerX - 10][centerY + 10] = RECT;


    Resource[1][3] = BARRIER;
    Resource[2][4] = BARRIER;
    Resource[4][2] = BARRIER;
    Resource[5][1] = BARRIER;
    Resource[9][6] = BARRIER;
    Resource[10][5] = BARRIER;
    Resource[9][3] = BARRIER;
    Resource[12][7] = BARRIER;
    Resource[8][9] = BARRIER;
    Resource[12][9] = BARRIER;
    Resource[12][10] = BARRIER;
    Resource[13][9] = BARRIER;
    Resource[13][8] = BARRIER;
    Resource[14][7] = BARRIER;
    Resource[2][13] = BARRIER;
    Resource[2][17] = BARRIER;
    Resource[3][14] = BARRIER;
    Resource[3][15] = BARRIER;
    Resource[4][10] = BARRIER;
    Resource[4][13] = BARRIER;
    Resource[5][14] = BARRIER;
    Resource[5][17] = BARRIER;
    Resource[6][18] = BARRIER;
    Resource[6][19] = BARRIER;
    Resource[7][20] = BARRIER;
    Resource[8][21] = BARRIER;
    Resource[9][23] = BARRIER;
    Resource[9][20] = BARRIER;
    Resource[10][14] = BARRIER;
    Resource[11][13] = BARRIER;
    Resource[12][14] = BARRIER;
    Resource[14][14] = BARRIER;
*/
}



void GameMap::SecondMap()
{
    /*
    // 增加资源
    Resource[9][16] = CYCLE;
    Resource[10][16] = CYCLE;
    Resource[10][17] = CYCLE;
    Resource[10][18] = CYCLE;
    Resource[11][17] = CYCLE;
    Resource[11][18] = CYCLE;
    Resource[12][18] = CYCLE;
    Resource[3][5] = RECT;
    Resource[3][6] = RECT;
    Resource[4][6] = RECT;
    Resource[4][7] = RECT;
    Resource[4][8] = RECT;
    Resource[5][7] = RECT;
    Resource[5][6] = RECT;
    Resource[5][5] = RECT;
    Resource[6][5] = RECT;
    Resource[6][8] = RECT;
*/
}

void GameMap::ClearBarriers()
{
    /*
    Resource[1][3] = NONE;
    Resource[2][4] = NONE;
    Resource[4][2] = NONE;
    Resource[5][1] = NONE;
    Resource[9][6] = NONE;
    Resource[10][5] = NONE;
    Resource[9][3] = NONE;
    Resource[12][7] = NONE;
    Resource[8][9] = NONE;
    Resource[12][9] = NONE;
    Resource[12][10] = NONE;
    Resource[13][9] = NONE;
    Resource[13][8] = NONE;
    Resource[14][7] = NONE;
    Resource[2][13] = NONE;
    Resource[2][17] = NONE;
    Resource[3][14] = NONE;
    Resource[3][15] = NONE;
    Resource[4][10] = NONE;
    Resource[4][13] = NONE;
    Resource[5][14] = NONE;
    Resource[5][17] = NONE;
    Resource[6][18] = NONE;
    Resource[6][19] = NONE;
    Resource[7][20] = NONE;
    Resource[8][21] = NONE;
    Resource[9][23] = NONE;
    Resource[9][20] = NONE;
    Resource[10][14] = NONE;
    Resource[11][13] = NONE;
    Resource[12][14] = NONE;
    Resource[14][14] = NONE;
*/
}
int GameMap::GetResource(GridVec pos)
{
    return Resource[pos.i][pos.j];
}
void GameMap::SetBuilding(GridVec pos, Building *building, int direction, int name)
{
    for (auto pos : building->BuildingAllPos())
    {
        BuildingsMap[pos.i][pos.j] = building;
        Buildingsmap[pos.i][pos.j] = building->name;
    }
}

Building *GameMap::GetBuilding(GridVec pos)
{
    return BuildingsMap[pos.i][pos.j];
}

Building *GameMap::GetBuilding(int i, int j)
{
    return BuildingsMap[i][j];
}

void GameMap::RemoveBuilding(GridVec pos)
{
    if (BuildingsMap[pos.i][pos.j]->name != HUB)
    {
        for (auto pos : BuildingsMap[pos.i][pos.j]->BuildingAllPos())
        {
            BuildingsMap[pos.i][pos.j] = nullptr;
            Buildingsmap[pos.i][pos.j] = -1;
        }
    }
}
GridVec GameMap::GetTatget(GridVec source, int directionout)
{
    GridVec target = source;
    switch (directionout)
    {
        {
        case UP:
            target.i--;
            break;
        case DOWN:
            target.i++;
            break;
        case LEFT:
            target.j--;
            break;
        case RIGHT:
            target.j++;
            break;
        }
    default:
        break;
    }
    return target;
}
int GameMap::OppositeDirection(int direction)
{
    switch (direction)
    {
    case UP:
        return DOWN;
    case DOWN:
        return UP;
    case LEFT:
        return RIGHT;
    case RIGHT:
        return LEFT;
    default:
        return NONE;
    }
}
