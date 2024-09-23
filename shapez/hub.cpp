#include <windows.h>
#include "hub.h"
#include "config.h"
#include<cstdio>
#include "PlayScene.h"

int *Hub::need_shape_name = 0;
Hub::Hub()
{
    CreateMapFile();
    need = NEED_CYCLE;
    *need_shape_name = CYCLE;
    current_have = 0;
    money = 0;
    increase_item_value = false;
    upgradehub = false;
    received_objects_last_second = 0;  // 初始化为0
    last_receive_time.start();
    last_received_shape = NONE;  // 初始化为 NONE
    shape_update_timer.start();

}
Hub::Hub(GridVec pos, int name, int direction) : Building(pos, name, direction)
{
    need = NEED_CYCLE;
    *need_shape_name = CYCLE;
    current_have = 0;
    money = 0;
    increase_item_value = false;
    upgradehub = false;
}
std::vector<GridVec> Hub::BuildingAllPos()
{
    GridVec temp;
    std::vector<GridVec> allpos;
    if (!upgradehub)
    {
        allpos.push_back(pos);
        temp.i = pos.i + 1;
        temp.j = pos.j;
        allpos.push_back(temp);
        temp.i = pos.i + 1;
        temp.j = pos.j + 1;
        allpos.push_back(temp);
        temp.i = pos.i;
        temp.j = pos.j + 1;
        allpos.push_back(temp);
    }
    else
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                temp.i = pos.i + i;
                temp.j = pos.j + j;
                allpos.push_back(temp);
            }
        }
    }
    return allpos;
}
bool Hub::CanPlace(GridVec click, int picdirection, GameMap &gamemap)
{
    return true;
}
bool Hub::CanReceive(GridVec source, int directionin, int shapename)
{
    return true;
}
void Hub::Receive(GridVec source, int directionin, int shapename)
{
    PlayScene *scene = static_cast<PlayScene*>(this->parent());

    // 检查当前接收到的物体类型是否发生改变
    if (shapename != last_received_shape) {
        if (scene) {
            scene->current_received_shape = shapename;  // 更新物体类型
        }
        last_received_shape = shapename;  // 更新最后接收到的物体类型
        shape_update_timer.restart();     // 重置计时器
    }

    // 检查计时器是否超过 10 秒未接收到物体
    if (shape_update_timer.elapsed() >= 10000) {
        if (scene) {
            scene->current_received_shape = NONE;  // 10 秒未接收到物体，更新为 NONE
        }
        shape_update_timer.restart();  // 重启计时器
        last_received_shape = NONE;    // 重置最后接收到的物体类型为 NONE
    }
    if (shapename == *need_shape_name)
    {
        current_have++;
    }
    // 检查是否超过一秒
    if (last_receive_time.elapsed() >= 10000) {
        resetReceiveCounter();
    }
    // 增加接收到的物体计数
    received_objects_last_second++;
    if (!increase_item_value)
    {
        switch (shapename)
        {
        case CYCLE:
            money += CYCLE_MONEY_1;
            break;
        case RECT:
            money += RECT_MONEY_1;
            break;
        case LEFT_CYCLE:
            money += LEFT_CYCLE_MONEY_1;
            break;
        case RIGHT_CYCLE:
            money += RIGHT_CYCLE_MONEY_1;
            break;
        default:
            break;
        }
    }
    else
    {
        switch (shapename)
        {
        case CYCLE:
            money += CYCLE_MONEY_2;
            break;
        case RECT:
            money += RECT_MONEY_2;
            break;
        case LEFT_CYCLE:
            money += LEFT_CYCLE_MONEY_2;
            break;
        case RIGHT_CYCLE:
            money += RIGHT_CYCLE_MONEY_2;
            break;
        default:
            break;
        }
    }
    return;
}
void Hub::UpdateTickableState(GameMap &gamemap)
{
    return;
}
void Hub::TickableRunning()
{
    return;
}
void Hub::UpdateNeed()
{
    switch (*need_shape_name)
    {
    case CYCLE:
        *need_shape_name = RECT;
        current_have = 0;
        need = NEED_RECT;
        break;
    case RECT:
        *need_shape_name = LEFT_CYCLE;
        current_have = 0;
        need = NEED_LEFT_CYCLE;
        break;
    case LEFT_CYCLE:
        *need_shape_name = RIGHT_CYCLE;
        current_have = 0;
        need = NEED_RIGHT_CYCLE;
        break;
    default:
        break;
    }
}
void Hub::CreateMapFile(){
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,                    // 默认安全属性
        PAGE_READWRITE,          // 读/写权限
        0,                       // 文件的高32位大小
        sizeof(int),             // 文件的低32位大小（int 的大小）
        L"need_shape_name"
        );
    need_shape_name = (int*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int));

    if (need_shape_name == NULL) {
        printf("Could not map view of file (%d).\n", GetLastError());
        CloseHandle(hMapFile);
    }
}
void Hub::resetReceiveCounter()
{
    received_objects_last_second = 0;
    last_receive_time.restart();  // 重启计时器
}

