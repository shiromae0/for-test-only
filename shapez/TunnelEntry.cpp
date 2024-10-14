#include "TunnelEntry.h"
#include "GameMap.h"
#include <iostream>


// Default constructor
#include "TunnelEntry.h"

// Default constructor
TunnelEntry::TunnelEntry() {
    FirstRequire_ms = 2700;  // Initialize teleportation time
    SecondRequire_ms = 1800; // Secondary teleportation time
}

// Parameterized constructor
TunnelEntry::TunnelEntry(GridVec pos, int name, int direction)
    : Building(pos, name, direction) {
    FirstRequire_ms = 2700;  // Initialize teleportation time
    SecondRequire_ms = 1800; // Secondary teleportation time
    this->name = name;
}

std::vector<GridVec> TunnelEntry::BuildingAllPos()
{
    // TODO
    std::vector<GridVec> allpos;
    allpos.push_back(pos);
    return allpos;
}



bool TunnelEntry::CanPlace(GridVec click, int picdirection, GameMap &gamemap)
{
    for (auto pos : BuildingAllPos())
    {
        // 如果超出地图范围，返回false
        if (pos.i < 0 || pos.i >= HEIGHT || pos.j < 0 || pos.j >= WIDTH)
        {
            return false;
        }
        // 如果在矿地上，或有障碍物，返回false
        if (gamemap.GetResource(pos) != NONE)
        {
            return false;
        }
        // 如果点击的是hub，返回false
        if (gamemap.GetBuilding(pos) != nullptr)
        {
            if (gamemap.GetBuilding(pos)->name == HUB || gamemap.GetBuilding(pos)->name == CUTTER || gamemap.GetBuilding(pos)->name == TRASH)
            {
                return false;
            }
        }
    }
    return true;
}



bool TunnelEntry::CanReceive(GridVec target, int directionin, int shapename)
{
    if (state == EMPTY)
    {
        if (directionin == direction)
        {
            if (shapename == CYCLE)
            {
                return true;
            }
        }
    }
    return false;
}



void TunnelEntry::Receive(GridVec target, int directionin, int shapename)
{
    this->shape.name = shapename;
    this->state = RUNNING;
    this->timer.Reset();
}
void TunnelEntry::TickableRunning()
{

    this->timer.UpdateRuningTime(FirstRequire_ms);
    running_ms = this->timer.running_ms;
    return;
}
void TunnelEntry::UpdateTickableState(GameMap &gamemap)
{

    switch (state)
    {
    case EMPTY:
        running_ms = 0;
        break;
    case RUNNING:
        if (running_ms >= FirstRequire_ms)
        {
            // cutter好了，准备运输
            state = BLOCK;
            running_ms = 0;
        }
        else
        {
            this->TickableRunning();
        }
        break;
    case BLOCK:
        switch (direction)
        {
        case UP:
        case RIGHT:
            if (this->CanSend(BuildingAllPos()[0], direction, CYCLE, gamemap))
            {
                this->Send(BuildingAllPos()[0], direction, CYCLE, gamemap);
                state = EMPTY;
                this->shape.name = NONE;
            }
            break;
        case DOWN:
        case LEFT:
            if (this->CanSend(BuildingAllPos()[0], direction, RIGHT_CYCLE, gamemap))
            {
                this->Send(BuildingAllPos()[0], direction, RIGHT_CYCLE, gamemap);
                state = EMPTY;
                this->shape.name = NONE;
            }
            break;
        default:
            break;
        }
    default:
        break;
    }
    return;
}

bool TunnelEntry::HasTunnelExit(GridVec pos, GameMap map) {
    // 从 GridVec 对象 'pos' 中获取行和列的索引
    int row = pos.i;
    int col = pos.j;

    // 检查当前位置 'pos' 的建筑是否是隧道入口
    auto currentBuilding = map.GetBuilding(pos);
    if (currentBuilding == nullptr || currentBuilding->name != TUNNEL_ENTRY) {
        return false;  // 如果当前位置不是隧道入口，直接返回 false
    }

    // 第一层循环：检查水平范围内 -3 到 +3 的邻居
    for (int i = -3; i <= 3; i++) {
        int newRow = row + i;  // 计算新的行
        if (newRow >= 0 && newRow < WIDTH) {  // 检查边界
            auto neighborBuilding = map.GetBuilding(newRow, col);  // 获取邻居建筑
            if (neighborBuilding != nullptr && neighborBuilding->name == TUNNEL_EXIT
                && currentBuilding->direction == neighborBuilding->direction) {
                return true;  // 找到匹配的隧道出口
            }
        }
    }

    // 第二层循环：检查垂直范围内 -3 到 +3 的邻居
    for (int j = -3; j <= 3; j++) {
        int newCol = col + j;  // 计算新的列
        if (newCol >= 0 && newCol < WIDTH) {  // 检查边界
            auto neighborBuilding = map.GetBuilding(row, newCol);  // 获取邻居建筑
            if (neighborBuilding != nullptr && neighborBuilding->name == TUNNEL_EXIT
                && currentBuilding->direction == neighborBuilding->direction) {
                return true;  // 找到匹配的隧道出口
            }
        }
    }

    // 如果在两个循环中都没有找到匹配的邻居，返回 false
    return false;
}



