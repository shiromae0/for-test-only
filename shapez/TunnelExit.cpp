#include "TunnelExit.h"
#include "GameMap.h"


// Default constructor
TunnelExit::TunnelExit() {
    FirstRequire_ms = 2700;  // Initialize teleportation time
    SecondRequire_ms = 1800; // Secondary teleportation time
}

// Parameterized constructor
TunnelExit::TunnelExit(GridVec pos, int name, int direction)
    : Building(pos, name, direction) {
    FirstRequire_ms = 2700;  // Initialize teleportation time
    SecondRequire_ms = 1800; // Secondary teleportation time
}

std::vector<GridVec> TunnelExit::BuildingAllPos()
{
    // TODO
    std::vector<GridVec> allpos;
    allpos.push_back(pos);
    return allpos;
}



bool TunnelExit::CanPlace(GridVec click, int picdirection, GameMap &gamemap)
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



bool TunnelExit::CanReceive(GridVec target, int directionin, int shapename)
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



void TunnelExit::Receive(GridVec target, int directionin, int shapename)
{
    this->shape.name = shapename;
    this->state = RUNNING;
    this->timer.Reset();
}
void TunnelExit::TickableRunning()
{

    this->timer.UpdateRuningTime(FirstRequire_ms);
    running_ms = this->timer.running_ms;
    return;
}
void TunnelExit::UpdateTickableState(GameMap &gamemap)
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
            if (this->CanSend(BuildingAllPos()[0], direction, LEFT_CYCLE, gamemap) && this->CanSend(BuildingAllPos()[1], direction, RIGHT_CYCLE, gamemap))
            {
                this->Send(BuildingAllPos()[0], direction, LEFT_CYCLE, gamemap);
                this->Send(BuildingAllPos()[1], direction, RIGHT_CYCLE, gamemap);
                state = EMPTY;
                this->shape.name = NONE;
            }
            break;
        case DOWN:
        case LEFT:
            if (this->CanSend(BuildingAllPos()[0], direction, RIGHT_CYCLE, gamemap) && this->CanSend(BuildingAllPos()[1], direction, LEFT_CYCLE, gamemap))
            {
                this->Send(BuildingAllPos()[0], direction, RIGHT_CYCLE, gamemap);
                this->Send(BuildingAllPos()[1], direction, LEFT_CYCLE, gamemap);
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


bool TunnelExit::HasTunnelEntry(GridVec click, int picdirection, GameMap &gamemap){
    // 从 GridVec 对象 'click' 中获取行和列的索引
    int row = click.i;
    int col = click.j;

    // 第一层循环：检查水平范围内 -3 到 +3 的邻居
    for(int i = -3 ; i <= 3; i++) {
        // 检查当前位置 'click' 的建筑是否是隧道
        if (gamemap.GetBuilding(click)->name == TUNNEL_EXIT) {
            // 检查水平方向上相邻的建筑（水平偏移 'i'）是否也是隧道
            // 并且相邻隧道的方向是否与当前隧道的方向一致
            if (gamemap.GetBuilding(row + i, col)->name == TUNNEL_ENTRY
                && gamemap.GetBuilding(click)->direction == gamemap.GetBuilding(row + i, col)->direction) {
                // 如果找到匹配的隧道，返回 true（找到邻居）
                return true;
            }
        }
    }

    // 第二层循环：检查垂直范围内 -3 到 +3 的邻居
    for(int j = -3 ; j <= 3; j++) {
        // 检查当前位置 'click' 的建筑是否是隧道
        if (gamemap.GetBuilding(click)->name == TUNNEL_EXIT) {
            // 检查垂直方向上相邻的建筑（垂直偏移 'j'）是否也是隧道
            // 并且相邻隧道的方向是否与当前隧道的方向一致
            if (gamemap.GetBuilding(row, col + j)->name == TUNNEL_ENTRY
                && gamemap.GetBuilding(click)->direction == gamemap.GetBuilding(row, col + j)->direction) {
                // 如果找到匹配的隧道，返回 true（找到邻居）
                return true;
            }
        }
    }

    // 如果在两个循环中都没有找到匹配的邻居，返回 false
    return false;
}


