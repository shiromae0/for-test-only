#include "Rotator.h"
#include "GameMap.h"
Rotator::Rotator()
{
    FirstRequire_ms = 2700;
    SecondRequire_ms = 1800;
}
Rotator::Rotator(GridVec pos, int name, int direction)
    : Building(pos, name, direction)
{
    FirstRequire_ms = 2700;
    SecondRequire_ms = 1800;
}
std::vector<GridVec> Rotator::BuildingAllPos()
{
    std::vector<GridVec> allpos;
    allpos.push_back(pos);
    return allpos;
}
bool Rotator::CanPlace(GridVec click, int picdirection, GameMap &gamemap)
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
            if (gamemap.GetBuilding(pos)->name == HUB || gamemap.GetBuilding(pos)->name == ROTATOR || gamemap.GetBuilding(pos)->name == TRASH)
            {
                return false;
            }
        }
    }
    return true;
}
bool Rotator::CanReceive(GridVec target, int directionin, int shapename)
{
    if (state == EMPTY)
    {
        if (directionin == direction)
        {
            if (shapename == CYCLE or RECT or LEFT_CYCLE or shapename == LEFT_RECT or shapename == RIGHT_CYCLE or shapename == RIGHT_RECT or shapename == UP_RECT or shapename == DOWN_RECT or shapename == UP_CYCLE or shapename == DOWN_CYCLE)
            {
                return true;
            }
        }
    }
    return false;
}
void Rotator::Receive(GridVec target, int directionin, int shapename)
{
    this->shape.name = shapename;
    this->state = RUNNING;
    this->timer.Reset();
}
void Rotator::TickableRunning()
{

    this->timer.UpdateRuningTime(FirstRequire_ms);
    running_ms = this->timer.running_ms;
    return;
}
void Rotator::UpdateTickableState(GameMap &gamemap)
{

    switch (state)
    {
    case EMPTY:
        running_ms = 0;
        break;
    case RUNNING:
        if (running_ms >= FirstRequire_ms)
        {
            // Rotator好了，准备运输
            state = BLOCK;
            running_ms = 0;
        }
        else
        {
            this->TickableRunning();
        }
        break;
    case BLOCK:
        if (this->shape.name == CYCLE) {
            // 当前形状是右矩形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
        if (this->CanSend(BuildingAllPos()[0], direction, CYCLE, gamemap) )
            {
                this->Send(BuildingAllPos()[0], direction, CYCLE, gamemap);
                state = EMPTY;
                this->shape.name = NONE;
            }
                break;
            default:
                break;
            }
        } else if (this->shape.name == RECT) {
            // 当前形状是右矩形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, RECT, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, RECT, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == LEFT_RECT) {
            // 当前形状是左矩形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, LEFT_RECT, gamemap))
                {
                    this->Send(BuildingAllPos()[0], direction, UP_RECT, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == RIGHT_RECT) {
            // 当前形状是右矩形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, RIGHT_RECT, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, DOWN_RECT, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == UP_RECT) {
            // 当前形状是上矩形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, UP_RECT, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, RIGHT_RECT, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == DOWN_RECT) {
            // 当前形状是下矩形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, DOWN_RECT, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, LEFT_RECT, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == LEFT_CYCLE) {
            // 当前形状是左园形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, LEFT_CYCLE, gamemap))
                {
                    this->Send(BuildingAllPos()[0], direction, UP_CYCLE, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == RIGHT_CYCLE) {
            // 当前形状是右园形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, RIGHT_CYCLE, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, DOWN_CYCLE, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == UP_CYCLE) {
            // 当前形状是上园形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, UP_CYCLE, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, RIGHT_CYCLE, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        } else if (this->shape.name == DOWN_CYCLE) {
            // 当前形状是下园形
            switch (direction)
            {
            case UP:
            case RIGHT:
            case DOWN:
            case LEFT:
                if (this->CanSend(BuildingAllPos()[0], direction, DOWN_CYCLE, gamemap) )
                {
                    this->Send(BuildingAllPos()[0], direction, LEFT_CYCLE, gamemap);
                    state = EMPTY;
                    this->shape.name = NONE;
                }
                break;
            default:
                break;
            }
        }
        break;
        break;
    default:
        break;
    }
    return;
}
