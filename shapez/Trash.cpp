#include "Trash.h"
#include "gamemap.h"
Trash::Trash()
{
}
Trash::Trash(GridVec pos, int name, int direction) : Building(pos, name, direction)
{
}
std::vector<GridVec> Trash::BuildingAllPos()
{
    // TODO
    std::vector<GridVec> allpos;
    allpos.push_back(pos);
    return allpos;
}
bool Trash::CanPlace(GridVec click, int picdirection, GameMap &gamemap)
{
    // If outside map，return false
    if (pos.i < 0 || pos.i >= HEIGHT || pos.j < 0 || pos.j >= WIDTH)
    {
        return false;
    }
    // if on mine, return false
    if (gamemap.GetResource(pos) != NONE)
    {
        return false;
    }
    // if click hub，return false
    if (gamemap.GetBuilding(pos) != nullptr)
    {
        if (gamemap.GetBuilding(pos)->name == HUB || gamemap.GetBuilding(pos)->name == CUTTER || gamemap.GetBuilding(pos)->name == TRASH)
        {
            return false;
        }
    }
    return true;
}
bool Trash::CanReceive(GridVec source, int directionin, int shapename)
{
    return true;
}
void Trash::Receive(GridVec source, int directionin, int shapename)
{
    return;
}
void Trash::UpdateTickableState(GameMap &gamemap)
{
    return;
}
void Trash::TickableRunning()
{
    return;
}
