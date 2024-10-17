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
        // If out of bounds, return false
        if (pos.i < 0 || pos.i >= HEIGHT || pos.j < 0 || pos.j >= WIDTH)
        {
            return false;
        }
        // If on a mining area or there is an obstacle, return false
        if (gamemap.GetResource(pos) != NONE)
        {
            return false;
        }
        // If the clicked area is a hub, return false
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
    //this->timer.UpdateRunningTime(FirstRequire_ms);
    //running_ms = this->timer.running_ms;
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
            // The cutter is ready, preparing to transport
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
    // Retrieve row and column indices from the 'click' GridVec object
    int row = click.i;
    int col = click.j;

    // First loop: Check for neighbors in the horizontal range from -3 to +3
    for(int i = -3 ; i <= 3; i++) {
        // Check if the building at 'click' is a tunnel exit
        if (gamemap.GetBuilding(click)->name == TUNNEL_EXIT) {
            // Check if the neighboring building (with horizontal offset 'i') is also a tunnel
            // and if its direction matches the current tunnel's direction
            if (gamemap.GetBuilding(row + i, col)->name == TUNNEL_ENTRY
                && gamemap.GetBuilding(click)->direction == gamemap.GetBuilding(row + i, col)->direction) {
                // If a matching tunnel is found, return true (neighbor found)
                return true;
            }
        }
    }

    // Second loop: Check for neighbors in the vertical range from -3 to +3
    for(int j = -3 ; j <= 3; j++) {
        // Check if the building at 'click' is a tunnel exit
        if (gamemap.GetBuilding(click)->name == TUNNEL_EXIT) {
            // Check if the neighboring building (with vertical offset 'j') is also a tunnel
            // and if its direction matches the current tunnel's direction
            if (gamemap.GetBuilding(row, col + j)->name == TUNNEL_ENTRY
                && gamemap.GetBuilding(click)->direction == gamemap.GetBuilding(row, col + j)->direction) {
                // If a matching tunnel is found, return true (neighbor found)
                return true;
            }
        }
    }

    // If no matching neighbors are found in either loop, return false
    return false;
}
