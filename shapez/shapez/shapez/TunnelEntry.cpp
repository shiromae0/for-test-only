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
    //this->timer.UpdateRunningTime(FirstRequire_ms);
    //running_ms = this->timer.running_ms;
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
            // The rotator is ready, preparing to transport
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
    // Retrieve row and column indices from the 'pos' GridVec object
    int row = pos.i;
    int col = pos.j;

    // Check if the building at position 'pos' is a tunnel entry
    auto currentBuilding = map.GetBuilding(pos);
    if (currentBuilding == nullptr || currentBuilding->name != TUNNEL_ENTRY) {
        return false;  // If the current position is not a tunnel entry, return false
    }

    // First loop: check for neighbors in the horizontal range from -3 to +3
    for (int i = -3; i <= 3; i++) {
        int newRow = row + i;  // Calculate the new row
        if (newRow >= 0 && newRow < WIDTH) {  // Check boundaries
            auto neighborBuilding = map.GetBuilding(newRow, col);  // Get the neighboring building
            if (neighborBuilding != nullptr && neighborBuilding->name == TUNNEL_EXIT
                && currentBuilding->direction == neighborBuilding->direction) {
                return true;  // Matching tunnel exit found
            }
        }
    }

    // Second loop: check for neighbors in the vertical range from -3 to +3
    for (int j = -3; j <= 3; j++) {
        int newCol = col + j;  // Calculate the new column
        if (newCol >= 0 && newCol < WIDTH) {  // Check boundaries
            auto neighborBuilding = map.GetBuilding(row, newCol);  // Get the neighboring building
            if (neighborBuilding != nullptr && neighborBuilding->name == TUNNEL_EXIT
                && currentBuilding->direction == neighborBuilding->direction) {
                return true;  // Matching tunnel exit found
            }
        }
    }

    // If no matching neighbors are found in either loop, return false
    return false;
}


