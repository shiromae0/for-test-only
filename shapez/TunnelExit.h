#ifndef TUNNELEXIT_H
#define TUNNELEXIT_H
#include "Building.h"
/**
 * \brief 传送带
 */
class TunnelExit : public Building
{
public:
    TunnelExit();
    TunnelExit(GridVec pos, int name, int direction);
    std::vector<GridVec> BuildingAllPos() override;
    bool CanPlace(GridVec click, int picdirection, GameMap &gamemap) override;
    bool CanReceive(GridVec target, int directionin, int shapename) override;
    void Receive(GridVec target, int directionin, int shapename) override;
    void TickableRunning() override;
    void UpdateTickableState(GameMap &gamemap) override;
    bool HasTunnelEntry(GridVec click, int picdirection, GameMap &gamemap);
};
#endif // TUNNELEXIT_H
