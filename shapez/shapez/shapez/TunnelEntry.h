#ifndef TUNNELENTRY_H
#define TUNNELENTRY_H

#include "Building.h"
#include <string>
#include <vector>

/**
 * \brief 传送带
 */
class TunnelEntry : public Building
{
public:
    TunnelEntry();  // 默认构造函数
    TunnelEntry(GridVec pos, int name, int direction);  // 重载构造函数

    std::vector<GridVec> BuildingAllPos() override;
    bool CanPlace(GridVec click, int picdirection, GameMap &gamemap) override;
    bool CanReceive(GridVec target, int directionin, int shapename) override;
    void Receive(GridVec target, int directionin, int shapename) override;
    void TickableRunning() override;
    void UpdateTickableState(GameMap &gamemap) override;
    bool HasTunnelExit(GridVec pos, GameMap map);
};

#endif // TUNNELENTRY_H
