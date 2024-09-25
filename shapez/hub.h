#ifndef HUB_H
#define HUB_H
#include <QTime>
#include <QElapsedTimer>
#include <QObject>
#include "Building.h"
#include <QQueue>
class Hub final :public QObject, public Building
{
public:
    Hub();
    Hub(GridVec pos, int name, int direction);
    Hub(QObject* parent = nullptr);
    Hub(GridVec pos, int name, int direction, QObject* parent = nullptr);
    /**
     * \brief 当前等级需要的物品个数
     */
    int need;

    /**
     * \brief 当前等级需要的物品类型
     */
    static int *need_shape_name;

    /**
     * \brief 当前等级已提交物品个数
     */
    int current_have;

    int money;
    // 新增：过去一秒接收到的物体数量
    int* received_objects_last_10_second;

    // 新增：过去一秒的时间标记
    QElapsedTimer last_receive_time;
    QElapsedTimer shape_update_timer;
    int last_received_shape;
    bool increase_item_value;

    bool upgradehub;

    std::vector<GridVec> BuildingAllPos() override;
    bool CanPlace(GridVec click, int picdirection, GameMap &gamemap) override;
    bool CanReceive(GridVec source, int directionin, int shapename) override;
    void Receive(GridVec source, int directionin, int shapename) override;
    void TickableRunning() override;
    void UpdateTickableState(GameMap &gamemap) override;
    void UpdateNeed();
    void CreateMapFile();
    // 重置接收计数器
    void resetReceiveCounter();
    void updateReceivedObjectsCount();
    QQueue<QTime> receivedTimestamps;  // 存储接收物体的时间戳
    ~Hub() {
        delete received_objects_last_10_second;
    }
};

#endif // HUB_H
