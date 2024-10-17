#include <windows.h>
#include "hub.h"
#include "config.h"
#include<cstdio>
#include "PlayScene.h"

int *Hub::need_shape_name = 0;
Hub::Hub(QObject* parent) : QObject(parent)
{
    CreateMapFile();
    need = NEED_CYCLE;
    *need_shape_name = CYCLE;
    current_have = 0;
    money = 0;
    increase_item_value = false;
    upgradehub = false;
    received_objects_last_10_second = new int(0);
    last_receive_time.start();
    last_received_shape = NONE;
    shape_update_timer.start();
    receivedTimestamps.clear();
}
Hub::Hub(GridVec pos, int name, int direction, QObject* parent) : Building(pos, name, direction)
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
    shape_update_timer.restart();

    // Record the timestamp of the received object
    QTime currentTime = QTime::currentTime();
    receivedTimestamps.enqueue(currentTime);

    // Clean up timestamps older than 10 seconds
    while (!receivedTimestamps.isEmpty() && receivedTimestamps.head().secsTo(currentTime) > 10) {
        receivedTimestamps.dequeue();
    }
    updateReceivedObjectsCount();

    // Update the currently received shape
    if (shapename != last_received_shape) {
        if (scene) {
            scene->current_received_shape = shapename;
        }
        last_received_shape = shapename;
        shape_update_timer.restart();
    }

    // Update the count of needed shapes
    if (shapename == *need_shape_name)
    {
        current_have++;
    }

    // Add money based on the received shape and item value state
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
        NULL,                    // Default security attributes
        PAGE_READWRITE,          // Read/write access
        0,                       // High-order file size
        sizeof(int),             // Low-order file size (size of an int)
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
    last_receive_time.restart();  // Restart the timer
}

void Hub::updateReceivedObjectsCount()
{
    QTime currentTime = QTime::currentTime();

    // Clean up timestamps older than 10 seconds
    while (!receivedTimestamps.isEmpty() && receivedTimestamps.head().secsTo(currentTime) > 10) {
        receivedTimestamps.dequeue();
    }

    // Output the number of objects received in the last 10 seconds
    *received_objects_last_10_second = receivedTimestamps.size();

    //qDebug() << "Objects received in the last 10 seconds: " << *(received_objects_last_10_second);
}
