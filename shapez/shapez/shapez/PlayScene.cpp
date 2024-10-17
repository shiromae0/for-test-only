#include <windows.h>
#include "playscene.h"
#include "Miner.h"
#include "Belt.h"
#include "Cutter.h"
#include "Trash.h"
#include "TunnelEntry.h"
#include "TunnelExit.h"
#include "Rotator.h"
#include "config.h"
#include <QPainter>
#include <iostream>
#include <QSet>
#include <QList>
void PlayScene::CreateMapFile() {
    Qpointmapx = new int;
    Qpointmapy = new int;
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,   // Create an anonymous file mapping
        NULL,                   // Default security attributes
        PAGE_READWRITE,         // Read-write permission
        0,                      // High-order file size
        sizeof(int) * 2,        // Size for two ints (x and y)
        L"scrolloffset"         // Shared memory name
        );

    if (hMapFile == NULL) {
        return;
    }
    int* pMappedMemory = (int*) MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(int) * 2  // Space for two ints
        );
    if (pMappedMemory == NULL) {
        CloseHandle(hMapFile);
        return;
    }
    // Point the x and y pointers to the shared memory
    Qpointmapx = &pMappedMemory[0];
    Qpointmapy = &pMappedMemory[1];
    *Qpointmapx = 200;
    *Qpointmapy = 85;
    //CloseHandle(hMapFile);
    scaleFcator_map = new double;

    HANDLE ScaleMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,   // Create an anonymous file mapping
        NULL,                   // Default security attributes
        PAGE_READWRITE,         // Read-write permission
        0,                      // High-order file size
        sizeof(double),         // Size for one double
        L"scaleFactor"          // Shared memory name
        );

    if (ScaleMapFile == NULL) {
        return;
    }

    double* sMappedMemory = (double*) MapViewOfFile(
        ScaleMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(double)          // Space for one double
        );

    if (sMappedMemory == NULL) {
        CloseHandle(ScaleMapFile);  // Use ScaleMapFile instead of hMapFile
        return;
    }

    scaleFcator_map = sMappedMemory;  // No need to take the address
}

PlayScene::PlayScene() {
    CreateMapFile();
    // Initialize pixel images
    *scaleFcator_map = 1.0;
    hub_small_img.load(HUB_SMALL_PATH); // Delivery center
    hub_big_img.load(HUB_BIG_PATH);
    cycle_mine_field_img.load(CYCLE_MINE_PATH); // Mining field
    rect_mine_field_img.load(RECT_MINE_PATH);
    barrier_img.load(BARRIER_PATH); // Barrier
    miner_img_W.load(MINER_W_PATH); // Miner
    miner_img_A.load(MINER_A_PATH);
    miner_img_D.load(MINER_D_PATH);
    miner_img_S.load(MINER_S_PATH);
    belt_img_A.load(BELT_A_PATH); // Conveyor belt
    belt_img_A_S.load(BELT_A_S_PATH);
    belt_img_A_W.load(BELT_A_W_PATH);
    belt_img_D.load(BELT_D_PATH);
    belt_img_D_S.load(BELT_D_S_PATH);
    belt_img_D_W.load(BELT_D_W_PATH);
    belt_img_S.load(BELT_S_PATH);
    belt_img_S_A.load(BELT_S_A_PATH);
    belt_img_S_D.load(BELT_S_D_PATH);
    belt_img_W.load(BELT_W_PATH);
    belt_img_W_A.load(BELT_W_A_PATH);
    belt_img_W_D.load(BELT_W_D_PATH);
    cutter_img_A.load(CUTTER_A_PATH); // Cutter
    cutter_img_D.load(CUTTER_D_PATH);
    cutter_img_S.load(CUTTER_S_PATH);
    cutter_img_W.load(CUTTER_W_PATH);
    trash_img_A.load(TRASH_A_PATH); // Trash bin
    trash_img_D.load(TRASH_D_PATH);
    trash_img_S.load(TRASH_S_PATH);
    trash_img_W.load(TRASH_W_PATH);
    tunnel_entry_A.load(TUNNEL_ENTRY_A_PATH); // Tunnel entry
    tunnel_entry_D.load(TUNNEL_ENTRY_D_PATH);
    tunnel_entry_S.load(TUNNEL_ENTRY_S_PATH);
    tunnel_entry_W.load(TUNNEL_ENTRY_W_PATH);
    tunnel_exit_A.load(TUNNEL_EXIT_A_PATH); // Tunnel exit
    tunnel_exit_D.load(TUNNEL_EXIT_D_PATH);
    tunnel_exit_S.load(TUNNEL_EXIT_S_PATH);
    tunnel_exit_W.load(TUNNEL_EXIT_W_PATH);
    rotator_img_A.load(ROTATOR_A_PATH); // Rotator
    rotator_img_D.load(ROTATOR_D_PATH);
    rotator_img_S.load(ROTATOR_S_PATH);
    rotator_img_W.load(ROTATOR_W_PATH);
    miner_img_blue_W.load(MINER_BLUE_W_PATH); // Miner blueprint
    miner_img_blue_A.load(MINER_BLUE_A_PATH);
    miner_img_blue_D.load(MINER_BLUE_D_PATH);
    miner_img_blue_S.load(MINER_BLUE_S_PATH);
    belt_img_blue_A.load(BELT_BLUE_A_PATH); // Conveyor belt blueprint
    belt_img_blue_A_S.load(BELT_BLUE_A_S_PATH);
    belt_img_blue_A_W.load(BELT_BLUE_A_W_PATH);
    belt_img_blue_D.load(BELT_BLUE_D_PATH);
    belt_img_blue_D_S.load(BELT_BLUE_D_S_PATH);
    belt_img_blue_D_W.load(BELT_BLUE_D_W_PATH);
    belt_img_blue_S.load(BELT_BLUE_S_PATH);
    belt_img_blue_S_A.load(BELT_BLUE_S_A_PATH);
    belt_img_blue_S_D.load(BELT_BLUE_S_D_PATH);
    belt_img_blue_W.load(BELT_BLUE_W_PATH);
    belt_img_blue_W_A.load(BELT_BLUE_W_A_PATH);
    belt_img_blue_W_D.load(BELT_BLUE_W_D_PATH);
    cutter_img_blue_A.load(CUTTER_BLUE_A_PATH); // Cutter blueprint
    cutter_img_blue_D.load(CUTTER_BLUE_D_PATH);
    cutter_img_blue_S.load(CUTTER_BLUE_S_PATH);
    cutter_img_blue_W.load(CUTTER_BLUE_W_PATH);
    trash_img_blue_A.load(TRASH_BLUE_A_PATH); // Trash bin blueprint
    trash_img_blue_D.load(TRASH_BLUE_D_PATH);
    trash_img_blue_S.load(TRASH_BLUE_S_PATH);
    trash_img_blue_W.load(TRASH_BLUE_W_PATH);
    tunnel_blue_A.load(TUNNEL_BLUE_A_PATH); // Tunnel blueprint
    tunnel_blue_D.load(TUNNEL_BLUE_D_PATH);
    tunnel_blue_S.load(TUNNEL_BLUE_S_PATH);
    tunnel_blue_W.load(TUNNEL_BLUE_W_PATH);

    cycle_img.load(CYCLE_PATH); // Items
    rect_img.load(RECT_PATH);
    left_cycle_img.load(LEFT_CYCLE_PATH);
    right_cycle_img.load(RIGHT_CYCLE_PATH);
    up_cycle_img.load(UP_CYCLE_PATH);
    down_cycle_img.load(DOWN_CYCLE_PATH);

    left_rect_img.load(LEFT_RECT_PATH);
    right_rect_img.load(RIGHT_RECT_PATH);
    up_rect_img.load(UP_RECT_PATH);
    down_rect_img.load(DOWN_RECT_PATH);
    // Initialize buttons
    shop.setParent(this);
    shop.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    shop.setStyleSheet("QPushButton{image: url(:/res/shop.png)}");
    miner_button.setParent(this);
    miner_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    miner_button.setStyleSheet("QPushButton{image: url(:/res/miner_button.png)}");
    belt_button.setParent(this);
    belt_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    belt_button.setStyleSheet("QPushButton{image: url(:/res/belt_button.png)}");
    cutter_button.setParent(this);
    cutter_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    cutter_button.setStyleSheet("QPushButton{image: url(:/res/cutter_button.png)}");
    trash_button.setParent(this);
    trash_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    trash_button.setStyleSheet("QPushButton{image: url(:/res/trash_button.png)}");
    //tunnel_entry_button.setParent(this);
    //tunnel_entry_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    //tunnel_entry_button.setStyleSheet("QPushButton{image: url(:/res/tunnel_entry_W.png); }");
    //tunnel_exit_button.setParent(this);
    //tunnel_exit_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    //tunnel_exit_button.setStyleSheet("QPushButton{image: url(:/res/tunnel_exit_W.png)}");
    rotator_img_blue_A.load(ROTATOR_BLUE_A_PATH); // Rotator blueprint
    rotator_img_blue_D.load(ROTATOR_BLUE_D_PATH);
    rotator_img_blue_S.load(ROTATOR_BLUE_S_PATH);
    rotator_img_blue_W.load(ROTATOR_BLUE_W_PATH);
    rotator_button.setParent(this);
    rotator_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    rotator_button.setStyleSheet("QPushButton{image: url(:/res/rotator_button.png)}");
    help_button.setParent(this);
    help_button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE);
    help_button.setStyleSheet("QPushButton{image: url(:/res/help.png)}");
    // Map without mining fields added
    map.FirstMap();
    // No building is placed at the moment
    is_placing_belt = false;
    ready_to_place_belt = false;
    current_building_name = NONE;
    current_building_direction = UP;
    // No partial upgrades to buildings
    upgrade = NONE;
    // Game level
    round = 1;
    // Timer for factory operation
    timer.setInterval(GAME_HZ);
    this->FactoryRunning();
    // Capture mouse position in real-time
    setMouseTracking(true);
    cur_x = 0;
    cur_y = 0;
    // Right-click drag to delete
    right_button_pressed = false;
    QTimer::singleShot(0, this, &PlayScene::initializeView);
    elapsedTimer.start();  // Start timer
    current_received_shape = NONE;
    shape_output_timer = new QTimer(this);
    connect(shape_output_timer, &QTimer::timeout, this, &PlayScene::outputCurrentShape);
    shape_output_timer->start(1000);  // Trigger every second
    connect(shape_output_timer, &QTimer::timeout, this, &PlayScene::checkResetReceivedShape);
    hub = new Hub(this);
    QTimer* updateTimer = new QTimer(this);
    connect(updateTimer, &QTimer::timeout, this, &PlayScene::updateHubStats);  // Call updateHubStats every second
    updateTimer->start(1000);  // 1000 milliseconds == 1 second
    addTopLeftButton();

    //tunnel_check_timer = new QTimer(this);
    //tunnel_check_timer->start(100);
    //connect(tunnel_check_timer, &QTimer::timeout, this, &PlayScene::checkTunnels);
}

void PlayScene::draw_cells()
{
    // Draw the background light gray grid
    QPen pen;                            // Pen
    pen.setColor(QColor(220, 220, 220)); // Light gray color
    QPainter gridPainter(this);          // Painter
    gridPainter.setPen(pen);             // Painter holds the pen
    gridPainter.scale(scaleFactor, scaleFactor);
    gridPainter.translate(scroll_offset);
    for (int y = 0; y <= HEIGHT * CELLSIZE; y += CELLSIZE)
    {
        gridPainter.drawLine(0, y, WIDTH * CELLSIZE, y);
    }
    for (int x = 0; x <= WIDTH * CELLSIZE; x += CELLSIZE)
    {
        gridPainter.drawLine(x, 0, x, HEIGHT * CELLSIZE);
    }
}

void PlayScene::draw_map_resources()
{
    // Draw the mining fields
    QPainter mineFieldPainter(this);
    mineFieldPainter.scale(scaleFactor, scaleFactor);
    mineFieldPainter.translate(scroll_offset);
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            switch (map.Resource[i][j])
            {
            case CYCLE:
                mineFieldPainter.drawPixmap(j * CELLSIZE, i * CELLSIZE, CELLSIZE, CELLSIZE, cycle_mine_field_img);
                break;
            case RECT:
                mineFieldPainter.drawPixmap(j * CELLSIZE, i * CELLSIZE, CELLSIZE, CELLSIZE, rect_mine_field_img);
                break;
            case BARRIER:
                mineFieldPainter.drawPixmap(j * CELLSIZE, i * CELLSIZE, CELLSIZE, CELLSIZE, barrier_img);
                break;
            default:
                break;
            }
        }
    }
}

void PlayScene::draw_building()
{
    if (buildings.size() > 0)
    {
        QPainter painter(this);
        painter.scale(scaleFactor, scaleFactor);
        painter.translate(scroll_offset);
        for (int i = 0; i < buildings.size(); i++)
        {
            Building *building = buildings[i];
            switch (building->name)
            {
            case HUB:
                switch (round)
                {
                case 1:
                    aim_img.load(AIM1_PATH);
                    break;
                case 2:
                    aim_img.load(AIM2_PATH);
                    break;
                case 3:
                    aim_img.load(AIM3_PATH);
                    break;
                case 4:
                    aim_img.load(AIM4_PATH);
                    break;
                case 5:
                    aim_img.load(AIM1_PATH);
                    break;
                default:
                    aim_img.load(AIM1_PATH);
                    break;
                }
                if (!hub->upgradehub)
                {
                    // Unupgraded hub
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE * 2, CELLSIZE * 2, hub_small_img);

                    // Calculate the center point of the unupgraded hub and draw the scaled-down aim_img
                    int center_x = building->pos.j * CELLSIZE + CELLSIZE; // x-coordinate of the center point
                    int center_y = building->pos.i * CELLSIZE + CELLSIZE; // y-coordinate of the center point

                    // Scaled-down size
                    int aim_width = CELLSIZE / 1.5;
                    int aim_height = CELLSIZE / 1.5;
                    if (round < 3 || round == 5) {
                        painter.drawPixmap(center_x - aim_width / 2, center_y - aim_height / 2, aim_width, aim_height, aim_img);
                    }
                    else if (round >= 3 && round <= 4) {
                        painter.drawPixmap(center_x - aim_width / 2, center_y - aim_height / 2, aim_width / 2, aim_height, aim_img);
                    }
                    // Center the scaled-down aim_img

                    update();
                }
                else
                {
                    // Upgraded hub
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE * 4, CELLSIZE * 4, hub_big_img);

                    // Calculate the center point of the upgraded hub and draw aim_img
                    int center_x = building->pos.j * CELLSIZE + 2 * CELLSIZE; // x-coordinate of the center point
                    int center_y = building->pos.i * CELLSIZE + 2 * CELLSIZE; // y-coordinate of the center point
                    painter.drawPixmap(center_x - CELLSIZE / 2, center_y - CELLSIZE / 2, CELLSIZE, CELLSIZE, aim_img);
                }
                break;
            case TRASH:
                switch (building->direction)
                {
                case UP:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, trash_img_W);
                    break;
                case DOWN:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, trash_img_S);
                    break;
                case LEFT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, trash_img_A);
                    break;
                case RIGHT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, trash_img_D);
                    break;
                default:
                    break;
                }
                break;
            case MINER:
                switch (building->direction)
                {
                case UP:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, miner_img_W);
                    break;
                case DOWN:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, miner_img_S);
                    break;
                case LEFT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, miner_img_A);
                    break;
                case RIGHT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, miner_img_D);
                    break;
                default:
                    break;
                }
                break;
            case BELT_A:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_A);
                break;
            case BELT_A_S:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_A_S);
                break;
            case BELT_A_W:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_A_W);
                break;
            case BELT_D:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_D);
                break;
            case BELT_D_S:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_D_S);
                break;
            case BELT_D_W:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_D_W);
                break;
            case BELT_S:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_S);
                break;
            case BELT_S_A:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_S_A);
                break;
            case BELT_S_D:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_S_D);
                break;
            case BELT_W:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_W);
                break;
            case BELT_W_A:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_W_A);
                break;
            case BELT_W_D:
                painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_W_D);
                break;
            case CUTTER:
                switch (building->direction)
                {
                case UP:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE * 2, CELLSIZE, cutter_img_W);
                    break;
                case DOWN:
                    painter.drawPixmap((building->pos.j - 1) * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE * 2, CELLSIZE, cutter_img_S);
                    break;
                case LEFT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, (building->pos.i - 1) * CELLSIZE, CELLSIZE, CELLSIZE * 2, cutter_img_A);
                    break;
                case RIGHT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE * 2, cutter_img_D);
                    break;
                default:
                    break;
                }
                break;
            case ROTATOR:
                switch (building->direction)
                {
                case UP:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, rotator_img_W);
                    break;
                case DOWN:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, rotator_img_S);
                    break;
                case LEFT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, rotator_img_A);
                    break;
                case RIGHT:
                    painter.drawPixmap(building->pos.j * CELLSIZE, building->pos.i * CELLSIZE, CELLSIZE, CELLSIZE, rotator_img_D);
                    break;
                default:
                    break;
                }
                break;

            default:
                break;
            }
        }
    }
}



void PlayScene::draw_ui()
{
    shop.move(0, CELLSIZE * 15);
    shop.show();
    miner_button.move(BUTTON_SIZE * 3, CELLSIZE * 15);
    miner_button.show();
    belt_button.move(BUTTON_SIZE * 5, CELLSIZE * 15);
    belt_button.show();
    cutter_button.move(BUTTON_SIZE * 7, CELLSIZE * 15);
    cutter_button.show();
    trash_button.move(BUTTON_SIZE * 9, CELLSIZE * 15);
    trash_button.show();
    help_button.move(BUTTON_SIZE * 1, CELLSIZE * 15);
    help_button.show();
    //tunnel_entry_button.move(BUTTON_SIZE * 11, CELLSIZE * 15);
    //tunnel_entry_button.show();
    //tunnel_exit_button.move(BUTTON_SIZE * 12, CELLSIZE * 15);
    //tunnel_exit_button.show();
    rotator_button.move(BUTTON_SIZE * 11, CELLSIZE * 15);
    rotator_button.show();
}

void PlayScene::draw_hub_text()
{
    QPainter hubPainter(this);
    // Draw hub text
    hubPainter.scale(scaleFactor, scaleFactor);
    hubPainter.translate(scroll_offset);
    if (!hub->upgradehub)
    {
        // Unupgraded hub
        hubPainter.drawText(hub->pos.j * CELLSIZE + 35, hub->pos.i * CELLSIZE + 32, QString::number(hub->current_have) + QString("/") + QString::number(hub->need));
    }
    else
    {
        // Upgraded hub
        hubPainter.setFont(QFont("KaiTi", 35, QFont::Bold));
        hubPainter.drawText(hub->pos.j * CELLSIZE + 40, hub->pos.i * CELLSIZE + 45, QString::number(hub->current_have) + QString("/") + QString::number(hub->need));
    }
}

void PlayScene::draw_item()
{
    if (buildings.size() > 0)
    {
        QPainter itemPainter(this);
        itemPainter.scale(scaleFactor, scaleFactor);
        itemPainter.translate(scroll_offset);
        for (auto &building : buildings)
        {
            if (building->name >= BELT_A && building->name <= BELT_W_D)
            {
                if (building->state != EMPTY)
                {
                    switch (building->shape.name)
                    {
                    case CYCLE:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE, ITEM_SIZE, cycle_img);
                        break;
                    case RECT:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE, ITEM_SIZE, rect_img);
                        break;
                    case LEFT_CYCLE:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE / 2, ITEM_SIZE, left_cycle_img);
                        break;
                    case RIGHT_CYCLE:
                        itemPainter.drawPixmap(building->shape.x0, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE / 2, ITEM_SIZE, right_cycle_img);
                        break;
                    case UP_CYCLE:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE, ITEM_SIZE / 2, up_cycle_img);
                        break;
                    case DOWN_CYCLE:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE, ITEM_SIZE / 2, down_cycle_img);
                        break;
                    case LEFT_RECT:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE / 2, ITEM_SIZE, left_rect_img);
                        break;
                    case RIGHT_RECT:
                        itemPainter.drawPixmap(building->shape.x0, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE / 2, ITEM_SIZE, right_rect_img);
                        break;
                    case UP_RECT:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE, ITEM_SIZE / 2, up_rect_img);
                        break;
                    case DOWN_RECT:
                        itemPainter.drawPixmap(building->shape.x0 - ITEM_SIZE / 2, building->shape.y0 - ITEM_SIZE / 2, ITEM_SIZE, ITEM_SIZE / 2, down_rect_img);
                        break;
                    default:
                        break;
                    }
                }
            }
        }
    }
}



void PlayScene::draw_overlay(int x, int y)
{
    QPainter painter(this);
    painter.scale(scaleFactor, scaleFactor);
    painter.translate(scroll_offset);
    switch (current_building_name)
    {
    case MINER:
        switch (current_building_direction)
        {
        case UP:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, miner_img_blue_W);
            break;
        case DOWN:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, miner_img_blue_S);
            break;
        case LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, miner_img_blue_A);
            break;
        case RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, miner_img_blue_D);
            break;
        default:
            break;
        }
        break;
    case CUTTER:
        switch (current_building_direction)
        {
        case UP:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE * 2, CELLSIZE, cutter_img_blue_W);
            break;
        case DOWN:
            painter.drawPixmap(x / scaleFactor - 3 * CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE * 2, CELLSIZE, cutter_img_blue_S);
            break;
        case LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - 3 * CELLSIZE / 2, CELLSIZE, CELLSIZE * 2, cutter_img_blue_A);
            break;
        case RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE * 2, cutter_img_blue_D);
            break;
        default:
            break;
        }
        break;
    case TRASH:
        switch (current_building_direction)
        {
        case UP:
            // qDebug() << tr("try to draw trash");
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, trash_img_blue_W);
            break;
        case DOWN:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, trash_img_blue_S);
            break;
        case LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, trash_img_blue_A);
            break;
        case RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, trash_img_blue_D);
            break;
        default:
            break;
        }
        break;
    case ROTATOR:
        switch (current_building_direction)
        {
        case UP:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, rotator_img_blue_W);
            break;
        case DOWN:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, rotator_img_blue_S);
            break;
        case LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, rotator_img_blue_A);
            break;
        case RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, rotator_img_blue_D);
            break;
        default:
            break;
        }
        break;
    case BELT:
        switch (current_building_direction)
        {
        // Single direction
        case UP:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_W);
            break;
        case DOWN:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_S);
            break;
        case LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_A);
            break;
        case RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_D);
            break;
        case UP_RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_W_D);  // Up-Right
            break;
        case UP_LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_W_A);  // Up-Left
            break;
        case DOWN_RIGHT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_S_D);  // Down-Right
            break;
        case DOWN_LEFT:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_S_A);  // Down-Left
            break;
        case LEFT_UP:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_A_W);  // Left-Up
            break;
        case LEFT_DOWN:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_A_S);  // Left-Down
            break;
        case RIGHT_UP:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_D_W);  // Right-Up
            break;
        case RIGHT_DOWN:
            painter.drawPixmap(x / scaleFactor - CELLSIZE / 2, y / scaleFactor - CELLSIZE / 2, CELLSIZE, CELLSIZE, belt_img_blue_D_S);  // Right-Down
            break;

        default:
            break;
        }

    default:
        break;
    }
    if (current_building_name == BELT)
    {
        if (belt_grid_path.size() >= 2)
        {
            int which_belt = 0;
            for (int i = 0; i < belt_grid_path.size(); i++)
            {
                which_belt = WhichBeltImg(i);
                switch (which_belt)
                {
                case BELT_A:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_A);
                    break;
                case BELT_S_A:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_S_A);
                    break;
                case BELT_W_A:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_W_A);
                    break;
                case BELT_S:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_S);
                    break;
                case BELT_A_S:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_A_S);
                    break;
                case BELT_D_S:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_D_S);
                    break;
                case BELT_W:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_W);
                    break;
                case BELT_A_W:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_A_W);
                    break;
                case BELT_D_W:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_D_W);
                    break;
                case BELT_D:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_D);
                    break;
                case BELT_S_D:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_S_D);
                    break;
                case BELT_W_D:
                    painter.drawPixmap(belt_grid_path[i].j * CELLSIZE, belt_grid_path[i].i * CELLSIZE, CELLSIZE, CELLSIZE, belt_img_blue_W_D);
                    break;
                default:
                    break;
                }
            }
        }
    }
}



void PlayScene::paintEvent(QPaintEvent *)
{
    draw_cells();           // Draw the grid
    draw_map_resources();    // Draw the map resources (e.g., mining fields)
    draw_building();         // Draw the buildings
    draw_ui();               // Draw the UI elements (buttons, etc.)
    draw_hub_text();         // Draw text related to the hub
    draw_item();             // Draw items on the conveyor belts
    //cur_x = ((cur_x / CELLSIZE) - related_j_offset + 0.5) * CELLSIZE;
    //cur_y = ((cur_y / CELLSIZE) - related_i_offset + 0.5) * CELLSIZE;
    draw_overlay(cur_x, cur_y);   // Draw the overlay of current building placement
    draw_current_shape();         // Draw the current shape of items
}

void PlayScene::HandleLeftButtonPress(QMouseEvent *e, int grid_i, int grid_j)
{
    GridVec cur;
    cur.j = grid_j;
    cur.i = grid_i;
    Building *current_building;

    if (ready_to_place_belt)
    {
        PushBackNewBeltGridPoint(grid_j, grid_i);  // Add new grid point for placing a belt
        ready_to_place_belt = false;
        is_placing_belt = true;
        return;
    }

    if (current_building_name)  // If a building is selected
    {
        switch (current_building_name)
        {
        case MINER:
            current_building = new Miner(cur, current_building_name, current_building_direction);
            break;
        case CUTTER:
            current_building = new Cutter(cur, current_building_name, current_building_direction);
            break;
        case TRASH:
            current_building = new Trash(cur, current_building_name, current_building_direction);
            break;
        case TUNNEL_ENTRY:
            current_building = new TunnelEntry(cur, current_building_name, current_building_direction);
            printf("tunnelentry:\n", current_building->name);
            break;
        case TUNNEL_EXIT:
            current_building = new TunnelExit(cur, current_building_name, current_building_direction);
            break;
        case ROTATOR:
            current_building = new Rotator(cur, current_building_name, current_building_direction);
            break;
        default:
            return;
        }

        // Apply building upgrades if any
        switch (upgrade)
        {
        case MINER:
            current_building->FirstRequire_ms = MINER_SPEED_2;
            break;
        case CUTTER:
            current_building->FirstRequire_ms = CUTTER_SPEED_2;
            break;
        case ROTATOR:
            current_building->FirstRequire_ms = ROTATOR_SPEED_2;
            break;
        default:
            break;
        }

        // Place the building
        if (current_building->CanPlace(cur, current_building_direction, map))
        {
            map.SetBuilding(cur, current_building, current_building_direction, current_building_name);
            for (int i = 0; i < buildings.size(); i++)
            {
                for (auto poscur : current_building->BuildingAllPos())
                {
                    if (buildings[i]->pos == poscur)
                    {
                        buildings.remove(i);  // Remove any existing building at the new position
                    }
                }
            }
            buildings.push_back(current_building);  // Add the new building to the list
            current_building_name = NONE;
            update();  // Refresh the scene
        }
    }

    if (!ready_to_place_belt && !current_building_name)  // If no building is selected, handle dragging
    {
        dragging = true;
        start_pos = e->pos();  // Capture the starting position for dragging
    }
}

void PlayScene::HandleRightButtonPress(int grid_i, int grid_j)
{
    GridVec cur;
    cur.j = grid_j;
    cur.i = grid_i;

    if (0 <= grid_i && grid_i < HEIGHT && 0 <= grid_j && grid_j < WIDTH)
    {
        RemoveBuilding(cur);  // Remove the building at the specified grid position
        update();  // Refresh the scene
    }
}

void PlayScene::mousePressEvent(QMouseEvent *e)
{
    int x = (e->pos().x() - scroll_offset.x() * scaleFactor);
    int y = (e->pos().y() - scroll_offset.y() * scaleFactor);
    int grid_j = x / (CELLSIZE * scaleFactor);  // Calculate the grid column
    int grid_i = y / (CELLSIZE * scaleFactor);  // Calculate the grid row

    //printf("ex = %d; ey =%d ;x = %d y = %d j = %d , i = %d s= %lf\n", e->pos().x(), e->pos().y(), x, y, grid_j, grid_i, scaleFactor);
    fflush(stdout);

    if (e->button() == Qt::LeftButton)
    {
        HandleLeftButtonPress(e, grid_i, grid_j);  // Handle left mouse button press
    }
    else if (e->button() == Qt::RightButton)
    {
        right_button_pressed = true;
        HandleRightButtonPress(grid_i, grid_j);  // Handle right mouse button press
    }
}

void PlayScene::RemoveBuilding(GridVec pos)
{
    if (map.BuildingsMap[pos.i][pos.j] != nullptr)
    {
        for (auto curr : map.BuildingsMap[pos.i][pos.j]->BuildingAllPos())
        {
            for (int i = 0; i < buildings.size(); i++)
            {
                if (buildings[i]->pos == curr)
                {
                    if (buildings[i]->name != HUB)
                    {
                        buildings.remove(i);  // Remove the building from the list, except the hub
                        break;
                    }
                }
            }
        }
        map.RemoveBuilding(pos);  // Remove the building from the map
    }
}



void PlayScene::keyPressEvent(QKeyEvent *e)
{
    switch (e->key()) {
    case Qt::Key_1:
        current_building_direction = UP_RIGHT;
        break;
    case Qt::Key_2:
        current_building_direction = RIGHT_UP;
        break;
    case Qt::Key_3:
        current_building_direction = UP_LEFT;
        break;
    case Qt::Key_4:
        current_building_direction = LEFT_UP;
        break;
    case Qt::Key_5:
        current_building_direction = DOWN_RIGHT;
        break;
    case Qt::Key_6:
        current_building_direction = RIGHT_DOWN;
        break;
    case Qt::Key_7:
        current_building_direction = DOWN_LEFT;
        break;
    case Qt::Key_8:
        current_building_direction = LEFT_DOWN;
        break;
    case Qt::Key_A:
        current_building_direction = LEFT;
        break;
    case Qt::Key_D:
        current_building_direction = RIGHT;
        break;
    case Qt::Key_W:
        current_building_direction = UP;
        break;
    case Qt::Key_S:
        current_building_direction = DOWN;
        break;
    default:
        // If other undefined keys are pressed, choose to ignore or handle them
        break;
    }
}

void PlayScene::keyReleaseEvent(QKeyEvent *e)
{
    // Remove the key
    pressedKeys.remove(e->key());
    keySequence.removeOne(e->key());  // Remove the key from the key sequence
}

void PlayScene::PushBackNewBeltGridPoint(int grid_j, int grid_i)
{
    // Push new grid points into the list when dragging to place conveyor belts
    belt_grid_Point.i = grid_i;
    belt_grid_Point.j = grid_j;
    if (!belt_grid_path.contains(belt_grid_Point))  // If the point is not already in the list
    {
        belt_grid_path.append(belt_grid_Point);  // Add it to the path
    }
}

void PlayScene::HandleBuildingPlacement(QMouseEvent *e, int grid_i, int grid_j)
{
    int adjusted_x = (e->pos().x() - scroll_offset.x() * scaleFactor);
    int adjusted_y = (e->pos().y() - scroll_offset.y() * scaleFactor);

    // If a building is selected, update its position and refresh the display
    if (current_building_name)
    {
        cur_x = adjusted_x;
        cur_y = adjusted_y;
        update();

        // If placing a belt, record the grid point
        if (is_placing_belt)
        {
            PushBackNewBeltGridPoint(grid_j, grid_i);  // Add new point for the belt path
        }
    }
}

void PlayScene::HandleDragging(QMouseEvent *e)
{
    QPoint delta = e->pos() - start_pos;
    scroll_offset += delta;

    // Ensure the scroll_offset stays within the valid range (prevents scrolling too far)
    if (scroll_offset.x() > 0)
    {
        scroll_offset.setX(0);
    }
    if (scroll_offset.y() > 0)
    {
        scroll_offset.setY(0);
    }

    int maxX = (WIDGET_WIDTH - WIDTH * CELLSIZE * scaleFactor) / scaleFactor;
    int maxY = (WIDGET_HEIGHT - HEIGHT * CELLSIZE * scaleFactor) / scaleFactor;

    // Keep the scroll offset within the boundaries of the grid
    scroll_offset.setX((scroll_offset.x() < maxX) ? maxX : scroll_offset.x());
    scroll_offset.setY((scroll_offset.y() < maxY) ? maxY : scroll_offset.y());

    start_pos = e->pos();  // Update the starting position for the next movement
    *Qpointmapx = scroll_offset.x();
    *Qpointmapy = scroll_offset.y();
    update();  // Refresh the display
}

void PlayScene::HandleRightButtonDrag(int grid_i, int grid_j)
{
    GridVec cur;
    cur.j = grid_j;
    cur.i = grid_i;

    // If the grid coordinates are valid, remove the building at this location
    if (0 <= grid_i && grid_i < HEIGHT && 0 <= grid_j && grid_j < WIDTH)
    {
        RemoveBuilding(cur);
        update();  // Refresh the display
    }
}

void PlayScene::mouseMoveEvent(QMouseEvent *e)
{
    int adjusted_x = (e->pos().x() - scroll_offset.x() * scaleFactor);
    int adjusted_y = (e->pos().y() - scroll_offset.y() * scaleFactor);
    int grid_j = adjusted_x / (CELLSIZE * scaleFactor);  // Calculate the grid column (j)
    int grid_i = adjusted_y / (CELLSIZE * scaleFactor);  // Calculate the grid row (i)

    // Handle building placement when a building is selected
    if (current_building_name)
    {
        HandleBuildingPlacement(e, grid_i, grid_j);  // Update the position of the building
    }

    // Handle dragging (scrolling) action
    if (dragging)
    {
        HandleDragging(e);  // Update the scroll position based on mouse movement
    }

    // Handle right-button drag action to remove buildings
    if (right_button_pressed)
    {
        HandleRightButtonDrag(grid_i, grid_j);  // Remove buildings while dragging with the right button
    }
}


void PlayScene::ClearBeltGridPath()
{
    while (!belt_grid_path.empty())
    {
        belt_grid_path.removeLast();  // Remove all points from the belt grid path
    }
}

int PlayScene::WhichBeltImg(int belt_grid_path_index)
{
    if (belt_grid_path_index == 0)
    {
        if (belt_grid_path[1] - belt_grid_path[0] == RIGHT)
        {
            return BELT_D;
        }
        else if (belt_grid_path[1] - belt_grid_path[0] == LEFT)
        {
            return BELT_A;
        }
        else if (belt_grid_path[1] - belt_grid_path[0] == UP)
        {
            return BELT_W;
        }
        else if (belt_grid_path[1] - belt_grid_path[0] == DOWN)
        {
            return BELT_S;
        }
    }
    int end_index = belt_grid_path.size() - 1;
    if (belt_grid_path_index == end_index)
    {
        if (belt_grid_path[end_index] - belt_grid_path[end_index - 1] == RIGHT)
        {
            return BELT_D;
        }
        else if (belt_grid_path[end_index] - belt_grid_path[end_index - 1] == LEFT)
        {
            return BELT_A;
        }
        else if (belt_grid_path[end_index] - belt_grid_path[end_index - 1] == UP)
        {
            return BELT_W;
        }
        else if (belt_grid_path[end_index] - belt_grid_path[end_index - 1] == DOWN)
        {
            return BELT_S;
        }
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == RIGHT) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == RIGHT))
    {
        return BELT_D;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == LEFT) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == LEFT))
    {
        return BELT_A;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == UP) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == UP))
    {
        return BELT_W;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == DOWN) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == DOWN))
    {
        return BELT_S;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == RIGHT) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == UP))
    {
        return BELT_W_D;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == LEFT) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == UP))
    {
        return BELT_W_A;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == RIGHT) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == DOWN))
    {
        return BELT_S_D;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == LEFT) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == DOWN))
    {
        return BELT_S_A;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == UP) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == LEFT))
    {
        return BELT_A_W;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == DOWN) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == LEFT))
    {
        return BELT_A_S;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == UP) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == RIGHT))
    {
        return BELT_D_W;
    }
    else if ((belt_grid_path[belt_grid_path_index + 1] - belt_grid_path[belt_grid_path_index] == DOWN) && (belt_grid_path[belt_grid_path_index] - belt_grid_path[belt_grid_path_index - 1] == RIGHT))
    {
        return BELT_D_S;
    }
    else {
        return BELT_D;
    }
    return 0;
}

void PlayScene::HandleSingleBeltPlacement()
{
    int which_belt = 0;
    int belt_direction = current_building_direction;

    // Determine the type of conveyor belt based on the current building direction
    switch (current_building_direction)
    {
    // Single directions
    case UP:
        which_belt = BELT_W;  // Conveyor belt facing up
        break;
    case DOWN:
        which_belt = BELT_S;  // Conveyor belt facing down
        break;
    case LEFT:
        which_belt = BELT_A;  // Conveyor belt facing left
        break;
    case RIGHT:
        which_belt = BELT_D;  // Conveyor belt facing right
        break;

        // Eight new direction combinations
    case UP_RIGHT:
        which_belt = BELT_W_D;  // Conveyor belt up-right
        belt_direction = RIGHT;  // Adjust to right direction
        break;
    case UP_LEFT:
        which_belt = BELT_W_A;  // Conveyor belt up-left
        belt_direction = LEFT;  // Adjust to left direction
        break;
    case DOWN_RIGHT:
        which_belt = BELT_S_D;  // Conveyor belt down-right
        belt_direction = RIGHT;  // Adjust to right direction
        break;
    case DOWN_LEFT:
        which_belt = BELT_S_A;  // Conveyor belt down-left
        belt_direction = LEFT;  // Adjust to left direction
        break;
    case LEFT_UP:
        which_belt = BELT_A_W;  // Conveyor belt left-up
        belt_direction = UP;  // Adjust to up direction
        break;
    case LEFT_DOWN:
        which_belt = BELT_A_S;  // Conveyor belt left-down
        belt_direction = DOWN;  // Adjust to down direction
        break;
    case RIGHT_UP:
        which_belt = BELT_D_W;  // Conveyor belt right-up
        belt_direction = UP;  // Adjust to up direction
        break;
    case RIGHT_DOWN:
        which_belt = BELT_D_S;  // Conveyor belt right-down
        belt_direction = DOWN;  // Adjust to down direction
        break;

    default:
        break;
    }

    // Create a new conveyor belt and check if it can be placed
    Building *current_building = new Belt(belt_grid_path[0], which_belt, belt_direction);
    if (current_building->CanPlace(belt_grid_path[0], belt_direction, map))
    {
        map.SetBuilding(belt_grid_path[0], current_building, belt_direction, which_belt);
        buildings.push_back(current_building);
    }
    current_building_direction = UP;  // Reset the building direction to UP
}




void PlayScene::HandleBeltGridPlacement()
{
    for (int i = 0; i < belt_grid_path.size(); i++)
    {
        int which_belt = WhichBeltImg(i);
        int belt_direction = 0;

        // Determine belt direction based on the belt type
        switch (which_belt)
        {
        case BELT_A:
        case BELT_S_A:
        case BELT_W_A:
            belt_direction = LEFT;
            break;
        case BELT_S:
        case BELT_A_S:
        case BELT_D_S:
            belt_direction = DOWN;
            break;
        case BELT_W:
        case BELT_A_W:
        case BELT_D_W:
            belt_direction = UP;
            break;
        case BELT_D:
        case BELT_S_D:
        case BELT_W_D:
            belt_direction = RIGHT;
            break;
        default:
            break;
        }

        // Create a new belt and check if it can be placed
        Building *current_building = new Belt(belt_grid_path[i], which_belt, belt_direction);
        if (current_building->CanPlace(belt_grid_path[i], belt_direction, map))
        {
            if (upgrade == BELT)
            {
                current_building->FirstRequire_ms = BELT_SPEED_2;
            }
            map.SetBuilding(belt_grid_path[i], current_building, belt_direction, which_belt);
            buildings.push_back(current_building);
        }
    }
}

void PlayScene::HandleBeltPlacement()
{
    if (belt_grid_path.size() >= 2)
    {
        HandleBeltGridPlacement();
    }
    else
    {
        HandleSingleBeltPlacement();
    }

    // Clear the belt grid path and update the scene
    ClearBeltGridPath();
    update();
    current_building_name = NONE;
}

void PlayScene::HandleRightButtonRelease(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
    {
        right_button_pressed = false;
    }
}

void PlayScene::mouseReleaseEvent(QMouseEvent *e)
{
    dragging = false;

    if (is_placing_belt)
    {
        is_placing_belt = false;
        HandleBeltPlacement();  // Handle the belt placement logic
    }

    HandleRightButtonRelease(e);  // Handle right button release
}

void PlayScene::FactoryRunning()
{
    timer.start();
    // Listen to the timer
    connect(&timer, &QTimer::timeout, this, [=]()
            {
                if (elapsedTimer.elapsed() >= 10000) {  // Check if one second has passed
                    updateHubStats();  // Update statistics
                    elapsedTimer.restart();  // Reset the timer and start counting for the next second
                }
                // Redraw the images
                update();
                if (buildings.size())
                {
                    for (auto &building : buildings)
                    {
                        if (building->name != HUB && building->name != TRASH && building->name != TUNNEL_ENTRY && building->name != TUNNEL_EXIT)
                        {
                            building->UpdateTickableState(map);
                        }
                    }
                } });
}

void PlayScene::updateHubStats()
{
    // Update the count of objects received by the hub in the last second
    hub->updateReceivedObjectsCount();

    // Reset the hub's receive counter
    hub->resetReceiveCounter();

    // Request a window refresh to update the UI display (trigger paintEvent)
    update();
}

void PlayScene::UpgradeHub()
{
    hub->pos.i = HEIGHT / 2 - 2;
    hub->pos.j = WIDTH / 2 - 2;
    GridVec hubvec(WIDTH / 2 - 2, HEIGHT / 2 - 2);
    map.SetBuilding(hubvec, hub, UP, HUB);
}

void PlayScene::DoSave()
{
    QSettings setting("Parameter.ini", QSettings::IniFormat);
    setting.remove("building");
    setting.beginWriteArray("building");
    for (int i = 0; i < buildings.size(); i++)
    {
        setting.setArrayIndex(i);
        setting.setValue("j", buildings[i]->pos.j);
        setting.setValue("i", buildings[i]->pos.i);
        setting.setValue("direction", buildings[i]->direction);
        setting.setValue("name", buildings[i]->name);
    }
    setting.endArray();
    setting.beginGroup("savecondition");
    setting.setValue("saved", saved);
    setting.setValue("increased_mine", increased_mine);
    setting.setValue("cleared_barriers", clear_barriers);
    setting.setValue("increased_item_value", hub->increase_item_value);
    setting.setValue("upgradedhub", hub->upgradehub);
    setting.setValue("money", hub->money);
    setting.setValue("round", round);
    setting.setValue("upgradewhich", upgrade);
    setting.setValue("current_have", hub->current_have);
    setting.endGroup();
    setting.sync();
}

void PlayScene::LoadSave()
{
    QSettings setting("Parameter.ini", QSettings::IniFormat);
    setting.beginGroup("savecondition");
    saved = setting.value("saved").toBool();
    increased_mine = setting.value("increased_mine").toBool();
    clear_barriers = setting.value("cleared_barriers").toBool();
    bool tempincrease = setting.value("increased_item_value").toBool();
    int tempmoney = setting.value("money").toInt();
    bool tempupgradehub = setting.value("upgradedhub").toBool();
    int temphave = setting.value("current_have").toInt();
    upgrade = setting.value("upgradewhich").toInt();
    if (setting.value("round").toInt() > 1)
    {
        round = setting.value("round").toInt();
    }
    setting.endGroup();
    if (increased_mine)
    {
        map.SecondMap();
    }
    if (clear_barriers)
    {
        map.ClearBarriers();
    }
    if (saved)
    {
        int size = setting.beginReadArray("building");
        for (int i = 0; i < size; i++)
        {
            setting.setArrayIndex(i);
            GridVec cur = GridVec(setting.value("j").toInt(), setting.value("i").toInt());
            int name = setting.value("name").toInt();
            int direction = setting.value("direction").toInt();
            if (name == HUB)
            {
                hub = new Hub(this);
                hub->name = name;
                hub->direction = direction;
                hub->pos = cur;
                Building *smallhub = hub;
                map.SetBuilding(cur, smallhub, direction, name);
                buildings.push_back(smallhub);
                continue;
            }
            Building *cur_building;
            switch (name)
            {
            case MINER:
                cur_building = new Miner(cur, name, direction);
                break;
            case CUTTER:
                cur_building = new Cutter(cur, name, direction);
                break;
            case TRASH:
                cur_building = new Trash(cur, name, direction);
                break;
            case TUNNEL_ENTRY:
                cur_building = new TunnelEntry(cur, name, direction);
                break;
            case TUNNEL_EXIT:
                cur_building = new TunnelExit(cur, name, direction);
                break;
            case ROTATOR:
                cur_building = new Rotator(cur, name, direction);
                break;
            case BELT_A:
            case BELT_A_S:
            case BELT_A_W:
            case BELT_D:
            case BELT_D_S:
            case BELT_D_W:
            case BELT_S:
            case BELT_S_A:
            case BELT_S_D:
            case BELT_W:
            case BELT_W_A:
            case BELT_W_D:
                cur_building = new Belt(cur, name, direction);
                break;
            }
            map.SetBuilding(cur, cur_building, direction, name);
            buildings.push_back(cur_building);
        }
    }
    if (!saved)
    {
        hub = new Hub(this);
        hub->name = HUB;
        hub->pos.i = 10;
        hub->pos.j = 15;
        hub->direction = UP;
        GridVec hubvec(10, 15);
        Building *smallhub = hub;
        map.SetBuilding(hubvec, smallhub, hub->direction, hub->name);
        buildings.push_back(smallhub);
    }
    hub->increase_item_value = tempincrease;
    hub->money = tempmoney;
    hub->upgradehub = tempupgradehub;
    hub->current_have = temphave;
    if (tempupgradehub)
    {
        UpgradeHub();
    }
    switch (upgrade)
    {
    case NONE:
        break;
    case MINER:
        for (int i = 0; i < buildings.size(); i++)
        {
            switch (buildings[i]->name)
            {
            case MINER:
                buildings[i]->FirstRequire_ms = 1800;
                break;
            case CUTTER:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case ROTATOR:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case BELT_A:
            case BELT_A_S:
            case BELT_A_W:
            case BELT_D:
            case BELT_D_S:
            case BELT_D_W:
            case BELT_S:
            case BELT_S_A:
            case BELT_S_D:
            case BELT_W:
            case BELT_W_A:
            case BELT_W_D:
                buildings[i]->FirstRequire_ms = 1500;
                break;
            default:
                break;
            }
        }
        break;
    case BELT:
        for (int i = 0; i < buildings.size(); i++)
        {
            switch (buildings[i]->name)
            {
            case MINER:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case CUTTER:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case ROTATOR:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case BELT_A:
            case BELT_A_S:
            case BELT_A_W:
            case BELT_D:
            case BELT_D_S:
            case BELT_D_W:
            case BELT_S:
            case BELT_S_A:
            case BELT_S_D:
            case BELT_W:
            case BELT_W_A:
            case BELT_W_D:
                buildings[i]->FirstRequire_ms = 720;
                break;
            default:
                break;
            }
        }
        break;
    case CUTTER:
        for (int i = 0; i < buildings.size(); i++)
        {
            switch (buildings[i]->name)
            {
            case MINER:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case CUTTER:
                buildings[i]->FirstRequire_ms = 1800;
                break;
            case ROTATOR:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case BELT_A:
            case BELT_A_S:
            case BELT_A_W:
            case BELT_D:
            case BELT_D_S:
            case BELT_D_W:
            case BELT_S:
            case BELT_S_A:
            case BELT_S_D:
            case BELT_W:
            case BELT_W_A:
            case BELT_W_D:
                buildings[i]->FirstRequire_ms = 1500;
                break;
            default:
                break;
            }
        }
        break;
    case ROTATOR:
        for (int i = 0; i < buildings.size(); i++)
        {
            switch (buildings[i]->name)
            {
            case MINER:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case CUTTER:
                buildings[i]->FirstRequire_ms = 1800;
                break;
            case ROTATOR:
                buildings[i]->FirstRequire_ms = 2700;
                break;
            case BELT_A:
            case BELT_A_S:
            case BELT_A_W:
            case BELT_D:
            case BELT_D_S:
            case BELT_D_W:
            case BELT_S:
            case BELT_S_A:
            case BELT_S_D:
            case BELT_W:
            case BELT_W_A:
            case BELT_W_D:
                buildings[i]->FirstRequire_ms = 1500;
                break;
            default:
                break;
            }
        }
        break;
    default:
        break;
    }
    switch (round)
    {
    case 1:
        *hub->need_shape_name = CYCLE;
        hub->need = NEED_CYCLE;
        aim_img.load(AIM1_PATH);
        break;
    case 2:
        *hub->need_shape_name = RECT;
        hub->need = NEED_RECT;
        aim_img.load(AIM2_PATH);
        break;
    case 3:
        *hub->need_shape_name = LEFT_CYCLE;
        hub->need = NEED_LEFT_CYCLE;
        aim_img.load(AIM3_PATH);
        break;
    case 4:
        *hub->need_shape_name = RIGHT_CYCLE;
        hub->need = NEED_RIGHT_CYCLE;
        aim_img.load(AIM4_PATH);
        break;
    default:
        break;
    }
}



void PlayScene::closeEvent(QCloseEvent *)
{
    saved = true;
    DoSave();
}
void PlayScene::music()
{
    player = new QMediaPlayer(this);
    audioOutput = new QAudioOutput(this);
    player->setAudioOutput(audioOutput);
    player->setSource(QUrl("qrc:/res/theme-full.mp3"));
    player->play();
}
void PlayScene::setScaleFactor(double fac){
    scaleFactor = fac;
    if (scroll_offset.x()>0 ){
        scroll_offset.setX(0);
    }
    if(scroll_offset.y()>0){
        scroll_offset.setY(0);
    }
    int maxX = (WIDGET_WIDTH-WIDTH*CELLSIZE*scaleFactor)/scaleFactor;
    printf("maxX=%d",maxX);
    int maxY = (WIDGET_HEIGHT-HEIGHT*CELLSIZE*scaleFactor)/scaleFactor;
    fflush(stdout);
    scroll_offset.setX((scroll_offset.x() < maxX) ? maxX:scroll_offset.x());
    scroll_offset.setY((scroll_offset.y() < maxY) ? maxY:scroll_offset.y());
    *Qpointmapx = scroll_offset.x();
    *Qpointmapy = scroll_offset.y();
    update();
}
void PlayScene::wheelEvent(QWheelEvent *event) {
    // Get the current scroll value from the mouse wheel
    int delta = event->angleDelta().y();
    double scaleFactorStep = 0.1;

    // Get the current center position of the viewport
    int viewCenterX = (WIDGET_WIDTH / 2) / scaleFactor - scroll_offset.x();
    int viewCenterY = (WIDGET_HEIGHT / 2) / scaleFactor - scroll_offset.y();

    // Increase or decrease the scale factor based on the scroll direction
    if (delta > 0) {
        setScaleFactor(scaleFactor + scaleFactorStep);
    } else {
        setScaleFactor(scaleFactor - scaleFactorStep);
    }

    // Restrict the scale factor to a reasonable range
    if (scaleFactor < 0.6) {
        setScaleFactor(0.6);
    } else if (scaleFactor > 2.0) {
        setScaleFactor(2.0);
    }
    *scaleFcator_map = scaleFactor;

    // Adjust the scroll offset to ensure the center of the viewport stays the same
    scroll_offset.setX((WIDGET_WIDTH / 2) / scaleFactor - viewCenterX);
    scroll_offset.setY((WIDGET_HEIGHT / 2) / scaleFactor - viewCenterY);

    // Refresh the view
    update();
}

void PlayScene::initializeView() {


    setScaleFactor(1.0);
    // Calculate the width and height of the map (number of tiles * size of each tile)
    int mapWidth = 32 * 50;  // Total map width
    int mapHeight = 20 * 50; // Total map height

    // Calculate the center of the map (midpoint of the map)
    int mapCenterX = mapWidth / 2;
    int mapCenterY = mapHeight / 2;

    // Calculate the center of the window
    int windowCenterX = WIDGET_WIDTH / 2;
    int windowCenterY = WIDGET_HEIGHT / 2;

    // Set scroll_offset to align the window center with the map center
    scroll_offset.setX(windowCenterX - (mapCenterX / scaleFactor));
    scroll_offset.setY(windowCenterY - (mapCenterY / scaleFactor));

    // Refresh the view
    update();


}

void PlayScene::outputCurrentShape() {
    switch (current_received_shape) {
    case CYCLE:
        std::cout << "Current received shape: CYCLE" << std::endl;
        break;
    case RECT:
        std::cout << "Current received shape: RECT" << std::endl;
        break;
    case LEFT_CYCLE:
        std::cout << "Current received shape: LEFT CYCLE" << std::endl;
        break;
    case RIGHT_CYCLE:
        std::cout << "Current received shape: RIGHT CYCLE" << std::endl;
        break;
    case NONE:
        std::cout << "Current received shape: NONE" << std::endl;
        break;
    default:
        std::cout << "Current received shape: UNKNOWN" << std::endl;
        break;
    }
}
void PlayScene::draw_current_shape() {
    QPainter shapePainter(this);

    // Fixed position in the top-right corner, set the drawing position (offset from the top-right)
    int x = this->width() - 250;  // 200 pixels from the right edge of the window
    int y = 50;

    // Draw the number of objects received in the last 10 seconds
    shapePainter.setFont(QFont("Arial", 20));  // Set font size
    shapePainter.drawText(x, y, "Objects in 10s: " + QString::number(*(hub->received_objects_last_10_second)));

    // Display the current received shape below the previous text
    y += 30;  // Adjust y-coordinate to prevent text overlap
    switch (current_received_shape) {
    case CYCLE:
        shapePainter.drawText(x, y, "CYCLE");
        break;
    case RECT:
        shapePainter.drawText(x, y, "RECT");
        break;
    case LEFT_CYCLE:
        shapePainter.drawText(x, y, "LEFT CYCLE");
        break;
    case RIGHT_CYCLE:
        shapePainter.drawText(x, y, "RIGHT CYCLE");
        break;
    default:
        shapePainter.drawText(x, y, "NONE");
        break;
    }
}

void PlayScene::checkResetReceivedShape() {
    // If more than 10 seconds have passed without receiving any objects
    if (hub && hub->shape_update_timer.elapsed() >= 10000) {
        if (current_received_shape != NONE) {
            current_received_shape = NONE;
            hub->resetReceiveCounter();  // Reset the receive counter
        }
    }
}

void PlayScene::addTopLeftButton() {
    // Create a button and set its parent window to PlayScene
    QPushButton *topLeftButton = new QPushButton(this);

    // Set the button's size and position (top left corner)
    topLeftButton->setGeometry(1030, 730, 100, 100);  // Adjust the button's size and position

    // Remove the button's border to make it appear like a plain image area
    topLeftButton->setStyleSheet("QPushButton { border: none; }");

    // Set the button's icon (replace with your image path)
    QPixmap pixmap("://res/locate.png");  // Image path
    QIcon ButtonIcon(pixmap);
    topLeftButton->setIcon(ButtonIcon);
    topLeftButton->setIconSize(pixmap.rect().size());  // Set the icon size to match the button

    // Connect the button's click signal to the initializeView method of PlayScene
    connect(topLeftButton, &QPushButton::clicked, this, &PlayScene::initializeView);
}




