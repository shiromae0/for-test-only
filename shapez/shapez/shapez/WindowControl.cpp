#include <windows.h>
#include "WindowControl.h"
#include <QIcon>
#include <QPainter>
#include <QPen>
#include "config.h"
#include "iostream"

void WindowControl::CreateMapFile(){

    minspeed = new int;

    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,                    // Default security attributes
        PAGE_READWRITE,          // Read/write permissions
        0,                       // High-order file size
        sizeof(int),             // Low-order file size (size of int)
        L"minnspeed"             // Use a wide-character string as the shared memory name
        );
    LPVOID lpBase = MapViewOfFile(
        hMapFile,                // Handle to the file mapping object
        FILE_MAP_ALL_ACCESS,     // Read/write permissions
        0,                       // High-order offset
        0,                       // Low-order offset
        sizeof(int)              // Number of bytes to map
        );

    minspeed = static_cast<int*>(lpBase); // Cast the mapped memory to an int pointer
    *minspeed = 1;

}

WindowControl::WindowControl(QWidget *parent)
{
    // Check every 1 second if any level has been passed
    CreateMapFile();
    timer.setInterval(10);
    start = new StartScene;
    play = new PlayScene;
    round = new RoundScene;
    choose = new ChooseScene;
    shop = new ShopScene;
    win = new WinScene;
    help = new HelpScene;
    start->show();
    // Play music
    play->music();
    // Set up connections between the scenes
    connections();

}
WindowControl::~WindowControl()
{
    delete play;
    delete round;
    delete choose;
    delete shop;
}

/*
void WindowControl::CheckAllUpdated() {
    if (miner_updated == true && belt_updated == true && cutter_updated == true){
        *minspeed = 2;
    }
    else{
        *minspeed = 1;
    }
}
*/

void WindowControl::CheckPassCertainRound() {
    timer.start();
    connect(&timer, &QTimer::timeout, this, [=]() {
        if (play->round == 5) {
            *minspeed = 2;
            // New condition: receive 6 cycles within 10 seconds
            if (*(play->hub->received_objects_last_10_second) >= 6 && play->current_received_shape == CYCLE) {
                round->level++;
                play->round++;
                play->close();
                choose->show();
                play->hub->UpdateNeed();
            }
        } else {
            // Keep the same condition for other levels
            if (play->hub->current_have >= play->hub->need) {
                round->level++;
                play->round++;
                if (round->level >= 6) {
                    round->level = 6;
                    play->round = 6;
                    play->close();
                    choose->close();
                    round->close();
                    win->show();
                    QSettings setting("Parameter.ini", QSettings::IniFormat);
                    setting.clear();
                } else {
                    play->close();
                    choose->show();
                    play->hub->UpdateNeed();
                }
            }
        }
    });
}

void WindowControl::ChooseUpgradeMiner()
{
    // Update the speed of miner, belt, and cutter
    play->upgrade = MINER;
    for (int i = 0; i < play->buildings.size(); i++)
    {
        switch (play->buildings[i]->name)
        {
        case MINER:
            play->buildings[i]->FirstRequire_ms = MINER_SPEED_2;
            break;
        default:
            break;
        }
    }
}

void WindowControl::ChooseUpgradeBelt()
{
    // Update the speed of miner, belt, and cutter
    play->upgrade = BELT;
    for (int i = 0; i < play->buildings.size(); i++)
    {
        switch (play->buildings[i]->name)
        {
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
            play->buildings[i]->FirstRequire_ms = BELT_SPEED_2;
            break;
        default:
            break;
        }
    }
}

void WindowControl::ChooseUpgradeCutter()
{
    // Update the speed of miner, belt, and cutter
    play->upgrade = CUTTER;
    for (int i = 0; i < play->buildings.size(); i++)
    {
        switch (play->buildings[i]->name)
        {
        case CUTTER:
            play->buildings[i]->FirstRequire_ms = CUTTER_SPEED_2;
            break;
        default:
            break;
        }
    }
}

void WindowControl::connections()
{
    // start界面选择新游戏
    connect(&start->new_game, &QPushButton::clicked, this, [=]()
            {
        start->close();
        QSettings setting("Parameter.ini", QSettings::IniFormat);
        setting.clear();
        play->saved = false;
        play->LoadSave();
        round->level = play->round;
        // 开启计时器，每1s检查一下有没有通过某一关
        CheckPassCertainRound();
        round->show(); });
    // start界面选择继续游戏
    connect(&start->continue_game, &QPushButton::clicked, this, [=]()
            {
        start->close();
        play->LoadSave();
        round->level = play->round;
        // 开启计时器，每1s检查一下有没有通过某一关
        CheckPassCertainRound();
        if(round->level < 5)
        {
            round->show();
        } });
    // round界面确认键按下后，进入游戏
    connect(&round->confirm, &QPushButton::clicked, this, [=]()
            {
        round->close();
        play->show(); });
    // play界面点击help，进入帮助
    connect(&play->help_button, &QPushButton::clicked, this, [=]()
            {
        play->close();
        help->show(); });
    // help界面点击return，进入play
    connect(&help->return_button, &QPushButton::clicked, this, [=]()
            {
        help->close();
        play->show(); });
    // play界面点击shop，进入商店
    connect(&play->shop, &QPushButton::clicked, this, [=]()
            {
        play->close();
        shop->show(); });
    // shop界面点击bigger_hub
    connect(&shop->bigger_hub, &QPushButton::clicked, this, [=]()
            {
                //关闭shop界面
                shop->close();
                //如果金币足够
                if(play->hub->money >= BIGGER_HUB)
                {
                    if(!play->hub->upgradehub)
                    {
                        //更新hub
                        play->hub->upgradehub = true;
                        play->UpgradeHub();
                        //消耗金币
                        play->hub->money -= BIGGER_HUB;
                        //把新hub占据的位置的建筑清空
                        for(auto pos : play->hub->BuildingAllPos())
                        {
                            qDebug() << pos.j << pos.i;
                            play->RemoveBuilding(pos);
                        }
                    }
                }
                //进入play界面
                play->show(); });
    // shop界面点击clear_barriers
    connect(&shop->clear_barriers, &QPushButton::clicked, this, [=]()
            {
                //关闭shop界面
                shop->close();
                //如果金币足够
                if(play->hub->money >= CLEAR_BARRIERS)
                {
                    if(! play->clear_barriers)
                    {
                        //清空所有障碍物
                        play->clear_barriers = true;
                        play->map.ClearBarriers();
                        //消耗金币
                        play->hub->money -= CLEAR_BARRIERS;
                    }
                }
                //进入play界面
                play->show(); });
    // shop界面点击increase_mine
    connect(&shop->increase_mine, &QPushButton::clicked, this, [=]()
            {
                //关闭shop界面
                shop->close();
                //如果金币足够
                if(play->hub->money >= INCREASE_MINE)
                {
                    if(!play->increased_mine)
                    {
                        //增加矿地数量
                        play->increased_mine = true;
                        play->map.SecondMap();
                        //消耗金币
                        play->hub->money -= INCREASE_MINE;
                        //把新矿地占据的位置的建筑清空
                        play->RemoveBuilding(GridVec(16, 9));
                        play->RemoveBuilding(GridVec(16, 10));
                        play->RemoveBuilding(GridVec(17, 10));
                        play->RemoveBuilding(GridVec(18, 10));
                        play->RemoveBuilding(GridVec(17, 11));
                        play->RemoveBuilding(GridVec(18, 11));
                        play->RemoveBuilding(GridVec(18, 12));
                        play->RemoveBuilding(GridVec(5, 3));
                        play->RemoveBuilding(GridVec(6, 3));
                        play->RemoveBuilding(GridVec(6, 4));
                        play->RemoveBuilding(GridVec(7, 4));
                        play->RemoveBuilding(GridVec(8, 4));
                        play->RemoveBuilding(GridVec(7, 5));
                        play->RemoveBuilding(GridVec(6, 5));
                        play->RemoveBuilding(GridVec(5, 5));
                        play->RemoveBuilding(GridVec(5, 6));
                        play->RemoveBuilding(GridVec(8, 6));
                    }
                }
                //进入play界面
                play->show(); });
    // shop界面点击increase_item_value
    connect(&shop->increase_item_value, &QPushButton::clicked, this, [=]()
            {
                //关闭shop界面
                shop->close();
                //如果金币足够
                if(play->hub->money >= INCREASE_ITEM_VALUE)
                {
                    if(!play->hub -> increase_item_value)
                    {
                        //升级所有item价值
                        play->hub -> increase_item_value = true;
                        play->hub->money -= INCREASE_ITEM_VALUE;
                    }
                }
                //进入play界面
                play->show(); });
    // shop界面点击return
    connect(&shop->return_button, &QPushButton::clicked, this, [=]()
            {
                //关闭shop界面
                shop->close();
                //进入play界面
                play->show(); });
    // todo,没考虑none
    connect(&play->miner_button, &QPushButton::clicked, this, [=]()
            { play->current_building_name = MINER; });
    //选择传送带
    connect(&play->belt_button, &QPushButton::clicked, this, [=]()
            {
                play->current_building_name = BELT;
                play->ready_to_place_belt = true; });
    connect(&play->cutter_button, &QPushButton::clicked, this, [=]()
            { play->current_building_name = CUTTER; });
    connect(&play->trash_button, &QPushButton::clicked, this, [=]()
            { play->current_building_name = TRASH; });
    connect(&play->tunnel_entry_button, &QPushButton::clicked, this, [=]()
            { play->current_building_name = TUNNEL_ENTRY; });
    connect(&play->tunnel_exit_button, &QPushButton::clicked, this, [=]()
            { play->current_building_name = TUNNEL_EXIT; });
    connect(&play->rotator_button, &QPushButton::clicked, this, [=]()
            { play->current_building_name = ROTATOR; });
    // choose界面选择局部强化miner
    connect(&choose->choose_miner, &QPushButton::clicked, this, [=]()
            {
                choose->close();
                round->show();
                ChooseUpgradeMiner();
                miner_updated = true;
                //CheckAllUpdated();
            });
    // choose界面选择局部强化belt
    connect(&choose->choose_belt, &QPushButton::clicked, this, [=]()
            {
                choose->close();
                round->show();
                ChooseUpgradeBelt();
                belt_updated = true;
                //CheckAllUpdated();
            });
    // choose界面选择局部强化cutter
    connect(&choose->choose_cutter, &QPushButton::clicked, this, [=]()
            {
                choose->close();
                round->show();
                ChooseUpgradeCutter();
                cutter_updated = true;
                //CheckAllUpdated();
            });
}
