// main.cpp is the entry file
#include "WindowControl.h"
#include "RoundScene.h"
#include "PlayScene.h"

#include <QApplication>
#include <QResource>
#include "config.h"
int main(int argc, char *argv[])
{
    // QApplication is a class provided by the Qt framework
    // It handles events like mouse clicks and keyboard input
    QApplication a(argc, argv);
    // Register external resource files
    QResource::registerResource(GAME_RES_PATH);
    WindowControl w;
    // Create a window and display it
    // Enter the event loop until the application exits
    return a.exec();
}
