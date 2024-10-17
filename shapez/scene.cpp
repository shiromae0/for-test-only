#include "scene.h"
#include "config.h"
Scene::Scene(QWidget *parent)
    : QWidget{parent}
{
    // set window size
    this->setFixedSize(WIDGET_WIDTH, WIDGET_HEIGHT);
    // set window icon
    this->setWindowIcon(QIcon(GAME_ICON));
    // set window name
    this->setWindowTitle(GAME_TITLE);
    //
    setMouseTracking(true);
    //this->showFullScreen();
}
