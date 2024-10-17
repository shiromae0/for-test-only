#include "RoundScene.h"
#include "config.h"

RoundScene::RoundScene()
{
    // Pixel art loading
    cycle_img.load(CYCLE_PATH);
    rect_img.load(RECT_PATH);
    left_cycle_img.load(LEFT_CYCLE_PATH);
    right_cycle_img.load(RIGHT_CYCLE_PATH);
    // Initialize the level
    level = 1;
    // Initialize the confirm button
    confirm.setParent(this);
    confirm.setFixedSize(200, 100);
    confirm.setStyleSheet("QPushButton{image: url(:/res/confirm.png)}");
}

void RoundScene::paintEvent(QPaintEvent *e)
{
    QPainter painter(this);
    switch (level)
    {
    case 1:
        painter.setFont(QFont("楷体", 55, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 150, WIDGET_HEIGHT / 2 - 150, QString("LEVEL ONE"));
        painter.drawPixmap(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 - 100, 100, 100, cycle_img);
        painter.setFont(QFont("楷体", 45, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 + 100, QString("*20"));
        break;
    case 2:
        painter.setFont(QFont("楷体", 55, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 150, WIDGET_HEIGHT / 2 - 150, QString("LEVEL TWO"));
        painter.drawPixmap(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 - 100, 100, 100, rect_img);
        painter.setFont(QFont("楷体", 45, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 + 100, QString("*30"));
        break;
    case 3:
        painter.setFont(QFont("楷体", 55, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 150, WIDGET_HEIGHT / 2 - 160, QString("LEVEL THREE"));
        painter.drawPixmap(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 - 100, 50, 100, left_cycle_img);
        painter.setFont(QFont("楷体", 45, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 + 100, QString("*40"));
        break;
    case 4:
        painter.setFont(QFont("楷体", 55, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 150, WIDGET_HEIGHT / 2 - 155, QString("LEVEL FOUR"));
        painter.drawPixmap(WIDGET_WIDTH / 2, WIDGET_HEIGHT / 2 - 100, 50, 100, right_cycle_img);
        painter.setFont(QFont("楷体", 45, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 + 100, QString("*50"));
        break;
    case 5:
        painter.setFont(QFont("楷体", 55, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 150, WIDGET_HEIGHT / 2 - 155, QString("LEVEL FIVE"));
        painter.drawPixmap(WIDGET_WIDTH / 2 - 50, WIDGET_HEIGHT / 2 - 100, 100, 100, cycle_img);
        painter.setFont(QFont("楷体", 45, QFont::Bold));
        painter.drawText(WIDGET_WIDTH / 2 - 100, WIDGET_HEIGHT / 2 + 100, QString("4 in 10s"));
    default:
        break;
    }
    confirm.move(WIDGET_WIDTH / 2 - 100, WIDGET_HEIGHT / 2 + 150);
    confirm.show();
}
