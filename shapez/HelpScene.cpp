#include "HelpScene.h"
#include "config.h"
HelpScene::HelpScene()
{
    return_button.setParent(this);
    return_button.setFixedSize(115, 115);
    return_button.setStyleSheet("QPushButton{image: url(:/res/return.png)}");
    return_button.move(WIDGET_WIDTH - 150, WIDGET_HEIGHT - 150);
    return_button.show();
    setTxt();
}
void HelpScene::paintEvent(QPaintEvent *e)
{
    QPainter painter(this);
    painter.setFont(QFont("楷体", 25, QFont::Bold));
    painter.drawText(WIDGET_WIDTH - 155, WIDGET_HEIGHT - 165, QString("Back"));
}
void HelpScene::setTxt()
{
    QTextEdit *text = new QTextEdit(this);
    text->setPlainText("                           ABOUT THE GAME   \n"
                       "This is a small demo of Shapez, a game similar to Factorio. It's part of DECO3801 project in UQ.\n"
                       "The base code is on Github:(https://github.com/jiangqianyu/MyShapez)\n"
                       "Basic operations:\n"
                       "Use keyboard(ＷＡＳＤ)to control the direction of building，Click right button to remove it.\n"
                       "There are five kinds of buildings：\n"
                       "Miner：After placing it on a mining site, it can extract items. Please notice the direction of the items.\n"
                       "Belt：Transport items from Miner to Hub. Please. Please notice the direction of the items\n"
                       "Cutter：It cuts the items to the half. If you only need one half of the item, you'll need to use a trash.\n"
                       "Trash：It can receive items from all four directions and permanently dispose of them.。\n"
                       "Victory Condition: In each level, you need to collect a certain number of items. Clear four levels to win。\n"
                       "After each level, you can upgrade one of the buildings: the miner, belt, or cutter. Please note that the upgrade effects are only valid for the current level.\n"
                       "By spending coins, you can globally enhance aspects like the size of the delivery center, the number of mining site tiles, and more in the shop.\n"
                       "The game supports automatic saving.\n"
                       "Thanks for playing！");
    text->setReadOnly(true);
    text->setFixedSize(WIDGET_WIDTH, WIDGET_HEIGHT - 200);
    text->setStyleSheet("background: transparent; color: black; border-style: outset; font: bold 36px; font-family: 楷体;");
    text->move(0, 0);
    text->show();
}
