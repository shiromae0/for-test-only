#include "WinScene.h"
#include "config.h"
#include <QApplication>
#include <QPainter>
#include <QVBoxLayout>
#include <QMessageBox>

WinScene::WinScene(QWidget *parent) : QWidget(parent), win(WIN_PATH)
{
    // 加载胜利图片
    win.load(WIN_PATH);

    // 创建关闭按钮
    closeButton = new QPushButton(this);
    closeButton->setText(tr("CLOSE GAME"));
    closeButton->setFont(QFont("楷体", 28, QFont::Bold));
    closeButton->setGeometry((WIDGET_WIDTH - 400) / 2 + 100, (WIDGET_HEIGHT - 400) / 2 + 100, 200, 50);

    // 连接按钮点击信号到关闭槽函数
    connect(closeButton, &QPushButton::clicked, this, &WinScene::closeGame);
}

void WinScene::paintEvent(QPaintEvent *e)
{
    QPainter painter(this);
    painter.setFont(QFont("楷体", 35, QFont::Bold));

    // 绘制胜利文本
    painter.drawText((WIDGET_WIDTH - 200) / 2, (WIDGET_HEIGHT - 400) / 2 - 40, QString("WINNNNNN!!!"));

    // 绘制胜利图片
    painter.drawPixmap((WIDGET_WIDTH - 400) / 2, (WIDGET_HEIGHT - 400) / 2, 400, 400, win);
}

void WinScene::closeGame()
{
    // 弹出确认对话框，确认是否关闭
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "LEAVING GAME?", "Are you sure to leave the game？",
                                  QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        QApplication::quit();  // 退出应用程序
    }
}
