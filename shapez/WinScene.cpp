#include "WinScene.h"
#include "config.h"
#include <QApplication>
#include <QPainter>
#include <QVBoxLayout>
#include <QMessageBox>

WinScene::WinScene(QWidget *parent) : QWidget(parent), win(WIN_PATH)
{
    // Load the victory image
    win.load(WIN_PATH);

    // Create the close button
    closeButton = new QPushButton(this);
    closeButton->setText(tr("CLOSE GAME"));
    closeButton->setFont(QFont("Kaiti", 28, QFont::Bold));
    closeButton->setGeometry((WIDGET_WIDTH - 400) / 2 + 100, (WIDGET_HEIGHT - 400) / 2 + 100, 200, 50);

    // Connect the button click signal to the close slot function
    connect(closeButton, &QPushButton::clicked, this, &WinScene::closeGame);
}

void WinScene::paintEvent(QPaintEvent *e)
{
    QPainter painter(this);
    painter.setFont(QFont("Kaiti", 35, QFont::Bold));

    // Draw the victory text
    painter.drawText((WIDGET_WIDTH - 200) / 2, (WIDGET_HEIGHT - 400) / 2 - 40, QString("WINNNNNN!!!"));

    // Draw the victory image
    painter.drawPixmap((WIDGET_WIDTH - 400) / 2, (WIDGET_HEIGHT - 400) / 2, 400, 400, win);
}

void WinScene::closeGame()
{
    // Show a confirmation dialog to confirm if the user wants to close
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "LEAVING GAME?", "Are you sure you want to leave the game?",
                                  QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        QApplication::quit();  // Exit the application
    }
}
