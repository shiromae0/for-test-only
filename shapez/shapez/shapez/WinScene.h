
#ifndef WINSCENE_H
#define WINSCENE_H

#include <QWidget>
#include <QPushButton>
#include <QPixmap>

class WinScene : public QWidget
{
    Q_OBJECT  // 确保 Qt 元对象系统能处理此类

public:
    explicit WinScene(QWidget *parent = nullptr);  // 构造函数
    void paintEvent(QPaintEvent *e) override;  // 重写 paintEvent

private slots:
    void closeGame();  // 槽函数，用于关闭应用程序

private:
    QPushButton *closeButton;  // 关闭按钮
    QPixmap win;  // 用于显示的胜利图片
};

#endif // WINSCENE_H
