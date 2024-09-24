#ifndef TRANSPORTRATECALCULATOR_H
#define TRANSPORTRATECALCULATOR_H
class TransportRateCalculator {
public:
    TransportRateCalculator();

    // 设置运输物体的总数量
    void setTotalObjects(int total);

    // 设置经过的时间（秒）
    void setElapsedTime(double time);

    // 计算并返回运输速率（单位：物体/秒）
    double calculateRate() const;

private:
    int totalObjects;   // 物体总数
    double elapsedTime; // 时间
};
#endif // TRANSPORTRATECALCULATOR_H
