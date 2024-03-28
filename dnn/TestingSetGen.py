import numpy as np

# import math


def generate_modified_complex_curve():
    # 生成横坐标在90到105之间的点
    x_values = np.arange(90, 105, 0.01)

    y_values = (
        42
        + 10 * np.sin((x_values - 90) / 5)
        + 5 * np.cos((x_values - 95) / 5)
        + 3 * np.sin((x_values - 100) / 2)
        + 2 * np.exp((x_values - 105) / 10)
        + 2 * np.sin((x_values - 95) / 2) * np.cos((x_values - 100) / 5)
    )

    # 保证相邻点之间的距离小于0.05
    x_points = []
    y_points = []

    for i in range(len(x_values) - 1):
        x_points.append(x_values[i])
        y_points.append(y_values[i])

        # 计算当前点与下一个点之间的距离
        distance = np.sqrt(
            (x_values[i + 1] - x_values[i]) ** 2 + (y_values[i + 1] - y_values[i]) ** 2
        )

        # 如果距离大于0.05，则插入新点，使距离小于0.05
        while distance > 0.1:
            x_new = (x_values[i] + x_values[i + 1]) / 2
            y_new = (y_values[i] + y_values[i + 1]) / 2

            x_points.append(x_new)
            y_points.append(y_new)

            distance = np.sqrt(
                (x_values[i + 1] - x_new) ** 2 + (y_values[i + 1] - y_new) ** 2
            )

    return x_points, y_points


def getPoints():
    x_points, y_points = generate_modified_complex_curve()

    # result_test = np.zeros([len(x_points)-1,4])
    # for i in range(len(x_points)-2):
    #     result_test[i][0]=x_points[i]
    #     result_test[i][1]=y_points[i]
    #     result_test[i][2]=x_points[i+1]
    #     result_test[i][3]=y_points[i+1]

    return []
