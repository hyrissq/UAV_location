import Config as cf
import numpy as np
import math


def shrinkQuadrilateral(v1, v2, v3, v4, margin):
    # Calculate the centroid of the quadrilateral
    centroid = np.mean([v1, v2, v3, v4], axis=0)

    # Define a function to move a point towards the centroid by a given margin
    def move_point_towards_centroid(point, centroid, margin):
        # Calculate the direction vector from the point to the centroid
        direction_vector = centroid - point
        # Normalize the direction vector
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return point  # In case the point is already at the centroid
        normalized_direction_vector = direction_vector / norm
        # Calculate the new point that is 'margin' distance closer to the centroid
        new_point = point + margin * normalized_direction_vector
        return new_point

    # Move each vertex towards the centroid by the margin
    new_v1 = move_point_towards_centroid(v1, centroid, margin)
    new_v2 = move_point_towards_centroid(v2, centroid, margin)
    new_v3 = move_point_towards_centroid(v3, centroid, margin)
    new_v4 = move_point_towards_centroid(v4, centroid, margin)

    return new_v1, new_v2, new_v3, new_v4


#####


# Function to calculate the angle between vectors OA and AB for each pair of points
def calculate_angles_phi(array_A, array_B, origin):
    angles = []
    for A, B in zip(array_A, array_B):
        vector_OA = A - origin
        vector_AB = B - A
        dot_product = np.dot(vector_OA, vector_AB)
        norm_OA = np.linalg.norm(vector_OA)
        norm_AB = np.linalg.norm(vector_AB)
        # Avoid division by zero in case one of the vectors is zero
        if norm_OA == 0 or norm_AB == 0:
            angle = 0
        else:
            # Clip the cosine value to avoid numerical errors beyond the range [-1, 1]
            cos_angle = dot_product / (norm_OA * norm_AB)
            cos_angle_clipped = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle_clipped)
        angles.append(angle)
    return angles


# PUBLIC
def get_angle_phi(coords_A, coords_B):
    # Call the function and pass the arrays
    phi1 = calculate_angles_phi(coords_A, coords_B, cf.vertex1)
    phi2 = calculate_angles_phi(coords_A, coords_B, cf.vertex2)
    phi3 = calculate_angles_phi(coords_A, coords_B, cf.vertex3)
    phi4 = calculate_angles_phi(coords_A, coords_B, cf.vertex4)

    return [phi1, phi2, phi3, phi4]


def calculate_distance(x, y, a, b):
    """Calculate the Euclidean distance between two points (x, y) and (a, b)."""
    return math.sqrt((a - x) ** 2 + (b - y) ** 2)


# 定义计算夹角的函数，以计算与x轴的夹角
def calculate_angle_with_x_axis(x, y, a, b):
    # 向量p1p2
    vector = np.array([a - x, b - y])
    # x轴正方向的向量
    x_axis = np.array([1, 0])

    # 向量的点乘
    dot_product = np.dot(vector, x_axis)
    # 向量的模长
    norm_vector = np.linalg.norm(vector)

    # 防止除以零
    if norm_vector == 0:
        return 0

    # 计算夹角的余弦值
    cos_angle = dot_product / norm_vector
    # 余弦值的范围是[-1, 1]，可能由于浮点数误差超出这个范围
    # 这里将其限制在[-1, 1]内
    cos_angle = np.clip(cos_angle, -1, 1)

    # 计算夹角（弧度转换为度）
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)

    # 考虑向量在y轴上的方向，如果向量第二个分量是负的，则它在x轴的下方，夹角应该是360°-计算出的角度
    if vector[1] < 0:
        angle_degrees = 360 - angle_degrees

    return angle_degrees


# 继续使用之前定义的点和距离计算方法
def get_angle_theta(coords_A, coords_B):
    # o1a = []
    # o1b = []
    # o2a = []
    # o2b = []
    # o3a = []
    # o3b = []
    # o4a = []
    # o4b = []

    theta_1a = []
    theta_1b = []
    theta_2a = []
    theta_2b = []
    theta_3a = []
    theta_3b = []
    theta_4a = []
    theta_4b = []

    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []

    data_length = np.shape(coords_A)[0]

    for i in range(0, data_length):
        aPosition = coords_A[i]
        bPosition = coords_B[i]

        # 计算距离
        # o1a.append(
        #     calculate_distance(cf.vertex1[0], cf.vertex1[1], aPosition[0], aPosition[1])
        # )
        # o1b.append(
        #     calculate_distance(cf.vertex1[0], cf.vertex1[1], bPosition[0], bPosition[1])
        # )
        # o2a.append(
        #     calculate_distance(cf.vertex2[0], cf.vertex2[1], aPosition[0], aPosition[1])
        # )
        # o2b.append(
        #     calculate_distance(cf.vertex2[0], cf.vertex2[1], bPosition[0], bPosition[1])
        # )
        # o3a.append(
        #     calculate_distance(cf.vertex3[0], cf.vertex3[1], aPosition[0], aPosition[1])
        # )
        # o3b.append(
        #     calculate_distance(cf.vertex3[0], cf.vertex3[1], bPosition[0], bPosition[1])
        # )
        # o4a.append(
        #     calculate_distance(cf.vertex4[0], cf.vertex4[1], aPosition[0], aPosition[1])
        # )
        # o4b.append(
        #     calculate_distance(cf.vertex4[0], cf.vertex4[1], bPosition[0], bPosition[1])
        # )

        # 计算夹角
        n_theta_1a = calculate_angle_with_x_axis(
            cf.vertex1[0], cf.vertex1[1], aPosition[0], aPosition[1]
        )
        n_theta_1b = calculate_angle_with_x_axis(
            cf.vertex1[0], cf.vertex1[1], bPosition[0], bPosition[1]
        )
        n_theta_2a = calculate_angle_with_x_axis(
            cf.vertex2[0], cf.vertex2[1], aPosition[0], aPosition[1]
        )
        n_theta_2b = calculate_angle_with_x_axis(
            cf.vertex2[0], cf.vertex2[1], bPosition[0], bPosition[1]
        )
        n_theta_3a = calculate_angle_with_x_axis(
            cf.vertex3[0], cf.vertex3[1], aPosition[0], aPosition[1]
        )
        n_theta_3b = calculate_angle_with_x_axis(
            cf.vertex3[0], cf.vertex3[1], bPosition[0], bPosition[1]
        )
        n_theta_4a = calculate_angle_with_x_axis(
            cf.vertex4[0], cf.vertex4[1], aPosition[0], aPosition[1]
        )
        n_theta_4b = calculate_angle_with_x_axis(
            cf.vertex4[0], cf.vertex4[1], bPosition[0], bPosition[1]
        )

        theta_1a.append(n_theta_1a)
        theta_1b.append(n_theta_1b)
        theta_2a.append(n_theta_2a)
        theta_2b.append(n_theta_2b)
        theta_3a.append(n_theta_3a)
        theta_3b.append(n_theta_3b)
        theta_4a.append(n_theta_4a)
        theta_4b.append(n_theta_4b)

        theta1.append(abs(n_theta_1a - n_theta_1b))
        theta2.append(abs(n_theta_2a - n_theta_2b))
        theta3.append(abs(n_theta_3a - n_theta_3b))
        theta4.append(abs(n_theta_4a - n_theta_4b))

    return [
        np.radians(theta1),  # theta1
        np.radians(theta2),  # theta2
        np.radians(theta3),  # theta3
        np.radians(theta4),  # theta4
    ]


def getWAndDoppler(coords_A, coords_B):
    data_length = np.shape(coords_A)[0]

    [phi1, phi2, phi3, phi4] = get_angle_phi(coords_A, coords_B)
    [theta1, theta2, theta3, theta4] = get_angle_theta(coords_A, coords_B)

    w = np.zeros([data_length, 3])
    doppler = np.zeros([data_length, 3])
    for i in range(data_length):
        x = coords_A[i][0]
        y = coords_A[i][1]
        x1 = coords_B[i][0]
        y1 = coords_B[i][1]

        d11 = calculate_distance(x, y, cf.vertex1[0], cf.vertex1[1])
        d12 = calculate_distance(x1, y1, cf.vertex1[0], cf.vertex1[1])
        d21 = calculate_distance(x, y, cf.vertex2[0], cf.vertex2[1])
        d22 = calculate_distance(x1, y1, cf.vertex2[0], cf.vertex2[1])
        d31 = calculate_distance(x, y, cf.vertex3[0], cf.vertex3[1])
        d32 = calculate_distance(x1, y1, cf.vertex3[0], cf.vertex3[1])
        d41 = calculate_distance(x, y, cf.vertex3[0], cf.vertex3[1])
        d42 = calculate_distance(x1, y1, cf.vertex3[0], cf.vertex3[1])
        # dd12 = np.abs(d11+d21-d12-d22)
        # dd13 = np.abs(d11+d31-d12-d32)

        w12 = (
            d11
            * theta1[i]
            * (np.cos(phi1[i]) + np.cos(phi2[i]))
            * cf.fc
            / cf.c
            * (theta1[i] * np.cos(phi1[i]) + np.sin(phi1[i]))
        )
        w13 = (
            d31
            * theta3[i]
            * (np.cos(phi1[i]) + np.cos(phi3[i]))
            * cf.fc
            / cf.c
            * (theta3[i] * np.cos(phi3[i]) + np.sin(phi3[i]))
        )
        w14 = (
            d41
            * theta4[i]
            * (np.cos(phi1[i]) + np.cos(phi3[i]))
            * cf.fc
            / cf.c
            * (theta4[i] * np.cos(phi4[i]) + np.sin(phi4[i]))
        )

        v12 = cf.v * cf.fc * (np.cos(phi1[i]) + np.cos(phi2[i])) / cf.c
        v13 = cf.v * cf.fc * (np.cos(phi1[i]) + np.cos(phi3[i])) / cf.c
        v14 = cf.v * cf.fc * (np.cos(phi1[i]) + np.cos(phi4[i])) / cf.c

        # w12 = 2*6*10e9*3.14*dd12/(3*10e8)
        # w13 = 2*6*10e9*3.14*dd13/(3*10e8)
        # v12 = (6*10e9/(3*10e8))*dd12*100
        # v13 = (6*10e9/(3*10e8))*dd13*100
        w[i][0] = w12
        w[i][1] = w13
        w[i][2] = w14

        doppler[i][0] = v12
        doppler[i][1] = v13
        doppler[i][2] = v14

    return [w, doppler]
