import numpy as np
import math

import Config as cf


def shrink_quadrilateral(v1, v2, v3, v4, margin):
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


# def shrink_triangle(v1, v2, v3, margin):
#     # Calculate the centroid of the triangle
#     centroid = np.mean([v1, v2, v3], axis=0)

#     # Define a function to move a point towards the centroid by a given margin
#     def move_point_towards_centroid(point, centroid, margin):
#         # Calculate the direction vector from the point to the centroid
#         direction_vector = centroid - point
#         # Normalize the direction vector
#         norm = np.linalg.norm(direction_vector)
#         if norm == 0:
#             return point  # In case the point is already at the centroid
#         normalized_direction_vector = direction_vector / norm
#         # Calculate the new point that is 'margin' distance closer to the centroid
#         new_point = point + margin * normalized_direction_vector
#         return new_point

#     # Move each vertex towards the centroid by the margin
#     new_v1 = move_point_towards_centroid(v1, centroid, margin)
#     new_v2 = move_point_towards_centroid(v2, centroid, margin)
#     new_v3 = move_point_towards_centroid(v3, centroid, margin)

#     return new_v1, new_v2, new_v3


def get_random_samples_in_quad(x1, y1, x2, y2, x3, y3, x4, y4, train_points_num):
    # Coordinates of the quadrilateral's vertices
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    point3 = np.array([x3, y3])
    point4 = np.array([x4, y4])

    # Divide the quadrilateral into two triangles:
    # Triangle 1: point0, point1, point2
    # Triangle 2: point2, point3, point0

    # List to store the random points
    randomPoints = []

    for _ in range(train_points_num):
        # Randomly choose one of the two triangles to place a point
        if np.random.rand() < 0.5:
            # Working with triangle 1 (p123)
            base_point = point1
            edge0 = point2 - point1
            edge1 = point3 - point1
        else:
            # Working with triangle 2 (p234)
            base_point = point2
            edge0 = point3 - point2
            edge1 = point4 - point2

        # Generate random x, y in the range [0, 1]
        x, y = np.random.rand(2)
        # Ensure the random point (x, y) lies within the triangle
        if x + y > 1:
            x = 1 - x
            y = 1 - y

        # Calculate a random point within the selected triangle
        randomPoint = base_point + edge0 * x + edge1 * y
        randomPoints.append(randomPoint)

    return randomPoints


def get_coordinates():
    shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4 = shrink_quadrilateral(
        cf.vertex1, cf.vertex2, cf.vertex3, cf.vertex4, cf.margin
    )

    coordinates = get_random_samples_in_quad(
        shrinked_v1[0],
        shrinked_v1[1],
        shrinked_v2[0],
        shrinked_v2[1],
        shrinked_v3[0],
        shrinked_v3[1],
        shrinked_v4[0],
        shrinked_v4[1],
        cf.train_points_num,
    )
    return np.vstack(coordinates)


def getPoints():
    coordinates = get_coordinates()

    random_distances = np.random.uniform(0.0, 0.05, size=(len(coordinates)))

    # shift alt
    # Generate random angles in the range (0, 2*pi)
    random_angles = np.random.uniform(0, 2 * np.pi, size=(len(coordinates)))

    # Convert polar coordinates to cartesian coordinates for the second point
    delta_x = random_distances * np.cos(random_angles)
    delta_y = random_distances * np.sin(random_angles)

    # Create the second set of coordinates
    new_x_values = coordinates[:, 0] + delta_x
    new_y_values = coordinates[:, 1] + delta_y

    # Combine the original and new coordinates to form a 4D vector
    result_array = np.column_stack(
        (coordinates[:, 0], coordinates[:, 1], new_x_values, new_y_values)
    )

    return [
        result_array.reshape((len(coordinates), 4)),  # result_array
        np.array(list(zip(coordinates[:, 0], coordinates[:, 1]))),  # coords_A
        np.array(list(zip(new_x_values, new_y_values))),  # coords_B
    ]


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
def getAnglePhi(coords_A, coords_B):
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
def getAngleTheta(coords_A, coords_B):
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

    for i in range(0, cf.train_points_num):
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


def getWAndDoppler(
    result_array, theta1, theta2, theta3, theta4, phi1, phi2, phi3, phi4
):
    w = np.zeros([cf.train_points_num, 3])
    doppler = np.zeros([cf.train_points_num, 3])
    for i in range(cf.train_points_num):
        x = result_array[i][0]
        y = result_array[i][1]
        x1 = result_array[i][2]
        y1 = result_array[i][3]

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
