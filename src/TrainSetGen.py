import numpy as np
import src.Config as cf
import src.Common as cm


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
    shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4 = cm.shrinkQuadrilateral(
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