import numpy as np
import sympy
import math

import src.Config as cf
import src.Common as cm

import numpy as np
import src.Config as cf
import src.Common as cm

# import math


def coord_in_boundary(shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4, coordinate):
    # TODO: use shrinked coordinates as boundary, or make assertions so that xx and yy never exceeds the boundary
    return True


def get_random_samples_on_curve(
    shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4, numbers_to_generate
):

    lower_x_boundary = 90
    upper_x_boundary = 105
    step_size = (upper_x_boundary - lower_x_boundary) / numbers_to_generate
    xx = np.arange(90, 105, step_size)

    yy = (
        42
        + 10 * np.sin((xx - 90) / 5)
        + 5 * np.cos((xx - 95) / 5)
        + 3 * np.sin((xx - 100) / 2)
        + 2 * np.exp((xx - 105) / 10)
        + 2 * np.sin((xx - 95) / 2) * np.cos((xx - 100) / 5)
    )

    coordinates = []
    for i in range(numbers_to_generate):
        coordinate = [xx[i], yy[i]]
        assert coord_in_boundary(
            shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4, coordinate
        )
        coordinates.append(coordinate)

    return coordinates


def get_coordinates():
    shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4 = cm.shrinkQuadrilateral(
        cf.vertex1, cf.vertex2, cf.vertex3, cf.vertex4, cf.margin
    )

    coordinates = get_random_samples_on_curve(
        shrinked_v1,
        shrinked_v2,
        shrinked_v3,
        shrinked_v4,
        cf.test_points_num,
    )
    return np.vstack(coordinates)


# this is not clean!
def get_points_old():
    coordinates = get_coordinates()

    random_distances = np.random.uniform(0.0, 0.05, size=(len(coordinates)))

    # shift alt
    # Generate random angles in the srange (0, 2*pi)
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


def getY(x):
    return (
        42
        + 10 * np.sin((x - 90) / 5)
        + 5 * np.cos((x - 95) / 5)
        + 3 * np.sin((x - 100) / 2)
        + 2 * np.exp((x - 105) / 10)
        + 2 * np.sin((x - 95) / 2) * np.cos((x - 100) / 5)
    )


def getDy(x):
    delta = 0.0001
    return (getY(x + delta) - getY(x)) / delta


def normalize2D(v):
    sumOfSquares = v[0] ** 2 + v[1] ** 2
    v[0] = v[0] / math.sqrt(sumOfSquares)
    v[1] = v[1] / math.sqrt(sumOfSquares)
    return v


def multiply2D(v, scalar):
    v[0] = v[0] * scalar
    v[1] = v[1] * scalar
    return v


def coord_in_boundary(shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4, coordinate):
    # TODO: use shrinked coordinates as boundary, or make assertions so that xx and yy never exceeds the boundary
    return True


def get_random_samples_on_curve(
    shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4, numbers_to_generate
):

    lower_x_boundary = 90
    upper_x_boundary = 105
    step_size = (upper_x_boundary - lower_x_boundary) / numbers_to_generate
    xx = np.arange(90, 105, step_size)

    yy = getY(xx)

    coordinates = []
    for i in range(numbers_to_generate):
        coordinate = [xx[i], yy[i]]
        assert coord_in_boundary(
            shrinked_v1, shrinked_v2, shrinked_v3, shrinked_v4, coordinate
        )
        coordinates.append(coordinate)

    return coordinates


# this is not clean!
def get_points_new():
    coordinates = get_coordinates()

    new_x_values = np.array([])
    new_y_values = np.array([])

    for coordinate in coordinates:
        dY = getDy(coordinate[0])
        speedVector = [1, dY]
        speedVector = normalize2D(speedVector)
        speedVector = multiply2D(speedVector, 0.05)
        new_x_values = np.append(new_x_values, coordinate[0] + speedVector[0])
        new_y_values = np.append(new_y_values, coordinate[1] + speedVector[1])

    # Combine the original and new coordinates to form a 4D vector
    result_array = np.column_stack(
        (coordinates[:, 0], coordinates[:, 1], new_x_values, new_y_values)
    )

    return [
        result_array.reshape((len(coordinates), 4)),  # result_array
        np.array(list(zip(coordinates[:, 0], coordinates[:, 1]))),  # coords_A
        np.array(list(zip(new_x_values, new_y_values))),  # coords_B
    ]


def getPoints():
    if cf.use_new_test_gen_method:
        return get_points_new()
    else:
        return get_points_old()
