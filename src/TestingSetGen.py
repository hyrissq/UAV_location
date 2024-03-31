import numpy as np
import Config as cf
import Common as cm

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
def getPoints():
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
