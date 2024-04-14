import numpy as np


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def point_in_triangle(v1, v2, v3, pt):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def point_in_boundary(v1, v2, v3, v4, point):
    # check if point is in boundary
    return point_in_triangle(v1, v2, v3, point) or point_in_triangle(
        v3, v2, v4, point
    )


def get_random_sample_in_quad(v1, v2, v3, v4):
    # Randomly choose one of the two triangles to place a point
    if np.random.rand() < 0.5:
        # Working with triangle 1 (p123)
        base_point = v1
        edge0 = v2 - v1
        edge1 = v3 - v1
    else:
        # Working with triangle 2 (p234)
        base_point = v2
        edge0 = v3 - v2
        edge1 = v4 - v2
    # Generate random x, y in the range [0, 1]
    x, y = np.random.rand(2)
    # Ensure the random point (x, y) lies within the triangle
    if x + y > 1:
        x = 1 - x
        y = 1 - y
    # Calculate a random point within the selected triangle
    return base_point + edge0 * x + edge1 * y


angle_change_limit = 0.1
a_b_distance = 0.05  # Distance from this A to this B


def random_angle_change(original_angle, angle_change_tolerance):
    # Generate a random angle change within the tolerance
    ang = original_angle + (np.random.rand() - 0.5) * \
        2 * angle_change_tolerance

    # Ensure the angle is within the range [0, 2pi]
    if ang < 0:
        ang += 2 * np.pi
    if ang > 2 * np.pi:
        ang -= 2 * np.pi
    return ang


def random_walk(angle_change_per_step, length_per_step, last_angle, last_coords_a):
    this_angle = random_angle_change(last_angle, angle_change_per_step)
    this_coords_a = (
        last_coords_a
        + np.array([np.cos(this_angle), np.sin(this_angle)]) * length_per_step
    )
    return [this_angle, this_coords_a]


def get_coords_b(coords_a, angle):
    return coords_a + np.array([np.cos(angle), np.sin(angle)]) * a_b_distance


def try_create_line_in_bounding_box(detecting_region_info, step_count_per_line, length_per_step, angle_change_limit_per_step):
    # Start from a random point inside the boundary
    starting_coords_a = get_random_sample_in_quad(
        detecting_region_info.v1, detecting_region_info.v2, detecting_region_info.v3, detecting_region_info.v4
    )
    starting_angle = np.random.rand() * 2 * np.pi

    _angle = starting_angle
    _coords_a = starting_coords_a

    line_a = [starting_coords_a]
    line_b = [get_coords_b(starting_coords_a, starting_angle)]

    while True:
        [_angle, _coords_a] = random_walk(
            angle_change_limit_per_step, length_per_step, _angle, _coords_a
        )

        in_boundary = point_in_boundary(
            detecting_region_info.v1, detecting_region_info.v2, detecting_region_info.v3, detecting_region_info.v4, _coords_a
        )
        if not in_boundary:
            return [line_a, line_b, False]

        line_a.append(_coords_a)
        line_b.append(get_coords_b(_coords_a, _angle))

        if len(line_a) == step_count_per_line:
            return [line_a, line_b, True]


def generateLines(detecting_region_info, num_of_lines_to_generate, step_count_per_line, length_per_step, angle_change_limit_per_step):
    valid_line_count = 0

    lines_a = []
    lines_b = []

    while valid_line_count < num_of_lines_to_generate:
        [line_a, line_b, valid] = try_create_line_in_bounding_box(detecting_region_info,
                                                                  step_count_per_line, length_per_step, angle_change_limit_per_step)

        if valid:
            valid_line_count += 1
            lines_a.append(line_a)
            lines_b.append(line_b)

    lines_a = np.array(lines_a)
    lines_b = np.array(lines_b)

    return [lines_a, lines_b]
