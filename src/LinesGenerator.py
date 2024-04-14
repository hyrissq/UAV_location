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


def point_in_boundary(detecting_region_info, pt):
    v1 = detecting_region_info.v1
    v2 = detecting_region_info.v2
    v3 = detecting_region_info.v3
    v4 = detecting_region_info.v4

    # check if point is in boundary
    return point_in_triangle(v1, v2, v3, pt) or point_in_triangle(
        v3, v2, v4, pt
    )


def get_triangle_size(v1, v2, v3):
    # Arrange the vertices into a 3x3 matrix, where the last column is all ones
    matrix = np.array([
        [v1[0], v1[1], 1],
        [v2[0], v2[1], 1],
        [v3[0], v3[1], 1]
    ])

    # Calculate the determinant of the matrix
    det = np.linalg.det(matrix)

    # The area of the triangle is half the absolute value of the determinant
    area = 0.5 * np.abs(det)

    return area


def get_random_sample_in_detecting_region(detecting_region_info):
    v1 = detecting_region_info.v1
    v2 = detecting_region_info.v2
    v3 = detecting_region_info.v3
    v4 = detecting_region_info.v4

    tri_size_1 = get_triangle_size(v1, v2, v3)
    tri_size_2 = get_triangle_size(v2, v3, v4)

    # Randomly choose one of the two triangles to place a point
    if np.random.rand() < tri_size_1 / (tri_size_1 + tri_size_2):
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
    this_coords_a = (
        last_coords_a
        + np.array([np.cos(last_angle), np.sin(last_angle)]) * length_per_step
    )
    this_angle = random_angle_change(last_angle, angle_change_per_step)
    return [this_angle, this_coords_a]


def get_coords_b(a_b_distance, coords_a, angle):
    return coords_a + np.array([np.cos(angle), np.sin(angle)]) * a_b_distance


def try_create_line_in_bounding_box(a_b_distance, detecting_region_info, step_count_per_line, length_per_step, angle_change_limit_per_step):
    # Start from a random point inside the boundary
    _angle = np.random.rand() * 2 * np.pi

    _coords_a = get_random_sample_in_detecting_region(
        detecting_region_info)
    _coords_b = get_coords_b(a_b_distance, _coords_a, _angle)

    in_boundary = point_in_boundary(
        detecting_region_info, _coords_a
    ) and point_in_boundary(
        detecting_region_info, _coords_b
    )
    if not in_boundary:
        return [None, None, False]

    line_a = [_coords_a]
    line_b = [_coords_b]

    while True:
        [_angle, _coords_a] = random_walk(
            angle_change_limit_per_step, length_per_step, _angle, _coords_a
        )
        _coords_b = get_coords_b(a_b_distance, _coords_a, _angle)

        in_boundary = point_in_boundary(
            detecting_region_info, _coords_a
        ) and point_in_boundary(
            detecting_region_info, _coords_b
        )
        if not in_boundary:
            return [None, None, False]

        line_a.append(_coords_a)
        line_b.append(_coords_b)

        if len(line_a) == step_count_per_line:
            return [line_a, line_b, True]


def generateLines(detecting_region_info, a_b_distance, num_of_lines_to_generate, step_count_per_line, length_per_step, angle_change_limit_per_step):
    valid_line_count = 0

    lines_a = []
    lines_b = []

    while valid_line_count < num_of_lines_to_generate:
        [line_a, line_b, valid] = try_create_line_in_bounding_box(a_b_distance, detecting_region_info,
                                                                  step_count_per_line, length_per_step, angle_change_limit_per_step)

        if valid:
            valid_line_count += 1
            lines_a.append(line_a)
            lines_b.append(line_b)

    lines_a = np.array(lines_a)
    lines_b = np.array(lines_b)

    return [lines_a, lines_b]


def reparseSingleLineAsLines(line_a, line_b, num_of_lines_to_generate, step_count_per_line):
    lines_a = []
    lines_b = []

    point_size_of_line_a = len(line_a)
    print("point_size_of_line_a: ", point_size_of_line_a)

    assert (num_of_lines_to_generate +
            (step_count_per_line - 1)) <= point_size_of_line_a

    cursor = 0
    while cursor < num_of_lines_to_generate:
        lines_a.append(line_a[cursor: cursor + step_count_per_line])
        lines_b.append(line_b[cursor: cursor + step_count_per_line])
        cursor += 1

    return [lines_a, lines_b]
