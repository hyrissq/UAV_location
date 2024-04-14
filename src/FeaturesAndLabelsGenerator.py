import numpy as np


def get_d_phi(detecting_region_info, coord_a):
    ref = detecting_region_info.transmittor_position
    phi = np.arctan2(coord_a[1] - ref[1], coord_a[0] - ref[0])
    # change range from -pi to pi to 0 to 2pi
    if phi < 0:
        phi += 2 * np.pi
    d = np.sqrt((coord_a[0] - ref[0])**2 + (coord_a[1] - ref[1])**2)
    return [d, phi]


def get_labels(detecting_region_info, lines_a):
    labels = []
    for line in lines_a:
        for coord_a in line:
            labels.append(get_d_phi(detecting_region_info, coord_a))
    return np.array(labels)


def generateFeaturesAndLabels(detecting_region_info, lines_a, w, doppler, num_of_lines_to_generate, step_count_per_line):
    features = np.concatenate((w, doppler), axis=1)
    labels = get_labels(detecting_region_info, lines_a)
    reshaped_features = features.reshape(
        num_of_lines_to_generate, step_count_per_line, 6)
    reshaped_labels = labels.reshape(
        num_of_lines_to_generate, step_count_per_line, 2)

    # drop the sequence of the label
    reshaped_labels = reshaped_labels[:, -1, :]

    return [reshaped_features, reshaped_labels]
