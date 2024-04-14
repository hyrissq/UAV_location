import numpy as np

# train dataset gen
vertex1 = np.array([0, 0])
vertex2 = np.array([200, 0])
vertex3 = np.array([100, 100])
vertex4 = np.array([300, 150])

margin = 0.1  # Example margin
train_set_num = 10

line_seq_count = 5
line_total_length = 2.0
line_angle_change = 4

# testing dataset generation
use_new_test_gen_method = True

# test_points_num = 1500
# test_batch_size = test_points_num

epoch = 1000
train_batch_size = 128
learning_rate = 0.0003

# learning_rate = 0.0003

# oppler and w para
fc = 6 * 10e9
c = 3 * 10e8
v = 10

# using model
# using_model = "rbf"
using_model = "dnn"
# using_model = "Transformer"
