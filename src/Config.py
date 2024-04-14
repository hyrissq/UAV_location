import numpy as np

transmittor_position = np.array([0, 0])
receiver_position_1 = np.array([200, 0])
receiver_position_2 = np.array([100, 100])
receiver_position_3 = np.array([300, 150])

# margin = 0.1  # Example margin
train_set_num = 10  # Number of lines in the training set
test_set_num = 5  # Number of lines
line_seq_count = 5

line_step_length = 0.2
line_step_angle_change = 0.4


test_set_num = 1500
# test_batch_size = test_points_num

epoch = 8
train_batch_size = 128
learning_rate = 0.0003

# oppler and w para
fc = 6 * 10e9
c = 3 * 10e8
v = 10

# using_model = "rbf"
using_model = "dnn"
# using_model = "Transformer"
