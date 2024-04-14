import numpy as np

transmittor_position = np.array([0, 0])
receiver_position_1 = np.array([200, 0])
receiver_position_2 = np.array([100, 100])
receiver_position_3 = np.array([300, 150])

train_num_of_lines_to_generate = 51200
test_set_num = 512
train_step_count_per_line = 5
length_per_step = 0.2
angle_change_limit_per_step = 0.1
a_b_distance = 0.05

epoch = 1000
train_batch_size = 128
test_batch_size = test_set_num

learning_rate = 0.0003

# oppler and w para
fc = 6 * 10e9
c = 3 * 10e8
v = 10

# using_model = "rbf"
using_model = "dnn"
# using_model = "Transformer"
