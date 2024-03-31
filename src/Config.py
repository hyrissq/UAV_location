import numpy as np

# Train dataset generation
vertex1 = np.array([0, 0])
vertex2 = np.array([200, 0])
vertex3 = np.array([100, 100])
vertex4 = np.array([300, 150])
margin = 0.1  # Example margin

# Training para
train_points_num = 50000
test_points_num = 100
epoch = 50
train_batch_size = 128
test_batch_size = test_points_num
learning_rate = 0.01
# learning_rate = 0.0003

# oppler and w para
fc = 6 * 10e9
c = 3 * 10e8
v = 10

using_model = "rbf"
# using_model = "dnn"
