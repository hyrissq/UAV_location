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
learning_rate = 0.005

# oppler and w para
fc = 6 * 10e9
c = 3 * 10e8
v = 10
