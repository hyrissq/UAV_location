# 定义更深的神经网络模型
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(14, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.fc9 = nn.Linear(64, 32)
        self.bn9 = nn.BatchNorm1d(32)
        self.fc10 = nn.Linear(32, 4)
        # Activation function can be assigned as a member variable
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.activation(self.bn4(self.fc4(x)))
        x = self.activation(self.bn5(self.fc5(x)))
        x = self.activation(self.bn6(self.fc6(x)))
        x = self.activation(self.bn7(self.fc7(x)))
        x = self.activation(self.bn8(self.fc8(x)))
        x = self.activation(self.bn9(self.fc9(x)))
        x = self.fc10(x)  # Usually no batch norm just before the final layer
        return x


# class DNN(nn.Module):
#     def __init__(self):
#         super(DNN, self).__init__()
#         self.fc1 = nn.Linear(14, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 512)
#         self.fc4 = nn.Linear(512, 512)  # 新增层
#         self.fc5 = nn.Linear(512, 512)  # 新增层
#         self.fc6 = nn.Linear(512, 256)
#         self.fc7 = nn.Linear(256, 128)
#         self.fc8 = nn.Linear(128, 64)
#         self.fc9 = nn.Linear(64, 32)
#         self.fc10 = nn.Linear(32, 4)

#         self.activation_layer = nn.ReLU()
        

#     def forward(self, x):
#         x = self.activation_layer(self.fc1(x))
#         x = self.activation_layer(self.fc2(x))
#         x = self.activation_layer(self.fc3(x))
#         x = self.activation_layer(self.fc4(x))  # 新增层
#         x = self.activation_layer(self.fc5(x))  # 新增层
#         x = self.activation_layer(self.fc6(x))
#         x = self.activation_layer(self.fc7(x))
#         x = self.activation_layer(self.fc8(x))
#         x = self.activation_layer(self.fc9(x))
#         x = self.fc10(x)
#         return x
