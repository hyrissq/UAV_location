# 定义更深的神经网络模型
import torch
import torch.nn as nn


def gaussian(alpha):
    return torch.exp(-1 * alpha.pow(2))


# class RBF(nn.Module):

#     def __init__(self, in_features, out_features):
#         super(RBF, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
#         self.basis_func = gaussian
#         self.reset_parameters()

#     def reset_parameters(self):
#         # initializing centers from normal distributions with 0 mean and std 1
#         nn.init.normal_(self.centers, 0, 1)
#         # initializing sigmas with 0 value
#         nn.init.constant_(self.log_sigmas, 0)

#     def forward(self, x):
#         """Forward pass"""
#         input = x
#         size = (input.size(0), self.out_features, self.in_features)
#         x = input.unsqueeze(1).expand(size)
#         c = self.centers.unsqueeze(0).expand(size)
#         distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(
#             self.log_sigmas
#         ).unsqueeze(0)
#         return self.basis_func(distances)


class RBF(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.centers, -1, 1)
        nn.init.uniform_(self.log_sigmas, -1, 1)

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * torch.exp(
            -self.log_sigmas
        ).unsqueeze(0)
        return torch.exp(-distances.pow(2))


class RBFNetwork(nn.Module):
    def __init__(self):
        in_features = 14
        hidden_features = 8  #
        out_features = 4

        super(RBFNetwork, self).__init__()
        self.rbf_layer = RBF(in_features, hidden_features)
        self.fc1 = nn.Linear(hidden_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)  # 新增层
        self.fc5 = nn.Linear(512, 512)  # 新增层
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 32)
        self.fc10 = nn.Linear(32, out_features)

        self.activation_layer = nn.ReLU()

    def forward(self, x):
        x = self.rbf_layer(x)
        x = self.activation_layer(self.fc1(x))
        x = self.activation_layer(self.fc2(x))
        x = self.activation_layer(self.fc3(x))
        x = self.activation_layer(self.fc4(x))  # 新增层
        x = self.activation_layer(self.fc5(x))  # 新增层
        x = self.activation_layer(self.fc6(x))
        x = self.activation_layer(self.fc7(x))
        x = self.activation_layer(self.fc8(x))
        x = self.activation_layer(self.fc9(x))
        x = self.fc10(x)
        return x


# class RBF(nn.Module):

#     def _init_(self):
#         input_dim = 14
#         out_dim = 4
#         num_centers = 100  #

#         super(RBF, self).__init__()  #

#         self.input_dim = input_dim
#         self.num_centers = num_centers
#         self.out_dim = out_dim
#         self.beta = 8  # 扩展常数
#         self.centers = [
#             np.random.uniform(0, 300, input_dim) for i in range(num_centers)
#         ]
#         self.W = np.random((self.num_centers, self.out_dim))

#     def basisfunc(self, c, d):
#         return np.exp(-self.beta * norm(c - d) ** 2)

#     def calcAct(self, input):
#         G = np.zeros((input.shape[0], self.num_centers), dtype=np.float)
#         for ci, c in enumerate(self.centers):
#             for xi, x in enumerate(input):
#                 G[xi, ci] = self.basisfunc(c, x)
#         return G

#     def train(self, input, output):
#         rnd_idex = np.random.permutation(input.shape[0])[: self.num_centers]
#         self.centers = [input[i, :] for i in rnd_idex]

#         # 计算RBF激活函数的值
#         G = self.calcAct(input)
#         self.W = np.dot(pinv(G), output)

#     def predict(self, input):
#         G = self.calcAct(input)
#         output = np.dot(G, self.W)
#         return output
