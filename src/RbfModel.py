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
        self.linear_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.rbf_layer(x)
        x = self.linear_layer(x)
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
