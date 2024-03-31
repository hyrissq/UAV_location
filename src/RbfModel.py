# 定义更深的神经网络模型
import torch.nn as nn
import numpy as np
from scipy.linalg import norm, pinv


class RBF(nn.Module):

    def _init_(self, input_dim, num_centers, out_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.out_dim = out_dim
        self.beta = 8  # 扩展常数
        self.centers = [
            np.random.uniform(0, 300, input_dim) for i in range(num_centers)
        ]
        self.W = np.random((self.num_centers, self.out_dim))

    def basisfunc(self, c, d):
        return np.exp(-self.beta * norm(c - d) ** 2)

    def calcAct(self, input):
        G = np.zeros((input.shape[0], self.num_centers), dtype=np.float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(input):
                G[xi, ci] = self.basisfunc(c, x)
        return G

    def train(self, input, output):
        rnd_idex = np.random.permutation(input.shape[0])[: self.num_centers]
        self.centers = [input[i, :] for i in rnd_idex]

        # 计算RBF激活函数的值
        G = self.calcAct(input)
        self.W = np.dot(pinv(G), output)

    def predict(self, input):
        G = self.calcAct(input)
        output = np.dot(G, self.W)
        return output
