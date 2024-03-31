from torch.utils.data import Dataset
import torch


# 定义一个 PyTorch 数据集
class CoordDataset(Dataset):
    def __init__(self, inputs, coords):
        # 确保 inputs 和 coords 都是 torch.float32 类型
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.coords = torch.tensor(coords, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.coords[idx]
