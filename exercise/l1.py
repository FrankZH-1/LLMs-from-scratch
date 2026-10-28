import torch

x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
w = torch.ones((3,5), dtype=torch.float32)

y = x @ w
print("输出形状：", y.shape)
print("第一个 token 的输出向量：", y[0, 0])