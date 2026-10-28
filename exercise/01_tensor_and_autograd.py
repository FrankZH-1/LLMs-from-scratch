"""练习 01：张量、矩阵乘法、自动求导与一次权重更新。

合并原 l0.py 至 l3.py，保留最后的梯度累积现象作为下一课的起点。
运行：.venv/bin/python exercise/01_tensor_and_autograd.py
"""

import numpy as np
import torch


print("一、NumPy：形状、均值和矩阵乘法")
np_x = np.arange(24).reshape(2, 4, 3)
np_w = np.ones((3, 5))
np_y = np_x @ np_w

print("输入：", np_x)
print("沿分量轴求均值并保留轴：", np_x.mean(axis=2, keepdims=True))
print("输出形状：", np_y.shape)
print("第一个 token 的输出向量：", np_y[0, 0])


print("\n二、PyTorch：用浮点张量完成相同运算")
token_vectors = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
projection = torch.ones((3, 5), dtype=torch.float32)
projected = token_vectors @ projection

print("输出形状：", projected.shape)
print("第一个 token 的输出向量：", projected[0, 0])


print("\n三、标量模型：预测、平方误差与自动求导")
x = torch.tensor(2.0)
w = torch.tensor(6.0, requires_grad=True)
target = 10.0
learning_rate = 0.1

y = w * x
loss = (y - target) ** 2
loss.backward()

print("更新前权重：", w.item())
print("更新前预测：", y.item())
print("更新前损失：", loss.item())
print("更新前损失对权重的梯度：", w.grad.item())


print("\n四、更新一次权重，重新计算，并观察梯度累积")
with torch.no_grad():
    w -= learning_rate * w.grad

y = w * x
loss = (y - target) ** 2
# 此处保留原练习的行为：没有清除旧梯度就再次反向传播。
# 下一项练习将解释并处理累积梯度，不将当前现象视为已掌握。
loss.backward()

print("更新后权重：", w.item())
print("更新后预测：", y.item())
print("更新后损失：", loss.item())
print("累积后的梯度：", w.grad.item())
