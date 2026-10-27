import numpy as np

# 用模拟数字表示一批文本的向量
x = np.arange(24).reshape(2, 4, 3)

print(x)

print(x.mean(axis=2, keepdims=True))
