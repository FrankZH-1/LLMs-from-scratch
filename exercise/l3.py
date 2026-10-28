import torch

x = torch.tensor(2.0)
w = torch.tensor(6.0, requires_grad=True)

y = w * x
loss = (y - 10) ** 2
loss.backward()
learning_rate = 0.1

print("y的预测" , y.item())
print("loss的值", loss.item())
print("损失对权重的梯度", w.grad.item())
with torch.no_grad():
		w -= learning_rate * w.grad

y = w * x
loss = (y - 10) ** 2
loss.backward()

print("y的预测" , y.item())
print("loss的值", loss.item())
print("损失对权重的梯度", w.grad.item())
