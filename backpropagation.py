import torch

#y = 2x
x = torch.tensor(1.0)
y = torch.tensor(2.0)

print("X:",x)
print("Y:",y)

# This is the parameter we want to optimize -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True) #真实测试数据

#公式： y = w * x
# forward pass to compute loss
y_predicted = w * x   #预测数据
print("y_predicted:", y_predicted)
loss = (y_predicted - y) ** 2 #算平方 需要保证损失函数最后得到的值是正值 ?
print("loss:", loss)

# backward pass to compute gradient dLoss/dw
loss.backward()
print("w.grad:", w.grad)

# update weights
# next forward and backward pass...

# continue optimizing: #优化器
# update weights, this operation should not be part of the computational graph
with torch.no_grad():
    w -= 0.01 * w.grad
    print("w:", w)
# don't forget to zero the gradients
w.grad.zero_()