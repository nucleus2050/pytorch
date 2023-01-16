import torch


#autograd 自动求导

x = torch.randn(3, requires_grad=True)
y = x + 2

print(x)
print(y)

#损失函数： 真实y值与实际y值间的差值计算函数


