import torch

x = torch.empty(1)
print(x)

x = torch.empty(2,3)
print(x)

x = torch.empty(2,3,1) #从后往前推,最里层括号只有一个元素,倒数第二层括号中有三个元素,倒数第三层括号有两个元素
print(x)

y = torch.rand(2, 2)
x = torch.rand(2, 2)
z = x + y

print("x:",x)
print("Y:",y)
print("Z:",z)



