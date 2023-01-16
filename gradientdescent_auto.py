import torch

# Here we replace the manually computed gradient with autograd

# Linear regression
# f = w * x

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model output
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

# 基本逻辑：
# 1.预测
# 2.计算损失值
# 3.反向传播
# 4.更新w值
for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)         #损失函数并不参与到反向传播的过程，只是作为衡量学习过程是否应该结束

    # calculate gradients = backward pass
    l.backward()                #这里发生了什么？调试发现这里将w.grad设置值，自动计算了梯度 通过运算符重载维护关联关系？
                                #通过偏导数计算了梯度（计算图）

    print("loss:", l)

    # update weights
    # w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad
        print("w:", w, "grad:", w.grad)

    # zero the gradients after updating
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')