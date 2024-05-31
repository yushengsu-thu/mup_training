import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 钩子函数
def forward_hook(module, input, output):
    print(11111111)
    print(f"Forward hook: {module}")
    print(f"Input: {input}")
    print(f"Output: {output}")
    pass

def backward_hook(module, grad_input, grad_output):
    print(2222222222)
    print(f"Backward hook: {module}")
    print(f"Grad Input: {grad_input}")
    print(f"Grad Output: {grad_output}")
    pass

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 注册钩子
hooks = []
for layer in model.children():
    hooks.append(layer.register_forward_hook(forward_hook))
    hooks.append(layer.register_backward_hook(backward_hook))

# 训练数据
inputs = Variable(torch.randn(10))
targets = Variable(torch.randn(1))

# 训练步骤
for step in range(3):  # 简单训练 3 个步骤
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    # 清除钩子
    for hook in hooks:
        hook.remove()

    # 重新注册钩子（如果需要在每个step都注册）
    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(forward_hook))
        hooks.append(layer.register_backward_hook(backward_hook))

    print(f"Step {step + 1} completed.\n")

print("Training completed.")

