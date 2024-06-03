import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(4, 1, bias=True)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


forward_hook_dict = {}
backward_hook_dict = {}

# 钩子函数
# def forward_hook(module, input, output):
#     # print(11111111)
#     # print(f"Forward hook: {module}")
#     # print(f"Input: {input}")
#     # print(f"Output: {output}")
#     # pass
#     # forward_hook = {}
#     forward_hook_dict[module] = output


# def backward_hook(module, grad_input, grad_output):
#     # print(2222222222)
#     # print(f"Backward hook: {module}")
#     # print(f"Grad Input: {grad_input}")
#     # print(f"Grad Output: {grad_output}")
#     # pass
#     # backward_hook = {}
#     backward_hook_dict[module] = grad_output


def forward_hook(name):
    def f_hook(module, input, output):
        forward_hook_dict[name] = output
    return f_hook

def backward_hook(name):
    def b_hook(module, grad_input, grad_output):
        backward_hook_dict[name] = grad_output
    return b_hook


# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 注册钩子
hooks = []
# for layer in model.children():
#     print(layer)
#     print("----")
#     hooks.append(layer.register_forward_hook(forward_hook))
#     hooks.append(layer.register_backward_hook(backward_hook))


# build the index of name of each parameters and its coresponding weights:
# param_index = {}
# for name, param in model.named_parameters():
#     param_index[name] = param
# print(param_index)
# exit()

# the nework is f(), and the input is x, the output is y, how do I get the inverse f()? aka. f'()?


for name, module in model.named_modules():
    print(name)
    if hasattr(module, 'weight'):
        # print(module)
        # print("----")
        # hooks.append(module.register_forward_hook(forward_hook))
        # hooks.append(module.register_backward_hook(backward_hook))
        module.register_forward_hook(forward_hook(name))
        module.register_backward_hook(backward_hook(name))

# 训练数据
# inputs = Variable(torch.randn(2))
# targets = Variable(torch.randn(1))

inputs = torch.FloatTensor([1,2])
targets = torch.FloatTensor([0])

# 训练步骤
for step in range(2):  # 简单训练 3 个步骤
    optimizer.zero_grad()

    #print(model.fc2.weight.data.T)
    #print(model.fc2.weight)
    #print(model.fc2.weight.grad)
    #print("------")
    #exit()
    
    outputs = model(inputs)
    #print("out:", outputs)
    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    print(step)
    #print(forward_hook_dict)
    for name, output in forward_hook_dict.items():
        print(name, output)
    # exit()
    # print("------")

    #print(1111)
    # print(model.fc2.weight.grad)
    # print("------")
    print(backward_hook_dict)
    # for name, output in backward_hook_dict.items():
    #     print(name, output)
    #print("------")
    #exit()
    
    # # 清除钩子
    # for hook in hooks:
    #     hook.remove()
    forward_hook_dict.clear()
    backward_hook_dict.clear()

    # 重新注册钩子（如果需要在每个step都注册）
    # hooks = []
    # for layer in model.children():
    #     hooks.append(layer.register_forward_hook(forward_hook))
    #     hooks.append(layer.register_backward_hook(backward_hook))

    print(f"Step {step + 1} completed.\n")

print("Training completed.")






# # Write the code below
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define a simple neural network
# class QuadraticModel(nn.Module):
#     def __init__(self):
#         super(QuadraticModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(1, 2, bias=False),
#             #nn.Linear(1, 1)
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x

# # Initialize the model, loss function, and optimizer
# model = QuadraticModel()

# # print(model)
# # for n, par in model.named_parameters():
# #     print(n, par.shape)
# # exit()

# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # Training data
# inputs = torch.randn(1, 1)
# #targets = 2 * inputs.pow(2) + 3 * inputs + 1
# #targets = inputs.pow(2)
# targets = inputs

# # Training steps
# for epoch in range(100):
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     print("inputs:", inputs)
#     print("model_weights:", model.model[0].weight.data) # grad=True
#     print("------")
#     print("caculate_output:", inputs * model.model[0].weight.data)
#     print("output:", outputs)
#     print("------")
#     #outputs.backward(torch.ones_like(outputs)
#     #print(outputs)
#     #print(loss)
#     loss.backward()
#     optimizer.step()
#     #optimizer.zero_grad()
#     print(model.model[0].weight.grad)
#     print("------")
#     exit()

#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# print("Training completed.")


