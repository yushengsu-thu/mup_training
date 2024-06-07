# import torch
# import torch.nn as nn
# import torch.optim as optim

# # 假设我们有一个新的 tensor，用于替换 grad_input
# new_grad_input_tensor = torch.randn(1, 10)  # 示例 tensor

# def full_backward_hook_fn(module, grad_input, grad_output, is_before=True):
#     if is_before:
#         # 将 new_grad_input_tensor 替换 grad_input
#         #modified_grad_input = tuple([new_grad_input_tensor if g is not None else None for g in grad_input])
#         print(f"grad_input: {grad_input}", len(grad_input))
#         modified_grad_input = grad_input * 2
#         print("New grad_input before backward:", modified_grad_input)
#         return modified_grad_input
#     else:
#         # 在计算之后查看输出的梯度 (grad_output)
#         print("grad_output after backward:", grad_output)
#         return grad_output

# # 创建一个简单的线性模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(10, 10)
#         self.fc2 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

# model = SimpleModel()

# # 创建优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 定义一个损失函数
# criterion = nn.MSELoss()

# # 模拟一些训练数据
# inputs = torch.randn(5, 10, requires_grad=True)
# targets = torch.randn(5, 1)

# # 用于记录前向传播值的字典
# forward_pass_values = {'with_hook': [], 'without_hook': []}

# # 定义前向钩子函数
# def forward_hook_fn(module, input, output):
#     forward_pass_values['current'].append(output.detach().numpy())

# # 为模型的每一层注册前向钩子
# forward_hooks = []
# for layer in model.children():
#     hook = layer.register_forward_hook(forward_hook_fn)
#     forward_hooks.append(hook)

# # 为模型的每一层注册后向钩子
# backward_hooks = []
# for layer in model.children():
#     hook = layer.register_full_backward_hook(full_backward_hook_fn)
#     backward_hooks.append(hook)

# for epoch in range(2):  # 简单的训练循环
#     for input, target in zip(inputs, targets):
#         # 重置梯度
#         optimizer.zero_grad()
#         if input.grad is not None:
#             input.grad.zero_()

#         # 设置当前记录的类型为 'with_hook'
#         forward_pass_values['current'] = forward_pass_values['with_hook']

#         # 前向传播
#         output = model(input)
#         loss = criterion(output, target)

#         # 反向传播 - 使用修改后的 grad_input
#         loss.backward()

#         # 更新参数 - 使用修改后的梯度
#         optimizer.step()


#         #optimizer.zero_grad()
#         print(11111111)
#         print(forward_pass_values['current'])
#         exit()

#         # 再次前向传播和反向传播
#         optimizer.zero_grad()
#         if input.grad is not None:
#             input.grad.zero_()

#         # 设置当前记录的类型为 'without_hook'
#         forward_pass_values['current'] = forward_pass_values['without_hook']

#         output = model(input)
#         loss = criterion(output, target)

#         # 反向传播 - 使用原始的 grad_input
#         loss.backward()

#         # 更新参数 - 使用原始的梯度
#         optimizer.step()

#         # 打印结果以进行比较
#         print("Epoch:", epoch, "input.grad after original grad_input:", input.grad)

# # 打印前向传播值
# print("Forward pass values with hook modification:")
# print(forward_pass_values['with_hook'])
# print("Forward pass values without hook modification:")
# print(forward_pass_values['without_hook'])

# # 取消前向钩子
# for hook in forward_hooks:
#     hook.remove()

# # 取消后向钩子
# for hook in backward_hooks:
#     hook.remove()





from turtle import back
import torch
import torch.nn as nn
import torch.optim as optim

#from test_code import forward_hook

# # 假设我们有一个新的 tensor，用于替换 grad_input
# new_grad_input_tensor = torch.randn(10)  # 示例 tensor

# def full_backward_hook_fn(module, grad_input, grad_output, is_before=True):
#     if is_before:
#         # 将 new_grad_input_tensor 替换 grad_input
#         modified_grad_input = (new_grad_input_tensor,) + grad_input[1:]
#         #print("New grad_input before backward:", modified_grad_input)
#         return modified_grad_input
#     else:
#         #return grad_output
#         pass



# 创建一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = SimpleModel()

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义一个损失函数
criterion = nn.MSELoss()

# 模拟一些训练数据
inputs = torch.randn(5, 10, requires_grad=True)
targets = torch.randn(5, 1)

# 用于记录前向传播值的字典
#forward_pass_values = {'with_hook': [], 'without_hook': []}


forward_hooks = []
backward_hooks = []

# 假设我们有一个新的 tensor，用于替换 grad_input
#new_grad_input_tensor = torch.randn(10)  # 示例 tensor
def full_backward_hook_fn(module, grad_input, grad_output, is_before=True):
    if is_before:
        backward_hooks.append(grad_output)
        # 将 new_grad_input_tensor 替换 grad_input
        #modified_grad_input = (new_grad_input_tensor,) + grad_input[1:]
        #print(module)
        modified_grad_input = tuple()
        for i in range(len(grad_input)):
            modified_grad_input += (grad_input[i]*2, )
        #print("New grad_input before backward:", modified_grad_input)
        return modified_grad_input
    else:
        #return grad_output
        backward_hooks.append(grad_output)
        #pass

# 定义前向钩子函数
def forward_hook_fn(module, input, output):
    #forward_pass_values['current'].append(output.detach().numpy())
    forward_hooks.append(output)

# 为模型的每一层注册前向钩子
# forward_hooks = []
for layer in model.children():
    layer.register_forward_hook(forward_hook_fn)
    #forward_hooks.append(hook)

# 为模型的每一层注册后向钩子
# backward_hooks = []
for layer in model.children():
    layer.register_full_backward_hook(full_backward_hook_fn)
    #backward_hooks.append(hook)

# for epoch in range(5):  # 简单的训练循环
max_epoch = 5
epoch = 0
for input, target in zip(inputs, targets):
    epoch += 1
    if epoch >= max_epoch:
        break

    # 重置梯度
    optimizer.zero_grad()
    #if input.grad is not None:
    #    input.grad.zero_()

    # 设置当前记录的类型为 'with_hook'
    #forward_pass_values['current'] = forward_pass_values['with_hook']

    # 前向传播
    output = model(input)
    loss = criterion(output, target)

    # 反向传播 - 使用修改后的 grad_input
    loss.backward()

    # 更新参数 - 使用修改后的梯度
    optimizer.step()

    # print(forward_hooks)
    #print(backward_hooks)
    # for n, p in backward_hooks:
    #     print(f"{n}: {p}")
    #     print("---")
    #print("====")
    #print(forward_hooks)

    # optimizer.zero_grad()
    # print(forward_pass_values['current'])
    # exit()

    # # 再次前向传播和反向传播
    # optimizer.zero_grad()
    # if input.grad is not None:
    #     input.grad.zero_()

    # # 设置当前记录的类型为 'without_hook'
    # #forward_pass_values['current'] = forward_pass_values['without_hook']

    # output = model(input)
    # loss = criterion(output, target)

    # # 反向传播 - 使用原始的 grad_input
    # loss.backward()

    # # 更新参数 - 使用原始的梯度
    # optimizer.step()

    # # 打印结果以进行比较
    # #print("Epoch:", epoch, "input.grad after original grad_input:", input.grad)



print(backward_hooks)
exit()

# # 打印前向传播值
# print("Forward pass values with hook modification:")
# print(forward_pass_values['with_hook'])
# print("Forward pass values without hook modification:")
# print(forward_pass_values['without_hook'])



# 取消前向钩子
#for hook in forward_hooks:
#    hook.remove()

# 取消后向钩子
#for hook in backward_hooks:
#    hook.remove()
