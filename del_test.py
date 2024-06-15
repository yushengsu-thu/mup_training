# import torch
# import torch.nn as nn

# # 定义一个简单的模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         #self.conv1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
#         self.l1 = nn.Linear(1, 2)
#         self.l2 = nn.Linear(2, 1)

#     def forward(self, x):
#         return self.l2(self.l1(x))

# model = SimpleModel()

# # 定义一个 hook 函数
# def hook_fn(module, input, output):
#     print('------')
#     print(f"Module: {module}")
#     print(f"Input: {input}")
#     print(f"Output: {output}")
#     print('------')

# # 注册 forward hook
# #hook = model.conv1.register_forward_hook(hook_fn)
# list_ = []
# # hook1 = model.linear.register_forward_hook(hook_fn)
# # hook2 = model.linear.register_forward_hook(hook_fn)
# for n, module in model.named_modules():
#     #print(n)
#     hook = module.register_forward_hook(hook_fn)
#     list_.append(hook)

# # 创建一个输入张量
# #input_tensor = torch.randn(1, 1, 5, 5)
# input_tensor = torch.randn(2,1)
# print("==============")
# print(input_tensor)
# print("==============")

# # 进行前向传播
# output = model(input_tensor)

# # 移除 hook
# hook.remove()




# import torch
# import torch.nn as nn

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SimpleNN, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim, bias = False)

#     def forward(self, x):
#         x = self.fc(x)
#         return x

# # 設定輸入維度 n
# n = 10  # 這個數值可以是任何你需要的維度
# model = SimpleNN(input_dim=n, output_dim=5)  # 輸出維度設定為 5

# # 創建一個 n 維的輸入向量，所有元素都是1
# input_tensor = torch.ones((1, n))  # 批次大小設為 1

# # 執行模型的前向傳播
# output = model(input_tensor)
# print(output)
# print("=======")
# output = model(input_tensor)
# print(output)
# print("=======")
# output = model(input_tensor)
# print(output)





import torch
import torch.nn as nn

# 定义一个简单的线性层，输入特征为 5，输出特征为 3
linear_layer = nn.Linear(5, 3)

# 定义一个 hook 函数，该函数只会作用于权重
def weights_hook(grad, pass_in):
    print(f"{pass_in}")
    # 这里可以添加自定义的梯度处理逻辑
    print("Gradient of weights:", grad)
    # 返回新的梯度值，如果不需要修改梯度，可以直接返回原始梯度
    return grad

# 使用 register_full_backward_hook 注册 hook
# 注意，这里只注册了权重的 hook
weight_handle = linear_layer.weight.register_hook(weights_hook, "hello")

# 通过网络传递一些数据
input_tensor = torch.randn(1, 5)
output = linear_layer(input_tensor)

# 反向传播
output.sum().backward()
