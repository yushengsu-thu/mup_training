import torch
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        #self.conv1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.l1 = nn.Linear(1, 2)
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.l2(self.l1(x))

model = SimpleModel()

# 定义一个 hook 函数
def hook_fn(module, input, output):
    print('------')
    print(f"Module: {module}")
    print(f"Input: {input}")
    print(f"Output: {output}")
    print('------')

# 注册 forward hook
#hook = model.conv1.register_forward_hook(hook_fn)
list_ = []
# hook1 = model.linear.register_forward_hook(hook_fn)
# hook2 = model.linear.register_forward_hook(hook_fn)
for n, module in model.named_modules():
    #print(n)
    hook = module.register_forward_hook(hook_fn)
    list_.append(hook)

# 创建一个输入张量
#input_tensor = torch.randn(1, 1, 5, 5)
input_tensor = torch.randn(2,1)
print("==============")
print(input_tensor)
print("==============")

# 进行前向传播
output = model(input_tensor)

# 移除 hook
hook.remove()

