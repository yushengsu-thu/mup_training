import torch
import torch.nn as nn

def pre_hook_fn(name):
    def f_hook(module, input):
        #print(f"Pre-hook for {name}")
        # 修改輸入
        modified_input = (input[0] * 1.5,)  # 將輸入值增加50%
        return modified_input
    return f_hook

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化網絡
net = Net()

# 存儲所有的hooks
hooks = []

# 為每一層註冊pre-hook
for name, module in net.named_modules():
    print(name)
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        hook = module.register_forward_pre_hook(pre_hook_fn(name))
        hooks.append(hook)

# 創建輸入數據
input_data = torch.randn(1, 1, 32, 32)

# 前向傳播
output = net(input_data)

# 如果不再需要hooks，可以移除它們
for hook in hooks:
    hook.remove()

print("Forward pass completed")
