import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define a class to handle hook registration and data capture
class HookManager:
    def __init__(self):
        self.hooks = []
        self.layer_outputs = {}

    def forward_hook(self, module_name, model_name):
        def hook(module, input, output):
            print(f"Forward pass through {model_name}.{module_name}")
            self.layer_outputs[f"{model_name}.{module_name}"] = output
        return hook

    def register_hook(self, model, model_name, hook_type):
        if hook_type == "forward":
            for module_name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Register hooks only on leaf modules
                    handle = module.register_forward_hook(self.forward_hook(module_name, model_name))
                    self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

# Create the network and a hook manager
net = SimpleNet()
hook_manager = HookManager()

# Register hooks
hook_manager.register_hook(net, "SimpleNet", "forward")

# Dummy data to pass through the network
input_data = torch.randn(1, 1, 28, 28)
output = net(input_data)


for i in hook_manager.hooks:
    print(i)
exit()

# Removing hooks after use
hook_manager.remove_hooks()

# Display captured outputs
for name, output in hook_manager.layer_outputs.items():
    print(f"{name} output size: {output.size()} and some data: {output.data}")
