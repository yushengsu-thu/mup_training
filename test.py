import torch
import torch.nn as nn

# 定義一個簡單的模型
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 定義一個函數來處理 hook 被呼叫時的操作
#def print_grad_hook(grad):
def print_grad_hook(module, input, output):
    print('Gradient:', grad)

#for n, p in model.named_modules():
#    print(n, p)

# 選擇模型中的第一個線性層，並在其權重上設置 hook
#hook = model[0].weight.register_hook(print_grad_hook)

for n, p in model.named_parameters():
    hook = p.register_hook(print_grad_hook)
    print(n)

# 進行一些運算來觸發 hook
input = torch.randn(1, 10)
output = model(input)
output.backward(torch.tensor([[0.1, 0.1]]))

# 如果你想移除這個 hook，可以呼叫 hook.remove()
hook.remove()
