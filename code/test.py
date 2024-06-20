# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()

# Dummy dataset
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple model
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare everything with accelerator
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training loop
for epoch in range(10):
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete")

