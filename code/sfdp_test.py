# train_model.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
import numpy as np

# Create a simple dataset
x = np.random.rand(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)
dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  # Increased batch size

# Define a simple neural network with dropout for sfdp
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.5)  # Sparse fast dropout simulation
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Initialize the model, loss, and optimizer
model = NeuralNetwork()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Initialize Accelerator
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Run training
train(dataloader, model, loss_fn, optimizer)
print("Training completed")

