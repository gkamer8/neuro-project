import pickle
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn, optim
import statistics

from reservoir import Reservoir
from custom_dataset import CustomImageDataset

"""

Reservoir computing tests

"""

BATCH_SIZE = 256

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

# Note: got to > 80% in epoch 3 with 5000, discriminating 2 images of 3 x 28 x 28
RESERVOIR_OUTPUT_LENGTH = 3_000

class RNNOut(nn.Module):
    def __init__(self):
        super().__init__()
        natural_size = RESERVOIR_OUTPUT_LENGTH + INPUT_SIZE
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(natural_size,  natural_size // 10),
            torch.nn.ReLU(),
            torch.nn.Linear(natural_size // 10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        return prediction

device = "cpu"

if __name__ == '__main__':

    recognizer = RNNOut().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(recognizer.parameters(), lr=.1)

    learning_rates = {
        0: 1,
        5: .5,
        10: .1,
        20: .05,
        30: .01,
        40: .005,
    }

    num_epochs = 50

    # NOTE: On macbook, I changed line 160 in torch/storage.py to return torch.load(io.BytesIO(b), map_location="cpu") - used to not have "map location" bit
    dataset = CustomImageDataset('generated')
    train_size = int(.8 * len(dataset))  # .8 for 80%
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    for epoch in range(num_epochs):

        # Switch lr according to curve
        if epoch in learning_rates:
            optimizer = torch.optim.SGD(recognizer.parameters(), lr=learning_rates[epoch]) 

        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

        train_losses = []
        accuracies = []

        print(f"Epoch #{epoch+1}")
        show_batch = True

        # Train
        recognizer.train()
        for batch, (X, y) in enumerate(train_loader):

            X = X.to(device)
            y = y.to(device).float()

             # Select images that are memories
            memory1 = X[:, 0, :, :, :]
            memory2 = X[:, 1, :, :, :]

            # Flatten memories
            memory1 = torch.flatten(memory1, start_dim=1, end_dim=3)
            memory2 = torch.flatten(memory2, start_dim=1, end_dim=3)

            bsize = X.shape[0]
            reservoir_outputs = torch.zeros((bsize, RESERVOIR_OUTPUT_LENGTH))

            # NOTE: This code puts sensory input into reservoir
            sensory_input = X[:, 2, :, :]
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)

            # Reservoir acts on full batch at a time
            reservoir = Reservoir(Nu=INPUT_SIZE, Nx=RESERVOIR_OUTPUT_LENGTH, batch_size=len(X))
            reservoir.evolve(memory1)
            reservoir.evolve(memory2)
            # reservoir.evolve(flattened)
            reservoir_outputs = reservoir.get_states()

            # Height and width are weird for the matrix multiplication
            # Need to put it back to normal here
            reservoir_outputs = torch.transpose(reservoir_outputs, 1, 0)

            total_input = torch.cat((reservoir_outputs, flattened), 1).float()

            pred = recognizer(total_input)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            accuracies.append(batch_acc.item())

            if show_batch is True:
                print(f"batch {batch} acc: {batch_acc.item()}")
            
        current_loss = statistics.mean(train_losses)
        total_acc = statistics.mean(accuracies)
        print(f"Train loss: {current_loss}")
        print(f"Train accuracy: {total_acc}")