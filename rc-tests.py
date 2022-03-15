import pickle
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn, optim
import statistics

from reservoir import Reservoir
from custom_dataset import CustomImageDataset, LongMatchOrNoGame
from shape_gen import long_match_or_no

"""

Reservoir computing tests

"""

BATCH_SIZE = 256

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

# Note: got to > 80% in epoch 3 with 5000, discriminating 2 images of 3 x 28 x 28
RESERVOIR_OUTPUT_LENGTH = 5_000

"""
Sequential was originally:
torch.nn.Linear(natural_size,  natural_size // 10),
            torch.nn.ReLU(),
            torch.nn.Linear(natural_size // 10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
            torch.nn.Softmax(dim=1)
"""

class RNNOut(nn.Module):
    def __init__(self):
        super().__init__()
        natural_size = RESERVOIR_OUTPUT_LENGTH + INPUT_SIZE
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(natural_size,  natural_size // 10),
            torch.nn.ReLU(),
            torch.nn.Linear(natural_size // 10,  16),
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

    reservoir = Reservoir(density=.0002, Nu=INPUT_SIZE, Nx=RESERVOIR_OUTPUT_LENGTH, batch_size=BATCH_SIZE)

    learning_rates = {
        0: 1,
        4: .1,
        10: .05,
        15: .01,
        20: .005,
        25: .001,
    }

    num_epochs = 30

    # Should we regenerate data after each epoch
    change_data = True

    # Should the reservoir be cleared after every batch
    clear_reservoir = False

    # NOTE: On macbook, I changed line 160 in torch/storage.py to return torch.load(io.BytesIO(b), map_location="cpu") - used to not have "map location" bit
    dataset = LongMatchOrNoGame('long_match_or_no')
    train_size = int(.8 * len(dataset))  # .8 for 80%
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    if len(dataset) % BATCH_SIZE != 0:
        print("Warning: last batch size will be inconsistent and will thus be ignored.")
    
    print(f"Num pics: {dataset.num_pics}")

    all_val_accuracies = []
    all_train_accuracies = []

    for epoch in range(num_epochs):

        # Switch lr according to curve
        if epoch in learning_rates:
            optimizer = torch.optim.SGD(recognizer.parameters(), lr=learning_rates[epoch]) 

        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

        train_losses = []
        val_losses = []
        accuracies = []
        val_accuracies = []

        print(f"Epoch #{epoch+1}")
        show_batch = True

        # Train
        recognizer.train()
        for batch, (X, y) in enumerate(train_loader):

            X = X.to(device)
            y = y.to(device).float()

            if len(X) != BATCH_SIZE:
                continue

            # Select images that are memories
            memories = []
            n = X.shape[1]  # how many images are there per example
            for i in range(n-1):  # n-1 because last one is fed directly to network
                memories.append(X[:, i, :, :, :])

            # Flatten memories
            for i in range(n-1):
                memories[i] = torch.flatten(memories[i], start_dim=1, end_dim=3)

            bsize = X.shape[0]
            reservoir_outputs = torch.zeros((bsize, RESERVOIR_OUTPUT_LENGTH))

            sensory_input = X[:, n-1, :, :]  # n-1 is last image, remember
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)

            # Reservoir acts on full batch at a time
            for i in range(n-1):
                reservoir.evolve(memories[i])
            reservoir_outputs = reservoir.get_states()
            
            if clear_reservoir:
                reservoir.clear()

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

        recognizer.eval()
        for batch, (X, y) in enumerate(val_loader):

            X = X.to(device)
            y = y.to(device).float()

            if len(X) != BATCH_SIZE:
                continue

            # Select images that are memories
            memories = []
            n = X.shape[1]  # how many images are there per example
            for i in range(n-1):  # n-1 because last one is fed directly to network
                memories.append(X[:, i, :, :, :])

            # Flatten memories
            for i in range(n-1):
                memories[i] = torch.flatten(memories[i], start_dim=1, end_dim=3)

            bsize = X.shape[0]
            reservoir_outputs = torch.zeros((bsize, RESERVOIR_OUTPUT_LENGTH))

            sensory_input = X[:, n-1, :, :]  # n-1 is last image, remember
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)

            # Reservoir acts on full batch at a time
            for i in range(n-1):
                reservoir.evolve(memories[i])
            reservoir_outputs = reservoir.get_states()
            
            if clear_reservoir:
                reservoir.clear()

            # Height and width are weird for the matrix multiplication
            # Need to put it back to normal here
            reservoir_outputs = torch.transpose(reservoir_outputs, 1, 0)

            total_input = torch.cat((reservoir_outputs, flattened), 1).float()

            pred = recognizer(total_input)

            loss = loss_fn(pred, y)
            val_losses.append(loss.item())
            
            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            val_accuracies.append(batch_acc.item())

            if show_batch is True:
                print(f"val batch {batch} acc: {batch_acc.item()}")
            
        current_loss = statistics.mean(train_losses)
        total_acc = statistics.mean(accuracies)
        val_current_loss = statistics.mean(val_losses)
        val_total_acc = statistics.mean(val_accuracies)
        print(f"Train loss: {current_loss}")
        print(f"Train accuracy: {total_acc}")
        print(f"Val loss: {val_current_loss}")
        print(f"Val accuracy: {val_total_acc}")
        all_val_accuracies.append(val_total_acc)
        all_train_accuracies.append(total_acc)

        reservoir.clear()

        if change_data:
            long_match_or_no(n=dataset.num_pics)
            dataset = LongMatchOrNoGame('long_match_or_no')
            train_size = int(.8 * len(dataset))  # .8 for 80%
            val_size = len(dataset) - train_size
            train, val = random_split(dataset, [train_size, val_size])
    
    to_dump = {'num_pics': dataset.num_pics, 'val_accuracies': all_val_accuracies,
                    'train_accuracies': all_train_accuracies, 'lrs': learning_rates}
    pickle.dump(open('rc_test_header.pkl', 'wb'), to_dump)
