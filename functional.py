import enum
import pickle
import matplotlib.pyplot as plt
import os
from sympy import I
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomImageDataset
import torch
from torch import nn, optim
import statistics

from shape_mem_encoder import MemoryEncoder

"""

Train functional NN to perform memory task

"""

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

MEMORY_SIZE = 128

MEMORY_LENGTH = 2

device = 'cuda'

class FunctionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH,  MEMORY_SIZE * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(MEMORY_SIZE * 2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        return prediction

class MemoryBank():
    def __init__(self) -> None:
        self.rows = MEMORY_LENGTH
        self.cols = MEMORY_SIZE
        self.bank = torch.zeros(self.rows, self.cols, dtype=torch.float).to(device)

    # Moves each row forward one position, placing new memory at 0
    # x should be a tensor
    @torch.no_grad()
    def add(self, x):
        new_mem = [x] + [self.bank[i] for i in range(self.rows-1)]
        self.bank = torch.vstack(new_mem)

if __name__ == '__main__':
    
    recognizer = FunctionalNN().to(device)

    encoder_file = 'encoder-.011.pkl'
    mem_encoder = pickle.load(open(encoder_file, 'rb'))

    learning_rates = {
        0: 1
    }

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(recognizer.parameters(), lr=100)

    num_epochs = 10

    dataset = CustomImageDataset('generated')
    train_size = int(.8 * len(dataset))  # .8 for 80%
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    epoch_val_accuracies = []

    for epoch in range(num_epochs):

        # Switch lr according to curve
        if epoch in learning_rates:
            optimizer = torch.optim.SGD(recognizer.parameters(), lr=learning_rates[epoch]) 

        train_loader = DataLoader(train, batch_size=128, shuffle=True)
        val_loader = DataLoader(val, batch_size=128, shuffle=True)

        train_losses = []
        accuracies = []
        val_accuraces = []


        print(f"Epoch #{epoch+1}")

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

            # Run memories through auto encoder
            memory1 = mem_encoder.encode(memory1)
            memory2 = mem_encoder.encode(memory2)

            # Flatten input
            sensory_input = X[:, 2, :, :]
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)
            # Concatenate memories and input
            total_input = torch.cat((memory1, memory2, flattened), 1).float()

            pred = recognizer(total_input)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            accuracies.append(batch_acc.item())

        # Validate
        recognizer.eval()
        for batch, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device).float()
            
            # Select images that are memories
            memory1 = X[:, 0, :, :, :]
            memory2 = X[:, 1, :, :, :]

            # Flatten memories
            memory1 = torch.flatten(memory1, start_dim=1, end_dim=3)
            memory2 = torch.flatten(memory2, start_dim=1, end_dim=3)

            # Run memories through auto encoder
            memory1 = mem_encoder.encode(memory1)
            memory2 = mem_encoder.encode(memory2)

            # Flatten input
            sensory_input = X[:, 2, :, :]
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)
            # Concatenate memories and input
            total_input = torch.cat((memory1, memory2, flattened), 1).float()

            pred = recognizer(total_input)

            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            val_accuraces.append(batch_acc.item())

        current_loss = statistics.mean(train_losses)
        total_acc = statistics.mean(accuracies)
        total_val_acc = statistics.mean(val_accuraces)
        print(f"Train loss: {current_loss}")
        print(f"Train accuracy: {total_acc}")
        print(f"Val   accuracy: {total_val_acc}")
        epoch_val_accuracies.append(total_val_acc)

    recognizer_train_info = {}
    recognizer_train_info['val_accuracies'] = epoch_val_accuracies
    recognizer_train_info['lrs'] = learning_rates
    recognizer_train_info['num_epochs'] = num_epochs

    pickle.dump(recognizer_train_info, open("recognizer_train_info.pkl", 'wb'))
    pickle.dump(recognizer, open("functionalnn.pkl", 'wb'))

