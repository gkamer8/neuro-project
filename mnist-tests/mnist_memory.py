from cmath import inf
import pickle
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from mem_encoder import MemoryEncoder  # needed to load mnist model
import numpy as np
import copy
import torch.utils.data as Data
import matplotlib.pyplot as plt


mnist_model = pickle.load(open('mem_encoder.bin', 'rb'))

MEMORY_LENGTH = 5
MEMORY_SIZE = 64

SENSORY_INPUT_SIZE = 28 * 28

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

class FunctionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(SENSORY_INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH,  MEMORY_SIZE * (MEMORY_LENGTH+1)),
            torch.nn.ReLU(),
            torch.nn.Linear(MEMORY_SIZE * (MEMORY_LENGTH+1), MEMORY_SIZE * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(MEMORY_SIZE * 2, MEMORY_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(MEMORY_SIZE, 2),
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

recognizer = FunctionalNN().to(device)
# Initialize weights and biases

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])

loss = nn.MSELoss()
mnist_model = mnist_model.to(device)

params = recognizer.parameters()
optimizer = optim.Adam(params, lr=1e-3)

validate = True
target_epoch = inf  # Epoch after which to change learning rate i.e. 10
lower_lr = None  # New, lower learning rate i.e. 1e-4

torch.autograd.set_detect_anomaly(True)

nb_epochs = 1
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    recognizer.train()

    train_loader = DataLoader(train, batch_size=256, shuffle=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=True)

    if epoch >= target_epoch and lower_lr is not None:
        optimizer = optim.Adam(recognizer.parameters(), lr=lower_lr)

    num_batch = 0
    for batch in train_loader:
        x, y = batch
        b = x.size(0)  # current batch size, length of y (32 by default)
        x = x.view(b, -1).to(device)
        y = y.to(device)

        # change 2 to 1 when using cross entropy loss
        match_or_no = torch.zeros(b, 2, dtype=torch.int64).to(device).float()  # 0 for no match; 1 for match

        memory_reps = mnist_model.encode(x).to(device)
        total_memory_reps = torch.zeros(b, MEMORY_SIZE * MEMORY_LENGTH).to(device)

        # CONSTRUCT MEMORY FOR EACH X
        mem_rep_index = 0
        memory_bank = MemoryBank()
        for xi in x:
            memory_bank.add(memory_reps[mem_rep_index])
            total_memory_reps[mem_rep_index] = copy.deepcopy(memory_bank).bank.reshape(-1)
            mem_rep_index += 1

        total_input = torch.cat((x, total_memory_reps), 1).float()

        # Construct match_or_no from y
        match_index = 0
        true_memory_contents = [-1 for _ in range(MEMORY_LENGTH)]
        for yi in y:
            if yi.item() in true_memory_contents:
                match_or_no[match_index] = torch.tensor([1., 0.])  # for cross entropy loss, just 1
            else:
                match_or_no[match_index] = torch.tensor([0., 1.])
            match_index += 1
            true_memory_contents = [yi.item()] + true_memory_contents[:len(true_memory_contents)-1]

        # match_or_no = torch.reshape(match_or_no, (-1,))

        pred = recognizer(total_input)
        J = loss(pred, match_or_no)

        # Backpropagation
        optimizer.zero_grad()
        J.backward()

        optimizer.step()

        num_batch += 1

        losses.append(J.item())

        correct = 0
        total = 0
        for i, yhat in enumerate(pred):
            total += 1
            if match_or_no[i].argmax(dim=-1) == yhat.argmax(dim=-1):
                correct += 1
        accuracies.append(correct / total)
                
        # accuracies.append(match_or_no.eq(pred.detach().argmax(dim=1)).float().mean())  # for cross entropy loss

    print(f'Epoch {epoch + 1}, train loss: {torch.tensor(losses).mean():.2f}')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')

    if validate is False:
        continue

    recognizer.eval()
    for batch in val_loader:
        x, y = batch
        b = x.size(0)  # current batch size, length of y (32 by default)
        x = x.view(b, -1).to(device)
        y = y.to(device)

        match_or_no = torch.empty(b, 2, dtype=torch.int64).to(device)  # 0 for no match; 1 for match

        memory_reps = mnist_model.encode(x).to(device)
        total_memory_reps = torch.empty(b, MEMORY_SIZE * MEMORY_LENGTH).to(device)

        # CONSTRUCT MEMORY FOR EACH X
        mem_rep_index = 0
        memory_bank = MemoryBank()
        for xi in x:
            memory_bank.add(memory_reps[mem_rep_index])
            total_memory_reps[mem_rep_index] = copy.deepcopy(memory_bank.bank.reshape((-1,)))
            mem_rep_index += 1

        total_input = torch.cat((x, total_memory_reps), 1)

        # Construct match_or_no from y
        match_index = 0
        true_memory_contents = [-1 for _ in range(MEMORY_LENGTH)]
        for yi in y:
            if yi.item() in true_memory_contents:
                match_or_no[match_index] = torch.tensor([1., 0.])  # for cross entropy loss, just 1
            else:
                match_or_no[match_index] = torch.tensor([0., 1.])
            match_index += 1
            true_memory_contents = [yi.item()] + true_memory_contents[:len(true_memory_contents)-1]

        with torch.no_grad():
            pred = recognizer(total_input)

        J = loss(pred, match_or_no)

        losses.append(J.item())
        correct = 0
        total = 0
        for i, yhat in enumerate(pred):
            total += 1
            if match_or_no[i].argmax(dim=-1) == yhat.argmax(dim=-1):
                correct += 1
        accuracies.append(correct / total)

    print(f'Epoch {epoch + 1}', end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')

