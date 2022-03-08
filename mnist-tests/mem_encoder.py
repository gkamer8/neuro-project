import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import copy
import matplotlib.pyplot as plt

import pickle

MEMORY_INPUT = 28 * 28
MEMORY_REPRESENTATION = 64

class MemoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding_layers = torch.nn.Sequential(
            nn.Linear(MEMORY_INPUT, MEMORY_REPRESENTATION),
            torch.nn.ReLU(),
            nn.Linear(MEMORY_REPRESENTATION, MEMORY_REPRESENTATION),
        )

        self.decoding_layers = torch.nn.Sequential(
            nn.Linear(MEMORY_REPRESENTATION, MEMORY_REPRESENTATION),
            torch.nn.ReLU(),
            nn.Linear(MEMORY_REPRESENTATION, MEMORY_INPUT),
        )
    
    def forward(self, x):
        encoded = self.encode(x)
        return self.decode(encoded)

    def encode(self, x):
        return self.encoding_layers(x)

    def decode(self, x):
        return self.decoding_layers(x)

if __name__ == '__main__':

    # Gordon's Note: I'm using this code on two machines - one with a GPU, one without
    # So I'm adding this special code not in the tutorial
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MemoryEncoder().to(device)

    params = model.parameters()
    optimizer = optim.SGD(params, lr=2e-1)

    loss = nn.MSELoss()

    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_data, [55000, 5000])

    validate = True
    see_digit = True  # In order to see digit, validate must be true

    nb_epochs = 35
    for epoch in range(nb_epochs):
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        val_loader = DataLoader(val, batch_size=32, shuffle=True)

        losses = list()
        accuracies = list()
        model.train()
        for batch_features, _ in train_loader:
            
            x = batch_features.view(-1, MEMORY_INPUT).to(device).to(device)

            x = x.to(device)
            b = x.size(0)

            l = model(x)  # predict
            J = loss(l, x)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()  # actually shift the params by gradients
            losses.append(J.item())

        print(f'Epoch {epoch + 1}, train loss: {torch.tensor(losses).mean():.2f}')

        # NOW EVALUATE
        if validate is False:
            continue

        losses = list()
        accuracies = list()
        model.eval()
        seen = False
        for batch_features, _ in val_loader:

            x = batch_features.view(-1, MEMORY_INPUT).to(device).to(device)

            x = x.to(device)
            b = x.size(0)

            l = model(x)  # predict
            
            with torch.no_grad():
                J = loss(l, x)

            if see_digit is True and not seen:
                x = x.cpu()
                l = l.detach().cpu()
                plt.imshow(x[0].view(28, 28))
                plt.show()
                plt.imshow(l[0].view(28, 28))
                plt.show()
                seen = True

            losses.append(J.item())

        print(f'Epoch {epoch + 1}', end=', ')
        print(f'validation loss: {torch.tensor(losses).mean():.2f}')

    # pickle.dump(model, open('mem_encoder.bin', 'wb'))
