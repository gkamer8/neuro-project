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


"""

Use data to train autoencoder and save

"""

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
TOTAL_INPUTS = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

device = "cuda"

example_directory = "generated"

MEMORY_REPRESENTATION = 128

class MemoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding_layers = torch.nn.Sequential(
            nn.Linear(TOTAL_INPUTS, MEMORY_REPRESENTATION),
            torch.nn.ReLU(),
            nn.Linear(MEMORY_REPRESENTATION, MEMORY_REPRESENTATION),
        )

        self.decoding_layers = torch.nn.Sequential(
            nn.Linear(MEMORY_REPRESENTATION, MEMORY_REPRESENTATION),
            torch.nn.ReLU(),
            nn.Linear(MEMORY_REPRESENTATION, TOTAL_INPUTS),
        )
    
    def forward(self, x):

        encoded = self.encode(x)
        return self.decode(encoded)

    def encode(self, x):
        return self.encoding_layers(x)

    def decode(self, x):
        return self.decoding_layers(x)

if __name__ == '__main__':

    model = MemoryEncoder().to(device)

    learning_rates = {
        0: 1,
        30: 50,
        80: 35,
        150: 25,
        250: 15,
        350: 5,
        450: 3,
        500: 1,
        600: .5,
        700: .2,
        800: .1,
        900: .05,
        1000: .01,
        1100: .005,
        1200: .0005
    }

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=100)    

    num_epochs = 1500

    dataset = CustomImageDataset('generated')
    train_size = int(.8 * len(dataset))  # .8 for 80%
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    # Note: batch size is multiplied by three
    train_loader = DataLoader(train, batch_size=128, shuffle=False)
    val_loader = DataLoader(val, batch_size=128, shuffle=False)

    # EXAMINE EXISTING ENCODER
    examine = True

    if examine:
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        to_view = 3
        for batch, (X, y) in enumerate(val_loader):
            # Note: we're ignoring y for the autoencoder
            X = X.to(device)

            flattened = torch.flatten(X, start_dim=0, end_dim=1)

            flattened = flattened.view(-1, TOTAL_INPUTS).to(device)
            encoded = encoder.encode(flattened)
            decoded = encoder.decode(encoded)

            for i in range(to_view):
                print("Encoded shape: " + str(encoded[0].shape))
                plt.imshow(decoded[i].cpu().reshape(INPUT_SHAPE).permute(1, 2, 0).detach().numpy())
                plt.show()

                plt.imshow(flattened[i].cpu().reshape(INPUT_SHAPE).permute(1, 2, 0).detach().numpy())
                plt.show()
            break

    for epoch in range(num_epochs):

        # Switch lr according to curve
        if epoch in learning_rates:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[epoch]) 

        train_losses = []

        print(f"Epoch #{epoch+1}")

        # Train
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            # Note: we're ignoring y for the autoencoder
            X = X.to(device)

            # Flattening takes the three images normally used for memory task
            # And makes them three separate data poitns
            # So 64 examples becomes 64*3 = 192
            flattened = torch.flatten(X, start_dim=0, end_dim=1)

            flattened = flattened.view(-1, TOTAL_INPUTS).to(device)
            pred = model(flattened)

            """
            if epoch == num_epochs - 1 and batch < 5:
                plt.imshow(pred[0].cpu().reshape(INPUT_SHAPE).permute(1, 2, 0).detach().numpy())
                plt.show()

                plt.imshow(flattened[0].cpu().reshape(INPUT_SHAPE).permute(1, 2, 0).detach().numpy())
                plt.show()"""

            loss = loss_fn(pred, flattened)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        current_loss = statistics.mean(train_losses)
        print(f"Train loss: {current_loss}")

    pickle.dump(model, open("encoder.pkl", 'wb'))
