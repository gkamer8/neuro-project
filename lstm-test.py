import pickle
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomImageDataset, LongMatchOrNoGame
import torch
from torch import nn, optim
import statistics

from shape_mem_encoder import MemoryEncoder
from shape_gen import long_match_or_no, left_right_match

"""

Train functional NN to perform memory task

"""

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

HIDDEN_SIZE = 250  # Essentially, memory representation

device = 'cpu'

class LSTMRecognizer(nn.Module):
    def __init__(self):
        super(LSTMRecognizer, self).__init__()
        self.lstm = nn.LSTM(
                input_size = INPUT_SIZE,
                hidden_size = HIDDEN_SIZE,
                num_layers = 1,
                batch_first=True
            )
        self.lin1 =  nn.Linear(HIDDEN_SIZE, 128)
        self.lin2 =  nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        rnn_out, _ = self.lstm(x)
        last_out = rnn_out[:, 2, :]

        first_lin = self.lin1(last_out)
        first_lin = self.relu(first_lin)

        out_layer = self.lin2(first_lin)
        out_layer = self.softmax(out_layer)
        return out_layer


if __name__ == '__main__':

    learning_rates = {
        0: 10,
        1: 5,
        2: 1,
        10: .5,
        20: 1e-4,
        30: 1e-10,
        40: 2,
        50: 1e-3
    }

    game_names = ['left or right', 'seen or not']
    # Note: when adding a game, make sure to change the data creation at the end of the epoch
    game = game_names[0]  # 'seen or not'

    total_pics_shown = 0
    output_length = 0
    if game == 'seen or not':
        dataset = LongMatchOrNoGame('long_match_or_no')
        total_pics_shown = dataset.num_pics
        output_length = 2
    elif game == 'left or right':
        dataset = CustomImageDataset('generated')
        total_pics_shown = 3
        output_length = 2

    lstm_recognizer = LSTMRecognizer()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(lstm_recognizer.parameters(), lr=1)

    num_epochs = 60

    show_batch = True

    change_data = True  # create new data after every epoch

    print(f"Playing {game} with sequence of length {total_pics_shown}.")

    train_size = int(.8 * len(dataset))  # .8 for 80%
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    all_train_accuracies = []
    all_train_losses = []

    for epoch in range(num_epochs):

        # Switch lr according to curve
        if epoch in learning_rates:
            optimizer = torch.optim.SGD(lstm_recognizer.parameters(), lr=learning_rates[epoch]) 

        train_loader = DataLoader(train, batch_size=256, shuffle=True)
        val_loader = DataLoader(val, batch_size=256, shuffle=True)

        train_losses = []
        accuracies = []

        print(f"Epoch #{epoch+1}")

        # Train
        lstm_recognizer.train()
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()

            total_input = X.flatten(start_dim=2, end_dim=4)

            output = lstm_recognizer(total_input)

            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            correctness_tensor = output.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            accuracies.append(batch_acc.item())

            if show_batch is True:
                print(f"batch {batch} acc: {batch_acc.item()}")

        current_loss = statistics.mean(train_losses)
        train_total_acc = statistics.mean(accuracies)
        print(f"Train loss: {current_loss}")
        print(f"Train accuracy: {train_total_acc}")

        all_train_accuracies.append(train_total_acc)
        all_train_losses.append(current_loss)

        val_losses = []
        val_accuracies = []

        # Validate
        """recognizer.eval()
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()

            # Select images that are memories
            memories = []
            for i in range(memories_to_load):
                memories.append(X[:, i, :, :, :])
                memories[i] = torch.flatten(memories[i], start_dim=1, end_dim=3)
                memories[i] = mem_encoder.encode(memories[i])

            # Flatten input
            # Note: memories_to_load would equal the index of the first non memory
            sensory_input = X[:, memories_to_load, :, :]
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)
            # Concatenate memories and input
            total_input = torch.cat(memories + [flattened], 1).float()

            pred = recognizer(total_input)

            loss = loss_fn(pred, y)
            val_losses.append(loss.item())
            
            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            val_accuracies.append(batch_acc.item())

        current_val_loss = statistics.mean(val_losses)
        val_total_acc = statistics.mean(val_accuracies)
        print(f"Val loss: {current_val_loss}")
        print(f"Val accuracy: {val_total_acc}")"""

        if change_data:
            
            if game == 'seen or not':
                long_match_or_no(n=dataset.num_pics)
                dataset = LongMatchOrNoGame('long_match_or_no')
            elif game == 'left or right':
                left_right_match()
                dataset = CustomImageDataset('generated')

            train_size = int(.8 * len(dataset))  # .8 for 80%
            val_size = len(dataset) - train_size
            train, val = random_split(dataset, [train_size, val_size])

    pickle.dump(lstm_recognizer, open("lstmfunctional.pkl", 'wb'))

    to_dump = {
            'game': game,
            'total_pics': total_pics_shown,
            'train_accuracies': all_train_accuracies,
            'train_losses': all_train_losses,
            'lrs': learning_rates
        }
    pickle.dump(to_dump, open('lstm_test_header.pkl', 'wb'))
