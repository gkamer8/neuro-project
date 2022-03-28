import matplotlib.pyplot as plt
from sympy import false, true
import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomImageDataset
import pickle

"""

Used to produce visualizations for the paper

"""

def show_dataset():
    # Show shape dataset
    # You're expected to manually save the pictures you want as pngs

    dataset_src = "generated"

    TO_SHOW = 1

    dataset = CustomImageDataset('generated')
    loader = DataLoader(dataset, batch_size=TO_SHOW, shuffle=True)

    # Just look at one batch
    for (X, y) in loader:
        for i in range(TO_SHOW):
            img1 = X[i][0]
            img2 = X[i][1]
            img3 = X[i][2]

            plt.imshow(img1.detach().cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(img2.detach().cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(img3.detach().cpu().permute(1, 2, 0))
            plt.show()
        break


def plot_encoder_training():

    SHOW_LRS = true

    train_info_file = "encoder_train_info.pkl"
    info = pickle.load(open(train_info_file, "rb"))

    y = info['val_losses']
    x = [i+1 for i in range(len(y))]

    # only look at first _ epochs
    crop_to_first = 100
    x = x[:crop_to_first]
    y = y[:crop_to_first]
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plt.plot(x, y)
    plt.title("Validation Loss of Memory Autoencoder")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.show()

def plot_recognizer_training():

    SHOW_LRS = false

    train_info_file = "recognizer_train_info.pkl"
    info = pickle.load(open(train_info_file, "rb"))

    y = info['val_accuracies']
    x = [i+1 for i in range(len(y))]

    # only look at first _ epochs
    crop_to_first = 10
    x = x[:crop_to_first]
    y = y[:crop_to_first]
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plt.plot(x, y)
    plt.title("Validation Accuracy of Delayed Match-to-Sample Game 1")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.show()


if __name__ == '__main__':

    # show_dataset()
    plot_recognizer_training()
