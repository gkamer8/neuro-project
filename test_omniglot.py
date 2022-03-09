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
from torchvision import datasets, transforms
import random

from shape_mem_encoder import MemoryEncoder
from functional import FunctionalNN

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

MEMORY_SIZE = 128

MEMORY_LENGTH = 2

device = 'cuda'

if __name__ == '__main__':

    functional_source = 'functionalnn.pkl'
    functional = pickle.load(open(functional_source, 'rb'))

    mem_encoder_source = 'encoder-.011.pkl'
    mem_encoder = pickle.load(open(mem_encoder_source, 'rb'))

    # NOTE: ALL DATA IS VALIDATE

    # Invert: because omniglot data is black on white, but we want
    # white on black

    all_transforms = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2])),
                    transforms.RandomInvert(p=1)])

    data = datasets.Omniglot('omniglot', download=True, transform=all_transforms)
    loader = DataLoader(data, batch_size=256, shuffle=True)

    new_data = []

    # Create new dataset based on omniglot to perform memory task
    # NOTE: NOT QUITE DONE YET

    for batch, (X, y) in enumerate(loader):

        # Expand into three color channels
        X = X.repeat((1, INPUT_SHAPE[0] // X.shape[1], 1, 1))

        # Sort characters into type
        label_dict = {}  # Label --> Image tensor
        label_index = {}  # Dict to keep track of something later
        for i in range(len(X)):
            if y[i] not in label_dict:
                label_dict[y[i]] = []
                label_index[y[i]] = 0
            label_dict[y[i]].append(X[i])

        for k in random.shuffle(label_index.keys()):
            i = 0
            while label_index[k] < len(label_dict[k]):
                new_tensors = []
                new_y = torch.zeros((2,))
                # Half get same type; half different type (roughly)
                if i % 2 == 0:
                    if len(label_dict[k]) > label_index[k] + 1:
                        new_tensors.append(label_dict[k][label_index[k]])
                        new_tensors.append(label_dict[k][label_index[k+1]])
                        label_index[k] += 2
                    else:
                        break
                else:
                    new_tensors.append(label_dict[k][label_index[k]])
                    tries = 0
                    num_tries = 5
                    # Try five times to
                    while tries < num_tries:
                        rand_other_type = random.shuffle(label_dict.keys())[0]
                        if len(label_dict[rand_other_type]) >= 1:
                            new_tensors.append(label_dict[rand_other_type][label_index[rand_other_type]])
                            label_index[rand_other_type] += 1
                            break
                        tries += 1
                    if num_tries == tries:
                        break
                    label_index[k] += 1

                new_tensors = tuple(new_tensors)
                i += 1

        for batch, (X, y) in enumerate(loader):
            pass

        exit(0)