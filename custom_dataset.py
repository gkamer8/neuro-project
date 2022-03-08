import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import pickle
import torch

"""

Expects:
- examples_dir to be a folder with a bunch of pickle files
- one pickle file should be "header.pkl", which contains a dictionary
- The dictionary should have "paths" and "len"
- len is size of dataset
- paths is dictionary of index -> path of pkl object
- that pickle object should be dictionary with "img1", "img2", "img3", "y" - all tensors

"""

class CustomImageDataset(Dataset):
    def __init__(self, examples_dir):

        self.examples_dir = examples_dir

        header_path = os.path.join(examples_dir, 'header.pkl')
        header_obj = pickle.load(open(header_path, "rb"))
        self.len = header_obj['len']

        self.paths = header_obj['paths']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        """

        3 images each get put in a dimension
        0: memory 1
        1: memory 2
        2: input

        """

        pickle_path = self.paths[idx]

        obj = pickle.load(open(pickle_path, "rb"))

        # X: tensor with a dimension added for each image
        img_shape = list(obj['img1'].shape)

        x = torch.empty([3] + img_shape)
        x[0] = obj['img1']
        x[1] = obj['img2']
        x[2] = obj['img3']

        y = obj['y']

        return x, y