import torch.utils.data as data
import random as rd
import numpy as np

class ngcfdataset(data.Dataset):
    def __init__(self, train_data):
        self.train_data = train_data #딕셔너리임
        self.keys = list(train_data.keys())

    def __getitem__(self, index):
        self.init_keys = self.keys[index]
        return self.init_keys, self.train_data[self.init_keys]

    def __len__(self):
        self.count = len(self.keys)
        return self.count


