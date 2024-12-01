

import cv2
import json
import os
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.tgt_path = os.path.join(data_path, 'labels.json')
        with open(self.tgt_path, 'r') as f:
            self.data = json.load(f)
        self.data = self.data['labels']
        self.data = [(image_name, label) for image_name, label in self.data.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        image_path = os.path.join(self.data_path, 'images', image_name)
        image = cv2.imread(image_path)
        # TODO: add online augmentation pipeline
        if self.transform:
            image = self.transform(image)
        return image, label


# TODO: function to get vocab from data
VOCAB = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "â", "á"]