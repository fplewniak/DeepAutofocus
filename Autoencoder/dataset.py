import os
from pathlib import Path

import numpy as np
import tifffile
from torch.utils.data import Dataset


class FocusImageDataset(Dataset):
    def __init__(self, datalist, img_dir, transform=None, target_transform=None):
        self.img_labels = datalist
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename, label = self.img_labels[idx]
        label = np.float32(label)
        img_path = os.path.join(self.img_dir, filename)
        image = tifffile.imread(Path(str(img_path)))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, filename
