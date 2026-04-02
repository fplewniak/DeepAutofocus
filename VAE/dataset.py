import os
import re
from pathlib import Path

import tifffile
from torch.utils.data import Dataset


class FocusImageDataset(Dataset):
    def __init__(self, data_list, img_dir, transform=None, paired_transform=None):
        self.data_list = data_list
        self.img_dir = img_dir
        self.transform = transform
        self.paired_transform = paired_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        filename, delta_z = self.data_list[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = tifffile.imread(Path(str(img_path)))
        in_focus_filename = re.sub(r'_z\d\d\d\.tif', '_z030.tif', filename)
        img_path = os.path.join(self.img_dir, in_focus_filename)
        in_focus = tifffile.imread(Path(str(img_path)))

        if self.transform:
            image = self.transform(image)
            in_focus = self.transform(in_focus)
        if self.paired_transform:
            image, in_focus = self.paired_transform(image, in_focus)
        return image, in_focus, delta_z
