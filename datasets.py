import numpy as np
from tifffile import tifffile
from torch.utils.data import Dataset


class FocusImageDataset(Dataset):
    def __init__(self, datalist, transform=None, target_transform=None):
        self.img_labels = datalist
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename, label = self.img_labels[idx]
        label = np.float32(label)
        image = tifffile.imread(filename)

        if self.transform:
            image = self.transform(image)
            # if idx == 0:
            #     print(f'{image=}, {image.dtype}')
            #     print(f'{min(image.flatten())=} - {max(image.flatten())=}')
            #     plt.imshow(torch.permute(image, (1, 2, 0)))
            #     plt.show()
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, filename

