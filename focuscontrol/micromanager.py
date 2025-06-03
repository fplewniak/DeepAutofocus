import os
from pycromanager import Core
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, transforms
import numpy as np
import sys


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_liste, img_dir, transform=None, target_transform=None):
#         self.img_labels = annotations_liste
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         filename, label = self.img_labels[idx]
#         label = np.float32(label)
#         img_path = os.path.join(self.img_dir, filename)
#         image = tifffile.imread(img_path)
#
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


class NumpyImageDataset(Dataset):
    def __init__(self, image_array, transform=None):
        self.image = image_array
        self.transform = transform

    def __len__(self):
        return 1  # car on n'a qu'une image

    def __getitem__(self, idx):
        img = self.image
        if self.transform:
            img = self.transform(img)
        return img


def pred(dataloader, model, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            # print(pred)
            deltaz = pred.item()
    return deltaz


if __name__ == '__main__':
    param = sys.argv
    core = Core()
    z_stage = core.get_focus_device()

    # Définir le dispositif de caméra actif (si ce n’est pas déjà fait)
    # core.set_camera_device('Camera')  # optionnel selon la config

    # Capture d'une image (snapshot)
    core.snap_image()
    image = core.get_image()

    z = core.get_position()
    # print("hauteur actuelle:",z)

    # Si l'image est une seule couche, transformer en tableau numpy
    image_np = np.reshape(image, newshape=[core.get_image_height(), core.get_image_width()])

    # 3. Définir des transformations (facultatif)
    transform = transforms.Compose(
            [transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.5,), (0.5,))])

    # 4. Créer le dataset
    dataset = NumpyImageDataset(image_np, transform=transform)

    # 5. (Optionnel) Utiliser un DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model2 = torch.jit.load("C:/DeepAutofocus/Models/resnet18.pt", map_location=torch.device("cpu"))
    model2.eval()  # Très important pour désactiver dropout/batchnorm si utilisé
    delta = pred(dataloader, model2, device)

    core.set_position(z - delta)
    z = core.get_position()
    core.wait_for_device(z_stage)

    # print("hauteur actuelle apres changement:",z)
    # print(param)
    # print('OK')

    # for line in sys.stdin:
    #     print("Python a reçu :", line.strip())
