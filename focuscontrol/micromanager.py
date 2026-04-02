from pycromanager import Core
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, transforms
import numpy as np
import sys

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
            print(f'{min(X.flatten())} - {max(X.flatten())}')
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
    initial_z = z
    # print("hauteur actuelle:",z)

    # transformer l'image en tableau numpy et la tourner de 90° pour l'orienter comme les images d'apprentissage
    image_np = np.reshape(image, shape=[core.get_image_width(), core.get_image_height()])
    # image_np = np.reshape(image, shape=[core.get_image_height(), core.get_image_width()])
    # image_np = np.rot90(image_np, axes=[0, 1]).copy()
    print(f'{min(image_np.flatten())} - {max(image_np.flatten())}')
    #plt.imshow(image_np / 65536.0)
    #plt.show()
    
    # 3. Définir des transformations
    # transform = transforms.Compose(
    #         [transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    #          transforms.Normalize((0.5,), (0.5,))])

    model = "Resnet34_resize512_best_model.pt"

    match model:
        case "Resnet34_resize512_best_model.pt":
            image_size = 512
            crop = False
        case "Resnet18reg_resize256_AdamW_1e-4_best_model.pt":
            image_size = 256
            crop = False
        case "Resnet18reg_resize512_AdamW_1e-4_best_model.pt":
            image_size = 512
            crop = False
        case "Resnet18reg_resize1024_AdamW_1e-4_best_model.pt":
            image_size = 1024
            crop = False
        case "Resnet18reg_1024_AdamW_1e-4_best_model.pt":
            image_size = 1024
            crop = True
        case _:
            image_size = image_np.shape[0]
            crop = False

    image_sizing = transforms.CenterCrop((image_size, image_size)) if crop else v2.Resize(image_size)

    transform = transforms.Compose(
            [transforms.ToTensor(),
             v2.ToDtype(torch.float, scale=True),
             #v2.functional.autocontrast,
             transforms.Normalize((0.5,), (0.5,)),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             image_sizing
             ])

    # 4. Créer le dataset
    dataset = NumpyImageDataset(image_np, transform=transform)

    # 5. (Optionnel) Utiliser un DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model2 = torch.jit.load(f"C:/DeepAutofocus/Models/{model}", map_location=torch.device(device))
    model2.eval()  # Très important pour désactiver dropout/batchnorm si utilisé
    delta = pred(dataloader, model2, device)

    with open('C:/DeepAutofocus/log.txt', 'a+') as logfile:
        logfile.write(f'initial Z: {z} - delta Z: {delta}')
        logfile.write('\n')
        if abs(initial_z - z + delta) <= 10:
            core.set_position(z - delta)
            core.wait_for_device(z_stage)
            z = core.get_position()
            logfile.write(f'new Z: {z}')
        else:
            logfile.write(f'shift too big: {initial_z - z + delta}')
        logfile.write('\n')

    # print("hauteur actuelle apres changement:",z)
    # print(param)
    # print('OK')

    # for line in sys.stdin:
    #     print("Python a reçu :", line.strip())
