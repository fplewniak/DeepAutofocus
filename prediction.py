import torch
from torch.utils.data import DataLoader
from Dataset import CustomImageDataset
from torchvision.transforms import  transforms, v2

im = [("position2_001/image_separee_4.tif","0")]
print(f'{im=}')
transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Normalize((0.5,), (0.5,))])

predset = CustomImageDataset(im,"./data",transform)
Datapred= DataLoader(predset, batch_size=1, shuffle=True)

def predi(dataloader, model, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            print(pred)

if __name__ == '__main__':
    batch_size = 1
    # Sauvegarder uniquement les poids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model2 = torch.jit.load("mon_modele3.pt")
    model2.eval()  # Très important pour désactiver dropout/batchnorm si utilisé
    print(f'{model2=}')
    predi(Datapred, model2, device)

