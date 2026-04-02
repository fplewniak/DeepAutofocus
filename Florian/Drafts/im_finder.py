import torch
import torch.nn as nn
from Test import test_loop
from Dataset import cust_dataloader, tuples_images, cust_single_dataset
from main import ResNet18Model

def im_find():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    model = ResNet18Model().to(device)
    loss_fn = nn.MSELoss()
    ltrain, ltest, lval = tuples_images()
    ltrain = ltrain + ltest+ lval
    batch_size = 64
    model2 = torch.jit.load("resnet18.pt")
    model2.eval()  # Très important pour désactiver dropout/batchnorm si utilisé
    while len(ltrain) >40:
        print(len(ltrain)//2)
        train_bas = [ltrain[k] for k in range(len(ltrain)//2)]
        train_haut = [ltrain[k] for k in range(len(ltrain)//2,len(ltrain))]
        print(ltrain == train_bas+train_haut)

        dataset_bas = cust_single_dataset(train_bas)
        dataset_haut = cust_single_dataset(train_haut)
        Data_bas = cust_dataloader(batch_size,dataset_bas)
        Data_haut = cust_dataloader(batch_size,dataset_haut)

        test_loop(Data_bas, model2, loss_fn, device)
        test_loop(Data_haut, model2, loss_fn, device)
        test = input("quel image choisir, 1 ou2?")
        if test == 1:
            ltrain= train_bas
            print(ltrain == train_bas)
        else:
            ltrain = train_haut
    print(ltrain)




if __name__ == '__main__':
    im_find()