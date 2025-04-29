import numpy as np
import os
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
from traitement_image import limages, limage_uni
from torchvision.transforms import  transforms, v2
import pandas as pd

def Listes():
    liste_train = limages()
    nbr_ech = len(liste_train)
    nbr_train = int(0.6*nbr_ech)
    nbr_test = int(0.2*nbr_ech)
    L_train = []
    L_test = []
    L_val = []
    for k in range(nbr_train):
        L_train.append(liste_train[k])
    for l in range(nbr_train,nbr_train+nbr_test):
        L_test.append(liste_train[l])
    for m in range(nbr_train+nbr_test,nbr_ech):
        L_val.append(liste_train[m])
    return L_train,L_test,L_val

def tuples_images():
    a,b,c=Listes()
    df = pd.read_csv("resultat.csv")
    # Garder les lignes où 'nom' commence par un des préfixes
    df1 = df[df["Nom du fichier complet"].apply(lambda x: any(x.startswith(p) for p in a))]
    df2 = df[df["Nom du fichier complet"].apply(lambda x: any(x.startswith(p) for p in b))]
    df3 = df[df["Nom du fichier complet"].apply(lambda x: any(x.startswith(p) for p in c))]
    ltrain = list(df1.itertuples(index=False, name=None))
    ltest = list(df2.itertuples(index=False, name=None))
    lval = list(df3.itertuples(index=False, name=None))
    return ltrain,ltest,lval

def tuple_image(l):
    li = l
    df = pd.read_csv("resultat.csv")
    # Garder les lignes où 'nom' commence par un des préfixes
    df1 = df[df["Nom du fichier complet"].apply(lambda x: any(x.startswith(p) for p in li))]
    ltrain = list(df1.itertuples(index=False, name=None))
    #print(ltrain)
    return ltrain



class CustomImageDataset(Dataset):
    def __init__(self, annotations_liste, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_liste
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename, label = self.img_labels[idx]
        label = np.float32(label)
        img_path = os.path.join(self.img_dir, filename)
        image = tifffile.imread(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def cust_datasets():
    L_train,L_test,L_val = tuples_images()
    transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True),transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize((0.5,), (0.5,))])
    Dataset_train = CustomImageDataset(L_train,"/home/fred/Public/DeepAutoFocus/Z-stacks", transform)
    Dataset_test = CustomImageDataset(L_test,"/home/fred/Public/DeepAutoFocus/Z-stacks",transform)
    Dataset_val = CustomImageDataset(L_val,"/home/fred/Public/DeepAutoFocus/Z-stacks",transform)
    return Dataset_train,Dataset_test,Dataset_val

def cust_dataset(l):
    L_train = tuple_image(l)
    transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True),transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize((0.5,), (0.5,))])
    Dataset = CustomImageDataset(L_train,"/home/fred/Public/DeepAutoFocus/Z-stacks",transform)
    return Dataset

def cust_single_dataset(L_train):
    transform = transforms.Compose(
        [transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize((0.5,), (0.5,))])
    Dataset_train = CustomImageDataset(L_train, "/home/fred/Public/DeepAutoFocus/Z-stacks", transform)
    return Dataset_train


def dataloaders(batch_size):
    Dataset_train, Dataset_test, Dataset_val = cust_datasets()
    train_dataloader = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(Dataset_test, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Dataset_val, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader,val_dataloader

def cust_dataloader(batch_size,Dataset_train):
    train_dataloader = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    return train_dataloader

def Dataloader(batch_size,l):
    Dataset_train = cust_dataset(l)
    train_dataloader = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    return train_dataloader

