import csv
import glob
import random
import numpy as np
import os
import tifffile
import torch
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import  transforms, v2
import pandas as pd



def ecriturecsv(modif=False):
    """
    écrit le csv resultat avec le nom de toutes les images et leurs deltaz dans /Z-stacks, parametre modif True or False
    decide si les zfocus sont modifiés avec des fichies csv
    :return:
    """
    chemin_racine = '/data3/DeepAutoFocus'  # à modifier selon ton besoin
    # Parcourir tous les dossiers et fichiers
    # ouvre le csv du nom des images dans le dossier et numérote en z les images
    if not modif:
        with open('resultat.csv', mode='w', newline='', encoding='utf-8') as fichier_csv:
            writer = csv.writer(fichier_csv)
            writer.writerow(['Nom du fichier complet', "deltaz [µm]"])  # En-têtes
            for dossier_racine, sous_dossiers, fichiers in os.walk(chemin_racine):
                z = 30 * 0.2
                fichiers = natsorted(fichiers)
                for fichier in fichiers:
                    if fichier.lower().endswith(('.tif', '.tiff')):
                        writer.writerow([f"{dossier_racine}/" + f"{fichier}", f"{z}"])
                        z = z - 0.2
    else:
        epsilon = 1e-10
        with open('resultat.csv', mode='w', newline='', encoding='utf-8') as fichier_csv:
            writer = csv.writer(fichier_csv)
            writer.writerow(['Nom du fichier complet', "deltaz [µm]"])  # En-têtes
            for dossier_racine, sous_dossiers, fichiers in os.walk(chemin_racine):
                for soussous_dossiers in sous_dossiers:

                    df = pd.read_csv("/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/BadData.csv")
                    if (df[df.iloc[:, 0].str.contains(f"{soussous_dossiers}")].empty):  # enleve les positions dans Baddata
                        # verifier si le nom de dossier est dans les zfoc à modifié
                        df = pd.read_csv("/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/CorrectionFocusOnCells.csv")
                        df1 = df[df['FOV'] == f"{soussous_dossiers}"]
                        z = 30 * 0.2
                        if not df1.empty:
                            ser = df1.iloc[:, 1]
                            z = int(ser.iloc[0]) * 0.2
                        dos_comp = os.path.join(dossier_racine, soussous_dossiers)
                        lfichier = natsorted(os.listdir(dos_comp))

                        for fichier in lfichier:
                            if fichier.lower().endswith(('.tif', '.tiff')):
                                writer.writerow([f"{dos_comp}/" + f"{fichier}", f"{z:.2}"])
                                z = z - 0.2
                                if abs(z) < epsilon:
                                    z = 0.0

def limages():
    random.seed(42)
    liste_train=[]
    main_file = []
    for main_file in os.listdir("/data3/DeepAutoFocus/"):
        for dossier in os.listdir("/data3/DeepAutoFocus/"+f"{main_file}/"):
            chemin = os.path.join("/data3/DeepAutoFocus/",main_file, dossier)+"/"
            if dossier[-1].isdigit(): # verfier quil s'agit d'un sous dossier POS
                liste_train.append(chemin)
    random.shuffle(liste_train)
    return liste_train

def limagesB():
    random.seed(42)
    df = pd.read_csv("./resultat.csv")
    liste_train = list(df['Nom du fichier complet'])

    # liste_train = glob.glob('/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/*/*/*.tif')

    # for dossier in os.listdir("/data3/DeepAutoFocus/"):
    #     print(f'{dossier=}')
    #     if dossier == "20250610_Nikon_zStacks_W52_YAK1-09":
    #         # print(main_file)
    #         #print("dossier",dossier)
    #         chemin = os.path.join("/data3/DeepAutoFocus/", dossier) + "/"
    #         #print("chemin",chemin)
    #         if not dossier.endswith((".txt", ".json",".csv")):
    #             #print("oui")
    #             if not dossier.startswith("Pos"):
    #                 #print("/data3/DeepAutoFocus/" + f"{main_file}/"+ f"{dossier}/")
    #                 for sousdossier in os.listdir("/data3/DeepAutoFocus/" + f"{dossier}/"):
    #                     print("sousdossier",sousdossier)
    #                     if not sousdossier.endswith(".csv"):
    #                         chemin = os.path.join("/data3/DeepAutoFocus/",dossier,sousdossier) + "/"
    #                         if chemin.startswith(("/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09")):
    #                             for soussousdossier in os.listdir("/data3/DeepAutoFocus/" + f"{dossier}/"+ f"{sousdossier}/"):
    #                                 chemin = os.path.join("/data3/DeepAutoFocus/", dossier,sousdossier,soussousdossier) + "/"
    #                                 if not soussousdossier.endswith((".txt", ".json",".csv")):
    #                                     liste_train.append(chemin)
    #                         if not sousdossier.endswith((".txt", ".json",".csv")):
    #                             liste_train.append(chemin)
    print(f'{len(liste_train)=}')
    random.shuffle(liste_train)
    return liste_train

def Listes(type=None):
    """ listes de toutes les pos dans 3 listes"""
    if type == "B":
        liste_train = limagesB()
    else:
        liste_train = limages()
    nbr_ech = len(liste_train)
    nbr_train = int(0.6*nbr_ech)
    nbr_test = int(0.2*nbr_ech)
    # print(f'{nbr_ech=} {nbr_train=} {nbr_test=}')
    L_train = []
    L_test = []
    L_val = []
    for k in range(nbr_train):
        L_train.append(liste_train[k])
    for l in range(nbr_train,nbr_train+nbr_test):
        L_test.append(liste_train[l])
    for m in range(nbr_train+nbr_test,nbr_ech):
        L_val.append(liste_train[m])
    # print(f'{len(L_train)=} {len(L_test)=} {len(L_val)=}')
    return L_train,L_test,L_val

def tuple_image(l):
    """
    récupère toutes les hauteurs pour les position dans la liste l
    :param l:
    :return:
    """
    df = pd.read_csv("./resultat.csv")
    # Garder les lignes où 'nom' commence par un des préfixes
    df1 = df[df["Nom du fichier complet"].apply(lambda x: any(x.startswith(p) for p in l))]
    print(df1)
    # ltuple = list(df1.itertuples(index=False, name=None))
    #print(ltrain)
    return list(df1.itertuples(index=False, name=None))

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
        # print(f'{image=}')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, filename

def cust_dataset(l,crop = True):
    """
    crée un dataset à partir de la liste l de positions
    :param l:
    :return:
    """
    print(f'{len(l)=}')
    L_train = tuple_image(l)
    print(f'{len(L_train)=}')
    if crop:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.CenterCrop((1024,1024)),
                                        v2.ToDtype(torch.float, scale=True),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), v2.ToDtype(torch.float, scale=True),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize((0.5,), (0.5,))])
    # print(f'{L_train[0]=}')
    Dataset = CustomImageDataset(L_train,"/data3/DeepAutoFocus",transform)
    # print(f'{len(Dataset)=}')
    return Dataset

def cust_dataloader(batch_size,Dataset_train):
    train_dataloader = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    return train_dataloader

if __name__ == '__main__':
    """à lancer une fois pour créer le fichier resultat.csv, utilisé pour le dataset"""
    ecriturecsv(True)
