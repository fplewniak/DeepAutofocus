import csv
import os
import random
import pandas as pd
from natsort import natsorted

#liste de tous les dossiers images
def limages():
    liste_train=[]
    main_file = []
    for main_file in os.listdir("/home/fred/Public/DeepAutoFocus/Z-stacks/"):
        for dossier in os.listdir("/home/fred/Public/DeepAutoFocus/Z-stacks/"+f"{main_file}/"):
            chemin = os.path.join("/home/fred/Public/DeepAutoFocus/Z-stacks/",main_file, dossier)+"/"
            if dossier[-1].isdigit(): # verfier quil s'agit d'un sous dossier POS
                liste_train.append(chemin)
    random.shuffle(liste_train)
    return liste_train

#utilise l image n de limage
def limage_uni(l,n):
    return l[n]




if __name__ == '__main__':
    # Chemin du répertoire de départ
    chemin_racine = '/home/fred/Public/DeepAutoFocus/Z-stacks'  # à modifier selon ton besoin
    # Parcourir tous les dossiers et fichiers
    #ouvre le csv du nom des images dans le dossier et numérote en z les images
    print(limages())
    with open('resultat.csv', mode='w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(['Nom du fichier complet',"deltaz [µm]"])  # En-têtes
        for dossier_racine, sous_dossiers, fichiers in os.walk(chemin_racine):

            nom_sous_dossier = os.path.basename(dossier_racine)  # juste le nom du dossier courant
            #print(nom_sous_dossier)
            z = 30*0.2
            fichiers = natsorted(fichiers)
            for fichier in fichiers:
                if fichier.lower().endswith(('.tif', '.tiff')):
                    writer.writerow([f"{dossier_racine}/"+f"{fichier}",f"{z}"])
                    z = z-0.2




