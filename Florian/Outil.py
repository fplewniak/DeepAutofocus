import random
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from Dataset import Dataloader, tuple_image, CustomImageDataset, CustomImageDataset2
from models import ResNet18Model
from traitement2 import limages
from torchvision.transforms import  transforms, v2
import re

"""effectue une simulation de l'utilisation avec microscope pour connaître le changement de position avec la prédiction du modle utilisé"""
def init_sim():
    # return ('/home/fred/Public/DeepAutoFocus/Z-stacks/Zeiss_30zStack0-2-30_Cells_1/Pos3/img_channel000_position003_time000000000_z002.tif', 5.6)
    """renvoie le deltaz et le dossier d'une image d'une position aleatoire"""
    L = limages()
    rand = random.randint(0, len(L) - 1)
    l = [L[rand]]
    T =(tuple_image(l))
    rand2 = random.randint(0, len(T)-1)
    deltaz0 = T[rand2][1]
    pos0 = T[rand2][0]
    return T[rand2]



def simul(data,device,model):
    """
    :param data:dtaset d'une seul image
    :param device:
    :param model:
    :return: deltaz, z, respectivement le delta z calculé par le modèle, et la différence entre le deltaz réel et calculé
    """
    with torch.no_grad():
        for X, y in data:
            X = X.to(device)
            pred = model(X)
            # plt.scatter(y, pred.item(), marker='o', label='Points')
            # plt.show()

            deltaz = pred.item()
            deltaz0sim = y.cpu().numpy().tolist()[0]
    return deltaz


def deplacement(deltaz,incr):
    """
    :param deltaz: deltaz  calculé
    :param incr minimum d'hauteur entre deux positions z
    :return:
    """
    print(deltaz)
    nb_incr = deltaz / incr
    if nb_incr > nb_incr//1 +0.5:
        nb_incr = nb_incr//1 +1
        deltaz = nb_incr*incr
    else:
        nb_incr = nb_incr // 1
        deltaz = nb_incr*incr

    return deltaz

for i in range(10):
    l = init_sim()
    nouvelle_place_fichier = l[0]
    distfocus0 = l[1]
    depl=0
    Lpos = []
    for k in range(10):
        l = (nouvelle_place_fichier,round(distfocus0 - depl,1))

        #delta z exact fournie dans le dossier de l'image
        distfocus0 = l[1]
        Lpos.append(l[1])
        #donnée du deltaz de l'image de départ
        im= [(l[0],str(l[1]))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet18Model().to(device)
        model2 = torch.jit.load("resnet18.pt")
        model2.eval()  # Très important pour désactiver dropout/batchnorm si utilisé
        transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True),transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize((0.5,), (0.5,))])
        predset = CustomImageDataset(im,"./data",transform)
        Datapred= DataLoader(predset, batch_size=1, shuffle=True)
        deltaz = simul(Datapred,device,model2)

        incr =0.2
        depl = deplacement(deltaz,incr)
        incr_depart = distfocus0/incr + 30
        print("delta de depart",distfocus0)
        print("déplacement",depl)
        print("position après déplacement",round(distfocus0- depl,1))
        incr_depl = round(depl,1)/incr
        # z =
        print("nombre d'incréments",incr_depl)
        z = re.findall(r"_z(\d{3})\.tif", l[0])
        print("z",z[0])
        z = z[0]
        #retrouver image correspondante
        place_fichier = int(int(z) + incr_depl)
        place_fichier = f"{place_fichier:03}"
        print("place de l'image",int(place_fichier))

        z = place_fichier
        texte = l[0]
        nouvelle_place_fichier = re.sub(r"_z\d{3}\.tif", "_z"+place_fichier +".tif", texte)
        #nouvelle_place_fichier = l[0][:-7] + place_fichier + l[0][-4:]
        print("nouvelle image utilisée:",nouvelle_place_fichier)

    x = [k for k in range(len(Lpos))]
    plt.plot(x,Lpos, marker='o', label='Points')

plt.show()