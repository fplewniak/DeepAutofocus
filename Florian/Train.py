import sys
import torch
from matplotlib import pyplot as plt
import mplcursors
from torch import nn
from Dataset import cust_dataloader, Listes, cust_dataset
from Models import ResNet34Model, ResNet18Model, ResNetModel


def train_loop_eval(dataloader,evalloader, model, loss_fn, optimizer, device,Ltrain_loss,Lval_loss,lossmin,epoch,nom):
    model.train()
    running_loss = 0.0
    for X, y,_ in dataloader:
        X = X.to(device)
        y = y.to(device)
        y = y.squeeze()
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()


        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = running_loss / len(dataloader)
    print(f"loss: {avg_loss:>7f}")
    # Validation phase
    model.eval()
    val_loss = 0.0
    for X, y,_ in evalloader:
        X = X.to(device)
        y = y.to(device)
        y = y.squeeze()
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)
        val_loss += loss.item()
    avg_train_loss = running_loss / len(dataloader)
    avg_val_loss = val_loss / len(evalloader)
    if epoch==0:
        lossmin = avg_val_loss
        model_script = torch.jit.script(model)
        model_script.save(nom)
    elif avg_val_loss < lossmin:
        lossmin = avg_val_loss
        model_script = torch.jit.script(model)
        model_script.save(nom)
    print("epoch:", epoch, "train loss:", avg_train_loss, "val loss:", avg_val_loss, "loss min",lossmin)
    Ltrain_loss.append(avg_train_loss)
    Lval_loss.append(avg_val_loss)
    return Ltrain_loss,Lval_loss,lossmin

def test_loop2(dataloader, model, loss_fn, device,titre="test"):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    ytot= []
    predtot= []
    diff = []
    descriptions = []
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y, etiq in dataloader:

            descriptions += etiq
            # print(len(descriptions))
            ytot.append(y)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            predtot += pred
            test_loss += loss_fn(pred.squeeze(), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            diff  += (pred.squeeze() - y).tolist()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    ytot = torch.cat(ytot).cpu().numpy().tolist()
    predtot = [t.item() for t in predtot]
    fig, ax = plt.subplots()
    sc = ax.scatter(ytot, predtot)
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(descriptions[sel.index])

    plt.title(titre)
    plt.xlabel("distance réelle (µm)")
    plt.ylabel("distance prédite (µm)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    return diff

def mult_train(Lmodel,DataloadA,evalloader,epochs,learning_rate = 1e-3):
    Llosstrain= []
    Llosseval =[]
    for l in (Lmodel):
        lossmin = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = nn.MSELoss()
        losstrain, losseval = [], []
        for k in range(epochs):
            optimizer = torch.optim.Adam(l[0].parameters(), lr=learning_rate)
            losstrain, losseval, lossmin = train_loop_eval(DataloadA, evalloader, l[0], loss_fn, optimizer, device,
                                                             losstrain, losseval, lossmin, k, l[1])
        Llosstrain.append(losstrain)
        Llosseval.append(losseval)

        model = l[1]
    return Llosstrain,Llosseval,model

def mult_testbox(Lmodel,DataloadAtest):
    delta = []
    for l in (Lmodel):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = nn.MSELoss()
        modelA = torch.jit.load(l)
        modelA.eval()  # Très important pour désactiver dropout/batchnorm si utilisé
        dataA = test_loop2(DataloadAtest, modelA, loss_fn, device,DataloadAtest)
        delta.append(dataA)
    plt.boxplot(delta, labels=["18", "34"])
    plt.show()

def affichage(Llosstrain,Llosseval,epochs, dataset, model):
    x = [k for k in range(epochs)]

    plt.figure()
    plt.plot(x, Llosstrain[0], label=f'model{model}_{dataset}train', marker='o', linestyle='-')
    plt.plot(x, Llosseval[0], label=f'model{model}_{dataset}eval', marker='o', linestyle='-')
    plt.yscale('log')
    plt.xlabel('epoch')  # Label de l’axe x
    plt.ylabel('loss')  # Label de l’axe y
    plt.title('loss de train et evaluation')
    plt.legend()  # 👈 Affiche la légende (labels des courbes)
    plt.grid(True)
    plt.savefig(f'training_history_{dataset}_{model}.png')

def execute(choix,Lmodel):
    Lmod= Lmodel
    if  "34" in choix:
        Lmod = [[ResNet34Model().to(device),"r34test.pt",4]]
        Dataloadex = cust_dataloader(8, Data)
        print(Lmod)
    if "18" in choix:
        Lmod = [[ResNet18Model().to(device),"r18test.pt"]]
        Dataloadex = cust_dataloader(16, Data)
    if "50" in choix:
        Lmod = [[ResNetModel().to(device),"r50test.pt",4]]
        Dataloadex = cust_dataloader(4, Data)

    if  "apprentissage" in choix :
            Llosstrain, Llosseval,model = mult_train(Lmod, Dataloadex, DataloadevalA, epochs, learning_rate)
            print("apprentissage")
            affichage(Llosstrain, Llosseval, epochs, "A", model)

    if "test" in choix:
            mult_testbox(Lmod[1], DataloadAtest)

if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 2
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Lmodel = [[ResNet34Model().to(device),"resnet342test.pt",4],[ResNet18Model().to(device),"resnet18test.pt",8],[ResNetModel().to(device),"resnet200test.pt",4]]
    Ltest = ["r18fulldonnees.pt"]
    loss_fn = nn.MSELoss()

    Ltrain, Ltest, Leval = Listes("B")

    Data = cust_dataset(Ltrain, crop=False)
    DataloadA = cust_dataloader(batch_size, Data)
    DataAtest = cust_dataset(Ltest, crop=False)
    DataloadAtest = cust_dataloader(batch_size, DataAtest)
    DataAeval = cust_dataset(Leval, crop=False)
    DataloadevalA = cust_dataloader(batch_size, DataAeval)

    choix = sys.argv[1:]
    print(choix)
    execute(choix, Lmodel)

