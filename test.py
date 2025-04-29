import torch
from matplotlib import pyplot as plt


def test_loop(dataloader, model, loss_fn, device,k,pos):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    ytot= []
    predtot= []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            ytot.append(y)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            predtot += pred
            test_loss += loss_fn(pred.squeeze(), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #ytot = [t.item() for t in ytot]
    ytot = torch.cat(ytot).cpu().numpy().tolist()
    #print(ytot)
    #print("pred")
    predtot = [t.item() for t in predtot]
    #print(predtot)
    plt.subplot(8, 6, k+1)
    plt.scatter(ytot,predtot, marker='o', label='Points')
    #plt.xlabel("distance réelle (µm)")
    #plt.ylabel("distance prédie (µm)")
    plt.axis('equal')  # 👈 le plus important
    plt.title(f"{pos}",fontsize=6)
    plt.grid(True)


def test_loop2(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    ytot= []
    predtot= []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y,pos in dataloader:
            print(pos)
            ytot.append(y)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            predtot += pred
            test_loss += loss_fn(pred.squeeze(), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #ytot = [t.item() for t in ytot]
    ytot = torch.cat(ytot).cpu().numpy().tolist()
    #print(ytot)
    #print("pred")
    predtot = [t.item() for t in predtot]
    #print(predtot)
    plt.figure()
    plt.scatter(ytot,predtot, marker='o', label='Points')
    plt.xlabel("distance réelle (µm)")
    plt.ylabel("distance prédie (µm)")
    #plt.axis('equal')  # 👈 le plus important
    plt.grid(True)
    plt.show()

