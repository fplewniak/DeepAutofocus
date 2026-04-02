import torch
from torch import nn
from Dataset import Dataloaders
from main import ResNet18Model

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)
        running_loss += loss
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = running_loss / len(dataloader)
    print(f"loss: {avg_loss:>7f}")

if __name__ == '__main__':
    learning_rate = 1e-3
    batch_size = 8
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet18Model().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_dataloader,test_dataloader,val_dataloader = Dataloaders(batch_size)

    for k in range(epochs):
        train_loop(train_dataloader,model,loss_fn, optimizer, device)
    model_script = torch.jit.script(model)
    model_script.save('mon_modele3.pt')