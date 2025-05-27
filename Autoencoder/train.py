import random

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import v2

from models import Autoencoder, Autoencoder8_64, Autoencoder512
from dataset import FocusImageDataset


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device):
    model.train()
    if not isinstance(loss_fn, list):
        loss_fn = [loss_fn]
    # decoder.train()
    running_loss = 0.0
    for images, _, _ in training_loader:
        images = images.to(device)
        outputs = model(images)
        if epoch < epochs * 0.25:
            loss = loss_fn[0](outputs, images)
        elif epoch < epochs * 0.5:
            loss = 0.5 * (loss_fn[0](outputs, images) + loss_fn[1](torch.sigmoid(outputs), images))
        elif epoch < epochs * 0.8:
            loss = 0.2 * loss_fn[0](outputs, images) + 0.8 * loss_fn[1](torch.sigmoid(outputs), images)
        else:
            loss = loss_fn[1](torch.sigmoid(outputs), images)
        # loss = loss_fn(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    # Validation phase
    model.eval()
    # decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _, _ in validation_loader:
            images = images.to(device)
            # images = images.to(memory_format=torch.channels_last)
            # encoded = encoder(images)
            outputs = model(images)
            if epoch < 5:
                loss = loss_fn[0](outputs, images)
            elif epoch < 10:
                loss = 0.5 * (loss_fn[0](outputs, images) + loss_fn[1](torch.sigmoid(outputs), images))
            else:
                loss = 0.2 * loss_fn[0](outputs, images) + 0.8 * loss_fn[1](torch.sigmoid(outputs), images)
            # loss = loss_fn(outputs, images)
            val_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    avg_val_loss = val_loss / len(validation_loader)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Training Loss: {avg_train_loss:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, ")
    return {'train loss': avg_train_loss, 'val loss': avg_val_loss}


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, num_batches = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for images, _, _ in test_loader:
            num_batches += 1
            images = images.to(device)
            # encoded = encoder(images)
            outputs = model(images)
            test_loss += loss_fn(outputs, images).item()
            _, predicted = torch.max(outputs.data, 1)

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    learning_rate = 1e-3
    batch_size = 64
    epochs = 25
    # criterion = nn.MSELoss()
    criterion = [nn.BCEWithLogitsLoss(), nn.L1Loss()]
    model = Autoencoder8_64().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: x.repeat(3,1,1))])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize(size=32)])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=64)])
    # transform = transforms.ToTensor()
    # transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Normalize((0.5,), (0.5,)),])
    # transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Resize(size=(1024, 1024))])
    transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.RandomCrop(size=(256, 256))])

    train_df = pd.read_csv('/data3/DeepAutoFocus/20250417/train_noPos16.csv')
    train_list = list(zip(train_df.filename, train_df.deltaz))
    train_dataset = FocusImageDataset(train_list, '/data3/DeepAutoFocus/20250417', transform, None)

    val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/val.csv')
    val_list = list(zip(val_df.filename, val_df.deltaz))
    val_dataset = FocusImageDataset(val_list, '/data3/DeepAutoFocus/20250417', transform, None)

    test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/testPos16.csv')
    test_list = list(zip(test_df.filename, test_df.deltaz))
    test_dataset = FocusImageDataset(test_list, '/data3/DeepAutoFocus/20250417', transform, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # train_dataset = datasets.MNIST(root='/data2/PyTorch/data/MNIST', train=True, download=True, transform=transform)
    #
    # train_size = int(0.9 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
    #
    # test_dataset = datasets.MNIST(root='/data2/PyTorch/data/MNIST', train=False, download=True, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)

    history = []

    try:
        model.apply(weights_init)
        for epoch in range(epochs):
            history.append(train_loop(train_loader, val_loader, model, criterion, optimizer, device))
            # scheduler.step()
            # print(f'Learning rate: {scheduler.get_last_lr()}')
        ##################################################################
        # Testing the model on test dataset ##############################
        test_loop(test_loader, model, criterion[0])

        # Get a batch of test images
        dataiter = iter(test_loader)
        images, _, _ = next(dataiter)
        images = images.to(device)
        print(f'identity loss: {criterion[1](images, images).item()}')

        # Run through model
        with torch.no_grad():
            # encoded = model.encoder(images)
            # decoded = model.decoder(encoded)
            decoded = model(images)

        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 3))
        for i in range(10):
            # Original
            # print(f'{i}: {np.max(images[i].cpu().numpy().flatten())=}')
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # Reconstructed
            # print(f'{i}: {np.max(decoded[i].cpu().numpy().flatten())=}')
            axes[1, i].imshow(decoded[i].cpu().squeeze(), cmap="gray")
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

        ##################################################################
    finally:
        ##################################################################
        # torch.save(model.state_dict(), 'NetMNISTmodel.pth')
        encoder_scripted = torch.jit.script(model.encoder)  # Export to TorchScript
        # encoder_scripted.save('NetMNIST_encoder.pt')  # Save
        encoder_scripted.save('Zeiss_encoder.pt')  # Save
        decoder_scripted = torch.jit.script(model.decoder)  # Export to TorchScript
        # decoder_scripted.save('NetMNIST_decoder.pt')  # Save
        decoder_scripted.save('Zeiss_decoder.pt')  # Save
        model_scripted = torch.jit.script(model)
        # model_scripted.save('NetMNIST_autoencoder.pt')  # Save
        model_scripted.save('Zeiss_autoencoder.pt')  # Save
        ##################################################################
        df = pd.DataFrame(history)
        # print(history)
        # print(df)
        plt.figure(1)
        df['train loss'].plot()
        df['val loss'].plot()
        # plt.plot([hist['train loss'] for hist in history])
        # plt.plot([hist['val loss'] for hist in history])
        plt.legend()
        plt.show()

        # print("Cleaning up DataLoader...")
        # cleanup_dataloader(train_loader)
        # cleanup_dataloader(val_loader)
        # cleanup_dataloader(test_loader)
        # monitor_threads_and_processes()
        # kill_all_pt_data_workers()
        print("Done.")

