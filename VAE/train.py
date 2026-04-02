import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, v2

from Florian.Drafts.dataset import FocusImageDataset
from models import VAE


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)


def train_loop(training_loader, validation_loader, model, optimizer, device):
    model.train()
    # decoder.train()
    running_loss = 0.0
    train_loss = 0
    for x_blur, x_sharp, _ in train_loader:
        x_blur, x_sharp = x_blur.to(device), x_sharp.to(device)
        x_blur, x_sharp = v2.functional.autocontrast(x_blur), v2.functional.autocontrast(x_sharp)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x_blur)
        # print(f'{x_blur.shape=}')
        # print(f'{x_sharp.shape=}')
        # print(f'{x_recon.shape=}')
        # print(f'{x_blur.max()=} {x_blur.min()=}')
        # print(f'{x_sharp.max()=} {x_sharp.min()=}')
        # print(f'{x_recon.max()=} {x_recon.min()=}')
        loss = vae_loss_function(x_recon, x_sharp, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_blur, x_sharp, _ in val_loader:
            x_blur, x_sharp = x_blur.to(device), x_sharp.to(device)
            x_blur, x_sharp = v2.functional.autocontrast(x_blur), v2.functional.autocontrast(x_sharp)
            x_recon, mu, logvar = model(x_blur)
            loss = vae_loss_function(x_recon, x_sharp, mu, logvar)
            val_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    avg_val_loss = val_loss / len(validation_loader)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Training Loss: {avg_train_loss:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, ")
    return {'train loss': avg_train_loss, 'val loss': avg_val_loss}


def test_loop(dataloader, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_blur, x_sharp, _ in test_loader:
            x_blur, x_sharp = x_blur.to(device), x_sharp.to(device)
            x_blur, x_sharp = v2.functional.autocontrast(x_blur), v2.functional.autocontrast(x_sharp)
            x_recon, mu, logvar = model(x_blur)
            loss = vae_loss_function(x_recon, x_sharp, mu, logvar)
            test_loss += loss.item()

    test_loss = test_loss / len(dataloader)
    print(f"Test Error: Avg loss: {test_loss:.4f} \n")


def vae_loss_function(x_recon, x_sharp, mu, logvar, beta=1e-4):
    # recon_loss = F.mse_loss(x_recon, x_sharp)
    # recon_loss = F.l1_loss(x_recon, x_sharp, reduction='mean')
    recon_loss = F.binary_cross_entropy_with_logits(x_recon, x_sharp, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


class PairedRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.size)
        # return v2.functional.crop(img1, i, j, h, w), v2.functional.crop(img2, i, j, h, w)
        return v2.functional.crop(img1, 128, 128, h, w), v2.functional.crop(img2, 128, 128, h, w)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    learning_rate = 1e-3
    batch_size = 64
    epochs = 15
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: x.repeat(3,1,1))])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize(size=32)])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=64)])
    # transform = transforms.ToTensor()
    # transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Normalize((0.5,), (0.5,)),])
    # transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True), transforms.Resize(size=(1024, 1024))])
    transform = transforms.Compose([transforms.ToTensor(), v2.ToDtype(torch.float, scale=True)])
    paired_transform = PairedRandomCrop(size=(512, 512))

    train_df = pd.read_csv('/data3/DeepAutoFocus/20250417/train_noPos16.csv')
    train_list = list(zip(train_df.filename, train_df.deltaz))
    train_dataset = FocusImageDataset(train_list, '/data3/DeepAutoFocus/20250417', transform, paired_transform)

    val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/val.csv')
    val_list = list(zip(val_df.filename, val_df.deltaz))
    val_dataset = FocusImageDataset(val_list, '/data3/DeepAutoFocus/20250417', transform, paired_transform)

    test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/testPos16.csv')
    test_list = list(zip(test_df.filename, test_df.deltaz))
    test_dataset = FocusImageDataset(test_list, '/data3/DeepAutoFocus/20250417', transform, paired_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    history = []

    try:
        model.apply(weights_init)
        for epoch in range(epochs):
            history.append(train_loop(train_loader, val_loader, model, optimizer, device))

        ##################################################################
        # Testing the model on test dataset ##############################
        test_loop(test_loader, model)

        # Get a batch of test images
        dataiter = iter(test_loader)
        images, in_focus, _ = next(dataiter)
        images = images.to(device)
        in_focus = in_focus.to(device)
        diff = images - in_focus

        # Run through model
        with torch.no_grad():
            # encoded = model.encoder(images)
            # decoded = model.decoder(encoded)
            decoded, _, _ = model(images)
            print(f'{decoded.shape=}')

        fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(21, 3))
        for i in range(10):
            # Original
            # print(f'{i}: {np.max(images[i].cpu().numpy().flatten())=}')
            img = v2.functional.autocontrast(in_focus[i])
            axes[0, i].imshow(img.cpu().squeeze(), cmap="gray")
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # Reconstructed
            # print(f'{i}: {np.max(decoded[i].cpu().numpy().flatten())=}')
            img = v2.functional.autocontrast(decoded[i])
            axes[1, i].imshow(img.cpu().squeeze(), cmap="gray")
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

            # Blur input
            # print(f'{i}: {np.max(decoded[i].cpu().numpy().flatten())=}')
            img = v2.functional.autocontrast(images[i])
            axes[2, i].imshow(img.cpu().squeeze(), cmap="gray")
            axes[2, i].set_title("Blur input")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.show()
    finally:
        ##################################################################
        # torch.save(model.state_dict(), 'NetMNISTmodel.pth')
        encoder_scripted = torch.jit.script(model.encoder)  # Export to TorchScript
        # encoder_scripted.save('NetMNIST_encoder.pt')  # Save
        encoder_scripted.save('Zeiss_vae_encoder.pt')  # Save
        decoder_scripted = torch.jit.script(model.decoder)  # Export to TorchScript
        # decoder_scripted.save('NetMNIST_decoder.pt')  # Save
        decoder_scripted.save('Zeiss_vae_decoder.pt')  # Save
        model_scripted = torch.jit.script(model)
        # model_scripted.save('NetMNIST_autoencoder.pt')  # Save
        model_scripted.save('Zeiss_vae.pt')  # Save
        ##################################################################
        df = pd.DataFrame(history)

        plt.figure(1)
        df['train loss'].plot()
        df['val loss'].plot()

        plt.legend()
        plt.show()

        print("Done.")
