import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=2, padding=1),
                # nn.BatchNorm2d(32),
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                # nn.BatchNorm2d(64),
                nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                # nn.BatchNorm2d(128),
                nn.ReLU()
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                # nn.BatchNorm2d(256),
                nn.ReLU()
                )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 32, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 32 * 32)
        self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                # nn.BatchNorm2d(128),
                nn.ReLU()
                )
        self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                # nn.BatchNorm2d(64),
                nn.ReLU()
                )
        self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                # nn.BatchNorm2d(32),
                nn.ReLU()
                )
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)  # Final layer without BatchNorm

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 256, 32, 32)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        # x = torch.sigmoid(x)
        # x = torch.tanh(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_blur):
        mu, logvar = self.encoder(x_blur)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
