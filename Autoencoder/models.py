from torch import nn

"""
Estimation of encoded size based on initial size of 512 x 512 with one channel : 262144
"""

class Autoencoder(nn.Module):
    """
    Encoded size: 32 * 256 * 256 = 2097152 (x8)
    """
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [B, 1, 28, 28] → [B, 16, 14, 14]
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 16, 14, 14] → [B, 32, 7, 7]
                nn.ReLU(),
                )

        # Decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [B, 32, 7, 7] → [B, 16, 14, 14]
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14] → [B, 1, 28, 28]
                nn.Sigmoid()
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder8_32(nn.Module):
    """
    Encoded size: 32 * 32 * 32 = 32768 (12.5%)
    """
    def __init__(self):
        super(Autoencoder8_32, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),  # [B, 1, 28, 28] → [B, 16, 14, 14]
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),  # [B, 16, 14, 14] → [B, 32, 7, 7]
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU()
                )

        # Decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # [B, 32, 7, 7] → [B, 16, 14, 14]
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14] → [B, 1, 28, 28]
                nn.Sigmoid()
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder16_64(nn.Module):
    """
    Encoded size: 64 * 32 * 32 = 65536 (25%)
    """
    def __init__(self):
        super(Autoencoder16_64, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [B, 1, 28, 28] → [B, 16, 14, 14]
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 16, 14, 14] → [B, 32, 7, 7]
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                )

        # Decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [B, 32, 7, 7] → [B, 16, 14, 14]
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14] → [B, 1, 28, 28]
                nn.Sigmoid()
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder512(nn.Module):
    """
    Ecoded size: 128 * 32 * 32 = 131072 (50%)
    """
    def __init__(self):
        super(Autoencoder512, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 512 -> 256
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 256 -> 128
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 128 -> 64
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                nn.ReLU(),
                # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32 -> 16
                # nn.ReLU()
                )

        # Decoder
        self.decoder = nn.Sequential(
                # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 32
                # nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 128
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 128 -> 256
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 256 -> 512
                # nn.Sigmoid()  # normalize output to [0, 1]
                # nn.Tanh()
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder8_64(nn.Module):
    """
    Ecoded size: 64 * 32 * 32 = 65536 (25%)
    """
    def __init__(self):
        super(Autoencoder8_64, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 512 -> 256
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 256 -> 128
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                nn.ReLU(),
                )
        #
        # Decoder
        self.decoder = nn.Sequential(

                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 64
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 128
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 128 -> 256
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 256 -> 512
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder8_32(nn.Module):
    """
    Encoded size : 32 * 64 * 64 = 131072 (50%)
    """
    def __init__(self):
        super(Autoencoder8_32, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 512 -> 256
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 256 -> 128
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
                nn.ReLU(),
                )
        #
        # Decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 128
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 128 -> 256
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 256 -> 512
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
