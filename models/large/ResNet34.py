import torch
import torch.nn as nn
from torchvision import models

class ResNet34Model(nn.Module):
    def __init__(self, freeze=False):
        super(ResNet34Model, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)
        self.regression = nn.Sequential(
                self.global_avg_pool,
                nn.Flatten(),
                # self.batch_norm1,
                nn.LeakyReLU(),
                self.dropout,
                self.fc
                )

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        # x = self.global_avg_pool(x)
        # x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        # x = self.dropout(x)
        # x = self.fc(x)
        x = self.regression(x)
        return x
