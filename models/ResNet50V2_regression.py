import torch
import torch.nn as nn
from torchvision import models


class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()

        resnet = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc(x)
        return x
