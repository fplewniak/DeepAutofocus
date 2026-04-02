import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2


class ResNet50Reg(nn.Module):
    def __init__(self):
        super(ResNet50Reg, self).__init__()

        # resnet = models.resnet18(weights=None)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.batch_norm1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.batch_norm1(x)  # normalize
        x = self.dropout(x)
        x = self.fc(x)
        return x
