import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2


class MobileNetV3_l(nn.Module):
    def __init__(self):
        super(MobileNetV3_l, self).__init__()

        # resnet = models.resnet18(weights=None)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.batch_norm1 = nn.BatchNorm1d(960)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(960, 1)

    def forward(self, x):
        x = self.mobilenet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.batch_norm1(x)  # normalize
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MobileNetV3_s(nn.Module):
    def __init__(self):
        super(MobileNetV3_s, self).__init__()

        # resnet = models.resnet18(weights=None)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.batch_norm1 = nn.BatchNorm1d(576)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(576, 1)

    def forward(self, x):
        x = self.mobilenet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.batch_norm1(x)  # normalize
        x = self.dropout(x)
        x = self.fc(x)
        return x
