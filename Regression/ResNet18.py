import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2


class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()

        # resnet = models.resnet18(weights=None)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.batch_norm1(x)  # normalize
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResNet18Model2DenseLayers(nn.Module):
    def __init__(self):
        super(ResNet18Model2DenseLayers, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.batch_norm1(x) # normalize
        x = self.dropout(x)
        x = self.fc1(x)
        # x = self.batch_norm2(x) # normalize
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ResNet18Model3DenseLayers(nn.Module):
    def __init__(self):
        super(ResNet18Model3DenseLayers, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout_1(x)
        x = self.fc1(x)
        x = self.dropout_2(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class ResNet18ModelConv(nn.Module):
    def __init__(self):
        super(ResNet18ModelConv, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512, 64)
        self.conv2d_1 = nn.Conv2d(1, 36, 2, 1)
        self.fc2 = nn.Linear(36, 1)
        # self.fc3 = nn.Linear(49, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        # print(f'resnet {x.shape=}')
        x = self.global_avg_pool(x)
        # print(f'avg pool {x.shape=}')
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        # print(f'flatten {x.shape=}')
        x = self.dropout_1(x)
        x = self.fc1(x)
        # print(f'fc1 {x.shape=}')
        x = torch.unflatten(x, -1, (1, 8, 8))  # Flatten to (batch, features)
        # print(f'unflatten {x.shape=}')
        x = self.conv2d_1(x)
        # print(f'conv2D {x.shape=}')
        x = self.global_avg_pool(x)
        # print(f'avg pool  {x.shape=}')
        x = torch.flatten(x, start_dim=1)
        x = self.dropout_2(x)
        # print(f'flatten {x.shape=}')
        x = self.fc2(x)
        # print(f'fc2 {x.shape=}')
        # x = self.fc3(x)
        # print(f'fc3 {x.shape=}')
        return x
