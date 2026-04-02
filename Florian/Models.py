import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    """
    Class for simple neural network
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.regression = nn.Linear(968256, 1)

    def forward(self, x):
        """
        Forward loop
        :param x: image data
        :return: predicted delta z
        """
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        output = self.regression(x)
        return output

class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet34Model(nn.Module):
    def __init__(self):
        super(ResNet34Model, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()

        # Load ResNet50 (since PyTorch lacks ResNet50V2)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 1)  # BiLSTM output size = 2 * hidden_size

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc(x)
        return x