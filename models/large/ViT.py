import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2


class ViTRegressor(nn.Module):
    def __init__(self):
        super(ViTRegressor, self).__init__()
        # Load pretrained ViT base
        self.vit = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)

        # Freeze all layers if needed
        # for param in self.vit.parameters():
        #     param.requires_grad = False

        # Replace the classifier head
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.vit(x).squeeze(1)  # shape: (batch,)

class Vit_l_32(nn.Module):
    def __init__(self):
        super(Vit_l_32, self).__init__()

        # resnet = models.resnet18(weights=None)
        vit_l_32 = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.vit_l_32 = nn.Sequential(*list(vit_l_32.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.vit_l_32(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Vit_l_32_2DenseLayers(nn.Module):
    def __init__(self):
        super(Vit_l_32_2DenseLayers, self).__init__()

        vit_l_32 = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.vit_l_32 = nn.Sequential(*list(vit_l_32.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.vit_l_32(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Vit_b_16(nn.Module):
    def __init__(self):
        super(Vit_b_16, self).__init__()

        # resnet = models.resnet18(weights=None)
        vit_b_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.vit_b_16 = nn.Sequential(*list(vit_b_16.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        x = self.vit_b_16(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Vit_b_16_2DenseLayers(nn.Module):
    def __init__(self):
        super(Vit_b_16_2DenseLayers, self).__init__()

        vit_b_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        ## Change input layer for only one channel
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.vit_b_16 = nn.Sequential(*list(vit_b_16.children())[:-2])  # Remove FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.vit_b_16(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
