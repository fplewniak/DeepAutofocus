import collections

import torch
from torch import nn

from layers.filters import LaplacianFilterLayer
from models import Nonlinear


class ResNet(nn.Module):
    """
    Complete ResNet architecture
    Can be configured for different depths (18, 34, 50, 101, 152)
    """

    def __init__(self, block, layers=None, planes=None, input_size=None, batch_size=None):
        super(ResNet, self).__init__()

        if layers is None:
            layers = [2, 2, 2, 2]
        if planes is None:
            planes = [64, 128, 256, 512]
        # Initial number of channels
        self.inplanes = planes[0]

        self.laplacian = LaplacianFilterLayer()

        # Initial convolutional layer
        # Converts 3-channel input to 64 channels
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.bn2 = nn.BatchNorm2d(planes[-1])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.5)

        # Max pooling to reduce spatial dimensions
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four groups of residual blocks
        # Each group handles a different feature resolution
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)

        self.blocks = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer1', self.layer1),
                    ('layer2', self.layer2),
                    ('layer3', self.layer3),
                    ('layer4', self.layer4),
                ]
            )
        )

        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.regression = nn.Sequential(
            collections.OrderedDict(
                [
                    # ('nonlinear', Nonlinear.layers[nonlinear]()),
                    ('linear1', nn.LazyLinear(1)),
                ]
            )
        )
        self.dropout2d = nn.Dropout2d(0.1)

        if input_size is not None:
            dummy =  torch.ones(batch_size, 1, input_size, input_size)
            self.forward(dummy)
        # Initialize weights properly
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a group of residual blocks

        Args:
            block: The type of residual block to use
            planes: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for the first block (for downsampling)
        """
        downsample = None

        # Create downsampling layer if needed
        # This happens when we change the number of channels or spatial size
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        # First block might need downsampling
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # Remaining blocks don't need downsampling
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial processing
        x = self.laplacian(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through residual block groups
        x = self.blocks(x)
        x = self.dropout2d(x)

        # Final classification
        x = self.avgpool(x)
        # x = self.bn2(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.regression(x)

        return x

class ResNetreg(nn.Module):
    """
    Complete ResNet architecture
    Can be configured for different depths (18, 34, 50, 101, 152)
    """

    def __init__(self, block, layers=None, planes=None, input_size=None, batch_size=16,
                 nonlinear='LeakyReLU', dropout: float = 0.5):
        super(ResNetreg, self).__init__()

        if layers is None:
            layers = [2, 2, 2, 2]
        if planes is None:
            planes = [64, 128, 256, 512]
        # Initial number of channels
        self.inplanes = planes[0]

        self.laplacian = LaplacianFilterLayer()

        # Initial convolutional layer
        # Converts 3-channel input to 64 channels
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.bn2 = nn.BatchNorm2d(planes[-1])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        # Max pooling to reduce spatial dimensions
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four groups of residual blocks
        # Each group handles a different feature resolution
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)

        self.blocks = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer1', self.layer1),
                    ('layer2', self.layer2),
                    ('layer3', self.layer3),
                    ('layer4', self.layer4),
                ]
            )
        )

        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.regression = nn.Sequential(
            collections.OrderedDict(
                [
                    ('nonlinear', Nonlinear.layers[nonlinear]()),
                    ('linear1', nn.LazyLinear(1)),
                ]
            )
        )
        self.dropout2d = nn.Dropout2d(0.1)
        if input_size is not None:
            dummy =  torch.ones(batch_size, 1, input_size, input_size)
            self.forward(dummy)
        # Initialize weights properly
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a group of residual blocks

        Args:
            block: The type of residual block to use
            planes: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for the first block (for downsampling)
        """
        downsample = None

        # Create downsampling layer if needed
        # This happens when we change the number of channels or spatial size
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        # First block might need downsampling
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # Remaining blocks don't need downsampling
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial processing
        x = self.laplacian(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through residual block groups
        x = self.blocks(x)

        # Regression head
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.regression(x)

        return x