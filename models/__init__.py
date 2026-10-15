import collections

from torch import nn
import torch.nn.functional as F

class Nonlinear:
    layers = {
        'Identity': nn.Identity,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Mish': nn.Mish,
        'Sigmoid': nn.Sigmoid,
        'GLU': nn.GLU,
        'Tanhshrink': nn.Tanhshrink,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'LogSigmoid': nn.LogSigmoid,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'PReLU': nn.PReLU,
    }


class ConvUnit(nn.Module):
    def __init__(self, initw: str = 'kaiming', in_channels: int = 1, out_channels: int = 8, nonlinear: str = 'ReLU'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2)
        match initw:
            case 'kaiming':
                nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='conv2d')
            case 'xavier':
                nn.init.xavier_normal_(self.conv.weight, gain=1)
        self.conv_block = nn.Sequential(
            self.conv,
            # nn.Dropout2d(0.2),
            Nonlinear().layers[nonlinear](),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class ConvBlocks(nn.Module):
    def __init__(self, n_blocks: int = 2, initw: str = 'kaiming', channels: int = 8, nonlinear: str = 'ReLU'):
        super(ConvBlocks, self).__init__()
        self.n_channels = channels
        self.n_blocks = n_blocks
        self.blocks = nn.Sequential(
            collections.OrderedDict(
                [
                    (f'conv{i}', ConvUnit(initw, in_channels=max(i * channels, 1), out_channels=(i + 1) * channels, nonlinear=nonlinear))
                    for i in range(n_blocks)
                ]
            )
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34
    This block consists of two 3x3 convolutional layers
    with batch normalization and ReLU activation
    """
    expansion = 1  # Output channels multiplier

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer for matching dimensions
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # Store the input for the skip connection
        identity = x

        # First conv block: conv -> batch norm -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block: conv -> batch norm (no ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to identity if needed
        # This handles cases where input and output dimensions differ
        if self.downsample is not None:
            identity = self.downsample(x)

        # The key ResNet operation: add the skip connection
        out += identity

        # Apply ReLU after the addition
        out = F.relu(out)

        return out