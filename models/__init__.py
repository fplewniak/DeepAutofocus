import collections

from torch import nn


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
