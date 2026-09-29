import collections

import torch
import torch.nn as nn

from layers.filters import LaplacianFilterLayer
from models import ConvBlocks, Nonlinear


class LaplacianBlocks(nn.Module):
    """
    Class for simple neural network
    """
    def __init__(self, initw: str = 'kaiming', n_blocks: int = 2,
                 nonlinear='LeakyReLU', nonlinearh: str = 'LeakyReLU',
                 channels: int = 8, input_size: int = 512, dropout: float = 0.5):
        super(LaplacianBlocks, self).__init__()
        self.laplacian = LaplacianFilterLayer()
        self.blocks = ConvBlocks(n_blocks=n_blocks, initw=initw, channels=channels, nonlinear=nonlinearh)
        # self.regression = nn.Sequential(
        #     nn.LazyLinear(128),
        #     Nonlinear.layers[nonlinear](),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 1)
        # )
        self.regression = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear1', nn.LazyLinear(128)),
                    ('nonlinear', Nonlinear.layers[nonlinear]()),
                    ('dropout', nn.Dropout(0.4)),
                    ('linear2', nn.Linear(128, 1)),
                ]
            )
        )
        # self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout2d(0.1)
        dummy =  torch.ones(16, 1, input_size, input_size)
        self.forward(dummy)
        match initw:
            case 'kaiming':
                nn.init.kaiming_normal_(self.regression[0].weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(self.regression[3].weight, mode='fan_in', nonlinearity='leaky_relu')
            case 'xavier':
                nn.init.xavier_normal_(self.regression[0].weight, gain=1)
                nn.init.xavier_normal_(self.regression[3].weight, gain=1)

    def forward(self, x):
        """
        Forward loop
        :param x: image data
        :return: predicted delta z
        """
        x = self.laplacian(x)
        x = self.blocks(x)
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.regression(x)
        return x