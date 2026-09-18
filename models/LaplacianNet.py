import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.filters import LaplacianFilterLayer


class LaplacianNet(nn.Module):
    """
    Class for simple neural network
    """
    def __init__(self, initw: str = 'kaiming'):
        super(LaplacianNet, self).__init__()
        self.laplacian = LaplacianFilterLayer()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        match initw:
            case 'kaiming':
                torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
            case 'xavier':
                torch.nn.init.xavier_normal_(self.conv1.weight, gain=1)
                torch.nn.init.xavier_normal_(self.conv2.weight, gain=1)
        self.dropout = nn.Dropout(0.5)
        self.regression = nn.Linear(254016, 1)

    def forward(self, x):
        """
        Forward loop
        :param x: image data
        :return: predicted delta z
        """
        # Pass data through Sobel layer
        x = self.laplacian(x)
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