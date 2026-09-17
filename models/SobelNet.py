import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.filters import SobelLayer


class SobelNet(nn.Module):
    """
    Class for simple neural network
    """
    def __init__(self):
        super(SobelNet, self).__init__()
        self.sobel = SobelLayer()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.regression = nn.Linear(254016, 1)

    def forward(self, x):
        """
        Forward loop
        :param x: image data
        :return: predicted delta z
        """
        # Pass data through Sobel layer
        x = self.sobel(x)
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for w in self.sobel.weights:
            w = w.to(*args, **kwargs)
        return self