import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fviz


class LaplacianFilterLayer(nn.Module):
    def __init__(self, blur= False):
        super(LaplacianFilterLayer, self).__init__()
        self.blur = blur
        kernel = [[0, -1.0, 0],
                  [-1.0,  4.0, -1.0],
                  [0, -1.0, 0]]
        self.weight = nn.Parameter(data=torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        if self.blur:
            x = Fviz.gaussian_blur(x, [1])
        x = F.conv2d(x, self.weight, stride=1, padding=1)
        return x + 1e-6


class SobelFilterLayer(nn.Module):
    def __init__(self, blur= False):
        super(SobelFilterLayer, self).__init__()
        self.blur = blur
        kernel_y = [[-0.5, -1, -0.5],
                    [0, 0, 0],
                    [0.5, 1, 0.5]]
        kernel_x = [[-0.5, 0, 0.5],
                    [-1.0, 0, 1.0],
                    [-0.5, 0, 0.5]]
        self.weight_x = nn.Parameter(data=torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.weight_y = nn.Parameter(data=torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        if self.blur:
            x = Fviz.gaussian_blur(x, [1])
        x_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        x_y = F.conv2d(x, self.weight_y, stride=1, padding=1)

        return torch.sqrt(torch.pow(x_x, 2) + torch.pow(x_y, 2) + 1e-6)