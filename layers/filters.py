import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class FilterLayer(nn.Module):
    def __init__(self, kernels, blur=False):
        super(FilterLayer, self).__init__()
        self.blur = blur
        if not isinstance(kernels, tuple):
            kernels = (kernels,)
        self.weights = []
        for kernel in kernels:
            self.weights.append(nn.Parameter(data=torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0), requires_grad=False))

    def forward(self, x):
        if self.blur:
            x = torchvision.transforms.GaussianBlur(1)(x)
        out = torch.pow(F.conv2d(x, self.weights[0], stride=1, padding=1), 2)
        if len(self.weights) > 1:
            for w in self.weights[1:]:
                out +=  torch.pow(F.conv2d(x, w, stride=1, padding=1), 2)
        return torch.sqrt(out + 1e-6)


class SobelLayer(FilterLayer):
    def __init__(self):
        kernel_v = [[-0.5, -1, -0.5],
                    [ 0,    0,  0  ],
                    [ 0.5,  1,  0.5]]
        kernel_h = [[-0.5,  0,  0.5],
                    [-1.0,  0,  1.0],
                    [-0.5,  0,  0.5]]
        super(SobelLayer, self).__init__((kernel_v, kernel_h))

class LaplacianLayer(FilterLayer):
    def __init__(self):
        kernel = [[0, -1.0, 0],
                  [-1.0,  0, -1.0],
                  [0, -1.0, 0]]
        super(LaplacianLayer, self).__init__(kernel, blur=True)