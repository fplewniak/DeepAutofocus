import torch
from torch import nn
import torch.nn.functional as F


class WeighedMSELoss(nn.Module):
    """
    Computes a weighed loss where larger predicted values are less important as long as they are in the right
     direction (same sign as ground truth). This means that precision should be favoured around the correct focus and
     diverging predictions are penalized.
    """
    def __init__(self, method = 'gauss', l = None):
        super(WeighedMSELoss, self).__init__()
        self.method = method
        self.l = l

    def forward(self, predictions, targets):
        error = (predictions - targets)
        same_sign = F.sigmoid(predictions * targets)

        match self.method:
            case 'gauss':
                if self.l is None:
                    self.l = 0.01
                weights = same_sign * torch.exp(-abs(self.l) * error ** 2) + (1 - same_sign)
            case 'lorentz':
                if self.l is None:
                    self.l = 0.05
                weights = same_sign * (1 / (1 + abs(self.l) * error**2)) + (1 - same_sign)
            case 'plain':
                if self.l is None:
                    self.l = 0.25
                weights = (same_sign + self.l * (1 - same_sign) * error**2)
        mse_loss = torch.mean(weights * error ** 2)
        return mse_loss

