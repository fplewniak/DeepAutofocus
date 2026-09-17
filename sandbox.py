import cv2
import numpy as np
import skimage
import torch
from matplotlib import pyplot as plt
from tifffile import tifffile

from layers.filters import SobelLayer, LaplacianLayer, SobelLike

# class GradLoss(nn.Module):
#
#     def __init__(self):
#         super(GradLoss, self).__init__()
#         self.loss = nn.L1Loss()
#         self.grad_layer = SobelLayer()
#
#     def forward(self, output, gt_img):
#         output_grad = self.grad_layer(output)
#         gt_grad = self.grad_layer(gt_img)
#         return self.loss(output_grad, gt_grad)


if __name__ == '__main__':
    net = LaplacianLayer()

    fig, ax = plt.subplots()

    filename = '/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/_2/8-Pos004_003/img_channel000_position019_time000000000_z036.tif'
    img = tifffile.imread(filename)
    img = skimage.util.img_as_float(img).astype(np.float32)
    img = torch.DoubleTensor(img).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)

    print(img.shape)
    print(torch.max(img), torch.min(img))
    img = net(img)
    img = img[0, :, :, :].permute(1, 2, 0).numpy()
    img =  cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print(np.max(img), np.min(img))

    ax.imshow(img, cmap='gray')

    plt.show()