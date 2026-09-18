import argparse
import sys

import torch
from torchinfo import summary

from ResNet18 import ResNet18Model, ResNet18Model2DenseLayers, ResNet18Model3DenseLayers
from ResNet34 import ResNet34Model
from ResNet50 import ResNet50Reg
from MobileNetV3 import MobileNetV3_l, MobileNetV3_s
from models.LaplacianNet import LaplacianNet
from models.SobelNet import SobelNet


def get_params(argv):
    parser = argparse.ArgumentParser(description='Evaluate model.')

    parser.add_argument('--model', metavar='STR', help='Model',
                        choices=['ResNet18', 'ResNet18_2Dense', 'ResNet18_3Dense', 'ResNet34', 'ResNet50', 'MobileNetV3_l',
                                 'MobileNetV3_s', 'SobelNet', 'LaplacianNet'], default='SobelNet')
    parser.add_argument('--channels', metavar='INT', help='Number of channels', default=3, type=int)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)
    parser.add_argument('--image_size', metavar='INT', help='size of image', type=int, default=512)

    a = parser.parse_args()

    return a.model, a.batch_size, a.image_size, a.channels

if __name__ == '__main__':
    model_name, batch_size, image_size, channels = get_params(sys.argv[1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    match model_name:
        case 'ResNet18':
            model = ResNet18Model()
        case 'ResNet18_2Dense':
            model = ResNet18Model2DenseLayers()
        case 'ResNet18_3Dense':
            model = ResNet18Model3DenseLayers()
        case 'ResNet34':
            model = ResNet34Model()
        case 'ResNet50':
            model = ResNet50Reg()
        case 'MobileNetV3_l':
            model = MobileNetV3_l()
        case 'MobileNetV3_s':
            model = MobileNetV3_s()
        case 'SobelNet':
            model = SobelNet()
        case 'LaplacianNet':
            model = LaplacianNet()
        case _:
            raise NotImplementedError(f'Model {model_name} is not implemented')

    summary(model, (batch_size, channels, image_size, image_size, ), device='cpu')
