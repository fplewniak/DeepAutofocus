import argparse
import sys

import torch
from torchview import draw_graph
import graphviz

from Regression.ResNet18 import ResNet18Model
from models import ResidualBlock
from models.LaplacianBlocks import LaplacianBlocks
from models.LaplacianNet import LaplacianNet
from models.ResNet import ResNet
from models.SobelNet import SobelNet


def get_params(argv):
    parser = argparse.ArgumentParser(description='Draw model graph.')

    parser.add_argument('--model', metavar='STR', help='Model',
                        choices=['SobelNet', 'LaplacianNet', 'LaplacianBlocks', 'ResNet', 'ResNet18Model' ], default='SobelNet')
    parser.add_argument('--channels', metavar='INT', help='Number of channels', default=3, type=int)
    parser.add_argument('--blocks', metavar='INT', help='Number of blocks', default=2, type=int)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)
    parser.add_argument('--image_size', metavar='INT', help='size of image', type=int, default=512)
    parser.add_argument('--depth', metavar='INT', help='depth of model tree', type=int, default=3)
    parser.add_argument('--layers', help='numbers of residual blocks per layer', type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument('--planes', help='numbers of planes in each resnet layer', type=int, nargs=4, default=[64, 128, 256, 512])

    a = parser.parse_args()

    return a.model, a.batch_size, a.image_size, a.channels, a.blocks, a.depth, a.layers, a.planes

if __name__ == '__main__':
    model_name, batch_size, image_size, channels, blocks, depth, layers, planes = get_params(sys.argv[1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    channels = 1

    match model_name:
        case 'SobelNet':
            model = SobelNet()
        case 'LaplacianNet':
            model = LaplacianNet(n_blocks=blocks, channels=channels, input_size=image_size)
        case 'LaplacianBlocks':
            model = LaplacianBlocks(n_blocks=blocks, channels=channels).to(device)
        case 'ResNet':
            model = ResNet(ResidualBlock, layers, planes, input_size=image_size, batch_size=batch_size).to(device)
        case 'ResNet18Model':
            model = ResNet18Model().to(device)
            channels = 3
        case _:
            raise NotImplementedError(f'Model {model_name} is not implemented')

    # summary(model, (batch_size, channels, image_size, image_size, ), depth=depth, device='cpu')
    model_graph = draw_graph(model, input_size=(batch_size, channels, image_size, image_size), device='meta',
                             save_graph=True, filename=f'{model_name}', depth=depth,
                             hide_inner_tensors=True, hide_module_functions=False, roll=True, expand_nested=True)
