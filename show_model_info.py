import argparse
import sys

import torch
from torchinfo import summary

from models.LaplacianBlocks import LaplacianBlocks
from models.LaplacianNet import LaplacianNet
from models.SobelNet import SobelNet


def get_params(argv):
    parser = argparse.ArgumentParser(description='Evaluate model.')

    parser.add_argument('--model', metavar='STR', help='Model',
                        choices=['SobelNet', 'LaplacianNet', 'LaplacianBlocks' ], default='SobelNet')
    parser.add_argument('--channels', metavar='INT', help='Number of channels', default=3, type=int)
    parser.add_argument('--blocks', metavar='INT', help='Number of blocks', default=2, type=int)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)
    parser.add_argument('--image_size', metavar='INT', help='size of image', type=int, default=512)
    parser.add_argument('--depth', metavar='INT', help='depth of model tree', type=int, default=3)

    a = parser.parse_args()

    return a.model, a.batch_size, a.image_size, a.channels, a.blocks, a.depth

if __name__ == '__main__':
    model_name, batch_size, image_size, channels, blocks, depth = get_params(sys.argv[1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    match model_name:
        case 'SobelNet':
            model = SobelNet()
        case 'LaplacianNet':
            model = LaplacianNet(n_blocks=blocks, channels=channels, input_size=image_size)
        case 'LaplacianBlocks':
            model = LaplacianBlocks(n_blocks=blocks, channels=channels).to(device)
        case _:
            raise NotImplementedError(f'Model {model_name} is not implemented')

    summary(model, (batch_size, 1, image_size, image_size, ), depth=depth, device='cpu')
