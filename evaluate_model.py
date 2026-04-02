import argparse
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import models
from torchvision.transforms import transforms, v2
from torch.utils.data import DataLoader

from loss_functions import WeighedMSELoss
from datasets import FocusImageDataset
from matplotlib import pyplot as plt

def get_params(argv):
    parser = argparse.ArgumentParser(description='Evaluate model.')

    parser.add_argument('--model', metavar='STR', help='model file name', type=str, required=True)
    parser.add_argument('--gt', metavar='STR', help='Ground truth file name', type=str, required=True)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)
    parser.add_argument('--crop', help='toggle crop at the centre instead of resizing', action='store_true')
    parser.add_argument('--image_size', metavar='INT', help='size of image', type=int, default=512)
    parser.add_argument('--title', metavar='STR', help='Plot title', type=str, default=None)
    parser.add_argument('--out_stats', metavar='STR', help='Save stats as csv', type=str, default=None)
    parser.add_argument('--out_data', metavar='STR', help='Save predictions vs ground truth as csv', type=str, default=None)
    parser.add_argument('--savefig', metavar='FILE', help='Save plot to file', default=None)
    parser.add_argument('--weighed_loss', metavar='STR', help='weighing loss', choices=['gauss', 'lorentz', 'plain'],
                        default=None)

    a = parser.parse_args()

    return a.model, a.batch_size, a.image_size, a.title, a.gt, a.crop, a.out_stats, a.out_data, a.savefig, a.weighed_loss

def test_loop(test_loader, model, loss_fn, device, name):
    model.eval()
    test_loss = 0.0
    comparison = []
    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.squeeze(model(images))
            if outputs.dim() == 0:
                outputs = torch.unsqueeze(outputs, 0)
            comparison += list(zip(outputs, labels, filenames))
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
    print(f'{name} average loss: {avg_test_loss}')

    return comparison

if __name__ == '__main__':
    (model_name, batch_size, image_size, title, ground_truth_file, crop,
     out_stats, out_data, savefig, weighed_loss) = get_params(sys.argv[1:])
    print(out_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # torch.set_default_device(device)
    # torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    image_sizing = transforms.CenterCrop((image_size, image_size)) if crop else v2.Resize(image_size)

    transform = transforms.Compose(
            [transforms.ToTensor(),
             v2.ToDtype(torch.float, scale=True),
             v2.functional.autocontrast,
             # transforms.Normalize((0.5,), (0.5,)),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             image_sizing
             # models.ViT_B_16_Weights.DEFAULT.transforms()
             ])

    test_df = pd.read_csv(ground_truth_file, header=0, names=['filename', 'deltaz', 'pos'])

    test_list = list(zip(test_df.filename, test_df.deltaz))

    test_dataset = FocusImageDataset(test_list, transform, None)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f'Test dataset: {len(test_loader)}')

    model = torch.jit.load(model_name)
    # model = torch.jit.load('Resnet18Florian_10epochs_batch8.pt')

    if weighed_loss is not None:
        criterion = WeighedMSELoss(method=weighed_loss)
    else:
        criterion = nn.MSELoss()

    positions = {}

    comparison = test_loop(test_loader, model, criterion, device, 'Evaluation')
    gt_vs_pred_test = pd.DataFrame([[p.item(), gt.item(), f] for p, gt, f in comparison], columns=['pred', 'gt', 'filename'])
    gt_vs_pred_test['error'] = gt_vs_pred_test['pred'] - gt_vs_pred_test['gt']

    if out_data is not None:
        gt_vs_pred_test.to_csv(out_data, sep=',', header=True, index=False)

    median_df = gt_vs_pred_test.groupby('gt', as_index=False).quantile(0.5, numeric_only=True)
    stats_df = median_df

    first_quant = gt_vs_pred_test.groupby('gt', as_index=False).quantile(0.05, numeric_only=True,)
    stats_df = stats_df.merge(first_quant, on='gt', suffixes=('', '_quant5'))

    second_quant = gt_vs_pred_test.groupby('gt', as_index=False).quantile(0.95, numeric_only=True,)
    stats_df = stats_df.merge(second_quant, on='gt', suffixes=('', '_quant95'))

    if out_stats is not None:
        stats_df.to_csv(out_stats, sep=',', header=True, index=False)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    if title is None:
        title = f'{model_name}'
    fig.suptitle(title)
    # axes[0].set_title('Test')
    axes[0].set(xlim=(-8, 8), ylim=(-8, 8))
    axes[0].axline((0, 0), slope=1, linestyle='--', color='k')
    # gt_vs_pred_test.plot.scatter('gt', 'pred', ax=axes[0], grid=True, markersize=1.5, alpha=0.3)
    axes[0].plot('gt', 'pred', 'b.', markersize=2, alpha=0.3, data=gt_vs_pred_test)
    axes[0].grid(visible=True, axis='both', which='major')
    # axes[0].set_aspect('equal', adjustable='datalim', anchor='S')
    axes[0].set_aspect('equal', anchor='S')
    # stats_df.plot('gt', ['error_quant5', 'error', 'error_quant95'], kind='line',
    #               style={'error_quant5': ':', 'error': '-', 'error_quant95': ':'}, ax=axes[1], grid=True)
    # stats_df.plot('gt', 'error', kind='line', style={'error': '-'}, ax=axes[1], grid=True)
    axes[1].set(xlim=(-8, 8), ylim=(-8, 8))
    axes[1].plot('gt', 'error', 'k-', data=stats_df)
    axes[1].plot('gt', 'error_quant5', 'k:', data=stats_df, alpha=0.3)
    axes[1].plot('gt', 'error_quant95', 'k:', data=stats_df, alpha=0.3)
    axes[1].fill_between('gt', 'error_quant5', 'error_quant95', alpha=0.2, data=stats_df)
    axes[1].plot( 'gt', 'error', 'b.', markersize=1, data=gt_vs_pred_test, alpha=0.1)
    # axes[1].legend(['quant 5%', 'median', 'quant 95%'])
    # axes[1].legend(['median'])
    axes[1].grid(visible=True, axis='both', which='major')
    # axes[1].set_aspect('equal', adjustable='datalim', anchor='S')
    axes[1].set_aspect('equal', anchor='S')
    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()
    print("Done.")
