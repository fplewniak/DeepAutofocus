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

from datasets import FocusImageDataset
from matplotlib import pyplot as plt

def get_params(argv):
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('--model', metavar='STR', help='model file name', type=str, required=True)
    parser.add_argument('--filelist', metavar='STR', help='CSV file containing the list of image files and'
                                                          ' the corresponding ground-truth delta Z value separated with a comma',
                        required=True, type=str)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)
    parser.add_argument('--crop', help='toggle crop at the centre instead of resizing', action='store_true')
    parser.add_argument('--image_size', metavar='INT', help='size of image (cropped at the centre)', type=int, default=512)
    parser.add_argument('--title', metavar='STR', help='Plot title', type=str, default=None)

    argscope = parser.parse_args()

    return argscope.model, argscope.batch_size, argscope.crop, argscope.image_size, argscope.title, argscope.filelist

def test_loop(test_loader, model, loss_fn, device, name):
    model.eval()
    test_loss = 0.0
    comparison = []
    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = (torch.squeeze(model(images)))
            comparison += list(zip(outputs, labels, filenames))
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'{name} average loss: {avg_test_loss}')

    return comparison

if __name__ == '__main__':
    model_name, batch_size, crop, image_size, title, filelist = get_params(sys.argv[1:])

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

    all_df = pd.read_csv(filelist, header=0, names=['filename', 'deltaz'])
    print(all_df)
    group_indices = np.arange(len(all_df)) // 61
    unique_groups = np.unique(group_indices)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n1 = int(n_groups * 0.6)
    n2 = int(n_groups * 0.2)

    train_grp = unique_groups[:n1]
    val_grp = unique_groups[n1:n1 + n2]
    test_grp = unique_groups[n1 + n2:]

    all_df['group'] = group_indices

    # train_df = pd.read_csv('/data3/DeepAutoFocus/20250417/train_noPos16_25_24.csv')
    # train_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/train.csv', names=['filename', 'deltaz'])
    df = all_df[all_df['group'].isin(train_grp)]
    # df.to_csv('train.csv', sep=',', header=True, index=False)
    train_df = df.drop(columns='group')
    train_df.to_csv('train.csv', sep=',', header=True, index=False)
    train_list = list(zip(train_df.filename, train_df.deltaz))
    train_dataset = FocusImageDataset(train_list, transform, None)

    # val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/val.csv')
    # val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/val.csv', names=['filename', 'deltaz'])
    df = all_df[all_df['group'].isin(val_grp)]
    # df.to_csv('val.csv', sep=',', header=True, index=False)
    val_df = df.drop(columns='group')
    val_df.to_csv('val.csv', sep=',', header=True, index=False)
    val_list = list(zip(val_df.filename, val_df.deltaz))
    val_dataset = FocusImageDataset(val_list,  transform, None)

    # test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/test.csv')
    # test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/test.csv', names=['filename', 'deltaz'])
    df = all_df[all_df['group'].isin(test_grp)]
    # df.to_csv('test.csv', sep=',', header=False, index=False)
    test_df = df.drop(columns='group')
    test_df.to_csv('test.csv', sep=',', header=False, index=False)
    test_list = list(zip(test_df.filename, test_df.deltaz))
    test_dataset = FocusImageDataset(test_list, transform, None)

    # train_df = pd.read_csv('/data3/DeepAutoFocus/20250417/train_noPos16_25_24.csv')
    # train_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/train.csv', names=['filename', 'deltaz'])
    # train_list = list(zip(train_df.filename, train_df.deltaz))
    # train_dataset = FocusImageDataset(train_list, '/data3/DeepAutoFocus/20250417', transform, None)
    #
    # # val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/val.csv')
    # val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/val.csv', names=['filename', 'deltaz'])
    # val_list = list(zip(val_df.filename, val_df.deltaz))
    # val_dataset = FocusImageDataset(val_list, '/data3/DeepAutoFocus/20250417', transform, None)
    #
    # # test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/test.csv')
    # test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/test.csv', names=['filename', 'deltaz'])
    # test_list = list(zip(test_df.filename, test_df.deltaz))
    # test_dataset = FocusImageDataset(test_list, '/data3/DeepAutoFocus/20250417', transform, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f'Training dataset: {len(train_loader)}')
    print(f'Validation dataset: {len(val_loader)}')
    print(f'Test dataset: {len(test_loader)}')

    model = torch.jit.load(model_name)
    # model = torch.jit.load('Resnet18Florian_10epochs_batch8.pt')

    #### Training the model ##################"
    criterion = nn.MSELoss()

    positions = {}
    comparison = test_loop(train_loader, model, criterion, device, 'Training')
    gt_vs_pred_train = pd.DataFrame([[p.item(), gt.item(), f] for p, gt, f in comparison], columns=['pred', 'gt', 'filename'])
    gt_vs_pred_train.replace(to_replace=r'.*(Pos\d+).*', value=r'\1', regex=True)
    print(gt_vs_pred_train)

    comparison = test_loop(val_loader, model, criterion, device, 'Validation')
    gt_vs_pred_val = pd.DataFrame([[p.item(), gt.item(), f] for p, gt, f in comparison], columns=['pred', 'gt', 'filename'])
    print(gt_vs_pred_val)

    comparison = test_loop(test_loader, model, criterion, device, 'Test')
    gt_vs_pred_test = pd.DataFrame([[p.item(), gt.item(), f] for p, gt, f in comparison], columns=['pred', 'gt', 'filename'])
    print(gt_vs_pred_test)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    if title is None:
        title = f'{model}'
    fig.suptitle(title)
    axes[0].set_title('Training')
    axes[0].set(xlim=(-8, 8), ylim=(-8, 8))
    axes[0].axline((0, 0), slope=1, linestyle='--', color='k')
    axes[1].set_title('Validation')
    axes[1].set(xlim=(-8, 8), ylim=(-8, 8))
    axes[1].axline((0, 0), slope=1, linestyle='--', color='k')
    axes[2].set_title('Test')
    axes[2].set(xlim=(-8, 8), ylim=(-8, 8))
    axes[2].axline((0, 0), slope=1, linestyle='--', color='k')
    gt_vs_pred_train.plot.scatter('gt', 'pred', ax=axes[0], grid=True)
    gt_vs_pred_val.plot.scatter('gt', 'pred', ax=axes[1], grid=True)
    gt_vs_pred_test.plot.scatter('gt', 'pred', ax=axes[2], grid=True)
    # axs.axline((0, 0), slope=1, linestyle='--', color='k')
    plt.show()
    print("Done.")
