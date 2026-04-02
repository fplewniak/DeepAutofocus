import sys
import argparse

import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision.transforms import transforms, v2
from torch.utils.data import DataLoader

from loss_functions import WeighedMSELoss
from ResNet18 import ResNet18Model, ResNet18Model2DenseLayers, ResNet18Model3DenseLayers
from ResNet34 import ResNet34Model
from ResNet50 import ResNet50Reg
from MobileNetV3 import MobileNetV3_l, MobileNetV3_s
from datasets import FocusImageDataset
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def get_params(argv):
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('--model', metavar='STR', help='Model',
                        choices=['ResNet18', 'ResNet18_2Dense', 'ResNet18_3Dense', 'ResNet34', 'ResNet50', 'MobileNetV3_l',
                                 'MobileNetV3_s'], default='ResNet18'),
    parser.add_argument('--freeze', help='Freeze pretrained weights and only train the regression head',
                        action='store_true')
    parser.add_argument('--optim', metavar='STR', help='optimizer', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'], default='AdamW')
    parser.add_argument('--epochs', metavar='INT', help='number of epochs', type=int, default=10)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)
    parser.add_argument('--lr', metavar='FLOAT', help='learning rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', metavar='FLOAT', help='weight decay', type=float, default=0)
    parser.add_argument('--out', metavar='FILE', help='The output files prefix', default=None)
    parser.add_argument('--crop', help='toggle crop at the centre instead of resizing', action='store_true')
    parser.add_argument('--image_size', metavar='INT', help='size of image (cropped at the centre)', type=int, default=512)
    parser.add_argument('--lambda1', metavar='FLOAT', help='L1 lambda value', type=float, default=0.0)
    parser.add_argument('--lambda2', metavar='FLOAT', help='L2 lambda value', type=float, default=0.0)
    parser.add_argument('--savefig', metavar='FILE', help='Save plot to file', default=None)
    parser.add_argument('--title', metavar='STR', help='Plot title', type=str, default=None)
    parser.add_argument('--weighed_loss', metavar='STR', help='weighing loss', choices=['gauss', 'lorentz', 'plain'],
                        default=None)

    argscope = parser.parse_args()

    return (argscope.model, argscope.epochs, argscope.batch_size, argscope.out, argscope.optim, argscope.lr, argscope.weight_decay,
            argscope.crop, argscope.image_size, argscope.lambda1, argscope.lambda2, argscope.savefig, argscope.title,
            argscope.weighed_loss, argscope.freeze)


def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device, lambda1, lambda2):
    model.train()
    running_loss = 0.0
    for images, labels, filename in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = torch.squeeze(model(images))
        loss = loss_fn(outputs, labels)
        # Apply L1 & L2 regularization
        loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                 + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()


    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, filename in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.squeeze(model(images))
            loss = loss_fn(outputs, labels)
            loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                     + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())
            val_loss += loss.item()

        avg_train_loss = running_loss / len(training_loader)
        avg_val_loss = val_loss / len(validation_loader)

    return {'train loss': avg_train_loss, 'val loss': avg_val_loss}


def fix_filename(filename):
    # Regular expression to match /PosX/PosX/ where X is one or two digits
    pattern = r'(/Pos\d{1,2})\/\1/'
    return re.sub(pattern, r'\1/', filename)


if __name__ == '__main__':
    (model_name, n_epochs, batch_size, outprefix, optim_name, lr, weight_decay, crop,
     image_size, lambda1, lambda2, savefig, title, weighed_loss, freeze) = get_params(sys.argv[1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # torch.set_default_device(device)
    # torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.fp32_precision = "tf32"

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

    all_df = pd.read_csv('/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/CorrectFocusOnCells.csv', header=0, names=['filename', 'deltaz'])
    # print(all_df)
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
    df.to_csv('train_training.csv', sep=',', header=True, index=False)
    train_df = df.drop(columns='group')
    # print(f'{train_df=}')
    train_list = list(zip(train_df.filename, train_df.deltaz))
    train_dataset = FocusImageDataset(train_list, transform, None)

    # val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/val.csv')
    # val_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/val.csv', names=['filename', 'deltaz'])
    df = all_df[all_df['group'].isin(val_grp)]
    df.to_csv('val_training.csv', sep=',', header=True, index=False)
    val_df = df.drop(columns='group')
    # print(f'{val_df=}')
    val_list = list(zip(val_df.filename, val_df.deltaz))
    val_dataset = FocusImageDataset(val_list, transform, None)

    # test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/test.csv')
    ## test_df = pd.read_csv('/data3/DeepAutoFocus/20250417/BetterDispatch/test.csv', names=['filename', 'deltaz'])
    df = all_df[all_df['group'].isin(test_grp)]
    df.to_csv('test_training.csv', sep=',', header=False, index=False)
    test_df = df.drop(columns='group')
    # test_list = list(zip(test_df.filename, test_df.deltaz))
    # test_dataset = FocusImageDataset(test_list, transform, None)

    # batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f'Training dataset: {len(train_loader)}')
    print(f'Validation dataset: {len(val_loader)}')

    match model_name:
        case 'ResNet18':
            model = ResNet18Model(freeze=freeze).to(device)
        case 'ResNet18_2Dense':
            model = ResNet18Model2DenseLayers().to(device)
        case 'ResNet18_3Dense':
            model = ResNet18Model3DenseLayers().to(device)
        case 'ResNet34':
            model = ResNet34Model(freeze=freeze).to(device)
        case 'ResNet50':
            model = ResNet50Reg(freeze=freeze).to(device)
        case 'MobileNetV3_l':
            model = MobileNetV3_l().to(device)
        case 'MobileNetV3_s':
            model = MobileNetV3_s().to(device)
        case _:
            raise NotImplementedError(f'Model {model_name} is not implemented')

    # summary(model, input_size=(batch_size, 3, image_size, image_size))

    #### Training the model ##################"
    if weighed_loss is not None:
        criterion = WeighedMSELoss(method=weighed_loss)
    else:
        criterion = nn.MSELoss()

    match optim_name:
        case 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        case 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        case 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        case 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    history = []

    writer = SummaryWriter()


    try:
        min_val_loss = torch.finfo(torch.float).max
        for epoch in range(n_epochs):
            history.append(train_loop(train_loader, val_loader, model, criterion, optimizer, device, lambda1, lambda2))
            if history[-1]['val loss'] < min_val_loss:
                min_val_loss = history[-1]['val loss']
                model_scripted = torch.jit.script(model)
                model_scripted.save(f'{outprefix}_best_model.pt')
                print(
                    f"Saving best model at epoch {epoch + 1} with val loss {min_val_loss} and train loss {history[-1]['train loss']}")
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"Training Loss: {history[-1]['train loss']:.4f}, "
                  f"Validation Loss: {history[-1]['val loss']:.4f}, "
                  f"learning rate: {scheduler.get_last_lr()}, "
                  f" -- ({datetime.now().strftime('%H:%M:%S')})")
            writer.add_scalars('Loss',
                               {'Training Loss': history[-1]['train loss'],
                                'Validation Loss': history[-1]['val loss']},
                               epoch + 1)
            # Log the histogram of the model's weights
            for name, param in model.regression.named_parameters():
            # for name, param in model.named_parameters():
                writer.add_histogram(f'weights/{name}', param, epoch + 1)
                writer.add_histogram(f'gradients/{name}.grad', param.grad, epoch + 1)
                grad_norm = param.grad.norm()  # L2 norm
                writer.add_scalar(f'gradients/norm/{name}', grad_norm, epoch + 1)
                writer.add_scalar(f'norm_mean/{name}', grad_norm.mean(), epoch + 1)
                writer.add_scalar(f'norm_max/{name}', grad_norm.max(), epoch + 1)
                writer.add_scalar(f'norm_min/{name}', grad_norm.min(), epoch + 1)

            scheduler.step(history[-1]['val loss'])
    finally:
        ##################################################################
        # torch.save(model.state_dict(), 'ResNet18_reg.pth')
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(f'{outprefix}.pt')  # Save
        ##################################################################
        df = pd.DataFrame(history)
        plt.figure(1)
        plt.title(title)
        df['train loss'].plot()
        df['val loss'].plot()
        plt.legend()

        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()

        writer.flush()
        writer.close()
        #
        # print("Cleaning up DataLoader...")
        # cleanup_dataloader(train_loader)
        # cleanup_dataloader(val_loader)
        # cleanup_dataloader(test_loader)
        # monitor_threads_and_processes()
        # kill_all_pt_data_workers()
        print("Done.")
