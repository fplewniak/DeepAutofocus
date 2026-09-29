import gc
import multiprocessing
import os
import sys
import argparse

from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import TrialState
from torch import nn, optim
from torchvision.transforms import transforms, v2
from torch.utils.data import DataLoader

from loss_functions import WeightedMSELoss
from datasets import FocusImageDataset
from models.LaplacianBlocks import LaplacianBlocks
from models.LaplacianNet import LaplacianNet
from models.SobelNet import SobelNet

from train_model import train_loop


def get_params(argv):
    parser = argparse.ArgumentParser(description='Tune hyperparameters with optuna.')

    parser.add_argument('--model', metavar='STR', help='Model',
                        choices=['SobelNet', 'LaplacianNet', 'LaplacianBlocks'], default='SobelNet'),
    parser.add_argument('--filelist', metavar='STR', help='CSV file containing the list of image files and'
                                                          ' the corresponding ground-truth delta Z value separated with a comma',
                        required=True, type=str)

    # parser.add_argument('--config', metavar='STR', help='Optuna configuration', required=True, type=str)
    parser.add_argument('--epochs', metavar='INT', help='number of epochs', type=int, default=10)
    parser.add_argument('--crop', help='toggle crop at the centre instead of resizing', action='store_true')
    parser.add_argument('--image_size', metavar='INT', help='size of image (cropped at the centre)', type=int, default=512)
    parser.add_argument('--batch_size', metavar='INT', help='size of batch', type=int, default=16)

    parser.add_argument('--trials', metavar='INT', help='number of trials', type=int, default=10)
    parser.add_argument('--nonlinear', metavar='STR', help='Regression nonlinear layer',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'Identity'], default='ReLU')
    parser.add_argument('--nonlinearh', metavar='STR', help='Hidden nonlinear layer',
                        choices=['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'PReLU'], default='ReLU')


    a = parser.parse_args()

    return (a.model, a.filelist, a.trials, a.crop, a.image_size, a.batch_size, a.epochs, a.nonlinear)


class HyperparameterTuner:
    def __init__(self, model_name, filelist, n_trials, crop, image_size, batch_size, n_epochs, train_loader, val_loader):
        self.model_name = model_name
        self.filelist = filelist
        self.n_trials = n_trials
        self.crop = crop
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.path = '/home/fred/Projects/DeepFocus/optuna'
        os.makedirs(self.path, exist_ok=True)
        self.min_val_loss = torch.finfo(torch.float).max

    def tune_hyperparameters(self, n_trials: int = 100) -> optuna.Trial:
            """
            Runs an optuna study for hyperparameter tuning.
            """
            print('Starting Optuna study')

            study_name = f"{datetime.now().strftime("%y%m%d%H%M%s")}"
            os.makedirs(f'{self.path}/{study_name}', exist_ok=True)
            study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{self.path}/optuna.sqlite3",
                                        direction="minimize")
            print('Optimization of objective function')
            study.optimize(self.objective, n_trials=n_trials)
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            return trial

    def objective(self, trial: optuna.Trial) -> float:
        """
            The objective function. This method suggests values for the hyperparameters and then runs the train_model method, passing
            the trial variable thereto. It returns the metric that should be optimized by optuna.
            :param trial: the trial object
            :return: the value of the metric to optimize
            """
        print('Running objective function')
        parameters = {
            # 'optim_name': 'AdamW',
            # 'optim_name': trial.suggest_categorical('optim_name', ['AdamW', 'RMSprop']),
            # 'init_weights': trial.suggest_categorical('init_weights', ['kaiming', 'xavier']),
            # 'lr': trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            # 'L2': trial.suggest_float("L2", 1e-5, 1e-1, log=True),
            # 'weighted_loss': trial.suggest_categorical('weighted_loss', ['gauss', 'lorentz', 'plain']),
            'blocks': trial.suggest_int('blocks', 4, 7),
            'channels': trial.suggest_int('channels', 4, 8),
            'dropout': trial.suggest_float("dropout", 0.1, 0.5),
            # 'nonlinear': trial.suggest_categorical('nonlinear', ['ReLU', 'LeakyReLU']),
            # 'nonlinearh': trial.suggest_categorical('nonlinearh', ['ReLU', 'LeakyReLU']),
            'L1': 0.0,
            'optim_name': 'AdamW',
            'init_weights': 'kaiming',
            'lr': 0.00027760306206483,
            'L2': 0.0000148159971684519,
            'weighted_loss': 'lorentz',
            # 'blocks': 6,
            # 'channels': 8,
            'nonlinear': 'LeakyReLU',
            'nonlinearh': 'ReLU',
            'input_size': image_size,
        }

        val_loss, model = self.train_model(trial, parameters)
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        del model
        gc.collect()
        return self.min_val_loss

    def train_model(self, trial, parameters):
        match model_name:
            case 'SobelNet':
                model = SobelNet(parameters['init_weights']).to(device)
            case 'LaplacianNet':
                model = LaplacianNet(parameters['init_weights'],
                                        parameters['blocks'],
                                        parameters['nonlinear'],
                                        parameters['nonlinearh'],
                                        parameters['channels'],
                                        parameters['input_size'],
                                        parameters['dropout'],
                                     ).to(device)
            case 'LaplacianBlocks':
                model = LaplacianBlocks(parameters['init_weights'],
                                        parameters['blocks'],
                                        parameters['nonlinear'],
                                        parameters['nonlinearh'],
                                        parameters['channels'],
                                        parameters['input_size'],
                                        parameters['dropout'],
                                        ).to(device)
            case _:
                raise NotImplementedError(f'Model {model_name} is not implemented')

        # summary(model, input_size=(batch_size, 3, image_size, image_size))

        #### Training the model ##################"
        if parameters['weighted_loss'] is not None:
            criterion = WeightedMSELoss(method=parameters['weighted_loss'])
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])
        match parameters['optim_name']:
            case 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])
            case 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=parameters['lr'])
            case 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=parameters['lr'], momentum=0.9)
            case 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=parameters['lr'], momentum=0.9)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        history = []

        try:
            min_val_loss = torch.finfo(torch.float).max
            for epoch in range(n_epochs):
                history.append(
                    train_loop(self.train_loader, self.val_loader, model, criterion, optimizer, device,
                               parameters['L1'], parameters['L2']))
                if history[-1]['val loss'] < min_val_loss:
                    min_val_loss = history[-1]['val loss']
                    model_scripted = torch.jit.script(model)
                    model_scripted.save(f'{self.path}/{trial.study.study_name}/{trial.number}_best_model.pt')
                    print(
                        f"Saving best model at epoch {epoch + 1} with val loss {min_val_loss} and train loss {history[-1]['train loss']}")
                print(f"Epoch {epoch + 1}/{n_epochs}, "
                      f"Training Loss: {history[-1]['train loss']:.4f}, "
                      f"Validation Loss: {history[-1]['val loss']:.4f}, "
                      f"learning rate: {parameters['lr']}, "
                      f" -- ({datetime.now().strftime('%H:%M:%S')})")
                if trial is not None:
                    trial.report(history[-1]['val loss'], epoch)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                scheduler.step(history[-1]['val loss'])
        finally:
            print("Done.")
        return history[-1]['val loss'], model


if __name__ == '__main__':
    (model_name, filelist, n_trials, crop, image_size, batch_size, n_epochs, nonlinear) = get_params(sys.argv[1:])
    multiprocessing.set_start_method('fork')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fp32_precision = "tf32"

    image_sizing = transforms.CenterCrop((image_size, image_size)) if crop else v2.Resize(image_size)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         v2.ToDtype(torch.float, scale=True),
         v2.functional.autocontrast,
         image_sizing
         ])

    all_df = pd.read_csv(filelist, header=0, names=['filename', 'deltaz'])
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

    df = all_df[all_df['group'].isin(train_grp)]
    train_df = df.drop(columns='group')
    train_list = list(zip(train_df.filename, train_df.deltaz))
    train_dataset = FocusImageDataset(train_list, transform, None)

    df = all_df[all_df['group'].isin(val_grp)]
    val_df = df.drop(columns='group')
    val_list = list(zip(val_df.filename, val_df.deltaz))
    val_dataset = FocusImageDataset(val_list, transform, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    img, labels, filenames = train_dataset.__getitem__(0)
    print(f'{img.shape=} {labels=} {filenames=}')

    print(f'Training dataset: {len(train_loader)}')
    print(f'Validation dataset: {len(val_loader)}')

    tuner = HyperparameterTuner(model_name, filelist, n_trials, crop, image_size, batch_size, n_epochs, train_loader, val_loader)

    best_trial = tuner.tune_hyperparameters(n_trials)



