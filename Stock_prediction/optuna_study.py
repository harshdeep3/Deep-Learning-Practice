from datetime import datetime
import optuna
import os
import torch
import torch.nn as nn
import numpy as np

from optuna.trial import TrialState
from torch.nn import CrossEntropyLoss
from MT5_Link import MT5Class

from Stock_prediction.models.basic_nn_model import NeuralNetwork
from Stock_prediction.models.lstm_model import LSTM
from Stock_prediction.dataHandling import create_dataloader
from Stock_prediction.training import train_one_epoch


def objective(trial):
    """

    Args:
        trial:

    Returns:

    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    ##################################################
    # Hyper-parameter selections
    ##################################################
    loss_fn_choice = trial.suggest_categorical("loss_fun", ["L1", "MSE", "CrossEntropy"])
    n_layer = trial.suggest_categorical("n_layers", [1, 2, 3, 4, 5, 6])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 124, 256, 512])
    lr = trial.suggest_categorical('lr', [0.00001, 0.0001, 0.001, 0.01, 0.1])
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512])
    look_back = trial.suggest_categorical('look_back', [1, 2, 4, 8, 16, 32, 64, 128])
    n_epochs =  trial.suggest_categorical('n_epochs', [1, 2])

    print(f"n_layer:  {n_layer}, batch_size:  {batch_size}, lr:  {lr}, hidden_dim:  {hidden_dim}, look_back:  "
          f"{look_back}, n_epochs:  {n_epochs}, loss_fn-:  {loss_fn_choice}")

    ##################################################
    # Loading data
    ##################################################
    symbol = 'USDJPY'
    model_type = "lstm"
    input_dim = 7
    output_dim = 7

    train_dl, _ = create_dataloader(symbol, look_back, batch_size, model_type)

    ##################################################
    # create model
    ##################################################
    if model_type == "lstm":

        # create model and move to device (cuda or cpu)
        model = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=n_layer,
                     device=device, batch_size=batch_size).to(device)
    else:

        # create model and move to device (cuda or cpu)
        model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)

    # set loss function and optimiser
    if loss_fn_choice == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_fn_choice == "L1":
        loss_fn = nn.L1Loss()
    elif loss_fn_choice == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(n_epochs):
        # run one epoch
        loss, mae = train_one_epoch(train_dl, model, loss_fn, optimiser, device, batch_size)
        if epoch % 10 == 0:
            print(f"-------------------------------\nEpoch {epoch + 1}")
            print(f"Average loss: {np.average(loss)}, mae: {np.average(mae)}")
            print(f"Last loss: {loss[-1]}, mae: {mae[-1]}")

    return mae[-1], loss[-1]


def main():

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    hyperparameter_filepath = (f"C:/Users/Harsh/Desktop/Coding Projects/GitHub/Deep-Learning-Practice"
                               f"/Stock_prediction/hyperparameters/{now}.txt")

    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=2)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trials[-1]

    print("  Value: Mae", best_trial.values[0], " Loss:", best_trial.values[1])

    print("  Params: ")
    if not os.path.exists(hyperparameter_filepath):
        with open(hyperparameter_filepath, 'w') as file:
            file.write('')
            file.write("\n")

            for trial in study.trials:
                for key, value in trial.params.items():
                    file.write(f'{key}  {value}, ')

                file.write(f' -- {trial.values}')
                file.write("\n")

            file.write("\n")
            file.write(f'Best trail')
            file.write("\n")

            for key, value in best_trial.params.items():
                print("    {}: {}".format(key, value))
                file.write(f'{key}  {value}, ')

            file.write(f'Mae: {best_trial.values[0]}, loss: {best_trial.values[1]}')
            file.write("\n")


if __name__ == '__main__':
    main()