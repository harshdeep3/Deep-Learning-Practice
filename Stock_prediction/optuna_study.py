from datetime import datetime
import optuna
import os
import torch
import torch.nn as nn
import numpy as np

from optuna.trial import TrialState

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

    ##################################################
    # Hyper-parameter selections
    ##################################################
    loss_fn_choice = trial.suggest_categorical("loss_fun", ["L1"])
    n_layer = trial.suggest_categorical("n_layers", [1, 2, 3, 4, 5, 6])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 124, 256, 512])
    lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512])
    look_back = trial.suggest_categorical('look_back', [1, 2, 4, 8, 16, 32, 64, 128])
    n_epochs =  trial.suggest_categorical('n_epochs', [10, 25, 50, 100, 150, 200, 300, 500, 1000])

    print(f"n_layer:  {n_layer}, batch_size:  {batch_size}, lr:  {lr}, hidden_dim:  {hidden_dim}, look_back:  "
          f"{look_back}, n_epochs:  {n_epochs}")

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
    model = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=n_layer,
                 device=device, batch_size=batch_size).to(device)

    # set loss function and optimiser
    if loss_fn_choice == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_fn_choice == "L1":
        loss_fn = nn.L1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(n_epochs):
        # run one epoch
        loss = train_one_epoch(train_dl, model, loss_fn, optimiser, device)
        if epoch % 1 == 0:
            print(f"-------------------------------\nEpoch {epoch + 1}")
            print(f"Average loss: {np.average(loss)}")

        trial.report(np.average(loss), epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.average(loss)


def main():

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    hyperparameter_filepath = (f"C:/Users/Harsh/Desktop/Coding Projects/GitHub/Deep-Learning-Practice"
                               f"/Stock_prediction/hyperparameters/{now}.txt")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=2000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

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

            file.write(f' -- {best_trial.value}')
            file.write("\n")


if __name__ == '__main__':
    main()