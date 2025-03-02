
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from MT5_Link import MT5Class
from Stock_prediction.models.basic_nn_model import NeuralNetwork
from Stock_prediction.models.lstm_model import LSTM
from Stock_prediction.dataHandling import create_dataloader


def train_one_epoch(dataloader, model, loss_fn, optimiser, device, batch_size):
    """
    Trains a single epoch for the provided model using the given dataloader, loss function,
    optimizer, and device. Computes and accumulates the batch loss and mean absolute error (MAE)
    for each batch during training. Moves the model and input data to the specified device.
    Executes forward propagation, computes loss, backpropagation, and updates model parameters.
    Handles scenarios for both batch sizes of one and multi-sample batches.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader providing batches of input data
            and corresponding target values for training.
        model (torch.nn.Module): Model to be trained during the epoch.
        loss_fn (torch.nn.Module): Loss function used to compute the error between predicted
            and target values.
        optimiser (torch.optim.Optimizer): Optimizer responsible for updating the model's
            parameters based on the computed gradients.
        device (torch.device): Device (CPU or GPU) to be used for model, inputs, and targets
            during training.
        batch_size (int): Batch size defining the number of samples processed in a single
            batch during training.

    Returns:
        tuple: A tuple containing two lists:
            - batch_loss (list[float]): List of loss values for each batch within the epoch.
            - batch_mae (list[float]): List of mean absolute error (MAE) values for each batch
                within the epoch.
    """
    model.to(device)
    model.train()
    
    batch_loss = []
    batch_mae = []
    for batch, (inputs, labels) in enumerate(dataloader):
        # moving to device and changing to
        inputs, labels = inputs.type(torch.double).to(device), labels.type(torch.double).to(device)
        # Compute prediction error
        if batch_size == 1:
            pred = model(inputs)
        else:
            pred = torch.squeeze(model(inputs))
        loss = loss_fn(pred, labels)

        mae = mean_absolute_error(pred.detach().cpu().numpy(), labels.detach().cpu().numpy())

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(inputs)
        batch_loss.append(loss)
        batch_mae.append(mae)

    return batch_loss, batch_mae


def train(epochs: int, dataloader, model: NeuralNetwork, loss_fn, optimiser: torch.optim.Optimizer,
          device: str, batch_size: int) -> None:
    """
    Trains a model over a specified number of epochs using a given dataloader, loss
    function, optimiser, device, and batch size. Logs and plots metrics such as
    loss and mean absolute error (MAE) at regular intervals for monitoring.

    Args:
        epochs (int): Number of epochs to train the model.
        dataloader: DataLoader object providing batches
            of training data.
        model (torch.nn.Module): PyTorch model to be trained.
        loss_fn: Loss function used for optimization.
        optimiser (torch.optim.Optimizer): Optimizer used for updating the model's
            parameters.
        device str: Device on which the training will be performed
            (e.g., "cpu" or "cuda").
        batch_size (int): Number of samples used in each training batch.
    """
    epoch_loss = []
    epoch_mae = []
    for epoch in range(epochs):
        loss, mae = train_one_epoch(dataloader, model, loss_fn, optimiser, device, batch_size)
        if epoch % 100 == 0:
            print(f"-------------------------------\nEpoch {epoch + 1}")
            print(f"Average loss: {np.average(loss)}, mae: {np.average(mae)}")
        epoch_loss.append(np.average(loss))
        epoch_mae.append((np.average(mae)))

    plot_data(epoch_loss)
    plot_data(epoch_mae)


def save_model(model: torch.nn.Module, filepath: str) -> None:
    """
    Saves the state dictionary of a PyTorch model to a specified file path. The function
    is typically used to store the weights and biases of a trained model, enabling reusability
    and portability of the trained model across different environments or later use.

    Args:
        model (torch.nn.Module): The PyTorch model whose state dictionary is to be saved.
        filepath (str): The file path where the state dictionary will be saved, should
            include filename and extension (e.g., 'model.pth').

    """
    torch.save(model.state_dict(), filepath)


def plot_data(data: list[float]) -> None:
    """
    Plots the provided data on a 2D graph and displays it using Matplotlib.

    This function takes in numerical data, visualizes it with a simple line
    plot using the Matplotlib library, and then renders the graph via the
    `show()` function. It is designed to provide quick and visual
    interpretation of the given data set.

    Args:
        data (list or any iterable of numerical values): The numerical data
            to be plotted on the graph.
    """
    plt.plot(data)
    plt.show()

if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # hyperparameters
    batch_size = 128
    lr = 0.0001
    dropout = 0.05
    epochs = 2000
    symbol = 'USDJPY'
    short_term_time_period = 30
    spilt_indices = 5
    model_type = "mlp"

    if model_type == "lstm":
        input_dim = 7
        output_dim = 7
        hidden_dim = 128
        look_back = 64
        num_layers = 6

        # create model and move to device (cuda or cpu)
        model = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                     device=device, batch_size=batch_size).to(device)
    else:
        input_dim = 7
        output_dim = spilt_indices * 2 * 4
        hidden_dim = 128
        look_back = 1

        # create model and move to device (cuda or cpu)
        model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim, dropout=dropout,
                              hidden_dim=hidden_dim).to(device)

    # set loss function and optimiser
    loss_fn = nn.L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # create dataloader
    train_dl, _ = create_dataloader(symbol, look_back, batch_size, short_term_time_period,
                                    spilt_indices, model_type)

    # training
    train(epochs=epochs, dataloader=train_dl, model=model, loss_fn=loss_fn, optimiser=optimiser, device=device,
          batch_size=batch_size)

    if model_type == "lstm":
        filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                    "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\lstm_model.pt")
    else:
        filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                    "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\nn_model.pt")

    save_model(model, filepath=filepath)
