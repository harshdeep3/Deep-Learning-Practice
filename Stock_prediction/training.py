
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
    """This train the model for one epoch

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        optimiser (_type_): _description_
    """
    model.to(device)
    model.train()
    
    batch_loss = []
    batch_mae = []
    for batch, (inputs, targets) in enumerate(dataloader):
        # moving to device and changing to
        inputs, targets = inputs.type(torch.float32).to(device), targets.type(torch.float32).to(device)
        # Compute prediction error
        if batch_size == 1:
            pred = model(inputs)
        else:
            pred = torch.squeeze(model(inputs))
        loss = loss_fn(pred, targets)
        mae = mean_absolute_error(pred.detach().cpu().numpy(), targets.detach().cpu().numpy())

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(inputs)
        batch_loss.append(loss)
        batch_mae.append(mae)

    return batch_loss, batch_mae


def train(epochs, dataloader, model, loss_fn, optimiser, device, batch_size):
    """this trains the model for a number of epochs

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        optimiser (_type_): _description_
    """
    epoch_loss = []
    epoch_mae = []
    for epoch in range(epochs):
        loss, mae = train_one_epoch(dataloader, model, loss_fn, optimiser, device, batch_size)
        if epoch % 1 == 0:
            print(f"-------------------------------\nEpoch {epoch + 1}")
            print(f"Average loss: {np.average(loss)}, mae: {np.average(mae)}")
        epoch_loss.append(np.average(loss))
        epoch_mae.append((np.average(mae)))

    plot_data(epoch_loss)
    plot_data(epoch_mae)


def save_model(model, filepath: str):
    torch.save(model.state_dict(), filepath)


def plot_data(data):
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
    batch_size = 1
    lr = 0.01
    epochs = 10000
    symbol = 'USDJPY'
    model_type = "nn"

    if model_type == "lstm":
        input_dim = 7
        output_dim = 1
        hidden_dim = 128
        look_back = 8
        num_layers = 6
    else:
        input_dim = 7
        output_dim = 7
        hidden_dim = 128
        look_back = 1

    # create model and move to device (cuda or cpu)
    model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
    # model = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers,
    #              device=device, batch_size=batch_size).to(device)

    # set loss function and optimiser
    loss_fn = nn.L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # create dataloader
    train_dl, _ = create_dataloader(symbol, look_back, batch_size, model_type)

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
