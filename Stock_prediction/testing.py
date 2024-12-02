import numpy as np
import torch
from sqlalchemy import label
from torch import nn
import matplotlib.pyplot as plt

from MT5_Link import MT5Class
from Stock_prediction.models.basic_nn_model import NeuralNetwork
from Stock_prediction.models.lstm_model import LSTM
from Stock_prediction.dataHandling import create_dataloader


def plot_data(actual_values, preds):

    close_pred = []
    close = []
    high_pred = []
    high = []
    open_pred = []
    open = []
    low_pred = []
    low = []

    for index in range(len(actual_values)):
        close_preds = preds[index][:, 3]
        close_value = actual_values[index][:, 3]
        high_preds = preds[index][:, 1]
        high_value = actual_values[index][:, 1]
        open_preds = preds[index][:, 0]
        open_value = actual_values[index][:, 0]
        low_preds = preds[index][:, 2]
        low_value = actual_values[index][:, 2]

        high_pred.append(high_preds)
        high.append(high_value)
        close_pred.append(close_preds)
        close.append(close_value)
        low_pred.append(low_preds)
        low.append(low_value)
        open_pred.append(open_preds)
        open.append(open_value)

    fig, axs = plt.subplots(4)

    axs[0].plot(high_pred, label="High Pred")
    axs[0].plot(high, label="High")
    axs[1].plot(close_pred, label="Close Pred")
    axs[1].plot(close, label="Close")
    axs[2].plot(low_pred, label="Low Pred")
    axs[2].plot(low, label="Low")
    axs[3].plot(open_pred, label="Open Pred")
    axs[3].plot(open, label="Open")

    # Create a legend
    plt.legend()
    plt.xlabel('Time-Step')
    plt.ylabel('Price')
    plt.show()



def test(dataloader, model, loss_fn, device):
    """ This tests performance of the model

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    actual_values = []
    preds = []
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.type(torch.float32).to(device), target.type(torch.float32).to(device)
            pred = model(inputs)
            test_loss += loss_fn(pred, target).item()

            preds.append(pred.cpu().numpy())
            actual_values.append(target.cpu().numpy())

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    plot_data(actual_values, preds)


def load_nn_model(filepath, input_dim, output_dim, hidden_dim):

    # create model and move to device (cuda or cpu)
    model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)

    model.load_state_dict(torch.load(filepath))

    return model

def load_lstm_model(filepath, input_dim, output_dim, hidden_dim, num_layers, device, batch_size):

    # create model and move to device (cuda or cpu)
    model = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                 device=device, batch_size=batch_size).to(device)
    model.load_state_dict(torch.load(filepath))

    return model

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
    input_dim = 7
    output_dim = 7
    look_back = 8
    hidden_dim = 128
    num_layers = 6
    symbol = 'USDJPY'
    model_type = "nn"

    filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\lstm_model.pt")

    # create and load model
    # model = load_lstm_model(filepath, input_dim, output_dim, hidden_dim, num_layers, device, batch_size)
    model = load_nn_model(filepath, input_dim, output_dim, hidden_dim)
    model  = model.to(device)

    # set loss function and optimiser
    loss_fn = nn.L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # create dataloader for MNIST dataset
    _, test_dl = create_dataloader(symbol, look_back, batch_size, model_type)

    # testing
    test(dataloader=test_dl, model=model, loss_fn=loss_fn, device=device)