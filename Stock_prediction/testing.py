import numpy as np
import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.functional import dropout

from MT5_Link import MT5Class
from Stock_prediction.models.basic_nn_model import NeuralNetwork
from Stock_prediction.models.lstm_model import LSTM
from Stock_prediction.dataHandling import create_dataloader


def plot_data(actual_values, preds):
    """
    Visualizes the comparison between predicted and actual values across four categories: Open, High, Low, and Close.
    This function generates a subplot for each category, plots the predicted and actual values for each,
    and labels them for easier interpretation. The x-axis represents time-steps and the y-axis values represent prices.

    Args:
        actual_values (ndarray): A 2D numpy array containing the actual values for Open, High, Low, and Close.
            Each row corresponds to a time-step, and columns represent the respective categories.
        preds (ndarray): A 2D numpy array containing the predicted values for Open, High, Low, and Close.
            Each row corresponds to a time-step, and columns represent the respective categories.
    """
    open_pred = preds[:, 0]
    open = actual_values[:, 0]
    high_pred = preds[:, 1]
    high = actual_values[:, 1]
    low_pred = preds[:, 2]
    low = actual_values[:, 2]
    close_pred = preds[:, 3]
    close = actual_values[:, 3]

    fig, axs = plt.subplots(4)

    axs[0].set_title('High')
    axs[0].plot(high_pred, label="High Pred")
    axs[0].plot(high, label="High")

    axs[1].set_title('Close')
    axs[1].plot(close_pred, label="Close Pred")
    axs[1].plot(close, label="Close")

    axs[2].set_title('Low')
    axs[2].plot(low_pred, label="Low Pred")
    axs[2].plot(low, label="Low")

    axs[3].set_title('Open')
    axs[3].plot(open_pred, label="Open Pred")
    axs[3].plot(open, label="Open")

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
    # precision point
    torch.set_printoptions(precision=20)
    np.set_printoptions(precision=20)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = []
    actual_values = []
    preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.type(torch.double).to(device), labels.type(torch.double).to(device)

            # change the shape from (1,7) to (7)
            labels = torch.squeeze(labels)

            pred = model(inputs)
            test_loss.append(loss_fn(pred, labels).item())

            preds.append(pred.cpu().numpy())
            actual_values.append(labels.cpu().numpy())

    test_loss = np.average(test_loss)
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    actual_values = np.array(actual_values)
    preds = np.array(preds)

    preds = preds.reshape(actual_values.shape)
    plot_data(actual_values, preds)


def load_nn_model(filepath, input_dim, output_dim, dropout, hidden_dim):
    """
    Loads a pre-trained neural network model from a file.

    This function initializes a neural network with the provided input, output,
    and hidden dimensions, moves it to the appropriate device (CUDA or CPU), and
    loads the pre-trained weights from the specified file.

    Args:
        filepath (str): Path to the file containing the stored model weights.
        input_dim (int): Dimension of the input features for the neural network.
        output_dim (int): Dimension of the output features for the neural network.
        hidden_dim (int): Dimension of the hidden layer(s) in the neural network.

    Returns:
        NeuralNetwork: A neural network model with the pre-trained weights loaded.
    """
    # create model and move to device (cuda or cpu)
    model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim, dropout=dropout,
                          hidden_dim=hidden_dim).to(device)

    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
    else:
        print("Model file not found")

    return model

def load_lstm_model(filepath, input_dim, output_dim, hidden_dim, num_layers, device, batch_size):
    """
    Loads a pre-trained LSTM model from a given file path and configures it based on provided parameters.
    The model is transferred to the appropriate device (CUDA or CPU), and its state is restored from the
    specified file.

    Args:
        filepath (str): Path to the file containing the saved model state.
        input_dim (int): Size of the input features for the LSTM model.
        output_dim (int): Size of the output features for the LSTM model.
        hidden_dim (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of stacked LSTM layers in the model.
        device (torch.device): Target device to load the model onto
            (e.g., CPU or CUDA device).
        batch_size (int): Number of samples per batch that the LSTM model
            will process.

    Returns:
        nn.Module: The loaded LSTM model with pre-trained parameters.
    """
    # create model and move to device (cuda or cpu)
    model = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                 device=device, batch_size=batch_size).to(device)
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
    else:
        print("Model file not found")

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
    dropout=0.02
    look_back = 8
    hidden_dim = 128
    num_layers = 6
    symbol = 'USDJPY'
    model_type = "mlp"

    if model_type == "lstm":

        filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                    "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\lstm_model.pt")
        # create and load model
        model = load_lstm_model(filepath, input_dim, output_dim, hidden_dim, num_layers, device, batch_size)
    else:
        filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                    "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\nn_model.pt")
        model = load_nn_model(filepath, input_dim, output_dim, dropout, hidden_dim)

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