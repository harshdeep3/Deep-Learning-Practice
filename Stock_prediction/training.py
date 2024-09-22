import numpy as np
import torch
from networkx.utils.backends import backends
from torch import nn

from MT5_Link import MT5Class
from Stock_prediction.models.basic_nn_model import NeuralNetwork
from Stock_prediction.models.lstm_model import LSTM

from Stock_prediction.dataHandling import create_dataloader


def train_one_epoch(dataloader, model, loss_fn, optimiser):
    """This train the model for one epoch

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        optimiser (_type_): _description_
    """

    model.train()
    batch_loss = []
    for batch, (X, y) in enumerate(dataloader):
        # moving to device and changing to
        X, y = X.type(torch.float32).to(device), y.type(torch.float32).to(device)
        # Compute prediction error
        pred = torch.unsqueeze(model(X), 0)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        batch_loss.append(loss)

    return np.array(batch_loss)


def train(epochs, dataloader, model, loss_fn, optimiser):
    """this trains the model for a number of epochs

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        optimiser (_type_): _description_
    """

    for epoch in range(epochs):
        print(f"-------------------------------\nEpoch {epoch + 1}")
        loss = train_one_epoch(dataloader, model, loss_fn, optimiser)
        print(f"Average loss: {np.average(loss)}")


def save_model(model, filepath: str):
    torch.save(model.state_dict(), filepath)


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
    lr = 1e-3
    epochs = 100
    input_dim = 7
    look_back = 32
    symbol = 'USDJPY'
    model_type = "lstm"

    # create model and move to device (cuda or cpu)
    # model = NeuralNetwork().to(device)
    model = LSTM(input_dim=input_dim*batch_size, output_dim=7, hidden_dim=64, num_layers=1, device=device).to(device)

    # set loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # create dataloader for MNIST dataset
    train_dl, test_dl = create_dataloader(symbol, look_back, batch_size, model_type)

    # training
    train(epochs=epochs, dataloader=train_dl, model=model, loss_fn=loss_fn, optimiser=optimiser)

    filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\nn_model.pt")

    save_model(model, filepath=filepath)
