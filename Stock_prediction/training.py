import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

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
    model.to(device)
    model.train()
    
    batch_loss = []
    batch_acc = []
    for batch, (inputs, targets) in enumerate(dataloader):
        # moving to device and changing to
        inputs, targets = inputs.type(torch.float32).to(device), targets.type(torch.float32).to(device)

        inputs = torch.squeeze(inputs, 0)
        targets = torch.squeeze(targets, 0)

        pred = model(inputs)

        loss = loss_fn(pred, targets)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(inputs)

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

    epoch_loss = []
    for epoch in range(epochs):
        # run one epoch
        loss = train_one_epoch(dataloader, model, loss_fn, optimiser)

        if epoch % 1000 == 0:
            print(f"-------------------------------\nEpoch {epoch + 1}")
            print(f"Average loss: {np.average(loss)}")

        epoch_loss.append(np.average(loss))

    plt.plot(epoch_loss)
    plt.show()


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
    batch_size = 16
    lr = 1e-3
    epochs = 5000
    input_dim = 7
    look_back = 32
    symbol = 'USDJPY'
    model_type = "lstm"

    # create model and move to device (cuda or cpu)
    # model = NeuralNetwork().to(device)
    model = LSTM(input_dim=input_dim, output_dim=7, hidden_dim=64, num_layers=1,
                 device=device, batch_size=batch_size).to(device)

    # set loss function and optimiser
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # create dataloader for MNIST dataset
    train_dl, _ = create_dataloader(symbol, look_back, batch_size, model_type)

    # training
    train(epochs=epochs, dataloader=train_dl, model=model, loss_fn=loss_fn, optimiser=optimiser)

    filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\nn_model.pt")

    save_model(model, filepath=filepath)
