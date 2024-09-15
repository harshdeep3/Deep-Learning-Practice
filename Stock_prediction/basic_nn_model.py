
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
from MT5_Link import get_historic_data, MT5Class


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_one_epoch(dataloader, model, loss_fn, optimiser):
    """This train the model for one epoch

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        optimiser (_type_): _description_
    """
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 


def train(epochs, dataloader, model, loss_fn, optimiser):
    """this trains the model for a number of epochs

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        optimiser (_type_): _description_
    """
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_one_epoch(dataloader, model, loss_fn, optimiser)


def test(dataloader, model, loss_fn):
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
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    


if __name__ == "__main__":
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 100

    # create model and move to device (cuda or cpu)
    nn_model = NeuralNetwork().to(device)
    
    # set loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(nn_model.parameters(), lr=lr)
    
    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()
    
    # metatrader info
    # timeframe objects https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
    timeframe = mt5.TIMEFRAME_M5
    symbol = 'USDJPY'
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(2010, 1, 10, tzinfo=timezone)
    utc_to = datetime(2020, 1, 11, tzinfo=timezone)
    
    # goes back to 1971-08-11
    count = 13500

    # print account info
    # mt5_obj.get_acc_info()
    
    # get data
    df = get_historic_data(symbol, timeframe, count)
    df = df.set_index('time')
    
    # create dataloader for MNIST dataset
    # train_dl, test_dl = create_dataloader(batch_size)
    
    # training
    # train(epochs=epochs, dataloader=train_dl, model=nn_model, loss_fn=loss_fn, optimiser=optimiser)
    
    # test(dataloader=train_dl, model=nn_model, loss_fn=loss_fn)
    
    