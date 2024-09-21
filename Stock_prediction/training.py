
import torch
from torch import nn

from MT5_Link import MT5Class
from basic_nn_model import NeuralNetwork

from torch.utils.data import DataLoader
from Stock_prediction.datasets.stockDataset import StockDataset


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
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_one_epoch(dataloader, model, loss_fn, optimiser)


def create_dataloader(symbol: str='USDJPY', look_back: int= 32, batch_size: int = 64):
    """
    Creates and returns DataLoader objects for training and testing stock data.

    This function constructs training and testing datasets using the provided stock symbol and look-back period.
    It then initializes PyTorch DataLoader instances for both the training and testing datasets with the specified batch size.

    Args:
        batch_size (int, optional): The number of samples per batch to load. Defaults to 64.
        symbol (str): The stock symbol or ticker for which the dataset is to be created.
        look_back (int): The number of previous time steps (look-back period) used as input to predict the next time step.

    Returns:
        tuple: A tuple containing two DataLoader instances:
            - train_dataloader: DataLoader for the training dataset.
            - test_dataloader: DataLoader for the testing dataset.

    Example:
        train_loader, test_loader = create_dataloader(batch_size=32, symbol='USDJPY', look_back=30)
    """
    train_dataset = StockDataset(symbol, look_back, True)
    test_dataset = StockDataset(symbol, look_back, False)

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


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

    # create dataloader for MNIST dataset
    train_dl, test_dl = create_dataloader('USDJPY', 32, batch_size)

    # training
    train(epochs=epochs, dataloader=train_dl, model=nn_model, loss_fn=loss_fn, optimiser=optimiser)

    # test(dataloader=train_dl, model=nn_model, loss_fn=loss_fn)