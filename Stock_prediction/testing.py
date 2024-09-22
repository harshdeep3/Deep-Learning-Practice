import numpy as np
import torch
from torch import nn

from MT5_Link import MT5Class
from Stock_prediction.models.basic_nn_model import NeuralNetwork
from Stock_prediction.models.lstm_model import LSTM

from Stock_prediction.dataHandling import create_dataloader


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
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.type(torch.float32).to(device), y.type(torch.float32).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct)}%, Avg loss: {test_loss:>8f} \n")


def load_model(filepath):

    # create model and move to device (cuda or cpu)
    model = NeuralNetwork().to(device)
    # model = LSTM(input_dim=7, output_dim=7, hidden_dim=32, num_layers=1).to(device)

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
    batch_size = 64
    lr = 1e-3
    epochs = 100

    filepath = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\"
                "Deep-Learning-Practice\\Stock_prediction\\models\\saved_models\\nn_model.pt")

    # create and load model
    model  = load_model(filepath)
    model  = model.to(device)

    # set loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # create dataloader for MNIST dataset
    train_dl, test_dl = create_dataloader('USDJPY', 32, batch_size, "nn")

    # testing
    test(dataloader=train_dl, model=model, loss_fn=loss_fn, device=device)