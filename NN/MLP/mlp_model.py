import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


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


def create_dataloader(batch_size: int = 64):
    # Download training data from open datasets.
    training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    return train_dataloader, test_dataloader


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
    epochs = 10
    
    # create dataloader for MNIST dataset
    train_dl, test_dl = create_dataloader(batch_size)

    # create model and move to device (cuda or cpu)
    nn_model = NeuralNetwork().to(device)
    
    # set loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(nn_model.parameters(), lr=lr)
    
    # training
    train(epochs=epochs, dataloader=train_dl, model=nn_model, loss_fn=loss_fn, optimiser=optimiser)
    
    test(dataloader=train_dl, model=nn_model, loss_fn=loss_fn)
    
    