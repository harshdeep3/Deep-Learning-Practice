import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


class CNN(nn.Module):

  def __init__(self) -> None:
    super(CNN, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(
          in_channels=1, #num of channels in the input image
          out_channels=16, # num of channels produce by CNN
          kernel_size=5, # Size of Kernel
          stride=1, # the number of pixels to pass at a time when sliding the C kernel
          padding=2 # preserve exactly the size of the input image
      ),
      nn.ReLU(), # activation node
      nn.MaxPool2d(kernel_size=2),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(16, 32, 5, 1, 2),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.out = nn.Linear(32*7*7, 10)

  def forward(self, x):
    x = self.conv1(x) # get the output of the conv1
    x = self.conv2(x) # get output of conv2 using the output of conv1 as input

    #flatten the output of conv2 to (batch_size, 32*7*7)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return output # return x to visualise


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


def show_multiple_images(train_data):
    #plot multiple images
    figure = plt.figure(figsize=(10,8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data.dataset.data), size=(1,)).item() # get random index
        img, label = train_data.dataset[sample_idx] # get image and label for index
        figure.add_subplot(rows, cols, i) # add to subplot
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap='gray')

    plt.show()
 
    
def show_single_image(train_data, index):
    # visualising MNIST Data

    plt.imshow(train_data.dataset.data[index], cmap='gray')
    plt.title('%i' % train_data.dataset.targets[index])
    plt.show()


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
            test_loss += loss_fn(pred, y)
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
    
    batch_size = 64
    
    # create dataloader for MNIST dataset
    train_dl, test_dl = create_dataloader(batch_size)
    
    # show image
    # show_single_image(train_dl, 10)
    # show_multiple_images(train_dl)
    
    # cnn model
    cnn = CNN().to(device)
    
    # num of epochs
    epochs = 10
    
    # defining a loss function
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(), lr=0.01)
    
    # training
    train(epochs=epochs, dataloader=train_dl, model=cnn, loss_fn=loss_fn, optimiser=optimiser)
    
    # testing
    test(dataloader=train_dl, model=cnn, loss_fn=loss_fn)