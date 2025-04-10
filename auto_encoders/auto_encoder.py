"""_summary_

Returns:
    _type_: _description_
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Hyperparameters
LR = 1e-4
PATIENCE = 2
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 64
EMBEDDING_DIM = 2
EPOCHS = 10


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, output_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # -> based on the values of the data if between 0 and 1 then sigmode, -1 and 1 then tanH
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class AutoEncoder2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # -> N, 64, 1, 1
        )

        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid(),
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def get_dataloaders_mnist(train_transforms=None, test_transforms=None):
    """Load MNIST data set

    Args:
        train_transforms (_type_, optional): _description_. Defaults to None.
        test_transforms (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: train and test dataset
    """
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, transform=train_transforms, download=True)
    test_dataset = datasets.MNIST(root="data", train=False, transform=test_transforms)
    return train_dataset, test_dataset


def get_dataloaders_fashion_mnist(train_transforms=None, test_transforms=None):
    """Load Fashion MNIST data set

    Args:
        train_transforms (_type_, optional): _description_. Defaults to None.
        test_transforms (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: train and test dataset
    """
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root="data", train=True, transform=train_transforms, download=True)
    test_dataset = datasets.FashionMNIST(root="data", train=False, transform=test_transforms)

    return train_dataset, test_dataset


def show_images_as_grid(train_data):
    """show dataset as a grid

    Args:
        train_data (_type_): _description_
    """
    # plot multiple images
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()  # get random index
        img, label = train_data[sample_idx]  # get image and label for index
        figure.add_subplot(rows, cols, i)  # add to subplot
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.show()


def training(images_size, train_loader, loss_fn, optimizer, n_epochs, encoder=None, decoder=None, autoencoder=None):

    # initialize the best validation loss as infinity
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, n_epochs + 1)):
        # initialize running loss as 0
        running_loss = 0.0

        if encoder is not None and decoder is not None:
            encoder.train()
            decoder.train()
        else:
            autoencoder.train()
        ###################
        # train the model #
        ###################
        # loop over the batches of the training dataset
        for batch_idx, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # _ stands in for labels
            images = images.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            if encoder is not None and decoder is not None:
                # forward pass: encode the data and decode the encoded representation
                encoded = encoder(images)
                decoded = decoder(encoded)

                # calculate the loss
                loss = loss_fn(decoded, images)
            else:
                # only for autocode1 which is the linear model
                # images = images.reshape(-1, images_size)
                output = autoencoder(images)
                # calculate the loss
                loss = loss_fn(output, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            # loss.requires_grad = True
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # accumulate the loss for the current batch
            running_loss += loss.item()

        # print avg training statistics
        train_loss = running_loss / len(train_loader)

        if train_loss < best_val_loss:
            best_val_loss = train_loss

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    print(f"Best Training Loss: {best_val_loss}")


def testing(test_loader, loss_fn, optimizer, encoder=None, decoder=None, autoencoder=None):

    for batch_idx, (images, label) in tqdm(enumerate(test_loader), total=2):
        # _ stands in for labels
        images = images.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        if encoder is not None and decoder is not None:
            # forward pass: encode the data and decode the encoded representation
            encoded = encoder(images)
            decoded = decoder(encoded)

            # calculate the loss
            loss = loss_fn(decoded, images)
        else:
            # only for autocode1 which is the linear model
            # images = images.reshape(-1, images_size)
            output = autoencoder(images)
            # calculate the loss
            loss = loss_fn(output, images)
            print(f"loss -> {loss}")
            output_imgs = output.cpu().detach().numpy()
            output_imgs = np.squeeze(output_imgs)
            plt.imshow(output_imgs)
            plt.show()
            images = images.cpu().detach().numpy()
            images = np.squeeze(images)
            plt.imshow(images)
            plt.show()
        break


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # training and testing dataloader
    train_dataset, test_dataset = get_dataloaders_mnist()
    # train_dataset, test_dataset = get_dataloaders_fashion_mnist()

    show_images_as_grid(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=2, shuffle=False)

    # understand the data
    images, labels = train_dataset[0]
    print(f"image shape {images.size()}")
    print(torch.min(images), torch.max(images))

    input_size = 1

    for i in images.size():
        input_size = input_size * i

    # initialize the NN
    autoencoder = AutoEncoder2(input_dim=input_size, output_dim=3).to(device)
    # encoder = Encoder(channels=CHANNELS, image_size=IMAGE_SIZE, embedding_dim=EMBEDDING_DIM).to(device)
    # decoder = Decoder(
    #     channels=CHANNELS, shape_before_flattening=encoder.shape_before_flattening, embedding_dim=EMBEDDING_DIM
    # ).to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    training(
        images_size=input_size,
        autoencoder=autoencoder,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=EPOCHS,
    )

    # testing
    show_images_as_grid(test_dataset)

    testing(
        autoencoder=autoencoder,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
