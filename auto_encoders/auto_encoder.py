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


class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()

        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)

        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]

        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder, self).__init__()

        # define fully connected layer to create embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))

        # store the shape before flattening
        self.reshape_dim = shape_before_flattening

        # encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            torch.sigmoid(),
        )

    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)

        x = self.decoder(x)
        return x


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
                images = images.reshape(-1, images_size)
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # training and testing dataloader
    train_dataset, test_dataset = get_dataloaders_mnist()
    # train_dataset, test_dataset = get_dataloaders_fashion_mnist()

    show_images_as_grid(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

    # understand the data
    images, labels = train_dataset[0]
    print(f"image shape {images.size()}")
    print(torch.min(images), torch.max(images))

    input_size = 1

    for i in images.size():
        input_size = input_size * i

    # initialize the NN
    autoencoder = AutoEncoder(input_dim=input_size, output_dim=3).to(device)
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
