
import torch
from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    """
    Defines a feedforward neural network using PyTorch's `nn.Module`.

    This class implements a neural network architecture with multiple layers,
    including linear layers with ReLU activations and a dropout layer for
    regularization. The network is designed to take an input tensor, process it
    through the defined layers, and output logits, which represent the unnormalized
    scores for the output classes. The structure is flexible, allowing the
    specification of input dimensions, output dimensions, a hidden layer size, and
    a dropout probability.

    Attributes:
        flatten (nn.Flatten): This attribute flattens the input tensor to a
            1-dimensional tensor, enabling it to be passed through fully connected
            layers.
        linear_relu_stack (nn.Sequential): A sequential container of the network
            layers, including linear layers, ReLU activations, dropout, and the
            output layers.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float, hidden_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 256).double(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512).double(),
            nn.ReLU(),
            nn.Linear(512, hidden_dim).double(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim).double()
        )

    def forward(self, x):
        """
        Applies a forward pass through the neural network layers.

        This method takes the input tensor, flattens it, and passes it through a sequential
        stack of linear and ReLU layers to produce the logits, which are the unnormalized
        scores for each output class of the network.

        Args:
            x: Input tensor containing data to process. This can be a single input
                or a batch of inputs.

        Returns:
            The logits tensor, containing the unnormalized scores for the output
            classes after processing the input through the network layers.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    print("This is a Model class. This should not be run.")
    
    