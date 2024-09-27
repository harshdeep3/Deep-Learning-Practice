import torch
from torch import nn


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device, batch_size):
        super(LSTM, self).__init__()

        # device
        self.device = device

        # output dim
        self.output_dim = output_dim

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch size
        self.batch_size = batch_size

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x)

        # Index hidden state of last time step
        out = self.fc(out)

        if self.batch_size > 1:
            return out[:, -1]
        return out[-1]

if __name__ == '__main__':
    print("Lstm model file should not be run")
