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
        self.lstm_1 = nn.LSTM(input_dim, 128, num_layers, batch_first=True)
        self.activation_1 = nn.ReLU()
        self.lstm_2 = nn.LSTM(128, 256, num_layers, batch_first=True)
        self.activation_2 = nn.ReLU()
        self.lstm_3 = nn.LSTM(256, hidden_dim, num_layers, batch_first=True)
        self.activation_3 = nn.ReLU()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        if self.batch_size == 1:
            x = torch.squeeze(x)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm_1(x)
        out = self.activation_1(out)
        out, _ = self.lstm_2(out)
        out = self.activation_2(out)
        out, _ = self.lstm_3(out)
        out = self.activation_3(out)

        # Index hidden state of last time step
        out = self.fc(out)

        if self.batch_size > 1:
            return out[:, -1]
        return out[-1]

if __name__ == '__main__':
    print("Lstm model file should not be run")
