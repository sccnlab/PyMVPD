# MVPD - 1-Layer Linear Neural Network
import torch
import torch.nn as nn

class NN_1layer(nn.Module):
    """
    Build a 1-layer fully connected linear neural network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(self.bn1(x))
        out = self.fc2(out)
        return out


