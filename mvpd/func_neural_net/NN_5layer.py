# MVPD - 5-Layer Linear Neural Network
import torch
import torch.nn as nn

class NN_5layer(nn.Module):
    """
    Build a 5-layer fully connected linear neural network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(self.bn1(x))
        out = self.fc2(self.bn2(out))
        out = self.fc3(self.bn3(out))
        out = self.fc4(self.bn4(out))
        out = self.fc5(self.bn5(out))
        out = self.fc6(self.bn6(out))
        return out

