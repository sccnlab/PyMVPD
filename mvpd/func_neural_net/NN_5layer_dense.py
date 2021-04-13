# MVPD - 5-Layer Linear Neural Network with Dense Connections
import torch
import torch.nn as nn

class NN_5layer_dense(nn.Module):
    """
    Build a 5-layer fully connected linear neural network with dense connections.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(input_size+hidden_size)
        self.fc2 = nn.Linear(input_size+hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(input_size+hidden_size*2)
        self.fc3 = nn.Linear(input_size+hidden_size*2, hidden_size)
        self.bn4 = nn.BatchNorm1d(input_size+hidden_size*3)
        self.fc4 = nn.Linear(input_size+hidden_size*3, hidden_size)
        self.bn5 = nn.BatchNorm1d(input_size+hidden_size*4)
        self.fc5 = nn.Linear(input_size+hidden_size*4, hidden_size)
        self.bn6 = nn.BatchNorm1d(input_size+hidden_size*5)
        self.fc6 = nn.Linear(input_size+hidden_size*5, output_size)

    def forward(self, x):
        out1 = self.fc1(self.bn1(x))
        in2 = torch.cat((x, out1), dim=1)
        out2 = self.fc2(self.bn2(in2))
        in3 = torch.cat((in2, out2), dim=1)
        out3 = self.fc3(self.bn3(in3))
        in4 = torch.cat((in3, out3), dim=1)
        out4 = self.fc4(self.bn4(in4))
        in5 = torch.cat((in4, out4), dim=1)
        out5 = self.fc5(self.bn5(in5))
        in6 = torch.cat((in5, out5), dim=1)
        out = self.fc6(self.bn6(in6))
        return out

