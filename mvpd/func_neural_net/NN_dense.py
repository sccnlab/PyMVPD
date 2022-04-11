# MVPD - Linear Neural Network with Dense Connections
import torch
import torch.nn as nn

class NN_dense(nn.Module):
    """
    Build a fully densely connected linear neural network with equal hidden layer size.

    INPUT FORMAT
    input_size - the number of voxels in the predictor ROI 
    hidden_size - the number of units per hidden layer 
    output_size - the number of voxels in the target ROI 
    num_hLayer - the number of hidden layers 

    x - the functional data in the predictor ROI [t, n]
       t - the number of timepoints in the experimental run(s)
       n - the number of voxels in the predictor ROI 
    """
    def __init__(self, input_size, hidden_size, output_size, num_hLayer):
        super().__init__() 
        self.bn = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.nLayers = num_hLayer+2 

        self.bn.append(nn.BatchNorm1d(input_size))
        self.linear.append(nn.Linear(input_size, hidden_size))

        for nLayer in range(1, num_hLayer):
            self.bn.append(nn.BatchNorm1d(input_size+hidden_size*nLayer))
            self.linear.append(nn.Linear(input_size+hidden_size*nLayer, hidden_size))

        self.bn.append(nn.BatchNorm1d(input_size+hidden_size*num_hLayer))
        self.linear.append(nn.Linear(input_size+hidden_size*num_hLayer, output_size)) 

    def forward(self, x):
        for i in range(self.nLayers-2):
            out = self.linear[i](self.bn[i](x))
            x = torch.cat((x, out), dim=1) 
        x = self.linear[self.nLayers-2](self.bn[self.nLayers-2](x))
        return x


